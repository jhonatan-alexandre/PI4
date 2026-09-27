"""Report a completed search without changing model selection or predictions."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .reporting import markdown_table


def generate_tuning_report(output: Path) -> Path:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    final = json.loads((output / "final" / "model_config.json").read_text(encoding="utf-8"))
    trials = pd.read_csv(output / "search_trials.csv")
    table = pd.read_csv(output / "test_metrics.csv")
    predictions = pd.read_csv(output / "test_predictions.csv", parse_dates=["time"])
    winner, neural = selection["winner"], selection["best_neural"]
    ranking = trials.pivot(index="model", columns="fold", values="score_kwh")
    ranking["mean_kwh"] = ranking.mean(axis=1)
    ranking = ranking.sort_values("mean_kwh").reset_index()
    ranking.to_csv(output / "validation_ranking.csv", index=False)
    blends = pd.DataFrame([{"Peso da árvore": float(weight),
                            "Peso da LSTM": 1-float(weight), "Critério médio (kWh)": value}
                           for weight, value in selection["blend_scores"].items()])
    blends.to_csv(output / "blend_validation_scores.csv", index=False)

    changes = []
    for name in dict.fromkeys([winner, neural]):
        selected = table[table.model == name].set_index("horizonte")
        for baseline in ("previous_lstm_original_saved", "previous_hgb_full", "previous_lstm_features"):
            old = table[table.model == baseline].set_index("horizonte")
            for horizon in old.index:
                changes.append({"model": name, "reference": baseline, "horizonte": horizon,
                                "MAE_reduction_pct": 100*(1-selected.loc[horizon, "MAE"]/old.loc[horizon, "MAE"]),
                                "RMSE_reduction_pct": 100*(1-selected.loc[horizon, "RMSE"]/old.loc[horizon, "RMSE"])})
    improvements = pd.DataFrame(changes)
    improvements.to_csv(output / "improvements.csv", index=False)
    chosen_models = list(dict.fromkeys(["previous_lstm_original_saved", "previous_hgb_full", neural, winner]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, metric in zip(axes, ("MAE", "RMSE")):
        for i, name in enumerate(chosen_models):
            part = table[table.model == name]
            ax.bar(np.arange(3)+(i-(len(chosen_models)-1)/2)*.18, part[metric],
                   width=.18, label=name)
        ax.set(xticks=np.arange(3), xticklabels=["15 min", "30 min", "1 h"],
               ylabel=f"{metric} (kWh)", title=f"{metric} — comparação histórica")
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "comparison_kwh.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(ranking))
    for i, fold in enumerate(("fold1", "fold2")):
        ax.bar(x+(i-.5)*.35, ranking[fold], width=.35, label=fold)
    ax.set(xticks=x, xticklabels=ranking.model, ylabel="0,5 × MAE + 0,5 × RMSE (kWh)",
           title="Busca de hiperparâmetros — duas validações cronológicas")
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "validation_search.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    preview = predictions.iloc[:864]
    for ax, horizon in zip(axes, ("15min", "30min", "1h")):
        ax.plot(preview.time, preview[f"actual_{horizon}"], color="#18332F", linewidth=1, label="Real")
        ax.plot(preview.time, preview[f"{winner}_{horizon}"], color="#087F78", linewidth=1, label=winner)
        ax.set(ylabel=f"{horizon} (kWh)")
        ax.legend(fontsize=8)
    axes[0].set_title("Primeiros três dias — energia observada e prevista")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output / "predictions_kwh.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, fold in zip(axes, ("fold1", "fold2")):
        for name in (p["name"] for p in manifest["search"]["networks"]):
            history = pd.read_csv(output / fold / f"{name}_history.csv")
            ax.plot(np.arange(1, len(history)+1), history.val_kwh_score, label=name)
        ax.set(title=fold, xlabel="Época", ylabel="Critério de validação (kWh)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "network_validation_kwh.png", dpi=180)
    plt.close(fig)

    selected_parameters = json.dumps({"tree": final["tree_parameters"],
                                      "network": final["neural_parameters"],
                                      "final_neural_epochs": final["final_neural_epochs"],
                                      "ensemble_tree_weight": final["tree_weight"]}, ensure_ascii=False, indent=2)
    text = f"""# Retreinamento e ajuste de hiperparâmetros em kWh

**Modelo escolhido pela validação: {winner}. Melhor rede: {neural}.**
O resultado anterior e os modelos antigos foram preservados em suas pastas.

## Protocolo

A busca avaliou cinco configurações de HistGradientBoosting e três da LSTM,
cada uma em dois períodos cronológicos consecutivos. São 16 ajustes de busca,
seguidos do retreinamento final das configurações de árvore e rede selecionadas.
Não foram usados alvos futuros como entrada, nem interpoladas medições ausentes.
As janelas afetadas por dados ausentes foram excluídas.

O critério fixado antes da busca foi a média, entre horizontes, de
**0,5 × MAE + 0,5 × RMSE, ambos em kWh**. Os dois períodos têm o mesmo peso.
Não se divide o erro pela duração do horizonte: os erros de 1 hora, naturalmente
maiores em kWh, têm maior influência absoluta neste critério que no experimento anterior.
O objetivo e a escolha da melhor época da rede passam a considerar a unidade física.

As fronteiras iniciais dos dois períodos de validação são
{manifest['boundaries']['fold1']['start']} e {manifest['boundaries']['fold2']['start']}.
O conjunto final contém {len(predictions):,} previsões, de
{predictions.time.min()} até {predictions.time.max()}.
Os alvos de uma janela de treino terminam antes do início da validação correspondente.

**Limite da avaliação:** o período de teste já havia sido examinado na análise anterior.
Ele é uma comparação histórica, não um novo teste independente. A seleção desta busca
usou somente as duas validações, e foi salva em `selection.json` antes da comparação
final. Confirmação independente exige dados ainda não examinados. Os resultados se
referem a uma residência e uma semente.

## O que foi ajustado

- Árvores: 15/31/63 folhas, 200–600 iterações, taxas 0,04–0,08,
  regularização L2 de 5–20, mínimo de 40–100 observações por folha e perdas MSE/MAE.
- Entradas das árvores: contexto anterior e, nas variantes ampliadas, trajetória
  da potência nos 24 blocos observados, tendência, médias recentes e amplitudes.
- Redes: LSTM de 32/64 unidades, camadas densas de 96/48/3, dropout de 10%/15%,
  taxas 0,0007/0,001 e perdas MSE, Huber ou mista calculadas em kWh.
- Parada antecipada e redução da taxa orientadas pelo critério de validação em kWh.
- Teste de combinações entre a melhor árvore e a melhor rede com pesos predefinidos
  de 25%, 50% e 75% para a árvore; a combinação só é escolhida se melhorar a validação.
- O PCA anterior não trouxe ganho preditivo; esta busca manteve as medições originais
  e as features causais. PCA continua disponível no diagnóstico anterior.
- Treino cobre o histórico completo disponível antes de cada partição, com janelas
  a cada {manifest['search']['stride']} minutos; validações/teste usam cinco minutos.

## Busca na validação

{markdown_table(ranking)}

Combinações da melhor árvore com a melhor rede, avaliadas nas mesmas validações:

{markdown_table(blends)}

![Busca temporal](validation_search.png)
![Validação das redes](network_validation_kwh.png)

## Configuração final e retreinamento

```json
{selected_parameters}
```

Depois da seleção, os dois modelos escolhidos foram ajustados novamente em
**{final['train_samples']:,} janelas de todo o histórico anterior ao teste, incluindo
os períodos de validação**. O número de épocas final foi fixado pela média das
melhores épocas dos dois períodos. A sequência de taxas de aprendizado da segunda
validação foi reaplicada; não houve parada antecipada nem ajuste pelo teste.

O resultado final combina efeitos dos hiperparâmetros, features e inclusão da
validação no retreinamento. Não se deve atribuir todo o ganho exclusivamente a
uma dessas mudanças. As duas validações controlam melhor a comparação dos candidatos.

## Métricas no período histórico

{markdown_table(table[['model','horizonte','MAE','RMSE','MAPE_%','WAPE_%','R2']])}

![Erros em kWh](comparison_kwh.png)
![Previsões](predictions_kwh.png)

### Alterações em relação à análise anterior

Valores positivos indicam redução do erro; negativos indicam piora.

{markdown_table(improvements[improvements.reference == 'previous_hgb_full'])}

Também estão salvas as comparações com a LSTM original e com a LSTM da análise
anterior em `improvements.csv`. As previsões completas estão em `test_predictions.csv`.

## Arquivos e reprodução

- `final/model_config.json`: configuração e identificação do modelo selecionado.
- `final/{selection['best_tree']}.joblib`: árvore retreinada.
- `final/{neural}.keras` e `final/{neural}_transforms.joblib`: rede e transformadores.
- `search_trials.csv`, `validation_ranking.csv` e `fold*/`: histórico da busca.
- `manifest.json`: parâmetros, versões e hashes dos dados, código e referência.
- `prepared_samples.joblib`: cache das janelas para retomada da busca.

```powershell
.\\.venv\\Scripts\\python.exe tune.py --output outputs/experiments/nova_busca --reference "{manifest['search']['reference']}" --epochs {manifest['search']['epochs']} --train-stride {manifest['search']['stride']}
.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v
```

Use uma pasta nova. Para retomar uma busca interrompida, use a mesma pasta e os
mesmos parâmetros com `--resume`; os ajustes concluídos são reaproveitados.
"""
    report = output / "RELATORIO_AJUSTE.md"
    report.write_text(text, encoding="utf-8")
    return report

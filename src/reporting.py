"""Generate a Portuguese report and standalone plots from completed experiments."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def markdown_table(frame: pd.DataFrame) -> str:
    rows = ["| " + " | ".join(map(str, frame.columns)) + " |",
            "| " + " | ".join("---" for _ in frame.columns) + " |"]
    for values in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in values) + " |")
    return "\n".join(rows)


def generate_report(output: Path) -> Path:
    audit = json.loads((output / "dataset_audit.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    test = pd.read_csv(output / "test_metrics.csv")
    curve = pd.read_csv(output / "learning_curve.csv")
    predictions = pd.read_csv(output / "test_predictions.csv", parse_dates=["time"])
    winner, neural = selection["winner"], selection["best_neural"]
    pca_full = pd.read_csv(output / "pca_full_train.csv")
    pca_engineered = pd.read_csv(output / "pca_engineered_train.csv")
    engineered_95 = int(np.searchsorted(pca_engineered.cumulative_variance, .95)+1)
    # Persist the engineered PCA summary too, including for a run made by an older runner.
    engineered_summary = audit.setdefault("pca_engineered_train", {})
    engineered_summary.update({
        "features": len(pca_engineered), "components_95pct": engineered_95,
        "samples": manifest["sample_counts"]["neural_train"],
        "explained_variance_ratio": pca_engineered.explained_variance_ratio.tolist()})
    for threshold in (.90, .99):
        engineered_summary[f"components_{int(threshold*100)}pct"] = int(
            np.searchsorted(pca_engineered.cumulative_variance, threshold)+1)
    (output / "dataset_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    display = test[["model", "horizonte", "MAE", "RMSE", "MAPE_%", "WAPE_%", "R2"]]
    ranking = pd.DataFrame([{"Modelo": name, "RMSE médio (kW)": item["score"]}
                            for name, item in selection["candidates"].items()]).sort_values("RMSE médio (kW)")
    comparison = ""
    if "lstm_original_saved" in set(test.model):
        old = test[test.model == "lstm_original_saved"].set_index("horizonte")
        improvements = []
        for name in dict.fromkeys([winner, neural]):
            new = test[test.model == name].set_index("horizonte")
            for horizon in old.index:
                improvements.append({"Modelo": name, "Horizonte": horizon,
                                     "Redução MAE (%)": 100*(1-new.loc[horizon,"MAE"]/old.loc[horizon,"MAE"]),
                                     "Redução RMSE (%)": 100*(1-new.loc[horizon,"RMSE"]/old.loc[horizon,"RMSE"])})
        improvement_table = pd.DataFrame(improvements)
        improvement_table.to_csv(output / "improvements_vs_original.csv", index=False)
        comparison = "\n\nReduções em relação ao modelo original salvo (valores negativos indicam piora):\n\n" + markdown_table(improvement_table)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    models = list(test.model.unique())
    colors = ["#64736F", "#087F78", "#C19335", "#D86B50"]
    for ax, metric in zip(axes, ("MAE", "RMSE")):
        for i, name in enumerate(models):
            part = test[test.model == name]
            x = np.arange(len(part)) + (i-(len(models)-1)/2)*.18
            ax.bar(x, part[metric], width=.18, color=colors[i % len(colors)], label=name)
        ax.set(xticks=np.arange(3), xticklabels=["15 min", "30 min", "1 h"],
               ylabel=f"{metric} (kWh)", title=f"{metric} — mesmo período de teste")
        ax.grid(axis="y", alpha=.15)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "model_comparison.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    # A fixed first three-day display avoids selecting a visually favorable interval.
    preview = predictions.iloc[:3*288]
    for ax, horizon in zip(axes, ("15min", "30min", "1h")):
        ax.plot(preview.time, preview[f"actual_{horizon}"], color="#18332F", linewidth=1, label="Real")
        ax.plot(preview.time, preview[f"{winner}_{horizon}"], color="#087F78", linewidth=1, label=winner)
        ax.set(ylabel=f"{horizon} (kWh)")
        ax.legend(fontsize=8)
    axes[0].set_title("Previsão selecionada — primeiros três dias do teste")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output / "selected_predictions.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, name in zip(axes, ("lstm_raw", "lstm_features", "lstm_features_pca")):
        history = pd.read_csv(output / f"history_{name}.csv")
        ax.plot(np.arange(1, len(history)+1), history.loss, label="Treino")
        ax.plot(np.arange(1, len(history)+1), history.val_loss, label="Validação")
        ax.set(title=name, xlabel="Época", ylabel="MSE (alvos padronizados)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "neural_training_curves.png", dpi=180)
    plt.close(fig)

    original_score = curve.loc[curve.history == "original", "validation_score"].iloc[0]
    annual_score = curve.loc[curve.history == "365days", "validation_score"].iloc[0]
    full_score = curve.loc[curve.history == "full", "validation_score"].iloc[0]
    text = f"""# Diagnóstico e retreinamento da previsão de energia

O modelo selecionado pela validação foi **{winner}**. A melhor rede foi **{neural}**.
O teste foi calculado depois de salvar a seleção em `selection.json`.

## O dataset é suficiente?

O arquivo contém **{audit['total_rows']:,} registros**, de {audit['start']} a {audit['end']}.
Há {audit['missing_rows']:,} linhas com medições ausentes ({100*audit['missing_fraction']:.2f}%).
A maior lacuna tem **{audit['longest_gap_minutes']:,} minutos**, aproximadamente {audit['longest_gap_minutes']/1440:.2f} dias.
O treino original continha apenas {audit['original_train_rows']:,} minutos
({audit['original_train_days']:.2f} dias) e **{100*audit['original_train_missing_fraction']:.2f}%** deles eram ausentes antes da interpolação.
O pipeline original interpolava sem limite, inclusive usando valores posteriores.
Também reutilizava um cache truncado e ignorava aumentos de `--max-rows`; esse erro foi corrigido.

Mantendo o mesmo modelo HGB, as mesmas features e o mesmo período de validação:

{markdown_table(curve)}

Passar do recorte original para um ano reduziu o erro de validação em
**{100*(1-annual_score/original_score):.2f}%**. Passar de um ano para todo o histórico anterior
à validação trouxe mais **{100*(1-full_score/annual_score):.2f}%**.
Isso sustenta usar mais cobertura temporal que o recorte inicial. O ganho marginal menor
depois de um ano sugere retornos decrescentes para esse modelo e essa validação.
**Não demonstra suficiência universal**: quantidade e estações do ano variam juntas nesta curva;
os minutos/janelas se sobrepõem e não são observações independentes.
O conjunto descreve **uma residência**, portanto não permite concluir desempenho em outras casas.
Para uma conclusão mais forte, usar múltiplas validações cronológicas em diferentes estações
e repetir as redes com outras sementes. Novos dados de outras residências seriam necessários
para avaliar generalização entre domicílios.

![Curva de aprendizado](learning_curve.png)

## Variância, correlação e PCA

Todas as sete medições foram analisadas, além das {len(pca_engineered)} features de contexto.
Potência ativa e corrente têm correlação **{audit['power_current_correlation']:.6f}** no treino completo.
Depois de padronizar cada variável, cinco componentes das sete medidas preservam
**{100*pca_full.cumulative_variance.iloc[4]:.2f}%** da variância e seis preservam
**{100*pca_full.cumulative_variance.iloc[5]:.2f}%**.
Nas features de contexto, **{engineered_95} componentes** preservam pelo menos 95%.
O PCA é descritivo e não usa os alvos: alta variância explicada não garante menor erro de previsão.
Ele também não mede se o número de registros é suficiente.

O experimento `lstm_features_pca` aplica PCA de 95% **às sete medições da sequência**;
mantém o contexto de engenharia de features. O PCA das features de contexto foi realizado
para diagnóstico, sem reduzi-las automaticamente na rede.
Scaler, imputação das defasagens opcionais e PCA são ajustados exclusivamente no treino.
As estatísticas por variável, variâncias, correlações e pesos dos componentes estão em
`feature_statistics_*.csv`, `correlation_*.csv`, `pca_*.csv` e `pca_*_weights.csv`.

![PCA das medições](pca_full_train.png)
![PCA das features](pca_engineered_train.png)

## Mudanças e protocolo

- Alvos: energia efetivamente observada nos próximos 15, 30 e 60 minutos, em kWh.
- Nenhum alvo é interpolado. A linha temporal é mantida; janelas com medições ausentes
  no histórico ou nos 60 minutos futuros são excluídas igualmente dos candidatos novos.
- Histórico da rede: {manifest['lookback_minutes']} minutos, agrupados em médias de cinco minutos.
- Features: medições atuais, médias móveis, desvio padrão, mínimos/máximos, defasagens,
  consumo no mesmo intervalo do dia/semana anterior, calendário cíclico e carga não submedida.
- Redes: LSTM de 32 unidades, dropout de 10%, Dense 64/32/3, Adam 0,0005,
  clipnorm 1, early stopping e redução de learning rate. As versões com features
  concatenam o contexto à saída da LSTM; a versão raw usa apenas a sequência.
- As três redes usam as mesmas {manifest['sample_counts']['neural_train']:,} janelas de treino
  de até {manifest['neural_days']} dias, semente {manifest['seed']} e no máximo {manifest['epochs_limit']} épocas.
- As previsões neurais recebem a regra física predefinida de energia não negativa e
  acumulada não decrescente entre horizontes, aplicada na validação e no teste.
- Comparação adicional: persistência da potência média dos últimos cinco minutos e HGB.
- Escolha: menor média de `RMSE_kWh / duração_em_horas` na validação, dando peso igual
  aos três horizontes em uma unidade comparável. MAPE é reportado, mas pode aumentar
  muito com consumos pequenos; MAE, RMSE e WAPE complementam a análise.
- Validação: {manifest['partitions']['validation']['start']} a {manifest['partitions']['validation']['end']}.
- Teste: {manifest['partitions']['test']['start']} a {manifest['partitions']['test']['end']}.
- São {manifest['sample_counts']['validation']:,} janelas de validação e
  {manifest['sample_counts']['test']:,} de teste, com cadência de cinco minutos.
  Cada histórico e todos os seus alvos ficam dentro da respectiva partição.
- Os arquivos originais de modelo/métricas em `outputs/models` e `outputs/metrics`
  foram preservados. A LSTM salva foi reavaliada nas mesmas origens do teste.

## Comparação na validação

{markdown_table(ranking)}

`lstm_raw` versus `lstm_features` versus `lstm_features_pca` é a comparação controlada
entre as novas redes. A comparação com o modelo antigo muda também volume de dados,
limpeza, cadência e arquitetura; não permite atribuir todo o ganho a uma única alteração.

## Resultados no teste

{markdown_table(display)}{comparison}

![Comparação](model_comparison.png)
![Curvas das redes](neural_training_curves.png)
![Previsões](selected_predictions.png)

Os resultados representam esta execução com uma semente e um período de teste.
Nenhuma nova busca foi feita com base nos erros do teste. As métricas e previsões
por origem estão em `test_metrics.csv` e `test_predictions.csv`.

## Reproduzir

No diretório PI4, usando uma pasta de saída nova:

```powershell
.\\.venv\\Scripts\\python.exe experiments.py --output outputs/experiments/nova_execucao --epochs {manifest['epochs_limit']} --neural-days {manifest['neural_days']} --train-stride {manifest['train_stride']} --lookback {manifest['lookback_minutes']} --reference-rows {manifest['reference_rows']} --seed {manifest['seed']}
.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v
```

As versões e configurações usadas estão em `manifest.json`. Os modelos neurais e seus
transformadores estão nos pares `*.keras` / `*_transforms.joblib`; os modelos HGB em
`hgb_*.joblib`. `selection.json` identifica o modelo selecionado.

Referências metodológicas: [dataset UCI](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption),
[PCA do scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
e [curvas de aprendizado](https://scikit-learn.org/stable/modules/learning_curve.html).
"""
    report = output / "RELATORIO.md"
    report.write_text(text, encoding="utf-8")
    return report

# Arquivo de referência do relatório

Este repositório é uma cópia independente do projeto PI4, com código e evidências
dos experimentos. A versão citável é `v1.0-relatorio`.

- [Anexo A em texto](docs/ANEXO_A.md)
- [Anexo A compatível com Word](docs/ANEXO_A.rtf)
- [Como reproduzir o fluxo](docs/REPRODUCAO.md)
- [Fonte e identificação dos dados](docs/FONTE_DOS_DADOS.md)
- `ARQUIVOS_SHA256.csv`: inventário da versão arquivada.

Os experimentos arquivados e seus modelos são versionados nesta entrega.
Novas pastas de resultados permanecem ignoradas pelo Git até inclusão explícita.
O dataset bruto e o ambiente virtual são reconstruídos conforme a documentação.

---

# Previsão de Consumo de Energia com LSTM (15 min, 30 min e 1 h)

Pipeline completo para prever a energia elétrica residencial consumida nos
próximos 15 min, 30 min e 1 h usando uma rede LSTM, a partir do dataset
público da UCI **Individual Household Electric Power Consumption** (id=235).

## Retreinamento com diagnóstico e engenharia de features

### Busca de hiperparâmetros com objetivo em kWh

Para ajustar a LSTM e o HistGradientBoosting em duas validações cronológicas e
retreinar as configurações escolhidas com todo o histórico anterior ao teste:

```powershell
.\.venv\Scripts\python.exe tune.py --output outputs/experiments/nova_busca --epochs 35 --train-stride 30
```

O critério combina MAE e RMSE em **kWh**, inclusive para selecionar a melhor época
da rede. São comparadas cinco configurações de árvores, três de LSTM e combinações
entre as melhores. Os modelos finais ficam em `final/`, junto de
`model_config.json`; o relatório é `RELATORIO_AJUSTE.md`.

O argumento `--reference` aponta para a execução anterior (padrão:
`outputs/experiments/2026-09-26`). Para retomar uma busca interrompida, repita
o comando com os mesmos argumentos e `--resume`. A retomada verifica dados,
parâmetros, código do treinamento e versões das bibliotecas.

O período final já foi examinado no trabalho anterior: essa comparação histórica
não deve ser apresentada como um novo teste independente. A seleção desta busca
usa somente as validações.

Na busca concluída, a combinação **50% HGB + 50% LSTM com Huber** foi selecionada.
O MAE de 1 hora caiu de **0,3056 para 0,2758 kWh** frente ao melhor modelo da
comparação inicial, e o R² passou de 0,7048 para 0,7378. Houve redução de MAE e
RMSE também em 15 e 30 minutos. A rede isolada foi retreinada com 64 unidades,
perda Huber em kWh e 22 épocas.

- [Resultados do ajuste e hiperparâmetros finais](docs/ajuste_hiperparametros.md)
- [Relatório completo da busca](outputs/experiments/2026-09-27-tuning/RELATORIO_AJUSTE.md)
- [Configuração dos modelos finais](outputs/experiments/2026-09-27-tuning/final/model_config.json)

### Diagnóstico e comparação inicial

Para executar a comparação atualizada, use uma **pasta de saída nova**:

```powershell
.\.venv\Scripts\python.exe experiments.py --output outputs/experiments/nova_execucao --epochs 25
```

Esse comando audita o dataset completo, gera PCA e estatísticas de todas as
variáveis, compara volumes de histórico, treina três variantes da LSTM e modelos
HistGradientBoosting e seleciona pela validação antes de avaliar o teste.
As lacunas permanecem explícitas: janelas com medições ausentes são excluídas,
sem fabricar alvos por interpolação. As features usam apenas informações anteriores
à previsão. Os resultados antigos são preservados e copiados como referência.

Opções principais: `--neural-days 365`, `--train-stride 15`, `--lookback 120`,
`--reference-rows 200000` e `--seed 42`. `--reference-rows` fixa as fronteiras
de validação/teste a partir do recorte original; o histórico anterior pode ser usado
no treino. As comparações de volume usam 30 dias, o recorte original, um ano e
todo o histórico; as redes usam o período configurado em `--neural-days`.
O teste sempre tem cadência de cinco minutos.

Resultados desta execução: a nova LSTM reduziu o MAE em **8,8%, 10,8% e 13,1%**
nos horizontes de 15, 30 e 60 minutos. O melhor candidato foi o HGB com histórico
completo, com reduções de **20,2%, 17,1% e 15,6%**. PCA não melhorou a rede nesta
execução. Isso se refere a uma residência e a um período de teste fixo.

- [Resultados e interpretação](docs/retreinamento.md)
- [Relatório completo, métricas e gráficos](outputs/experiments/2026-09-26/RELATORIO.md)
- Código: `src/features.py`, `src/diagnostics.py`, `src/experiments.py` e `src/reporting.py`.
- Cada execução salva `RELATORIO.md`, `manifest.json`, `selection.json`, métricas,
  previsões, curvas de aprendizado, PCA, modelos e transformadores na pasta escolhida.
- `legacy_provenance.json` registra os hashes da referência; novas execuções copiam
  os artefatos antigos antes do treino. A primeira execução registrou a cópia após
  terminar e confirmou que as métricas antigas foram reproduzidas.

Verificação:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

O fluxo `main.py`, descrito abaixo, é a implementação original para referência.
Ele mantém a interpolação antiga; use `experiments.py` para o novo protocolo.

## Estrutura

```
PI4/
├── src/
│   ├── config.py         # caminhos, hiperparâmetros, colunas, horizontes
│   ├── data_loader.py     # download/cache do dataset via ucimlrepo
│   ├── preprocessing.py   # limpeza, interpolação, split cronológico
│   ├── scaling.py         # normalização (StandardScaler ajustado no treino)
│   ├── windowing.py       # janelas -> energia acumulada futura em kWh
│   ├── model.py           # arquitetura da LSTM (três horizontes de consumo)
│   ├── train.py           # loop de treino, callbacks, curva de perda
│   └── evaluate.py        # métricas (MAE/RMSE/MAPE/R2) + gráficos
├── main.py                 # orquestra o pipeline ponta a ponta
├── data/                    # cache local dos dados (raw/processed)
└── outputs/
   ├── models/              # modelo treinado e scalers de features/alvos
    ├── metrics/             # test_metrics.csv / .json
   └── figures/             # gráficos, tabela de métricas e ficha técnica
```

## Como executar

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

Parâmetros opcionais:

```powershell
.\.venv\Scripts\python.exe main.py --epochs 50 --lookback 60 --max-rows 200000
```

- `--max-rows`: limita o dataset às últimas N amostras (1 min/linha) para acelerar
  experimentos. Para treinar com a série completa (~2M linhas / 4 anos), edite
  `MAX_ROWS = None` em `src/config.py`.
- `--force-download`: ignora o cache e baixa o dataset novamente.

## O que o pipeline faz

1. **Dados**: baixa (ou usa cache) o dataset de consumo (medições a cada 1 minuto:
   potência ativa/reativa, tensão, corrente, 3 submedições).
2. **Limpeza**: converte marcadores `?` em `NaN`, interpola por tempo, ordena por
   data/hora.
3. **Split cronológico**: 70% treino / 15% validação / 15% teste (sem embaralhar,
   para não vazar informação futura).
4. **Normalização**: scalers de features e de alvos ajustados apenas no treino.
5. **Janelamento e alvo energético**: usa os últimos `LOOKBACK_MINUTES`
   (padrão 120) minutos como entrada, amostrando uma janela a cada 5 minutos
   para reduzir a redundância de janelas quase idênticas. Para cada horizonte, integra a potência
   ativa média dos minutos futuros após a janela: `energia (kWh) = soma(kW) / 60`.
   As saídas são o consumo acumulado nos próximos 15 min, 30 min e 1 h; nenhum
   minuto previsto pertence à janela que a rede já observou.
6. **Modelo**: LSTM empilhada (64→32 unidades) + dropout + camada densa com 3
   saídas, treinada com MSE/Adam, `EarlyStopping` e `ReduceLROnPlateau`.
7. **Avaliação**: MAE, RMSE, MAPE e R² por horizonte, salvos em
   `outputs/metrics/`, além de 3 figuras prontas para o relatório em
   `outputs/figures/`:
   - `training_history.png` — curvas de perda/MAE por época;
   - `predictions_vs_actual.png` — energia real vs. prevista em kWh;
   - `scatter_residuals.png` — dispersão real×previsto e resíduos em kWh;
   - `metrics_summary.png` — tabela de métricas por horizonte;
   - `experiment_summary.png` — ficha técnica dos dados e hiperparâmetros.

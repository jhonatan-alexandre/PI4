# Ajuste de hiperparâmetros e retreinamento em kWh

Foram treinadas cinco configurações de árvores e três da LSTM em dois períodos
cronológicos: 16 ajustes de busca. As configurações selecionadas passaram por
mais dois treinamentos finais, incluindo os períodos de validação e mantendo
todos os alvos anteriores ao teste.

O modelo escolhido foi a média de **50% HistGradientBoosting e 50% LSTM com
perda Huber**. Os pesos foram definidos na validação e fixados antes da comparação
histórica. Não foram ajustados a partir dos resultados abaixo.

## Resultados frente ao melhor modelo da análise anterior

Todos os erros estão em kWh, nas mesmas 5.965 janelas de previsão.

| Horizonte | MAE anterior | MAE ajustado | RMSE anterior | RMSE ajustado | R² ajustado |
| --- | --- | --- | --- | --- | --- |
| 15 min | 0,05344 | **0,05116** | 0,09133 | **0,08690** | 0,8571 |
| 30 min | 0,12932 | **0,11895** | 0,20462 | **0,19363** | 0,8052 |
| 1 h | 0,30564 | **0,27584** | 0,44592 | **0,42025** | 0,7378 |

A LSTM isolada também melhorou frente à LSTM da análise anterior:

| Horizonte | MAE LSTM anterior | MAE LSTM ajustada | RMSE LSTM anterior | RMSE LSTM ajustada |
| --- | --- | --- | --- | --- |
| 15 min | 0,06112 | **0,05559** | 0,09982 | **0,09247** |
| 30 min | 0,13921 | **0,12359** | 0,21398 | **0,20106** |
| 1 h | 0,31483 | **0,27439** | 0,45650 | **0,42990** |

O modelo de referência inicial, anterior a todas as melhorias, tinha MAE de
1 hora de 0,36227 kWh e R² de 0,6128. As tabelas acima usam como referência
os modelos já melhorados na análise anterior, para mostrar o ganho adicional.

## Hiperparâmetros escolhidos

**Árvores:** HistGradientBoosting, 600 iterações, 31 folhas, taxa de aprendizado
0,04, L2 de 20, mínimo de 100 amostras por folha e perda quadrática.
Além das 34 features de contexto, usa 24 valores históricos de potência
e 21 medidas de tendência, médias recentes e amplitude por canal.

**Rede:** LSTM de 64 unidades, Dense 96/48/3, dropout de 15%, Adam com taxa
inicial 0,001, clipnorm 1 e perda Huber com delta de 0,2 kWh. A perda converte
os resíduos dos alvos padronizados de volta para kWh antes do cálculo.

As melhores épocas das duas validações foram 24 e 20. O retreinamento final
usou **22 épocas**, com a sequência de taxas da segunda validação. Não houve
parada antecipada ou escolha de épocas baseada no teste.

Os dois modelos finais foram ajustados em **66.894 janelas** de todo o histórico
disponível antes do teste, com cadência de 30 minutos. Cada janela contém
120 minutos observados, agrupados em blocos de cinco minutos. As validações
e o teste usam uma previsão a cada cinco minutos. As janelas com dados
ausentes permanecem excluídas, sem interpolar alvos.

## Seleção e limites da comparação

O critério de seleção foi a média entre horizontes de
**0,5 × MAE + 0,5 × RMSE, em kWh**, com o mesmo peso para os dois períodos.
O modelo combinado obteve 0,203113 kWh, frente a 0,209908 do modelo de referência
retreinado na mesma busca: redução de **3,24%** na validação.

Como o critério usa kWh diretamente, os erros de horizontes longos têm maior
influência absoluta que no critério anterior, que dividia pela duração.
O PCA não havia mostrado ganho preditivo e não foi imposto ao novo modelo.

O período final já foi examinado na análise anterior. Trata-se de uma
**comparação histórica, não de um novo teste independente**. A confirmação
independente requer novos dados. Os resultados se referem a uma residência
e uma semente; incluem o efeito dos novos hiperparâmetros, features e da
inclusão da validação no retreinamento final.

## Reproduzir e utilizar

```powershell
.\.venv\Scripts\python.exe tune.py --output outputs/experiments/nova_busca --epochs 35 --train-stride 30
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Use uma pasta de saída nova. O novo fluxo é `tune.py`; `main.py` permanece como
referência do treinamento original. A busca é retomável com `--resume` se os
dados, parâmetros, código e bibliotecas permanecerem iguais.

`src.tuning.predict_selected(output, samples)` carrega os arquivos finais e
retorna previsões em kWh, na ordem 15 min / 30 min / 1 h. `samples` deve conter
a sequência e o contexto produzidos por `src.features.make_samples`.

- [Relatório completo](../outputs/experiments/2026-09-27-tuning/RELATORIO_AJUSTE.md)
- [Comparação visual em kWh](../outputs/experiments/2026-09-27-tuning/comparison_kwh.png)
- [Métricas completas](../outputs/experiments/2026-09-27-tuning/test_metrics.csv)
- [Configuração final](../outputs/experiments/2026-09-27-tuning/final/model_config.json)
- [Decisão registrada antes do teste](../outputs/experiments/2026-09-27-tuning/selection.json)

Os modelos, transformadores, previsões e gráficos estão em
`outputs/experiments/2026-09-27-tuning/`. O diretório `outputs/` é ignorado pelo
Git; os artefatos devem acompanhar o relatório ao compartilhar o trabalho.

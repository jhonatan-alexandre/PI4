# Análise do dataset, PCA e retreinamento

A nova LSTM melhorou os três horizontes no mesmo conjunto de teste. O melhor
modelo selecionado na validação foi o HistGradientBoosting com engenharia de
features e todo o histórico anterior à validação.

## Comparação no teste

Os modelos foram comparados nas mesmas 5.965 janelas de 6 a 26 de novembro de
2010. A escolha dos modelos usou apenas a validação, de 16 de outubro a
6 de novembro de 2010.

| Horizonte | MAE original (kWh) | MAE nova LSTM | MAE HGB | R² original | R² nova LSTM | R² HGB |
| --- | --- | --- | --- | --- | --- | --- |
| 15 min | 0,06701 | 0,06112 | 0,05344 | 0,7916 | 0,8114 | 0,8421 |
| 30 min | 0,15607 | 0,13921 | 0,12932 | 0,7219 | 0,7622 | 0,7825 |
| 1 h | 0,36227 | 0,31483 | 0,30564 | 0,6128 | 0,6906 | 0,7048 |

Redução do MAE da nova LSTM: **8,8%, 10,8% e 13,1%**.
Redução do MAE do HGB: **20,2%, 17,1% e 15,6%**.

As métricas antigas foram reproduzidas a partir do modelo salvo; os arquivos
anteriores foram preservados. O ranking de validação ficou registrado antes
de calcular o teste. Os resultados descrevem uma execução e não uma garantia
para outras épocas ou residências.

## O dataset é suficiente?

O arquivo completo contém **2.075.259 minutos**, quase quatro anos, de uma
única residência. O projeto utilizava apenas os últimos 200 mil registros,
dos quais 140 mil, aproximadamente **97 dias**, entravam no treino.

Foram encontradas 25.979 linhas com medições ausentes no conjunto completo.
No treino original, **8,9% dos minutos eram ausentes e tinham sido
interpolados**. A maior lacuna tinha 7.226 minutos, cerca de cinco dias.
O cache também ignorava alterações no limite de linhas; esse erro foi corrigido.

Uma curva com o mesmo HGB, as mesmas features e a mesma validação mostrou:

| Histórico de treino | Janelas válidas | RMSE médio equivalente em kW |
| --- | --- | --- |
| 30 dias | 2.515 | 0,5096 |
| Recorte original | 8.454 | 0,4787 |
| 365 dias | 33.681 | 0,4377 |
| Todo o histórico anterior | 131.804 | 0,4320 |

Um ano melhorou o erro em **8,6%** sobre o recorte inicial; o histórico completo
acrescentou aproximadamente **1,3%**. Há evidência de que o recorte original
limitava o desempenho e de que existe ganho marginal menor depois de um ano
nesta comparação. A curva mistura aumento de quantidade e cobertura de estações;
as janelas sobrepostas não são amostras independentes.

O dataset sustenta um estudo de previsão para essa residência. Para afirmar
suficiência com mais segurança, faltam validações temporais em outras estações
e repetições com outras sementes. Para prever outras casas, seriam necessários
dados de outras casas.

## O que o PCA mostrou

As sete medições têm redundância: a correlação entre potência ativa e corrente
é aproximadamente **0,999** no treino completo. Após padronização:

- Cinco componentes preservam **96,9%** da variância das sete medições.
- Seis preservam **99,99%**.
- Entre as 34 features de contexto, 17 componentes preservam ao menos 95%.

PCA descreve a variância das entradas, não a quantidade de dados necessária nem
a capacidade de prever energia. O teste controlado das novas redes mostrou
RMSE médio equivalente na validação de 0,4781 sem contexto, **0,4684 com features**
e 0,4690 com features e PCA. Portanto, a rede selecionada usa as features
sem reduzir a sequência por PCA.

O PCA foi testado nas sete variáveis da sequência; a redução das 34 features
de contexto foi apenas analisada, sem aplicação automática ao modelo.

## O que mudou

- Features de horário, dia da semana e época do ano com codificação cíclica.
- Médias móveis, dispersão, mínimos/máximos e defasagens da potência.
- Energia no mesmo intervalo do dia e da semana anteriores e carga não submedida.
- Histórico de 120 minutos agrupado em 24 blocos de cinco minutos.
- LSTM de 32 unidades, dropout de 10%, camadas densas 64/32/3 e contexto adicional.
- Adam com taxa 0,0005, controle da norma do gradiente, redução da taxa e early stopping.
- Exclusão de janelas com medições ausentes no histórico/alvo, sem interpolar energia real.
- Transformadores ajustados só no treino; fronteiras temporais fixas para comparação.
- Métricas MAE, RMSE, MAPE, WAPE e R², persistência como referência e HGB como alternativa.
- Testes de alinhamento, ausência de vazamento, cache, transformadores e cópia da referência.

A comparação entre as três redes novas mantém os demais fatores constantes.
A comparação com a rede antiga muda arquitetura, dados, limpeza e cadência,
portanto não isola a contribuição individual de cada alteração.

## Executar e consultar

```powershell
.\.venv\Scripts\python.exe experiments.py --output outputs/experiments/nova_execucao --epochs 25
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Use uma pasta nova. Os modelos anteriores não são sobrescritos.
O novo fluxo é `experiments.py`; `main.py` mantém o fluxo original para referência.

- [Relatório completo](../outputs/experiments/2026-09-26/RELATORIO.md)
- [Comparação visual](../outputs/experiments/2026-09-26/model_comparison.png)
- [Curva de aprendizado](../outputs/experiments/2026-09-26/learning_curve.png)
- [PCA das medições](../outputs/experiments/2026-09-26/pca_full_train.png)
- [PCA das features](../outputs/experiments/2026-09-26/pca_engineered_train.png)
- [Métricas](../outputs/experiments/2026-09-26/test_metrics.csv)
- [Modelos selecionados](../outputs/experiments/2026-09-26/selection.json)

Os artefatos em `outputs/` são locais e ignorados pelo Git; devem acompanhar o
relatório ao compartilhar o trabalho. `manifest.json` registra a configuração e
as versões. `legacy_provenance.json` contém os hashes do modelo de referência.

Referências: [dataset UCI](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption),
[PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
e [curvas de aprendizado](https://scikit-learn.org/stable/modules/learning_curve.html).

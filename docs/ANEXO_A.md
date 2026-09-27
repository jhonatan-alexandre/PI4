# ANEXO A – REPOSITÓRIO DO FLUXO DE PREVISÃO DE CONSUMO DE ENERGIA

O desenvolvimento computacional deste trabalho foi organizado em um repositório
Git específico do Projeto Integrador IV, denominado **pi4-consumo-energia**.
A versão de referência para o relatório é identificada pela tag
**v1.0-relatorio**. Ela reúne o código-fonte, os testes automatizados, a
documentação metodológica, as configurações dos experimentos, os modelos
treinados, as métricas e os gráficos utilizados na análise.

O endereço de hospedagem remota ainda não foi definido. Nesta etapa, a versão
está preservada em um repositório Git local e será acompanhada de uma cópia
portátil do histórico (`.bundle`) e de um arquivo ZIP. O identificador completo
do commit está registrado no arquivo `REGISTRO_DE_VERSAO.json` da entrega.

O fluxo documentado compreende: obtenção e caracterização dos dados; tratamento
de valores ausentes; construção dos alvos de energia acumulada em kWh para os
horizontes de 15, 30 e 60 minutos; análise de variância, correlações e componentes
principais; engenharia de atributos temporais e estatísticos; divisão
cronológica dos dados; treinamento das redes LSTM e dos modelos de árvores;
ajuste de hiperparâmetros; comparação em duas validações temporais; seleção da
combinação de modelos; retreinamento final e avaliação no período histórico.

Os dados são provenientes do conjunto *Individual Household Electric Power
Consumption*, disponibilizado pela UCI Machine Learning Repository. Os arquivos
brutos são obtidos pelo código do projeto e não integram o histórico Git; a
fonte, a forma de obtenção e o hash do arquivo utilizado estão documentados no
repositório. Os dados derivados apresentados nos resultados mantêm a referência
à fonte original.

O modelo selecionado combina, com pesos iguais, HistGradientBoosting e uma
LSTM com perda Huber calculada a partir de resíduos em kWh. A busca avaliou oito
configurações em dois períodos cronológicos, seguida do retreinamento dos dois
modelos selecionados em 66.894 janelas anteriores ao teste. A rede final possui
64 unidades LSTM e foi treinada por 22 épocas.

| Horizonte | MAE (kWh) | RMSE (kWh) | R² |
| --- | --- | --- | --- |
| 15 minutos | 0,05116 | 0,08690 | 0,8571 |
| 30 minutos | 0,11895 | 0,19363 | 0,8052 |
| 60 minutos | 0,27584 | 0,42025 | 0,7378 |

As métricas correspondem às mesmas 5.965 janelas do período histórico de
comparação. Esse período já havia sido examinado em análises anteriores; assim,
os resultados não constituem um novo teste independente. A escolha dos
hiperparâmetros e dos pesos da combinação utilizou somente as validações
cronológicas. O estudo contempla uma residência e uma semente aleatória.

Para possibilitar a conferência do trabalho, o repositório inclui as versões
das dependências, as instruções de execução, os resultados por experimento,
os manifestos e os testes de alinhamento temporal, conversão para kWh,
ausência de vazamento nos transformadores e carregamento dos modelos.
O inventário `ARQUIVOS_SHA256.csv` permite verificar a integridade dos arquivos
da versão arquivada.

**Referência dos dados:** HEBRAIL, G.; BERARD, A. *Individual Household Electric
Power Consumption*. UCI Machine Learning Repository, 2006.
Disponível em: <https://doi.org/10.24432/C58K54>.

**Registro para a versão final do relatório:** após a publicação, acrescentar
a URL do repositório e o link permanente para a tag `v1.0-relatorio`, mantendo
o commit identificado em `REGISTRO_DE_VERSAO.json`.

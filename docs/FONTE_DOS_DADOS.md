# Fonte e identificação dos dados

- Conjunto: Individual Household Electric Power Consumption.
- Autores: Georges Hebrail e Alice Berard.
- Repositório: UCI Machine Learning Repository, identificador 235.
- DOI: https://doi.org/10.24432/C58K54.
- Página: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption.
- Licença informada pela UCI: Creative Commons Attribution 4.0 International
  (https://creativecommons.org/licenses/by/4.0/).
- Origem dos resultados: medições de uma residência, com frequência de um minuto.
- Arquivo CSV utilizado: `data/raw/household_power_consumption.csv`.
- Tamanho observado: 128.862.195 bytes.
- SHA-256 do CSV utilizado:
  `d8c4a0f4d6a47358c79f1670c78013d494d20db8f64a308bd5fd69290ead3498`.

O CSV é produzido pelo carregador do projeto a partir do dataset UCI. Uma nova
serialização pode produzir bytes diferentes; o hash acima identifica o arquivo
exato usado nas execuções arquivadas, não todas as distribuições do dataset.

Os alvos derivados são a soma da potência ativa média de cada minuto futuro
dividida por 60, em kWh. Os experimentos atualizados preservam a linha temporal
e excluem janelas com medições ausentes nos históricos ou alvos. As previsões,
estatísticas e gráficos em `outputs/` são produtos dessas transformações.

Referência: HEBRAIL, G.; BERARD, A. *Individual Household Electric Power
Consumption* [Dataset]. UCI Machine Learning Repository, 2006.
https://doi.org/10.24432/C58K54.

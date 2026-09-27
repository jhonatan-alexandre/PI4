# Reprodução e preservação do fluxo

Esta entrega contém um retrato do fluxo completo, incluindo os resultados e
modelos das duas etapas de melhoria. O código original de referência está em
`main.py`; o diagnóstico ampliado em `experiments.py`; e a busca final em
`tune.py`. Para novas análises, use os dois últimos comandos.

## Ambiente

Ambiente observado: Windows, Python 3.12.6, TensorFlow 2.19.1, Keras 3.15.1,
scikit-learn 1.9.1, NumPy 2.1.3 e pandas 3.0.6. O treinamento ocorreu em CPU.
`requirements-lock.txt` registra as versões instaladas. Resultados numéricos
podem variar com o hardware e as bibliotecas.

No diretório do repositório:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Conferir os resultados sem retreinar

Os arquivos em `outputs/` incluem as métricas, previsões, gráficos, modelos e
transformadores efetivamente utilizados. Os relatórios principais são:

- `outputs/experiments/2026-09-26/RELATORIO.md`;
- `outputs/experiments/2026-09-27-tuning/RELATORIO_AJUSTE.md`;
- `outputs/experiments/2026-09-27-tuning/final/model_config.json`.

O diretório de cada validação contém seus parâmetros, métricas e previsões.
As referências copiadas para os experimentos preservam os modelos anteriores.
Não é necessário baixar o dataset para ler essas evidências ou executar os testes.

## Executar novamente

```powershell
.\.venv\Scripts\python.exe experiments.py --output outputs/experiments/nova_analise --epochs 25
.\.venv\Scripts\python.exe tune.py --output outputs/experiments/nova_busca --reference outputs/experiments/2026-09-26 --epochs 35 --train-stride 30
```

O segundo comando usa a referência arquivada, reproduzindo a comparação da busca.
Para comparar com uma análise recém-executada, substitua `--reference` pela pasta
`outputs/experiments/nova_analise`.

O dataset público é obtido automaticamente quando o cache local está ausente.
O download requer acesso à internet. As pastas de saída devem ser novas para
preservar os resultados anteriores. A busca pode ser retomada com `--resume`
somente se seus dados, código, parâmetros e bibliotecas permanecerem iguais.
Os caches de amostras preparados não são distribuídos; são reconstruídos a partir
do dataset. Para uma entrega arquivada, prefira iniciar uma nova pasta de execução.

## Integridade e escopo

`ARQUIVOS_SHA256.csv` registra os arquivos da entrega, seus tamanhos e hashes,
exceto o próprio inventário. O arquivo não acompanha alterações futuras: ele
identifica o conteúdo arquivado nesta versão.

O repositório inclui código, testes, documentação, resultados e modelos.
Não inclui o ambiente `.venv`, os dados brutos, caches processados ou documentos
pessoais de outros trabalhos. Os resultados arquivados são versionados; saídas
de novos experimentos permanecem ignoradas pelo Git até inclusão explícita.

Os manifestos das execuções preservam os hashes registrados na época, inclusive
anotações de manutenção posterior. Eles não devem ser reescritos para aparentar
que todo o código atual é idêntico ao executado anteriormente.

O teste final é uma comparação histórica já examinada. Novos dados são necessários
para uma avaliação independente; não se deve usar o período histórico para ajustar
novamente os hiperparâmetros e apresentá-lo como teste inédito.

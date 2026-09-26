# PI4

Este repositório pode ser usado para demonstrar um fluxo simples de `git add` e verificação com `git status --short` no Windows PowerShell.

## Pré-requisitos

- Git for Windows instalado
- Uma cópia local deste repositório no computador de cada pessoa
- Ajustar o caminho local do repositório antes de executar o fluxo

## Fluxo para compartilhar com colegas

Cada colega deve adaptar o valor de `$repoPath` para a raiz local clonada deste repositório, onde ficam o arquivo `README.md` e a pasta `PI4`.

```powershell
$git = "C:\Program Files\Git\cmd\git.exe"
$repoPath = "C:\caminho\para\o\repositorio"

Set-Location $repoPath

& $git add -- "PI4"
& $git status --short
```

## O que o comando faz

1. Entra na pasta local do repositório
2. Adiciona a pasta `PI4` ao índice do Git
3. Exibe o status resumido para confirmar o resultado

## Saída esperada

Se a pasta `PI4` ainda não estiver rastreada, a saída esperada é:

```text
A  PI4/README.md
```

Se nada tiver mudado desde a última adição ou commit, o `git status --short` pode não exibir nenhuma linha.

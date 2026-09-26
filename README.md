# PI4

Este repositório pode ser usado para demonstrar um fluxo simples de `git add` e verificação com `git status --short` no Windows PowerShell.

## Pré-requisitos

- Git for Windows instalado
- Uma cópia local deste repositório no computador de cada pessoa
- Ajustar o caminho da raiz local do repositório antes de executar o fluxo

## Fluxo para compartilhar com colegas

Cada colega deve adaptar o valor de `$repoPath` para a raiz local deste repositório, isto é, a pasta que contém este `README.md`. Antes de continuar, confirme que a subpasta `PI4` existe dentro desse caminho.

Se `.\PI4` não existir depois do `Set-Location`, ajuste o valor de `$repoPath` antes de continuar.

```powershell
$git = "C:\Program Files\Git\cmd\git.exe"
$repoPath = "C:\caminho\para\uma-pasta-com-README-e-subpasta-PI4"

Set-Location $repoPath
Set-Content -Path ".\PI4\exemplo.txt" -Value "arquivo de teste"

& $git add -- "PI4"
& $git status --short
```

## O que o comando faz

1. Entra na raiz local do repositório
2. Cria um arquivo de teste não rastreado dentro da pasta `PI4`
3. Adiciona a pasta `PI4` ao índice do Git
4. Exibe o status resumido para confirmar o resultado

## Saída esperada

Assumindo que o repositório já estava limpo antes de criar o arquivo de teste acima e que a pasta `PI4` já está rastreada por causa do arquivo versionado `PI4/README.md`, a saída esperada é:

```text
A  PI4/exemplo.txt
```

Se quiser repetir a demonstração do zero, remova ou limpe o arquivo `PI4/exemplo.txt` antes de executar o fluxo novamente.

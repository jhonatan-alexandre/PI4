"""Copy a reviewed PI4 snapshot to a new, independent repository directory.

Usage: python scripts/package_repository.py --destination entrega/pi4-consumo-energia
Git initialization, commits and archives are intentionally separate operations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_rtf(markdown: str, destination: Path) -> None:
    """Create a Unicode RTF appendix that Word can open without extra packages."""
    def escape(text):
        result = []
        for char in text:
            if char in "\\{}":
                result.append("\\" + char)
            elif char == "\t":
                result.append("\\tab ")
            elif ord(char) > 127:
                encoded = char.encode("utf-16-le")
                for pos in range(0, len(encoded), 2):
                    unit = int.from_bytes(encoded[pos:pos+2], "little")
                    result.append(f"\\u{unit if unit < 32768 else unit-65536}?")
            else:
                result.append(char)
        return "".join(result)
    content = [r"{\rtf1\ansi\deff0\uc1{\fonttbl{\f0 Times New Roman;}}",
               r"\paperw11906\paperh16838\margl1701\margr1134\margt1701\margb1134\f0\fs24"]
    for block in markdown.strip().split("\n\n"):
        block = block.replace("**", "").replace("`", "")
        if block.startswith("# "):
            content.append(r"\pard\qc\b\sa240 " + escape(block[2:]) + r"\b0\par")
        elif block.startswith("|"):
            for line in block.splitlines():
                if set(line.replace(" ", "")) <= {"|", "-", ":"}:
                    continue
                row = "\t".join(cell.strip() for cell in line.strip("|").split("|"))
                content.append(r"\pard\ql\tx2400\tx4200\tx6000\sa100 " + escape(row) + r"\par")
        else:
            content.append(r"\pard\qj\sl360\slmult1\sa160 " + escape(" ".join(block.splitlines())) + r"\par")
    content.append("}")
    destination.write_text("\n".join(content), encoding="ascii")


def build(destination: Path) -> None:
    destination = destination.resolve()
    if not destination.is_relative_to(ROOT) or destination == ROOT:
        raise ValueError("Choose a new destination directory inside the project.")
    if destination.exists():
        raise FileExistsError(f"Destination already exists; nothing was overwritten: {destination}")
    destination.mkdir(parents=True)
    paths = [ROOT / name for name in ("README.md", "requirements.txt", "main.py", "experiments.py", "tune.py")]
    for folder in ("src", "tests", "docs", "scripts", "outputs"):
        for path in (ROOT / folder).rglob("*"):
            if (path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
                    and path.name != "prepared_samples.joblib"):
                paths.append(path)
    for path in paths:
        target = destination / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    lock = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    if any(" @ " in line or line.startswith("-e ") for line in lock.splitlines()):
        raise ValueError("Review local/URL dependencies before distributing the environment lock.")
    (destination / "requirements-lock.txt").write_text(lock, encoding="utf-8")
    (destination / ".gitignore").write_text(""".venv/
__pycache__/
*.pyc
.env
.env.*
data/raw/
data/processed/
entrega/
**/prepared_samples.joblib
outputs/*
!outputs/experiments/
outputs/experiments/*
!outputs/experiments/2026-09-26/
!outputs/experiments/2026-09-27-tuning/
!outputs/models/
!outputs/metrics/
!outputs/figures/
""", encoding="utf-8")
    (destination / ".gitattributes").write_text("""# Preserve the exact bytes identified by the archived experiment hashes.
* -text
*.rtf binary
*.png binary
*.keras binary
*.joblib binary
*.npy binary
""", encoding="utf-8")
    (destination / "data").mkdir(exist_ok=True)
    (destination / "data" / "README.md").write_text(
        "# Dados\n\nO dataset bruto e os caches não são versionados. Consulte "
        "[a fonte e os hashes](../docs/FONTE_DOS_DADOS.md). O carregador do projeto "
        "obtém os dados automaticamente quando necessário.\n", encoding="utf-8")
    source_note = """# Arquivo de referência do relatório

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

"""
    readme = destination / "README.md"
    readme.write_text(source_note + readme.read_text(encoding="utf-8"), encoding="utf-8")
    write_rtf((destination / "docs" / "ANEXO_A.md").read_text(encoding="utf-8"),
              destination / "docs" / "ANEXO_A.rtf")
    (destination / "AMBIENTE.json").write_text(json.dumps({
        "python": platform.python_version(), "platform": platform.platform(),
        "version_tag": "v1.0-relatorio", "source_scope": "PI4 only",
        "raw_dataset_sha256": "d8c4a0f4d6a47358c79f1670c78013d494d20db8f64a308bd5fd69290ead3498",
        "excluded": [".venv", "raw/processed data", "prepared_samples.joblib", "unrelated personal documents"]
    }, indent=2), encoding="utf-8")
    files = sorted(p for p in destination.rglob("*") if p.is_file())
    with (destination / "ARQUIVOS_SHA256.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["path", "bytes", "sha256"])
        for path in files:
            writer.writerow([path.relative_to(destination).as_posix(), path.stat().st_size, digest(path)])
    print(json.dumps({"repository": str(destination), "files": len(files)+1,
                      "size_mib": round(sum(p.stat().st_size for p in files)/1024**2, 2)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "entrega" / "pi4-consumo-energia")
    build(parser.parse_args().destination)

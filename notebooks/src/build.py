"""
Build notebooks from percent-format sources in notebooks/src/.

    python notebooks/src/build.py                      # build every notebooks/src/NN_*.py
    python notebooks/src/build.py 03_main_algorithm_analysis --execute

Cell markers: a line `# %%` starts a code cell, `# %% [markdown]` a markdown
cell whose lines are written as `# text` (the leading `# ` is stripped).
The .ipynb is written to notebooks/<name>.ipynb; --execute runs it in place
with nbclient (working directory notebooks/, same as opening it in Jupyter).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import nbformat

SRC = Path(__file__).resolve().parent
NB_DIR = SRC.parent
_MARK = re.compile(r"^# %%(?P<md>\s*\[markdown\])?\s*$")


def parse(text: str) -> list:
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        while buf and not buf[-1].strip():
            buf.pop()
        if kind == "markdown":
            body = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in buf)
            cells.append(nbformat.v4.new_markdown_cell(body))
        else:
            cells.append(nbformat.v4.new_code_cell("\n".join(buf)))

    for line in text.splitlines():
        m = _MARK.match(line)
        if m:
            flush()
            kind, buf = ("markdown" if m.group("md") else "code"), []
        elif kind is not None:
            buf.append(line)
    flush()
    return cells


def build(name: str, execute: bool, timeout: int) -> Path:
    src = SRC / f"{name}.py"
    nb = nbformat.v4.new_notebook()
    nb.cells = parse(src.read_text())
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    out = NB_DIR / f"{name}.ipynb"
    if execute:
        from nbclient import NotebookClient
        NotebookClient(nb, timeout=timeout, kernel_name="python3",
                       resources={"metadata": {"path": str(NB_DIR)}}).execute()
    nbformat.write(nb, out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="source stems (default: every NN_*.py)")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--timeout", type=int, default=7200, help="per-cell timeout [s]")
    a = ap.parse_args()
    names = a.names or sorted(p.stem for p in SRC.glob("[0-9][0-9]_*.py"))
    for n in names:
        print(build(n, a.execute, a.timeout))

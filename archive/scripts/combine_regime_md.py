"""
Assemble the per-regime markdown of notebooks 03 and 04 (week 30) into one file.

    dlent_venv/bin/python scripts/combine_regime_md.py small_cluster
        -> reports/week30/nb03_nb04_small_cluster.md

Inputs (written by the notebooks run with DLENT_REGIME=configs/regime_<tag>.yaml):
  reports/week30/<tag>_nb/nb03_numbers.md     (notebooks/03_main_algorithm_analysis_<tag>.ipynb)
  reports/week30/pareto_<tag>/nb04_tables.md  (notebooks/04_pareto_lambda_<tag>.ipynb)
The summary written by hand goes at the top (`--header FILE`, optional).
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W30 = ROOT / "reports" / "week30"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--header", default=None, help="markdown file with the hand-written summary")
    a = ap.parse_args()
    nb03 = (W30 / f"{a.tag}_nb" / "nb03_numbers.md").read_text()
    nb04 = (W30 / f"pareto_{a.tag}" / "nb04_tables.md").read_text()
    shift = lambda t: "\n".join(("#" + l) if l.startswith("#") else l for l in t.splitlines())   # one level down
    parts = [Path(a.header).read_text().rstrip() + "\n"] if a.header else []
    parts += ["\n---\n\n# Appendix A — notebook 03 tables (generated)\n", shift(nb03),
              "\n---\n\n# Appendix B — notebook 04 tables (generated)\n", shift(nb04)]
    out = W30 / f"nb03_nb04_{a.tag}.md"
    out.write_text("\n".join(parts) + "\n")
    print(out)


if __name__ == "__main__":
    main()

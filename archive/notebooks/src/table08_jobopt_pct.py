"""
Add "% of job OPT" to table08 (week 30). It is computed from the columns that nb03 Section 8 already wrote
(R <policy> $ / R job OPT (MILP) $ of the same configuration), so nb03 is not re-run; no other change.

    dlent_venv/bin/python notebooks/src/table08_jobopt_pct.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TAB = Path(__file__).resolve().parents[2] / "reports" / "week30" / "tables"
POLS = ("DLENT-RT", "StaticPolicy (P0)", "G1", "G2")


def add(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in df.columns if c.endswith("% of job OPT")])
    for p in POLS:
        df[f"{p} % of job OPT"] = 100 * df[f"R {p} $"] / df["R job OPT (MILP) $"]
    return df


def main():
    for name in ("table08_capacity_grid_sweep", "table08_capacity_grid_sweep_full"):
        path = TAB / f"{name}.csv"
        df = add(pd.read_csv(path, index_col=[0, 1]))
        df.to_csv(path)
    for f in (0.3, 0.5, 1.0):
        path = TAB / f"table08_capacity_grid_sweep_f{f}.csv"
        add(pd.read_csv(path, index_col=0)).to_csv(path)
    df = pd.read_csv(TAB / "table08_capacity_grid_sweep.csv", index_col=[0, 1])
    print(df[[f"{p} % of job OPT" for p in POLS]].to_string(float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()

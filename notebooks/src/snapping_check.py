"""
Hardware-profile snapping (week 30, read-only).

    dlent_venv/bin/python notebooks/src/snapping_check.py

Every A_ij in the model is the SNAPPED profile value: data._build_job_arrays sets
A = catalogue.centroids[catalogue.assign(A_raw)], and JobArrays.A is what the capacity sizing, the
drop filter, the physical gate (stream.A), the energy model (C_elec, K -> R_job), the bid cost base
and the type volume v = A_type * d_bar use. A_raw is used only to pick the profile.

For (k* 128, 32 profiles) and (k* 64, 16 profiles) on the committed default config (post-outlier H +
stream jobs, the set the catalogue is fitted on with hardware.fit_on = "full"): quantiles 0.5 / 0.9 /
0.99 of |A_snapped - A_raw| / A_raw per resource, and the share of jobs whose A changes by > 5%.
A depends on n_profiles only (k* is the bid grid); both are set as asked.
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks" / "src"))

from capacity_scan import committed_default
from dlent_rt.data import _read_csv, load_dataset
from dlent_rt.preprocess import fit_hardware_catalogue, handle_outliers

warnings.simplefilter("ignore")
GRIDS = {"k*=128, hw=32": (128, 32), "k*=64, hw=16": (64, 16)}


def main():
    base = committed_default()
    rows = []
    for name, (k, h) in GRIDS.items():
        cfg = replace(base, grid=replace(base.grid, k_star=k),
                      hardware=replace(base.hardware, n_profiles=h)).validate()
        col = cfg.data.columns
        a_cols = [col.requested_cpu, col.requested_ram]
        df, _ = handle_outliers(_read_csv(cfg), cfg, a_cols, col.duration_hours)
        A_raw = df[a_cols].to_numpy(dtype=float)
        A_fit = A_raw if cfg.hardware.fit_on == "full" else A_raw[: cfg.data.n0]
        cat = fit_hardware_catalogue(A_fit, cfg)
        A_snap = cat.centroids[cat.assign(A_raw)]
        ds = load_dataset(cfg)                           # the model's own A, for the check
        assert np.array_equal(np.vstack([ds.H.A, ds.stream.A]), A_snap)
        rel = np.abs(A_snap - A_raw) / A_raw
        row = {"grid": name, "jobs": len(A_raw), "profiles": cat.n_profiles}
        for i, r in enumerate(("CPU", "RAM")):
            for q in (0.5, 0.9, 0.99):
                row[f"{r} rel. change q{q}"] = float(np.quantile(rel[:, i], q))
            row[f"{r} > 5% (share)"] = float((rel[:, i] > 0.05).mean())
        row["any resource > 5% (share)"] = float((rel > 0.05).any(axis=1).mean())
        rows.append(row)
    df = pd.DataFrame(rows).set_index("grid")
    df.to_csv(ROOT / "reports" / "week30" / "tables" / "snapping_check.csv")
    print(df.T.to_string(float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main()

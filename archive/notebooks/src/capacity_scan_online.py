"""
Online runs at one fraction of the capacity scan (week 30, read-only): DLENT-RT, StaticPolicy (Phase 0),
G1, G2 and the job-level OPT (MILP, 600 s), seed 42, same stream / R_job / physical gate.

    dlent_venv/bin/python notebooks/src/capacity_scan_online.py 0.2

Config = capacity_scan.scan_config(f, drop=True). Everything comes from capacity_grid_sweep.run_one.
Writes reports/week30/tables/capacity_scan_online_f<f>.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks" / "src"))

from capacity_grid_sweep import run_one
from capacity_scan import scan_config


def main(f: float):
    out = run_one(scan_config(f, drop=True), f"f={f}", milp_time_limit=600.0)
    opt = out["R_job_opt"]
    for k in ("DLENT", "static", "G1", "G2"):
        out[f"pct_jobopt_{k}"] = 100 * out[f"R_{k}"] / opt
    path = ROOT / "reports" / "week30" / "tables" / f"capacity_scan_online_f{f}.json"
    path.write_text(json.dumps({k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in out.items()},
                               indent=1, default=float))
    for k, v in out.items():
        print(f"{k:<34} {v}")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.2)

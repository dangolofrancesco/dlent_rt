"""
The discretization cost of the grid (D6): two job-level MILPs on the stream arrivals of the first --days days, with
the same jobs, the same capacity checks (at every arrival, true durations), the same carbon map and s_C, lambda = 1,
that differ only in the value: MILP_true values each job at its own hourly bid b_j, MILP_bin at its bin's lower edge
L_k (benchmarks.discretization_milps). Each MILP on its own HiGHS instance with --time-limit seconds.

    dlent_venv/bin/python scripts/discretization_milp.py                                  # 7 days, 3,600 s, seed 42
    dlent_venv/bin/python scripts/discretization_milp.py --days 7 --time-limit 3600 --seed 42
        -> reports/week33/cache/discretization_milp.pkl

Uses configs/default.yaml at its capacity.f.
"""
from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "reports" / "week33" / "cache" / "discretization_milp.pkl"


def main(days: float = 7.0, time_limit: float = 3600.0, seed=None) -> dict:
    from dlent_rt import theory_eval as te
    from dlent_rt.config import default_config
    cfg = default_config()
    if seed is not None:
        cfg = replace(cfg, run=replace(cfg.run, seed=int(seed))).validate()
    r = te.window_discretization(cfg, days=(0.0, float(days)), time_limit=float(time_limit))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "wb") as fh:
        pickle.dump(r, fh)
    print(f"OPT_true {r['true']['best']:,.2f} (bound {r['true']['bound']:,.2f}), "
          f"OPT_bin {r['bin']['best']:,.2f} (bound {r['bin']['bound']:,.2f})")
    print(f"value loss {r['value_loss']:,.2f} $ ({r['value_loss_pct']:.2f}%), decision loss {r['decision_loss']:,.2f} $ "
          f"({r['decision_loss_pct']:.2f}%), grid worst case {r['worst_case_pct']:.2f}%")
    print(f"saved {OUT}")
    return r


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=float, default=7.0, help="window: the first DAYS days of the stream")
    ap.add_argument("--time-limit", type=float, default=3600.0, help="seconds per MILP (each its own HiGHS)")
    ap.add_argument("--seed", type=int, default=None, help="run.seed (default: the configured one)")
    a = ap.parse_args()
    main(a.days, a.time_limit, a.seed)

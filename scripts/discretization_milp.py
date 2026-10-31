"""
The discretization cost of the grid (D6): two job-level MILPs on the stream arrivals of the first --days days, with
the same jobs, the same capacity checks (at every arrival, true durations), the same carbon map and s_C, lambda = 1,
that differ only in the value: MILP_true values each job at its own hourly bid b_j, MILP_bin at its bin's lower edge
L_k (benchmarks.discretization_milps). Each MILP on its own HiGHS instance with --time-limit seconds. MILP_bin is
solved first and warm-starts MILP_true. Every solve prints our timer, the wall clock and HiGHS's run time: if the
machine sleeps, the wall clock and HiGHS's time run ahead of our timer (run it under caffeinate -i).

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
    for k in ("bin", "true"):
        o = r[k]
        print(f"MILP_{k}: best {o['best']:,.2f}, bound {o['bound']:,.2f}, gap {100 * o['gap']:.3f}%, {o['status']}; "
              f"time: our timer {o['runtime_s']:,.0f} s, wall clock {o['wall_s']:,.0f} s, HiGHS {o['highs_run_time_s']:,.0f} s"
              + (f"; warm start {o['warm_start']:,.2f}" if o.get("warm_start") is not None else ""), flush=True)
    (vl0, vl1), (vp0, vp1) = r["value_loss_range"], r["value_loss_range_pct"]
    (dl0, dl1), (dp0, dp1) = r["decision_loss_range"], r["decision_loss_range_pct"]
    print(f"value loss OPT_true - OPT_bin: {vl0:,.2f} .. {vl1:,.2f} $ ({vp0:.2f} .. {vp1:.2f}%)")
    print(f"decision loss OPT_true - MILP_bin's schedule at the true rewards ({r['bin_schedule_at_true']:,.2f}): "
          f"{dl0:,.2f} .. {dl1:,.2f} $ ({dp0:.2f} .. {dp1:.2f}%)")
    print(f"grid worst case 1 - 1/rho: {r['worst_case_pct']:.2f}%")
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

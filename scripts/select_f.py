"""
The f rule (D3) on H only: at each f, Phase 0 on H alone (data.load_history: the quota fixed point, C = f x D_bar,
hardware profiles on H; no stream job is read), then H's jobs replayed through the static ThresholdPolicy of that
Phase 0 state with capacity C.
  (a) tau > 0 (CPU or RAM) in the Phase 0 LP;
  (b) among those, the f whose '% of H's time with the cluster >= 95% full' is closest to the 20-30% band (ties: 25%).

    dlent_venv/bin/python scripts/select_f.py                              # f in theory_eval.F_SCAN, configured seed
    dlent_venv/bin/python scripts/select_f.py --f 0.3 0.4 0.5 --seed 42    # -> reports/week33/cache/f_scan_H.pkl

Prints the chosen f. It does not change configs/default.yaml.
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "reports" / "week33" / "cache" / "f_scan_H.pkl"


def main(f_grid=None, seed=None) -> dict:
    from dlent_rt import theory_eval as te
    from dlent_rt.config import default_config
    cfg = default_config()
    if seed is not None:
        cfg = replace(cfg, run=replace(cfg.run, seed=int(seed))).validate()
    out = []
    for f in (f_grid or te.F_SCAN):
        t = time.perf_counter()
        r = te.evaluate_f_H(cfg, float(f))
        r["runtime_s"] = time.perf_counter() - t
        out.append(r)
        print(f"f = {f}: tau = {[round(x, 6) for x in r['tau']]}, H >= 95% full {100 * r['full_time']:.1f}% of the time "
              f"({r['runtime_s']:.0f} s)", flush=True)
    res = dict(results=out, selected=te.select_f(out), config_f=cfg.capacity.f, seed=cfg.run.seed, on="H")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "wb") as fh:
        pickle.dump(res, fh)
    print(f"saved {OUT}")
    print(f"the f rule on H picks f = {res['selected']} (configs/default.yaml has f = {cfg.capacity.f})")
    return res


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--f", type=float, nargs="+", default=None, help="f grid (default: theory_eval.F_SCAN)")
    ap.add_argument("--seed", type=int, default=None, help="run.seed (default: the configured one)")
    a = ap.parse_args()
    main(a.f, a.seed)

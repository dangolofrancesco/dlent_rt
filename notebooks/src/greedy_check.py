"""
Greedy policies G1 / G2 (week 30) on the real trace, seed = run.seed, through policy.simulate
with the physical gate; scored with Benchmarks.score (R_job, Phase 0's bid model and s_C).

    dlent_venv/bin/python notebooks/src/greedy_check.py [config.yaml ...]

Default: configs/default.yaml and configs/binding_capacity.yaml. Per policy: R, % of fluid OPT,
admitted, rejections by reason, and for G1 the decision-error rate (sign(R_job) != sign(r~_j)
over the arrivals with an observed type). Check: G1 on frozen H types vs StaticPolicy (Phase 0).
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dlent_rt.benchmarks import Benchmarks
from dlent_rt.config import load_config
from dlent_rt.data import load_dataset
from dlent_rt.models.policy import (REJ_NAMES, REJ_PHYS, REJ_R, REJ_UNSEEN, REJ_NOTYPE,
                                    JobGreedyPolicy, StaticPolicy, TypeGreedyPolicy, run_policy)
from dlent_rt.phase0 import run_phase0

warnings.simplefilter("ignore")


def greedy_rows(B: Benchmarks, ds):
    """Run G1, G1 frozen, G2 and StaticPolicy (Phase 0) on B's stream. Returns (rows, sims)."""
    S, st0, R = B.S, B.state0, B.R
    pols = {
        "G1 (type greedy, cumulative r~)": TypeGreedyPolicy.from_state(st0, S, R, ds.H_span_steps),
        "G1 frozen H types (check)": TypeGreedyPolicy.from_state(st0, S, R, ds.H_span_steps, update=False),
        "G2 (job greedy, R_job > 0)": JobGreedyPolicy(R),
        "static Phase 0 (H)": StaticPolicy(st0, S),
    }
    fluid = B.fluid_opt()
    rows, sims = [], {}
    for name, pol in pols.items():
        sim = run_policy(S, ds.capacity, pol)
        sims[name] = (sim, pol)
        Ry = B.score(sim.admit)
        row = {"policy": name, "R $": Ry, "% of fluid OPT": 100 * Ry / fluid,
               "admitted": int(sim.admit.sum()), "physical": int(sim.phys_reject.sum())}
        if hasattr(pol, "reason"):
            for code in (REJ_R, REJ_UNSEEN, REJ_NOTYPE):
                row[REJ_NAMES[code]] = int((pol.reason == code).sum())
            assert int((pol.reason == REJ_PHYS).sum()) == row["physical"]
        if isinstance(pol, TypeGreedyPolicy):
            ok = np.isfinite(pol.r_dec)
            row["decision errors %"] = 100 * np.mean(np.sign(R[ok]) != np.sign(pol.r_dec[ok]))
        row["adm with R_job<0 %"] = 100 * (R[sim.admit] < 0).mean() if sim.admit.any() else 0.0
        rows.append(row)
    return rows, sims, fluid


def main(paths):
    for path in paths:
        t0 = time.time()
        cfg = load_config(path)
        ds = load_dataset(cfg)
        st0 = run_phase0(ds, cfg)
        B = Benchmarks(ds, st0, cfg)
        rows, sims, fluid = greedy_rows(B, ds)
        print(f"=== {Path(path).name}: capacity {cfg.capacity.mode}  seed={cfg.run.seed}  |H|={ds.H.n:,}  "
              f"|stream|={B.S.n:,}  tau Phase 0={st0.tau}  fluid OPT = {fluid:,.2f} $")
        cols = ["R $", "% of fluid OPT", "admitted", "r<=0", "unseen type", "no type (IR)", "physical",
                "decision errors %", "adm with R_job<0 %"]
        print(f"{'policy':<34}" + "".join(f"{c:>20}" for c in cols))
        for r in rows:
            cells = []
            for c in cols:
                v = r.get(c, "")
                cells.append(f"{v:>20,.2f}" if isinstance(v, float) else f"{v:>20,}" if isinstance(v, int) else f"{v:>20}")
            print(f"{r['policy']:<34}" + "".join(cells))
        g1f = sims["G1 frozen H types (check)"][0]
        g1 = sims["G1 (type greedy, cumulative r~)"][0]
        sp = sims["static Phase 0 (H)"][0]
        print(f"check: G1 frozen vs StaticPolicy(Phase 0): admit masks differ on {int((g1f.admit != sp.admit).sum())} "
              f"arrivals   (tau Phase 0 = {st0.tau}: identical expected iff tau = 0)")
        print(f"       G1 cumulative vs StaticPolicy(Phase 0): admit masks differ on {int((g1.admit != sp.admit).sum())} arrivals")
        print(f"({time.time() - t0:.0f} s)\n")


if __name__ == "__main__":
    main(sys.argv[1:] or [ROOT / "configs" / "default.yaml", ROOT / "configs" / "binding_capacity.yaml"])

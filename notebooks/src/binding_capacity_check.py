"""
Binding-capacity check (week 30, read-only): one online run on configs/binding_capacity.yaml
(capacity fraction_of_concurrency_quantile, 0.30 / 0.60; everything else = default.yaml).
No maths is changed. Reports the capacity, mean / peak occupancy (time-weighted over [1, T]),
tau in Phase 0, in every phase and in the full-stream hindsight state, and the phase end reasons
(does Test 1 fire?).

    dlent_venv/bin/python notebooks/src/binding_capacity_check.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dlent_rt.benchmarks import Benchmarks, regret
from dlent_rt.config import load_config
from dlent_rt.data import load_dataset
from dlent_rt.models.theory import TheoryModel
from dlent_rt.phase0 import run_phase0

warnings.simplefilter("ignore")


def occ_stats(events, T):
    """Time-weighted mean and peak of U(t) on ticks [1, T] from the occupancy event log."""
    ev = np.array([(e[0], *e[3:]) for e in events], float)
    te, Ue = ev[:, 0].astype(np.int64), ev[:, 1:]
    keep = np.r_[te[1:] != te[:-1], True]                  # last level at each tick
    te, Ue = te[keep], Ue[keep]
    pts = np.r_[1, te]
    lev = np.vstack([np.zeros(Ue.shape[1]), Ue])
    dur = np.diff(np.r_[np.clip(pts, 1, T), T + 1])
    return (lev * dur[:, None]).sum(0) / T, lev.max(0)


def main(path=ROOT / "configs" / "binding_capacity.yaml"):
    t0 = time.time()
    cfg = load_config(path)
    ds = load_dataset(cfg)
    st0 = run_phase0(ds, cfg)
    model = TheoryModel(ds, st0, cfg)
    res = model.run()
    c = np.asarray(ds.capacity, float)
    mean, peak = occ_stats(res.occupancy_events, int(ds.T))
    ph = res.phases
    reasons = {r: sum(p["reason"] == r for p in ph) for r in ("test1", "test2", "test3", "timeout", "horizon")}
    tau_ph = np.array([[p["tau_cpu"], p["tau_ram"]] for p in ph])
    B = Benchmarks.from_model(model)
    tau_ref = B.reference_state.tau
    R_alg = res.summary["R_total"]

    print(f"config {path.name}: capacity {cfg.capacity.mode} fraction={cfg.capacity.fraction} "
          f"q={cfg.capacity.concurrency_quantile}  |H|={ds.H.n:,}  |stream|={model.stream.n:,}  T={ds.T:,}")
    print(f"capacity c = cpu {c[0]:.4g}, ram {c[1]:.4g}   xi = a_max/c = {model.ep.xi:.4g}")
    print(f"occupancy mean  cpu {100 * mean[0] / c[0]:.1f}%  ram {100 * mean[1] / c[1]:.1f}%")
    print(f"occupancy peak  cpu {100 * peak[0] / c[0]:.1f}%  ram {100 * peak[1] / c[1]:.1f}%")
    print(f"tau Phase 0 = {st0.tau}   tau* hindsight = {tau_ref}")
    print(f"phases = {len(ph)}  end reasons = {reasons}   Test 1 fires: {reasons['test1'] > 0}")
    print(f"tau > 0 in {int((tau_ph > 0).any(1).sum())} / {len(ph)} phases")
    for m, p in enumerate(ph, start=1):
        print(f"  phase {m:>2}: [{int(p['tau_start']):>9,}, {int(p['tau_end']):>9,}] {p['reason']:<8} "
              f"arrivals={int(p['n_arrivals']):>6,} admitted={int(p['n_admitted']):>6,} "
              f"phys_rej={int(p['n_phys_reject']):>5,} tau=({p['tau_cpu']:.4g}, {p['tau_ram']:.4g})")
    print(f"admitted {int(sum(p['n_admitted'] for p in ph)):,} / {model.stream.n:,}   "
          f"physical rejects {int(sum(p['n_phys_reject'] for p in ph)):,}")
    fluid = B.fluid_opt()
    print(f"R online = {R_alg:,.2f} $   fluid OPT = {fluid:,.2f} $   regret vs fluid = {regret(R_alg, fluid):,.2f} $")
    print(f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "configs" / "binding_capacity.yaml")

"""
Binding-capacity check (week 30, read-only): one online run on configs/binding_capacity.yaml
(capacity fraction_of_concurrency_quantile, 0.30 / 0.60; everything else = default.yaml).
No maths is changed. Reports

  * Phase 0: LP capacity usage sum_j p_j v_ij x*_j / c_i and the profitable load
    sum_{j: r~_j > 0} p_j v_ij / c_i (both per resource, % of c);
  * the online run: capacity, mean / peak occupancy (time-weighted over [1, T]), tau in
    Phase 0, in every phase and in the full-stream hindsight state, phase end reasons;
  * Test 1 per phase, rebuilt from the arrival log: its LHS (resources of the jobs admitted
    in the phase after the warm-up tau_bar = tau_prev + d_max, while t < u <= t + D) against
    its threshold c - a_max_eff (monitor.PhaseMonitor.scan).

--rho R picks the capacity fraction so that the most loaded resource has profitable load
= c / R (c is linear in the fraction: c_i = fraction * Q_q(occupancy_i)).

    dlent_venv/bin/python notebooks/src/binding_capacity_check.py [config.yaml] [--rho 0.6]
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dlent_rt.benchmarks import Benchmarks, regret
from dlent_rt.config import load_config
from dlent_rt.data import load_dataset
from dlent_rt.lp_oracle import solve_lp_rs
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


def phase0_load(ds, st, cfg):
    """LP capacity usage and profitable load per resource, as fractions of c."""
    c = np.asarray(ds.capacity, float)
    lp = solve_lp_rs(p=st.p, r=st.r_tilde, v=st.v, c=c, presolve=cfg.lp.presolve)
    x = np.zeros(st.n_types)
    x[lp.support_idx] = lp.x
    prof = st.r_tilde > 0
    return dict(lp_use=(st.p * x) @ st.v / c, prof_load=(st.p * prof) @ st.v / c,
                total_load=st.p @ st.v / c, n_prof=int((prof & (st.p > 0)).sum()),
                n_pop=int((st.p > 0).sum()))


def max_window(t, D, A, lo, hi):
    """max over u in [lo, hi] of sum_k A_k 1(t_k < u <= t_k + D_k)  (Test 1's LHS)."""
    if t.size == 0:
        return np.zeros(A.shape[1])
    s, e = t + 1, np.minimum(t + D, hi)
    ok = e >= np.maximum(s, lo)
    s, e, A = np.maximum(s[ok], lo), e[ok], A[ok]
    if s.size == 0:
        return np.zeros(A.shape[1])
    times = np.r_[s, e + 1]
    d = np.vstack([A, -A])
    o = np.lexsort((np.r_[np.zeros(s.size), -np.ones(e.size)], times))   # releases before adds
    return np.cumsum(d[o], axis=0).max(0)


def test1_table(model, res):
    S, ph, c = model.stream, res.phases, model.c
    thr = (c - model.a_max_eff) / c
    adm = np.array([a["i"] for a in res.arrivals if a["admit"]], np.int64)
    rows = []
    for m, p in enumerate(ph, start=1):
        tau_prev, tau_end = int(p["tau_start"]) - 1, int(p["tau_end"])
        tau_bar = tau_prev + model.d_max
        a = adm[(S.tick[adm] > tau_prev) & (S.tick[adm] <= tau_end) & (S.tick[adm] >= tau_bar + 1)]
        lhs = max_window(S.tick[a].astype(np.int64), S.D[a].astype(np.int64), S.A[a],
                         tau_bar + 2, tau_end + 1) / c
        rows.append(dict(m=m, active=tau_end + 1 > tau_bar + 1, n_post=a.size, lhs=lhs))
    return thr, rows


def run(cfg, label):
    t0 = time.time()
    ds = load_dataset(cfg)
    st0 = run_phase0(ds, cfg)
    L = phase0_load(ds, st0, cfg)
    model = TheoryModel(ds, st0, cfg)
    res = model.run()
    c = np.asarray(ds.capacity, float)
    a_max = np.asarray(ds.a_max, float)
    mean, peak = occ_stats(res.occupancy_events, int(ds.T))
    ph = res.phases
    reasons = {r: sum(p["reason"] == r for p in ph) for r in ("test1", "test2", "test3", "timeout", "horizon")}
    tau_ph = np.array([[p["tau_cpu"], p["tau_ram"]] for p in ph])
    B = Benchmarks.from_model(model)
    R_alg = res.summary["R_total"]
    thr, t1 = test1_table(model, res)
    pc = lambda x: "/".join(f"{100 * v:.1f}%" for v in x)       # cpu/ram

    print(f"=== {label}: capacity {cfg.capacity.mode} fraction={cfg.capacity.fraction:.6g} "
          f"q={cfg.capacity.concurrency_quantile}  |H|={ds.H.n:,}  |stream|={model.stream.n:,}  T={ds.T:,}")
    print(f"capacity c = cpu {c[0]:.4g}, ram {c[1]:.4g}   data a_max/c = {pc(a_max / c)} (cpu/ram)   "
          f"xi used by the tests = {model.ep.xi:.4g}   stream jobs with A > c: "
          f"{int((model.stream.A > c).any(1).sum())}")
    print(f"Phase 0 (cpu/ram, % of c): LP usage {pc(L['lp_use'])}   profitable load {pc(L['prof_load'])}   "
          f"total load {pc(L['total_load'])}   profitable types {L['n_prof']} / {L['n_pop']} populated")
    print(f"occupancy mean {pc(mean / c)}   peak {pc(peak / c)}")
    print(f"tau Phase 0 = {st0.tau}   tau* hindsight = {B.reference_state.tau}")
    print(f"phases = {len(ph)}  end reasons = {reasons}   Test 1 fires: {reasons['test1'] > 0}   "
          f"tau > 0 in {int((tau_ph > 0).any(1).sum())} / {len(ph)} phases")
    print(f"Test 1 threshold (c - a_max_eff)/c = {pc(thr)}   d_max = {model.d_max:,} ticks")
    for (m, p), r in zip(enumerate(ph, start=1), t1):
        print(f"  phase {m:>2}: [{int(p['tau_start']):>9,}, {int(p['tau_end']):>9,}] {p['reason']:<8} "
              f"arr={int(p['n_arrivals']):>6,} adm={int(p['n_admitted']):>6,} phys_rej={int(p['n_phys_reject']):>4,} "
              f"tau=({p['tau_cpu']:.4g}, {p['tau_ram']:.4g})  T1 active={str(r['active']):<5} "
              f"adm after warm-up={r['n_post']:>5,}  max T1 LHS={pc(r['lhs'])}")
    print(f"admitted {int(sum(p['n_admitted'] for p in ph)):,} / {model.stream.n:,}   "
          f"physical rejects {int(sum(p['n_phys_reject'] for p in ph)):,}")
    fluid = B.fluid_opt()
    print(f"R online = {R_alg:,.2f} $   fluid OPT = {fluid:,.2f} $   regret vs fluid = {regret(R_alg, fluid):,.2f} $")
    print(f"({time.time() - t0:.0f} s)\n")
    return ds, st0, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default=str(ROOT / "configs" / "binding_capacity.yaml"))
    ap.add_argument("--rho", type=float, default=None,
                    help="also run with the fraction giving profitable load = c / rho on the most loaded resource")
    ap.add_argument("--skip-base", action="store_true", help="with --rho: only the rho run")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.rho is None or not a.skip_base:
        run(cfg, Path(a.config).name)
    if a.rho is not None:
        assert cfg.capacity.mode.startswith("fraction_of"), "--rho needs a fraction_of_* capacity mode"
        ds = load_dataset(cfg)
        L = phase0_load(ds, run_phase0(ds, cfg), cfg)
        f = cfg.capacity.fraction * a.rho * float(L["prof_load"].max())
        cfg_r = replace(cfg, capacity=replace(cfg.capacity, fraction=f)).validate()
        print(f"rho = {a.rho}: fraction {cfg.capacity.fraction} -> {f:.6g} "
              f"(max profitable load {100 * L['prof_load'].max():.1f}% of c at the base fraction)")
        run(cfg_r, f"{Path(a.config).name} with rho = {a.rho}")


if __name__ == "__main__":
    main()

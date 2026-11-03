"""
Calibrated thresholds of Tests 2 and 3 (week 34; dlent_rt.models.core.calibration), on the real data.

  --history        kappa2, kappa3 from H alone (the no-learning model replayed on H; practical switches) and the
                   LHS / threshold ratio curves on H -> reports/week34/cache/calibration/H.pkl
  --replay PKL     how often the calibrated tests would fire on a cached stationary run (no new run): the run's own
                   arrivals, rewards and admissions are fed to the monitor phase by phase, with each phase's state
                   rebuilt by ReInit from the cached arrivals; the ratio curves are traced at kappa = 1 and evaluated
                   at the calibrated kappas (ratio > 1 = the test fires). Test 3's per-gap ratio is the monitor's
                   screen bound (LHS at the gap's first tick / RHS at its last), so its count is an upper bound.
                   A firing is one episode: a run of consecutive gaps of one phase with ratio > 1 (the first would end
                   the phase in a real run). The stream's trace is kept, so --count re-counts at another target.
                   -> reports/week34/cache/calibration/replay_<name>.pkl
  --count PKL      the firing count of a replay at --target, from its kept trace (no new replay)
  --set-kappa      writes kappa2, kappa3 at --target to reports/week34/cache/calibration/kappa.pkl: the kappas every
                   practical_calibrated run of scripts/run_scenarios.py uses (H is the same in every environment, so
                   they are computed once)
  --target X       the largest LHS / threshold ratio on H (default: tests.calibration_target, 0.5)

    caffeinate -i dlent_venv/bin/python scripts/calibrate_tests.py --history
    caffeinate -i dlent_venv/bin/python scripts/calibrate_tests.py --replay reports/week34/cache/scenarios/none/practical.pkl
    dlent_venv/bin/python scripts/calibrate_tests.py --count reports/week34/cache/calibration/replay_none_practical.pkl --target 0.25
    dlent_venv/bin/python scripts/calibrate_tests.py --set-kappa --target 0.5
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
import warnings
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
OUT = ROOT / "reports" / "week34" / "cache" / "calibration"
H_PER_TICK = 1.0 / 3600.0


def practical_cfg():
    from run_scenarios import config_for
    return config_for("none", "practical")


def setup(cfg):
    from dlent_rt.data import load_dataset
    from dlent_rt.phase0 import run_phase0
    ds = load_dataset(cfg)
    return ds, run_phase0(ds, cfg)


def history() -> dict:
    from dlent_rt.models.core.calibration import kappas_from_trace, replay_history
    cfg = practical_cfg()
    ds, st0 = setup(cfg)
    t = time.perf_counter()
    tr, phases = replay_history(ds, st0, cfg)
    k = kappas_from_trace(tr, cfg.tests.calibration_target)
    h = cfg.time.tick_seconds / 3600.0
    out = dict(kappa=k, phases=phases, runtime_s=time.perf_counter() - t, target=cfg.tests.calibration_target, trace=tr,
               curve2=pd.DataFrame(dict(hour=tr.r2_tick * h, ratio_theory=tr.r2, ratio=tr.ratio2(k["kappa2"]))),
               curve3=_curve3(tr, k["kappa3"], h))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "H.pkl", "wb") as fh:
        pickle.dump(out, fh)
    print(f"H replay: {len(phases)} phases {[(p['tau_end'] * h, p['reason']) for p in phases]}, "
          f"{out['runtime_s'] / 60:.1f} min")
    print(f"kappa2 = {k['kappa2']:.6g} (max Test 2 ratio on H at kappa 1: {k['max_ratio2_theory']:.4g})")
    print(f"kappa3 = {k['kappa3']:.6g} (max Test 3 ratio on H at kappa 1: {k['max_ratio3_theory']:.4g})")
    return out


def kappa_for(target: float) -> dict:
    """kappa2, kappa3 at `target`, from H's kept trace."""
    from dlent_rt.models.core.calibration import kappas_from_trace
    H = pickle.load(open(OUT / "H.pkl", "rb"))
    k = kappas_from_trace(H["trace"], target) if "trace" in H else H["kappa"] if target == H["target"] else None
    if k is None:
        raise SystemExit("H.pkl has no trace: rerun --history")
    return dict(k, target=float(target))


def _curve3(tr, kappa3, h) -> pd.DataFrame:
    """Test 3's ratio per gap: the max over the window starts s (theoretical and calibrated)."""
    df = pd.DataFrame(dict(tick=tr.t3_tick, ratio_theory=tr.ratio3(1.0), ratio=tr.ratio3(kappa3)))
    g = df.groupby("tick", sort=True).max().reset_index()
    g.insert(0, "hour", g.pop("tick") * h)
    return g


def replay(pkl: Path, kappa: dict) -> dict:
    """The cached run's arrivals through the monitor (see the module docstring)."""
    from dlent_rt.models.core.calibration import ThresholdTrace
    from dlent_rt.models.core.error_params import make_error_params
    from dlent_rt.models.core.monitor import PhaseMonitor
    from dlent_rt.models.core.reinit import build_window_state
    from dlent_rt.models.core.stream import build_stream
    from dlent_rt.phase0 import assign_type_ids
    run = pickle.load(open(pkl, "rb"))
    cfg = practical_cfg()
    want = asdict(cfg)
    assert run["config"]["scenario"]["name"] == "none", "a stationary run"
    for sec in ("ablation", "oracle", "stats", "bid", "capacity", "run"):
        assert run["config"][sec] == want[sec], sec
    ds, st0 = setup(cfg)
    S = build_stream(ds, cfg)
    a = run["arrays"]
    assert np.array_equal(a["tick"], S.tick)
    R, admit = np.asarray(a["R"], float), np.asarray(a["admit"], bool)
    ph = run["phases"]
    ep = make_error_params(st0, ds, cfg)                                     # kappa = 1: the trace
    ep_cal = replace(ep, kappa2=kappa["kappa2"], kappa3=kappa["kappa3"])
    c = np.asarray(ds.capacity, float)
    a_max_eff = ep.xi * c
    r_unit, d_max = float(st0.r_max), int(ds.d_max_steps)
    types = assign_type_ids(st0.type_space, st0.grid.assign(S.v), S.priority, S.hw_idx)   # grid / types: Phase 0's

    # each phase's state, rebuilt by ReInit from the cached arrivals (as the run did)
    t0 = time.perf_counter()
    states, st, tau_prev = [st0], st0, 0
    for _, p in ph.iloc[:-1].iterrows():
        arr = np.flatnonzero((S.tick > tau_prev) & (S.tick <= p.tau_end))
        st, _ = build_window_state(S, arr, arr[admit[arr]], int(p.tau_end - tau_prev), st, ds, cfg, reason=p.reason)
        states.append(st)
        tau_prev = int(p.tau_end)
    print(f"{len(states)} phase states rebuilt in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)

    tr, ends = ThresholdTrace(), []
    mk = lambda e, st, tp, W, **kw: PhaseMonitor(e, st.r_tilde, st.v, c, a_max_eff, tp, W, d_max, cfg,
                                                 r_unit=r_unit, **kw)
    m, tau_prev, W_prev = 0, 0, d_max
    mon = mk(ep, states[0], 0, d_max, trace=tr, tests23=False)
    for i in range(S.n):
        t = int(S.tick[i])
        while True:
            sr = mon.scan(t)
            if sr.fail_tick is None:
                break
            tau_m = sr.fail_tick - 1
            ends.append((tau_m, sr.reason))
            m, W_prev, tau_prev = m + 1, tau_m - tau_prev, tau_m
            mon = mk(ep, states[min(m, len(states) - 1)], tau_prev, W_prev, trace=tr, tests23=False)
        mon.register_arrival(t, int(types[i]), R[i], bool(admit[i]), i, S.A[i], int(S.D[i]))
    tr.finish()
    cached = [(int(e), r) for e, r in zip(ph.tau_end, ph.reason)][:-1]
    h = cfg.time.tick_seconds / 3600.0
    out = dict(ends_replay=ends, ends_cached=cached, same_phases=ends == cached, trace=tr, phases=ph, h=h,
               hours=float(S.tick[-1] * h))
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"replay_{pkl.parent.name}_{pkl.stem}.pkl"
    with open(f, "wb") as fh:
        pickle.dump(out, fh)
    print(f"phase ends as cached: {out['same_phases']}")
    count(f, kappa)
    return out


def count(f: Path, kappa: dict) -> pd.DataFrame:
    """The firing count of a kept replay at `kappa`: per phase, each calibrated test's largest ratio, its first
    firing hour (since the phase start) and its episodes (runs of consecutive gaps with ratio > 1)."""
    r = pickle.load(open(f, "rb"))
    tr, ph, h = r["trace"], r["phases"], r["h"]
    curves = dict(test2=pd.DataFrame(dict(hour=tr.r2_tick * h, ratio=tr.ratio2(kappa["kappa2"]))),
                  test3=_curve3(tr, kappa["kappa3"], h))
    rows = []
    for _, p in ph.iterrows():
        lo, hi = (p.tau_start - 1) * h, p.tau_end * h
        row = dict(phase=int(p.m), start_h=lo, length_h=hi - lo, end=p.reason)
        for name, cv in curves.items():
            w = cv[(cv.hour > lo) & (cv.hour <= hi + h)].sort_values("hour")
            up = (w.ratio > 1).to_numpy()
            row[f"max {name} ratio"] = float(w.ratio.max()) if len(w) else np.nan
            row[f"{name} first fire h"] = float(w.hour[up].iloc[0] - lo) if up.any() else np.nan
            row[f"{name} episodes"] = int(np.sum(up & ~np.r_[False, up[:-1]]))
        rows.append(row)
    df = pd.DataFrame(rows)
    n = int(df["test2 episodes"].sum() + df["test3 episodes"].sum())
    print(f"target {kappa.get('target')}: kappa2 = {kappa['kappa2']:.6g}, kappa3 = {kappa['kappa3']:.6g}")
    print(df.to_string(index=False))
    print(f"FIRINGS: Test 2 + Test 3 = {n} over {r['hours']:.0f} h "
          f"(Test 2: {int(df['test2 episodes'].sum())}, Test 3: {int(df['test3 episodes'].sum())})")
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history", action="store_true")
    ap.add_argument("--replay", type=Path)
    ap.add_argument("--count", type=Path)
    ap.add_argument("--set-kappa", action="store_true")
    ap.add_argument("--target", type=float, default=None)
    a = ap.parse_args(argv)
    target = a.target if a.target is not None else practical_cfg().tests.calibration_target
    if a.history:
        history()
    if a.replay:
        replay(a.replay, kappa_for(target))
    if a.count:
        count(a.count, kappa_for(target))
    if a.set_kappa:
        k = kappa_for(target)
        with open(OUT / "kappa.pkl", "wb") as fh:
            pickle.dump(dict(kappa2=k["kappa2"], kappa3=k["kappa3"], target=k["target"]), fh)
        print(f"kappa.pkl: target {k['target']}, kappa2 = {k['kappa2']:.6g}, kappa3 = {k['kappa3']:.6g}")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    main()

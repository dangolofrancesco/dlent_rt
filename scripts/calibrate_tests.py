"""
Calibrated thresholds of Tests 2 and 3 (week 34; dlent_rt.models.core.calibration), on the real data.

  --history        kappa2, kappa3 from H alone (the no-learning model replayed on H; practical switches) and the
                   LHS / threshold ratio curves on H -> reports/week34/cache/calibration/H.pkl
  --replay PKL     how often the calibrated tests would fire on a cached stationary run (no new run): the run's own
                   arrivals, rewards and admissions are fed to the monitor phase by phase, with each phase's state
                   rebuilt by ReInit from the cached arrivals; the ratio curves are traced at kappa = 1 and evaluated
                   at the calibrated kappas (ratio > 1 = the test fires). Test 3's per-gap ratio is the monitor's
                   screen bound (LHS at the gap's first tick / RHS at its last), so its count is an upper bound.
                   -> reports/week34/cache/calibration/replay_<name>.pkl

    caffeinate -i dlent_venv/bin/python scripts/calibrate_tests.py --history
    caffeinate -i dlent_venv/bin/python scripts/calibrate_tests.py --replay reports/week34/cache/scenarios/none/practical.pkl
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
    out = dict(kappa=k, phases=phases, runtime_s=time.perf_counter() - t, target=cfg.tests.calibration_target,
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
    c2, c3 = tr.ratio2(kappa["kappa2"]), _curve3(tr, kappa["kappa3"], h)
    out = dict(kappa=kappa, ends_replay=ends, ends_cached=cached, same_phases=ends == cached,
               curve2=pd.DataFrame(dict(hour=tr.r2_tick * h, ratio_theory=tr.r2, ratio=c2)), curve3=c3)
    out["fires"] = _fires(out, ph, h)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"replay_{pkl.parent.name}_{pkl.stem}.pkl", "wb") as fh:
        pickle.dump(out, fh)
    print(f"phase ends as cached: {out['same_phases']}")
    print(out["fires"].to_string(index=False))
    return out


def _fires(out, ph, h) -> pd.DataFrame:
    """Per cached phase: the calibrated tests' largest ratio, and the first hour (since the phase start) with
    ratio > 1, or NaN."""
    rows = []
    for _, p in ph.iterrows():
        lo, hi = (p.tau_start - 1) * h, p.tau_end * h
        row = dict(phase=int(p.m), start_h=lo, length_h=hi - lo, end=p.reason)
        for name, cv in (("test2", out["curve2"]), ("test3", out["curve3"])):
            w = cv[(cv.hour > lo) & (cv.hour <= hi + h)]
            row[f"max {name} ratio"] = float(w.ratio.max()) if len(w) else np.nan
            f = w[w.ratio > 1]
            row[f"{name} fires at h"] = float(f.hour.iloc[0] - lo) if len(f) else np.nan
            row[f"{name} gaps > 1"] = int(len(f))
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history", action="store_true")
    ap.add_argument("--replay", type=Path)
    a = ap.parse_args(argv)
    if a.history:
        history()
    if a.replay:
        k = pickle.load(open(OUT / "H.pkl", "rb"))["kappa"]
        replay(a.replay, k)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    main()

"""Event-driven exact monitor == literal tick-by-tick reference (phase boundaries, reasons, decisions)."""
import tempfile, warnings
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest

from dlent_rt.data import load_dataset
from dlent_rt.phase0 import run_phase0
from dlent_rt.models.theory import TheoryModel
from dlent_rt.models.core import monitor as mon_mod
from tests.brute import BruteMonitor
from tests.synthetic import synthetic_config


def _setup(gamma, xi, d):
    cfg = synthetic_config(Path(d), n0=150, n_jobs=900, hours=4.0, drift=True)
    cfg = replace(cfg, time=replace(cfg.time, tick_seconds=5.0),
                  constants=replace(cfg.constants, gamma_override=gamma, xi_override=xi))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = load_dataset(cfg)
        st = run_phase0(ds, cfg)
    return cfg, ds, st


def _run(cfg, ds, st, brute):
    import dlent_rt.models.theory as th
    orig = th.PhaseMonitor
    if brute:
        th.PhaseMonitor = BruteMonitor
    try:
        return TheoryModel(ds, st, cfg).run()
    finally:
        th.PhaseMonitor = orig


@pytest.mark.parametrize("gamma,xi", [(2e-4, 1e-4), (1e-3, 1e-4), (1.0, 1e-3), (1e-3, 0.6), (5e-3, 0.3)])
def test_exact_equals_bruteforce(gamma, xi):
    with tempfile.TemporaryDirectory() as d:
        cfg, ds, st = _setup(gamma, xi, d)
        a = _run(cfg, ds, st, brute=False)
        b = _run(cfg, ds, st, brute=True)
    pa = [(p["tau_start"], p["tau_end"], p["reason"]) for p in a.phases]
    pb = [(p["tau_start"], p["tau_end"], p["reason"]) for p in b.phases]
    print(gamma, pa)
    assert pa == pb
    assert [x["admit"] for x in a.arrivals] == [x["admit"] for x in b.arrivals]


def test_test1_unit_equals_bruteforce():
    from types import SimpleNamespace
    from dlent_rt.models.core.error_params import ErrorParams
    from dlent_rt.models.core.monitor import PhaseMonitor
    rng = np.random.default_rng(3)
    nt, c = 30, np.array([1.0, 1.0])
    r, V = rng.normal(0.05, 0.05, nt), rng.lognormal(0, 1, (nt, 2))
    cfg = SimpleNamespace(test3=SimpleNamespace(s_grid="geometric", geometric_base=2),
                          tests=SimpleNamespace(piecewise_tol=1e-9))
    ep = ErrorParams(gamma=1.0, xi=0.5, c_min=1.0, n_res=2, T=10_000, delta=1e-12)
    for rep in range(20):
        mons = [cls(ep, r, V, c, 0.5 * c, 0, 400, 50, cfg) for cls in (PhaseMonitor, BruteMonitor)]
        t, outs = 0, [[], []]
        ticks = np.cumsum(rng.integers(1, 12, 200))
        for t in ticks:
            res = [m.scan(int(t)) for m in mons]
            outs[0].append((res[0].fail_tick, res[0].reason)); outs[1].append((res[1].fail_tick, res[1].reason))
            if res[0].fail_tick is not None:
                break
            adm = rng.random() < 0.5
            A, D = rng.uniform(0.05, 0.25, 2), int(rng.integers(1, 60))
            for m in mons:
                m.register_arrival(int(t), 0, 0.0, adm, int(t), A, D)
        assert outs[0] == outs[1]
        assert outs[0][-1][1] in ("test1", "timeout")
    assert any(o[-1][1] == "test1" for o in [outs[0]])

"""The discretization cost of the grid (week 33): MILP_true (each job at its own hourly bid) vs MILP_bin (each job at
its bin's lower edge L_k), on the same jobs and capacity, with true durations (benchmarks.discretization_milps).
Tiny hand-made instances: each MILP solves in well under a second."""
from types import SimpleNamespace

import numpy as np
import pytest

from dlent_rt.benchmarks import bin_values, discretization_milps, peak_occupancy
from dlent_rt.bid import MyersonModel
from dlent_rt.config import default_config
from dlent_rt.grid import build_valuation_grid

LAM = default_config().objective


def world(v, seed=0):
    """Jobs with hourly bids v on a 8-bin grid over [0.01, 1] $/h; cost bases near the bids (markups 0.8-1.6), so most
    rewards are positive; two resources, overlapping jobs, a cluster that holds about a third of them at once."""
    rng = np.random.default_rng(seed)
    n = len(v)
    grid = build_valuation_grid(np.array([0.01, 1.0]), k_star=8)
    state = SimpleNamespace(grid=grid, myerson=MyersonModel.known_markup(0.3), carbon_price=2.0)
    d = rng.uniform(0.5, 3.0, n)                                           # true durations, hours
    S = SimpleNamespace(tick=np.arange(1, n + 1) * 600, D=np.rint(d * 3600).astype(np.int64),
                        A=rng.uniform(0.05, 0.4, (n, 2)), v=np.asarray(v, float),
                        cost_base=np.asarray(v, float) / rng.uniform(0.8, 1.6, n), duration_hours=d,
                        C_elec=rng.uniform(0.0, 0.01, n), C_carbon=rng.uniform(0.0, 0.002, n))
    return S, state, np.array([1.0, 1.0])


def test_bin_reward_below_true_reward_and_opt_bin_below_opt_true():
    rng = np.random.default_rng(1)
    S, state, c = world(np.exp(rng.uniform(np.log(0.012), np.log(0.95), 40)))
    o = discretization_milps(S, state, LAM, c, time_limit=10)
    assert np.all(bin_values(S.v, state.grid) <= S.v)
    assert np.all(o["R_bin"] <= o["R_true"] + 1e-12) and o["n_jobs_reward_bin_gt_true"] == 0
    assert o["true"]["status"] == "Optimal" and o["bin"]["status"] == "Optimal"
    assert o["bin"]["best"] <= o["true"]["best"] + 1e-9                    # OPT_bin <= OPT_true
    assert o["value_loss"] >= -1e-9 and o["decision_loss"] >= -1e-9
    assert o["bin_schedule_at_true"] <= o["true"]["best"] + 1e-9
    assert o["worst_case_pct"] == pytest.approx(100 * (1 - 1 / state.grid.ratio))
    assert not np.any(o["x_bin"] & (o["R_true"] <= 0))                     # candidates = R_true > 0, for both


def test_one_job_per_bin_edge_gives_identical_milps():
    grid = build_valuation_grid(np.array([0.01, 1.0]), k_star=8)
    edges = grid.lower_edges[1:]                                            # b = L_k for bins 2..8 (bin 1's edge is 0)
    S, state, c = world(np.repeat(edges, 3), seed=2)
    np.testing.assert_array_equal(bin_values(S.v, state.grid), S.v)
    o = discretization_milps(S, state, LAM, c, time_limit=10)
    np.testing.assert_array_equal(o["R_bin"], o["R_true"])
    np.testing.assert_array_equal(o["x_bin"], o["x_true"])
    assert o["bin"]["best"] == o["true"]["best"] and o["value_loss"] == 0 and o["decision_loss"] == 0


def test_both_schedules_feasible_on_true_durations():
    rng = np.random.default_rng(3)
    S, state, c = world(np.exp(rng.uniform(np.log(0.012), np.log(0.95), 40)), seed=3)
    o = discretization_milps(S, state, LAM, c, time_limit=10)
    for x in (o["x_true"], o["x_bin"]):
        assert x.sum() > 0
        assert np.all(peak_occupancy(S.tick[x], S.D[x], S.A[x]) <= c + 1e-9)
    assert o["true"]["feasible_true"] and o["bin"]["feasible_true"]
    # the capacity binds: taking every candidate would not fit
    cand = o["R_true"] > 0
    assert np.any(peak_occupancy(S.tick[cand], S.D[cand], S.A[cand]) > c)


def test_window_discretization_on_the_synthetic_trace(tmp_path):
    """The script's path (theory_eval.window_discretization): Phase 0, the stream, the window, both MILPs."""
    import warnings
    from dlent_rt.theory_eval import window_discretization
    from tests.synthetic import synthetic_config
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = synthetic_config(tmp_path, n0=400, n_jobs=1500)
        w = window_discretization(cfg, days=(0.0, 0.25), time_limit=10.0)
    assert 0 < w["n_candidates"] <= w["n_jobs"] and w["days"] == (0.0, 0.25)
    assert w["n_jobs_reward_bin_gt_true"] == 0 and w["true"]["feasible_true"] and w["bin"]["feasible_true"]
    assert w["bin"]["best"] <= w["true"]["bound"] + 1e-6
    assert "R_true" not in w and w["f"] == cfg.capacity.f

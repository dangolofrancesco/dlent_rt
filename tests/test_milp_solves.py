"""benchmarks.occupancy_milp (week 32): every HiGHS solve gets its own time limit (the LP relaxation does not use the
MILP's budget, two solves in one process each get the full limit); the tail capacity checks after the last
arrival (tail_step) never change the solution, because occupancy cannot grow after the last arrival."""
import time

import numpy as np
import scipy.sparse as sp

from dlent_rt.benchmarks import _highs_max, occupancy_milp


def _hard_knapsack():
    rng = np.random.default_rng(0)
    A = rng.integers(1, 100, (60, 400)).astype(float)
    return sp.csr_matrix(A), rng.integers(1, 100, 400).astype(float), A.sum(1) / 4


def test_each_solve_gets_its_own_time_limit():
    M, obj, ub = _hard_knapsack()
    lp = _highs_max(M, obj, ub, integer=False, time_limit=2.0)                # the relaxation first
    assert lp["runtime_s"] < 1.0
    for _ in range(2):                                                        # two MILPs in one process
        ip = _highs_max(M, obj, ub, integer=True, time_limit=2.0)
        assert "time limit" in ip["status"].lower()
        assert 1.8 <= ip["runtime_s"] <= 3.0 and ip["highs_run_time_s"] >= 1.8
        assert ip["dual_bound"] >= ip["obj"] - 1e-9


def test_tail_checks_never_bind():
    rng = np.random.default_rng(1)
    n = 120
    t = np.sort(rng.integers(0, 500, n))
    D = rng.integers(1, 300, n)
    A = rng.uniform(0.05, 0.6, (n, 2))
    obj = rng.uniform(-1, 3, n)
    c = np.array([1.5, 1.2])
    a = occupancy_milp(t, D, A, obj, c, time_limit=30)
    b = occupancy_milp(t, D, A, obj, c, time_limit=30, tail_step=7)
    assert b["n_tail_checks"] > 0 and b["nonzeros"] > a["nonzeros"]
    assert abs(a["ip_obj"] - b["ip_obj"]) < 1e-6 and abs(a["lp_obj"] - b["lp_obj"]) < 1e-6

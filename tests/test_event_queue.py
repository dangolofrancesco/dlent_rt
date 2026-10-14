"""Event-driven occupancy (heap of departures) == brute-force per-tick occupancy."""
import numpy as np
import pytest

from dlent_rt.data import _OccupancyBasis, _occupancy_events
from dlent_rt.models.core.occupancy import Occupancy


def _brute(T, ticks, A, D, admitted):
    """U[t] = sum of A_j over admitted j with tick_j <= t < tick_j + D_j (per-tick scan)."""
    U = np.zeros((T + 1, A.shape[1]))
    for t0, a, d, ok in zip(ticks, A, D, admitted):
        if ok:
            U[t0:min(t0 + d, T + 1)] += a
    return U


def _replay(events, T, n_res):
    """Per-tick U from the Occupancy event log: the level after the last event at or before t."""
    U = np.zeros((T + 1, n_res))
    for tick, _, _, *u in events:
        U[tick:] = u
    return U


@pytest.mark.parametrize("seed", range(5))
def test_heap_occupancy_matches_per_tick(seed):
    rng = np.random.default_rng(seed)
    n, n_res = 400, 2
    ticks = np.sort(rng.choice(np.arange(1, 3000), n, replace=False))   # one arrival per tick
    A = rng.uniform(0.01, 0.3, (n, n_res))
    D = rng.integers(1, 200, n)
    c = np.array([1.5, 1.2])
    occ = Occupancy(c)
    admitted = np.zeros(n, bool)
    for j, (t, a, d) in enumerate(zip(ticks, A, D)):
        occ.release_until(int(t))                       # every departure with end <= t, before deciding
        # the occupancy seen by the decision equals the per-tick ground truth at t
        truth = _brute(int(t), ticks[:j], A[:j], D[:j], admitted[:j])[int(t)]
        np.testing.assert_allclose(occ.U, truth, atol=1e-12)
        if occ.fits(a):                                 # physical gate
            occ.add(j, a, int(t), int(d))
            admitted[j] = True
    T = int(ticks[-1] + D.max())
    occ.release_until(T)
    assert occ.n_active == 0 and np.allclose(occ.U, 0.0)
    assert 0 < admitted.sum() < n                       # the gate binds in this instance
    U_heap = _replay(occ.events, T, n_res)
    U_brute = _brute(T, ticks, A, D, admitted)
    np.testing.assert_allclose(U_heap, U_brute, atol=1e-12)
    assert np.all(U_brute <= c + 1e-12)


def test_heap_matches_capacity_sizing_sweep():
    """Admit-all heap occupancy == the sort/cumsum sweep data._occupancy_events used for capacity sizing."""
    rng = np.random.default_rng(7)
    n = 300
    ticks = np.sort(rng.choice(np.arange(0, 5000), n, replace=False))
    A = rng.uniform(0.0, 0.2, (n, 2))
    D = rng.integers(1, 400, n)
    occ = Occupancy(np.full(2, np.inf))
    for j, (t, a, d) in enumerate(zip(ticks, A, D)):
        occ.release_until(int(t))
        occ.add(j, a, int(t), int(d))
    T = int(ticks[-1] + D.max())
    occ.release_until(T)
    times, lev = _occupancy_events(_OccupancyBasis(arrival_step=ticks, duration_steps=D, A=A))
    U_sweep = np.zeros((T + 1, 2))
    for t, u in zip(times, lev):
        U_sweep[t:] = u
    np.testing.assert_allclose(_replay(occ.events, T, 2), U_sweep, atol=1e-12)
    np.testing.assert_allclose(U_sweep, _brute(T, ticks, A, D, np.ones(n, bool)), atol=1e-12)

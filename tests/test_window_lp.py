import numpy as np
from dlent_rt.lp_oracle import solve_lp_rs
from dlent_rt.models.core.window_lp import WindowLP


def _instance(seed, nt=60):
    rng = np.random.default_rng(seed)
    r = rng.normal(0.05, 0.08, nt)
    V = rng.lognormal(0, 1, (nt, 2)) * 5
    c = np.array([3.0, 2.0])
    ids = rng.choice(nt, 25, replace=False)
    cnt = rng.integers(1, 6, 25)
    return r, V, c, ids, cnt


def test_matches_reference_lp():
    for seed in range(5):
        r, V, c, ids, cnt = _instance(seed)
        w = WindowLP(r, V, c); w.set_counts(ids, cnt)
        for n in [1, 3, 10, 57, 400, 5000]:
            p = np.zeros(len(r)); p[ids] = cnt / n
            ref = solve_lp_rs(p, r, V, c, presolve=False).optimum
            assert np.isclose(w.lam(n), ref, rtol=1e-8, atol=1e-12), (seed, n)


def test_piecewise_exact_and_monotone():
    for seed in range(5):
        r, V, c, ids, cnt = _instance(seed)
        w = WindowLP(r, V, c); w.set_counts(ids, cnt)
        lam = w.lam_on_integers(1, 3000)
        brute = np.array([w.lam(n) for n in range(1, 3001, 37)])
        np.testing.assert_allclose(lam[::37], brute, rtol=1e-8, atol=1e-12)
        assert np.all(np.diff(lam) <= 1e-12)

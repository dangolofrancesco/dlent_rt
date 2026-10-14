"""Known bid distribution (bid.known_distribution): markup v / c ~ LogNormal(-sigma^2/2, sigma),
phi_v(v | c) = c * phi_m(v / c), one phi for Phase 0, ReInit and the online reward."""
import tempfile, warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.stats import norm

from dlent_rt.bid import MyersonModel
from dlent_rt.data import load_dataset
from dlent_rt.phase0 import run_phase0, assign_type_ids
from dlent_rt.models.theory import TheoryModel
from dlent_rt.config import default_config


GOLDEN = Path(__file__).parent / "data" / "golden_known_false.npz"


class _Recording(TheoryModel):
    def run(self):
        self.states = []
        return super().run()

    def _new_phase(self, state, tau_prev, W_prev):
        self.states.append(state)
        super()._new_phase(state, tau_prev, W_prev)


def _run(known: bool):
    with tempfile.TemporaryDirectory() as d:
        cfg = default_config()
        cfg = replace(cfg, bid=replace(cfg.bid, known_distribution=known))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = load_dataset(cfg)
            st = run_phase0(ds, cfg)
            m = _Recording(ds, st, cfg)
            res = m.run()
    return cfg, ds, m, res


@pytest.fixture(scope="module")
def known_run():
    return _run(True)


def z_star(sigma: float) -> float:
    """phi_m(m) < 0  <=>  sigma * Mills(z) > 1  <=>  z < z*(sigma), z = (ln m - mu) / sigma."""
    return brentq(lambda z: sigma * norm.sf(z) / norm.pdf(z) - 1.0, -30.0, 30.0)


def test_scaling():                                                    # (a)
    mdl = MyersonModel.known_markup(0.3)
    rng = np.random.default_rng(0)
    m = rng.lognormal(-0.045, 0.3, 1000)
    c = np.exp(rng.uniform(-5, 9, 1000))
    np.testing.assert_allclose(mdl.phi_job(c * m, c), c * mdl.phi(m), rtol=1e-12)
    assert mdl.mu == pytest.approx(-0.045) and mdl.sigma == 0.3 and mdl.known


@pytest.mark.parametrize("sigma", [0.3, 0.8])
def test_negative_share_matches_theory(sigma):                         # (b)
    mdl = MyersonModel.known_markup(sigma)
    n = 400_000
    m = np.random.default_rng(1).lognormal(-sigma ** 2 / 2, sigma, n)
    share = float((np.asarray(mdl.phi(m)) < 0).mean())
    p = norm.cdf(z_star(sigma))
    assert abs(share - p) < 4 * np.sqrt(p * (1 - p) / n)
    if sigma == 0.3:
        assert p == pytest.approx(0.167, abs=1e-3)


def test_same_phi_everywhere(known_run):                               # (c)
    cfg, ds, m, res = known_run
    st0, S = m.state0, m.stream
    f0 = st0.myerson.phi_job
    # Phase 0: phi_bar = per-type mean of phi_job over H (= the generator phi, analytically)
    H, b = ds.H, st0.bids
    phi_H = np.asarray(f0(b.v_rate, b.cost_base))
    np.testing.assert_allclose(phi_H, b.phi_rate, rtol=1e-9, atol=1e-9)
    tid = assign_type_ids(st0.type_space, st0.grid.assign(b.v_rate, st0.myerson), H.priority, H.hw_idx)
    N = np.bincount(tid, minlength=st0.n_types)
    s = np.bincount(tid, weights=phi_H, minlength=st0.n_types)
    np.testing.assert_allclose(st0.phi_bar, np.divide(s, N, out=np.zeros_like(s), where=N > 0),
                               rtol=1e-12, atol=1e-12)
    # ReInit states: same function (same output on a fixed set of bids) and their per-job phi
    assert len(m.states) >= 2
    v, c = S.v[:500], S.cost_base[:500]
    ref = np.asarray(f0(v, c))
    for st in m.states[1:]:
        np.testing.assert_array_equal(np.asarray(st.myerson.phi_job(v, c)), ref)
        np.testing.assert_allclose(st.bids.phi_rate, st0.myerson.phi_job(st.bids.v_rate, st.bids.cost_base),
                                   rtol=1e-12)
    # online realised reward
    phi_mech = np.array([a["phi_mech"] for a in res.arrivals])
    np.testing.assert_allclose(phi_mech, np.asarray(f0(S.v, S.cost_base)), rtol=1e-12)


def test_sigma_constant_across_phases(known_run):                      # (d)
    cfg, ds, m, res = known_run
    sig = cfg.bid.sigma
    assert all(st.myerson.known and st.myerson.sigma == sig for st in m.states)
    assert {a["sigma_used"] for a in res.arrivals} == {sig}
    assert {p["sigma"] for p in res.phases} == {sig}
    assert all(np.isnan(st.phi_tilde).all() for st in m.states)        # no pooled phi(L) per type


def test_threshold_price_logged(known_run):
    cfg, ds, m, res = known_run
    for a in res.arrivals:
        if a["k_hat"]:
            assert 1 <= a["threshold_k"] <= a["k_bin"]
        if a["threshold_k"] == 0:
            assert np.isnan(a["threshold_price"])
        else:
            assert a["threshold_markup"] == pytest.approx(a["threshold_price"] / a["cost_base"])


def test_known_false_reproduces_previous_numbers():                    # (e)
    g = np.load(GOLDEN)
    cfg, ds, m, res = _run(False)
    st = m.state0
    for f in ["p", "v_bar", "phi_bar", "r_tilde", "phi_tilde", "tau", "L"]:
        np.testing.assert_allclose(getattr(st, f), g["p0_" + f], rtol=1e-10, atol=1e-12, err_msg=f)
    np.testing.assert_allclose([st.lam_star, st.myerson.mu, st.myerson.sigma, st.carbon_price, st.gamma],
                               g["p0_scalars"], rtol=1e-10)
    for f in ["R", "phi_mech", "payment", "admit", "k_hat", "type"]:
        np.testing.assert_allclose(np.array([a[f] for a in res.arrivals], float), g["arr_" + f],
                                   rtol=1e-10, atol=1e-12, err_msg=f)
    np.testing.assert_allclose([p["lam_hat_start"] for p in res.phases], g["ph_lam"], rtol=1e-10)
    np.testing.assert_array_equal([p["tau_end"] for p in res.phases], g["ph_tau_end"])
    s = res.summary
    np.testing.assert_allclose([s["R_total"], s["n_admitted"], s["n_phases"]], g["sum"], rtol=1e-10)
    assert len({p["sigma"] for p in res.phases}) > 1                   # sigma_hat refit per phase

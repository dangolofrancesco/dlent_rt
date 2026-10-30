"""The bid (D2) and the customer value (D1): E[phi_m] = 0 and phi_m increasing for the known lognormal markup; the
hourly cost base U0 (1 + ln q) + gamma1 cores + gamma2 GB on a hand-made job; U0 = the cost rate of the median H job;
the sigma refit; customer value V = b d (hourly bid x true hours, no priority factor)."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import integrate, stats

from dlent_rt.bid import MyersonModel, bid_model, calibrate_u0, cost_base, fit_sigma, synthesize_bids
from dlent_rt.config import default_config
from dlent_rt.reward import job_terms

G1, G2 = 0.0332, 0.00445


@pytest.mark.parametrize("sigma", [0.1, 0.3, 0.5])
def test_expected_phi_markup_is_zero(sigma):
    mu = -sigma ** 2 / 2
    # E[phi(m)] = E[m] - int_0^inf (1 - F(m)) dm, and in z (m = exp(mu + sigma z)): int (1 - F) dm = sigma int m (1 - Phi) dz
    I, _ = integrate.quad(lambda z: np.exp(mu + sigma * z) * stats.norm.sf(z), -60, 60, limit=500, epsabs=1e-13)
    assert abs(1.0 - sigma * I) < 5e-3
    z = np.linspace(-6, 6, 2001)
    m = np.exp(mu + sigma * z)
    mills = np.exp(stats.norm.logsf(z) - stats.norm.logpdf(z))
    phi = MyersonModel.known_markup(sigma).phi_markup(m)
    np.testing.assert_allclose(phi, m * (1 - sigma * mills), rtol=1e-6, atol=1e-6)
    assert np.all(np.diff(phi) > 0)                                 # regular: phi_m increasing


def _jobs(cpu, ram, d, q):
    cpu, ram = np.asarray(cpu, float), np.asarray(ram, float)
    return SimpleNamespace(A=np.column_stack([cpu, ram]), duration_hours=np.asarray(d, float),
                           priority=np.asarray(q, float), n=cpu.size)


def test_cost_base_hand_made_job():
    cfg = default_config()
    u0 = 0.5
    c = cost_base([0.5], [0.25], [4.0], u0, cfg)                   # 0.5 machine = 32 cores, 0.25 machine = 64 GB
    assert c[0] == pytest.approx(u0 * (1 + np.log(4.0)) + G1 * 32 + G2 * 64, rel=1e-12)    # $ per hour, no d


def test_u0_is_the_rate_of_the_median_job():
    cfg = default_config()
    H = _jobs([0.01, 0.02, 0.5], [0.01, 0.30, 0.01], [1, 1, 1], [1, 1, 1])
    rates = G1 * 64 * H.A[:, 0] + G2 * 256 * H.A[:, 1]
    assert calibrate_u0(H, cfg) == pytest.approx(np.sort(rates)[1], rel=1e-12)


def test_sigma_refit_and_customer_value_is_the_bid():
    cfg = default_config()
    n = 200_000
    J = _jobs(np.full(n, 0.01), np.full(n, 0.01), np.full(n, 1.0), np.random.default_rng(1).integers(1, 6, n))
    b = synthesize_bids(J, cfg, u0=0.02, seed=3)
    np.testing.assert_allclose(b.v_rate.mean() / b.cost_base.mean(), 1.0, atol=5e-3)    # mean bid = cost base
    assert fit_sigma(b.v_rate, b.cost_base) == pytest.approx(cfg.bid.sigma, abs=2e-3)
    assert bid_model(cfg, b.v_rate, b.cost_base).sigma == pytest.approx(fit_sigma(b.v_rate, b.cost_base))
    from dataclasses import replace
    fixed = replace(cfg, bid=replace(cfg.bid, refit_sigma=False))
    assert bid_model(fixed, b.v_rate, b.cost_base).sigma == cfg.bid.sigma
    d = np.random.default_rng(2).uniform(0.1, 24.0, n)
    t = job_terms(b.v_rate, b.cost_base, d, np.zeros(n), np.zeros(n), MyersonModel.known_markup(0.3), 0.0, cfg.objective)
    np.testing.assert_array_equal(t.cv, b.v_rate * d)                # V = b d, no priority factor

"""Week 32: (1) the virtual value of the known lognormal markup: E[phi_m] = 0 (exact identity), the code's phi_m equals
the stable form m (1 - sigma Mills(z)) over the range that draws reach; (2) bid.mode size_u0: cost base
(u0 r + gamma1 cores + gamma2 GB) d, no priority in the bid, customer value = q x bid; the other modes unchanged."""
import numpy as np
import pytest
from scipy import integrate, stats

from dlent_rt.bid import LognormalBidParams, MyersonModel, compute_base_utility, synthesize_lognormal
from dlent_rt.reward import job_terms
from dlent_rt.config import ObjectiveCfg

G1, G2, U0 = 0.0332, 0.00445, 23.93


@pytest.mark.parametrize("sigma", [0.1, 0.3, 0.5])
def test_expected_phi_markup_is_zero(sigma):
    mu = -sigma ** 2 / 2
    # E[phi(m)] = E[m] - int_0^inf (1 - F(m)) dm, and in z (m = exp(mu + sigma z)): int (1 - F) dm = sigma int m (1 - Phi) dz
    I, _ = integrate.quad(lambda z: np.exp(mu + sigma * z) * stats.norm.sf(z), -60, 60, limit=500, epsabs=1e-13)
    assert abs(1.0 - sigma * I) < 5e-3
    z = np.linspace(-6, 6, 2001)
    m = np.exp(mu + sigma * z)
    mills = np.exp(stats.norm.logsf(z) - stats.norm.logpdf(z))
    np.testing.assert_allclose(MyersonModel.known_markup(sigma).phi_markup(m), m * (1 - sigma * mills), rtol=1e-6, atol=1e-6)


def _bid(mode, cpu, ram, d, q, seed=0):
    p = LognormalBidParams(sigma=0.3, base_utility_u0=23.926, base_utility_mode="log", gamma1=G1, gamma2=G2,
                           machine_cores=64, machine_ram_gb=256, mode=mode, u0_per_machine_hour=U0)
    return synthesize_lognormal(np.asarray(cpu, float), np.asarray(ram, float), np.asarray(d, float),
                                np.asarray(q, float), p, np.random.default_rng(seed))


def test_size_u0_cost_base_hand_made_job():
    b = _bid("size_u0", [0.5], [0.25], [2.0], [4.0])            # 0.5 machine = 32 cores, 0.25 machine = 64 GB, 2 h
    assert b.cost_base[0] == pytest.approx((U0 * 0.5 + G1 * 32 + G2 * 64) * 2.0, rel=1e-12)


def test_size_u0_bid_does_not_depend_on_priority_and_cv_is_q_times_bid():
    cpu, ram, d = np.full(5, 0.1), np.full(5, 0.05), np.full(5, 1.5)
    q = np.array([1., 2., 3., 4., 5.])
    b = _bid("size_u0", cpu, ram, d, q)
    b1 = _bid("size_u0", cpu, ram, d, np.ones(5))
    np.testing.assert_array_equal(b.cost_base, b1.cost_base)
    np.testing.assert_array_equal(b.v_rate, b1.v_rate)
    m = MyersonModel.known_markup(0.3)
    t = job_terms(q, b.v_rate, b.cost_base, np.zeros(5), np.zeros(5), m, 0.0, ObjectiveCfg())
    np.testing.assert_allclose(t.cv, q * b.v_rate, rtol=1e-15)


def test_old_modes_unchanged():
    cpu, ram, d, q = np.array([0.01, 0.2]), np.array([0.02, 0.1]), np.array([0.1, 1.0]), np.array([1., 4.])
    old = _bid("base_utility", cpu, ram, d, q)
    np.testing.assert_allclose(old.cost_base, (compute_base_utility(q, 23.926, "log") + G1 * 64 * cpu + G2 * 256 * ram) * d,
                               rtol=1e-12)
    size = _bid("size", cpu, ram, d, q)
    np.testing.assert_allclose(size.cost_base, (1 + np.log(q)) * (G1 * 64 * cpu + G2 * 256 * ram) * d, rtol=1e-12)

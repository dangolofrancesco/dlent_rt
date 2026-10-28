"""bid.mode = "size" (week 32): cost base = (1 + ln q) * (gamma1 * cores + gamma2 * GB) * duration, no u0; the bid is the
same lognormal markup around it. The default mode "base_utility" is unchanged."""
from dataclasses import replace

import numpy as np
import pytest

from dlent_rt.bid import LognormalBidParams, compute_base_utility, synthesize_lognormal
from dlent_rt.config import default_config

G1, G2 = 0.0332, 0.00445


def _bid(mode, cpu, ram, d, q, seed=0, **kw):
    p = LognormalBidParams(sigma=0.3, base_utility_u0=23.926, base_utility_mode="log", gamma1=G1, gamma2=G2,
                           machine_cores=64, machine_ram_gb=256, mode=mode, **kw)
    return synthesize_lognormal(np.asarray(cpu, float), np.asarray(ram, float), np.asarray(d, float),
                                np.asarray(q, float), p, np.random.default_rng(seed))


def test_size_cost_base_hand_made_job():
    # 0.5 machine CPU = 32 cores, 0.25 machine RAM = 64 GB, priority tier 4, 2 hours
    b = _bid("size", [0.5], [0.25], [2.0], [4.0])
    assert b.cost_base[0] == pytest.approx((1 + np.log(4)) * (G1 * 32 + G2 * 64) * 2.0, rel=1e-12)


def test_size_bid_is_linear_in_cores_gb_and_duration():
    cpu, ram, d, q = np.array([0.01, 0.2, 0.5]), np.array([0.02, 0.1, 0.3]), np.array([0.1, 1.0, 5.0]), np.array([1., 3., 5.])
    base = _bid("size", cpu, ram, d, q)
    for k in (0.5, 2.0, 3.0):
        both = _bid("size", k * cpu, k * ram, d, q)             # same seed -> same markups
        np.testing.assert_allclose(both.cost_base, k * base.cost_base, rtol=1e-12)
        np.testing.assert_allclose(both.v_rate, k * base.v_rate, rtol=1e-12)
        longer = _bid("size", cpu, ram, k * d, q)
        np.testing.assert_allclose(longer.v_rate, k * base.v_rate, rtol=1e-12)
    only_cpu = _bid("size", 2 * cpu, ram, d, q).cost_base - base.cost_base
    np.testing.assert_allclose(only_cpu, (1 + np.log(q)) * G1 * 64 * cpu * d, rtol=1e-10)
    # the markup v / cost_base is the same lognormal in both modes (same seed)
    old = _bid("base_utility", cpu, ram, d, q)
    np.testing.assert_allclose(base.v_rate / base.cost_base, old.v_rate / old.cost_base, rtol=1e-12)


def test_default_mode_is_unchanged():
    assert default_config().bid.mode == "base_utility"
    cpu, ram, d, q = np.array([0.01, 0.2]), np.array([0.02, 0.1]), np.array([0.1, 1.0]), np.array([1., 4.])
    b = _bid("base_utility", cpu, ram, d, q)
    want = (compute_base_utility(q, 23.926, "log") + G1 * 64 * cpu + G2 * 256 * ram) * d
    np.testing.assert_allclose(b.cost_base, want, rtol=1e-12)
    rng = np.random.default_rng(0)
    np.testing.assert_allclose(b.v_rate, rng.lognormal(np.log(want) - 0.045, 0.3), rtol=1e-12)

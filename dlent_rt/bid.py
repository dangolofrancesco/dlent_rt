"""
Bid synthesis and Myerson virtual value computation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import lognorm


# Moved from grid.py — single source of truth for virtual value logic
@dataclass
class MyersonModel:
    mu: float                 # log-space mean
    sigma: float              # log-space std

    def phi(self, v):
        """Myerson virtual value: phi(v) = v - (1 - F(v)) / f(v)."""
        v = np.asarray(v, dtype=float)
        scale = np.exp(self.mu)
        F = lognorm.cdf(v, s=self.sigma, scale=scale)
        f = lognorm.pdf(v, s=self.sigma, scale=scale)
        inv_hazard = np.where(f > 1e-12, (1.0 - F) / f, 0.0)
        return v - inv_hazard

    def phi_inv(self, target_phi, v_lo: float = 1e-9, v_hi: float = 1e12):
        """Invert phi numerically (monotone for a regular lognormal)."""
        target = np.atleast_1d(np.asarray(target_phi, dtype=float))
        lo = np.full_like(target, v_lo)
        hi = np.full_like(target, v_hi)
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            pm = np.asarray(self.phi(mid))
            go_up = pm < target
            lo = np.where(go_up, mid, lo)
            hi = np.where(go_up, hi, mid)
        out = 0.5 * (lo + hi)
        return out if out.size > 1 else float(out[0])


def fit_myerson(v_rate: np.ndarray) -> MyersonModel:
    """Lognormal MLE in log-space on positive bids."""
    v = v_rate[v_rate > 0]
    logs = np.log(v)
    mu = float(logs.mean())
    sigma = float(logs.std(ddof=1)) if v.size > 1 else 1.0
    sigma = max(sigma, 1e-6)
    return MyersonModel(mu=mu, sigma=sigma)


@dataclass
class UniformBidParams:
    """Parameters for the cost-based uniform bid synthesis."""
    gamma1: float = 0.02       # $/core-hour (infrastructure reserve price)
    gamma2: float = 0.004      # $/GB-hour (infrastructure reserve price)
    base_utility: float = 50.0
    spec_cpu_core: float = 64  # cores per 1.0 normalized CPU
    spec_ram_gb: float = 256   # GB per 1.0 normalized RAM


@dataclass
class LognormalBidParams:
    """Parameters for the legacy lognormal bid synthesis."""
    sigma: float = 1.5         
    base_multiplier: float = 1.0
    base_utility: float = 50.0
    gamma1: float = 0.01
    gamma2: float = 0.002
    spec_cpu_core: float = 64  # cores per 1.0 normalized CPU
    spec_ram_gb: float = 256   # GB per 1.0 normalized RAM


@dataclass
class BidResult:
    """Output of bid synthesis for a batch of jobs."""
    v_rate: np.ndarray        
    phi_rate: np.ndarray      
    bid_low: np.ndarray       
    bid_high: np.ndarray      


def synthesize_uniform(
    cpu_norm: np.ndarray, ram_norm: np.ndarray,
    duration_hours: np.ndarray, priority: np.ndarray,
    params: UniformBidParams, rng: np.random.Generator,
) -> BidResult:
    """
    Cost-based uniform bid synthesis.
    """
    cpu_cores = cpu_norm * params.spec_cpu_core
    ram_gb = ram_norm * params.spec_ram_gb
    cost_base = (
        params.base_utility + params.gamma1 * cpu_cores + params.gamma2 * ram_gb
    ) * duration_hours
    cost_base = np.maximum(cost_base, 1e-9)  # avoid zero

    M = 1.0 + np.log(priority + 1.0)
    bid_low = cost_base
    bid_high = (1.0 + 2.0 * M) * cost_base

    v_rate = rng.uniform(bid_low, bid_high)
    phi_rate = 2.0 * v_rate - bid_high  # closed-form Myerson for U[a,b]

    return BidResult(v_rate=v_rate, phi_rate=phi_rate,
                     bid_low=bid_low, bid_high=bid_high)


def synthesize_lognormal(
    cpu_norm: np.ndarray, ram_norm: np.ndarray,
    duration_hours: np.ndarray,
    params: LognormalBidParams, rng: np.random.Generator,
) -> BidResult:
    cpu_cores = cpu_norm * params.spec_cpu_core
    ram_gb = ram_norm * params.spec_ram_gb
    cost_base = (
        params.base_utility + params.gamma1 * cpu_cores + params.gamma2 * ram_gb
    ) * duration_hours
    cost_base = np.maximum(cost_base, 1e-9)
    mu = np.log(cost_base) - (params.sigma ** 2) / 2.0

    v_rate = rng.lognormal(mean=mu, sigma=params.sigma)

    # Numerical Myerson: phi(v) = v - (1-F(v))/f(v)
    F = lognorm.cdf(v_rate, s=params.sigma, scale=np.exp(mu))
    f = lognorm.pdf(v_rate, s=params.sigma, scale=np.exp(mu))
    inv_hazard = np.where(f > 1e-12, (1.0 - F) / f, 0.0)
    phi_rate = v_rate - inv_hazard

    # For lognormal, bid_low/high are theoretical (not compact support)
    bid_low = np.full_like(v_rate, 0.0)
    bid_high = v_rate * 3.0  # rough upper bound for normalization

    return BidResult(v_rate=v_rate, phi_rate=phi_rate,
                     bid_low=bid_low, bid_high=bid_high)


def compute_phi_tilde(
    L: np.ndarray,
    bid_high_bar: np.ndarray,
    strategy: str,
    myerson: "MyersonModel",
) -> np.ndarray:
    if strategy == "uniform":
        return 2.0 * L - bid_high_bar
    else:
        # bid_high_bar has no meaning for lognormal bids (no compact support)
        return np.asarray(myerson.phi(L))
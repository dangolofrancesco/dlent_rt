"""
Bid synthesis and Myerson virtual value computation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import lognorm


# Moved from grid.py — single source of truth for virtual value logic
@dataclass
class MyersonModel:
    """
    Lognormal Myerson model. Two modes:
      known=True (default, bid.known_distribution): the distribution is KNOWN a
        priori from the config and never refit. (mu, sigma) = (-sigma^2/2, sigma)
        parametrise the MARKUP m = v / c ~ LogNormal(-sigma^2/2, sigma), c = the
        job's reference cost (cost_base), and phi_job(v, c) = c * phi_markup(v / c).
      known=False (ablation, legacy): F = LogNormal(mu, sigma) fitted on the pooled
        dollar bids v (dlent_rt.legacy.pooled_mle.fit_myerson); phi_job(v, c) = phi of
        that fit at v, c is ignored.

    phi_job(v, c_j) is the ONLY entry point for the virtual value of a dollar bid; it
    is used by Phase 0, ReInit, the hindsight benchmark and the online reward.
    phi_markup(m) takes a markup m = v / c_j (a ratio, NOT a dollar bid).
    """
    mu: float                 # log-space mean (known=True: of the markup m)
    sigma: float              # log-space std  (known=True: bid.sigma from the config)
    known: bool = False

    @classmethod
    def known_markup(cls, sigma: float) -> "MyersonModel":
        return cls(mu=-0.5 * float(sigma) ** 2, sigma=float(sigma), known=True)

    def phi_job(self, v, c=None):
        """Virtual value of the DOLLAR bid v of a job with reference cost c (see class doc)."""
        if not self.known:
            return self._phi(v)
        if c is None:
            raise ValueError("known-distribution phi needs the job's reference cost c")
        c = np.asarray(c, dtype=float)
        return c * np.asarray(self.phi_markup(np.asarray(v, dtype=float) / c))

    def phi_markup(self, m):
        """Virtual value of the MARKUP m = v / c_j under LogNormal(-sigma^2/2, sigma).
        The argument is a ratio, not a dollar bid: for a bid use phi_job(v, c_j).
        Known distribution only (the legacy pooled fit is over dollar bids)."""
        if not self.known:
            raise ValueError("phi_markup needs the known markup distribution; "
                             "for the pooled fit use phi_job(v)")
        return self._phi(m)

    def _phi(self, x):
        """Myerson virtual value of the model's own variable x (markup if known, dollar
        bid if pooled): phi(x) = x - (1 - F(x)) / f(x)."""
        x = np.asarray(x, dtype=float)
        scale = np.exp(self.mu)
        F = lognorm.cdf(x, s=self.sigma, scale=scale)
        f = lognorm.pdf(x, s=self.sigma, scale=scale)
        inv_hazard = np.where(f > 1e-12, (1.0 - F) / f, 0.0)
        return x - inv_hazard

    def phi_inv(self, target_phi, v_lo: float = 1e-9, v_hi: float = 1e12):
        """Invert the model's own phi numerically (monotone for a regular lognormal):
        a markup if known (phi_inv(0) = reserve markup), a dollar bid if pooled."""
        target = np.atleast_1d(np.asarray(target_phi, dtype=float))
        lo = np.full_like(target, v_lo)
        hi = np.full_like(target, v_hi)
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            pm = np.asarray(self._phi(mid))
            go_up = pm < target
            lo = np.where(go_up, mid, lo)
            hi = np.where(go_up, hi, mid)
        out = 0.5 * (lo + hi)
        return out if out.size > 1 else float(out[0])


def bid_model(cfg, v_rate: np.ndarray) -> MyersonModel:
    """The mechanism's bid model: known markup distribution (bid.known_distribution,
    sigma = bid.sigma, no fit) or the legacy pooled lognormal MLE on v_rate."""
    if cfg.bid.known_distribution:
        return MyersonModel.known_markup(cfg.bid.sigma)
    from .legacy.pooled_mle import fit_myerson      # legacy ablation (audit L2)
    return fit_myerson(v_rate)


def compute_base_utility(
    priority: np.ndarray, u0: float, mode: str,
) -> np.ndarray:
    """
    Priority-dependent base utility rate ($/hour).

    "linear": U_base(q) = u0 * q
    "log":    U_base(q) = u0 * (1 + ln(q))

    Ref: Zheng et al. (2015) "Star: SLA-aware autonomic management of
    cloud resources" — base utility scales with SLA/priority tier;
    Zhang et al. (2017) "An Online Mechanism for Cloud Resource
    Allocation" — job-type coefficient modulating valuation floor.
    """
    if mode == "linear":
        return u0 * priority
    elif mode == "log":
        return u0 * (1.0 + np.log(priority))
    else:
        raise ValueError(f"unknown base_utility_mode: {mode}")


@dataclass
class LognormalBidParams:
    """Parameters for the lognormal bid synthesis."""
    sigma: float = 1.5
    base_utility_u0: float = 50.0   # U_0, the free parameter of base_utility_mode
    base_utility_mode: str = "log"  # "linear" | "log"
    mode: str = "base_utility"      # "base_utility" | "size" | "size_u0" (cost base, see synthesize_lognormal)
    u0_per_machine_hour: float = 23.93   # size_u0: $ per CPU machine-hour
    gamma1: float = 0.01
    gamma2: float = 0.002
    machine_cores: float = 64     # DECLARED ASSUMPTION (not from Google): cores of the largest machine (1.0 CPU)
    machine_ram_gb: float = 256   # DECLARED ASSUMPTION (not from Google): GB of the largest machine (1.0 RAM)


@dataclass
class BidResult:
    """Output of bid synthesis for a batch of jobs."""
    v_rate: np.ndarray        
    phi_rate: np.ndarray      # generator phi (diagnostic; legacy Phase 0 reward)
    bid_high: np.ndarray      
    cost_base: Optional[np.ndarray] = None   # reference cost c_j (public, from features)


def synthesize_lognormal(
    cpu_norm: np.ndarray, ram_norm: np.ndarray,
    duration_hours: np.ndarray, priority: np.ndarray,
    params: LognormalBidParams, rng: np.random.Generator,
) -> BidResult:
    cpu_cores = cpu_norm * params.machine_cores
    ram_gb = ram_norm * params.machine_ram_gb
    if params.mode == "size_u0":
        # size-based bid with a per-machine-hour utility; no priority (it enters only the customer value q x bid)
        cost_base = (
            params.u0_per_machine_hour * cpu_norm + params.gamma1 * cpu_cores + params.gamma2 * ram_gb
        ) * duration_hours
    elif params.mode == "size":
        # size-based bid: the reserve price of the requested resources, scaled by priority (1 + ln q); no u0
        cost_base = (1.0 + np.log(priority)) * (
            params.gamma1 * cpu_cores + params.gamma2 * ram_gb
        ) * duration_hours
    else:
        base_utility = compute_base_utility(
            priority, params.base_utility_u0, params.base_utility_mode,
        )
        cost_base = (
            base_utility + params.gamma1 * cpu_cores + params.gamma2 * ram_gb
        ) * duration_hours
    cost_base = np.maximum(cost_base, 1e-9)
    mu = np.log(cost_base) - (params.sigma ** 2) / 2.0

    v_rate = rng.lognormal(mean=mu, sigma=params.sigma)

    # Numerical Myerson: phi(v) = v - (1-F(v))/f(v)
    F = lognorm.cdf(v_rate, s=params.sigma, scale=np.exp(mu))
    f = lognorm.pdf(v_rate, s=params.sigma, scale=np.exp(mu))
    inv_hazard = np.where(f > 1e-12, (1.0 - F) / f, 0.0)
    phi_rate = v_rate - inv_hazard

    # no compact support for lognormal bids: bid_high is only a rough scale (diagnostic)
    bid_high = v_rate * 3.0  # rough upper bound for normalization

    return BidResult(v_rate=v_rate, phi_rate=phi_rate,
                     bid_high=bid_high, cost_base=cost_base)


def compute_phi_tilde(L: np.ndarray, myerson: "MyersonModel") -> np.ndarray:
    """Per-type virtual value at the bin floor L (payment rule)."""
    if myerson.known:
        # per-type phi(L) is undefined: phi depends on the job's c_j, which is not
        # part of the type. The payment is computed per job, phi_job(L_k, c_j).
        return np.full(np.shape(L), np.nan)
    return np.asarray(myerson.phi_job(L))
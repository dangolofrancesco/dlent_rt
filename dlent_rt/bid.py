"""
Bid synthesis and the Myerson virtual value (decision D2, week 33: HOURLY bids).

    cost_base_j = U0 (1 + ln q_j) + gamma1 cores_j + gamma2 GB_j               [$ per hour]
    v_j         ~ LogNormal(ln cost_base_j - sigma^2 / 2, sigma)               [$ per hour], mean cost_base_j
    phi_j       = cost_base_j * phi_m(v_j / cost_base_j)                       [$ per hour]

The bid is a price per hour: it does not depend on the job's duration. cost_base_j is known at arrival (priority and
request). The total value V_j = v_j d_j (true duration) is used only in the results (reward.job_terms).
U0 ($ per hour per job) = gamma1 cores + gamma2 GB of the median job of H by that rate (calibrate_u0), computed once
in Phase 0 and then fixed. cores = CPU x machine_cores, GB = RAM x machine_ram_gb.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import norm

PHI_M_FLOOR = -1e12    # floor of phi_m (markup units): phi_m -> -inf as m -> 0 (binds only near overflow, z < -7)


@dataclass
class MyersonModel:
    """
    The KNOWN markup distribution m = v / c ~ LogNormal(-sigma^2/2, sigma), c = the job's cost_base.
    phi_job(v, c) = c * phi_markup(v / c) is the ONLY entry point for the virtual value of a dollar bid; it is used
    by Phase 0, ReInit, the benchmarks and the online reward. phi_markup(m) takes a markup (a ratio).
    """
    mu: float                 # log-space mean of the markup (= -sigma^2/2)
    sigma: float              # log-space std of the markup

    @classmethod
    def known_markup(cls, sigma: float) -> "MyersonModel":
        return cls(mu=-0.5 * float(sigma) ** 2, sigma=float(sigma))

    def phi_job(self, v, c):
        """Virtual value of the DOLLAR bid v of a job with cost base c."""
        c = np.asarray(c, dtype=float)
        return c * np.asarray(self.phi_markup(np.asarray(v, dtype=float) / c))

    def phi_markup(self, m):
        """
        phi_m(m) = m - (1 - F(m)) / f(m) under LogNormal(-sigma^2/2, sigma), in the stable form
        m (1 - sigma Mills(z)), z = (ln m - mu) / sigma, Mills = (1 - Phi) / phi of the standard normal (log space).
        Increasing in m; floored at PHI_M_FLOOR (m <= 0 gives the floor).
        """
        m = np.asarray(m, dtype=float)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            z = (np.log(np.where(m > 0, m, 1.0)) - self.mu) / self.sigma
            out = m * (1.0 - self.sigma * np.exp(norm.logsf(z) - norm.logpdf(z)))
        return np.maximum(np.where((m > 0) & ~np.isnan(out), out, PHI_M_FLOOR), PHI_M_FLOOR)

    def phi_inv(self, target_phi, m_lo: float = 1e-9, m_hi: float = 1e12):
        """Invert phi_markup numerically (it is increasing): phi_inv(0) = the reserve markup."""
        target = np.atleast_1d(np.asarray(target_phi, dtype=float))
        lo = np.full_like(target, m_lo)
        hi = np.full_like(target, m_hi)
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            go_up = np.asarray(self.phi_markup(mid)) < target
            lo = np.where(go_up, mid, lo)
            hi = np.where(go_up, hi, mid)
        out = 0.5 * (lo + hi)
        return out if out.size > 1 else float(out[0])


def log_markups(v: np.ndarray, cost_base: np.ndarray) -> np.ndarray:
    """ln(v / cost_base) of every bid."""
    return np.log(np.asarray(v, float) / np.asarray(cost_base, float))


def fit_sigma(v: np.ndarray, cost_base: np.ndarray) -> float:
    """sigma refit on the markups: the standard deviation of ln(v / cost_base) (theoretical model, D2)."""
    return float(np.std(log_markups(v, cost_base)))


def bid_model(cfg, v: Optional[np.ndarray] = None, cost_base: Optional[np.ndarray] = None) -> MyersonModel:
    """The mechanism's bid model: sigma fit on (v, cost_base) when at least two bids are given (Phase 0 on H; ReInit
    by bid.sigma_update), else bid.sigma (the generator's)."""
    if v is not None and np.size(v) >= 2:
        return MyersonModel.known_markup(fit_sigma(v, cost_base))
    return MyersonModel.known_markup(cfg.bid.sigma)


def cost_rate(cpu: np.ndarray, ram: np.ndarray, cfg) -> np.ndarray:
    """gamma1 cores + gamma2 GB per job [$ per hour]."""
    em = cfg.energy_model
    return cfg.bid.gamma1 * np.asarray(cpu, float) * em.machine_cores + cfg.bid.gamma2 * np.asarray(ram, float) * em.machine_ram_gb


def calibrate_u0(H, cfg) -> float:
    """U0 [$ per hour per job] = the cost rate gamma1 cores + gamma2 GB of the median job of H by that rate."""
    return float(np.median(cost_rate(H.A[:, 0], H.A[:, 1], cfg)))


@dataclass
class BidResult:
    """Output of bid synthesis for a batch of jobs."""
    v_rate: np.ndarray                       # bid v [$ per hour]
    phi_rate: np.ndarray                     # phi_job(v, cost_base) under the GENERATOR's sigma (diagnostic)
    cost_base: Optional[np.ndarray] = None   # c_j (public, from the job's features)


def cost_base(cpu, ram, priority, u0: float, cfg) -> np.ndarray:
    """cost_base_j = U0 (1 + ln q_j) + gamma1 cores_j + gamma2 GB_j  [$ per hour]; no duration."""
    return u0 * (1.0 + np.log(np.asarray(priority, float))) + cost_rate(cpu, ram, cfg)


def synthesize_bids(jobs, cfg, u0: float, seed: int) -> BidResult:
    """Hourly bids of `jobs` (JobArrays): v ~ LogNormal(ln cost_base - sigma^2/2, sigma), sigma = bid.sigma. The
    duration is not read."""
    rng = np.random.default_rng(seed)
    c = cost_base(jobs.A[:, 0], jobs.A[:, 1], jobs.priority, u0, cfg)
    sigma = cfg.bid.sigma
    v = rng.lognormal(mean=np.log(c) - sigma ** 2 / 2.0, sigma=sigma)
    phi = np.asarray(MyersonModel.known_markup(sigma).phi_job(v, c))
    return BidResult(v_rate=v, phi_rate=phi, cost_base=c)

"""
Myerson virtual valuation and the geometric discretization grid.

The static DataGenerator already fits a lognormal to bids and precomputes
phi_rate per job. For Phase 0 and for phase transitions we re-fit the empirical
distribution on the batch at hand (lognormal MLE in log-space, matching the
generator's model) and build the geometric grid over virtual values.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import lognorm


@dataclass
class MyersonModel:
    """A fitted lognormal bid model plus its Myerson transform."""
    mu: float                 # log-space mean
    sigma: float              # log-space std
    phi_floor: float

    def phi(self, v: np.ndarray | float) -> np.ndarray | float:
        """Myerson virtual value: phi(v) = v - (1 - F(v)) / f(v)."""
        v = np.asarray(v, dtype=float)
        scale = np.exp(self.mu)
        F = lognorm.cdf(v, s=self.sigma, scale=scale)
        f = lognorm.pdf(v, s=self.sigma, scale=scale)
        inv_hazard = np.where(f > 1e-12, (1.0 - F) / f, 0.0)
        return v - inv_hazard

    def phi_inv(self, target_phi: np.ndarray | float,
                v_lo: float = 1e-6, v_hi: float = 1e12) -> np.ndarray | float:
        """
        Invert phi numerically (phi is monotone increasing for a regular
        lognormal). Vectorised bisection.
        """
        target = np.atleast_1d(np.asarray(target_phi, dtype=float))
        lo = np.full_like(target, v_lo)
        hi = np.full_like(target, v_hi)
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            pm = np.asarray(self.phi(mid))
            go_up = pm < target
            lo = np.where(go_up, mid, lo)
            hi = np.where(go_up, hi, mid)
        out = 0.5 * (lo + hi)
        return out if out.size > 1 else float(out[0])


def fit_myerson(v_rate: np.ndarray, phi_floor: float) -> MyersonModel:
    """Lognormal MLE in log-space on positive bids."""
    v = v_rate[v_rate > 0]
    logs = np.log(v)
    mu = float(logs.mean())
    sigma = float(logs.std(ddof=1)) if v.size > 1 else 1.0
    sigma = max(sigma, 1e-6)
    return MyersonModel(mu=mu, sigma=sigma, phi_floor=phi_floor)


@dataclass
class GeometricGrid:
    """Geometric grid over virtual values with its bid-space image."""
    bnd_phi: np.ndarray       # (K*+1,) virtual-value bin edges
    bnd_v: np.ndarray         # (K*+1,) bid-space bin edges = phi^{-1}(bnd_phi)
    k_star: int
    ratio: float

    def bin_of(self, phi_val: np.ndarray | float) -> np.ndarray | int:
        """
        Return the bin index k in {1,...,K*} for a virtual value, or 0 (bottom
        sentinel, "below lowest bin floor" -> rigid IR reject) if below bnd_phi[0].

        Bin k covers [bnd_phi[k-1], bnd_phi[k]).
        """
        phi_val = np.atleast_1d(np.asarray(phi_val, dtype=float))
        # searchsorted on the edges: idx in 0..K*+1
        idx = np.searchsorted(self.bnd_phi, phi_val, side="right")
        # idx == 0  -> below the lowest floor -> sentinel 0 (bot / reject)
        # idx in 1..K* -> valid bin idx
        # idx == K*+1 -> above the top edge -> clamp into the top bin K*
        k = np.clip(idx, 0, self.k_star)
        return k if k.size > 1 else int(k[0])


def build_geometric_grid(
    phi_values: np.ndarray, k_star: int, phi_floor: float, myerson: MyersonModel
) -> GeometricGrid:
    """
    Build the geometric grid over observed virtual values.

        Phi_min = max(min phi, phi_floor)   (guardrail against ratio blowup)
        Phi_max = max phi
        ratio   = (Phi_max / Phi_min)^(1/K*)
        bnd_phi = [Phi_min * ratio^k for k in 0..K*]
        bnd_v   = phi^{-1}(bnd_phi)
    """
    phi_min = max(float(np.min(phi_values)), phi_floor)
    phi_max = float(np.max(phi_values))
    if phi_max <= phi_min:
        phi_max = phi_min * (1.0 + 1e-6)          # degenerate guard
    ratio = (phi_max / phi_min) ** (1.0 / k_star)
    ks = np.arange(k_star + 1)
    bnd_phi = phi_min * ratio ** ks
    bnd_v = np.asarray(myerson.phi_inv(bnd_phi))
    return GeometricGrid(bnd_phi=bnd_phi, bnd_v=bnd_v, k_star=k_star, ratio=ratio)

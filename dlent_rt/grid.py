"""
Myerson virtual valuation and the valuation discretization grid.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# MyersonModel and fit_myerson have moved to bid.py.
# Re-exported here for backwards compatibility.
from .bid import MyersonModel, fit_myerson


# Four spacing strategies
def _edges_quantile(values: np.ndarray, k: int) -> np.ndarray:
    """
    Equal-MASS bins: edges at evenly spaced quantiles, so each bin holds ~n/k
    observations. Strongly preferred for heavily skewed data -- geometric or
    linear spacing on a skewed distribution starves most bins and piles all the
    mass into one or two.
    """
    probs = np.linspace(0.0, 1.0, k + 1)
    e = np.quantile(values, probs)
    e = np.maximum.accumulate(e)
    for i in range(1, len(e)):
        if e[i] <= e[i - 1]:
            e[i] = np.nextafter(e[i - 1], np.inf)
    return e


def _edges_geometric(values: np.ndarray, k: int, floor: float) -> np.ndarray:
    """Multiplicative spacing (paper-faithful). Requires strictly positive values."""
    lo = max(float(np.min(values)), floor)
    hi = float(np.max(values))
    if hi <= lo:
        hi = lo * (1.0 + 1e-6)
    ratio = (hi / lo) ** (1.0 / k)
    return lo * ratio ** np.arange(k + 1)


def _edges_log_linear(values: np.ndarray, k: int, floor: float) -> np.ndarray:
    """Uniform spacing in log-space."""
    lo = np.log(max(float(np.min(values)), floor))
    hi = np.log(max(float(np.max(values)), floor * (1 + 1e-6)))
    if hi <= lo:
        hi = lo + 1e-6
    return np.exp(np.linspace(lo, hi, k + 1))


def _edges_linear(values: np.ndarray, k: int) -> np.ndarray:
    """Uniform spacing in the native space."""
    lo, hi = float(np.min(values)), float(np.max(values))
    if hi <= lo:
        hi = lo + 1e-9
    return np.linspace(lo, hi, k + 1)


def build_edges(values: np.ndarray, k: int, spacing: str, floor: float) -> np.ndarray:
    if spacing == "quantile":
        return _edges_quantile(values, k)
    if spacing == "geometric":
        return _edges_geometric(values, k, floor)
    if spacing == "log_linear":
        return _edges_log_linear(values, k, floor)
    if spacing == "linear":
        return _edges_linear(values, k)
    raise ValueError(f"unknown spacing: {spacing}")



# The valuation grid
@dataclass
class ValuationGrid:
    """
    A K*-bin discretization of the valuation axis.

    `space` says WHICH quantity the edges live in:
      "v"           -> edges are bids;            bnd_v = edges
      "phi"         -> edges are virtual values;  bnd_v = phi^{-1}(edges)
      "phi_shifted" -> edges are (phi + shift);   bnd_v = phi^{-1}(edges - shift)

    In every case `bnd_v` is the bid-space image of the edges, which is what the
    payment rule and the reward's bin-floor term need.

    `assign()` CLIPS rather than rejects when space is "v" or "phi_shifted", so
    every arrival receives a bin. Only the legacy "phi" space can emit k=0.
    """
    edges: np.ndarray
    bnd_v: np.ndarray
    bnd_phi: np.ndarray
    k_star: int
    space: str
    spacing: str
    shift: float = 0.0
    rejects_below_floor: bool = False

    @property
    def ratio(self) -> float:
        lo, hi = float(self.edges[0]), float(self.edges[-1])
        if lo <= 0 or hi <= 0:
            return float("nan")
        return (hi / lo) ** (1.0 / self.k_star)

    def _to_space(self, v, myerson):
        if self.space == "v":
            return v
        ph = np.asarray(myerson.phi(v))
        if self.space == "phi":
            return ph
        return ph + self.shift

    def assign(self, v, myerson) -> np.ndarray:
        """Map bids to bin indices k in 1..K* (0 = reject sentinel, legacy only)."""
        x = np.atleast_1d(self._to_space(np.atleast_1d(np.asarray(v, float)), myerson))
        idx = np.searchsorted(self.edges, x, side="right")
        if self.rejects_below_floor:
            k = np.clip(idx, 0, self.k_star)
        else:
            k = np.clip(idx, 1, self.k_star)
        return k


def build_valuation_grid(
    v_rate: np.ndarray, myerson: MyersonModel, k_star: int,
    space: str, spacing: str, phi_floor: float,
) -> ValuationGrid:
    """Build the valuation grid in the configured space with the configured spacing."""
    v_pos = v_rate[v_rate > 0]

    if space == "v":
        edges = build_edges(v_pos, k_star, spacing, phi_floor)
        bnd_v = edges.copy()
        bnd_phi = np.asarray(myerson.phi(bnd_v))
        shift, rejects = 0.0, False

    elif space == "phi":
        vals = np.asarray(myerson.phi(v_pos))
        edges = build_edges(vals, k_star, spacing, phi_floor)
        bnd_phi = edges.copy()
        bnd_v = np.asarray(myerson.phi_inv(bnd_phi))
        shift, rejects = 0.0, True

    elif space == "phi_shifted":
        raw = np.asarray(myerson.phi(v_pos))
        shift = float(-raw.min() + max(phi_floor, 1e-9))
        vals = raw + shift
        edges = build_edges(vals, k_star, spacing, phi_floor)
        bnd_phi = edges - shift
        bnd_v = np.asarray(myerson.phi_inv(bnd_phi))
        rejects = False

    else:
        raise ValueError(f"unknown grid space: {space}")

    return ValuationGrid(
        edges=edges, bnd_v=bnd_v, bnd_phi=bnd_phi, k_star=k_star,
        space=space, spacing=spacing, shift=shift, rejects_below_floor=rejects,
    )

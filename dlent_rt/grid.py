"""
Myerson virtual valuation and the valuation discretization grid.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .bid import MyersonModel


# Bin edges: geometric (multiplicative) spacing, the only spacing (paper-faithful)
def _edges_geometric(values: np.ndarray, k: int, floor: float) -> np.ndarray:
    """Multiplicative spacing (paper-faithful). Requires strictly positive values."""
    lo = max(float(np.min(values)), floor)
    hi = float(np.max(values))
    if hi <= lo:
        hi = lo * (1.0 + 1e-6)
    ratio = (hi / lo) ** (1.0 / k)
    return lo * ratio ** np.arange(k + 1)


def build_edges(values: np.ndarray, k: int, floor: float) -> np.ndarray:
    return _edges_geometric(values, k, floor)



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
        ph = np.asarray(myerson.phi_job(v))   # pooled fit only (phi spaces)
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
    space: str, phi_floor: float,
) -> ValuationGrid:
    """Build the geometric valuation grid in the configured space."""
    v_pos = v_rate[v_rate > 0]
    if getattr(myerson, "known", False) and space != "v":
        raise ValueError("known bid distribution: phi depends on the job's c_j, "
                         "so the grid must be built in bid space (grid.space='v')")

    if space == "v":
        edges = build_edges(v_pos, k_star, phi_floor)
        bnd_v = edges.copy()
        # known distribution: phi of a bin edge is undefined without c_j
        bnd_phi = (np.full(bnd_v.shape, np.nan) if getattr(myerson, "known", False)
                   else np.asarray(myerson.phi_job(bnd_v)))
        shift, rejects = 0.0, False

    elif space in ("phi", "phi_shifted"):
        from .legacy.pooled_mle import phi_space_grid      # legacy ablation (audit L4)
        edges, bnd_v, bnd_phi, shift, rejects = phi_space_grid(
            v_pos, myerson, k_star, space, phi_floor)

    else:
        raise ValueError(f"unknown grid space: {space}")

    return ValuationGrid(
        edges=edges, bnd_v=bnd_v, bnd_phi=bnd_phi, k_star=k_star,
        space=space, shift=shift, rejects_below_floor=rejects,
    )

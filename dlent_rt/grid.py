"""
The valuation grid (decision D4): geometric bins on bids that cover EVERY possible bid.

    edges   L_k = v_min * rho^k,  k = 0..K*,  rho = (v_max / v_min)^(1 / K*)

v_min / v_max = the smallest / largest (positive) bid of the sample the grid is built on. No fixed floor; bids
<= 0 are left out of the range (the only guard). The K* bins are

    bin 1        [0, L_1)
    bin k        [L_{k-1}, L_k)      k = 2..K*-1
    bin K*       [L_{K*-1}, +inf)

so a bid below v_min falls in bin 1 and a bid above v_max in bin K*. Allocation (assign) and the threshold
payment (lower_edges) use these SAME edges: the lower edge of bin 1 is 0, so a payment never exceeds the bid.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def build_edges(values: np.ndarray, k: int) -> np.ndarray:
    """K* + 1 geometric edges from the smallest to the largest positive value."""
    v = np.asarray(values, float)
    v = v[v > 0]
    if v.size == 0:
        raise ValueError("geometric grid: no positive bid")
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        hi = lo * (1.0 + 1e-6)
    ratio = (hi / lo) ** (1.0 / k)
    return lo * ratio ** np.arange(k + 1)


@dataclass
class ValuationGrid:
    """K* bins on the bid axis; bnd_v = the K* + 1 geometric edges L_0..L_K* (L_0 = v_min, L_K* = v_max)."""
    bnd_v: np.ndarray
    k_star: int

    @property
    def ratio(self) -> float:
        return float((self.bnd_v[-1] / self.bnd_v[0]) ** (1.0 / self.k_star))

    @property
    def lower_edges(self) -> np.ndarray:
        """Lower edge of every bin, k = 1..K*: 0 for bin 1, L_{k-1} for the others (the payment edges)."""
        return np.r_[0.0, self.bnd_v[1:self.k_star]]

    def assign(self, v) -> np.ndarray:
        """Bin index k in 1..K* of each bid: the interior edges L_1..L_{K*-1} split [0, +inf)."""
        x = np.atleast_1d(np.asarray(v, float))
        return np.searchsorted(self.bnd_v[1:self.k_star], x, side="right").astype(np.int64) + 1


def build_valuation_grid(v: np.ndarray, k_star: int) -> ValuationGrid:
    return ValuationGrid(bnd_v=build_edges(v, k_star), k_star=k_star)

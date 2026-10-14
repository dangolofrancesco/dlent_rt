"""
Offline Pareto-front tools (thesis scope: non-convex Pareto fronts).

The augmented Chebyshev scalarization reaches non-convex regions of the
(customer value, profit, carbon) front that a weighted sum cannot, but it
produces a reward <= 0 by construction: used as the online LP-RS reward it
makes the oracle admit nothing. It is therefore kept OUT of the online loop
(TheoryModel refuses scalarization.method != "linear"); Phase 0 accepts it
with a warning so the offline analysis can build Chebyshev type rewards.
"""
from __future__ import annotations

import numpy as np


def scalarize_chebyshev(
    f: np.ndarray, weights: np.ndarray, rho: float
) -> np.ndarray:
    """
    Augmented Chebyshev scalarization, evaluated per type independently.
    The ideal point is the per-objective max over types.
    """
    ideal = f.max(axis=0)                           # (3,)
    gap = ideal[None, :] - f                        # (n_types, 3), >= 0 ideally
    weighted = weights[None, :] * gap               # (n_types, 3)
    cheb = weighted.max(axis=1)                     # (n_types,)
    aug = rho * gap.sum(axis=1)                     # (n_types,)
    return -(cheb + aug)                            # reward = -distance

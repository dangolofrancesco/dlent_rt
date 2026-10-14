"""
LEGACY ablation (audit L2 + L4): the pooled lognormal MLE of the bid
distribution (bid.known_distribution = false) and the virtual-value grid
spaces (grid.space = "phi" / "phi_shifted", with the IR-reject sentinel k = 0).

The default model treats the bid distribution as KNOWN (markup v / c_j ~
LogNormal(-sigma^2/2, sigma), never refit) and builds the grid in bid space.
This module is reached only through dlent_rt.bid.bid_model (pooled fit),
ReInit's refit and dlent_rt.grid.build_valuation_grid (phi spaces). Remove it
once the thesis figures comparing the pooled fit with the known distribution
are final.
"""
from __future__ import annotations

import numpy as np

from ..bid import MyersonModel


def fit_myerson(v_rate: np.ndarray) -> MyersonModel:
    """Lognormal MLE in log-space on positive bids."""
    v = v_rate[v_rate > 0]
    logs = np.log(v)
    mu = float(logs.mean())
    sigma = float(logs.std(ddof=1)) if v.size > 1 else 1.0
    sigma = max(sigma, 1e-6)
    return MyersonModel(mu=mu, sigma=sigma)


def phi_space_grid(v_pos, myerson: MyersonModel, k_star: int, space: str, spacing: str,
                   phi_floor: float):
    """Edges of the valuation grid in virtual-value space (pooled fit only).
    Returns (edges, bnd_v, bnd_phi, shift, rejects_below_floor)."""
    from ..grid import build_edges
    if myerson.known:
        raise ValueError("phi-space grids need one phi(v) for every job (pooled fit only)")
    if space == "phi":
        vals = np.asarray(myerson.phi_job(v_pos))
        edges = build_edges(vals, k_star, spacing, phi_floor)
        bnd_phi = edges.copy()
        bnd_v = np.asarray(myerson.phi_inv(bnd_phi))
        shift, rejects = 0.0, True

    elif space == "phi_shifted":
        raw = np.asarray(myerson.phi_job(v_pos))
        shift = float(-raw.min() + max(phi_floor, 1e-9))
        vals = raw + shift
        edges = build_edges(vals, k_star, spacing, phi_floor)
        bnd_phi = edges - shift
        bnd_v = np.asarray(myerson.phi_inv(bnd_phi))
        rejects = False
    else:
        raise ValueError(f"unknown phi grid space: {space}")
    return edges, bnd_v, bnd_phi, shift, rejects

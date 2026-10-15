"""
The additive dollar reward: ONE source of truth for its terms (week-30 audit, Section 3).

    customer value  CV = q * v                        [$]
    profit          Pi = phi - C_elec                 [$]   phi = MyersonModel.phi_job(v, c)
    carbon          K  = E * CI                       [kgCO2]
    reward          R  = l1 * CV + l2 * Pi - l3 * s_C * K   [$ per job]

with C_elec = E * price [$] and the quantile_linear carbon price s_C [$/kgCO2], computed
once in Phase 0 and frozen. Two views of the same terms:

  * job_terms(...)   one value per job (the online loop's realised reward, the benchmarks,
                     the notebooks);
  * type_reward(...) one value per type from the per-type SAMPLE MEANS of v, phi, C_elec, K
                     (Phase 0, ReInit, the hindsight benchmark). Because every term is linear
                     in the job quantities, r_tilde_j is the mean of R over the type's jobs
                     (tests/test_reward_identity.py). The means are taken FIRST and combined
                     after, in the order Phase 0 always used, so the type reward is
                     bit-identical to the pre-module code.

Everything is vectorised (bincount for type means); no per-job or per-type loops.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .lp_oracle import scalarize


# --------------------------------------------------------------------------- #
# Job-level quantities
# --------------------------------------------------------------------------- #
def electricity_cost(energy_kwh, price):
    """C_elec = E [kWh] * price [$/kWh]  ->  $."""
    return energy_kwh * price


def carbon_kg(energy_kwh, carbon_intensity):
    """K = E [kWh] * CI [kgCO2/kWh]  ->  kgCO2."""
    return energy_kwh * carbon_intensity


@dataclass(frozen=True)
class JobTerms:
    """Per-job reward terms (arrays aligned with the jobs passed to job_terms)."""
    phi: np.ndarray          # virtual value phi_job(v, c)                [$]
    cv: np.ndarray           # customer value q * v                       [$]
    profit: np.ndarray       # phi - C_elec                               [$]
    carbon_kg: np.ndarray    # E * CI                                     [kgCO2]
    carbon_usd: np.ndarray   # s_C * K                                    [$]
    R_nocarbon: np.ndarray   # l1 * cv + l2 * profit                      [$]
    R: np.ndarray            # l1 * cv + l2 * profit - l3 * K * s_C       [$]


def job_terms(q, v, c, C_elec, K, model, s_C: float, lam) -> JobTerms:
    """
    Reward terms of every job. q priority, v bid [$], c reference cost (cost_base) [$],
    C_elec [$], K [kgCO2], model the phase's MyersonModel, s_C [$/kgCO2], lam the
    ObjectiveCfg (lambda1..3). Operation order = the online loop's scalar formula
    (l3 * K) * s_C, so R is bit-identical to it element by element.
    """
    l1, l2, l3 = lam.lambda1, lam.lambda2, lam.lambda3
    q, v, C_elec, K = (np.asarray(x, float) for x in (q, v, C_elec, K))
    phi = np.asarray(model.phi_job(v, c), float)
    cv = q * v
    profit = phi - C_elec
    R_nocarbon = l1 * cv + l2 * profit
    R = l1 * cv + l2 * profit - l3 * K * s_C
    return JobTerms(phi=phi, cv=cv, profit=profit, carbon_kg=K, carbon_usd=s_C * K,
                    R_nocarbon=R_nocarbon, R=R)


def job_reward(stream, state, lam) -> JobTerms:
    """job_terms for every job of a StreamData with the bid model and s_C of `state`."""
    return job_terms(stream.priority, stream.v, stream.cost_base, stream.C_elec, stream.C_carbon,
                     state.myerson, state.carbon_price, lam)


# --------------------------------------------------------------------------- #
# Type-level quantities
# --------------------------------------------------------------------------- #
def type_mean(tid: np.ndarray, N: np.ndarray, x) -> np.ndarray:
    """Per-type sample mean of x over the jobs with type ids tid (N = counts per type);
    0 for types without jobs."""
    s = np.bincount(tid, weights=np.asarray(x, float), minlength=len(N))
    return np.divide(s, N, out=np.zeros(len(N)), where=N > 0)


def weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    """Weighted quantile: atoms sorted by x, each placed at the midpoint of its
    cumulative weight, (cumsum(w) - w/2) / sum(w), then linear interpolation."""
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    order = np.argsort(x, kind="stable")
    x, w = x[order], w[order]
    cw = (np.cumsum(w) - 0.5 * w) / w.sum()
    return float(np.interp(q, cw, x))


def carbon_price(cv: np.ndarray, profit: np.ndarray, carbon: np.ndarray,
                 p: np.ndarray, q: float) -> dict:
    """
    quantile_linear carbon price, over populated types (p > 0) weighted by p:
      D_q = (Q_q(CV) + Q_q(|Pi|)) / 2,  K_q = Q_q(K),  s_C = D_q / K_q  [$/kgCO2],
    so the linear map through the origin K -> s_C * K sends [0, K_q] onto [0, D_q].
    """
    m = np.asarray(p) > 0
    w = np.asarray(p, float)[m]
    Q_cv     = weighted_quantile(np.asarray(cv)[m], w, q)
    Q_abs_pi = weighted_quantile(np.abs(np.asarray(profit)[m]), w, q)
    K_q      = weighted_quantile(np.asarray(carbon)[m], w, q)
    D_q = 0.5 * (Q_cv + Q_abs_pi)
    if K_q <= 0 or D_q <= 0:
        warnings.warn(
            f"quantile_linear: degenerate scale (D_q={D_q:.4g}, K_q={K_q:.4g}); "
            f"setting s_C = 0 (no carbon term).",
            RuntimeWarning,
        )
        s_C = 0.0
    else:
        s_C = D_q / K_q
    return dict(q=float(q), Q_cv=Q_cv, Q_abs_pi=Q_abs_pi, D_q=D_q, K_q=K_q, s_C=float(s_C))


def type_reward(q_type: np.ndarray, v_bar: np.ndarray, phi_bar: np.ndarray,
                C_elec_bar: np.ndarray, K_bar: np.ndarray, p: np.ndarray, cfg,
                s_C: Optional[float] = None):
    """
    Type reward from the per-type means. s_C = None computes the quantile_linear
    carbon price on these types (Phase 0); otherwise s_C is the frozen value (ReInit
    and the hindsight benchmark pass Phase 0's). Returns (r_tilde, r_nocarbon, norm_info).
    The scalarization is cfg.scalarization.method: linear online, augmented Chebyshev
    only for the offline Pareto analysis (dlent_rt.pareto).
    """
    lam = cfg.objective
    cv = q_type * v_bar
    profit = phi_bar - C_elec_bar
    if s_C is None:
        norm_info = carbon_price(cv, profit, K_bar, p, cfg.normalization.quantile)
        s_C = norm_info["s_C"]
    else:
        s_C = float(s_C)
        norm_info = dict(q=cfg.normalization.quantile, s_C=s_C, frozen=True)
    norm_info = dict(strategy="quantile_linear", **norm_info)
    carbon_usd = K_bar * s_C
    f = np.column_stack([lam.lambda1 * cv, lam.lambda2 * profit, -lam.lambda3 * carbon_usd])
    r_nocarbon = lam.lambda1 * cv + lam.lambda2 * profit
    r_tilde = np.asarray(scalarize(f, cfg.scalarization.method, np.ones(3), cfg.scalarization.rho))
    return r_tilde, r_nocarbon, norm_info

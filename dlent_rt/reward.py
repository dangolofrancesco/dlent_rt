"""
The additive dollar reward (decision D1): ONE source of truth for its terms.

    customer value  V  = b d                            [$]   b the HOURLY bid, d the true duration (hours)
    profit          Pi = phi d - C_elec                 [$]   phi = MyersonModel.phi_job(b, c) [$ per hour]
    carbon          K  = E * CI                         [kgCO2]
    reward          R  = l1 * V + l2 * Pi - l3 * s_C * K   [$ per job]   (linear; all lambda = 1)

    social welfare  SW = sum over admitted jobs of (b d - C_elec - s_C * K)   [$]   a METRIC only, never a decision

with C_elec = E * price [$] and the carbon price s_C [$/kgCO2] computed once in Phase 0 and frozen. Two views:

  * job_terms(...)   one value per job, with the job's TRUE duration (the realised results only: the online loop's
                     realised reward, the benchmarks, the notebooks);
  * type_reward(...) one value per type from the type terms (Phase 0, ReInit, the hindsight benchmark; week 33): the
                     bin sets only the value, the column (priority, profile) sets the costs,
                         V_type = d_(q,h) L_k,  Pi_type = d_(q,h) phi_(q,h)(L_k) - C_elec_(q,h),  K_type = K_(q,h)
                     (phase0.type_values). No term depends on a job's true duration or on a per-bin duration.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np


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
    phi: np.ndarray          # virtual value phi_job(b, c) d              [$]
    cv: np.ndarray           # customer value V = b d                     [$]
    profit: np.ndarray       # phi - C_elec                               [$]
    carbon_kg: np.ndarray    # E * CI                                     [kgCO2]
    carbon_usd: np.ndarray   # s_C * K                                    [$]
    R_nocarbon: np.ndarray   # l1 * V + l2 * profit                       [$]
    R: np.ndarray            # l1 * V + l2 * profit - l3 * K * s_C        [$]
    sw: np.ndarray           # social welfare of the job: b d - C_elec - s_C * K   [$] (metric only)


def job_terms(b, c, d_hours, C_elec, K, model, s_C: float, lam) -> JobTerms:
    """
    Realised reward terms of every job. b hourly bid [$/h], c hourly cost_base [$/h], d_hours the TRUE duration
    [h], C_elec [$], K [kgCO2], model the MyersonModel, s_C [$/kgCO2], lam the ObjectiveCfg (lambda1..3).
    """
    l1, l2, l3 = lam.lambda1, lam.lambda2, lam.lambda3
    b, d, C_elec, K = (np.asarray(x, float) for x in (b, d_hours, C_elec, K))
    v = b * d
    phi = np.asarray(model.phi_job(b, c), float) * d
    profit = phi - C_elec
    R_nocarbon = l1 * v + l2 * profit
    R = l1 * v + l2 * profit - l3 * K * s_C
    return JobTerms(phi=phi, cv=v, profit=profit, carbon_kg=K, carbon_usd=s_C * K,
                    R_nocarbon=R_nocarbon, R=R, sw=v - C_elec - s_C * K)


def job_reward(stream, state, lam) -> JobTerms:
    """job_terms for every job of a StreamData (true durations) with the bid model and s_C of `state`."""
    return job_terms(stream.v, stream.cost_base, stream.duration_hours, stream.C_elec, stream.C_carbon,
                     state.myerson, state.carbon_price, lam)


def social_welfare(terms: JobTerms, admitted) -> float:
    """SW = sum over the admitted jobs of (b d - C_elec - s_C K) [$] (metric only)."""
    return float(terms.sw[np.asarray(admitted, bool)].sum())


# --------------------------------------------------------------------------- #
# Type-level quantities
# --------------------------------------------------------------------------- #
def type_mean(tid: np.ndarray, N: np.ndarray, x) -> np.ndarray:
    """Per-group sample mean of x over the jobs with group ids tid (N = counts per group); 0 for empty groups."""
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
    The carbon price over the types with p > 0, weighted by p:
      D_q = (Q_q(V) + Q_q(|Pi|)) / 2,  K_q = Q_q(K),  s_C = D_q / K_q  [$/kgCO2],
    so the linear map through the origin K -> s_C * K sends [0, K_q] onto [0, D_q].
    """
    m = np.asarray(p) > 0
    w = np.asarray(p, float)[m]
    Q_cv     = weighted_quantile(np.asarray(cv)[m], w, q)
    Q_abs_pi = weighted_quantile(np.abs(np.asarray(profit)[m]), w, q)
    K_q      = weighted_quantile(np.asarray(carbon)[m], w, q)
    D_q = 0.5 * (Q_cv + Q_abs_pi)
    if K_q <= 0 or D_q <= 0:
        warnings.warn(f"carbon price: degenerate scale (D_q={D_q:.4g}, K_q={K_q:.4g}); setting s_C = 0.",
                      RuntimeWarning)
        s_C = 0.0
    else:
        s_C = D_q / K_q
    return dict(q=float(q), Q_cv=Q_cv, Q_abs_pi=Q_abs_pi, D_q=D_q, K_q=K_q, s_C=float(s_C))


def type_reward(v_bar: np.ndarray, phi_bar: np.ndarray, C_elec_bar: np.ndarray, K_bar: np.ndarray,
                p: np.ndarray, cfg, s_C: Optional[float] = None):
    """
    Type reward from the type terms: v_bar = V_type, phi_bar = d phi(L_k) (so Pi = phi_bar - C_elec_bar), K_bar
    (phase0.type_values). s_C = None computes the carbon price on these types (Phase 0);
    otherwise s_C is the frozen value (ReInit and the benchmarks pass Phase 0's). Returns (r_tilde, r_nocarbon,
    norm_info).
    """
    lam = cfg.objective
    profit = phi_bar - C_elec_bar
    if s_C is None:
        norm_info = carbon_price(v_bar, profit, K_bar, p, cfg.normalization.quantile)
        s_C = norm_info["s_C"]
    else:
        s_C = float(s_C)
        norm_info = dict(q=cfg.normalization.quantile, s_C=s_C, frozen=True)
    r_nocarbon = lam.lambda1 * v_bar + lam.lambda2 * profit
    r_tilde = r_nocarbon - lam.lambda3 * K_bar * s_C
    return r_tilde, r_nocarbon, norm_info

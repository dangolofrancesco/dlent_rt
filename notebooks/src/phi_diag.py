"""
Helpers for notebook 04 (virtual-value diagnostic). READ-ONLY: every φ is computed
with the mechanism's own estimator, dlent_rt.bid.fit_myerson (lognormal MLE on ln v)
+ MyersonModel.phi_job (pooled fit); only the CONDITIONING (which bids are pooled into one F) changes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import lognorm

from dlent_rt.bid import fit_myerson


# --------------------------------------------------------------------------- #
# phi under different conditionings (same estimator)
# --------------------------------------------------------------------------- #
def phi_pooled(v):
    """(a) one F for every bid."""
    m = fit_myerson(v)
    return np.asarray(m.phi_job(v), float), m


def phi_scaled(v, s):
    """(c)/(d) v = s * u with u ~ one pooled F (u independent of s).
    F_v|s(x) = F_u(x/s), f_v|s(x) = f_u(x/s)/s  =>  phi_v(v) = s * phi_u(v/s)."""
    u = np.asarray(v, float) / np.asarray(s, float)
    m = fit_myerson(u)
    return np.asarray(s, float) * np.asarray(m.phi_job(u), float), m


def phi_by_group(v, groups, fallback, min_n=2):
    """(b) one F per group; groups with < min_n bids keep `fallback` (fit_myerson needs >= 2)."""
    v = np.asarray(v, float)
    phi = np.asarray(fallback, float).copy()
    models, n_fb = {}, 0
    for g in np.unique(groups):
        ix = groups == g
        if ix.sum() < min_n:
            n_fb += int(ix.sum())
            continue
        m = fit_myerson(v[ix])
        phi[ix] = m.phi_job(v[ix])
        models[g] = m
    return phi, models, n_fb


def reserve_quantile(m) -> float:
    """F_hat(r*) with r* = phi^{-1}(0) (MyersonModel.phi_inv)."""
    r = float(m.phi_inv(0.0))
    return float(lognorm.cdf(r, s=m.sigma, scale=np.exp(m.mu)))


# --------------------------------------------------------------------------- #
# buckets and regression
# --------------------------------------------------------------------------- #
def size_buckets(priority, hw, V, min_size=200, max_split=10):
    """
    Buckets = priority x hw profile x quantile bins of the volume proxy V, every bucket >= min_size:
      1. cells (priority, hw) with >= min_size jobs are kept; smaller cells of the same priority are
         merged into one (priority, 'hw-other') cell; if that is still < min_size it joins a global leftover;
      2. each cell is split into k = min(max_split, n // min_size) equal-count quantile bins of V.
    Returns (labels, table).
    """
    n = len(V)
    cell = np.empty(n, dtype=object)
    for q in np.unique(priority):
        iq = priority == q
        for h in np.unique(hw[iq]):
            ix = iq & (hw == h)
            cell[ix] = f"q{int(q)}|hw{int(h)}" if ix.sum() >= min_size else f"q{int(q)}|hw-other"
        io = iq & (cell == f"q{int(q)}|hw-other")
        if 0 < io.sum() < min_size:
            cell[io] = "leftover"
    lab = np.empty(n, dtype=object)
    rows = []
    for c in pd.unique(cell):
        ix = np.flatnonzero(cell == c)
        k = max(1, min(max_split, len(ix) // min_size))
        parts = np.array_split(ix[np.argsort(V[ix], kind="stable")], k)
        for b, p in enumerate(parts):
            lab[p] = f"{c}|V{b + 1}/{k}"
            rows.append(dict(bucket=f"{c}|V{b + 1}/{k}", cell=c, n=len(p),
                             V_lo=float(V[p].min()), V_hi=float(V[p].max())))
    return lab, pd.DataFrame(rows)


def ols(X, y):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    fit = X @ coef
    res = y - fit
    return dict(coef=coef, fitted=fit, resid=res, R2=1 - res.var() / y.var(),
                resid_sd=float(res.std(ddof=X.shape[1] if X.shape[0] > X.shape[1] else 0)))


def dummies(x, drop_first=True):
    cats = np.unique(x)
    cats = cats[1:] if drop_first else cats
    return np.column_stack([(x == c).astype(float) for c in cats]) if len(cats) else np.zeros((len(x), 0))


# --------------------------------------------------------------------------- #
# summaries
# --------------------------------------------------------------------------- #
def nonmono_within(phi, key, cells):
    """Share of adjacent pairs (sorted by `key` inside each cell) where phi decreases."""
    bad = tot = 0
    for c in pd.unique(cells):
        ix = np.flatnonzero(cells == c)
        if ix.size < 2:
            continue
        p = phi[ix[np.argsort(key[ix], kind="stable")]]
        bad += int((np.diff(p) < -1e-12).sum())
        tot += ix.size - 1
    return bad / max(tot, 1)


def nonmono_bins(phi, k_bin, min_count=5):
    """Mean phi per bid bin k (bins with >= min_count jobs); share of adjacent kept bins where it decreases."""
    s = pd.Series(phi).groupby(k_bin).agg(["mean", "size"])
    s = s[s["size"] >= min_count].sort_index()
    d = np.diff(s["mean"].to_numpy())
    return float((d < 0).mean()) if d.size else np.nan


def summary_row(name, phi, phi_gen, k_bin, key, cells, rq, phi_within=None):
    """phi_within: the function checked for monotonicity in `key` inside each cell (default phi);
    for the scaled variants it is phi_u = phi_v / s, the phi of the pooled variable u."""
    neg, gneg = phi < 0, phi_gen < 0
    phi_within = phi if phi_within is None else phi_within
    q10, q50, q90 = np.quantile(phi, [0.1, 0.5, 0.9])
    return {
        "conditioning": name, "share phi<0 %": 100 * neg.mean(),
        "phi p10": q10, "phi p50": q50, "phi p90": q90,
        "F_hat(r*) (model)": rq,
        "non-monotone adj. pairs within cell %": 100 * nonmono_within(phi_within, key, cells),
        "non-monotone adj. v-bins (bin means) %": 100 * nonmono_bins(phi, k_bin),
        "E[phi+] $": float(np.maximum(phi, 0).mean()),
        "sign agree w/ phi_gen %": 100 * (neg == gneg).mean(),
        "(+,+) %": 100 * (~neg & ~gneg).mean(), "(+,-gen) %": 100 * (~neg & gneg).mean(),
        "(-,+gen) %": 100 * (neg & ~gneg).mean(), "(-,-) %": 100 * (neg & gneg).mean(),
    }

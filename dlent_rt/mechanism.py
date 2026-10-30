"""
Admission rule and payment of the online model (week 33): a truthful threshold mechanism (Myerson: monotone
admission + critical-value payment) on HOURLY bids.

    margin_k = r~_(k,q,h) - tau^T v_(q,h)   the oracle's score of bin k in the arrival's column (q, h)
               It is increasing in k: the bin sets only the value d (L_k + phi(L_k)), increasing in L_k; the column
               sets the costs and the volume (phase0.type_values). So the admitted bins are an upper set.
    k*       = the lowest bin with margin > 0; 0 = no bin is admitted. threshold_kstar raises if the admitted bins
               are not an upper set.
    r_j      = P_j * price(t_j)             the reserve per hour: the job's power (kW, from its request, PUE
               included) times the electricity price at arrival [$ per hour]. It does not depend on the bid.
    admit    <=> k* > 0 and bin(b_j) >= k* and b_j >= r_j   (then thinning, theory only, and the physical gate)
    p_j      = max(L_{k*}, r_j)             [$ per hour], L_{k*} = the lower edge of bin k* (0 for bin 1)
    payment  = p_j * the hours the job runs  [$]

Since bin(b) >= k* <=> b >= L_{k*}, a job is admitted (before thinning and the gate) iff b >= p_j, and p_j does not
depend on b: p_j is the lowest hourly bid that would still be admitted (truthful, IR: 0 < p_j <= b_j, monotone).
"""
from __future__ import annotations

import numpy as np


def threshold_kstar(margin: np.ndarray) -> np.ndarray:
    """
    margin: (K, ...) the oracle's score per bid bin (axis 0, bin 1 first). Returns k* (1-based, int) per column: the
    lowest bin with margin > 0; 0 if no bin. Raises ValueError if the admitted bins of a column are not an upper set
    (a rejected bin above an admitted one): it must hold by construction (margin increasing in k).
    """
    ok = np.asarray(margin) > 0
    if np.any(ok[:-1] & ~ok[1:]):
        bad = np.argwhere(ok[:-1] & ~ok[1:])[0]
        raise ValueError(f"admitted bins are not an upper set: bin {bad[0] + 1} admitted, bin {bad[0] + 2} rejected "
                         f"at column {tuple(bad[1:])}")
    return np.where(ok.any(axis=0), np.argmax(ok, axis=0) + 1, 0).astype(np.int64)


def reserve_price(power_kw, price) -> np.ndarray:
    """r = P [kW] * price [$/kWh]  ->  $ per hour (the electricity cost of one hour of the job)."""
    return np.asarray(power_kw, float) * np.asarray(price, float)


def admit_and_pay(b, grid, kstar: int, reserve: float) -> tuple[bool, bool, float, float]:
    """
    The price part of the admission of ONE job with hourly bid b: (bin ok, reserve ok, L_{k*}, hourly payment).
    bin ok = k* > 0 and bin(b) >= k*; reserve ok = b >= r; payment = max(L_{k*}, r) per hour (NaN if k* = 0).
    Thinning and the gate come after.
    """
    k = int(grid.assign(b)[0])
    L = float(grid.lower_edges[kstar - 1]) if kstar > 0 else np.nan
    bin_ok = kstar > 0 and k >= kstar
    res_ok = float(b) >= float(reserve)
    pay = max(L, float(reserve)) if kstar > 0 else np.nan
    return bin_ok, res_ok, L, pay

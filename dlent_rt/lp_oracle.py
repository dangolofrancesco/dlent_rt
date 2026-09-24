"""
LP-RS solver and scalarization.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import highspy


def scalarize_linear(f: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Linear scalarization: weighted sum. f is (n_types, 3), weights is (3,).
    Returns (n_types,).
    """
    return f @ weights


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


def scalarize(
    f: np.ndarray, method: str, weights: np.ndarray, rho: float,
) -> np.ndarray:
    if method == "linear":
        return scalarize_linear(f, weights)
    if method == "chebyshev":
        return scalarize_chebyshev(f, weights, rho)
    raise ValueError(f"unknown scalarization method: {method}")



# LP-RS
@dataclass
class LPResult:
    optimum: float               # objective value  hat_lambda_*
    x: np.ndarray                # primal admission fractions, (n_support,)
    tau: np.ndarray              # dual shadow prices on capacity, (|I|,)
    support_idx: np.ndarray      # indices (into the type list) that were in support
    status: str


def solve_lp_rs(
    p: np.ndarray,              # type probabilities (n_types,)
    r: np.ndarray,              # scalarized reward per type (n_types,)
    v: np.ndarray,              # expected resource volume (n_types, |I|)
    c: np.ndarray,              # capacity (|I|,)
    presolve: bool = True,
) -> LPResult:
    """
    Solve the fluid LP-RS:

        max_x   sum_j p_j r_j x_j
        s.t.    sum_j p_j v_ij x_j <= c_i   for all i
                0 <= x_j <= 1

    Returns optimum, primal x*, and dual tau* (>= 0) on the capacity rows.
    Only types with p_j > 0 enter the LP (support).
    """
    n_types = len(p)
    n_res = len(c)
    support = np.where(p > 0.0)[0]
    if support.size == 0:
        return LPResult(0.0, np.zeros(0), np.zeros(n_res), support, "empty_support")

    ps = p[support]
    rs = r[support]
    vs = v[support]                                  # (n_support, |I|)
    ns = support.size

    obj = -(ps * rs)                                 # (ns,)

    # constraint matrix rows: sum_j (ps_j v_ij) x_j <= c_i
    # build in CSC/row form for highspy
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("presolve", "on" if presolve else "off")

    inf = highspy.kHighsInf

    # variables: 0 <= x <= 1 with cost obj
    lp = highspy.HighsLp()
    lp.num_col_ = ns
    lp.num_row_ = n_res
    lp.sense_ = highspy.ObjSense.kMinimize
    lp.col_cost_ = obj.tolist()
    lp.col_lower_ = [0.0] * ns
    lp.col_upper_ = [1.0] * ns
    lp.row_lower_ = [-inf] * n_res
    lp.row_upper_ = c.tolist()

    # constraint matrix in column-wise (CSC) format
    # a_ij = ps_j * v_ij  (row i, col j)
    a = (ps[:, None] * vs).T                         # (n_res, ns)
    # CSC: iterate columns
    starts = [0]
    indices: list[int] = []
    values: list[float] = []
    for jcol in range(ns):
        col = a[:, jcol]
        nz = np.nonzero(col)[0]
        indices.extend(nz.tolist())
        values.extend(col[nz].tolist())
        starts.append(len(indices))

    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = starts
    lp.a_matrix_.index_ = indices
    lp.a_matrix_.value_ = values
    lp.a_matrix_.num_col_ = ns
    lp.a_matrix_.num_row_ = n_res

    h.passModel(lp)
    h.run()

    status = h.getModelStatus()
    status_str = h.modelStatusToString(status)
    sol = h.getSolution()

    x = np.array(sol.col_value, dtype=float)
    # dual on rows: highs returns row_dual; sign convention -> take absolute value
    # of the shadow price (capacity constraints are <=, duals are <= 0 in min
    # form; the economic shadow price is non-negative).
    row_dual = np.array(sol.row_dual, dtype=float)
    tau = np.abs(row_dual)

    optimum = float((ps * rs * x).sum())

    return LPResult(
        optimum=optimum,
        x=x,
        tau=tau,
        support_idx=support,
        status=status_str,
    )

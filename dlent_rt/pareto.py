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


# --------------------------------------------------------------------------- #
# Pareto frontier over the objective weights lambda (week 30, offline only)
# --------------------------------------------------------------------------- #
# Objectives, all from dlent_rt.reward (no reward is re-implemented):
#     CV = q * v               customer value     (maximised)
#     PR = phi_job - C_elec    profit             (maximised)
#     CO = s_C * K             carbon cost [$]    (minimised; s_C frozen in Phase 0)
#     R(lambda) = l1 * CV + l2 * PR - l3 * CO
# Every frontier point is the objective vector (sum CV, sum PR, sum CO) of one solution; as a vector to
# MAXIMISE it is F = (CV, PR, -CO).
from dataclasses import dataclass as _dataclass
from itertools import product as _product

from scipy.optimize import linprog as _linprog

OBJECTIVES = ("CV", "PR", "CO")
CORNERS = ((3.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 3.0))
CENTRE = (1.0, 1.0, 1.0)


@_dataclass(frozen=True)
class Components:
    """Per-item objective components (items = jobs or types)."""
    CV: np.ndarray
    PR: np.ndarray
    CO: np.ndarray

    def combine(self, lam) -> np.ndarray:
        """R(lambda) = l1 * CV + l2 * PR - l3 * CO."""
        l1, l2, l3 = lam
        return l1 * self.CV + l2 * self.PR - l3 * self.CO

    def stack(self) -> np.ndarray:
        """(n, 3) array of the objectives to MAXIMISE: (CV, PR, -CO)."""
        return np.column_stack([self.CV, self.PR, -self.CO])


def job_components(stream, state) -> Components:
    """CV, PR, CO of every stream job under `state`'s bid model and s_C (reward.job_reward; lambda-free)."""
    from .config import ObjectiveCfg
    from .reward import job_reward
    t = job_reward(stream, state, ObjectiveCfg(1.0, 1.0, 1.0))
    return Components(CV=np.asarray(t.cv), PR=np.asarray(t.profit), CO=np.asarray(t.carbon_usd))


def type_components(state) -> Components:
    """CV, PR, CO of every type from the state's per-type means, as reward.type_reward forms them."""
    from .phase0 import type_arrays
    _, prio, _ = type_arrays(state.type_space, state.grid)
    return Components(CV=prio * state.v_bar, PR=state.phi_bar - state.C_elec_bar,
                      CO=state.C_carbon_bar * state.carbon_price)


def weight_grid(step: float = 0.25, scale: float = 3.0, extra=(CENTRE,)) -> list:
    """lambda = scale * w for w on the simplex with the given step (step 1/4: 15 points, corners included),
    plus `extra` points (default: the centre (1, 1, 1), not on the 1/4 grid) and the corners if absent."""
    m = int(round(1 / step))
    pts = [tuple(scale * np.array([i, j, m - i - j]) / m) for i, j in _product(range(m + 1), repeat=2) if i + j <= m]
    for e in (*CORNERS, *extra):
        if not any(np.allclose(e, p) for p in pts):
            pts.append(tuple(float(x) for x in e))
    return [tuple(float(x) for x in p) for p in pts]


def subset7(scale: float = 3.0) -> list:
    """The 3 corners, the centre (1, 1, 1) and the 3 edge midpoints (scale/2, scale/2, 0) etc."""
    h = scale / 2
    return [*CORNERS, CENTRE, (h, h, 0.0), (h, 0.0, h), (0.0, h, h)]


def nondominated(F: np.ndarray) -> np.ndarray:
    """Boolean mask of the non-dominated rows of F (all columns maximised). Duplicates are all kept."""
    F = np.asarray(F, float)
    ge = (F[:, None, :] >= F[None, :, :]).all(axis=2)     # ge[a, b]: a >= b everywhere
    gt = (F[:, None, :] > F[None, :, :]).any(axis=2)
    dominated = (ge & gt).any(axis=0)                       # some a dominates b
    return ~dominated


def hypervolume_3d(F: np.ndarray, ref) -> float:
    """
    Exact hypervolume (all objectives maximised) of the region dominated by the rows of F and dominating
    the reference point `ref`; rows not strictly better than ref in every objective are ignored.
    Slicing along the third objective + 2-D staircase areas.
    """
    F = np.asarray(F, float)
    ref = np.asarray(ref, float)
    F = F[(F > ref).all(axis=1)]
    if F.size == 0:
        return 0.0
    levels = np.unique(F[:, 2])[::-1]                       # descending
    hv, prev = 0.0, None
    bounds = np.r_[levels, ref[2]]
    for k, z in enumerate(levels):
        P = F[F[:, 2] >= z][:, :2]
        P = P[np.argsort(-P[:, 0])]
        area, ymax = 0.0, ref[1]
        for x, y in P:                                      # staircase, x descending
            if y > ymax:
                area += (x - ref[0]) * (y - ymax)
                ymax = y
        hv += area * (z - bounds[k + 1])
    return float(hv)


def fluid_lp(p, comp: Components, v, c, lam, presolve: bool = True):
    """Weighted-sum fluid LP-RS (lp_oracle.solve_lp_rs) with r = R(lambda) per type. Returns the
    admission fractions x (all types) and the per-tick objective totals sum_j p_j f_j x_j."""
    from .lp_oracle import solve_lp_rs
    lp = solve_lp_rs(p=np.asarray(p, float), r=comp.combine(lam), v=v, c=c, presolve=presolve)
    x = np.zeros(len(p))
    x[lp.support_idx] = lp.x
    return x, fluid_totals(p, comp, x), lp


def fluid_totals(p, comp: Components, x) -> dict:
    w = np.asarray(p, float) * x
    return {"CV": float(w @ comp.CV), "PR": float(w @ comp.PR), "CO": float(w @ comp.CO)}


def ideal_nadir_fluid(p, comp: Components, v, c):
    """Single-objective fluid optima (max CV, max PR, min CO) and the payoff-table nadir.
    Returns (ideal, nadir) as vectors to MAXIMISE, (CV, PR, -CO)."""
    sols = []
    for lam in CORNERS:
        _, tot, _ = fluid_lp(p, comp, v, c, lam)
        sols.append([tot["CV"], tot["PR"], -tot["CO"]])
    pay = np.array(sols)                                    # row k: solution of objective k
    return pay.diagonal().copy(), pay.min(axis=0)


def _fluid_parts(p, comp: Components, v):
    p = np.asarray(p, float)
    sup = np.flatnonzero(p > 0)
    ps, vs = p[sup], np.asarray(v, float)[sup]
    return p, sup, ps, vs, comp.stack()[sup] * ps[:, None]       # Fk[:, k] = p_j f_kj


def chebyshev_fluid(p, comp: Components, v, c, w, ideal, nadir, rho: float = 1e-3):
    """
    STANDARD augmented Chebyshev LP on the fluid problem (larger weight = more important):
        max  t + rho * sum_k g_k(x)
        s.t. t <= g_k(x) / w_k   for every k with w_k > 0,
             g_k(x) = (f_k(x) - nadir_k) / (ideal_k - nadir_k),
             sum_j p_j v_ij x_j <= c_i,  0 <= x <= 1,
    f = (CV, PR, -CO) per tick; i.e. the point where the ray from the nadir in direction w meets the front.
    Objectives with w_k = 0 carry no t-constraint but stay in the augmentation term, so ties are broken towards
    them and the solution is Pareto-optimal (not only weakly); objectives with ideal_k = nadir_k are dropped.
    The augmentation trades objectives at a rate of at most rho / (1 + rho). (NOT equivalent to
    min max_k w_k (ideal_k - f_k) / range_k, which follows the ray from the ideal in direction 1 / w.)
    Returns x, the per-tick totals, t and the solver message.
    """
    p, sup, ps, vs, Fk = _fluid_parts(p, comp, v)
    n = sup.size
    rng = np.asarray(ideal, float) - np.asarray(nadir, float)
    use = rng > 0
    A, b = [], []
    for k in range(3):
        if w[k] <= 0 or not use[k]:
            continue
        # w_k t - Fk^T x / rng_k <= -nadir_k / rng_k
        A.append(np.r_[-Fk[:, k] / rng[k], w[k]])
        b.append(-nadir[k] / rng[k])
    for i in range(vs.shape[1]):
        A.append(np.r_[ps * vs[:, i], 0.0])
        b.append(float(c[i]))
    aug = sum(Fk[:, k] / rng[k] for k in range(3) if use[k])
    res = _linprog(np.r_[-rho * aug, -1.0], A_ub=np.array(A), b_ub=np.array(b),
                   bounds=[(0, 1)] * n + [(None, None)], method="highs")
    x = np.zeros(len(p))
    x[sup] = res.x[:n]
    return x, fluid_totals(p, comp, x), float(res.x[n]), res.message


def fluid_dominance_slack(p, comp: Components, v, c, x, ideal, nadir) -> float:
    """
    max sum_k (f_k(y) - f_k(x)) / range_k over feasible fluid y with f_k(y) >= f_k(x) for every k.
    0 (up to solver tolerance) iff x is Pareto-optimal; > 0 means some y dominates x.
    If x sits on the capacity boundary within the solver's tolerance, the exact LP can be numerically infeasible:
    it is then re-solved once with the lower bounds f_k(y) >= f_k(x) relaxed by 1e-7 * range_k; NaN if that fails too.
    """
    p, sup, ps, vs, Fk = _fluid_parts(p, comp, v)
    fx = Fk.T @ np.asarray(x, float)[sup]
    rng = np.where(np.asarray(ideal) - np.asarray(nadir) > 0, np.asarray(ideal) - np.asarray(nadir), 1.0)
    A = np.vstack([-Fk.T, (ps[:, None] * vs).T])
    obj = -(Fk / rng).sum(axis=1)
    for relax in (0.0, 1e-7):
        b = np.r_[-fx + relax * rng, np.asarray(c, float)]
        res = _linprog(obj, A_ub=A, b_ub=b, bounds=(0, 1), method="highs")
        if res.status == 0:
            return float(-res.fun - (fx / rng).sum())
    return float("nan")


def chebyshev_fluid_legacy(p, comp: Components, v, c, w, ideal, nadir):
    """
    LEGACY (inverted weight orientation, kept for reference only): a larger w_k makes objective k's
    constraint LOOSER, i.e. objective k matters LESS; not augmented, so the optimum is degenerate (t is set by
    the smallest positive weight) and only weakly Pareto.
        max t  s.t.  t <= w_k (f_k(x) - nadir_k) / (ideal_k - nadir_k)   for every k with w_k > 0,
                     sum_j p_j v_ij x_j <= c_i,  0 <= x <= 1,
    f = (CV, PR, -CO) per tick. Objectives with w_k = 0 (or ideal_k = nadir_k) carry no constraint.
    Returns x, the per-tick totals and t.
    """
    p = np.asarray(p, float)
    sup = np.flatnonzero(p > 0)
    ps, vs = p[sup], np.asarray(v, float)[sup]
    Fk = comp.stack()[sup] * ps[:, None]                   # (n, 3): p_j f_kj
    n = sup.size
    A, b = [], []
    for k in range(3):
        rng = ideal[k] - nadir[k]
        if w[k] <= 0 or rng <= 0:
            continue
        # t - w_k (Fk^T x - nadir_k) / rng <= 0
        A.append(np.r_[-w[k] * Fk[:, k] / rng, 1.0])
        b.append(-w[k] * nadir[k] / rng)
    for i in range(vs.shape[1]):
        A.append(np.r_[ps * vs[:, i], 0.0])
        b.append(float(c[i]))
    res = _linprog(np.r_[np.zeros(n), -1.0], A_ub=np.array(A), b_ub=np.array(b),
                   bounds=[(0, 1)] * n + [(None, None)], method="highs")
    x = np.zeros(len(p))
    x[sup] = res.x[:n]
    return x, fluid_totals(p, comp, x), float(res.x[n]), res.message


def chebyshev_discrete(F: np.ndarray, w, ideal, nadir, rho: float = 1e-3) -> int:
    """Index of the row of F (maximised) that maximises the STANDARD augmented Chebyshev value
    min_{k: w_k > 0} g_k / w_k + rho * sum_k g_k, g_k = (F_k - nadir_k) / (ideal_k - nadir_k), over a discrete
    candidate set (e.g. a non-convex front). Larger weight = more important."""
    F = np.asarray(F, float)
    w, ideal, nadir = (np.asarray(a, float) for a in (w, ideal, nadir))
    use = ideal > nadir
    g = (F[:, use] - nadir[use]) / (ideal[use] - nadir[use])
    wk = w[use]
    s = (g[:, wk > 0] / wk[wk > 0]).min(axis=1) + rho * g.sum(axis=1)
    return int(np.argmax(s))


def perturbations(factors=(2.0, 0.5), centre=CENTRE) -> list:
    """The centre and, for each k, lambda_k scaled by every factor with the other weights at the centre:
    [(label, k or None, factor or None, lambda), ...] (Kash: "does every lambda matter?")."""
    out = [("centre", None, None, tuple(centre))]
    for k in range(3):
        for f in factors:
            lam = list(centre)
            lam[k] = centre[k] * f
            out.append((f"lambda{k + 1} x{f:g}", k, f, tuple(float(x) for x in lam)))
    return out

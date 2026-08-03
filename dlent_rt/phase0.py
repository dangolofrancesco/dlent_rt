"""
Phase 0 Bootstrap (Algorithm 1, paper-faithful).

Simulates the online warm-up offline on the historical batch H. Fits the
geometric grid, computes empirical priors and per-type statistics, solves the
initial LP-RS, and returns everything the main loop needs to start phase m=1
with no exploration.

The output `PhaseState` is the object both the theory and practical models read;
it is also re-emitted (recomputed) at every phase transition.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import Config
from .data import Dataset, JobArrays
from .grid import MyersonModel, GeometricGrid, fit_myerson, build_geometric_grid
from .lp_oracle import scalarize, solve_lp_rs, LPResult


# --------------------------------------------------------------------------- #
# Type-space bookkeeping
# --------------------------------------------------------------------------- #
@dataclass
class TypeSpace:
    """
    The finite type space J = K x Q x A, plus the mapping from a (k,q,A) triple
    to a flat integer type-id. Hardware profiles A are the unique rows of the
    resource matrix (fixed for all phases, decision from earlier: cloud VM shapes
    are a closed catalogue).
    """
    k_star: int
    priorities: np.ndarray                 # sorted unique priority classes
    hw_profiles: np.ndarray                # (n_hw, |I|) unique hardware rows
    resources: list[str]
    # index maps
    _hw_lookup: dict = field(default_factory=dict)   # tuple(A) -> hw index

    @property
    def n_hw(self) -> int:
        return len(self.hw_profiles)

    @property
    def n_types(self) -> int:
        return self.k_star * len(self.priorities) * self.n_hw

    def hw_index(self, a_row: np.ndarray) -> int:
        return self._hw_lookup[tuple(np.round(a_row, 8))]

    def type_id_from_hw_idx(self, k: int, q: float, hw_idx: int) -> int:
        """Flat id using a precomputed hardware-profile index (fast path)."""
        ki = k - 1
        qi = int(np.searchsorted(self.priorities, q))
        return (ki * len(self.priorities) + qi) * self.n_hw + hw_idx

    def type_id(self, k: int, q: float, a_row: np.ndarray) -> int:
        """Flat id in [0, n_types). k in 1..K*, q in priorities, A a hw row."""
        ki = k - 1
        qi = int(np.searchsorted(self.priorities, q))
        ai = self.hw_index(a_row)
        return (ki * len(self.priorities) + qi) * self.n_hw + ai

    def decode(self, type_id: int) -> tuple[int, float, np.ndarray]:
        ai = type_id % self.n_hw
        rest = type_id // self.n_hw
        qi = rest % len(self.priorities)
        ki = rest // len(self.priorities)
        return ki + 1, float(self.priorities[qi]), self.hw_profiles[ai]


def build_type_space(H: JobArrays, ds, cfg: Config) -> TypeSpace:
    priorities = np.array(sorted(cfg.grid.priority_classes), dtype=float)
    # canonical hardware catalogue (fixed, fitted in the data layer)
    profiles = ds.hw_catalogue.centroids
    lookup = {tuple(np.round(row, 8)): i for i, row in enumerate(profiles)}
    return TypeSpace(
        k_star=cfg.grid.k_star,
        priorities=priorities,
        hw_profiles=profiles,
        resources=H.resources,
        _hw_lookup=lookup,
    )


# --------------------------------------------------------------------------- #
# Phase state: the full output of Phase 0 / EndPhaseAndReinitialize
# --------------------------------------------------------------------------- #
@dataclass
class PhaseState:
    """Everything the online loop needs to run one phase."""
    myerson: MyersonModel
    grid: GeometricGrid
    type_space: TypeSpace

    # per-type arrays, indexed by flat type-id (0..n_types-1)
    p: np.ndarray                 # empirical type probability
    d_bar: np.ndarray             # expected duration (steps) per type
    w: np.ndarray                 # power footprint per type
    r_tilde: np.ndarray           # frozen scalarized reward per type
    r_rev: np.ndarray             # revenue-only base per type (for time-aware oracle)
    C_elec_bar: np.ndarray        # phase-averaged elec cost per type
    C_carbon_bar: np.ndarray      # phase-averaged carbon cost per type
    v: np.ndarray                 # (n_types, |I|) expected resource volume
    L: np.ndarray                 # bid-space bin floor per type
    phi_tilde: np.ndarray         # discrete virtual value phi(L) per type

    # LP-RS output
    tau: np.ndarray               # shadow prices (|I|,)
    lam_star: float               # LP optimum benchmark

    # system constants
    gamma: float
    xi: float
    eps_D: float

    n_types: int
    norm_ref: Optional[tuple] = None    # (z_sat, z_prof, z_carb) if normalized


# --------------------------------------------------------------------------- #
# System constants
# --------------------------------------------------------------------------- #
def _system_constants(
    H: JobArrays, ds: Dataset, cfg: Config, r_max: float, v_min: float
) -> tuple[float, float, float]:
    """gamma, xi, eps_D from the calibration block."""
    a_max = ds.a_max
    c_min = ds.capacity.min()
    d_max = ds.d_max_steps
    c_max = ds.capacity.max()
    r_max_val = max(r_max, 1e-12)
    v_min_val = max(v_min, 1e-12)

    gamma = max(r_max_val, H.v_rate.max(), d_max, c_max, r_max_val / v_min_val)

    xi = float(np.max(a_max / ds.capacity))       # max_i a_max,i / c_i  (c_min per-i)

    # delta
    T = ds.T
    delta = cfg.confidence.delta
    if delta is None:
        delta = 1.0 / (T ** cfg.confidence.delta_power)
    n_res = ds.n_resources
    eps_D = 2.0 * np.sqrt(xi * np.log(n_res / delta)) + 2.0 * xi * np.log(n_res / delta)

    return float(gamma), float(xi), float(eps_D)


# --------------------------------------------------------------------------- #
# Per-type statistics from a batch
# --------------------------------------------------------------------------- #
def _per_type_stats(
    jobs: JobArrays, phi_rate: np.ndarray, grid: GeometricGrid,
    ts: TypeSpace, ds: Dataset, cfg: Config,
    n_total_for_p: int,
) -> dict:
    """
    Compute per-type empirical statistics over `jobs`. Returns dict of arrays
    indexed by flat type-id. Uses duration in *steps*.
    """
    n_types = ts.n_types
    n_res = ds.n_resources

    # assign each job to a type (use precomputed hw_idx, robust to float snapping)
    k_arr = np.asarray(grid.bin_of(phi_rate))          # 0..K* (0 = reject sentinel)
    type_ids = np.full(len(jobs.q), -1, dtype=np.int64)
    valid = k_arr >= 1
    for idx in np.where(valid)[0]:
        type_ids[idx] = ts.type_id_from_hw_idx(
            int(k_arr[idx]), float(jobs.q[idx]), int(jobs.hw_idx[idx])
        )

    N = np.zeros(n_types, dtype=np.int64)
    d_bar = np.zeros(n_types)
    C_elec_bar = np.zeros(n_types)
    C_carbon_bar = np.zeros(n_types)
    w = np.zeros(n_types)

    scc = cfg.objective.scc

    for tid in range(n_types):
        mask = type_ids == tid
        cnt = int(mask.sum())
        N[tid] = cnt
        if cnt == 0:
            continue
        d_steps = jobs.duration_steps[mask]
        d_bar[tid] = d_steps.mean()
        w[tid] = jobs.w_kw[mask].mean()
        # per-job grid impact uses hours for cost scale (native cost units),
        # consistent with the CSV's C_elec / C_carbon definition.
        d_hours = jobs.duration_hours[mask]
        C_elec_bar[tid] = np.mean(jobs.elec_price[mask] * w[tid] * d_hours)
        C_carbon_bar[tid] = np.mean(
            jobs.carbon_intensity[mask] * w[tid] * d_hours
        ) * scc

    p = N / max(n_total_for_p, 1)
    return dict(N=N, p=p, d_bar=d_bar, w=w,
                C_elec_bar=C_elec_bar, C_carbon_bar=C_carbon_bar,
                type_ids=type_ids)


def _single_objective_max(
    p: np.ndarray, obj: np.ndarray, v: np.ndarray, c: np.ndarray, cfg: Config
) -> float:
    """
    Utopian point z*: maximise a single objective sum_j p_j obj_j x_j subject to
    the same capacity constraints. Used to normalize each objective to [0,1].
    """
    lp = solve_lp_rs(p=p, r=obj, v=v, c=c, presolve=cfg.lp.presolve)
    return lp.optimum


# --------------------------------------------------------------------------- #
# Phase 0 entry point
# --------------------------------------------------------------------------- #
def run_phase0(ds: Dataset, cfg: Config) -> PhaseState:
    H = ds.H

    # --- 1. System calibration: Myerson from lognormal fit on H bids --------
    myerson = fit_myerson(H.v_rate, cfg.grid.phi_floor)
    phi_H = np.asarray(myerson.phi(H.v_rate))

    # --- 2. Geometric grid over virtual valuations --------------------------
    grid = build_geometric_grid(phi_H, cfg.grid.k_star, cfg.grid.phi_floor, myerson)

    # --- 3. Type space ------------------------------------------------------
    ts = build_type_space(H, ds, cfg)
    n_types = ts.n_types
    n_res = ds.n_resources

    # --- 4-5. Per-type statistics on H --------------------------------------
    stats = _per_type_stats(H, phi_H, grid, ts, ds, cfg, n_total_for_p=H.n)
    p = stats["p"]
    d_bar = stats["d_bar"]
    w = stats["w"]
    C_elec_bar = stats["C_elec_bar"]
    C_carbon_bar = stats["C_carbon_bar"]

    # bin floors L_j and discrete virtual value phi(L_j) per type
    L = np.zeros(n_types)
    phi_tilde = np.zeros(n_types)
    q_of_type = np.zeros(n_types)
    A_of_type = np.zeros((n_types, n_res))
    for tid in range(n_types):
        k, q, a_row = ts.decode(tid)
        L[tid] = grid.bnd_v[k - 1]
        q_of_type[tid] = q
        A_of_type[tid] = a_row
    phi_tilde = np.asarray(myerson.phi(L))

    # --- expected resource volume v_ij = A_i * d_bar_j (needed for utopian LPs) ---
    v = A_of_type * d_bar[:, None]                          # (n_types, |I|)

    # --- reward objective vectors (n_types, 3), with optional normalization ---
    lam = cfg.objective
    # raw per-type objective components
    raw_sat = q_of_type * L
    raw_prof = phi_tilde - C_elec_bar
    raw_carb = C_carbon_bar                                 # lower is better

    if lam.normalize:
        # Utopian points z*: single-objective LP maxima under the capacity
        # constraints. Matches the static simulator's normalization.
        # For satisfaction and profit: maximise. For carbon: the "best" is the
        # minimum achievable, but the static sim normalizes V_sus = 1 - C/z*_carb
        # using z*_carb = the max carbon under the profit-max allocation scale;
        # to stay faithful we take z*_carb as the max carbon single-objective LP
        # (the largest carbon the LP would ever spend), giving V_sus in [0,1].
        z_sat = _single_objective_max(p, raw_sat, v, ds.capacity, cfg)
        z_prof = _single_objective_max(p, raw_prof, v, ds.capacity, cfg)
        z_carb = _single_objective_max(p, raw_carb, v, ds.capacity, cfg)
        z_sat = max(z_sat, 1e-12)
        z_prof = max(z_prof, 1e-12)
        z_carb = max(z_carb, 1e-12)

        V_sat = raw_sat / z_sat
        V_prof = raw_prof / z_prof
        V_sus = 1.0 - raw_carb / z_carb                    # inverted: lower C -> higher V
        f = np.column_stack([
            lam.lambda1 * V_sat,
            lam.lambda2 * V_prof,
            lam.lambda3 * V_sus,
        ])
        r_rev = lam.lambda1 * V_sat + lam.lambda2 * V_prof  # revenue-only, normalized
        norm_ref = (z_sat, z_prof, z_carb)
    else:
        f = np.column_stack([
            lam.lambda1 * raw_sat,
            lam.lambda2 * raw_prof,
            -lam.lambda3 * raw_carb,
        ])
        r_rev = lam.lambda1 * q_of_type * L + lam.lambda2 * phi_tilde
        norm_ref = None

    # --- scalarize to frozen reward r~_j ------------------------------------
    weights = np.ones(3)
    r_tilde = np.asarray(scalarize(
        f, cfg.scalarization.method, weights, cfg.scalarization.rho
    ))

    # --- system constants ---------------------------------------------------
    r_max = float(np.max(r_tilde)) if np.any(p > 0) else 1.0
    v_min = float(np.min(v[v > 0])) if np.any(v > 0) else 1.0
    gamma, xi, eps_D = _system_constants(H, ds, cfg, r_max, v_min)

    # --- 6. Solve LP-RS -----------------------------------------------------
    lp: LPResult = solve_lp_rs(
        p=p, r=r_tilde, v=v, c=ds.capacity, presolve=cfg.lp.presolve
    )
    tau = lp.tau if lp.tau.size == n_res else np.zeros(n_res)

    return PhaseState(
        myerson=myerson, grid=grid, type_space=ts,
        p=p, d_bar=d_bar, w=w, r_tilde=r_tilde, r_rev=r_rev,
        C_elec_bar=C_elec_bar, C_carbon_bar=C_carbon_bar,
        v=v, L=L, phi_tilde=phi_tilde,
        tau=tau, lam_star=lp.optimum,
        gamma=gamma, xi=xi, eps_D=eps_D,
        n_types=n_types,
        norm_ref=norm_ref,
    )

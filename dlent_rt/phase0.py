"""
Phase 0 — Offline Bootstrap

Computes everything the online loop needs to start its first phase:
  - U0 calibration (median H job) and HOURLY bid synthesis (lognormal, on the fly)
  - sigma fit on H's markups and the geometric grid on hourly bids (fixed for the run)
  - Energy model (power draw from CPU/RAM/scheduling_class)
  - Grid-data lookup (electricity price + carbon intensity at request_time)
  - Per-type arrival rates p and per-COLUMN statistics (priority, hardware profile): duration, energy, costs,
    request, hourly cost base
  - Carbon price s_C, additive linear dollar reward r_tilde (dlent_rt.reward)
  - LP-RS -> shadow prices tau* and benchmark lambda*

Type terms (week 33, D1/D4): the bin sets only the value, the column sets the costs. For type (k, q, h),
    r~ = d_(q,h) (L_k + phi_(q,h)(L_k)) - C_elec_(q,h) - s_C K_(q,h)        (lambda = 1),   volume v = v_(q,h),
with L_k the lower edge of bin k, phi_(q,h)(L) = c_(q,h) phi_m(L / c_(q,h)), and d, C_elec, K, the request and c the
column means over H (every H job ran). None depends on the bid or on a per-bin duration. Columns with no H job get
fallback estimates (hardware-profile means over H). r~ - tau^T v is increasing in k inside a column (phi_m is
increasing), so the admitted bins of a column are an upper set (dlent_rt.mechanism).

The output `PhaseState` is the object the online model reads.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import Config
from .data import Dataset, JobArrays
from .grid import ValuationGrid, build_valuation_grid
from .lp_oracle import solve_lp_rs, LPResult
from .reward import type_reward, type_mean, weighted_quantile, electricity_cost, carbon_kg
from .models.core.error_params import ErrorParams
from .energy import EnergyParams, compute_energy_kwh, compute_power_kw
from .bid import MyersonModel, BidResult, bid_model, calibrate_u0, cost_base, synthesize_bids

# Bid and energy helpers
def _make_energy_params(cfg: Config) -> EnergyParams:
    return EnergyParams(
        machine_cores=cfg.energy_model.machine_cores,
        machine_ram_gb=cfg.energy_model.machine_ram_gb,
        p_core=cfg.energy_model.p_core,
        p_gb=cfg.energy_model.p_gb,
        pue=cfg.energy_model.pue,
    )


def _synthesize_bids(jobs: JobArrays, cfg: Config, seed: int, u0: float) -> BidResult:
    """Generate bids on the fly from resource requests (bid.synthesize_bids)."""
    return synthesize_bids(jobs, cfg, u0, seed)


def _compute_job_energy(jobs: JobArrays, cfg: Config) -> np.ndarray:
    """Compute energy_kwh per job using the physical energy model."""
    params = _make_energy_params(cfg)
    return compute_energy_kwh(
        jobs.A[:, 0], jobs.A[:, 1],
        jobs.scheduling_class, jobs.duration_hours, params,
    )


def _compute_job_power(jobs: JobArrays, cfg: Config) -> np.ndarray:
    """Compute instantaneous power in kW per job."""
    params = _make_energy_params(cfg)
    return compute_power_kw(
        jobs.A[:, 0], jobs.A[:, 1],
        jobs.scheduling_class, params,
    )


def _lookup_grid(jobs: JobArrays, ds: Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Get electricity price and carbon intensity at each job's request_time."""
    if ds.grid_data is not None:
        return ds.grid_data.lookup(jobs.datetime)
    return (np.full(jobs.n, 0.035), np.full(jobs.n, 380.0))



# Type-space bookkeeping
@dataclass
class TypeSpace:
    """The finite type space J = K x Q x A."""
    k_star: int
    priorities: np.ndarray
    hw_profiles: np.ndarray
    resources: list[str]

    @property
    def n_hw(self) -> int:
        return len(self.hw_profiles)

    @property
    def n_cols(self) -> int:
        """Columns (priority, hardware profile): the types without the bid bin."""
        return len(self.priorities) * self.n_hw

    @property
    def n_types(self) -> int:
        return self.k_star * self.n_cols

    def decode(self, type_id: int) -> tuple[int, float, np.ndarray]:
        ai = type_id % self.n_hw
        rest = type_id // self.n_hw
        qi = rest % len(self.priorities)
        ki = rest // len(self.priorities)
        return ki + 1, float(self.priorities[qi]), self.hw_profiles[ai]


def assign_type_ids(
    ts: TypeSpace, k_arr: np.ndarray, priority: np.ndarray, hw_idx: np.ndarray,
) -> np.ndarray:
    """type_id = ((k - 1) * |Q| + q_idx) * |A| + hw_idx, vectorised (k in 1..K*)."""
    k_arr = np.asarray(k_arr, dtype=np.int64)
    qi = np.searchsorted(ts.priorities, np.asarray(priority, float)).astype(np.int64)
    return ((k_arr - 1) * len(ts.priorities) + qi) * ts.n_hw + np.asarray(hw_idx, np.int64)


def type_arrays(ts: TypeSpace, grid: ValuationGrid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-type (L = lower edge of the type's bid bin, 0 for bin 1 (grid.lower_edges), priority, hardware profile A)."""
    n = ts.n_types
    ids = np.arange(n)
    ai = ids % ts.n_hw
    rest = ids // ts.n_hw
    qi = rest % len(ts.priorities)
    ki = rest // len(ts.priorities)
    L = grid.lower_edges[ki]
    return L, ts.priorities[qi].astype(float), ts.hw_profiles[ai]


def column_ids(ts: "TypeSpace", priority: np.ndarray, hw_idx: np.ndarray) -> np.ndarray:
    """Column id of (priority, hardware profile) = q_idx * |A| + hw_idx: the type id without the bid bin."""
    qi = np.searchsorted(ts.priorities, np.asarray(priority, float)).astype(np.int64)
    return qi * ts.n_hw + np.asarray(hw_idx, np.int64)


def cols_to_types(ts: "TypeSpace", x) -> np.ndarray:
    """Spread a per-column array (n_cols, ...) over the K* bins: type id = (k - 1) n_cols + column id."""
    x = np.asarray(x)
    return np.tile(x, (ts.k_star,) + (1,) * (x.ndim - 1))


def column_means(cid: np.ndarray, n_cols: int, cols: dict) -> tuple[np.ndarray, dict]:
    """Per-column job counts and sample means of each array of cols (2-D arrays column by column; 0 if empty)."""
    N = np.bincount(cid, minlength=n_cols)

    def mean(x):
        x = np.asarray(x, float)
        if x.ndim == 1:
            return type_mean(cid, N, x)
        return np.column_stack([type_mean(cid, N, x[:, i]) for i in range(x.shape[1])])
    return N, {k: mean(x) for k, x in cols.items()}


def column_fallback(hw_idx: np.ndarray, ts: "TypeSpace", cols: dict) -> dict:
    """
    Per-column estimates for the columns with no job (D4): the mean over the jobs with the same hardware profile (all
    priorities); the mean over all jobs if the profile has none.
    """
    hw_col = np.arange(ts.n_cols) % ts.n_hw
    _, by_hw = column_means(np.asarray(hw_idx, np.int64), ts.n_hw, cols)
    n_hw = np.bincount(np.asarray(hw_idx, np.int64), minlength=ts.n_hw)
    out = {}
    for k, x in cols.items():
        overall = np.mean(np.asarray(x, float), axis=0)
        g = np.where((n_hw > 0).reshape((-1,) + (1,) * (by_hw[k].ndim - 1)), by_hw[k], overall)
        out[k] = g[hw_col]
    return out


def type_values(ts: "TypeSpace", grid: ValuationGrid, myerson: MyersonModel, c_col, d_hours_col):
    """
    Value terms of every type (k, q, h), from the bin's lower edge and the column (D1, week 33):
        v_bar   = d_(q,h) L_k                  [$]  (customer value)
        phi_bar = d_(q,h) phi_(q,h)(L_k)       [$]  phi_(q,h)(L) = c_(q,h) phi_m(L / c_(q,h))
    d_(q,h) the column's estimated duration [h], c_(q,h) its hourly cost base [$/h], L_k the lower edge of bin k
    (0 for bin 1, where phi_m is at its floor).
    """
    L = grid.lower_edges[:, None]
    d = np.asarray(d_hours_col, float)[None, :]
    phi = np.asarray(myerson.phi_job(L, np.asarray(c_col, float)[None, :]))
    return (L * d).ravel(), (phi * d).ravel()


def column_cost_base(ts: "TypeSpace", A_col: np.ndarray, u0: float, cfg) -> np.ndarray:
    """c_(q,h) = U0 (1 + ln q) + gamma1 cores + gamma2 GB of the column's mean request [$ per hour] (= the mean of
    the jobs' c_j, the cost base is linear in the request)."""
    q_col = ts.priorities[np.arange(ts.n_cols) // ts.n_hw]
    return cost_base(A_col[:, 0], A_col[:, 1], q_col, u0, cfg)


def build_type_space(H: JobArrays, ds, cfg: Config) -> TypeSpace:
    priorities = np.array(sorted(cfg.grid.priority_classes), dtype=float)
    profiles = ds.hw_catalogue.centroids
    return TypeSpace(
        k_star=cfg.grid.k_star, priorities=priorities,
        hw_profiles=profiles, resources=H.resources,
    )



# Phase state
@dataclass
class PhaseState:
    """Everything the online loop needs to run one phase."""
    myerson: MyersonModel
    grid: ValuationGrid
    type_space: TypeSpace
    bids: BidResult

    # per-type arrays (0..n_types-1). p is per type; the duration, energy, costs and request are the COLUMN's
    # (priority, profile) means spread over the bins (cols_to_types); v_bar / phi_bar are the type value terms
    p: np.ndarray                         # empirical type arrival rate (per tick)
    d_bar: np.ndarray                     # column's estimated duration (steps)
    v_bar: np.ndarray                     # customer value d_(q,h) L_k [$] (type_values)
    phi_bar: np.ndarray                   # d_(q,h) phi_(q,h)(L_k) [$] (type_values)
    energy_bar: np.ndarray                # column's mean energy_kwh
    power_bar: np.ndarray                 # column's mean power_kw
    r_tilde: np.ndarray                   # scalarized type reward
    r_nocarbon: np.ndarray                # r_tilde without the carbon term (diagnostic)
    C_elec_bar: np.ndarray                # column's mean electricity cost [$]
    C_carbon_bar: np.ndarray              # column's mean carbon emissions (kgCO2)
    v: np.ndarray                         # (n_types, |I|) column's volume = mean request * mean duration
    L: np.ndarray                         # lower edge of the type's bid bin (value and threshold payment)

    # LP-RS output
    tau: np.ndarray
    lam_star: float

    # system constants
    gamma: float
    xi: float
    eps_D: float

    n_types: int
    n_assigned: int = 0
    n_types_hw_degenerate: int = 0
    # $/kgCO2 applied to C_carbon_bar: s_C, frozen at Phase 0
    carbon_price: float = 0.0             # always overwritten in run_phase0
    norm_info: Optional[dict] = None      # q, Q_cv, Q_abs_pi, D_q, K_q, s_C
    A_type: Optional[np.ndarray] = None   # (n_types, |I|) column's mean request of the volume
    u0: float = 0.0                       # $ per hour per job, calibrated on H (bid.calibrate_u0), then fixed
    r_max: float = 1.0                    # Tests 2-3 reward unit: max r~ over H's types with r~ > 0 (reward_unit)
    c_col: Optional[np.ndarray] = None    # (n_cols,) column's hourly cost base c_(q,h) [$/h] (column_cost_base)
    # per COLUMN, where its statistics come from: 0 = fresh (H / this phase), 1 = carried forward, 2 = fallback
    src_cost: Optional[np.ndarray] = None    # hourly cost base c_(q,h) (from arrivals)
    src_dur: Optional[np.ndarray] = None     # duration, energy, costs, request of the volume (from admitted jobs)
    # the counts behind the statistics (stats.update "by_end_reason" merges old and new with these weights)
    n_type: Optional[np.ndarray] = None      # (n_types,) arrivals behind p
    T_obs: float = 0.0                       # ticks behind p (p = n_type / T_obs)
    n_col_arr: Optional[np.ndarray] = None   # (n_cols,) arrivals behind c_col
    n_col_obs: Optional[np.ndarray] = None   # (n_cols,) observed (admitted) jobs behind d_bar, energy, costs, request

    def col(self, x) -> np.ndarray:
        """The per-column values of a per-type array spread by cols_to_types (the bin-1 block)."""
        return np.asarray(x)[:self.type_space.n_cols]



# System constants
def _system_constants(
    ds: Dataset, cfg: Config, r_max: float, v_min: float, v_rate_max: float,
) -> tuple[float, float, float]:
    d_max = ds.d_max_steps
    c_max = ds.capacity.max()
    r_max_val = max(r_max, 1e-12)
    v_min_val = max(v_min, 1e-12)

    gamma = max(r_max_val, v_rate_max, d_max, c_max, r_max_val / v_min_val)
    xi = float(np.max(ds.a_max / ds.capacity))

    # eps_D of the DATA xi, from the one implementation of (A.7) in error_params.py
    # (the online loop builds its own ErrorParams with the forced constants)
    T = ds.T
    delta = cfg.confidence.delta
    if delta is None:
        delta = 1.0 / (T ** cfg.confidence.delta_power)
    ep = ErrorParams(gamma=float(gamma), xi=xi, c_min=float(np.min(ds.capacity)),
                     n_res=int(ds.n_resources), T=int(T), delta=float(delta),
                     drop_EF=cfg.tests.drop_dual_learning_terms)
    return float(gamma), float(xi), float(ep.eps_D)



def reward_unit(r_tilde: np.ndarray, p: np.ndarray) -> float:
    """
    The reward unit of Tests 2 and 3 (paper definition): rewards of admissible actions are bounded by gamma (forced
    gamma = 1), so r_max = the MAXIMUM of r~ over the populated types (p > 0) with r~ > 0. Types with r~ <= 0 are
    never admitted, so they do not count. 1.0 if no populated type has r~ > 0.
    """
    r, p = np.asarray(r_tilde, float), np.asarray(p, float)
    pos = (p > 0) & (r > 0)
    return float(r[pos].max()) if pos.any() else 1.0


# Phase 0 entry point
def run_phase0(ds: Dataset, cfg: Config) -> PhaseState:
    H = ds.H

    # 1. U0 from H's median job, then the hourly bids
    u0 = calibrate_u0(H, cfg)
    bids = _synthesize_bids(H, cfg, seed=cfg.run.seed, u0=u0)

    # 2. Bid model: sigma fit on H's markups (ReInit updates it by bid.sigma_update)
    myerson = bid_model(cfg, bids.v_rate, bids.cost_base)

    # 3. Geometric grid on the hourly bids, between the smallest and the largest bid (fixed for the run)
    grid = build_valuation_grid(bids.v_rate, cfg.grid.k_star)

    # 4. Build type space
    ts = build_type_space(H, ds, cfg)
    n_types = ts.n_types
    n_res = ds.n_resources

    # 5. Compute energy and grid prices per job
    energy_kwh = _compute_job_energy(H, cfg)
    power_kw   = _compute_job_power(H, cfg)
    elec_price, carbon_intensity = _lookup_grid(H, ds)

    # 6. p per type (bid bin, column): N_j / ticks spanned by H (real-tick time)
    tid = assign_type_ids(ts, grid.assign(bids.v_rate), H.priority, H.hw_idx)
    N = np.bincount(tid, minlength=n_types)
    p = N / max(ds.H_span_steps, 1)

    # 7. Column statistics over H (every H job ran); columns with no H job: hardware-profile fallback (D4)
    cols = dict(d_bar=H.duration_steps, energy_bar=energy_kwh, power_bar=power_kw,
                C_elec_bar=electricity_cost(energy_kwh, elec_price),
                C_carbon_bar=carbon_kg(energy_kwh, carbon_intensity), A_type=H.A)
    Nc, col = column_means(column_ids(ts, H.priority, H.hw_idx), ts.n_cols, cols)
    seen = Nc > 0
    fb = column_fallback(H.hw_idx, ts, cols)
    col = {k: np.where(seen.reshape((-1,) + (1,) * (x.ndim - 1)), x, fb[k]) for k, x in col.items()}
    src = np.where(seen, 0, 2).astype(np.int8)          # 0 = from H, 2 = fallback estimate (models.core.reinit)
    c_col = column_cost_base(ts, col["A_type"], u0, cfg)

    # 8. Type terms: value from the bin's lower edge, costs and volume from the column
    T = {k: cols_to_types(ts, x) for k, x in col.items()}
    d_bar, energy_bar_arr, power_bar_arr = T["d_bar"], T["energy_bar"], T["power_bar"]
    C_elec_bar, C_carbon_bar, A_vol = T["C_elec_bar"], T["C_carbon_bar"], T["A_type"]
    v_bar_arr, phi_bar_arr = type_values(ts, grid, myerson, c_col, col["d_bar"] * cfg.time.tick_seconds / 3600.0)
    L, _, A_of_type = type_arrays(ts, grid)
    v = A_vol * d_bar[:, None]

    # 9. Linear dollar reward from the type terms, with the carbon price s_C computed here
    r_tilde, r_nocarbon, norm_info = type_reward(
        v_bar_arr, phi_bar_arr, C_elec_bar, C_carbon_bar, p, cfg,
    )
    carbon_price = norm_info["s_C"]

    # 12. System constants
    # r_max for gamma: p-weighted quantile of |r_tilde| over populated types
    # (constants.r_max_quantile; 1.0 = plain max). A quantile < 1 keeps a few
    # extreme types (e.g. a lognormal phi(v) -> -inf in the left tail) from
    # dominating gamma through r_max and r_max / v_min.
    if np.any(p > 0):
        q_r = cfg.constants.r_max_quantile
        abs_r = np.abs(r_tilde[p > 0])
        r_max = (float(np.max(abs_r)) if q_r >= 1.0
                 else weighted_quantile(abs_r, p[p > 0], q_r))
    else:
        r_max = 1.0
    # gamma's r_max/v_min term must exclude types whose hardware profile is
    # numerically degenerate (a k-means centroid with a component collapsed
    # near zero). Such a type produces v_ij approx 0, inflating gamma by
    # orders of magnitude without reflecting any real workload category.
    # This does NOT affect which types are served -- only which types
    # contribute to the v_min term used for the gamma system constant.
    hw_floor = cfg.hardware.hw_floor
    hw_reliable_mask = np.all(A_of_type >= hw_floor, axis=1)  # (n_types,)
    reliable = (p > 0) & hw_reliable_mask
    if np.any(reliable) and np.any(v[reliable] > 0):
        v_min = float(np.min(v[reliable][v[reliable] > 0]))
    else:
        v_min = float(np.min(v[v > 0])) if np.any(v > 0) else 1e-12
        import warnings
        warnings.warn(
            f"No type has all hardware components >= hw_floor={hw_floor}; "
            f"falling back to unfiltered v_min. gamma may be unreliable.",
            RuntimeWarning,
        )
    n_types_hw_degenerate = int((~hw_reliable_mask & (p > 0)).sum())
    v_rate_max = float(bids.v_rate.max())
    gamma, xi, eps_D = _system_constants(ds, cfg, r_max, v_min, v_rate_max)

    # 13. Solve LP-RS
    lp: LPResult = solve_lp_rs(
        p=p, r=r_tilde, v=v, c=ds.capacity, presolve=cfg.lp.presolve
    )
    tau = lp.tau if lp.tau.size == n_res else np.zeros(n_res)

    return PhaseState(
        myerson=myerson, grid=grid, type_space=ts, bids=bids,
        p=p, d_bar=d_bar, v_bar=v_bar_arr, phi_bar=phi_bar_arr,
        energy_bar=energy_bar_arr, power_bar=power_bar_arr,
        r_tilde=r_tilde, r_nocarbon=r_nocarbon,
        C_elec_bar=C_elec_bar, C_carbon_bar=C_carbon_bar,
        v=v, L=L,
        tau=tau, lam_star=lp.optimum,
        gamma=gamma, xi=xi, eps_D=eps_D,
        n_types=n_types,
        n_assigned=int(tid.size),
        n_types_hw_degenerate=n_types_hw_degenerate,
        carbon_price=float(carbon_price), norm_info=norm_info, A_type=A_vol,
        u0=float(u0), r_max=reward_unit(r_tilde, p), c_col=c_col, src_cost=src, src_dur=src.copy(),
        n_type=N.astype(float), T_obs=float(max(ds.H_span_steps, 1)), n_col_arr=Nc.astype(float),
        n_col_obs=Nc.astype(float),
    )

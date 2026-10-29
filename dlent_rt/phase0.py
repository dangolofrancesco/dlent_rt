"""
Phase 0 — Offline Bootstrap

Computes everything the online loop needs to start its first phase:
  - U0 calibration (median H job) and bid synthesis (lognormal, on-the-fly)
  - sigma refit on H's markups (bid.refit_sigma) and the geometric bid grid
  - Energy model (power draw from CPU/RAM/scheduling_class)
  - Grid-data lookup (electricity price + carbon intensity at request_time)
  - Per-type statistics (p, d_bar, v_bar, phi_bar, energy, costs)
  - Carbon price s_C, additive linear dollar reward r_tilde (dlent_rt.reward)
  - LP-RS -> shadow prices tau* and benchmark lambda*

The type reward uses PER-TYPE SAMPLE MEANS of v and phi(v), not the bin floor L. L is kept for the threshold
payment (models.theory: the lower edge of the lowest admitted bin).

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
from .bid import MyersonModel, BidResult, bid_model, calibrate_u0, synthesize_bids

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
    def n_types(self) -> int:
        return self.k_star * len(self.priorities) * self.n_hw

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


def type_request(tid: np.ndarray, A: np.ndarray, A_of_type: np.ndarray) -> np.ndarray:
    """
    (n_types, |I|) request entering the type volume v = A_type * d_bar: the mean real request of the jobs with
    type id tid (rows of A); the hardware profile A_of_type for types without jobs.
    """
    N = np.bincount(tid, minlength=len(A_of_type))
    mean = np.column_stack([type_mean(tid, N, A[:, i]) for i in range(A.shape[1])])
    return np.where(N[:, None] > 0, mean, A_of_type)


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

    # per-type arrays (0..n_types-1)
    p: np.ndarray                         # empirical type probability
    d_bar: np.ndarray                     # expected duration (steps)
    v_bar: np.ndarray                     # mean bid v per type (for reward)
    phi_bar: np.ndarray                   # mean phi(v) per type (for reward)
    energy_bar: np.ndarray                # mean energy_kwh per type
    power_bar: np.ndarray                 # mean power_kw per type
    r_tilde: np.ndarray                   # frozen scalarized reward
    r_nocarbon: np.ndarray                # r_tilde without the carbon term (diagnostic)
    C_elec_bar: np.ndarray                # mean electricity cost per type
    C_carbon_bar: np.ndarray              # mean carbon emissions per type (kgCO2)
    v: np.ndarray                         # (n_types, |I|) expected resource volume = A_type * d_bar
    L: np.ndarray                         # lower edge of the type's bid bin (threshold payment)

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
    A_type: Optional[np.ndarray] = None   # (n_types, |I|) request of the type volume (type_request)
    u0: float = 0.0                       # $ per hour per job, calibrated on H (bid.calibrate_u0), then fixed
    r_max: float = 1.0                    # p-weighted quantile of |r~| over H's types: Tests 2-3 reward unit



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



# Per-type statistics
def _per_type_stats(
    jobs: JobArrays, bids: BidResult, energy_kwh: np.ndarray,
    power_kw: np.ndarray, elec_price: np.ndarray, carbon_intensity: np.ndarray,
    grid: ValuationGrid, ts: TypeSpace, ds: Dataset, cfg: Config,
    n_total_for_p: int, phi_job: np.ndarray,
) -> dict:
    """
    Compute per-type empirical statistics.

    v_bar and phi_bar are SAMPLE MEANS over the jobs assigned to each type
    (actual job values), not the discretization bin floor. `phi_job` is the
    per-job virtual value the type reward averages (see run_phase0, step 2).
    """
    n_types = ts.n_types

    tid = assign_type_ids(ts, grid.assign(bids.v_rate), jobs.priority, jobs.hw_idx)

    # per-type sample means over the jobs (reward.type_mean, bincount)
    N = np.bincount(tid, minlength=n_types).astype(np.int64)

    def _mean(x: np.ndarray) -> np.ndarray:
        return type_mean(tid, N, np.asarray(x, float))

    d_bar        = _mean(jobs.duration_steps)
    v_bar        = _mean(bids.v_rate)            # mean bid per type
    phi_bar      = _mean(phi_job)                # mean phi(v) per type
    energy_bar   = _mean(energy_kwh)
    power_bar    = _mean(power_kw)
    C_elec_bar   = _mean(electricity_cost(energy_kwh, elec_price))
    C_carbon_bar = _mean(carbon_kg(energy_kwh, carbon_intensity))     # E * CI [kgCO2]

    p = N / max(n_total_for_p, 1)
    return dict(N=N, p=p, d_bar=d_bar, v_bar=v_bar, phi_bar=phi_bar,
                energy_bar=energy_bar, power_bar=power_bar,
                C_elec_bar=C_elec_bar, C_carbon_bar=C_carbon_bar,
                type_ids=tid, n_assigned=int(tid.size))



# Phase 0 entry point
def run_phase0(ds: Dataset, cfg: Config) -> PhaseState:
    H = ds.H

    # 1. U0 from H's median job, then the bids
    u0 = calibrate_u0(H, cfg)
    bids = _synthesize_bids(H, cfg, seed=cfg.run.seed, u0=u0)

    # 2. Bid model: sigma refit on H's markups (bid.refit_sigma) or bid.sigma; phi_job(v, c_j) per job, the same
    #    function ReInit and the online reward use
    myerson = bid_model(cfg, bids.v_rate, bids.cost_base)
    phi_job = np.asarray(myerson.phi_job(bids.v_rate, bids.cost_base))

    # 3. Geometric bid grid between the smallest and the largest bid
    grid = build_valuation_grid(bids.v_rate, cfg.grid.k_star)

    # 4. Build type space
    ts = build_type_space(H, ds, cfg)
    n_types = ts.n_types
    n_res = ds.n_resources

    # 5. Compute energy and grid prices per job
    energy_kwh = _compute_job_energy(H, cfg)
    power_kw   = _compute_job_power(H, cfg)
    elec_price, carbon_intensity = _lookup_grid(H, ds)

    # 6. Per-type statistics
    p_normaliser = ds.H_span_steps                  # p_j = N_j / ticks spanned by H (real-tick time)

    stats = _per_type_stats(
        H, bids, energy_kwh, power_kw, elec_price, carbon_intensity,
        grid, ts, ds, cfg, n_total_for_p=p_normaliser, phi_job=phi_job,
    )
    p              = stats["p"]
    d_bar          = stats["d_bar"]
    v_bar_arr      = stats["v_bar"]
    phi_bar_arr    = stats["phi_bar"]
    energy_bar_arr = stats["energy_bar"]
    power_bar_arr  = stats["power_bar"]
    C_elec_bar     = stats["C_elec_bar"]
    C_carbon_bar   = stats["C_carbon_bar"]

    # 7. Bin floors per type (threshold payment)
    L, _, A_of_type = type_arrays(ts, grid)

    # 8. Expected resource volume: the mean real request of the type's jobs x its mean duration
    A_vol = type_request(stats["type_ids"], H.A, A_of_type)
    v = A_vol * d_bar[:, None]

    # 9. Linear dollar reward from the per-type SAMPLE MEANS, with the carbon price s_C computed here
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
        n_assigned=stats["n_assigned"],
        n_types_hw_degenerate=n_types_hw_degenerate,
        carbon_price=float(carbon_price), norm_info=norm_info, A_type=A_vol,
        u0=float(u0), r_max=float(r_max),
    )

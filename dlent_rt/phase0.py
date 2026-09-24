"""
Phase 0 — Offline Bootstrap

Computes everything the online loop needs to start its first phase:
  - Bid synthesis (uniform or lognormal, on-the-fly)
  - Myerson model and valuation grid
  - Energy model (power draw from CPU/RAM/scheduling_class)
  - Grid-data lookup (electricity price + carbon intensity at request_time)
  - Per-type statistics (p, d_bar, v_bar, phi_bar, energy, costs)
  - Normalization (per_job / moving_avg / common_currency / utopian)
  - Scalarized reward r_tilde
  - LP-RS -> shadow prices tau* and benchmark lambda*

RAW OBJECTIVES USE PER-TYPE SAMPLE MEANS:
  Raw objectives use PER-TYPE SAMPLE MEANS of v and phi(v), NOT the bin floor
  L. Using phi(L) causes systematically negative raw_prof because bin floors
  are small (geometric grid -> many tiny L values -> phi(L) << 0). Using
  phi_bar = mean(phi(v)) over the jobs in each type reflects the actual job
  values and produces meaningful single-objective ("utopian") optima.
  The bin floor L is retained only for the payment rule (phi_tilde).

The output `PhaseState` is the object both the theory and practical models read.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import Config
from .data import Dataset, JobArrays
from .grid import ValuationGrid, build_valuation_grid
from .lp_oracle import scalarize, solve_lp_rs, LPResult
from .energy import EnergyParams, compute_energy_kwh, compute_power_kw
from .bid import (
    MyersonModel, fit_myerson, compute_phi_tilde,
    UniformBidParams, LognormalBidParams, BidResult,
    synthesize_uniform, synthesize_lognormal,
)

# Bid and energy helpers
def _make_energy_params(cfg: Config) -> EnergyParams:
    return EnergyParams(
        spec_cpu_core=cfg.energy_model.spec_cpu_core,
        spec_ram_gb=cfg.energy_model.spec_ram_gb,
        p_core=cfg.energy_model.p_core,
        p_gb=cfg.energy_model.p_gb,
        pue=cfg.energy_model.pue,
    )


def _synthesize_bids(jobs: JobArrays, cfg: Config, seed: int) -> BidResult:
    """Generate bids on the fly from resource requests."""
    rng = np.random.default_rng(seed)
    if cfg.bid.strategy == "uniform":
        params = UniformBidParams(
            gamma1=cfg.bid.gamma1, gamma2=cfg.bid.gamma2,
            base_utility=cfg.bid.base_utility,
            spec_cpu_core=cfg.energy_model.spec_cpu_core,
            spec_ram_gb=cfg.energy_model.spec_ram_gb,
        )
        return synthesize_uniform(
            jobs.A[:, 0], jobs.A[:, 1],
            jobs.duration_hours, jobs.priority, params, rng,
        )
    else:  # lognormal
        params = LognormalBidParams(
            sigma=cfg.bid.sigma,
            base_multiplier=cfg.bid.base_multiplier,
            base_utility=cfg.bid.base_utility,
            gamma1=cfg.bid.gamma1,
            gamma2=cfg.bid.gamma2,
            spec_cpu_core=cfg.energy_model.spec_cpu_core,
            spec_ram_gb=cfg.energy_model.spec_ram_gb,
        )
        return synthesize_lognormal(
            jobs.A[:, 0], jobs.A[:, 1],
            jobs.duration_hours, params, rng,
        )


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
    _hw_lookup: dict = field(default_factory=dict)

    @property
    def n_hw(self) -> int:
        return len(self.hw_profiles)

    @property
    def n_types(self) -> int:
        return self.k_star * len(self.priorities) * self.n_hw

    def hw_index(self, a_row: np.ndarray) -> int:
        return self._hw_lookup[tuple(np.round(a_row, 8))]

    def type_id_from_hw_idx(self, k: int, priority: float, hw_idx: int) -> int:
        ki = k - 1
        qi = int(np.searchsorted(self.priorities, priority))
        return (ki * len(self.priorities) + qi) * self.n_hw + hw_idx

    def type_id(self, k: int, priority: float, a_row: np.ndarray) -> int:
        ki = k - 1
        qi = int(np.searchsorted(self.priorities, priority))
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
    profiles = ds.hw_catalogue.centroids
    lookup = {tuple(np.round(row, 8)): i for i, row in enumerate(profiles)}
    return TypeSpace(
        k_star=cfg.grid.k_star, priorities=priorities,
        hw_profiles=profiles, resources=H.resources, _hw_lookup=lookup,
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
    r_rev: np.ndarray                     # revenue-only base (time-aware oracle)
    C_elec_bar: np.ndarray                # mean electricity cost per type
    C_carbon_bar: np.ndarray              # mean carbon emissions per type (kgCO2)
    v: np.ndarray                         # (n_types, |I|) expected resource volume
    L: np.ndarray                         # bid-space bin floor (for payment rule)
    phi_tilde: np.ndarray                 # phi(L): virtual value at floor (payment rule)

    # LP-RS output
    tau: np.ndarray
    lam_star: float

    # system constants
    gamma: float
    xi: float
    eps_D: float

    n_types: int
    norm_ref: Optional[tuple] = None
    norm_strategy: str = "common_currency"  # always overwritten in run_phase0
    n_ir_rejected: int = 0
    n_assigned: int = 0
    norm_degenerate: tuple = (False, False, False)
    n_types_hw_degenerate: int = 0



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

    T = ds.T
    delta = cfg.confidence.delta
    if delta is None:
        delta = 1.0 / (T ** cfg.confidence.delta_power)
    n_res = ds.n_resources
    eps_D = (2.0 * np.sqrt(xi * np.log(n_res / delta))
             + 2.0 * xi * np.log(n_res / delta))

    return float(gamma), float(xi), float(eps_D)



# Per-type statistics
def _per_type_stats(
    jobs: JobArrays, bids: BidResult, energy_kwh: np.ndarray,
    power_kw: np.ndarray, elec_price: np.ndarray, carbon_intensity: np.ndarray,
    grid: ValuationGrid, ts: TypeSpace, ds: Dataset, cfg: Config,
    n_total_for_p: int, myerson: MyersonModel,
) -> dict:
    """
    Compute per-type empirical statistics.

    v_bar and phi_bar are SAMPLE MEANS over the jobs assigned to each type
    (actual job values), not the discretization bin floor.
    """
    n_types = ts.n_types

    k_arr = np.asarray(grid.assign(bids.v_rate, myerson))
    type_ids = np.full(jobs.n, -1, dtype=np.int64)
    valid = k_arr >= 1
    n_ir_rejected = int((~valid).sum())
    for idx in np.where(valid)[0]:
        type_ids[idx] = ts.type_id_from_hw_idx(
            int(k_arr[idx]), float(jobs.priority[idx]), int(jobs.hw_idx[idx])
        )

    N = np.zeros(n_types, dtype=np.int64)
    d_bar        = np.zeros(n_types)
    v_bar        = np.zeros(n_types)   # mean bid per type
    phi_bar      = np.zeros(n_types)   # mean phi(v) per type
    bid_high_bar = np.zeros(n_types)   # mean upper bid-support bound per type
    energy_bar   = np.zeros(n_types)
    power_bar    = np.zeros(n_types)
    C_elec_bar   = np.zeros(n_types)
    C_carbon_bar = np.zeros(n_types)

    for tid in range(n_types):
        mask = type_ids == tid
        cnt = int(mask.sum())
        N[tid] = cnt
        if cnt == 0:
            continue
        d_bar[tid]        = jobs.duration_steps[mask].mean()
        v_bar[tid]        = bids.v_rate[mask].mean()
        phi_bar[tid]      = bids.phi_rate[mask].mean()
        bid_high_bar[tid] = bids.bid_high[mask].mean()
        energy_bar[tid]   = energy_kwh[mask].mean()
        power_bar[tid]    = power_kw[mask].mean()
        C_elec_bar[tid]   = (energy_kwh[mask] * elec_price[mask]).mean()
        C_carbon_bar[tid] = (carbon_intensity[mask]).mean()         # energy_kwh[mask] * carbon_intensity[mask]

    p = N / max(n_total_for_p, 1)
    return dict(N=N, p=p, d_bar=d_bar, v_bar=v_bar, phi_bar=phi_bar,
                bid_high_bar=bid_high_bar,
                energy_bar=energy_bar, power_bar=power_bar,
                C_elec_bar=C_elec_bar, C_carbon_bar=C_carbon_bar,
                type_ids=type_ids, n_ir_rejected=n_ir_rejected,
                n_assigned=int(valid.sum()))



# Normalization
def _normalize_objectives(
    raw_sat: np.ndarray, raw_prof: np.ndarray, raw_carb: np.ndarray,
    p: np.ndarray, v: np.ndarray, ds: Dataset, cfg: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[tuple], tuple]:
    """
    Dispatch normalization by cfg.normalization.strategy.
    Returns (V_sat, V_prof, V_carb, norm_ref, norm_degenerate).

    Strategy reference (this function is the single source of truth; there is
    no separate normalization module):

      "per_job"        each objective is rescaled to roughly [-1, 1] by its own
                       largest absolute value across populated types. Simple,
                       scale-free, and the recommended default.
      "common_currency"  carbon (kgCO2) is first monetised via a carbon tax so
                       that all three objectives are in dollars, then all three
                       are divided by ONE shared scale so their real dollar
                       magnitudes stay comparable (see the branch comment).
    """
    strategy = cfg.normalization.strategy

    def _span(arr: np.ndarray) -> float:
        return max(float(np.max(np.abs(arr[p > 0]))) if np.any(p > 0) else 1.0, 1e-12)

    if strategy == "per_job":
        # Each objective scaled to ~[-1, 1] by its own max abs value over
        # populated types. Carbon gets no extra scalar here: its relative
        # weight is objective.lambda3, applied uniformly across strategies.
        V_sat  = raw_sat  / _span(raw_sat)
        V_prof = raw_prof / _span(raw_prof)
        V_carb = raw_carb / _span(raw_carb)
        norm_ref = (_span(raw_sat), _span(raw_prof), _span(raw_carb))
        norm_deg = (False, False, False)

    elif strategy == "common_currency":
        # Convert carbon from kgCO2 to $ via carbon tax. All three are now in dollars.
        carbon_tax = cfg.normalization.carbon_tax_per_kg
        raw_carb_dollar = raw_carb * carbon_tax  # kgCO2 -> $

        # Divide ALL THREE by ONE shared scale (the largest dollar magnitude
        # across the three objectives, over populated types). 
        common = max(_span(raw_sat), _span(raw_prof), _span(raw_carb_dollar))
        V_sat  = raw_sat         / common
        V_prof = raw_prof        / common
        V_carb = raw_carb_dollar / common

        norm_ref = (common, common, common)
        norm_deg = (False, False, False)

    else:
        raise ValueError(f"Unknown normalization strategy: {strategy}")

    return V_sat, V_prof, V_carb, norm_ref, norm_deg


def _single_objective_max(
    p: np.ndarray, obj: np.ndarray, v: np.ndarray, c: np.ndarray, cfg: Config
) -> float:
    lp = solve_lp_rs(p=p, r=obj, v=v, c=c, presolve=cfg.lp.presolve)
    return lp.optimum


# Phase 0 entry point
def run_phase0(ds: Dataset, cfg: Config) -> PhaseState:
    H = ds.H

    # 1. Synthesize bids on-the-fly
    bids = _synthesize_bids(H, cfg, seed=cfg.run.seed)

    # 2. Fit Myerson model on synthesized bids
    myerson = fit_myerson(bids.v_rate)

    # 3. Build valuation grid
    grid = build_valuation_grid(
        bids.v_rate, myerson, cfg.grid.k_star,
        cfg.grid.space, cfg.grid.spacing, cfg.grid.phi_floor,
    )

    # 4. Build type space
    ts = build_type_space(H, ds, cfg)
    n_types = ts.n_types
    n_res = ds.n_resources

    # 5. Compute energy and grid prices per job
    energy_kwh = _compute_job_energy(H, cfg)
    power_kw   = _compute_job_power(H, cfg)
    elec_price, carbon_intensity = _lookup_grid(H, ds)

    # 6. Per-type statistics
    p_normaliser = ds.H_span_steps if cfg.time.step_semantics == "real_tick" else H.n

    stats = _per_type_stats(
        H, bids, energy_kwh, power_kw, elec_price, carbon_intensity,
        grid, ts, ds, cfg, n_total_for_p=p_normaliser, myerson=myerson,
    )
    p              = stats["p"]
    d_bar          = stats["d_bar"]
    v_bar_arr      = stats["v_bar"]
    phi_bar_arr    = stats["phi_bar"]
    bid_high_bar   = stats["bid_high_bar"]
    energy_bar_arr = stats["energy_bar"]
    power_bar_arr  = stats["power_bar"]
    C_elec_bar     = stats["C_elec_bar"]
    C_carbon_bar   = stats["C_carbon_bar"]

    # 7. Bin floors per type for the PAYMENT RULE only
    L = np.zeros(n_types)
    priority_of_type = np.zeros(n_types)
    A_of_type = np.zeros((n_types, n_res))
    for tid in range(n_types):
        k, priority, a_row = ts.decode(tid)
        L[tid] = grid.bnd_v[k - 1]
        priority_of_type[tid] = priority
        A_of_type[tid] = a_row

    phi_tilde = compute_phi_tilde(L, bid_high_bar, cfg.bid.strategy, myerson)

    # 8. Expected resource volume
    v = A_of_type * d_bar[:, None]

    # 9. Raw per-type objectives using PER-TYPE SAMPLE MEANS (see module docstring)
    #    W  = priority * v_bar     (was: priority * L)
    #    Pi = phi_bar - C_elec     (was: phi(L) - C_elec)
    #    C  = C_carbon_bar (kgCO2)
    #    The relative weight of carbon against the other objectives is set
    #    solely by objective.lambda3, applied after normalization.
    lam = cfg.objective
    raw_sat  = priority_of_type * v_bar_arr
    raw_prof = phi_bar_arr - C_elec_bar
    raw_carb = C_carbon_bar

    # 10. Normalization
    V_sat, V_prof, V_carb, norm_ref, norm_deg = _normalize_objectives(
        raw_sat, raw_prof, raw_carb, p, v, ds, cfg,
    )

    # 11. Build objective matrix and scalarize
    # For utopian: V_carb is already inverted (1 - C/z_carb), so add it.
    # For all others: subtract V_carb (lower carbon is better).
    if cfg.normalization.strategy == "utopian":
        f = np.column_stack([
            lam.lambda1 * V_sat,
            lam.lambda2 * V_prof,
            lam.lambda3 * V_carb,
        ])
    else:
        f = np.column_stack([
            lam.lambda1 *  V_sat,
            lam.lambda2 *  V_prof,
            -lam.lambda3 * V_carb,
        ])

    r_rev = lam.lambda1 * V_sat + lam.lambda2 * V_prof

    if cfg.scalarization.method == "chebyshev":
        import warnings
        warnings.warn(
            "Chebyshev scalarization produces reward <= 0 by construction; the "
            "LP-RS will likely reject all types. Use 'linear' for the online "
            "admission reward, or use Chebyshev only in offline Pareto-"
            "enumeration notebooks.",
            RuntimeWarning,
        )

    weights = np.ones(3)
    r_tilde = np.asarray(scalarize(
        f, cfg.scalarization.method, weights, cfg.scalarization.rho
    ))

    # 12. System constants
    r_max      = float(np.max(np.abs(r_tilde[p > 0]))) if np.any(p > 0) else 1.0
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
        r_tilde=r_tilde, r_rev=r_rev,
        C_elec_bar=C_elec_bar, C_carbon_bar=C_carbon_bar,
        v=v, L=L, phi_tilde=phi_tilde,
        tau=tau, lam_star=lp.optimum,
        gamma=gamma, xi=xi, eps_D=eps_D,
        n_types=n_types, norm_ref=norm_ref,
        norm_strategy=cfg.normalization.strategy,
        n_ir_rejected=stats["n_ir_rejected"],
        n_assigned=stats["n_assigned"],
        norm_degenerate=norm_deg,
        n_types_hw_degenerate=n_types_hw_degenerate,
    )
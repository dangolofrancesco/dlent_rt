"""
Phase 0 — Offline Bootstrap

Computes everything the online loop needs to start its first phase:
  - Bid synthesis (lognormal, on-the-fly)
  - Myerson model and valuation grid
  - Energy model (power draw from CPU/RAM/scheduling_class)
  - Grid-data lookup (electricity price + carbon intensity at request_time)
  - Per-type statistics (p, d_bar, v_bar, phi_bar, energy, costs)
  - Carbon price s_C (quantile_linear), additive dollar reward
  - Scalarized reward r_tilde
  - LP-RS -> shadow prices tau* and benchmark lambda*

RAW OBJECTIVES USE PER-TYPE SAMPLE MEANS:
  Raw objectives use PER-TYPE SAMPLE MEANS of v and phi(v), NOT the bin floor
  L. Using phi(L) causes systematically negative raw_prof because bin floors
  are small (geometric grid -> many tiny L values -> phi(L) << 0). Using
  phi_bar = mean(phi(v)) over the jobs in each type reflects the actual job
  values.
  The bin floor L is retained only for the payment rule (phi_tilde).

The output `PhaseState` is the object the online model reads.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import Config
from .data import Dataset, JobArrays
from .grid import ValuationGrid, build_valuation_grid
from .lp_oracle import scalarize, solve_lp_rs, LPResult
from .models.core.error_params import ErrorParams
from .energy import EnergyParams, compute_energy_kwh, compute_power_kw
from .bid import (
    MyersonModel, fit_myerson, bid_model, compute_phi_tilde,
    LognormalBidParams, BidResult, synthesize_lognormal,
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
    params = LognormalBidParams(
        sigma=cfg.bid.sigma,
        base_utility_u0=cfg.bid.base_utility_u0,
        base_utility_mode=cfg.bid.base_utility_mode,
        gamma1=cfg.bid.gamma1,
        gamma2=cfg.bid.gamma2,
        spec_cpu_core=cfg.energy_model.spec_cpu_core,
        spec_ram_gb=cfg.energy_model.spec_ram_gb,
    )
    return synthesize_lognormal(
        jobs.A[:, 0], jobs.A[:, 1],
        jobs.duration_hours, jobs.priority, params, rng,
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
    """type_id = ((k - 1) * |Q| + q_idx) * |A| + hw_idx, vectorised; -1 where k < 1 (legacy phi-space reject)."""
    k_arr = np.asarray(k_arr, dtype=np.int64)
    qi = np.searchsorted(ts.priorities, np.asarray(priority, float)).astype(np.int64)
    tid = ((k_arr - 1) * len(ts.priorities) + qi) * ts.n_hw + np.asarray(hw_idx, np.int64)
    return np.where(k_arr >= 1, tid, -1)


def type_arrays(ts: TypeSpace, grid: ValuationGrid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-type (L = bid-space bin floor, priority, hardware profile A)."""
    n = ts.n_types
    ids = np.arange(n)
    ai = ids % ts.n_hw
    rest = ids // ts.n_hw
    qi = rest % len(ts.priorities)
    ki = rest // len(ts.priorities)
    L = np.asarray(grid.bnd_v)[ki]
    return L, ts.priorities[qi].astype(float), ts.hw_profiles[ai]


def scalarized_reward(
    priority_of_type: np.ndarray, v_bar: np.ndarray, phi_bar: np.ndarray,
    C_elec_bar: np.ndarray, C_carbon_bar: np.ndarray, p: np.ndarray,
    v: np.ndarray, ds, cfg: Config, carbon_price: Optional[float] = None,
):
    """
    Steps 9-11 of run_phase0 factored out so EndPhaseAndReinitialize reuses the
    exact same reward construction. `carbon_price` freezes the quantile_linear
    carbon price s_C (ReInit passes Phase 0's s_C; it is never recomputed on the
    stream). Returns (r_tilde, r_nocarbon, norm_info).
    """
    lam = cfg.objective
    raw_sat  = priority_of_type * v_bar
    raw_prof = phi_bar - C_elec_bar
    raw_carb = C_carbon_bar
    V_sat, V_prof, V_carb, norm_info = _normalize_objectives(
        raw_sat, raw_prof, raw_carb, p, cfg, carbon_price=carbon_price,
    )
    f = np.column_stack([lam.lambda1 * V_sat, lam.lambda2 * V_prof, -lam.lambda3 * V_carb])
    r_nocarbon = lam.lambda1 * V_sat + lam.lambda2 * V_prof
    r_tilde = np.asarray(scalarize(f, cfg.scalarization.method, np.ones(3), cfg.scalarization.rho))
    return r_tilde, r_nocarbon, norm_info


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
    v: np.ndarray                         # (n_types, |I|) expected resource volume
    L: np.ndarray                         # bid-space bin floor (for payment rule)
    phi_tilde: np.ndarray                 # phi(L): virtual value at floor (payment rule);
                                          # NaN under known_distribution (per job: phi_job(L, c_j))

    # LP-RS output
    tau: np.ndarray
    lam_star: float

    # system constants
    gamma: float
    xi: float
    eps_D: float

    n_types: int
    norm_strategy: str = "quantile_linear"
    n_ir_rejected: int = 0
    n_assigned: int = 0
    n_types_hw_degenerate: int = 0
    # $/kgCO2 applied to C_carbon_bar: s_C (quantile_linear), frozen at Phase 0
    carbon_price: float = 0.0             # always overwritten in run_phase0
    norm_info: Optional[dict] = None      # quantile_linear: q, Q_sw, Q_abs_pi, D_q, K_q, s_C



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
    n_total_for_p: int, myerson: MyersonModel, phi_job: np.ndarray,
) -> dict:
    """
    Compute per-type empirical statistics.

    v_bar and phi_bar are SAMPLE MEANS over the jobs assigned to each type
    (actual job values), not the discretization bin floor. `phi_job` is the
    per-job virtual value the type reward averages (see run_phase0, step 2).
    """
    n_types = ts.n_types

    k_arr = np.asarray(grid.assign(bids.v_rate, myerson))
    valid = k_arr >= 1
    n_ir_rejected = int((~valid).sum())
    type_ids = assign_type_ids(ts, k_arr, jobs.priority, jobs.hw_idx)

    # Vectorised per-type sample means (bincount). Numerically equivalent to
    # the previous per-type mask loop (O(n * n_types)); see tests/test_phase0_refactor.py.
    tid = type_ids[valid]
    N = np.bincount(tid, minlength=n_types).astype(np.int64)

    def _mean(x: np.ndarray) -> np.ndarray:
        s = np.bincount(tid, weights=np.asarray(x, float)[valid], minlength=n_types)
        return np.divide(s, N, out=np.zeros(n_types), where=N > 0)

    d_bar        = _mean(jobs.duration_steps)
    v_bar        = _mean(bids.v_rate)            # mean bid per type
    phi_bar      = _mean(phi_job)                # mean phi(v) per type
    energy_bar   = _mean(energy_kwh)
    power_bar    = _mean(power_kw)
    C_elec_bar   = _mean(energy_kwh * elec_price)
    # FIX (week 28): emissions per job = E [kWh] * CI [kgCO2/kWh] -> kgCO2.
    # Previously this averaged carbon_intensity alone (the energy factor was
    # commented out), so C_carbon did not scale with job size or duration.
    C_carbon_bar = _mean(energy_kwh * carbon_intensity)

    p = N / max(n_total_for_p, 1)
    return dict(N=N, p=p, d_bar=d_bar, v_bar=v_bar, phi_bar=phi_bar,
                energy_bar=energy_bar, power_bar=power_bar,
                C_elec_bar=C_elec_bar, C_carbon_bar=C_carbon_bar,
                type_ids=type_ids, n_ir_rejected=n_ir_rejected,
                n_assigned=int(valid.sum()))



# Normalization
def _weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    """Weighted quantile: atoms sorted by x, each placed at the midpoint of its
    cumulative weight, (cumsum(w) - w/2) / sum(w), then linear interpolation."""
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    order = np.argsort(x, kind="stable")
    x, w = x[order], w[order]
    cw = (np.cumsum(w) - 0.5 * w) / w.sum()
    return float(np.interp(q, cw, x))


def carbon_scale_from_quantile(
    raw_sat: np.ndarray, raw_prof: np.ndarray, raw_carb: np.ndarray,
    p: np.ndarray, q: float,
) -> dict:
    """
    quantile_linear carbon price, over populated types (p > 0) weighted by p:
      D_q = (Q_q(SW) + Q_q(|Pi|)) / 2,  K_q = Q_q(K),  s_C = D_q / K_q  [$/kgCO2],
    so the linear map through the origin K -> s_C * K sends [0, K_q] onto [0, D_q].
    """
    m = np.asarray(p) > 0
    w = np.asarray(p, float)[m]
    Q_sw     = _weighted_quantile(np.asarray(raw_sat)[m], w, q)
    Q_abs_pi = _weighted_quantile(np.abs(np.asarray(raw_prof)[m]), w, q)
    K_q      = _weighted_quantile(np.asarray(raw_carb)[m], w, q)
    D_q = 0.5 * (Q_sw + Q_abs_pi)
    if K_q <= 0 or D_q <= 0:
        import warnings
        warnings.warn(
            f"quantile_linear: degenerate scale (D_q={D_q:.4g}, K_q={K_q:.4g}); "
            f"setting s_C = 0 (no carbon term).",
            RuntimeWarning,
        )
        s_C = 0.0
    else:
        s_C = D_q / K_q
    return dict(q=float(q), Q_sw=Q_sw, Q_abs_pi=Q_abs_pi, D_q=D_q, K_q=K_q, s_C=float(s_C))


def _normalize_objectives(
    raw_sat: np.ndarray, raw_prof: np.ndarray, raw_carb: np.ndarray,
    p: np.ndarray, cfg: Config, carbon_price: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    quantile_linear (the only normalization): carbon is mapped to dollars
    linearly through the origin, C$ = s_C * K, with s_C = D_q / K_q chosen so
    the q-quantile of K lands on the mean of the q-quantiles of SW and |Pi|
    (carbon_scale_from_quantile). No divisor: the reward stays in dollars.
    s_C is computed once on Phase 0's H; `carbon_price` freezes it (ReInit
    passes Phase 0's, it is never recomputed on the stream).
    Returns (V_sat, V_prof, V_carb, norm_info).
    """
    if carbon_price is None:
        norm_info = carbon_scale_from_quantile(
            raw_sat, raw_prof, raw_carb, p, cfg.normalization.quantile)
        s_C = norm_info["s_C"]
    else:
        s_C = float(carbon_price)
        norm_info = dict(q=cfg.normalization.quantile, s_C=s_C, frozen=True)
    norm_info = dict(strategy="quantile_linear", **norm_info)
    return raw_sat, raw_prof, raw_carb * s_C, norm_info


# Phase 0 entry point
def run_phase0(ds: Dataset, cfg: Config) -> PhaseState:
    H = ds.H

    # 1. Synthesize bids on-the-fly
    bids = _synthesize_bids(H, cfg, seed=cfg.run.seed)

    # 2. Bid model. known_distribution (default): the markup distribution is
    #    known (sigma = bid.sigma, never fit) and the per-job phi is the
    #    mechanism's phi_job(v, c_j), the same function ReInit and the online
    #    reward use. known_distribution=false (ablation): pooled MLE fit, and the
    #    type reward keeps the legacy generator phi (bids.phi_rate).
    myerson = bid_model(cfg, bids.v_rate)
    phi_job = (np.asarray(myerson.phi_job(bids.v_rate, bids.cost_base))
               if myerson.known else bids.phi_rate)

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
        phi_job=phi_job,
    )
    p              = stats["p"]
    d_bar          = stats["d_bar"]
    v_bar_arr      = stats["v_bar"]
    phi_bar_arr    = stats["phi_bar"]
    energy_bar_arr = stats["energy_bar"]
    power_bar_arr  = stats["power_bar"]
    C_elec_bar     = stats["C_elec_bar"]
    C_carbon_bar   = stats["C_carbon_bar"]

    # 7. Bin floors per type for the PAYMENT RULE only
    L, priority_of_type, A_of_type = type_arrays(ts, grid)
    phi_tilde = compute_phi_tilde(L, myerson)

    # 8. Expected resource volume
    v = A_of_type * d_bar[:, None]

    # 9-11. Raw objectives (per-type SAMPLE MEANS), normalization, scalarization
    if cfg.scalarization.method == "chebyshev":
        import warnings
        warnings.warn(
            "Chebyshev scalarization produces reward <= 0 by construction; the "
            "LP-RS will likely reject all types. Use 'linear' for the online "
            "admission reward, or use Chebyshev only in offline Pareto-"
            "enumeration notebooks.",
            RuntimeWarning,
        )
    r_tilde, r_nocarbon, norm_info = scalarized_reward(
        priority_of_type, v_bar_arr, phi_bar_arr, C_elec_bar, C_carbon_bar,
        p, v, ds, cfg,
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
                 else _weighted_quantile(abs_r, p[p > 0], q_r))
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
        v=v, L=L, phi_tilde=phi_tilde,
        tau=tau, lam_star=lp.optimum,
        gamma=gamma, xi=xi, eps_D=eps_D,
        n_types=n_types,
        n_ir_rejected=stats["n_ir_rejected"],
        n_assigned=stats["n_assigned"],
        n_types_hw_degenerate=n_types_hw_degenerate,
        carbon_price=float(carbon_price), norm_info=norm_info,
    )

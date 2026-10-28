"""
Configuration layer for DLENT-RT.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional

import yaml


# Sub-configs
@dataclass(frozen=True)
class RunCfg:
    name: str = "default"
    seed: int = 42
    output_dir: str = "runs"


@dataclass(frozen=True)
class ColumnsCfg:
    collection_id: str = "collection_id"
    datetime: str = "arrival_utc"          # with its time zone; converted to naive UTC (the grid's time base)
    priority: str = "priority_tier"
    scheduling_class: str = "scheduling_class"
    requested_cpu: str = "CPU"             # SUM of the instances' first requests (machines)
    requested_ram: str = "RAM"
    duration_hours: str = "duration"       # job span (first start to last end); used where CPU is 0
    cpu_hours_real: str = "CPU_hours_real" # effective duration = cpu_hours_real / requested_cpu


@dataclass(frozen=True)
class DataCfg:
    batch_csv: str = "data/batch_v2_30k.csv"   # scripts/data/fetch_v2.py + scripts/data/build_v2.py
    grid_csv: str = "data/grid_data_may_2019.csv"
    n0: int = 10000
    columns: ColumnsCfg = field(default_factory=ColumnsCfg)
    # H = the jobs that arrive in the first h_days days (None = the first n0 jobs)
    h_days: Optional[float] = None


@dataclass(frozen=True)
class EnergyModelCfg:
    # DECLARED ASSUMPTION (not from Google): the size of the largest machine of the trace, i.e. of 1.0 in the
    # trace's units. Google publishes only the normalised values. Used only where cores / GB are needed (bids,
    # energy). Requests and capacity stay in machines.
    machine_cores: float = 64       # cores of the largest machine (1.0 CPU)
    machine_ram_gb: float = 256     # GB of the largest machine (1.0 RAM)
    p_core: float = 15.0            # watts per physical core
    p_gb: float = 0.35              # watts per GB RAM
    pue: float = 1.10


@dataclass(frozen=True)
class NormalizationCfg:
    # quantile_linear (the only normalization): q of the weighted quantiles
    # Q_q(CV), Q_q(|Pi|), Q_q(K) that set the carbon price s_C = D_q / K_q
    # (frozen at Phase 0). The reward stays in dollars (no divisor).
    quantile: float = 0.90

@dataclass(frozen=True)
class BidCfg:
    # bid model (the only one): ln v = ln cost_base - sigma^2/2 + sigma Z, Z ~ N(0, 1)
    # Base utility rate ($/hour): minimum hourly value a user derives from
    # job completion, independent of resource footprint. Multiplied by
    # duration. Ref: reinterpretation suggested by advisor — demand-side
    # utility rather than supply-side fee.
    # U_0: the single free parameter of base_utility_mode. Calibrated via
    # scripts/calibrate_u0.py (bisection on Phase 0's pos_reward_types,
    # target_pos_frac=0.90) at sigma=0.3: if sigma (or gamma1/gamma2) is
    # changed, this value no longer hits that 90% target; recalibrate.
    base_utility_u0: float = 23.926
    # How U_base(q) scales with priority q in {1,...,5}:
    #   "linear" -> U_base(q) = u0 * q
    #   "log"    -> U_base(q) = u0 * (1 + ln(q))
    base_utility_mode: str = "log"
    # Reserve prices ($/core-hour, $/GB-hour) in cost_base. CALIBRATED values, not
    # list prices: they are ~3.3x the 2024 GCP Preemptible n1-standard prices
    # (~$0.00996/core-hour, ~$0.00135/GB-hour), i.e. the same order of magnitude.
    # base_utility_u0 was calibrated WITH these values (scripts/calibrate_u0.py):
    # changing gamma1/gamma2 requires recalibrating base_utility_u0.
    gamma1: float = 0.0332          # $/core-hour reserve price
    gamma2: float = 0.00445         # $/GB-hour reserve price
    # lognormal params
    sigma: float = 0.3              # bid dispersion (log-space std of the markup)
    # true  -> the bid distribution is KNOWN a priori: markup v / cost_base ~
    #          LogNormal(-sigma^2/2, sigma) with sigma above; phi_v(v | c) = c * phi_m(v / c).
    #          Never refit (Phase 0, ReInit, online). Requires grid.space="v".
    # false -> legacy ablation: pooled lognormal MLE on the bids (refit at every ReInit).
    known_distribution: bool = True
    


@dataclass(frozen=True)
class TimeCfg:
    tick_seconds: float = 1.0           # wall-clock length of one step (real-tick time model)
    # Durations longer than d_max are cut (not removed); the one capped duration is used for the time in the
    # cluster, the bids and the energy. d_max = min(d_max_hours, d_max_horizon_fraction * T); None -> the ceiling.
    d_max_hours: Optional[float] = None
    d_max_horizon_fraction: float = 0.05
    d_max_ratio_abort: float = 0.10
    # Minimum job duration for allocation purposes. Jobs shorter than this
    # are treated as lasting this long, matching cloud billing minimums
    # (GCP/AWS bill 1 min minimum) and preventing near-zero resource
    # volumes from inflating gamma.
    duration_floor_hours: float = 0.016667    # ~1 minute (the YAML value in use, not exactly 1/60)


@dataclass(frozen=True)
class HardwareCfg:
    # Hardware profiles build ONLY the type id (hw_idx). The request A of a job is always its real request, and the
    # type volume v = A_type * d_bar uses the mean real request of the type's jobs (phase0.type_request).
    n_profiles: int = 32
    # profiles = k-means++ centroids, in log space if log_space (the only method)
    log_space: bool = True
    fit_on: str = "full"            # "full" | "H"
    # Minimum value for any hardware profile component (CPU or RAM).
    # Centroids from k-means clustering can occasionally collapse a
    # component near zero on heavy-tailed resource-request distributions;
    # such degenerate profiles produce near-zero expected resource volume
    # v_ij, which inflates gamma's r_max/v_min term by orders of magnitude
    # when used in the system-constant calculation. This is NOT a filter
    # on which types are served -- only on which types contribute to
    # gamma's v_min term.
    hw_floor: float = 1.0e-5


@dataclass(frozen=True)
class LimitsCfg:
    # Jobs with a CPU or RAM request larger than r_max_machines are removed (None = no limit).
    r_max_machines: Optional[float] = 0.1
    xi_warn_threshold: float = 0.10     # warn if the data xi = max_i a_max,i / c_i exceeds this
    # Duration cap (None = off): d_j = min(effective duration, d_cap_hours); jobs are kept, only shortened. When set,
    # d_max = d_cap_hours (the time.d_max_hours / horizon-ceiling rule is not used; the abort check still applies).
    d_cap_hours: Optional[float] = None
    # Job quota (None = off): a job with a CPU request > job_quota_machines OR a RAM request > job_quota_mem
    # (machines) is rejected before the market: it never enters Phase 0 or the online run.
    job_quota_machines: Optional[float] = None
    job_quota_mem: Optional[float] = None
    # "fixed": the two values above. "share_of_C": Q_i = quota_share * C_i, C = capacity.f * D_bar on H after the
    # quota, found by a fixed point that starts from Q = the 99th percentile of H's requests and stops when Q moves
    # by < 1% on both resources (needs capacity.mode mean_demand and d_cap_hours).
    job_quota_rule: str = "fixed"
    quota_share: float = 0.2


@dataclass(frozen=True)
class AbsoluteCapCfg:
    cpu: Optional[float] = None
    ram: Optional[float] = None


@dataclass(frozen=True)
class CapacityCfg:
    mode: str = "exceeded_share"   # "exceeded_share" | "mean_demand" | "absolute"
    # exceeded_share: c_i = the demand level of resource i exceeded this share of the time when every job is
    # admitted (time-weighted, on the same A and duration_steps the model uses).
    exceeded_share: float = 0.25
    absolute: AbsoluteCapCfg = field(default_factory=AbsoluteCapCfg)   # mode "absolute": c in machines
    # mean_demand: C_i = f * D_bar_i, D_bar_i = lambda_hat * mean_j(r_ij d_j) = sum_{j in H} r_ij d_j / |H|
    # (machines; on H only, after the limits, with the capped durations; |H| = H's span in hours)
    f: Optional[float] = None


@dataclass(frozen=True)
class ObjectiveCfg:
    lambda1: float = 1.0
    lambda2: float = 1.0
    lambda3: float = 1.0


@dataclass(frozen=True)
class GridCfg:
    k_star: int = 128
    phi_floor: float = 1.0e-6
    priority_classes: tuple = (1, 2, 3, 4, 5)
    # WHICH SPACE the valuation grid is built in:
    #   "v"          -> grid over bids directly. Always positive, so EVERY job
    #                   receives a type; rejection is left to the oracle, which
    #                   uses the full multi-objective reward. RECOMMENDED.
    #   "phi"        -> grid over Myerson virtual values (legacy). Cannot span
    #                   phi<=0, so jobs with negative virtual value are dropped
    #                   upstream of the optimisation.
    #   "phi_shifted"-> grid over (phi - min(phi) + eps): keeps phi ordering but
    #                   makes all values positive, so nothing is dropped.
    space: str = "v"
    # bin edges are always geometric (multiplicative) in that space


@dataclass(frozen=True)
class ConfidenceCfg:
    delta: Optional[float] = None
    delta_power: int = 3


@dataclass(frozen=True)
class ModelCfg:
    kind: str = "theory"          # only "theory" (the practical model is not implemented)


@dataclass(frozen=True)
class ScalarizationCfg:
    # "linear" (default) -> weighted sum; the reward keeps an absolute sign, which
    #                       is what the online LP-RS admission oracle expects.
    # "chebyshev"        -> augmented Chebyshev distance-to-ideal. Still available,
    #                       but it produces reward <= 0 by construction, so as a
    #                       direct LP-RS reward it makes the oracle admit nothing.
    #                       Intended for offline Pareto-front enumeration in the
    #                       analysis notebooks, NOT for the online loop.
    method: str = "linear"
    rho: float = 1.0e-3


@dataclass(frozen=True)
class OracleCfg:
    kind: str = "frozen"          # only "frozen" (phase-averaged costs, preserves Lemma 1)


@dataclass(frozen=True)
class LpCfg:
    presolve: bool = True


@dataclass(frozen=True)
class Test3Cfg:
    s_grid: str = "geometric"     # "geometric" | "full"
    geometric_base: int = 2


@dataclass(frozen=True)
class PhaseTransitionCfg:
    n_type_min: int = 5
    n_min: int = 80


@dataclass(frozen=True)
class ConstantsCfg:
    # System constants of Zhang & Cheung (2025, Appendix A). null -> the
    # data-derived value from Phase 0. The defaults are FORCED, NOT
    # DATA-JUSTIFIED: the data-derived gamma/xi make every threshold vacuous
    # and p_U ~ 0 (week-28 notes); the forced values are an open decision. When xi is forced,
    # Test 1 uses the implied a_max,i = xi * c_i so that xi = max_i a_max,i/c_i
    # still holds; the physical gate always uses the true A of each job.
    xi_override: Optional[float] = 0.001
    gamma_override: Optional[float] = 1.0
    # r_max entering the data-derived gamma: p-weighted quantile of |r~| over
    # populated types. 1.0 = max (original definition).
    r_max_quantile: float = 0.8


@dataclass(frozen=True)
class TestsCfg:
    # Exact-LP variant (week 17): shadow prices come from the exact LP-RS, not
    # from the stochastic dual-learning Algorithm 2, so the dual-learning error
    # terms eps_E^(m) (A.8) and eps_F^(m) (A.9) are set to 0 in p_U and Test 2.
    # All other Appendix-A terms are kept (paper-faithful).
    drop_dual_learning_terms: bool = True
    # relative tolerance of the Eisner-Severance parametric-LP reconstruction
    piecewise_tol: float = 1e-7


@dataclass(frozen=True)
class OnlineCfg:
    bid_seed_offset: int = 1        # stream bids: rng(run.seed + offset)
    thinning_seed_offset: int = 2   # Bernoulli U(t): rng(run.seed + offset)


@dataclass(frozen=True)
class AblationCfg:
    """
    Ablations of the theory model (week 30, round 1). Every default is OFF: with all of them off the model is the
    theory run, bit-identical. They change only the quantity named here.
      thinning_off           A1a: p_U(t) = 1 (no Bernoulli thinning); the oracle and the physical gate stay.
      thinning_delta_power   A1b: eps_D INSIDE p_U is computed with delta = 1/T^k (None = the run's delta);
                             the tests keep the run's delta (confidence.*).
      thinning_eps_D_zero    A1c: eps_D = 0 INSIDE p_U only (cap (1 - xi)); Test 2 keeps 1 - eps_D - xi.
      fixed_grid             A3: ReInit never refits the valuation grid; every phase keeps the previous phase's
                             grid, i.e. Phase 0's (the per-type statistics are still re-estimated).
    """
    thinning_off: bool = False
    thinning_delta_power: Optional[int] = None
    thinning_eps_D_zero: bool = False
    fixed_grid: bool = False


@dataclass(frozen=True)
class LoggingCfg:
    per_step: bool = True
    per_phase: bool = True
    short_phase_log: bool = True



# Top-level config
@dataclass(frozen=True)
class Config:
    run: RunCfg = field(default_factory=RunCfg)
    data: DataCfg = field(default_factory=DataCfg)
    time: TimeCfg = field(default_factory=TimeCfg)
    hardware: HardwareCfg = field(default_factory=HardwareCfg)
    limits: LimitsCfg = field(default_factory=LimitsCfg)
    capacity: CapacityCfg = field(default_factory=CapacityCfg)
    energy_model: EnergyModelCfg = field(default_factory=EnergyModelCfg)
    bid: BidCfg = field(default_factory=BidCfg)
    normalization: NormalizationCfg = field(default_factory=NormalizationCfg)
    objective: ObjectiveCfg = field(default_factory=ObjectiveCfg)
    grid: GridCfg = field(default_factory=GridCfg)
    confidence: ConfidenceCfg = field(default_factory=ConfidenceCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    scalarization: ScalarizationCfg = field(default_factory=ScalarizationCfg)
    oracle: OracleCfg = field(default_factory=OracleCfg)
    lp: LpCfg = field(default_factory=LpCfg)
    test3: Test3Cfg = field(default_factory=Test3Cfg)
    phase_transition: PhaseTransitionCfg = field(default_factory=PhaseTransitionCfg)
    logging: LoggingCfg = field(default_factory=LoggingCfg)
    constants: ConstantsCfg = field(default_factory=ConstantsCfg)
    tests: TestsCfg = field(default_factory=TestsCfg)
    online: OnlineCfg = field(default_factory=OnlineCfg)
    ablation: AblationCfg = field(default_factory=AblationCfg)

    #  validation 
    def validate(self) -> "Config":
        assert self.time.tick_seconds > 0
        assert self.model.kind == "theory", "model.kind: only 'theory' is implemented"
        assert self.scalarization.method in ("chebyshev", "linear")
        assert self.oracle.kind == "frozen", "oracle.kind: only 'frozen' is implemented"
        assert self.capacity.mode in ("exceeded_share", "mean_demand", "absolute")
        if self.capacity.mode == "mean_demand":
            assert self.capacity.f is not None and self.capacity.f > 0
        for x in (self.limits.d_cap_hours, self.limits.job_quota_machines, self.limits.job_quota_mem, self.data.h_days):
            assert x is None or x > 0
        assert self.limits.job_quota_rule in ("fixed", "share_of_C")
        if self.limits.job_quota_rule == "share_of_C":
            assert self.capacity.mode == "mean_demand" and self.limits.d_cap_hours is not None
            assert 0 < self.limits.quota_share <= 1
        assert 0.0 <= self.capacity.exceeded_share < 1.0
        assert self.time.d_max_hours is None or self.time.d_max_hours > 0
        assert self.limits.r_max_machines is None or self.limits.r_max_machines > 0
        ab = self.ablation
        assert ab.thinning_delta_power is None or int(ab.thinning_delta_power) >= 1
        assert not (ab.thinning_eps_D_zero and ab.thinning_delta_power is not None), \
            "ablation: thinning_eps_D_zero and thinning_delta_power both set the thinning eps_D"
        assert self.grid.k_star >= 2
        assert 0.0 < self.time.d_max_horizon_fraction <= 1.0
        assert self.hardware.fit_on in ("full", "H")
        assert self.hardware.n_profiles >= 1
        assert self.grid.space in ("v", "phi", "phi_shifted")
        if self.bid.known_distribution:
            assert self.grid.space == "v", \
                "bid.known_distribution=true requires grid.space='v' (phi depends on c_j)"
            assert self.bid.sigma > 0
        assert self.bid.base_utility_mode in ("linear", "log")
        assert 0.0 < self.normalization.quantile < 1.0
        assert self.energy_model.machine_cores > 0
        assert self.energy_model.machine_ram_gb > 0
        assert self.test3.s_grid in ("geometric", "full")
        assert self.test3.geometric_base >= 2
        if self.constants.xi_override is not None:
            assert 0.0 < self.constants.xi_override < 1.0
        if self.constants.gamma_override is not None:
            assert self.constants.gamma_override > 0.0
        assert 0.0 < self.constants.r_max_quantile <= 1.0
        if self.capacity.mode == "absolute":
            assert self.capacity.absolute.cpu is not None
            assert self.capacity.absolute.ram is not None
        return self



# Recursive dict -> dataclass with strict unknown-key checking
def _from_dict(cls: type, data: dict[str, Any], path: str = "") -> Any:
    if not is_dataclass(cls):
        return data
    kwargs: dict[str, Any] = {}
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(
            f"Unknown config key(s) at '{path or 'root'}': {sorted(unknown)}. "
            f"Allowed: {sorted(known)}"
        )
    for name, f in known.items():
        if name not in data:
            continue  # use dataclass default
        val = data[name]
        ftype = f.type
        # resolve nested dataclasses
        nested_cls = _resolve_dataclass_type(ftype)
        if nested_cls is not None and isinstance(val, dict):
            kwargs[name] = _from_dict(nested_cls, val, f"{path}.{name}".lstrip("."))
        elif name == "priority_classes" and isinstance(val, list):
            kwargs[name] = tuple(val)
        else:
            kwargs[name] = val
    return cls(**kwargs)


def _resolve_dataclass_type(ftype: Any) -> Optional[type]:
    """Return the dataclass type if ftype is (or names) one, else None."""
    if is_dataclass(ftype):
        return ftype
    # dataclass field types may arrive as strings under `from __future__ import annotations`
    if isinstance(ftype, str):
        return _DATACLASS_REGISTRY.get(ftype)
    return None


_DATACLASS_REGISTRY = {
    "RunCfg": RunCfg, "ColumnsCfg": ColumnsCfg, "DataCfg": DataCfg,
    "TimeCfg": TimeCfg, "HardwareCfg": HardwareCfg, "LimitsCfg": LimitsCfg,
    "AbsoluteCapCfg": AbsoluteCapCfg, "CapacityCfg": CapacityCfg,
    "ObjectiveCfg": ObjectiveCfg, "GridCfg": GridCfg, "ConfidenceCfg": ConfidenceCfg,
    "ModelCfg": ModelCfg, "ScalarizationCfg": ScalarizationCfg, "OracleCfg": OracleCfg,
    "LpCfg": LpCfg, "Test3Cfg": Test3Cfg, "PhaseTransitionCfg": PhaseTransitionCfg,
    "LoggingCfg": LoggingCfg, "EnergyModelCfg": EnergyModelCfg,
    "BidCfg": BidCfg, "NormalizationCfg": NormalizationCfg,
    "ConstantsCfg": ConstantsCfg, "TestsCfg": TestsCfg, "OnlineCfg": OnlineCfg,
    "AblationCfg": AblationCfg,
}


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config into a frozen Config tree."""
    path = Path(path)
    with path.open("r") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = _from_dict(Config, raw)
    return cfg.validate()


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def resolve_path(path: str | Path) -> Path:
    """Data paths in the config are relative to the repository root (absolute paths are kept)."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def default_config() -> Config:
    """
    The project default configuration.

    Loads ``configs/default.yaml`` when it is present (the repo layout). The
    dataclass defaults are kept EQUAL to that file (tests/test_config_defaults.py),
    so the fallback used when the YAML is missing gives the same configuration.
    """
    if DEFAULT_CONFIG_PATH.is_file():
        return load_config(DEFAULT_CONFIG_PATH)
    return Config().validate()

"""
Configuration layer for DLENT-RT. One configuration: configs/default.yaml = the final model (week 33). The model
variants are switches inside it:

    theoretical model   oracle.kind frozen, ablation.thinning_off false, bid.refit_sigma true   (the default)
    practical model     oracle.kind live,   ablation.thinning_off true
    no-learning model   the Phase 0 state forever (models.policy.ThresholdPolicy with thinning); Phase 0's sigma
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
    batch_csv: str = "data/batch_v2_300k.csv"            # scripts/data/fetch_v2.py + scripts/data/build_v2.py
    grid_csv: str = "data/grid_data_may_2019_filled.csv"  # 1-3 June filled by scripts/data/fill_grid.py
    columns: ColumnsCfg = field(default_factory=ColumnsCfg)
    h_days: float = 7.0                                   # H = the jobs that arrive in the first h_days days


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
    # The carbon map, the only normalization: s_C = D_q / K_q with D_q = (Q_q(V) + Q_q(|Pi|)) / 2, K_q = Q_q(K),
    # p-weighted quantiles over H's types, computed once in Phase 0. The reward stays in dollars (no divisor).
    quantile: float = 0.90


@dataclass(frozen=True)
class BidCfg:
    # HOURLY bid: cost_base_j = U0 (1 + ln q_j) + gamma1 cores_j + gamma2 GB_j [$/h] (no duration),
    # b_j ~ lognormal with mean cost_base_j [$/h].
    # U0 ($ per hour per job) = gamma1 cores + gamma2 GB of the median H job by that rate (bid.calibrate_u0, Phase 0).
    gamma1: float = 0.0332          # $ per vCPU-hour
    gamma2: float = 0.00445         # $ per GiB-hour
    sigma: float = 0.3              # log-space std of the markup b / cost_base (the generator)
    # true (theoretical model): the mechanism's sigma is refit on the markups (H in Phase 0, the previous phase's
    # arrivals at every ReInit). false (no-learning model): the mechanism keeps sigma above, never refit.
    refit_sigma: bool = True


@dataclass(frozen=True)
class TimeCfg:
    tick_seconds: float = 1.0           # wall-clock length of one step (real-tick time model)
    # Effective durations are cut at d_cap_hours (jobs kept, only shortened) and floored at duration_floor_hours.
    # The one capped duration is used for the time in the cluster, the bids and the energy. d_max = d_cap_hours.
    d_cap_hours: float = 24.0
    duration_floor_hours: float = 0.016667    # 1 minute (cloud billing minimum)
    d_max_ratio_abort: float = 0.10           # abort if d_max / T exceeds this (the theory needs d_max << T)


@dataclass(frozen=True)
class HardwareCfg:
    # Hardware profiles build ONLY the type id (hw_idx). The request A of a job is always its real request, and the
    # type volume v = A_type * d_bar uses the column's mean real request and mean duration (phase0.column_means).
    n_profiles: int = 32
    # profiles = k-means++ centroids, in log space if log_space (the only method)
    log_space: bool = True
    fit_on: str = "full"            # "full" | "H"
    # Minimum hardware-profile component that may enter gamma's v_min term (degenerate k-means centroids are
    # left out of that term only; it is NOT a filter on which types are served).
    hw_floor: float = 1.0e-5


@dataclass(frozen=True)
class LimitsCfg:
    # Job quota: a job with a CPU request > Q_cpu or a RAM request > Q_ram is rejected before the market. Q_i =
    # quota_share x C_i of the cluster actually used (capacity.f), by a fixed point on H that starts from the 99th
    # percentile of H's requests and stops when Q moves by < quota_tol on both resources.
    quota_share: float = 0.2
    quota_tol: float = 0.01
    xi_warn_threshold: float = 0.10     # warn if the data xi = max_i a_max,i / c_i exceeds this


@dataclass(frozen=True)
class CapacityCfg:
    # C_i = f * D_bar_i, D_bar_i = sum_{j in H} r_ij d_j / |H| (machines; H only, after the quota, capped durations).
    # ONE cluster for every model.
    f: float = 0.5
    # Job-level enlargement (off by default): if a job-level model rejects more than job_level_reject_max of the
    # positive-reward jobs for lack of space (benchmarks.job_level_space_check), C is multiplied by
    # job_level_enlarge_step until it does not. The quota and the jobs stay those of f.
    job_level_enlarge: bool = False
    job_level_reject_max: float = 0.5
    job_level_enlarge_step: float = 1.25


@dataclass(frozen=True)
class ObjectiveCfg:
    lambda1: float = 1.0
    lambda2: float = 1.0
    lambda3: float = 1.0


@dataclass(frozen=True)
class GridCfg:
    # Geometric grid on HOURLY bids: L_k = v_min rho^k, rho = (v_max / v_min)^(1 / k_star), v_min / v_max = the smallest /
    # largest bid. Types = (bid bin, priority, hardware profile).
    k_star: int = 128
    priority_classes: tuple = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class ConfidenceCfg:
    delta: Optional[float] = None
    delta_power: int = 3


@dataclass(frozen=True)
class OracleCfg:
    # "frozen" (theoretical model): r~ and tau of the phase (Lemma 1). "live" (practical model): the type's value
    # terms with the job's own electricity cost and carbon at arrival.
    kind: str = "frozen"


@dataclass(frozen=True)
class LpCfg:
    presolve: bool = True


@dataclass(frozen=True)
class Test3Cfg:
    s_grid: str = "geometric"     # the exact event-driven monitor supports only the geometric grid
    geometric_base: int = 2


@dataclass(frozen=True)
class PhaseTransitionCfg:
    n_min: int = 80


@dataclass(frozen=True)
class ConstantsCfg:
    # System constants of Zhang & Cheung (2025, Appendix A). null -> the data-derived value from Phase 0. The
    # defaults are FORCED, NOT DATA-JUSTIFIED (the data gamma / xi make every threshold vacuous). When xi is forced,
    # Test 1 uses a_max,i = xi * c_i; the physical gate always uses the true A of each job.
    xi_override: Optional[float] = 0.001
    gamma_override: Optional[float] = 1.0
    # The p-weighted quantile of |r~| over H's types that enters the DATA gamma only. (The reward unit of Tests 2-3 is
    # the maximum positive r~ of Phase 0, phase0.reward_unit.)
    r_max_quantile: float = 0.8


@dataclass(frozen=True)
class TestsCfg:
    # Exact LP-RS in place of the dual-learning Algorithm 2: eps_E (A.8) and eps_F (A.9) are 0.
    drop_dual_learning_terms: bool = True
    piecewise_tol: float = 1e-7   # relative tolerance of the parametric-LP reconstruction


@dataclass(frozen=True)
class OnlineCfg:
    bid_seed_offset: int = 1        # stream bids: rng(run.seed + offset)
    thinning_seed_offset: int = 2   # Bernoulli U(t): rng(run.seed + offset)


@dataclass(frozen=True)
class AblationCfg:
    # thinning_off (practical model): p_U(t) = 1; the oracle and the physical gate stay.
    thinning_off: bool = False


@dataclass(frozen=True)
class LoggingCfg:
    per_step: bool = True
    per_phase: bool = True
    short_phase_log: bool = True
    # progress lines on stdout during an online run: one every progress_every of the arrivals (0 = off) and one at
    # every phase end, each with the elapsed time and an estimate of the time left
    progress_every: float = 0.10
    progress_phases: bool = True


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
        assert self.time.d_cap_hours > 0 and self.time.duration_floor_hours > 0
        assert self.data.h_days > 0
        assert self.oracle.kind in ("frozen", "live")
        assert self.capacity.f > 0
        assert 0 < self.capacity.job_level_reject_max < 1 and self.capacity.job_level_enlarge_step > 1
        assert 0 < self.limits.quota_share <= 1 and 0 < self.limits.quota_tol < 1
        assert self.grid.k_star >= 2
        assert self.hardware.fit_on in ("full", "H")
        assert self.hardware.n_profiles >= 1
        assert self.bid.sigma > 0
        assert 0.0 < self.normalization.quantile < 1.0
        assert self.energy_model.machine_cores > 0
        assert self.energy_model.machine_ram_gb > 0
        assert self.test3.s_grid == "geometric"
        assert self.test3.geometric_base >= 2
        if self.constants.xi_override is not None:
            assert 0.0 < self.constants.xi_override < 1.0
        if self.constants.gamma_override is not None:
            assert self.constants.gamma_override > 0.0
        assert 0.0 < self.constants.r_max_quantile <= 1.0
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
    "CapacityCfg": CapacityCfg, "ObjectiveCfg": ObjectiveCfg, "GridCfg": GridCfg,
    "ConfidenceCfg": ConfidenceCfg, "OracleCfg": OracleCfg,
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

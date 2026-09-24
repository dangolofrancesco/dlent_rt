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
    log_level: str = "INFO"


@dataclass(frozen=True)
class ColumnsCfg:
    collection_id: str = "collection_id"
    datetime: str = "request_time"
    priority: str = "priority"
    scheduling_class: str = "scheduling_class"
    requested_cpu: str = "CPU"
    requested_ram: str = "RAM"
    duration_hours: str = "duration"


@dataclass(frozen=True)
class DataCfg:
    batch_csv: str = "data/batch_may2019_30k.csv"
    grid_csv: str = "data/grid_data_may_2019.csv"
    n0: int = 5000
    columns: ColumnsCfg = field(default_factory=ColumnsCfg)


@dataclass(frozen=True)
class EnergyModelCfg:
    spec_cpu_core: float = 64       # cores per 1.0 normalized CPU
    spec_ram_gb: float = 256        # GB per 1.0 normalized RAM
    p_core: float = 15.0            # watts per physical core
    p_gb: float = 0.35              # watts per GB RAM
    pue: float = 1.10


@dataclass(frozen=True)
class NormalizationCfg:
    # "per_job" | "common_currency"
    strategy: str = "common_currency"
    # $/kgCO2 ($190/ton, US EPA 2023 Social Cost of Carbon, EPA-HQ-OAR-2021-0317, based on Rennert et al. Nature 2022)
    carbon_tax_per_kg: float = 0.19

@dataclass(frozen=True)
class BidCfg:
    # "uniform" (recommended) | "lognormal" (legacy)
    strategy: str = "uniform"
    # Base utility rate ($/hour): minimum hourly value a user derives from
    # job completion, independent of resource footprint. Multiplied by
    # duration. Ref: reinterpretation suggested by advisor — demand-side
    # utility rather than supply-side fee.
    base_utility: float = 50.0
    # uniform params
    gamma1: float = 0.02            # $/core-hour reserve price
    gamma2: float = 0.004           # $/GB-hour reserve price
    # lognormal params
    sigma: float = 1.5              # bid dispersion
    base_multiplier: float = 1.0
    


@dataclass(frozen=True)
class TimeCfg:
    step_semantics: str = "real_tick"
    tick_seconds: float = 1.0           # wall-clock length of one step (real_tick only)
    d_max_quantile: float = 1.0     # 1.0 = max of survivors (no double-trim; see outliers.d_max_quantile)
    d_max_horizon_fraction: float = 0.05
    d_max_ratio_abort: float = 0.10
    # Minimum job duration for allocation purposes. Jobs shorter than this
    # are treated as lasting this long, matching cloud billing minimums
    # (GCP/AWS bill 1 min minimum) and preventing near-zero resource
    # volumes from inflating gamma.
    duration_floor_hours: float = 1.0 / 60.0  # 1 minute


@dataclass(frozen=True)
class HardwareCfg:
    n_profiles: int = 12
    # "kmeans"          -> k-means in (optionally) log space; adapts to clusters
    # "quantile_grid"   -> per-dimension quantile bins, cross-product cells
    # "geometric_grid"  -> per-dimension geometric bins, cross-product cells
    # "quantile_1d"     -> quantile bins on a single scalar summary (cpu+ram),
    #                      then centroid per bin. Robust when the two resources
    #                      are strongly correlated (as in most cloud traces).
    method: str = "kmeans"
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
    hw_floor: float = 1.0e-6


@dataclass(frozen=True)
class OutliersCfg:
    policy: str = "drop"            # "drop" | "clip"
    a_max_quantile: float = 0.95
    d_max_quantile: float = 0.95
    xi_warn_threshold: float = 0.10


@dataclass(frozen=True)
class AbsoluteCapCfg:
    cpu: Optional[float] = None
    ram: Optional[float] = None


@dataclass(frozen=True)
class CapacityCfg:
    mode: str = "fraction_of_concurrency_quantile" # "fraction_of_volume" | "fraction_of_peak" | "fraction_of_concurrency_quantile" | "absolute"
    fraction: float = 0.60
    concurrency_quantile: float = 0.90   # used only by fraction_of_concurrency_quantile
    absolute: AbsoluteCapCfg = field(default_factory=AbsoluteCapCfg)


@dataclass(frozen=True)
class ObjectiveCfg:
    lambda1: float = 0.3
    lambda2: float = 0.3
    lambda3: float = 0.3


@dataclass(frozen=True)
class GridCfg:
    k_star: int = 16
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
    # HOW the bin edges are spaced within that space:
    #   "quantile"  -> equal-MASS bins (each bin holds ~n/K jobs). Best for
    #                  heavily skewed distributions: no bin is starved.
    #   "geometric" -> multiplicative spacing.
    #   "log_linear"-> uniform in log-space.
    #   "linear"    -> uniform spacing.
    spacing: str = "geometric"


@dataclass(frozen=True)
class ConfidenceCfg:
    delta: Optional[float] = None
    delta_power: int = 3


@dataclass(frozen=True)
class ModelCfg:
    kind: str = "theory"          # "theory" | "practical"


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
    kind: str = "frozen"          # "frozen" | "time_aware"


@dataclass(frozen=True)
class AdmissionCfg:
    a_buffer_fraction: float = 1.0


@dataclass(frozen=True)
class RevisionCfg:
    mode: str = "continuous"      # "continuous" | "at_completion"


@dataclass(frozen=True)
class PhantomCfg:
    revise_in_flight: bool = False


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
    short_phase_tracking: bool = True


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
    outliers: OutliersCfg = field(default_factory=OutliersCfg)
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
    admission: AdmissionCfg = field(default_factory=AdmissionCfg)
    revision: RevisionCfg = field(default_factory=RevisionCfg)
    phantom: PhantomCfg = field(default_factory=PhantomCfg)
    lp: LpCfg = field(default_factory=LpCfg)
    test3: Test3Cfg = field(default_factory=Test3Cfg)
    phase_transition: PhaseTransitionCfg = field(default_factory=PhaseTransitionCfg)
    logging: LoggingCfg = field(default_factory=LoggingCfg)

    #  validation 
    def validate(self) -> "Config":
        assert self.time.step_semantics == "real_tick", \
            "step_semantics must be 'real_tick' (one_arrival legacy mode is not supported)."
        assert self.time.tick_seconds > 0
        assert self.model.kind in ("theory", "practical")
        assert self.scalarization.method in ("chebyshev", "linear")
        assert self.oracle.kind in ("frozen", "time_aware")
        assert self.revision.mode in ("continuous", "at_completion")
        assert self.capacity.mode in (
            "fraction_of_volume", "fraction_of_peak",
            "fraction_of_concurrency_quantile", "absolute",
        )
        assert 0.0 < self.capacity.concurrency_quantile <= 1.0
        assert 0.0 < self.capacity.fraction <= 1.0
        assert self.grid.k_star >= 2
        assert 0.0 <= self.admission.a_buffer_fraction <= 1.0
        assert 0.0 < self.time.d_max_quantile <= 1.0
        assert 0.0 < self.time.d_max_horizon_fraction <= 1.0
        assert self.hardware.method in (
            "kmeans", "quantile_grid", "geometric_grid", "quantile_1d")
        assert self.hardware.fit_on in ("full", "H")
        assert self.hardware.n_profiles >= 1
        assert self.grid.space in ("v", "phi", "phi_shifted")
        assert self.grid.spacing in ("quantile", "geometric", "log_linear", "linear")
        assert self.outliers.policy in ("drop", "clip")
        assert 0.0 < self.outliers.a_max_quantile <= 1.0
        assert 0.0 < self.outliers.d_max_quantile <= 1.0
        assert self.bid.strategy in ("uniform", "lognormal")
        assert self.normalization.strategy in (
            "per_job", "moving_avg", "common_currency", "utopian")
        assert self.energy_model.spec_cpu_core > 0
        assert self.energy_model.spec_ram_gb > 0
        assert self.normalization.carbon_tax_per_kg >= 0
        if self.capacity.mode == "absolute":
            assert self.capacity.absolute.cpu is not None
            assert self.capacity.absolute.ram is not None
        # theory model consistency: it must not use practical-only features
        if self.model.kind == "theory" and self.oracle.kind == "time_aware":
            # allowed but flagged — crossing axes is a legitimate ablation
            pass
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
    "TimeCfg": TimeCfg, "HardwareCfg": HardwareCfg, "OutliersCfg": OutliersCfg,
    "AbsoluteCapCfg": AbsoluteCapCfg, "CapacityCfg": CapacityCfg,
    "ObjectiveCfg": ObjectiveCfg, "GridCfg": GridCfg, "ConfidenceCfg": ConfidenceCfg,
    "ModelCfg": ModelCfg, "ScalarizationCfg": ScalarizationCfg, "OracleCfg": OracleCfg,
    "AdmissionCfg": AdmissionCfg, "RevisionCfg": RevisionCfg, "PhantomCfg": PhantomCfg,
    "LpCfg": LpCfg, "Test3Cfg": Test3Cfg, "PhaseTransitionCfg": PhaseTransitionCfg,
    "LoggingCfg": LoggingCfg, "EnergyModelCfg": EnergyModelCfg,
    "BidCfg": BidCfg, "NormalizationCfg": NormalizationCfg,
}


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config into a frozen Config tree."""
    path = Path(path)
    with path.open("r") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = _from_dict(Config, raw)
    return cfg.validate()


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def default_config() -> Config:
    """
    The project default configuration.

    Loads ``configs/default.yaml`` when it is present (the repo layout), so that
    file is the single effective source of truth. Falls back to the dataclass
    defaults only when the YAML is missing (e.g. an installed wheel without the
    configs/ tree).
    """
    if DEFAULT_CONFIG_PATH.is_file():
        return load_config(DEFAULT_CONFIG_PATH)
    return Config().validate()

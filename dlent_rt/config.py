"""
Configuration layer for DLENT-RT.

A single frozen dataclass tree loaded from YAML. Every experimental axis is a
field here, so an ablation is a config sweep rather than a code change. Loading
is strict: unknown keys raise, so a typo in a YAML never silently no-ops.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional

import yaml


# --------------------------------------------------------------------------- #
# Sub-configs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RunCfg:
    name: str = "default"
    seed: int = 42
    output_dir: str = "runs"
    log_level: str = "INFO"


@dataclass(frozen=True)
class ColumnsCfg:
    collection_id: str = "collection_id"
    datetime: str = "job_datetime"
    q: str = "q_j"
    a_cpu: str = "A_cpu"
    a_ram: str = "A_ram"
    duration_hours: str = "D (hours)"
    v_rate: str = "v_rate"
    phi_rate: str = "phi_rate"
    w_kw: str = "w_j_kw"
    elec_price: str = "elec_price_per_kWh"
    carbon_intensity: str = "carbon_intensity_gCO2_per_kWh"
    c_elec: str = "C_elec"
    c_carbon: str = "C_carbon"


@dataclass(frozen=True)
class DataCfg:
    batch_csv: str = "data/batch_may300k.csv"
    n0: int = 5000
    columns: ColumnsCfg = field(default_factory=ColumnsCfg)


@dataclass(frozen=True)
class TimeCfg:
    step_semantics: str = "one_arrival"
    duration_unit: str = "hours"
    d_max_quantile: float = 0.95
    d_max_horizon_fraction: float = 0.05
    d_max_ratio_abort: float = 0.10


@dataclass(frozen=True)
class HardwareCfg:
    n_profiles: int = 12
    method: str = "kmeans"          # "kmeans" | "quantile_grid"
    log_space: bool = True
    fit_on: str = "full"            # "full" | "H"


@dataclass(frozen=True)
class OutliersCfg:
    policy: str = "drop"            # "drop" | "clip"
    a_max_quantile: float = 0.99
    d_max_quantile: float = 0.95
    xi_warn_threshold: float = 0.10


@dataclass(frozen=True)
class AbsoluteCapCfg:
    cpu: Optional[float] = None
    ram: Optional[float] = None


@dataclass(frozen=True)
class CapacityCfg:
    mode: str = "fraction_of_volume"
    fraction: float = 0.60
    absolute: AbsoluteCapCfg = field(default_factory=AbsoluteCapCfg)


@dataclass(frozen=True)
class ObjectiveCfg:
    lambda1: float = 1.0
    lambda2: float = 1.0
    lambda3: float = 1.0
    scc: float = 0.05
    pue: float = 1.10
    cpu_watts: float = 4.0
    ram_watts: float = 0.5
    normalize: bool = True


@dataclass(frozen=True)
class GridCfg:
    k_star: int = 16
    phi_floor: float = 1.0e-6
    priority_classes: tuple = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class ConfidenceCfg:
    delta: Optional[float] = None
    delta_power: int = 3


@dataclass(frozen=True)
class ModelCfg:
    kind: str = "theory"          # "theory" | "practical"


@dataclass(frozen=True)
class ScalarizationCfg:
    method: str = "chebyshev"     # "chebyshev" | "linear" | "eps_constraint"
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
    solver: str = "highs"
    warm_start: bool = True
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


# --------------------------------------------------------------------------- #
# Top-level config
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Config:
    run: RunCfg = field(default_factory=RunCfg)
    data: DataCfg = field(default_factory=DataCfg)
    time: TimeCfg = field(default_factory=TimeCfg)
    hardware: HardwareCfg = field(default_factory=HardwareCfg)
    outliers: OutliersCfg = field(default_factory=OutliersCfg)
    capacity: CapacityCfg = field(default_factory=CapacityCfg)
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

    # ---- validation ----------------------------------------------------- #
    def validate(self) -> "Config":
        assert self.time.step_semantics == "one_arrival", \
            "Only one_arrival step semantics is implemented (decision A1)."
        assert self.model.kind in ("theory", "practical")
        assert self.scalarization.method in ("chebyshev", "linear", "eps_constraint")
        assert self.oracle.kind in ("frozen", "time_aware")
        assert self.revision.mode in ("continuous", "at_completion")
        assert self.capacity.mode in ("fraction_of_volume", "fraction_of_peak", "absolute")
        assert 0.0 < self.capacity.fraction <= 1.0
        assert self.grid.k_star >= 2
        assert 0.0 <= self.admission.a_buffer_fraction <= 1.0
        assert 0.0 < self.time.d_max_quantile <= 1.0
        assert 0.0 < self.time.d_max_horizon_fraction <= 1.0
        assert self.hardware.method in ("kmeans", "quantile_grid")
        assert self.hardware.fit_on in ("full", "H")
        assert self.hardware.n_profiles >= 1
        assert self.outliers.policy in ("drop", "clip")
        assert 0.0 < self.outliers.a_max_quantile <= 1.0
        assert 0.0 < self.outliers.d_max_quantile <= 1.0
        if self.capacity.mode == "absolute":
            assert self.capacity.absolute.cpu is not None
            assert self.capacity.absolute.ram is not None
        # theory model consistency: it must not use practical-only features
        if self.model.kind == "theory" and self.oracle.kind == "time_aware":
            # allowed but flagged — crossing axes is a legitimate ablation
            pass
        return self


# --------------------------------------------------------------------------- #
# Recursive dict -> dataclass with strict unknown-key checking
# --------------------------------------------------------------------------- #
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
    "LoggingCfg": LoggingCfg,
}


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config into a frozen Config tree."""
    path = Path(path)
    with path.open("r") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = _from_dict(Config, raw)
    return cfg.validate()


def default_config() -> Config:
    """The all-defaults config, without touching disk."""
    return Config().validate()

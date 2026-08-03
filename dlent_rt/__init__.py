"""
DLENT-RT: real-time DLENT-Exact simulator for multi-objective cloud resource
allocation with mechanism design and online learning.

First slice (this build): config + data + Phase 0.
"""
from .config import Config, load_config, default_config
from .data import Dataset, JobArrays, load_dataset
from .phase0 import PhaseState, run_phase0, TypeSpace, build_type_space
from .grid import MyersonModel, GeometricGrid, fit_myerson, build_geometric_grid
from .lp_oracle import scalarize, solve_lp_rs, LPResult

__version__ = "0.1.0"

__all__ = [
    "Config", "load_config", "default_config",
    "Dataset", "JobArrays", "load_dataset",
    "PhaseState", "run_phase0", "TypeSpace", "build_type_space",
    "MyersonModel", "GeometricGrid", "fit_myerson", "build_geometric_grid",
    "scalarize", "solve_lp_rs", "LPResult",
]

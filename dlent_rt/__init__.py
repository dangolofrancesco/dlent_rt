"""
DLENT-RT: real-time DLENT-Exact simulator for multi-objective cloud resource
allocation with mechanism design and online learning.

Pipeline: config -> data (outliers, hardware profiles, capacity) -> Phase 0
(bids, known bid distribution, type space, additive dollar reward with the
quantile_linear carbon price s_C, LP-RS) -> online theory model
(dlent_rt.models.theory: oracle, thinning, Tests 1-3, ReInit, hindsight
benchmark).
"""
from .config import Config, load_config, default_config
from .data import Dataset, JobArrays, load_dataset
from .phase0 import PhaseState, run_phase0, TypeSpace, build_type_space
from .bid import MyersonModel, bid_model
from .grid import ValuationGrid, build_valuation_grid
from .lp_oracle import scalarize, solve_lp_rs, LPResult

__version__ = "0.1.0"

__all__ = [
    "Config", "load_config", "default_config",
    "Dataset", "JobArrays", "load_dataset",
    "PhaseState", "run_phase0", "TypeSpace", "build_type_space",
    "MyersonModel", "bid_model",
    "ValuationGrid", "build_valuation_grid",
    "scalarize", "solve_lp_rs", "LPResult",
]

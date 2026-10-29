"""
DLENT-RT: real-time DLENT-Exact simulator for multi-objective cloud resource
allocation with mechanism design and online learning.

Pipeline: config -> data (24 h cap, job quota, cluster C = f x D_bar) -> Phase 0
(calibrated U0, bids, sigma refit, geometric bid grid, types, additive dollar
reward with the carbon price s_C, LP-RS) -> online model
(dlent_rt.models.theory: oracle, thinning, Tests 1-3, ReInit, threshold payments).
"""
from .config import Config, load_config, default_config
from .data import Dataset, JobArrays, load_dataset
from .phase0 import PhaseState, run_phase0, TypeSpace, build_type_space
from .bid import MyersonModel, bid_model
from .grid import ValuationGrid, build_valuation_grid
from .lp_oracle import solve_lp_rs, LPResult

__version__ = "0.1.0"

__all__ = [
    "Config", "load_config", "default_config",
    "Dataset", "JobArrays", "load_dataset",
    "PhaseState", "run_phase0", "TypeSpace", "build_type_space",
    "MyersonModel", "bid_model",
    "ValuationGrid", "build_valuation_grid",
    "solve_lp_rs", "LPResult",
]

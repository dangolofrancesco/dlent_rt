"""
Entry point: Phase 0 -> online loop -> logs.

Outputs (runs/<run.name>/<model>/):
  arrivals.parquet   one row per arrival (decision, reward, occupancy, test diagnostics)
  phases.parquet     one row per phase (length, end reason, reinit diagnostics, regret)
  short_phases.parquet  phases with n_arrivals < phase_transition.n_min
  occupancy.parquet  every physical occupancy change (tick, +1/-1, job, U_cpu, U_ram)
  summary.json       totals, per-phase and global hindsight regret, error parameters
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..config import Config
from ..data import load_dataset
from ..phase0 import run_phase0
from .theory import TheoryModel, RunResult


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def write_results(res: RunResult, out_dir: Path, cfg: Config) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.logging.per_step:
        pd.DataFrame(res.arrivals).to_parquet(out_dir / "arrivals.parquet")
        occ = pd.DataFrame(res.occupancy_events,
                           columns=["tick", "event", "job", "U_cpu", "U_ram"][: 3 + 2])
        occ.to_parquet(out_dir / "occupancy.parquet")
    ph = pd.DataFrame(res.phases)
    if cfg.logging.per_phase:
        ph.to_parquet(out_dir / "phases.parquet")
    if cfg.logging.short_phase_log and len(ph):
        ph[ph["short"]].to_parquet(out_dir / "short_phases.parquet")
    (out_dir / "summary.json").write_text(json.dumps(_jsonable(res.summary), indent=2))


def run_online(cfg: Config, out_dir: Optional[Path] = None, ds=None, state0=None) -> RunResult:
    if ds is None:
        ds = load_dataset(cfg)
    if state0 is None:
        state0 = run_phase0(ds, cfg)
    if cfg.capacity.job_level_enlarge:                  # D3 option (off by default)
        from ..benchmarks import enlarge_for_job_models
        ds, state0, _ = enlarge_for_job_models(ds, state0, cfg)
    model = TheoryModel(ds, state0, cfg)
    res = model.run()
    variant = "practical" if cfg.oracle.kind == "live" else "theory"
    if out_dir is None:
        out_dir = Path(cfg.run.output_dir) / cfg.run.name / variant
    write_results(res, Path(out_dir), cfg)
    return res


if __name__ == "__main__":   # python -m dlent_rt.models.runner [config.yaml]
    import sys
    from ..config import default_config, load_config
    cfg = load_config(sys.argv[1]) if len(sys.argv) > 1 else default_config()
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        r = run_online(cfg)
    print(json.dumps(_jsonable(r.summary), indent=2))

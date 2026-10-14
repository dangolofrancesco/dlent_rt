"""
Synthetic trace + grid CSVs with the SAME schema as the real inputs
(batch_may2019_30k.csv / grid_data_may_2019.csv). Used only by tests: the
numbers are not meant to be realistic, only to exercise every code path.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from dlent_rt.config import default_config


def write_synthetic(tmpdir: Path, n_jobs: int = 3000, hours: float = 36.0,
                    seed: int = 0, drift: bool = False) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    t0 = np.datetime64("2019-05-01T00:00:00")
    secs = np.sort(rng.uniform(0, hours * 3600, n_jobs))
    if drift:  # second half: bigger jobs -> exercises nonstationarity code paths
        pass
    cpu = np.exp(rng.normal(np.log(0.01), 0.8, n_jobs))
    ram = np.exp(rng.normal(np.log(0.006), 0.8, n_jobs))
    dur = np.exp(rng.normal(np.log(0.15), 1.0, n_jobs))
    if drift:
        late = secs > hours * 1800
        dur[late] *= 3.0
        cpu[late] *= 2.0
    df = pd.DataFrame({
        "collection_id": np.arange(n_jobs),
        "request_time": (t0 + secs.astype("timedelta64[s]")).astype("datetime64[s]"),
        "priority": rng.integers(1, 6, n_jobs),
        "scheduling_class": rng.integers(0, 4, n_jobs),
        "CPU": cpu, "RAM": ram, "duration": dur,
    })
    hrs = np.arange(int(hours) + 2)
    grid = pd.DataFrame({
        "timestamp_utc": (t0 + hrs.astype("timedelta64[h]")).astype("datetime64[s]"),
        "elec_price_per_kWh": 0.03 + 0.01 * np.sin(hrs / 24 * 2 * np.pi),
        "carbon_intensity_kgCO2_per_kWh": 0.40 + 0.05 * np.cos(hrs / 24 * 2 * np.pi),
    })
    b, g = tmpdir / "batch.csv", tmpdir / "grid.csv"
    df.to_csv(b, index=False)
    grid.to_csv(g, index=False)
    return b, g


def synthetic_config(tmpdir: Path, n0: int = 1000, **kw):
    b, g = write_synthetic(tmpdir, **kw)
    cfg = default_config()
    return replace(
        cfg,
        data=replace(cfg.data, batch_csv=str(b), grid_csv=str(g), n0=n0),
        grid=replace(cfg.grid, k_star=16),
        hardware=replace(cfg.hardware, n_profiles=6),
        time=replace(cfg.time, d_max_horizon_fraction=0.10, d_max_ratio_abort=0.2),
        normalization=replace(cfg.normalization, quantile=0.90),
    )

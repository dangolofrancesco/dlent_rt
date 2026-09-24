"""
Data ingestion for DLENT-RT.

The output is a `Dataset` object holding immutable numpy arrays. Everything
downstream (Phase 0, the online loop) reads from this; the CSV is never touched
again.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .energy import load_grid_data

from .config import Config
from .preprocess import (
    HardwareCatalogue, OutlierReport, fit_hardware_catalogue, handle_outliers,
)


@dataclass
class JobArrays:
    n: int
    resources: list[str]
    priority: np.ndarray                        # priority (1-5)
    scheduling_class: np.ndarray         # scheduling class (0-3)
    duration_hours: np.ndarray           # duration in hours (from CSV)
    duration_steps: np.ndarray           # duration in steps (derived)
    arrival_step: np.ndarray
    A: np.ndarray                        # (n, |I|) snapped resource requests (normalized)
    hw_idx: np.ndarray
    collection_id: np.ndarray
    datetime: np.ndarray                 # np.datetime64

    def __post_init__(self):
        assert self.A.shape == (self.n, len(self.resources))


@dataclass
class Dataset:
    """Everything the simulator needs, derived once from the CSV."""
    H: JobArrays
    stream: JobArrays
    resources: list[str]
    capacity: np.ndarray
    a_max: np.ndarray
    d_max_steps: int
    step_semantics: str
    tick_seconds: float
    T: int
    n_arrivals_stream: int
    H_span_steps: int
    hw_catalogue: HardwareCatalogue
    outlier_report: OutlierReport
    d_max_diag: dict
    xi: float
    grid_data: object              # GridData from energy.py (or None)

    @property
    def n_resources(self) -> int:
        return len(self.resources)


def _read_csv(cfg: Config) -> pd.DataFrame:
    col = cfg.data.columns
    df = pd.read_csv(cfg.data.batch_csv)
    df[col.datetime] = pd.to_datetime(df[col.datetime])
    df = df.sort_values(col.datetime).reset_index(drop=True)
    return df


def _assign_real_tick_steps(
    datetimes: np.ndarray, t0: np.datetime64, tick_seconds: float
) -> np.ndarray:
    """
    Assign each job an integer step index = elapsed ticks since t0.

        step = floor((timestamp - t0) / tick_seconds)

    COLLISIONS: the model allows at most one arrival per step. If two
    jobs land in the same tick, we nudge the later one to the next free tick. 
    """
    elapsed_s = (datetimes - t0) / np.timedelta64(1, "s")
    steps = np.floor(elapsed_s / tick_seconds).astype(np.int64)
    # forward pass to enforce strict monotonicity (at most one arrival per tick)
    out = np.empty_like(steps)
    prev = -1
    for i, s in enumerate(steps):
        cur = s if s > prev else prev + 1
        out[i] = cur
        prev = cur
    return out


def _duration_to_steps_real_tick(
    duration_hours: np.ndarray, tick_seconds: float, d_max_steps: Optional[int]
) -> np.ndarray:
    """
    EXACT conversion: a duration in hours is a physical quantity, and one step
    is a fixed number of wall-clock seconds, so

        steps = round(duration_hours * 3600 / tick_seconds)
    """
    steps = np.rint(duration_hours * 3600.0 / tick_seconds).astype(np.int64)
    steps = np.maximum(steps, 1)
    if d_max_steps is not None:
        steps = np.minimum(steps, d_max_steps)
    return steps


def _convert_durations(
    duration_hours: np.ndarray, cfg: Config, d_max_steps: Optional[int]
) -> np.ndarray:
    """Convert duration in hours to steps (real_tick mode only)."""
    if cfg.time.step_semantics != "real_tick":
        raise ValueError(
            f"step_semantics='{cfg.time.step_semantics}' not supported. "
            "Only 'real_tick' is currently supported."
        )
    return _duration_to_steps_real_tick(
        duration_hours, cfg.time.tick_seconds, d_max_steps
    )


def _build_job_arrays(
    df, cfg, resources, d_max_steps, catalogue, arrival_step,
):
    col = cfg.data.columns
    n = len(df)
    A_raw = np.column_stack([
        df[col.requested_cpu].to_numpy(dtype=float),
        df[col.requested_ram].to_numpy(dtype=float),
    ])
    hw_idx = catalogue.assign(A_raw)
    A = catalogue.centroids[hw_idx]
    dur_hours = df[col.duration_hours].to_numpy(dtype=float)
    # Duration floor: treat sub-minute jobs as 1-minute for allocation.
    # Cloud providers bill on a 1-minute minimum (GCP/AWS), and without
    # this floor near-zero durations produce near-zero resource volumes
    # that inflate gamma's r_max/v_min term by orders of magnitude.
    dur_hours = np.maximum(dur_hours, cfg.time.duration_floor_hours)
    dur_steps = _convert_durations(dur_hours, cfg, d_max_steps)
    return JobArrays(
        n=n, resources=resources,
        priority=df[col.priority].to_numpy(dtype=float),
        scheduling_class=df[col.scheduling_class].to_numpy(dtype=float),
        duration_hours=dur_hours,
        duration_steps=dur_steps,
        arrival_step=arrival_step,
        A=A, hw_idx=hw_idx,
        collection_id=df[col.collection_id].to_numpy(),
        datetime=df[col.datetime].to_numpy(),
    )


def _occupancy_events(stream: JobArrays) -> tuple[np.ndarray, np.ndarray]:
    """
    Event-driven concurrent-occupancy computation.

    Occupancy only changes at events: +A_s when job s arrives at step
    arrival_step[s], and -A_s when it releases at arrival_step[s] +
    duration_steps[s]. Between events, occupancy is CONSTANT. So we never
    iterate over ticks (which under real_tick semantics could be millions);
    we only sort ~2n event points.

    Returns (times, occupancy) where occupancy[k] is the resource vector held
    during the half-open interval [times[k], times[k+1]).
    """
    start = stream.arrival_step
    end = start + stream.duration_steps
    times = np.concatenate([start, end])
    deltas = np.concatenate([stream.A, -stream.A], axis=0)

    order = np.argsort(times, kind="stable")
    times = times[order]
    deltas = deltas[order]

    # collapse simultaneous events onto a single time point
    uniq_times, inv = np.unique(times, return_inverse=True)
    agg = np.zeros((len(uniq_times), stream.A.shape[1]))
    np.add.at(agg, inv, deltas)

    occ = np.cumsum(agg, axis=0)
    return uniq_times, occ


def _occupancy_stats(stream: JobArrays) -> dict:
    """
    Peak and time-weighted quantiles of concurrent occupancy.

    Time-weighting matters: an occupancy level that persists for 10,000 ticks
    should count 10,000x more than one that lasts a single tick. A naive
    quantile over event points would be badly biased toward short-lived spikes.

    Returns: peak, levels, weights (used by _size_capacity) plus times, occ
    (the raw step function, used by the diagnostics notebook).
    """
    times, occ = _occupancy_events(stream)
    if len(times) < 2:
        peak = occ.max(axis=0) if len(occ) else np.zeros(stream.A.shape[1])
        return dict(peak=peak, times=times, occ=occ, weights=None)

    # each occupancy level occ[k] holds for (times[k+1] - times[k]) ticks
    weights = np.diff(times).astype(float)
    levels = occ[:-1]                                  # level during each interval

    peak = levels.max(axis=0)
    return dict(peak=peak, times=times, occ=occ, levels=levels, weights=weights)


def _time_weighted_quantile(
    levels: np.ndarray, weights: np.ndarray, q: float
) -> np.ndarray:
    """
    Per-resource time-weighted quantile of the occupancy step function.
    levels: (m, |I|), weights: (m,) durations each level is held.
    """
    n_res = levels.shape[1]
    out = np.zeros(n_res)
    total = weights.sum()
    for i in range(n_res):
        order = np.argsort(levels[:, i], kind="stable")
        vals = levels[order, i]
        w = weights[order]
        cw = np.cumsum(w) / max(total, 1e-12)
        idx = int(np.searchsorted(cw, q, side="left"))
        idx = min(idx, len(vals) - 1)
        out[i] = vals[idx]
    return out


def _size_capacity(stream: JobArrays, cfg: Config) -> np.ndarray:
    if cfg.capacity.mode == "fraction_of_volume":
        volume = (stream.A * stream.duration_steps[:, None]).sum(axis=0)
        return cfg.capacity.fraction * volume
    elif cfg.capacity.mode == "fraction_of_peak":
        stats = _occupancy_stats(stream)
        return cfg.capacity.fraction * stats["peak"]
    elif cfg.capacity.mode == "fraction_of_concurrency_quantile":
        stats = _occupancy_stats(stream)
        if stats.get("weights") is None:
            return cfg.capacity.fraction * stats["peak"]
        qv = _time_weighted_quantile(
            stats["levels"], stats["weights"], cfg.capacity.concurrency_quantile
        )
        return cfg.capacity.fraction * qv
    else:  # absolute
        return np.array(
            [cfg.capacity.absolute.cpu, cfg.capacity.absolute.ram], dtype=float
        )


def _compute_d_max_steps(
    duration_hours_all: np.ndarray, T: int, cfg: Config
) -> tuple[int, dict]:
    # arrivals_per_hour = _estimate_arrivals_per_hour(df, cfg)
    steps = _convert_durations(
        duration_hours_all, cfg, d_max_steps=None
    )
    q = cfg.time.d_max_quantile
    candidate = int(np.quantile(steps, q))
    candidate = max(candidate, 1)

    ceiling = max(int(cfg.time.d_max_horizon_fraction * T), 1)
    d_max = min(candidate, ceiling)

    ratio = d_max / max(T, 1)
    diag = dict(
        quantile=q,
        candidate_from_quantile=candidate,
        horizon_ceiling=ceiling,
        chosen=d_max,
        ratio_to_T=ratio,
        clamped_by_ceiling=candidate > ceiling,
    )
    if ratio > cfg.time.d_max_ratio_abort:
        raise ValueError(
            f"d_max/T = {ratio:.3f} exceeds abort threshold "
            f"{cfg.time.d_max_ratio_abort}. d_max={d_max}, T={T}. "
            f"The theory requires d_max << T. Lower time.d_max_quantile or "
            f"time.d_max_horizon_fraction, or drop more long-duration outliers "
            f"(outliers.d_max_quantile)."
        )
    return d_max, diag


def _stratified_sample_by_hour(df, n0, datetime_col, seed=42,
                                pool_multiplier=3, min_stream_fraction=0.05):
    """
    Sample n0 jobs stratified by hour-of-day, drawing ONLY from the first
    pool_multiplier*n0 rows of df (chronological order preserved).
    This ensures:
      1. H jobs all precede stream jobs temporally (causal bootstrap).
      2. H covers all 24 hours uniformly (representative statistics).
    pool_multiplier: how many times n0 to look ahead for candidates.
                     Default 3 gives a pool of 15000 for n0=5000.
                     Raise if some hours are sparse in the first K rows.
    min_stream_fraction: fraction of df's tail that is NEVER included in the
                     pool, regardless of n0/pool_multiplier. Without this,
                     pool_size = min(pool_multiplier*n0, len(df)) can equal
                     len(df) when n0 is large relative to the dataset, so the
                     pool covers every row (including the chronologically
                     last ones). H's stratified sample could then include
                     rows at or near the dataset's max datetime, making
                     df_stream (rows strictly after H's max datetime) empty.
    """
    max_pool_size = max(len(df) - max(int(min_stream_fraction * len(df)), 1), 1)
    if n0 > max_pool_size:
        raise ValueError(
            f"n0={n0} leaves fewer than {min_stream_fraction:.0%} of rows "
            f"({len(df)} total) for the online stream. Lower n0 (<= "
            f"{max_pool_size}) or raise min_stream_fraction's budget."
        )
    pool_size = min(pool_multiplier * n0, max_pool_size)
    pool = df.iloc[:pool_size].copy()
    pool['_hour'] = pd.to_datetime(pool[datetime_col]).dt.hour
    per_hour = max(n0 // 24, 1)
    sampled = []
    for h in range(24):
        candidates = pool[pool['_hour'] == h]
        n_take = min(per_hour, len(candidates))
        if n_take > 0:
            sampled.append(candidates.sample(n=n_take, random_state=seed + h))
    result = pd.concat(sampled).sort_values(datetime_col).reset_index(drop=True)
    result = result.drop(columns=['_hour'])
    return result


# Public entry point
def load_dataset(cfg: Config) -> Dataset:
    df = _read_csv(cfg)
    col = cfg.data.columns
    resources = ["cpu", "ram"]            
    a_cols = [col.requested_cpu, col.requested_ram]

    df, outlier_report = handle_outliers(df, cfg, a_cols, col.duration_hours)

    n0 = cfg.data.n0
    if n0 >= len(df):
        raise ValueError(
            f"n0={n0} >= rows after outlier handling ({len(df)}). "
            "Lower n0 or relax outlier thresholds."
        )

    dts = df[col.datetime].to_numpy()

    if cfg.time.step_semantics == "real_tick":
        t0 = dts[0]
        all_steps = _assign_real_tick_steps(dts, t0, cfg.time.tick_seconds)
    else:
        raise ValueError(
            f"step_semantics='{cfg.time.step_semantics}' not supported. "
            "Only 'real_tick' is currently supported."
        )

    # Carry the step assignment as a column so it survives the stratified
    # split below (H is no longer a contiguous positional prefix of df, so
    # a positional slice of all_steps would misalign with df_H/df_stream).
    df = df.assign(_step=all_steps)

    df_H = _stratified_sample_by_hour(df, n0, col.datetime, seed=cfg.run.seed)
    last_h_time = df_H[col.datetime].max()
    df_stream = df[df[col.datetime] > last_h_time].reset_index(drop=True)

    steps_H = df_H["_step"].to_numpy()
    steps_stream = df_stream["_step"].to_numpy()
    stream_origin = steps_stream[0]
    steps_stream = steps_stream - stream_origin
    T = int(steps_stream[-1]) + 1
    H_span_steps = int(steps_H[-1] - steps_H[0]) + 1

    df_H = df_H.drop(columns=["_step"])
    df_stream = df_stream.drop(columns=["_step"])
    df = df.drop(columns=["_step"])

    if cfg.hardware.fit_on == "full":
        A_fit = df[a_cols].to_numpy(dtype=float)
    else:
        A_fit = df_H[a_cols].to_numpy(dtype=float)
    catalogue = fit_hardware_catalogue(A_fit, cfg)

    d_max_steps, d_max_diag = _compute_d_max_steps(
        df[col.duration_hours].to_numpy(dtype=float), T, cfg,
    )

    H = _build_job_arrays(df_H, cfg, resources,
                          d_max_steps, catalogue, steps_H)
    stream = _build_job_arrays(df_stream, cfg, resources,
                               d_max_steps, catalogue, steps_stream)

    
    capacity = _size_capacity(stream, cfg)
    a_max = stream.A.max(axis=0)
    xi = float(np.max(a_max / capacity))

    if xi > cfg.outliers.xi_warn_threshold:
        import warnings
        warnings.warn(
            f"xi = max_i a_max,i/c_i = {xi:.3f} exceeds "
            f"{cfg.outliers.xi_warn_threshold} (~1/log T). The DLENT guarantee "
            f"needs xi small. Consider a lower outliers.a_max_quantile or a "
            f"higher capacity.fraction.",
            RuntimeWarning,
        )

    grid_data = None  
    if cfg.data.grid_csv and os.path.exists(cfg.data.grid_csv):
        grid_data = load_grid_data(cfg.data.grid_csv)

    return Dataset(
        H=H,
        stream=stream,
        resources=resources,
        capacity=capacity,
        a_max=a_max,
        d_max_steps=d_max_steps,
        step_semantics=cfg.time.step_semantics,
        tick_seconds=cfg.time.tick_seconds,
        T=T,
        n_arrivals_stream=stream.n,
        H_span_steps=H_span_steps,
        hw_catalogue=catalogue,
        outlier_report=outlier_report,
        d_max_diag=d_max_diag,
        xi=xi,
        grid_data=grid_data
    )

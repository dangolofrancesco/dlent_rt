"""
Data ingestion for DLENT-RT.

The output is a `Dataset` object holding immutable numpy arrays. Everything
downstream (Phase 0, the online loop) reads from this; the CSV is never touched
again.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .energy import load_grid_data

from .config import Config, resolve_path
from .preprocess import HardwareCatalogue, LimitReport, apply_r_max, fit_hardware_catalogue


@dataclass
class JobArrays:
    n: int
    resources: list[str]
    priority: np.ndarray                        # priority (1-5)
    scheduling_class: np.ndarray         # scheduling class (0-3)
    duration_hours: np.ndarray           # effective duration in hours, floored and capped at d_max
    duration_steps: np.ndarray           # the same duration in steps
    arrival_step: np.ndarray
    A: np.ndarray                        # (n, |I|) real resource requests (machines)
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
    tick_seconds: float
    T: int
    n_arrivals_stream: int
    H_span_steps: int
    hw_catalogue: HardwareCatalogue
    limit_report: LimitReport
    d_max_diag: dict
    xi: float
    grid_data: object              # GridData from energy.py (or None)
    # capacity.mode mean_demand: lambda_hat (jobs / h), mean_rd and D_bar (machines, per resource), H span (h), f
    capacity_diag: Optional[dict] = None

    @property
    def n_resources(self) -> int:
        return len(self.resources)


def _read_csv(cfg: Config) -> pd.DataFrame:
    col = cfg.data.columns
    df = pd.read_csv(resolve_path(cfg.data.batch_csv))
    df[col.datetime] = pd.to_datetime(df[col.datetime])
    # the file must say its time zone; the model keeps naive UTC, the time base of the grid (timestamp_utc)
    if getattr(df[col.datetime].dt, "tz", None) is None:
        raise ValueError(f"'{col.datetime}' needs time-zone-aware timestamps (e.g. 2019-05-01 07:05:00+00:00); "
                         f"{cfg.data.batch_csv} has none")
    df[col.datetime] = df[col.datetime].dt.tz_convert("UTC").dt.tz_localize(None)
    # effective duration: real CPU-hours / SUM request (<= the span); a zero CPU request keeps the span
    cpu = df[col.requested_cpu].to_numpy(dtype=float)
    span = df[col.duration_hours].to_numpy(dtype=float)
    eff = np.divide(df[col.cpu_hours_real].to_numpy(dtype=float), cpu, out=span.copy(), where=cpu > 0)
    assert np.all(eff <= span * (1 + 1e-9)), "effective duration > span"
    df[col.duration_hours] = eff
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
    """Convert duration in hours to steps of time.tick_seconds (real-tick time model)."""
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
    hw_idx = catalogue.assign(A_raw)        # the profile builds only the type id; A stays the real request
    A = A_raw
    dur_hours = df[col.duration_hours].to_numpy(dtype=float)
    # Duration floor: treat sub-minute jobs as 1-minute for allocation.
    # Cloud providers bill on a 1-minute minimum (GCP/AWS), and without
    # this floor near-zero durations produce near-zero resource volumes
    # that inflate gamma's r_max/v_min term by orders of magnitude.
    dur_hours = np.maximum(dur_hours, cfg.time.duration_floor_hours)
    if d_max_steps is not None:
        # one duration everywhere: bids and energy (duration_hours) use the same cap as the cluster time
        dur_hours = np.minimum(dur_hours, d_max_steps * cfg.time.tick_seconds / 3600.0)
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


@dataclass
class _OccupancyBasis:
    """
    Minimal job-array view needed for occupancy/capacity sizing: arrival_step,
    duration_steps, A. Lets _size_capacity operate on an arbitrary UNION of
    job slices (e.g. H + stream, on a shared absolute timeline) without
    building a full JobArrays (priority, scheduling_class, etc. are unused
    here and would just be dead weight).
    """
    arrival_step: np.ndarray
    duration_steps: np.ndarray
    A: np.ndarray


def _occupancy_events(jobs) -> tuple[np.ndarray, np.ndarray]:
    """
    Event-driven concurrent-occupancy computation.

    Occupancy only changes at events: +A_s when job s arrives at step
    arrival_step[s], and -A_s when it releases at arrival_step[s] +
    duration_steps[s]. Between events, occupancy is CONSTANT. So we never
    iterate over ticks (which under real_tick semantics could be millions);
    we only sort ~2n event points.

    `jobs` needs only .arrival_step, .duration_steps, .A (JobArrays or
    _OccupancyBasis) on a SHARED absolute timeline -- arrival_step values
    from different slices that don't share an origin will silently produce
    a meaningless occupancy series.

    Returns (times, occupancy) where occupancy[k] is the resource vector held
    during the half-open interval [times[k], times[k+1]).
    """
    start = jobs.arrival_step
    end = start + jobs.duration_steps
    times = np.concatenate([start, end])
    deltas = np.concatenate([jobs.A, -jobs.A], axis=0)

    order = np.argsort(times, kind="stable")
    times = times[order]
    deltas = deltas[order]

    # collapse simultaneous events onto a single time point
    uniq_times, inv = np.unique(times, return_inverse=True)
    agg = np.zeros((len(uniq_times), jobs.A.shape[1]))
    np.add.at(agg, inv, deltas)

    occ = np.cumsum(agg, axis=0)
    return uniq_times, occ


def _occupancy_stats(jobs) -> dict:
    """
    Peak and time-weighted quantiles of concurrent occupancy.

    Time-weighting matters: an occupancy level that persists for 10,000 ticks
    should count 10,000x more than one that lasts a single tick. A naive
    quantile over event points would be badly biased toward short-lived spikes.

    `jobs` needs only .arrival_step, .duration_steps, .A on a shared absolute
    timeline (JobArrays or _OccupancyBasis) -- see _occupancy_events.

    Returns: peak, levels, weights (used by _size_capacity) plus times, occ
    (the raw step function, used by the diagnostics notebook).
    """
    times, occ = _occupancy_events(jobs)
    if len(times) < 2:
        peak = occ.max(axis=0) if len(occ) else np.zeros(jobs.A.shape[1])
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


def _size_capacity(basis: _OccupancyBasis, cfg: Config) -> np.ndarray:
    """
    Size cluster capacity from `basis`'s resource-request/occupancy profile.

    Capacity is sized on the FULL historical dataset (H + stream), not on
    stream alone. Sizing on stream alone can misrepresent typical
    concurrency if the stream window happens to capture a
    non-representative slice of load (verified empirically: stream-only
    concurrency was ~3x the full-dataset concurrency in one configuration).
    `basis` must therefore be H and stream COMBINED on one shared absolute
    timeline (see load_dataset), not `stream` by itself -- passing `stream`
    alone silently reproduces the old, biased sizing.
    """
    if cfg.capacity.mode == "exceeded_share":
        # the concurrent-request level exceeded `exceeded_share` of the time when every job is admitted
        stats = _occupancy_stats(basis)
        if stats.get("weights") is None:
            return stats["peak"]
        return _time_weighted_quantile(stats["levels"], stats["weights"], 1.0 - cfg.capacity.exceeded_share)
    else:  # absolute
        return np.array(
            [cfg.capacity.absolute.cpu, cfg.capacity.absolute.ram], dtype=float
        )


def mean_demand(A: np.ndarray, d_hours: np.ndarray, span_hours: float) -> dict:
    """
    Mean demand of a job history (capacity.mode mean_demand), per resource:
        lambda_hat = n / span_hours            (jobs per hour)
        mean_rd    = (1/n) sum_j r_j d_j       (machine-hours per job)
        D_bar      = lambda_hat * mean_rd      (machines) = total work / span
    """
    n = len(d_hours)
    lam = n / span_hours
    mean_rd = (np.asarray(A, float) * np.asarray(d_hours, float)[:, None]).sum(axis=0) / n
    return dict(lambda_hat=lam, mean_rd=mean_rd, D_bar=lam * mean_rd, span_hours=span_hours, n=n)


def _n_history(df: pd.DataFrame, cfg: Config) -> int:
    """Number of jobs in H: the first data.n0 rows, or (data.h_days) the rows that arrive in the first h_days days."""
    if cfg.data.h_days is None:
        return cfg.data.n0
    t = df[cfg.data.columns.datetime].to_numpy()
    return int(np.searchsorted(t, t[0] + np.timedelta64(int(round(cfg.data.h_days * 86400)), "s"), side="left"))


def _quota_fixed_point(df: pd.DataFrame, cfg: Config, a_cols: list[str],
                       tol: float = 0.01, max_iter: int = 50) -> tuple[np.ndarray, list]:
    """
    limits.job_quota_rule = share_of_C: Q = quota_share * C, C = capacity.f * D_bar(H after the quota Q).
    Start from Q = the 99th percentile of H's requests (df after R_max, no quota); repeat until Q moves by < tol
    on every resource. H, its span and the capped durations follow the same rules as load_dataset.
    """
    col, lim = cfg.data.columns, cfg.limits
    A = df[a_cols].to_numpy(dtype=float)
    t = df[col.datetime].to_numpy()
    d = np.minimum(np.maximum(df[col.duration_hours].to_numpy(dtype=float), cfg.time.duration_floor_hours),
                   lim.d_cap_hours)
    Q = np.quantile(A[:_n_history(df, cfg)], 0.99, axis=0)
    its = []
    for k in range(max_iter):
        idx = np.flatnonzero((A <= Q[None, :]).all(axis=1))
        h = idx[:_n_history(df.iloc[idx], cfg)]
        if not np.all(Q > 0) or h.size < 2:
            raise ValueError(f"quota fixed point collapsed (Q = {Q.tolist()}, {h.size} jobs in H after {k} steps): "
                             f"Q = {lim.quota_share} x f x D_bar(Q) has no positive solution at f = {cfg.capacity.f}")
        steps = _assign_real_tick_steps(t[h], t[h][0], cfg.time.tick_seconds)
        md = mean_demand(A[h], d[h], (steps[-1] - steps[0] + 1) * cfg.time.tick_seconds / 3600.0)
        C = cfg.capacity.f * md["D_bar"]
        Q_new = lim.quota_share * C
        change = np.abs(Q_new - Q) / Q
        its.append(dict(iteration=k, Q=Q.tolist(), n_kept=int(idx.size), n_H=int(h.size), D_bar=md["D_bar"].tolist(),
                        C=C.tolist(), Q_next=Q_new.tolist(), change=change.tolist()))
        Q = Q_new
        if np.all(change < tol):
            break
    return Q, its


def _compute_d_max_steps(T: int, cfg: Config) -> tuple[int, dict]:
    """d_max = min(time.d_max_hours, d_max_horizon_fraction * T) in steps; d_max_hours None -> the ceiling.
    limits.d_cap_hours set -> d_max = the cap (the ceiling rule is not used; the abort check still applies)."""
    ceiling = max(int(cfg.time.d_max_horizon_fraction * T), 1)
    if cfg.limits.d_cap_hours is not None:
        ceiling = max(int(np.rint(cfg.limits.d_cap_hours * 3600.0 / cfg.time.tick_seconds)), 1)
        candidate = ceiling
    elif cfg.time.d_max_hours is not None:
        candidate = max(int(np.rint(cfg.time.d_max_hours * 3600.0 / cfg.time.tick_seconds)), 1)
    else:
        candidate = ceiling
    d_max = min(candidate, ceiling)

    ratio = d_max / max(T, 1)
    diag = dict(
        d_max_hours=cfg.time.d_max_hours,
        candidate=candidate,
        horizon_ceiling=ceiling,
        chosen=d_max,
        ratio_to_T=ratio,
        clamped_by_ceiling=candidate > ceiling,
    )
    if ratio > cfg.time.d_max_ratio_abort:
        raise ValueError(
            f"d_max/T = {ratio:.3f} exceeds abort threshold "
            f"{cfg.time.d_max_ratio_abort}. d_max={d_max}, T={T}. "
            f"The theory requires d_max << T. Lower time.d_max_hours or "
            f"time.d_max_horizon_fraction."
        )
    return d_max, diag


# Public entry point
def load_dataset(cfg: Config) -> Dataset:
    df = _read_csv(cfg)
    col = cfg.data.columns
    resources = ["cpu", "ram"]
    a_cols = [col.requested_cpu, col.requested_ram]

    if cfg.limits.job_quota_rule == "share_of_C":
        df_r, _ = apply_r_max(df, cfg, a_cols, quota_values=(None, None))
        Q, its = _quota_fixed_point(df_r, cfg, a_cols)
        df, limit_report = apply_r_max(df, cfg, a_cols, quota_values=(float(Q[0]), float(Q[1])))
        limit_report.quota_iterations = its
    else:
        df, limit_report = apply_r_max(df, cfg, a_cols)

    n0 = _n_history(df, cfg)
    if n0 >= len(df):
        raise ValueError(
            f"n0={n0} >= rows after the size limit ({len(df)}). "
            "Lower n0 or raise limits.r_max_machines."
        )

    dts = df[col.datetime].to_numpy()

    # real-tick time model: one step = time.tick_seconds of wall clock
    all_steps = _assign_real_tick_steps(dts, dts[0], cfg.time.tick_seconds)

    # H = first n0 rows chronologically, stream = the rest. df is sorted by
    # datetime (_read_csv) and all_steps is row-aligned to df, so this is a
    # plain POSITIONAL split -- not a timestamp filter, which would lose or
    # duplicate jobs that share request_time with the H/stream boundary row.
    df_H = df.iloc[:n0].reset_index(drop=True)
    df_stream = df.iloc[n0:].reset_index(drop=True)
    steps_H = all_steps[:n0]
    # steps_stream_abs keeps the ORIGINAL absolute step index (shared origin
    # with steps_H, i.e. t0 = dts[0] of the full dataset) -- needed below to
    # build a capacity-sizing basis that spans H and stream on one timeline.
    # steps_stream is then re-based to 0 at the stream's own start, which is
    # what the rest of the online loop (T, stream.arrival_step) expects.
    steps_stream_abs = all_steps[n0:]

    assert len(df_H) + len(df_stream) == limit_report.n_after, (
        f"job accounting mismatch: len(df_H)+len(df_stream)="
        f"{len(df_H) + len(df_stream)} != limit_report.n_after="
        f"{limit_report.n_after} -- the H/stream split must partition the "
        f"filtered dataset with no gaps or overlap."
    )
    assert steps_H.max() < steps_stream_abs.min(), (
        f"causality violated: H's last absolute arrival_step ({steps_H.max()}) "
        f"is not strictly before stream's first ({steps_stream_abs.min()})."
    )

    stream_origin = steps_stream_abs[0]
    steps_stream = steps_stream_abs - stream_origin
    T = int(steps_stream[-1]) + 1
    H_span_steps = int(steps_H[-1] - steps_H[0]) + 1

    if cfg.hardware.fit_on == "full":
        A_fit = df[a_cols].to_numpy(dtype=float)
    else:
        A_fit = df_H[a_cols].to_numpy(dtype=float)
    catalogue = fit_hardware_catalogue(A_fit, cfg)

    d_max_steps, d_max_diag = _compute_d_max_steps(T, cfg)

    H = _build_job_arrays(df_H, cfg, resources,
                          d_max_steps, catalogue, steps_H)
    stream = _build_job_arrays(df_stream, cfg, resources,
                               d_max_steps, catalogue, steps_stream)

    # Capacity is sized on H + stream COMBINED, on the shared absolute
    # timeline (steps_H, steps_stream_abs) -- NOT on stream.arrival_step,
    # which is re-based to 0 and would misalign with steps_H. See
    # _size_capacity's docstring for why stream alone is not used.
    capacity_diag = None
    if cfg.capacity.mode == "mean_demand":
        # Phase 0 information only: H's jobs (after the limits), capped durations, H's span
        capacity_diag = mean_demand(H.A, H.duration_hours, H_span_steps * cfg.time.tick_seconds / 3600.0)
        capacity_diag["f"] = float(cfg.capacity.f)
        capacity = cfg.capacity.f * capacity_diag["D_bar"]
    else:
        capacity_basis = _OccupancyBasis(
            arrival_step=all_steps,
            duration_steps=np.concatenate([H.duration_steps, stream.duration_steps]),
            A=np.concatenate([H.A, stream.A], axis=0),
        )
        capacity = _size_capacity(capacity_basis, cfg)

        # Regression check: H+stream now partition the filtered dataset with no
        # gaps (positional split, no stratified pool), so capacity computed from
        # the H+stream basis must equal capacity computed directly on the WHOLE
        # filtered dataset's timeline in one shot.
        full_jobs = _build_job_arrays(df, cfg, resources, d_max_steps, catalogue, all_steps)
        full_basis = _OccupancyBasis(
            arrival_step=full_jobs.arrival_step,
            duration_steps=full_jobs.duration_steps,
            A=full_jobs.A,
        )
        capacity_full = _size_capacity(full_basis, cfg)
        assert np.allclose(capacity, capacity_full), (
            f"capacity from H+stream basis ({capacity}) does not match capacity "
            f"computed directly on the full filtered dataset's timeline "
            f"({capacity_full}) -- the H/stream split should partition df with "
            f"no gaps or reordering."
        )

    a_max = stream.A.max(axis=0)
    xi = float(np.max(a_max / capacity))

    if xi > cfg.limits.xi_warn_threshold:
        import warnings
        warnings.warn(
            f"xi = max_i a_max,i/c_i = {xi:.3f} exceeds "
            f"{cfg.limits.xi_warn_threshold} (~1/log T). The DLENT guarantee "
            f"needs xi small (constants.xi_override forces it in the online loop).",
            RuntimeWarning,
        )

    grid_data = None  
    if cfg.data.grid_csv and resolve_path(cfg.data.grid_csv).is_file():
        grid_data = load_grid_data(str(resolve_path(cfg.data.grid_csv)))

    return Dataset(
        H=H,
        stream=stream,
        resources=resources,
        capacity=capacity,
        a_max=a_max,
        d_max_steps=d_max_steps,
        tick_seconds=cfg.time.tick_seconds,
        T=T,
        n_arrivals_stream=stream.n,
        H_span_steps=H_span_steps,
        hw_catalogue=catalogue,
        limit_report=limit_report,
        d_max_diag=d_max_diag,
        xi=xi,
        grid_data=grid_data,
        capacity_diag=capacity_diag,
    )

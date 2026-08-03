"""
Data ingestion for DLENT-RT.

Responsibilities:
  1. Load the fused per-job CSV produced by the static DataGenerator.
  2. Convert job durations from native hours into an integer number of
     *arrival-steps* (decision A1: one step = one arrival).
  3. Split the first N0 rows into the historical batch H (offline, Phase 0)
     and the remainder into the online stream (never reuses H).
  4. Size cluster capacity c_i (fraction-of-total or absolute).
  5. Derive d_max, a_max, and the resource list I.

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

from .config import Config
from .preprocess import (
    HardwareCatalogue, OutlierReport, fit_hardware_catalogue, handle_outliers,
)


# --------------------------------------------------------------------------- #
# Job record: structure-of-arrays for vectorisation
# --------------------------------------------------------------------------- #
@dataclass
class JobArrays:
    """
    Structure-of-arrays view of a set of jobs, indexed 0..n-1 in arrival order.
    All arrays are 1-D and aligned. Resource arrays are (n, |I|).
    """
    n: int
    resources: list[str]                 # e.g. ["cpu", "ram"]  -> order of columns in A
    # per-job scalars
    q: np.ndarray                        # priority class (float)
    v_rate: np.ndarray                   # bid rate
    phi_rate: np.ndarray                 # Myerson virtual value rate
    w_kw: np.ndarray                     # power footprint (kW)
    elec_price: np.ndarray               # $/kWh at arrival
    carbon_intensity: np.ndarray         # gCO2/kWh at arrival
    duration_hours: np.ndarray           # native duration (hours)
    duration_steps: np.ndarray           # duration in arrival-steps (int, >=1, capped)
    # resource footprint matrix (n, |I|) — SNAPPED to canonical shapes
    A: np.ndarray
    # the canonical hardware-profile index per job (0..n_profiles-1)
    hw_idx: np.ndarray
    # bookkeeping
    collection_id: np.ndarray
    datetime: np.ndarray                 # np.datetime64

    def __post_init__(self):
        assert self.A.shape == (self.n, len(self.resources))

    def a_of(self, idx: int) -> np.ndarray:
        return self.A[idx]


@dataclass
class Dataset:
    """Everything the simulator needs, derived once from the CSV."""
    H: JobArrays                          # historical batch (offline, Phase 0)
    stream: JobArrays                     # online stream
    resources: list[str]                  # resource names, canonical order
    capacity: np.ndarray                  # c_i, shape (|I|,)
    a_max: np.ndarray                     # max single-job request per resource, (|I|,)
    d_max_steps: int                      # duration cap in steps
    arrivals_per_hour: float              # conversion factor used for hours->steps
    T: int                                # online horizon (len(stream))
    # preprocessing artefacts (for logging / reproducibility)
    hw_catalogue: HardwareCatalogue
    outlier_report: OutlierReport
    d_max_diag: dict                      # how d_max was determined + safety ratio
    xi: float                             # max_i a_max,i / c_i (post-handling)

    @property
    def n_resources(self) -> int:
        return len(self.resources)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _read_csv(cfg: Config) -> pd.DataFrame:
    col = cfg.data.columns
    path = Path(cfg.data.batch_csv)
    if not path.exists():
        raise FileNotFoundError(
            f"batch CSV not found at '{path}'. Set data.batch_csv in the config."
        )
    df = pd.read_csv(path)
    # chronological order = arrival order (the static generator already sorts,
    # but we enforce it so step index == arrival index unambiguously).
    df[col.datetime] = pd.to_datetime(df[col.datetime])
    df = df.sort_values(col.datetime).reset_index(drop=True)
    return df


def _estimate_arrivals_per_hour(df: pd.DataFrame, cfg: Config) -> float:
    """
    Arrivals per hour over the online span. This is the conversion factor from
    a duration in hours to a duration in arrival-steps:

        duration_steps = round(duration_hours * arrivals_per_hour)

    Rationale (decision A1): one step = one arrival. A job that lasts D hours
    occupies resources for however many *arrivals* land in that wall-clock
    window. On average that is D * (arrivals per hour). We use the global mean
    rate; a per-window rate is possible later but the global rate keeps the
    step<->hour map stationary, which the tests' sqrt(t) envelopes assume.
    """
    col = cfg.data.columns
    t = df[col.datetime]
    span_hours = (t.iloc[-1] - t.iloc[0]) / np.timedelta64(1, "h")
    if span_hours <= 0:
        # all arrivals at the same timestamp (degenerate / tiny test set):
        # fall back to 1 arrival per hour so durations map to themselves.
        return 1.0
    return float(len(df) / span_hours)


def _hours_to_steps(
    duration_hours: np.ndarray, arrivals_per_hour: float, d_max_steps: Optional[int]
) -> np.ndarray:
    steps = np.rint(duration_hours * arrivals_per_hour).astype(np.int64)
    steps = np.maximum(steps, 1)            # every admitted job occupies >=1 step
    if d_max_steps is not None:
        steps = np.minimum(steps, d_max_steps)
    return steps


def _build_job_arrays(
    df: pd.DataFrame, cfg: Config, resources: list[str],
    arrivals_per_hour: float, d_max_steps: Optional[int],
    catalogue: HardwareCatalogue,
) -> JobArrays:
    col = cfg.data.columns
    n = len(df)
    A_raw = np.column_stack([
        df[col.a_cpu].to_numpy(dtype=float),
        df[col.a_ram].to_numpy(dtype=float),
    ])
    # snap every request to its canonical VM shape
    hw_idx = catalogue.assign(A_raw)
    A = catalogue.centroids[hw_idx]
    dur_hours = df[col.duration_hours].to_numpy(dtype=float)
    dur_steps = _hours_to_steps(dur_hours, arrivals_per_hour, d_max_steps)
    return JobArrays(
        n=n,
        resources=resources,
        q=df[col.q].to_numpy(dtype=float),
        v_rate=df[col.v_rate].to_numpy(dtype=float),
        phi_rate=df[col.phi_rate].to_numpy(dtype=float),
        w_kw=df[col.w_kw].to_numpy(dtype=float),
        elec_price=df[col.elec_price].to_numpy(dtype=float),
        carbon_intensity=df[col.carbon_intensity].to_numpy(dtype=float),
        duration_hours=dur_hours,
        duration_steps=dur_steps,
        A=A,
        hw_idx=hw_idx,
        collection_id=df[col.collection_id].to_numpy(),
        datetime=df[col.datetime].to_numpy(),
    )


def _peak_concurrent_occupancy(stream: JobArrays) -> np.ndarray:
    """
    Peak concurrent physical occupancy per resource over the online stream,
    under one-arrival-per-step semantics. A job arriving at step s occupies
    resources during steps [s, s + duration_steps). We sweep a difference array.

    Returns (|I|,) peak occupancy.
    """
    n = stream.n
    n_res = stream.A.shape[1]
    # difference array over steps [0, n]; job s adds A at s, removes at s+dur
    horizon = n + int(stream.duration_steps.max()) + 1
    delta = np.zeros((horizon + 1, n_res))
    for s in range(n):
        d = int(stream.duration_steps[s])
        delta[s] += stream.A[s]
        delta[s + d] -= stream.A[s]
    occ = np.cumsum(delta, axis=0)
    return occ.max(axis=0)


def _size_capacity(stream: JobArrays, cfg: Config) -> np.ndarray:
    """
    c_i for each resource.

    fraction_of_volume (default, matches static sim):
        c_i = fraction * sum_j (A_ij * D_j)   [resource volume, core-steps]
        This is b_i = rho * sum_j V_ij. It puts capacity on the same scale as
        the fluid LP's expected consumption, so the LP binds.

    fraction_of_peak:
        c_i = fraction * peak concurrent occupancy of resource i over the stream.
        Physically meaningful "cluster is X% of peak demand"; makes Test 1 bind.

    absolute:
        taken directly from config (real deployments).
    """
    if cfg.capacity.mode == "fraction_of_volume":
        volume = (stream.A * stream.duration_steps[:, None]).sum(axis=0)
        return cfg.capacity.fraction * volume
    elif cfg.capacity.mode == "fraction_of_peak":
        peak = _peak_concurrent_occupancy(stream)
        return cfg.capacity.fraction * peak
    else:  # absolute
        return np.array(
            [cfg.capacity.absolute.cpu, cfg.capacity.absolute.ram], dtype=float
        )


def _compute_d_max_steps(
    duration_hours_all: np.ndarray, arrivals_per_hour: float, T: int, cfg: Config
) -> tuple[int, dict]:
    """
    d_max in steps, enforced to be << T.

    Candidate = configured quantile of step-durations. Then a HARD CEILING at
    d_max_horizon_fraction * T is applied (the theory requires d_max << T; the
    warm-up window is d_max steps, so d_max >= T would mean no phase ever exits
    warm-up). If the quantile-based candidate already sits below the ceiling we
    keep it; otherwise we clamp to the ceiling.

    Returns (d_max_steps, diagnostics). Aborts with a clear message if even the
    ceiling leaves d_max/T above d_max_ratio_abort (should not happen given the
    ceiling, but guards against misconfiguration).
    """
    steps = _hours_to_steps(duration_hours_all, arrivals_per_hour, d_max_steps=None)
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


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def load_dataset(cfg: Config) -> Dataset:
    df = _read_csv(cfg)
    col = cfg.data.columns
    resources = ["cpu", "ram"]            # A3: two resources; model generalises
    a_cols = [col.a_cpu, col.a_ram]

    # --- 1. Outlier handling (thresholds computed on the whole trace) --------
    df, outlier_report = handle_outliers(df, cfg, a_cols, col.duration_hours)

    n0 = cfg.data.n0
    if n0 >= len(df):
        raise ValueError(
            f"n0={n0} >= rows after outlier handling ({len(df)}). "
            "Lower n0 or relax outlier thresholds."
        )

    # --- 2. Hardware catalogue (fit on full trace or H) ----------------------
    if cfg.hardware.fit_on == "full":
        A_fit = df[a_cols].to_numpy(dtype=float)
    else:  # "H"
        A_fit = df.iloc[:n0][a_cols].to_numpy(dtype=float)
    catalogue = fit_hardware_catalogue(A_fit, cfg)

    # --- 3. Conversion factor and horizon ------------------------------------
    arrivals_per_hour = _estimate_arrivals_per_hour(df, cfg)
    T = len(df) - n0

    # --- 4. d_max with hard T-ceiling ---------------------------------------
    d_max_steps, d_max_diag = _compute_d_max_steps(
        df[col.duration_hours].to_numpy(dtype=float), arrivals_per_hour, T, cfg,
    )

    # --- 5. Split and build --------------------------------------------------
    df_H = df.iloc[:n0].reset_index(drop=True)
    df_stream = df.iloc[n0:].reset_index(drop=True)
    H = _build_job_arrays(df_H, cfg, resources, arrivals_per_hour, d_max_steps, catalogue)
    stream = _build_job_arrays(df_stream, cfg, resources, arrivals_per_hour, d_max_steps, catalogue)

    # --- 6. Capacity and a_max (on SNAPPED requests) -------------------------
    capacity = _size_capacity(stream, cfg)
    a_max = stream.A.max(axis=0)          # over snapped shapes -> bounded catalogue
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

    return Dataset(
        H=H,
        stream=stream,
        resources=resources,
        capacity=capacity,
        a_max=a_max,
        d_max_steps=d_max_steps,
        arrivals_per_hour=arrivals_per_hour,
        T=T,
        hw_catalogue=catalogue,
        outlier_report=outlier_report,
        d_max_diag=d_max_diag,
        xi=xi,
    )

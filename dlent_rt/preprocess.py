"""
Trace preprocessing: hardware-profile discretization and outlier handling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .config import Config


@dataclass
class HardwareCatalogue:
    """A fixed catalogue of canonical VM shapes and an assignment function."""
    centroids: np.ndarray          
    log_space: bool

    @property
    def n_profiles(self) -> int:
        return len(self.centroids)

    def assign(self, A: np.ndarray) -> np.ndarray:
        """
        Snap each request row to the nearest canonical shape. Returns the index
        of the assigned centroid per job (0..n_profiles-1). Distance is computed
        in the same space the catalogue was fitted in (log or linear).
        """
        X = np.log(np.clip(A, 1e-9, None)) if self.log_space else A
        C = np.log(np.clip(self.centroids, 1e-9, None)) if self.log_space else self.centroids
        # squared euclidean to each centroid, vectorised
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)   # (n, n_profiles)
        return d2.argmin(axis=1)


def _kmeans(X: np.ndarray, k: int, seed: int, iters: int = 100) -> np.ndarray:
    """
    Minimal k-means++ (no sklearn dependency). X is (n, d). Returns (k, d)
    centroids in the space of X.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    k = min(k, n)
    # k-means++ init
    centers = [X[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min(
            [((X - c) ** 2).sum(axis=1) for c in centers], axis=0
        )
        probs = d2 / max(d2.sum(), 1e-12)
        centers.append(X[rng.choice(n, p=probs)])
    C = np.array(centers)
    # Lloyd iterations
    for _ in range(iters):
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        labels = d2.argmin(axis=1)
        newC = np.array([
            X[labels == j].mean(axis=0) if np.any(labels == j) else C[j]
            for j in range(k)
        ])
        if np.allclose(newC, C):
            C = newC
            break
        C = newC
    return C


def fit_hardware_catalogue(A_fit: np.ndarray, cfg: Config) -> HardwareCatalogue:
    """Fit the canonical VM-shape catalogue on A_fit (native units): k-means++
    in (optionally) log space, the only method."""
    hw = cfg.hardware
    X = np.log(np.clip(A_fit, 1e-9, None)) if hw.log_space else A_fit
    C = _kmeans(X, hw.n_profiles, seed=cfg.run.seed)
    centroids = np.exp(C) if hw.log_space else C
    return HardwareCatalogue(centroids=centroids, log_space=hw.log_space)


# --------------------------------------------------------------------------- #
# Outlier handling
# --------------------------------------------------------------------------- #
@dataclass
class OutlierReport:
    """What the outlier step did, for logging and diagnostics."""
    a_max_thresholds: np.ndarray       # per-resource request cutoff (native units)
    d_max_hours_threshold: float       # duration cutoff (hours)
    n_dropped_resource: int
    n_dropped_duration: int
    n_dropped_total: int
    n_before: int
    n_after: int
    policy: str


def handle_outliers(
    df: pd.DataFrame, cfg: Config, a_cols: list[str], d_col: str
) -> tuple[pd.DataFrame, OutlierReport]:
    """
    Apply the outlier policy to a dataframe. Thresholds are computed on `df`.

    drop: rows exceeding any resource cutoff OR the duration cutoff are removed.
    clip: those values are clamped to the cutoff (values distorted, count kept).
    none: no filter; every row is kept (thresholds reported as NaN).
    r_max: rows with any request > outliers.r_max_machines are removed; durations are not filtered.
    """
    o = cfg.outliers
    n_before = len(df)
    if o.policy == "none":
        report = OutlierReport(
            a_max_thresholds=np.full(len(a_cols), np.nan), d_max_hours_threshold=float("nan"),
            n_dropped_resource=0, n_dropped_duration=0, n_dropped_total=0,
            n_before=n_before, n_after=n_before, policy="none",
        )
        return df.reset_index(drop=True), report
    if o.policy == "r_max":
        over = (df[a_cols].to_numpy(dtype=float) > o.r_max_machines).any(axis=1)
        report = OutlierReport(
            a_max_thresholds=np.full(len(a_cols), float(o.r_max_machines)), d_max_hours_threshold=float("nan"),
            n_dropped_resource=int(over.sum()), n_dropped_duration=0, n_dropped_total=int(over.sum()),
            n_before=n_before, n_after=int((~over).sum()), policy="r_max",
        )
        return df.loc[~over].reset_index(drop=True), report

    a_thresh = np.array([df[c].quantile(o.a_max_quantile) for c in a_cols])
    d_thresh = float(df[d_col].quantile(o.d_max_quantile))

    A = df[a_cols].to_numpy(dtype=float)
    D = df[d_col].to_numpy(dtype=float)

    over_resource = (A > a_thresh[None, :]).any(axis=1)
    over_duration = D > d_thresh

    if o.policy == "drop":
        keep = ~(over_resource | over_duration)
        out = df.loc[keep].reset_index(drop=True)
        report = OutlierReport(
            a_max_thresholds=a_thresh,
            d_max_hours_threshold=d_thresh,
            n_dropped_resource=int(over_resource.sum()),
            n_dropped_duration=int(over_duration.sum()),
            n_dropped_total=int((over_resource | over_duration).sum()),
            n_before=n_before, n_after=len(out), policy="drop",
        )
        return out, report
    else:  # clip
        out = df.copy()
        for j, c in enumerate(a_cols):
            out[c] = out[c].clip(upper=a_thresh[j])
        out[d_col] = out[d_col].clip(upper=d_thresh)
        report = OutlierReport(
            a_max_thresholds=a_thresh,
            d_max_hours_threshold=d_thresh,
            n_dropped_resource=int(over_resource.sum()),   # count affected
            n_dropped_duration=int(over_duration.sum()),
            n_dropped_total=int((over_resource | over_duration).sum()),
            n_before=n_before, n_after=len(out), policy="clip",
        )
        return out, report

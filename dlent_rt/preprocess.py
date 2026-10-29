"""
Trace preprocessing: hardware profiles (they build the type id only) and the job quota.
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
# Job quota
# --------------------------------------------------------------------------- #
@dataclass
class LimitReport:
    """What the job quota did, for logging and diagnostics."""
    n_before: int
    n_after: int
    n_quota_rejected: int = 0           # rejected by the job quota
    quota: Optional[tuple] = None       # (Q_cpu, Q_ram), machines
    quota_iterations: Optional[list] = None   # one dict per fixed-point step


def apply_quota(df: pd.DataFrame, a_cols: list[str], quota: tuple) -> tuple[pd.DataFrame, LimitReport]:
    """Reject the jobs over the job quota (CPU > Q_cpu or RAM > Q_ram)."""
    A = df[a_cols].to_numpy(dtype=float)
    over = (A[:, 0] > quota[0]) | (A[:, 1] > quota[1])
    report = LimitReport(n_before=len(df), n_after=int((~over).sum()), n_quota_rejected=int(over.sum()),
                         quota=(float(quota[0]), float(quota[1])))
    return df.loc[~over].reset_index(drop=True), report

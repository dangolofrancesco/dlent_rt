"""
Benchmarks for the regret of an online run (week-30 Stage D). Every number is scored with the
single job-level reward dlent_rt.reward (R_job, $ per job), on the online STREAM (the
outlier-filtered jobs after H), with Phase 0's bid model and frozen s_C.

    b = Benchmarks(ds, state0, cfg)          # or Benchmarks.from_model(theory_model)
    b.score(admitted_mask)    sum of R_job over the admitted stream jobs                 [$ per run]
    b.fluid_opt()             lambda_hat_ref * T: the full-stream LP-RS on the empirical
                              type statistics of the whole stream (true durations, every
                              arrival, no fallback). A FLUID quantity on type means, an upper
                              reference for algorithms that decide by type; not a feasible
                              schedule. Equals TheoryModel's summary["bench_global"].      [$ per run]
    b.job_level_opt()         offline job-level OPT (ORACLE): max sum_j R_job,j x_j, x in {0,1},
                              s.t. at every arrival tick the occupancy of the chosen jobs is <= c
                              (HiGHS MILP). Kept as the reference for the discretization loss
                              (fluid on types vs per-job decisions); also returns the LP
                              relaxation and the no-capacity bound sum_j R_job,j^+.          [$ per run]
    regret(alg_total, benchmark_total) = benchmark_total - alg_total                        [$ per run]

Units: R_job is in dollars per job; totals are dollars over the whole stream (T ticks). A
per-tick quantity (lambda) times T gives the same unit.
"""
from __future__ import annotations

from dataclasses import replace
from functools import cached_property

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from . import reward
from .models.core.reinit import build_window_state
from .models.core.stream import StreamData, build_stream


def regret(alg_total: float, benchmark_total: float) -> float:
    """Regret of an algorithm against a benchmark, both in $ over the same stream."""
    return float(benchmark_total) - float(alg_total)


class Benchmarks:
    def __init__(self, ds, state0, cfg, stream: StreamData | None = None):
        self.ds, self.state0, self.cfg = ds, state0, cfg
        self.S = build_stream(ds, cfg) if stream is None else stream
        self.c = np.asarray(ds.capacity, float)
        self.T = int(ds.T)

    @classmethod
    def from_model(cls, model) -> "Benchmarks":
        """Benchmarks on the stream and Phase-0 state of an online model (TheoryModel)."""
        return cls(model.ds, model.state0, model.cfg, stream=model.stream)

    # ------------------------------------------------------------------ #
    @cached_property
    def terms(self) -> reward.JobTerms:
        """Job reward terms of every stream job (Phase 0's bid model and s_C)."""
        return reward.job_reward(self.S, self.state0, self.cfg.objective)

    @property
    def R(self) -> np.ndarray:
        """R_job of every stream job [$]."""
        return self.terms.R

    def score(self, admitted_mask) -> float:
        """Total reward of an admitted set (boolean mask over the stream) [$]."""
        m = np.asarray(admitted_mask, bool)
        assert m.shape == (self.S.n,)
        return float(self.R[m].sum())

    # ------------------------------------------------------------------ #
    @cached_property
    def reference_state(self):
        """Full-stream hindsight state (ORACLE): every arrival, true D, no fallback, F refit on the
        whole stream when the bid distribution is not known. tau* and lambda_hat_ref come from it."""
        cfg_h = replace(self.cfg, phase_transition=replace(self.cfg.phase_transition, n_min=2))
        idx = np.arange(self.S.n)
        st, _ = build_window_state(self.S, idx, idx, self.T, self.state0, self.ds, cfg_h, fallback=False)
        return st

    def fluid_opt(self) -> float:
        """lambda_hat_ref * T [$]: the full-stream LP-RS optimum (per tick) times the horizon."""
        return float(self.reference_state.lam_star) * self.T

    def job_level_opt(self, time_limit: float = 300.0) -> dict:
        """Offline job-level OPT (MILP) over the jobs with R_job > 0 (others are never chosen)."""
        S, R = self.S, self.R
        cand = np.flatnonzero(R > 0)
        n = cand.size
        t, D, A = S.tick[cand], S.D[cand], S.A[cand]
        # job j occupies the arrival rows k = j .. last[j] (arrivals with t_k < t_j + D_j)
        last = np.searchsorted(t, t + D, side="left") - 1
        lengths = last - np.arange(n) + 1
        cols = np.repeat(np.arange(n), lengths)
        start = np.repeat(np.cumsum(lengths) - lengths, lengths)
        rows = cols + (np.arange(cols.size) - start)
        M = sp.vstack([sp.csr_matrix((A[cols, r], (rows, cols)), shape=(n, n))
                       for r in range(A.shape[1])]).tocsr()                     # one block per resource
        ub = np.repeat(self.c, n)
        lp = linprog(-R[cand], A_ub=M, b_ub=ub, bounds=(0, 1), method="highs")
        ip = milp(-R[cand], constraints=LinearConstraint(M, -np.inf, ub), integrality=np.ones(n),
                  bounds=Bounds(0, 1), options=dict(time_limit=time_limit))
        x = np.zeros(S.n, bool)
        x[cand] = np.round(ip.x).astype(bool)
        return dict(x=x, R_ip=float(R[x].sum()), R_lp=-float(lp.fun), R_nocap=float(R[cand].sum()),
                    gap=float(getattr(ip, "mip_gap", np.nan)), status=ip.message)

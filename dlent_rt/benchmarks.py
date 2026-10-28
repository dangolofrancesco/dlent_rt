"""
Benchmarks for the regret of an online run (week-30 Stage D). Every number is scored with the
single job-level reward dlent_rt.reward (R_job, $ per job), on the online STREAM (the
jobs after H, within the size limit R_max), with Phase 0's bid model and frozen s_C.

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
    b.job_opt_bucketed(source, duration)
                              the same per-tick-capacity MILP on BUCKETED jobs: the solver sees
                              each job's TYPE reward r~_k(j) instead of R_job (the professor's
                              benchmark between job OPT and fluid OPT, week 30). source =
                              "phase0" (Phase 0's types and r~ from H, the estimate the static /
                              online policies use) or "reference" (the full-stream reference
                              state of fluid_opt: its refit grid and stream means). duration =
                              "true" (variant A: true D in the occupancy) or "type" (variant B:
                              the type's d_bar). Reports the REALISED sum_j R_job,j x_j of the
                              chosen set (comparable with every other number here) and the
                              BELIEVED sum_j r~_k(j) x_j (the MILP's own objective).         [$ per run]
    regret(alg_total, benchmark_total) = benchmark_total - alg_total                        [$ per run]

Units: R_job is in dollars per job; totals are dollars over the whole stream (T ticks). A
per-tick quantity (lambda) times T gives the same unit.
"""
from __future__ import annotations

import time
from dataclasses import replace
from functools import cached_property

import numpy as np
import scipy.sparse as sp

from . import reward
from .models.core.reinit import build_window_state
from .models.core.stream import StreamData, build_stream
from .lp_oracle import solve_lp_rs
from .phase0 import assign_type_ids


def regret(alg_total: float, benchmark_total: float) -> float:
    """Regret of an algorithm against a benchmark, both in $ over the same stream."""
    return float(benchmark_total) - float(alg_total)


def _highs_max(M, obj, ub, integer: bool, time_limit: float) -> dict:
    """max obj^T x, M x <= ub, 0 <= x <= 1 (x integer if `integer`) on a NEW HiGHS instance with its own time
    limit, so every solve gets its full budget (no clock shared with earlier solves)."""
    import highspy
    M = sp.csc_matrix(M)
    n = M.shape[1]
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(time_limit))
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = n, M.shape[0]
    lp.sense_ = highspy.ObjSense.kMaximize
    lp.col_cost_ = np.asarray(obj, float)
    lp.col_lower_, lp.col_upper_ = np.zeros(n), np.ones(n)
    lp.row_lower_, lp.row_upper_ = np.full(M.shape[0], -highspy.kHighsInf), np.asarray(ub, float)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_, lp.a_matrix_.index_ = M.indptr.astype(np.int32), M.indices.astype(np.int32)
    lp.a_matrix_.value_ = M.data.astype(float)
    lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = n, M.shape[0]
    if integer:
        lp.integrality_ = [highspy.HighsVarType.kInteger] * n
    h.passModel(lp)
    t0 = time.perf_counter()
    h.run()
    wall = time.perf_counter() - t0
    info = h.getInfo()
    x = np.array(h.getSolution().col_value, float) if info.primal_solution_status > 0 else np.zeros(n)
    return dict(x=x, obj=float(np.asarray(obj, float) @ x), status=h.modelStatusToString(h.getModelStatus()),
                gap=float(info.mip_gap) if integer else 0.0,
                dual_bound=float(info.mip_dual_bound) if integer else float(info.objective_function_value),
                runtime_s=wall, highs_run_time_s=float(h.getRunTime()))


def occupancy_milp(t, D, A, obj, c, time_limit: float = 300.0, lp_time_limit: float | None = None,
                   tail_step: int | None = None) -> dict:
    """
    max obj^T x, x in {0,1}^n, s.t. at the arrival tick of every job k the occupancy of the chosen
    jobs is <= c: job j occupies the arrival rows k = j .. last[j] (arrivals with t_k < t_j + D_j).
    t must be sorted ascending (stream order). Returns x (bool), the MILP and LP-relaxation
    objective values, gap, status and the MILP runtime [s].

    The LP relaxation and the MILP each run on their own HiGHS instance with their own time limit
    (lp_time_limit, default = time_limit), so neither uses the other's budget.
    tail_step (ticks, None = off): also check the capacity at t_last + k * tail_step (k = 1, 2, ...) until the
    last candidate job ends, so the jobs that run past the last arrival (the end of a window) still use capacity.
    """
    t, D, A, obj = (np.asarray(a) for a in (t, D, A, obj))
    n = t.size
    last = np.searchsorted(t, t + D, side="left") - 1
    lengths = last - np.arange(n) + 1
    cols = np.repeat(np.arange(n), lengths)
    start = np.repeat(np.cumsum(lengths) - lengths, lengths)
    rows = cols + (np.arange(cols.size) - start)
    n_rows = n
    n_tail = 0
    if tail_step is not None and n:
        taus = np.arange(t[-1] + tail_step, (t + D).max(), tail_step)
        n_tail = taus.size
        if n_tail:
            # job j runs at tau (> every arrival) iff t_j + D_j > tau: rows n + k for k < its last tail check
            k_end = np.searchsorted(taus, t + D, side="left")              # number of tail checks job j covers
            tc = np.repeat(np.arange(n), k_end)
            tr = n + (np.arange(tc.size) - np.repeat(np.cumsum(k_end) - k_end, k_end))
            cols, rows = np.concatenate([cols, tc]), np.concatenate([rows, tr])
            n_rows = n + n_tail
    M = sp.vstack([sp.csr_matrix((A[cols, r], (rows, cols)), shape=(n_rows, n))
                   for r in range(A.shape[1])]).tocsr()                     # one block per resource
    ub = np.repeat(np.asarray(c, float), n_rows)
    lp = _highs_max(M, obj, ub, integer=False, time_limit=time_limit if lp_time_limit is None else lp_time_limit)
    ip = _highs_max(M, obj, ub, integer=True, time_limit=time_limit)
    return dict(x=np.round(ip["x"]).astype(bool), ip_obj=ip["obj"], lp_obj=lp["obj"], gap=ip["gap"],
                status=ip["status"], runtime_s=ip["runtime_s"], highs_run_time_s=ip["highs_run_time_s"],
                dual_bound=ip["dual_bound"], lp_runtime_s=lp["runtime_s"], n_tail_checks=int(n_tail),
                nonzeros=int(M.nnz))


def peak_occupancy(t, D, A) -> np.ndarray:
    """Peak over time of the occupancy of jobs [t, t + D) (releases before admissions at a tick)."""
    t, D, A = np.asarray(t), np.asarray(D), np.asarray(A, float)
    if t.size == 0:
        return np.zeros(A.shape[1] if A.ndim == 2 else 0)
    times = np.r_[t, t + D]
    d = np.vstack([A, -A])
    order = np.lexsort((np.r_[np.ones(t.size), np.zeros(t.size)], times))   # -A (0) before +A (1)
    return np.cumsum(d[order], axis=0).max(axis=0)


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
        o = occupancy_milp(S.tick[cand], S.D[cand], S.A[cand], R[cand], self.c, time_limit)
        x = np.zeros(S.n, bool)
        x[cand] = o["x"]
        return dict(x=x, R_ip=float(R[x].sum()), R_lp=o["lp_obj"], R_nocap=float(R[cand].sum()),
                    gap=o["gap"], status=o["status"], runtime_s=o["runtime_s"])

    # ------------------------------------------------------------------ #
    def bucket_types(self, source: str = "phase0") -> dict:
        """
        Type of every stream job under `source` ("phase0": Phase 0's grid / types / r~ from H;
        "reference": the reference state's refit grid and stream means) and its bucket reward
        r~_k(j) and type-average duration d_bar_k(j). A job whose type has no observation in that
        state (p = 0; or no type) gets r~ = 0 and is never chosen ("unseen").
        """
        st = {"phase0": self.state0, "reference": self.reference_state}[source]
        S = self.S
        k = st.grid.assign(S.v, st.myerson)
        j = assign_type_ids(st.type_space, np.asarray(k), S.priority, S.hw_idx)
        jj = np.maximum(j, 0)
        seen = (j >= 0) & (np.asarray(st.p)[jj] > 0)
        return dict(type=j, seen=seen, r=np.where(seen, st.r_tilde[jj], 0.0),
                    d_bar=np.where(seen, st.d_bar[jj], 0.0))

    def job_opt_bucketed(self, source: str = "phase0", duration: str = "true",
                         time_limit: float = 300.0, tail_step: int | None = None) -> dict:
        """
        Job-level MILP on BUCKETED jobs: max sum_j r~_k(j) x_j over the jobs with r~ > 0, same
        per-tick capacity, stream and real A as job_level_opt. duration "true"
        (variant A) uses each job's true D in the occupancy, "type" (variant B) the type's d_bar
        (rounded, >= 1 tick); B's set may then exceed c under the TRUE durations, reported as
        peak_true_occ / c. R_real = sum_j R_job,j x_j (realised), R_believed = sum_j r~ x_j.
        """
        S, R = self.S, self.R
        b = self.bucket_types(source)
        r = b["r"]
        cand = np.flatnonzero(r > 0)
        D_occ = S.D[cand] if duration == "true" else np.maximum(np.rint(b["d_bar"][cand]), 1).astype(np.int64)
        o = occupancy_milp(S.tick[cand], D_occ, S.A[cand], r[cand], self.c, time_limit, tail_step=tail_step)
        x = np.zeros(S.n, bool)
        x[cand] = o["x"]
        peak = peak_occupancy(S.tick[x], S.D[x], S.A[x]) / self.c
        unseen = ~b["seen"]
        return dict(x=x, source=source, duration=duration, R_real=float(R[x].sum()),
                    R_believed=float(r[x].sum()), R_lp_believed=o["lp_obj"],
                    n_candidates=int(cand.size), n_unseen_jobs=int(unseen.sum()),
                    n_unseen_types=int(np.unique(b["type"][unseen & (b["type"] >= 0)]).size),
                    n_chosen_R_neg=int((R[x] < 0).sum()), peak_true_occ_over_c=peak.tolist(),
                    feasible_true=bool(np.all(peak <= 1 + 1e-9)),
                    gap=o["gap"], status=o["status"], runtime_s=o["runtime_s"], dual_bound=o["dual_bound"],
                    lp_runtime_s=o["lp_runtime_s"], n_tail_checks=o["n_tail_checks"], nonzeros=o["nonzeros"])

    # ------------------------------------------------------------------ #
    def feasible_ceiling(self) -> float:
        """sum R_job^+ over the stream jobs with A_ij <= c_i on every resource: the most ANY policy can earn [$]."""
        fits = ~np.any(self.S.A > self.c[None, :], axis=1)
        return float(np.maximum(self.R, 0)[fits].sum())

    def fluid_opt_feasible_types(self) -> dict:
        """The fluid OPT (reference-state LP-RS x T) with every type whose request A_type exceeds c on some
        resource removed (p = 0): the fluid benchmark without the load that can never run."""
        ref = self.reference_state
        feas = ~np.any(ref.A_type > self.c[None, :], axis=1)
        lp = solve_lp_rs(p=ref.p * feas, r=ref.r_tilde, v=ref.v, c=self.c, presolve=self.cfg.lp.presolve)
        return dict(value=float(lp.optimum) * self.T, tau=np.asarray(lp.tau).tolist(),
                    n_types_removed=int((~feas & (ref.p > 0)).sum()), n_types_populated=int((ref.p > 0).sum()))

    def ladder(self, model, result, time_limit: float = 300.0) -> dict:
        """
        The benchmark ladder of an online run (model = the TheoryModel, result = model.run()), every row scored
        with R_job on the same stream and physical gate [$ per run]:
        no-capacity bound, feasible ceiling, job-level MILP, bucketed MILP A (reference / Phase 0 buckets),
        fluid OPT, fluid OPT on feasible types, G2, G1, static Phase 0, static Phase 0 + thinning at the cap
        p = (1 - xi) / (1 + eps_D) (the run's common random numbers), DLENT-RT.
        Checks: DLENT-RT's score = its R_total; a bucketed A set is feasible and its realised value is at most
        the job MILP (within the MILP's own gap). G1's decision-error rate (sign R_job != sign r~) is returned.
        """
        from .models.policy import JobGreedyPolicy, TypeGreedyPolicy, run_policy, run_static

        S, R = self.S, self.R
        adm = np.zeros(S.n, bool)
        adm[[a["i"] for a in result.arrivals if a["admit"]]] = True
        assert np.isclose(self.score(adm), result.summary["R_total"], rtol=1e-10)
        job = self.job_level_opt(time_limit=time_limit)
        buck = {src: self.job_opt_bucketed(src, "true", time_limit=time_limit) for src in ("reference", "phase0")}
        bound = job["R_ip"] * (1 + max(job["gap"], 0)) + 1e-6
        for src, o in buck.items():
            assert o["feasible_true"], src
            assert o["R_real"] <= bound, (src, o["R_real"], job["R_ip"], job["gap"])
        p_cap = (1 - model.ep.xi) / (1 + model.ep.eps_D)
        g1 = TypeGreedyPolicy.from_state(self.state0, S, R, self.ds.H_span_steps)
        sims = {"G2 job greedy": run_policy(S, self.c, JobGreedyPolicy(R)),
                "G1 type greedy": run_policy(S, self.c, g1),
                "static Phase 0": run_static(S, self.c, self.state0),
                "static Phase 0 + thinning": run_static(S, self.c, self.state0, thin_p=p_cap, u_rand=model.u_rand)}
        ff = self.fluid_opt_feasible_types()
        rows = {"job-level, no capacity": job["R_nocap"], "feasible ceiling": self.feasible_ceiling(),
                "job-level MILP": job["R_ip"],
                "bucketed MILP A (reference buckets)": buck["reference"]["R_real"],
                "bucketed MILP A (Phase 0 buckets)": buck["phase0"]["R_real"],
                "fluid OPT": self.fluid_opt(), "fluid OPT on feasible types": ff["value"],
                **{k: self.score(v.admit) for k, v in sims.items()}, "DLENT-RT": self.score(adm)}
        ok = np.isfinite(g1.r_dec)
        return dict(rows=rows, admitted={k: int(v.admit.sum()) for k, v in sims.items()} | {"DLENT-RT": int(adm.sum())},
                    job=dict((k, v) for k, v in job.items() if k != "x"),
                    bucketed={s: {k: v for k, v in o.items() if k != "x"} for s, o in buck.items()},
                    fluid_feasible=ff, p_cap=float(p_cap),
                    g1_decision_error=float(np.mean(np.sign(R[ok]) != np.sign(g1.r_dec[ok]))) if ok.any() else 0.0)

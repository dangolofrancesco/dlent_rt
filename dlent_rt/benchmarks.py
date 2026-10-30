"""
Benchmarks for the regret of an online run (D6). Every number is scored with the single job-level reward
dlent_rt.reward (R_job, $ per job), on the online STREAM (the jobs after H, within the job quota), with Phase 0's
bid model and frozen s_C. Kept for the theoretical model: fluid OPT, fluid on feasible types (upper bounds), the
discretization cost (two job-level MILPs, discretization_milps), the static Phase 0 policy and the no-learning model
(models.policy). Practical only (code kept, not run in the model check): Greedy-Pos and the type greedy (G1).

    b = Benchmarks(ds, state0, cfg)          # or Benchmarks.from_model(theory_model)
    b.score(admitted_mask)    sum of R_job over the admitted stream jobs                 [$ per run]
    b.fluid_opt()             lambda_hat_ref * T: the full-stream LP-RS on the empirical
                              type statistics of the whole stream (true durations, every
                              arrival, no fallback). A FLUID quantity on type means, an upper
                              reference for algorithms that decide by type; not a feasible
                              schedule. Equals TheoryModel's summary["bench_global"].      [$ per run]
    b.job_level_opt()         offline job-level OPT (ORACLE): max sum_j R_job,j x_j, x in {0,1},
                              s.t. at every arrival tick the occupancy of the chosen jobs is <= c
                              (HiGHS MILP); also returns the LP relaxation and the no-capacity
                              bound sum_j R_job,j^+.                                          [$ per run]
    discretization_milps(S, state, lam, c)
                              the discretization cost of the grid (week 33): MILP_true (each job
                              valued at its own hourly bid b_j) vs MILP_bin (b_j replaced by its
                              bin's lower edge L_k), same jobs, capacity, true durations.     [$ per run]
    b.job_opt_bucketed(source, duration)      (week 30, used only by notebook 03)
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


def bin_values(v, grid) -> np.ndarray:
    """L_k of every hourly bid: the lower edge of its bin (0 for bin 1)."""
    return grid.lower_edges[grid.assign(v) - 1]


def discretization_milps(S, state, lam, c, time_limit: float = 3600.0) -> dict:
    """
    The discretization cost of the grid (week 33). The grid changes only the VALUE (L_k instead of the bid), so two
    job-level MILPs on the same jobs, the same capacity checks (at every arrival, true durations), the same carbon
    map and s_C, differ only in that:
        MILP_true   R_j = b_j d_j   + (phi_j(b_j) d_j   - C_elec,j) - s_C K_j
        MILP_bin    R_j = L_k d_j   + (phi_j(L_k) d_j   - C_elec,j) - s_C K_j      L_k = lower edge of b_j's bin
    with phi_j(x) = c_j phi_m(x / c_j) (the job's own cost_base c_j), d_j the true duration. S: any stream-like object
    (tick, D, A, v, cost_base, duration_hours, C_elec, C_carbon); state: its grid, bid model and s_C.
    Candidates = the jobs with R_true > 0, for both. A candidate with R_bin <= 0 is left out of MILP_bin: a job with
    a negative reward is never in an optimal schedule, so OPT_bin does not change. L_k <= b_j and phi is increasing,
    so R_bin <= R_true for every job and OPT_bin <= OPT_true.
        value loss     OPT_true - OPT_bin (best - best; the range uses the MILPs' bounds)
        decision loss  OPT_true - (MILP_bin's schedule valued at the true rewards)
        worst case     1 - 1/rho of the grid
    Each MILP (and its LP relaxation) runs on its own HiGHS instance with its own time limit.
    """
    terms = lambda b: reward.job_terms(b, S.cost_base, S.duration_hours, S.C_elec, S.C_carbon, state.myerson,
                                       state.carbon_price, lam).R
    R_true = terms(np.asarray(S.v, float))
    R_bin = terms(bin_values(S.v, state.grid))
    c = np.asarray(c, float)
    n = len(R_true)

    def solve(R, cand):
        x = np.zeros(n, bool)
        if cand.size == 0:
            return x, dict(best=0.0, bound=0.0, lp=0.0, gap=0.0, status="no candidates", runtime_s=0.0,
                           n_candidates=0, nonzeros=0)
        o = occupancy_milp(S.tick[cand], S.D[cand], S.A[cand], R[cand], c, time_limit)
        x[cand] = o["x"]
        return x, dict(best=float(R[x].sum()), bound=max(o["dual_bound"], float(R[x].sum())), lp=o["lp_obj"],
                       gap=o["gap"], status=o["status"], runtime_s=o["runtime_s"], n_candidates=int(cand.size),
                       nonzeros=o["nonzeros"])
    cand = np.flatnonzero(R_true > 0)
    x_true, true = solve(R_true, cand)
    x_bin, binm = solve(R_bin, cand[R_bin[cand] > 0])
    peak = lambda x: peak_occupancy(S.tick[x], S.D[x], S.A[x])
    for o, x in ((true, x_true), (binm, x_bin)):
        o["feasible_true"] = bool(np.all(peak(x) <= c * (1 + 1e-9)))
    at_true = float(R_true[x_bin].sum())
    pct = lambda d: 100 * d / true["best"] if true["best"] > 0 else float("nan")
    rho = state.grid.ratio
    return dict(n_jobs=int(n), n_candidates=int(cand.size), true=true, bin=binm,
                value_loss=true["best"] - binm["best"], value_loss_pct=pct(true["best"] - binm["best"]),
                value_loss_range=(true["best"] - binm["bound"], true["bound"] - binm["best"]),
                bin_schedule_at_true=at_true, decision_loss=true["best"] - at_true,
                decision_loss_pct=pct(true["best"] - at_true), decision_loss_max=true["bound"] - at_true,
                rho=float(rho), worst_case_pct=100 * (1 - 1 / rho), n_jobs_reward_bin_gt_true=int((R_bin > R_true + 1e-12).sum()),
                x_true=x_true, x_bin=x_bin, R_true=R_true, R_bin=R_bin, time_limit=float(time_limit))


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
        """Full-stream hindsight state (ORACLE): every arrival, true D, Phase 0's grid, sigma refit on the whole
        stream. tau* and lambda_hat_ref come from it (types without arrivals have p = 0)."""
        cfg_h = replace(self.cfg, phase_transition=replace(self.cfg.phase_transition, n_min=2))
        idx = np.arange(self.S.n)
        st, _ = build_window_state(self.S, idx, idx, self.T, self.state0, self.ds, cfg_h)
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
        k = st.grid.assign(S.v)
        j = assign_type_ids(st.type_space, np.asarray(k), S.priority, S.hw_idx)
        jj = np.maximum(j, 0)
        seen = (j >= 0) & (np.asarray(st.p)[jj] > 0)
        return dict(type=j, seen=seen, r=np.where(seen, st.r_tilde[jj], 0.0),
                    d_bar=np.where(seen, st.d_bar[jj], 0.0))

    def job_opt_bucketed(self, source: str = "phase0", duration: str = "true",
                         time_limit: float = 300.0, tail_step: int | None = None) -> dict:
        """
        Week 30, used only by notebook 03 (the discretization cost is discretization_milps).
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
        no-capacity bound, feasible ceiling, job-level MILP (MILP_true), MILP_bin's schedule at the true rewards,
        fluid OPT, fluid OPT on feasible types, Greedy-Pos, G1, static Phase 0, the no-learning model, DLENT-RT.
        Greedy-Pos, static Phase 0 and the no-learning model use DLENT-RT's mechanism on the Phase 0 state
        (models.policy.ThresholdPolicy: threshold k* per column, hourly reserve, payment max(L_k*, r) x hours);
        Greedy-Pos with tau = 0, the no-learning model with the theoretical thinning p_U(W0, t) and the run's
        common random numbers.
        Checks: DLENT-RT's score = its R_total; both MILP schedules are feasible; OPT_bin and MILP_bin's schedule at
        the true rewards are at most OPT_true's bound. The discretization cost (discretization_milps, without the
        masks) and G1's decision-error rate (sign R_job != sign r~) are returned.
        """
        from .models.policy import ThresholdPolicy, TypeGreedyPolicy, mechanism_summary, run_policy

        S, R = self.S, self.R
        adm = np.zeros(S.n, bool)
        adm[[a["i"] for a in result.arrivals if a["admit"]]] = True
        assert np.isclose(self.score(adm), result.summary["R_total"], rtol=1e-10)
        disc = discretization_milps(S, self.state0, self.cfg.objective, self.c, time_limit=time_limit)
        assert np.allclose(disc["R_true"], R)
        assert disc["true"]["feasible_true"] and disc["bin"]["feasible_true"]
        for v in (disc["bin"]["best"], disc["bin_schedule_at_true"]):
            assert v <= disc["true"]["bound"] + 1e-6, (v, disc["true"])
        g1 = TypeGreedyPolicy.from_state(self.state0, S, R, self.ds.H_span_steps)
        mech_pols = {"Greedy-Pos": ThresholdPolicy(self.state0, S, tau=0.0),
                     "static Phase 0": ThresholdPolicy(self.state0, S),
                     "no-learning model": ThresholdPolicy(self.state0, S,
                                                          thin=(model.ep, model.u_rand, int(self.ds.d_max_steps)))}
        sims = {k: run_policy(S, self.c, p) for k, p in mech_pols.items()}
        sims = {"Greedy-Pos": sims["Greedy-Pos"], "G1 type greedy": run_policy(S, self.c, g1),
                "static Phase 0": sims["static Phase 0"], "no-learning model": sims["no-learning model"]}
        mech = {k: {a: b for a, b in mechanism_summary(p, sims[k], S).items() if a != "admit"}
                for k, p in mech_pols.items()}
        ff = self.fluid_opt_feasible_types()
        rows = {"job-level, no capacity": float(R[R > 0].sum()), "feasible ceiling": self.feasible_ceiling(),
                "job-level MILP": disc["true"]["best"],
                "job-level MILP on bin values, at true rewards": disc["bin_schedule_at_true"],
                "fluid OPT": self.fluid_opt(), "fluid OPT on feasible types": ff["value"],
                **{k: self.score(v.admit) for k, v in sims.items()}, "DLENT-RT": self.score(adm)}
        ok = np.isfinite(g1.r_dec)
        return dict(rows=rows, admitted={k: int(v.admit.sum()) for k, v in sims.items()} | {"DLENT-RT": int(adm.sum())},
                    discretization={k: v for k, v in disc.items() if k not in ("x_true", "x_bin", "R_true", "R_bin")},
                    fluid_feasible=ff, mechanism=mech,
                    g1_decision_error=float(np.mean(np.sign(R[ok]) != np.sign(g1.r_dec[ok]))) if ok.any() else 0.0)


# --------------------------------------------------------------------------- #
# Job-level enlargement (D3): a check, and a config option that is off by default
# --------------------------------------------------------------------------- #
def job_level_space_check(B: Benchmarks) -> dict:
    """
    Share of the positive-reward stream jobs (R_job > 0) that a job-level model rejects for lack of space. The
    job-level model is Greedy-Pos (G2: admit every R_job > 0 job that fits); it is run only as a capacity probe.
    """
    from .models.policy import JobGreedyPolicy, run_policy
    R = B.R
    pos = R > 0
    sim = run_policy(B.S, B.c, JobGreedyPolicy(R))
    n_pos = int(pos.sum())
    share = float(sim.phys_reject[pos].sum() / max(n_pos, 1))
    return dict(n_positive=n_pos, n_rejected_space=int(sim.phys_reject[pos].sum()), share_rejected_space=share,
                threshold=float(B.cfg.capacity.job_level_reject_max),
                exceeds=bool(share > B.cfg.capacity.job_level_reject_max))


def enlarge_for_job_models(ds, state0, cfg, max_steps: int = 30):
    """
    capacity.job_level_enlarge: multiply C by capacity.job_level_enlarge_step until job_level_space_check passes
    (the quota and the jobs stay those of f), re-running Phase 0 on the larger cluster. Off (the default): only the
    check. Returns (ds, state0, info), info = the checks in order and the final enlargement factor.
    """
    from dataclasses import replace as dc_replace
    from .phase0 import run_phase0
    checks = [job_level_space_check(Benchmarks(ds, state0, cfg))]
    factor = 1.0
    while cfg.capacity.job_level_enlarge and checks[-1]["exceeds"] and len(checks) <= max_steps:
        factor *= cfg.capacity.job_level_enlarge_step
        c = np.asarray(ds.capacity, float) * cfg.capacity.job_level_enlarge_step
        diag = dict(ds.capacity_diag, enlargement=factor)
        ds = dc_replace(ds, capacity=c, xi=float(np.max(np.asarray(ds.a_max) / c)), capacity_diag=diag)
        state0 = run_phase0(ds, cfg)
        checks.append(job_level_space_check(Benchmarks(ds, state0, cfg)))
    return ds, state0, dict(enabled=bool(cfg.capacity.job_level_enlarge), checks=checks, factor=factor)

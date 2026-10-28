"""
Capacity x type-grid sweep (week 30), used by notebook 03 Section 7. One call of run_one(cfg)
does everything for one configuration, so the configurations can run in parallel processes:

  * flag OFF (same c): the jobs that capacity.drop_jobs_exceeding_capacity WOULD remove, their
    share of the stream's sum of positive R_job and of Phase 0's profitable load;
  * flag as in cfg: Phase 0 (LP usage, profitable load, tau_0, types), the DLENT-RT online run,
    StaticPolicy (Phase 0), G1 (TypeGreedyPolicy, as implemented), G2 (JobGreedyPolicy), the
    fluid OPT and the job-level OPT (MILP), all on the same stream, R_job and physical gate.

No maths here: every number comes from dlent_rt (data, phase0, TheoryModel, benchmarks, policy).
"""
from __future__ import annotations

import time
import warnings

import numpy as np

from dlent_rt.benchmarks import Benchmarks, regret
from dlent_rt.data import load_dataset
from dlent_rt.lp_oracle import solve_lp_rs
from dlent_rt.models.policy import (REJ_R, REJ_UNSEEN, JobGreedyPolicy, StaticPolicy,
                                    TypeGreedyPolicy, run_policy)
from dlent_rt.models.theory import TheoryModel
from dlent_rt.phase0 import assign_type_ids, run_phase0, type_arrays
from dlent_rt.reward import job_reward

REASONS = ("timeout", "test1", "test2", "test3", "horizon")


def phase0_load(ds, st, cfg):
    """LP capacity usage sum_j p_j v_ij x*_j / c_i and profitable load sum_{r~_j>0} p_j v_ij / c_i."""
    c = np.asarray(ds.capacity, float)
    lp = solve_lp_rs(p=st.p, r=st.r_tilde, v=st.v, c=c, presolve=cfg.lp.presolve)
    x = np.zeros(st.n_types)
    x[lp.support_idx] = lp.x
    return (st.p * x) @ st.v / c, (st.p * (st.r_tilde > 0)) @ st.v / c


def _h_types(ds, st):
    """Phase-0 type id of every H job (-1 = no type)."""
    k = st.grid.assign(st.bids.v_rate, st.myerson)
    return assign_type_ids(st.type_space, np.asarray(k), ds.H.priority, ds.H.hw_idx)


def would_drop(cfg):
    """With the flag OFF (same c): jobs with A_ij > c_i, their share of sum R_job^+ (stream) and
    of the Phase-0 profitable load (H jobs of types with r~ > 0, volume A_type * D / H span)."""
    from dataclasses import replace
    cfg0 = replace(cfg, capacity=replace(cfg.capacity, drop_jobs_exceeding_capacity=False))
    ds = load_dataset(cfg0)
    st = run_phase0(ds, cfg0)
    c = np.asarray(ds.capacity, float)
    big_H = np.any(ds.H.A > c, axis=1)
    big_S = np.any(ds.stream.A > c, axis=1)
    from dlent_rt.models.core.stream import build_stream
    S = build_stream(ds, cfg0)
    R = job_reward(S, st, cfg0.objective).R
    Rp = np.maximum(R, 0.0)
    tid = _h_types(ds, st)
    _, _, A_type = type_arrays(st.type_space, st.grid)
    prof = (tid >= 0) & (st.r_tilde[np.maximum(tid, 0)] > 0)
    vol = A_type[np.maximum(tid, 0)] * ds.H.duration_steps[:, None] / ds.H_span_steps   # per job
    load_all = vol[prof].sum(0)
    load_big = vol[prof & big_H].sum(0)
    return dict(drop_n_H=int(big_H.sum()), drop_n_stream=int(big_S.sum()),
                drop_share_Rpos_pct=100 * Rp[big_S].sum() / Rp.sum(),
                drop_share_prof_load_cpu_pct=100 * load_big[0] / load_all[0],
                drop_share_prof_load_ram_pct=100 * load_big[1] / load_all[1])


def run_one(cfg, label: str = "", milp_time_limit: float = 600.0) -> dict:
    warnings.simplefilter("ignore")
    t0 = time.time()
    out = dict(label=label, fraction=cfg.capacity.fraction, q=cfg.capacity.concurrency_quantile,
               k_star=cfg.grid.k_star, n_profiles=cfg.hardware.n_profiles)
    out.update(would_drop(cfg))

    ds = load_dataset(cfg)
    st = run_phase0(ds, cfg)
    c = np.asarray(ds.capacity, float)
    em = cfg.energy_model
    lp_use, prof = phase0_load(ds, st, cfg)
    tid_H = _h_types(ds, st)
    n_pop = int((st.p > 0).sum())
    out.update(c_cpu=c[0], c_ram=c[1], c_cores=c[0] * em.machine_cores, c_GB=c[1] * em.machine_ram_gb,
               dropped_H=(ds.capacity_drop or {}).get("n_H", 0),
               dropped_stream=(ds.capacity_drop or {}).get("n_stream", 0),
               n_H=ds.H.n, n_stream=ds.stream.n, s_C=st.carbon_price,
               p0_lp_use_cpu_pct=100 * lp_use[0], p0_lp_use_ram_pct=100 * lp_use[1],
               p0_prof_load_cpu_pct=100 * prof[0], p0_prof_load_ram_pct=100 * prof[1],
               tau0_cpu=float(st.tau[0]), tau0_ram=float(st.tau[1]),
               n_types=int(st.n_types), n_types_populated=n_pop,
               jobs_per_populated_type=int((tid_H >= 0).sum()) / max(n_pop, 1))

    model = TheoryModel(ds, st, cfg)
    res = model.run()
    S = model.stream
    B = Benchmarks.from_model(model)
    R = B.R
    ph = res.phases
    tau_ph = np.array([[p["tau_cpu"], p["tau_ram"]] for p in ph])
    adm = np.zeros(S.n, bool)
    arr = res.arrivals
    adm[[a["i"] for a in arr if a["admit"]]] = True
    R_dlent = B.score(adm)
    assert np.isclose(R_dlent, res.summary["R_total"], rtol=1e-10)
    out.update(t_online_s=time.time() - t0, n_phases=len(ph),
               **{f"end_{r}": sum(p["reason"] == r for p in ph) for r in REASONS},
               phases_tau_pos=int((tau_ph > 0).any(1).sum()),
               tau_per_phase="; ".join(f"({a:.3g},{b:.3g})" for a, b in tau_ph),
               dlent_phys_rejects=int(sum(p["n_phys_reject"] for p in ph)),
               dlent_thinning_rejects=int(sum(p["n_thinned"] for p in ph)),
               dlent_admitted=int(adm.sum()))

    # policies on the same stream, R_job and gate
    g1 = TypeGreedyPolicy.from_state(st, S, R, ds.H_span_steps)
    sims = {"static": run_policy(S, c, StaticPolicy(st, S)), "G1": run_policy(S, c, g1),
            "G2": run_policy(S, c, JobGreedyPolicy(R))}
    fluid = B.fluid_opt()
    t1 = time.time()
    opt = B.job_level_opt(time_limit=milp_time_limit)
    out.update(t_milp_s=time.time() - t1, milp_status=opt["status"], milp_gap=opt["gap"],
               R_fluid=fluid, R_job_opt=opt["R_ip"], R_job_lp=opt["R_lp"], R_nocap=opt["R_nocap"],
               disc_loss=opt["R_ip"] - fluid, tau_star_cpu=float(B.reference_state.tau[0]),
               tau_star_ram=float(B.reference_state.tau[1]))
    totals = {"DLENT": R_dlent, **{k: B.score(v.admit) for k, v in sims.items()}}
    for k, Ry in totals.items():
        out[f"R_{k}"] = Ry
        out[f"pct_fluid_{k}"] = 100 * Ry / fluid
        out[f"regret_fluid_{k}"] = regret(Ry, fluid)
        out[f"regret_jobopt_{k}"] = regret(Ry, opt["R_ip"])
    for k, v in sims.items():
        out[f"admitted_{k}"] = int(v.admit.sum())
        out[f"phys_{k}"] = int(v.phys_reject.sum())
    out.update(G1_rej_r=int((g1.reason == REJ_R).sum()), G1_rej_unseen=int((g1.reason == REJ_UNSEEN).sum()))

    # decision errors: sign(R_job) != sign(r~ of the arrival's type)
    k = st.grid.assign(S.v, st.myerson)
    j = assign_type_ids(st.type_space, np.asarray(k), S.priority, S.hw_idx)
    ok = (j >= 0) & (st.p[np.maximum(j, 0)] > 0)
    out["dec_err_P0_pct"] = 100 * np.mean(np.sign(R[ok]) != np.sign(st.r_tilde[j[ok]]))
    ok1 = np.isfinite(g1.r_dec)
    out["dec_err_G1_pct"] = 100 * np.mean(np.sign(R[ok1]) != np.sign(g1.r_dec[ok1]))
    r_used = np.array([a["r_tilde"] for a in arr], float)
    i_arr = np.array([a["i"] for a in arr], np.int64)
    okd = np.isfinite(r_used)
    out["dec_err_DLENT_pct"] = 100 * np.mean(np.sign(R[i_arr[okd]]) != np.sign(r_used[okd]))
    out["t_total_s"] = time.time() - t0
    return out

"""
Small cluster WITHOUT the drop filter (week 30, report only; 30k trace).

    dlent_venv/bin/python scripts/small_cluster_nofilter.py fast     # Phase 0, tau_0 / tau*, rho, oversize, ceiling
    dlent_venv/bin/python scripts/small_cluster_nofilter.py full     # + job MILP (600 s), G2 / G1 / static, DLENT-RT

Base: the committed configs/default.yaml (git show HEAD into a temporary file), seed 42; xi_override 0.001 and
gamma_override 1.0 unchanged; outlier quantiles 0.90 / 0.95 unchanged; drop_jobs_exceeding_capacity OFF: jobs with
A > c stay in H and the stream and the physical gate rejects them.

Regimes: fraction_of_concurrency_quantile q = 0.7 with f in F, plus the sanity check = the earlier binding check's
rho = 0.6 run (binding_capacity_check.py --rho 0.6: q = 0.6, fraction 0.142948, no filter).
"Oversize" = A_ij > c_i on CPU or RAM; jobs' A is the snapped hardware profile, equal to their type's A, so the
spurious share of Phase 0's profitable load is sum_{H jobs in types with r~ > 0, oversize} A D / sum_{... all}.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "notebooks" / "src"))

OUT = ROOT / "reports" / "week30" / "small_cluster_nofilter"
F = (0.3, 0.2, 0.15, 0.1, 0.07, 0.05)
SANITY = ("sanity: q=0.6 f=0.142948 (binding check rho=0.6)", 0.6, 0.142948)
MAX_WORKERS = 4


def regimes():
    return [*[(f"f={f}", 0.7, f) for f in F], SANITY]


def cfg_for(q, f):
    from capacity_regime import committed
    base = committed()
    cap = replace(base.capacity, mode="fraction_of_concurrency_quantile", concurrency_quantile=q, fraction=f,
                  drop_jobs_exceeding_capacity=False)
    return replace(base, capacity=cap).validate()


def fast(label, q, f):
    warnings.simplefilter("ignore")
    from dlent_rt.benchmarks import Benchmarks
    from dlent_rt.data import load_dataset
    from dlent_rt.models.core.error_params import make_error_params
    from dlent_rt.models.core.stream import build_stream
    from dlent_rt.phase0 import assign_type_ids, run_phase0, type_arrays
    t0 = time.time()
    cfg = cfg_for(q, f)
    ds = load_dataset(cfg)
    st = run_phase0(ds, cfg)
    S = build_stream(ds, cfg)
    B = Benchmarks(ds, st, cfg, stream=S)
    c = np.asarray(ds.capacity, float)
    em = cfg.energy_model
    R = B.R
    big_S = np.any(S.A > c[None, :], axis=1)
    big_H = np.any(ds.H.A > c[None, :], axis=1)
    k = st.grid.assign(st.bids.v_rate, st.myerson)
    tid = assign_type_ids(st.type_space, np.asarray(k), ds.H.priority, ds.H.hw_idx)
    _, _, A_type = type_arrays(st.type_space, st.grid)
    jj = np.maximum(tid, 0)
    prof = (tid >= 0) & (st.r_tilde[jj] > 0)
    vol = A_type[jj] * ds.H.duration_steps[:, None] / ds.H_span_steps
    load = (st.p * (st.r_tilde > 0)) @ st.v
    assert np.allclose(vol[prof].sum(0), load, rtol=1e-9)          # same profitable load as the LP sees
    ref = B.reference_state
    ep = make_error_params(st, ds, cfg)
    pos = np.maximum(R, 0)
    a_max_H = ds.H.A.max(axis=0)
    o = {"regime": label, "q": q, "f": f,
         "capacity.absolute.cpu (norm)": c[0], "capacity.absolute.ram (norm)": c[1],
         "c CPU (cores)": c[0] * em.machine_cores, "c RAM (GB)": c[1] * em.machine_ram_gb,
         "a_max stream CPU / c": float(ds.a_max[0] / c[0]), "a_max stream RAM / c": float(ds.a_max[1] / c[1]),
         "a_max H CPU / c": float(a_max_H[0] / c[0]), "a_max H RAM / c": float(a_max_H[1] / c[1]),
         "oversize stream jobs": int(big_S.sum()), "oversize stream %": 100 * big_S.mean(),
         "oversize H jobs": int(big_H.sum()), "oversize H %": 100 * big_H.mean(),
         "spurious share of P0 profitable load CPU %": 100 * vol[prof & big_H, 0].sum() / vol[prof, 0].sum(),
         "spurious share of P0 profitable load RAM %": 100 * vol[prof & big_H, 1].sum() / vol[prof, 1].sum(),
         "s_C": st.carbon_price, "tau_0 (Phase 0)": str(np.round(st.tau, 6).tolist()),
         "tau* (hindsight)": str(np.round(ref.tau, 6).tolist()),
         "tau_0 > 0": bool(np.any(st.tau > 0)), "tau* > 0": bool(np.any(ref.tau > 0)),
         "rho CPU (P0 profitable load / c)": load[0] / c[0], "rho RAM (P0 profitable load / c)": load[1] / c[1],
         "rho CPU without oversize": vol[prof & ~big_H, 0].sum() / c[0],
         "rho RAM without oversize": vol[prof & ~big_H, 1].sum() / c[1],
         "R no capacity $": float(pos.sum()), "feasible ceiling $ (sum R+ over A <= c)": float(pos[~big_S].sum()),
         "R fluid OPT $": B.fluid_opt(), "xi data = max a_max/c": float(np.max(ds.a_max / c)),
         "p_U at phase-1 start (forced)": ep.p_U(int(ds.d_max_steps), 1),
         "populated types": int((st.p > 0).sum()),
         "H jobs per populated type": int((tid >= 0).sum()) / max(int((st.p > 0).sum()), 1),
         "fast s": time.time() - t0}
    return o, cfg, ds, st, S, B


def full(args):
    label, q, f = args
    warnings.simplefilter("ignore")
    from dlent_rt.models.policy import JobGreedyPolicy, StaticPolicy, TypeGreedyPolicy, run_policy
    import test_margins_check as tm
    o, cfg, ds, st, S, B = fast(label, q, f)
    R, c = B.R, np.asarray(ds.capacity, float)
    t0 = time.time()
    job = B.job_level_opt(time_limit=600.0)
    o.update({"R job MILP $": job["R_ip"], "job MILP gap": job["gap"], "job MILP status": job["status"],
              "job MILP runtime s": job["runtime_s"], "job MILP wall s": time.time() - t0})
    jm = job["R_ip"]
    o["capacity cost % ((no-cap - job MILP)/no-cap)"] = 100 * (o["R no capacity $"] - jm) / o["R no capacity $"]
    o["feasible ceiling % of no-cap"] = 100 * o["feasible ceiling $ (sum R+ over A <= c)"] / o["R no capacity $"]
    g2 = run_policy(S, c, JobGreedyPolicy(R))
    g1 = run_policy(S, c, TypeGreedyPolicy.from_state(st, S, R, ds.H_span_steps))
    sp = run_policy(S, c, StaticPolicy(st, S))
    for name, r in (("G2", g2), ("G1", g1), ("static P0", sp)):
        o[f"R {name} $"] = B.score(r.admit)
        o[f"{name} % of job MILP"] = 100 * o[f"R {name} $"] / jm
        o[f"{name} phys rejects % of arrivals"] = 100 * r.phys_reject.mean()
    o["fluid OPT % of job MILP"] = 100 * o["R fluid OPT $"] / jm
    o["job MILP - fluid OPT $"] = jm - o["R fluid OPT $"]
    t1 = time.time()
    df, info, thr, thr_no, res = tm.run(label, cfg)
    ph = res.phases
    tau = [(float(p["tau_cpu"]), float(p["tau_ram"])) for p in ph]
    reasons = res.summary["phase_end_reasons"]
    no_t1 = df[df["fired"] != "test1"]["T1 max ratio"]
    o.update({"DLENT-RT R $": res.summary["R_total"], "DLENT-RT % of job MILP": 100 * res.summary["R_total"] / jm,
              "DLENT-RT % of fluid OPT": 100 * res.summary["R_total"] / o["R fluid OPT $"],
              "DLENT-RT phases": len(ph), **{f"end {r}": int(reasons[r]) for r in ("timeout", "test1", "test2", "test3", "horizon")},
              "DLENT-RT phases with tau > 0": int(sum(a > 0 or b > 0 for a, b in tau)),
              "DLENT-RT tau per phase": "; ".join(f"({a:.3g},{b:.3g})" for a, b in tau),
              "Test 1 fires": bool(reasons["test1"] > 0),
              "Test 1 max ratio in phases where it does not fire": float(no_t1.max()) if len(no_t1) else np.nan,
              "DLENT-RT thinning rejects": int(sum(p["n_thinned"] for p in ph)),
              "DLENT-RT phys rejects": int(sum(p["n_phys_reject"] for p in ph)),
              "DLENT-RT phys rejects % of arrivals": 100 * sum(p["n_phys_reject"] for p in ph) / S.n,
              "DLENT-RT admitted": int(res.summary["n_admitted"]), "DLENT-RT s": time.time() - t1})
    df.to_csv(OUT / f"margins_{label.split(':')[0].replace('=', '').replace(' ', '')}.csv")
    return o


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "fast"
    pd.set_option("display.width", 250)
    t0 = time.time()
    if mode == "fast":
        rows = [fast(*r)[0] for r in regimes()]
        df = pd.DataFrame(rows).set_index("regime")
        df.to_csv(OUT / "fast.csv")
    else:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=mp.get_context("spawn")) as pool:
            rows = list(pool.map(full, regimes()))
        df = pd.DataFrame(rows).set_index("regime")
        df.to_csv(OUT / "full.csv")
    print(df.T.to_string())
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

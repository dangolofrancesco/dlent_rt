"""
Benchmark ladder with the bucketed job-level MILP (week 30, report only).

    dlent_venv/bin/python notebooks/src/bucketed_ladder.py

Configs (seed 42): the committed configs/default.yaml (from_amax, no filter) and the capacity-scan config
f = 0.2 (fraction_of_concurrency_quantile q = 0.7, drop_jobs_exceeding_capacity ON). Per config, on one
stream / R_job / physical gate: Sigma R_job+ (no capacity), job-level MILP, bucketed MILP
(Benchmarks.job_opt_bucketed: source phase0 / reference x duration true (A) / type (B)), fluid OPT, G2,
G1, StaticPolicy (Phase 0), DLENT-RT. Every MILP has a 600 s limit.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks" / "src"))

TAB = ROOT / "reports" / "week30" / "tables"
TL = 600.0


def ladder(name: str) -> dict:
    warnings.simplefilter("ignore")
    from capacity_scan import committed_default, scan_config
    from dlent_rt.benchmarks import Benchmarks
    from dlent_rt.data import load_dataset
    from dlent_rt.models.policy import JobGreedyPolicy, StaticPolicy, TypeGreedyPolicy, run_policy
    from dlent_rt.models.theory import TheoryModel
    from dlent_rt.phase0 import run_phase0

    cfg = committed_default() if name == "default" else scan_config(0.2, drop=True)
    ds = load_dataset(cfg)
    st = run_phase0(ds, cfg)
    model = TheoryModel(ds, st, cfg)
    res = model.run()
    B = Benchmarks.from_model(model)
    S, R, c = B.S, B.R, B.c
    adm = np.zeros(S.n, bool)
    adm[[a["i"] for a in res.arrivals if a["admit"]]] = True
    assert np.isclose(B.score(adm), res.summary["R_total"], rtol=1e-10)
    job = B.job_level_opt(time_limit=TL)
    buck = {(src, dur): B.job_opt_bucketed(src, dur, time_limit=TL)
            for src in ("phase0", "reference") for dur in ("true", "type")}
    pols = {"G2 job greedy": JobGreedyPolicy(R),
            "G1 type greedy": TypeGreedyPolicy.from_state(st, S, R, ds.H_span_steps),
            "StaticPolicy (Phase 0)": StaticPolicy(st, S)}
    sims = {k: run_policy(S, c, p) for k, p in pols.items()}

    def strip(o):
        return {k: v for k, v in o.items() if k != "x"}

    out = dict(config=name, n_stream=int(S.n), n_H=int(ds.H.n), c=c.tolist(),
               dropped=ds.capacity_drop, s_C=st.carbon_price, T=int(ds.T),
               R_nocap=job["R_nocap"], job=strip(job), fluid=B.fluid_opt(),
               bucketed={f"{s}|{d}": strip(o) for (s, d), o in buck.items()},
               policies={k: dict(R=B.score(v.admit), admitted=int(v.admit.sum()),
                                 phys=int(v.phys_reject.sum())) for k, v in sims.items()},
               dlent=dict(R=B.score(adm), admitted=int(adm.sum()),
                          reasons=res.summary["phase_end_reasons"]))
    # check: variant A is feasible for the job MILP, so its realised reward cannot exceed job OPT
    # (beyond the job MILP's own relative gap when it stops at the time limit)
    for (s, d), o in buck.items():
        bound = job["R_ip"] * (1 + max(job["gap"], 0)) + 1e-6
        o_chk = dict(realised_le_jobopt=bool(o["R_real"] <= job["R_ip"] + 1e-6),
                     realised_le_jobopt_bound=bool(o["R_real"] <= bound))
        out["bucketed"][f"{s}|{d}"].update(o_chk)
        if d == "true":
            assert o["feasible_true"], (name, s, d)
            assert o_chk["realised_le_jobopt_bound"], (name, s, d, o["R_real"], job["R_ip"], job["gap"])
    return out


def main():
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as pool:
        res = dict(zip(("default", "f0.2"), pool.map(ladder, ("default", "f0.2"))))
    (TAB / "bucketed_ladder.json").write_text(json.dumps(res, indent=1, default=str))
    for name, o in res.items():
        job = o["job"]["R_ip"]
        rows = [("job-level, no capacity (Σ R_job⁺)", o["R_nocap"], None),
                ("job-level MILP", job, o["job"]),
                *[(f"bucketed MILP {'A' if d == 'true' else 'B'} ({s})", o["bucketed"][f"{s}|{d}"]["R_real"],
                   o["bucketed"][f"{s}|{d}"]) for d in ("true", "type") for s in ("phase0", "reference")],
                ("fluid OPT", o["fluid"], None),
                *[(k, v["R"], None) for k, v in o["policies"].items()],
                ("DLENT-RT", o["dlent"]["R"], None)]
        print(f"=== {name}: |stream| = {o['n_stream']:,}  dropped = {o['dropped']}  s_C = {o['s_C']:.4f}")
        prev = None
        for lab, v, info in rows:
            extra = ""
            if info is not None and "status" in info:
                extra = f"  [{info['status']}, gap {info['gap']:.2e}, {info['runtime_s']:.1f} s"
                if "R_believed" in info:
                    extra += (f", believed {info['R_believed']:,.2f}, cand {info['n_candidates']}, "
                              f"unseen {info['n_unseen_jobs']} jobs / {info['n_unseen_types']} types, "
                              f"peak true occ/c {np.round(info['peak_true_occ_over_c'], 4).tolist()}")
                extra += "]"
            gap = "" if prev is None else f"{prev - v:>12,.2f}"
            print(f"  {lab:<40} {v:>14,.2f} {100 * v / job:>8.2f}% {gap:>12}{extra}")
            prev = v


if __name__ == "__main__":
    main()

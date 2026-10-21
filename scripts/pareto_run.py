"""
Pareto frontier over the objective weights lambda (week 30, offline only; nothing enters the online loop).

    dlent_venv/bin/python scripts/pareto_run.py            # writes reports/week30/pareto/*.csv (cache)

Instance (same as notebook 03): the COMMITTED configs/default.yaml (`git show HEAD:configs/default.yaml` into a
temporary file; an uncommitted working-copy edit is never read), seed 42, no drop filter.

Objectives per job (dlent_rt.pareto.job_components, from dlent_rt.reward): CV = q*v, PR = phi_job - C_elec,
CO = s_C*K, computed ONCE with Phase 0's bid model and frozen s_C; for every lambda the job reward is
R(lambda) = l1*CV + l2*PR - l3*CO (linear combination, no model re-implemented). Per-type components come from the
per-type means of the state (dlent_rt.pareto.type_components).

Models:
  * fluid LP OPT (weighted sum) and the STANDARD augmented Chebyshev LP on the fluid problem (reference state, as
    Benchmarks.fluid_opt), all grid points, plus the legacy (inverted-orientation) Chebyshev for reference;
    ideal / nadir = single-objective fluid optima / payoff-table nadir; dominance slack per fluid solution;
  * G2 job greedy (R(lambda) > 0, physical gate) and the job-level MILP (benchmarks.occupancy_milp on the jobs with
    R(lambda) > 0, 600 s limit), all grid points;
  * StaticPolicy (Phase 0) and DLENT-RT theory, the 7-point subset, Phase 0 rebuilt with each lambda through
    run_phase0 (checked: s_C, the valuation grid, the hardware profiles and the job components do not change);
  * "does every lambda matter?": the centre (1, 1, 1) and lambda_k x2 / x0.5 (pareto.perturbations) for the fluid
    LP, G2, the job MILP, StaticPolicy and DLENT-RT; jobs added / dropped vs the centre solution (perturbations.csv).
Totals are $ over the stream (fluid: per tick * T). One row per (model, lambda) in results.csv.
"""
from __future__ import annotations

import multiprocessing as mp
import subprocess
import sys
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dlent_rt import pareto
from dlent_rt.benchmarks import Benchmarks, occupancy_milp
from dlent_rt.config import load_config
from dlent_rt.data import load_dataset
from dlent_rt.models.core.stream import build_stream
from dlent_rt.models.policy import JobGreedyPolicy, StaticPolicy, run_policy
from dlent_rt.models.theory import TheoryModel
from dlent_rt.phase0 import run_phase0

OUT = ROOT / "reports" / "week30" / "pareto"
MILP_TL = 600.0
MAX_WORKERS = 4                                       # ~1 GB per DLENT-RT run: 13 at once swap a 16 GB machine
RHO = 1e-3                                            # augmentation of the standard Chebyshev LP


def committed_config():
    txt = subprocess.check_output(["git", "-C", str(ROOT), "show", "HEAD:configs/default.yaml"], text=True)
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "default_committed.yaml"
        p.write_text(txt)
        cfg = load_config(p)
    return cfg, commit


def with_lambda(cfg, lam):
    return replace(cfg, objective=replace(cfg.objective, lambda1=lam[0], lambda2=lam[1], lambda3=lam[2])).validate()


def subset_worker(lam):
    """Phase 0 rebuilt with lambda, StaticPolicy and DLENT-RT. Returns admitted masks and the invariance checks."""
    warnings.simplefilter("ignore")
    cfg, _ = committed_config()
    cfg_l = with_lambda(cfg, lam)
    ds = load_dataset(cfg_l)
    t0 = time.time()
    st = run_phase0(ds, cfg_l)
    S = build_stream(ds, cfg_l)
    comp = pareto.job_components(S, st)
    static = run_policy(S, ds.capacity, StaticPolicy(st, S))
    t_static = time.time() - t0
    t1 = time.time()
    model = TheoryModel(ds, st, cfg_l)
    res = model.run()
    adm = np.zeros(S.n, bool)
    adm[[a["i"] for a in res.arrivals if a["admit"]]] = True
    return dict(lam=lam, s_C=st.carbon_price, grid_edges=st.grid.edges, hw=ds.hw_catalogue.centroids,
                comp=comp, static=static.admit, static_phys=int(static.phys_reject.sum()),
                dlent=adm, dlent_R=res.summary["R_total"], dlent_reasons=res.summary["phase_end_reasons"],
                dlent_phys=int(res.summary["n_phys_reject"]), tau0=st.tau.tolist(),
                t_static=t_static, t_dlent=time.time() - t1)


def row(model, lam, tot, n_adm, runtime, **extra):
    l1, l2, l3 = lam
    return {"model": model, "l1": l1, "l2": l2, "l3": l3, "sum_CV": tot["CV"], "sum_PR": tot["PR"],
            "sum_CO": tot["CO"], "scalarised": l1 * tot["CV"] + l2 * tot["PR"] - l3 * tot["CO"],
            "n_admitted": n_adm, "runtime_s": runtime, **extra}


def main():
    warnings.simplefilter("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    cfg, commit = committed_config()
    blob = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD:configs/default.yaml"], text=True).strip()
    grid, sub = pareto.weight_grid(), pareto.subset7()
    pert = pareto.perturbations()
    pert_lams = [lam for *_, lam in pert]
    fast_lams = grid + [l for l in pert_lams if l not in grid]          # fluid / G2 / MILP
    slow_lams = sub + [l for l in pert_lams if l not in sub]            # StaticPolicy / DLENT-RT
    pool = ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=mp.get_context("spawn"))
    t_slow = time.time()
    futs = [pool.submit(subset_worker, lam) for lam in slow_lams]

    ds = load_dataset(cfg)
    st = run_phase0(ds, cfg)                               # lambda = (1, 1, 1)
    S = build_stream(ds, cfg)
    B = Benchmarks(ds, st, cfg, stream=S)
    J = pareto.job_components(S, st)
    assert np.max(np.abs(J.combine(pareto.CENTRE) - B.R)) < 0.005
    ref = B.reference_state
    TC = pareto.type_components(ref)
    assert np.max(np.abs(TC.combine(pareto.CENTRE) - ref.r_tilde)) < 0.005
    T, c = int(ds.T), np.asarray(ds.capacity, float)

    def job_totals(mask):
        return {"CV": float(J.CV[mask].sum()), "PR": float(J.PR[mask].sum()), "CO": float(J.CO[mask].sum())}

    rows, sets = [], {}                                    # sets[(model, lam)] = admitted mask or fluid x
    ideal, nadir = pareto.ideal_nadir_fluid(ref.p, TC, ref.v, c)
    for lam in fast_lams:
        on_grid = lam in grid
        t0 = time.time()
        x, tot, lp = pareto.fluid_lp(ref.p, TC, ref.v, c, lam)
        rows.append(row("fluid LP OPT", lam, {k: v * T for k, v in tot.items()}, float((ref.p * x).sum() * T),
                        time.time() - t0, status=lp.status,
                        dominance_slack=pareto.fluid_dominance_slack(ref.p, TC, ref.v, c, x, ideal, nadir)))
        sets[("fluid LP OPT", lam)] = x
        if on_grid:
            w = np.asarray(lam) / 3.0
            for name, fn in (("Chebyshev fluid LP", lambda: pareto.chebyshev_fluid(ref.p, TC, ref.v, c, w, ideal,
                                                                                  nadir, rho=RHO)),
                             ("Chebyshev fluid LP (legacy, inverted)",
                              lambda: pareto.chebyshev_fluid_legacy(ref.p, TC, ref.v, c, w, ideal, nadir))):
                t0 = time.time()
                xc, totc, tval, msg = fn()
                rows.append(row(name, lam, {k: v * T for k, v in totc.items()}, float((ref.p * xc).sum() * T),
                                time.time() - t0, status=msg, chebyshev_t=tval,
                                dominance_slack=pareto.fluid_dominance_slack(ref.p, TC, ref.v, c, xc, ideal, nadir)))
        R = J.combine(lam)
        t0 = time.time()
        g2 = run_policy(S, c, JobGreedyPolicy(R))
        rows.append(row("G2 job greedy", lam, job_totals(g2.admit), int(g2.admit.sum()), time.time() - t0,
                        phys_rejects=int(g2.phys_reject.sum())))
        sets[("G2 job greedy", lam)] = g2.admit
        cand = np.flatnonzero(R > 0)
        if cand.size:
            o = occupancy_milp(S.tick[cand], S.D[cand], S.A[cand], R[cand], c, MILP_TL)
            xm = np.zeros(S.n, bool)
            xm[cand] = o["x"]
            st_, gap, rt = o["status"], o["gap"], o["runtime_s"]
        else:
            xm, st_, gap, rt = np.zeros(S.n, bool), "no job with R > 0", 0.0, 0.0
        rows.append(row("job-level MILP", lam, job_totals(xm), int(xm.sum()), rt, status=st_, gap=gap,
                        candidates=int(cand.size)))
        sets[("job-level MILP", lam)] = xm
        print(f"lambda {lam}: fluid / Chebyshev / G2 / MILP done ({time.time() - t_all:.0f} s)", flush=True)

    checks, slow_times = [], {}
    for f in futs:
        o = f.result()
        lam = o["lam"]
        checks.append({"l1": lam[0], "l2": lam[1], "l3": lam[2], "s_C": o["s_C"],
                       "s_C same as (1,1,1)": o["s_C"] == st.carbon_price,
                       "grid edges same": bool(np.array_equal(o["grid_edges"], st.grid.edges)),
                       "hardware profiles same": bool(np.array_equal(o["hw"], ds.hw_catalogue.centroids)),
                       "job components same": all(np.array_equal(getattr(o["comp"], k), getattr(J, k))
                                                  for k in ("CV", "PR", "CO")),
                       "tau_0": str(np.round(o["tau0"], 6)), "DLENT end reasons": str(o["dlent_reasons"]),
                       "DLENT runtime s": o["t_dlent"]})
        assert np.isclose(J.combine(lam)[o["dlent"]].sum(), o["dlent_R"], rtol=1e-9, atol=1e-6)
        rows.append(row("StaticPolicy (Phase 0)", lam, job_totals(o["static"]), int(o["static"].sum()),
                        o["t_static"], phys_rejects=o["static_phys"]))
        rows.append(row("DLENT-RT theory", lam, job_totals(o["dlent"]), int(o["dlent"].sum()), o["t_dlent"],
                        phys_rejects=o["dlent_phys"]))
        sets[("StaticPolicy (Phase 0)", lam)] = o["static"]
        sets[("DLENT-RT theory", lam)] = o["dlent"]
        slow_times[lam] = o["t_dlent"]
    pool.shutdown()
    t_slow = time.time() - t_slow

    df = pd.DataFrame(rows)
    df["on_grid"] = [(a, b, c_) in grid for a, b, c_ in zip(df.l1, df.l2, df.l3)]
    df["in_subset7"] = [any(np.allclose((a, b, c_), s_) for s_ in sub) for a, b, c_ in zip(df.l1, df.l2, df.l3)]
    df["in_perturbations"] = [(a, b, c_) in pert_lams for a, b, c_ in zip(df.l1, df.l2, df.l3)]
    df["nondominated"] = False
    for m, g in df[df.on_grid | df.in_subset7].groupby("model"):
        F = np.column_stack([g.sum_CV, g.sum_PR, -g.sum_CO])
        df.loc[g.index, "nondominated"] = pareto.nondominated(F)
    # hypervolume on the grid / subset points: common reference point = componentwise worst over those rows,
    # pushed out by 1% of the range
    G = df[df.on_grid | df.in_subset7]
    Fall = np.column_stack([G.sum_CV, G.sum_PR, -G.sum_CO])
    lo, hi = Fall.min(axis=0), Fall.max(axis=0)
    ref_pt = lo - 0.01 * (hi - lo)
    hv = []
    for m, g in G.groupby("model"):
        for scope, gg in (("all its grid points", g[g.on_grid]), ("7-point subset", g[g.in_subset7])):
            if len(gg) == 0:
                continue
            F = np.column_stack([gg.sum_CV, gg.sum_PR, -gg.sum_CO])
            hv.append({"model": m, "points": scope, "n_points": len(gg),
                       "hypervolume": pareto.hypervolume_3d(F, ref_pt)})
    hv = pd.DataFrame(hv)

    # perturbations: jobs added / dropped and component changes vs the centre solution of the same model
    prow = []
    for model in ("fluid LP OPT", "G2 job greedy", "job-level MILP", "StaticPolicy (Phase 0)", "DLENT-RT theory"):
        base = sets[(model, pareto.CENTRE)]
        r0 = df[(df.model == model) & (df.l1 == 1) & (df.l2 == 1) & (df.l3 == 1)].iloc[0]
        for label, k, fct, lam in pert[1:]:
            x = sets[(model, lam)]
            r = df[(df.model == model) & (df.l1 == lam[0]) & (df.l2 == lam[1]) & (df.l3 == lam[2])].iloc[0]
            if model == "fluid LP OPT":                     # expected jobs: T * sum_j p_j (x_j - x0_j)^+
                added = float(T * (ref.p * np.maximum(x - base, 0)).sum())
                dropped = float(T * (ref.p * np.maximum(base - x, 0)).sum())
            else:
                added, dropped = int((x & ~base).sum()), int((base & ~x).sum())
            prow.append({"model": model, "perturbation": label, "lambda_k": f"lambda{k + 1}", "factor": fct,
                         "l1": lam[0], "l2": lam[1], "l3": lam[2], "n_admitted": r.n_admitted,
                         "n_admitted centre": r0.n_admitted, "jobs added": added, "jobs dropped": dropped,
                         "jobs changed": added + dropped, "d sum_CV": r.sum_CV - r0.sum_CV,
                         "d sum_PR": r.sum_PR - r0.sum_PR, "d sum_CO": r.sum_CO - r0.sum_CO})
    pdf = pd.DataFrame(prow)

    meta = pd.Series({"commit": commit, "configs/default.yaml blob": blob,
                      "config": "git show HEAD:configs/default.yaml -> temporary <tmpdir>/default_committed.yaml "
                                "(deleted after loading); the working-copy configs/default.yaml is not read",
                      "seed": cfg.run.seed, "T": T, "n_stream": S.n, "s_C": st.carbon_price, "c_cpu": c[0],
                      "c_ram": c[1], "chebyshev rho": RHO,
                      "fluid ideal (CV, PR, -CO) per tick": str(ideal.tolist()),
                      "fluid nadir (CV, PR, -CO) per tick": str(nadir.tolist()),
                      "hypervolume reference point (CV, PR, -CO) $": str(ref_pt.tolist()),
                      "StaticPolicy/DLENT-RT runs (parallel) wall s": t_slow,
                      "DLENT-RT perturbation runs, sum of run times s":
                          sum(slow_times[l] for l in pert_lams if l != pareto.CENTRE),
                      "total runtime s": time.time() - t_all})
    df.to_csv(OUT / "results.csv", index=False)
    hv.to_csv(OUT / "hypervolume.csv", index=False)
    pdf.to_csv(OUT / "perturbations.csv", index=False)
    pd.DataFrame(checks).to_csv(OUT / "lambda_invariance_checks.csv", index=False)
    meta.to_csv(OUT / "meta.csv", header=["value"])
    print(df.to_string(float_format=lambda x: f"{x:,.2f}"))
    print(hv.to_string())
    print(pdf.to_string(float_format=lambda x: f"{x:,.2f}"))
    print(pd.DataFrame(checks).to_string())
    print(meta.to_string())


if __name__ == "__main__":
    main()

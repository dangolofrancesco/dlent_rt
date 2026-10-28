"""
Ablation runner (week 30 rounds; week 32: on the committed default config and data/batch_v2_30k.csv).

    caffeinate -ims dlent_venv/bin/python scripts/ablation_runner.py run        # all ablations x seeds (resumes)
    dlent_venv/bin/python scripts/ablation_runner.py summary                    # re-aggregate the cached runs

Config: the committed configs/default.yaml (scripts/regime_config.py, from git HEAD), then the ablation's overrides.
The week-30 results (small-cluster regime, old data) are in archive/reports/week30/ablations/. Seeds: ONLY the thinning seed varies
(online.thinning_seed_offset in SEEDS); Phase 0, H, the stream and the bids are fixed (the thinning draws u_rand are
the model's only online randomness). Every run is cached in reports/ablations/runs/<ablation>__s<offset>.json
(a rerun skips the cached ones); runs.csv has one row per (ablation, seed), summary.csv the mean / std over seeds and
the paired difference vs the baseline with the same seed (mean, 95% t-interval).
"""
from __future__ import annotations

import json
import multiprocessing as mp
import subprocess
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

REGIME = None                                 # None = the committed default (regime_config.committed_with_regime)
OUT = ROOT / "reports" / "ablations"
SEEDS = (2, 3, 4, 5, 6)
MAX_WORKERS = 4                               # ~1 GB per DLENT-RT run
T_975_4 = 2.7764451051977987                  # Student t quantile 0.975, 4 degrees of freedom (5 seeds)

# name -> (family, level label, dotted overrides)
ABLATIONS = {
    "baseline":          ("baseline", "theory", {}),
    "A1a_thinning_off":  ("A1 thinning", "p_U = 1", {"ablation.thinning_off": True}),
    "A1b_delta_1_T":     ("A1 thinning", "delta = 1/T in p_U", {"ablation.thinning_delta_power": 1}),
    "A1c_epsD_0":        ("A1 thinning", "eps_D = 0 in p_U", {"ablation.thinning_eps_D_zero": True}),
    "A2a_xi_0.01":       ("A2a xi", "xi = 0.01", {"constants.xi_override": 0.01}),
    "A2a_xi_0.05":       ("A2a xi", "xi = 0.05", {"constants.xi_override": 0.05}),
    "A2b_gamma_0.3":     ("A2b gamma", "gamma = 0.3", {"constants.gamma_override": 0.3}),
    "A2b_gamma_0.1":     ("A2b gamma", "gamma = 0.1", {"constants.gamma_override": 0.1}),
    "A3_fixed_grid":     ("A3 grid", "Phase 0 grid kept", {"ablation.fixed_grid": True}),
}
# cumulative round (week 30): the stacked configurations, run under their own names (fresh cache entries)
CUMULATIVE = {
    "C0":  ("cumulative", "theory (all flags off)", {}),
    "C1":  ("cumulative", "A1c eps_D = 0 in p_U", {"ablation.thinning_eps_D_zero": True}),
    "C2":  ("cumulative", "C1 + A3 fixed grid", {"ablation.thinning_eps_D_zero": True, "ablation.fixed_grid": True}),
    "C3":  ("cumulative", "C2 + gamma = 0.3", {"ablation.thinning_eps_D_zero": True, "ablation.fixed_grid": True,
                                                "constants.gamma_override": 0.3}),
    "R1":  ("reference", "A1a thinning off", {"ablation.thinning_off": True}),
    "R2":  ("reference", "A1a + A3", {"ablation.thinning_off": True, "ablation.fixed_grid": True}),
}
CUM_PREV = {"C1": "C0", "C2": "C1", "C3": "C2", "R1": "C0", "R2": "R1"}       # the previous step of each
CUM_SAME_AS_ROUND1 = {"C0": "baseline", "C1": "A1c_epsD_0", "R1": "A1a_thinning_off"}
DETERMINISTIC = {"R1", "R2"}            # thinning off: no online randomness -> one run, not 5 replications
ROUND1 = list(ABLATIONS)                # round 1 (single factors); the cumulative names are added below
ABLATIONS.update(CUMULATIVE)

VOIDS = {   # which guarantee of the theory model each ablation gives up (stated from the code, not the paper)
    "baseline": "-",
    "A1a_thinning_off": "the Bernoulli thinning p_U = (1 - xi)/(1 + eps_U) that the theory model applies before the "
                        "oracle; without it admissions are no longer thinned to leave the eps_D / eta safety margin",
    "A1b_delta_1_T": "the confidence level of eps_D used by the thinning: delta = 1/T instead of 1/T^3 (the tests keep 1/T^3)",
    "A1c_epsD_0": "the eps_D margin of the thinning (cap (1 - xi) instead of (1 - xi)/(1 + eps_D)); Test 2 keeps eps_D",
    "A2a_xi_0.01": "xi is forced (not data): a larger xi changes eps_D, the thinning cap, Test 1's margin xi*c and "
                   "Test 2's factor 1 - eps_D - xi",
    "A2a_xi_0.05": "as xi = 0.01",
    "A2b_gamma_0.3": "gamma is forced (not data): a smaller gamma shrinks eta (thinning at phase start) and every "
                     "test radius, so the tests' confidence statements no longer use the reward bound gamma",
    "A2b_gamma_0.1": "as gamma = 0.3",
    "A3_fixed_grid": "ReInit's re-estimation of the valuation grid (write-up Alg. 3 lines 2-6): types are binned "
                     "with Phase 0's edges in every phase (the per-type statistics are still re-estimated)",
    "C0": "-", "C1": "as A1c", "C2": "as A1c + A3", "C3": "as A1c + A3 + A2b (gamma = 0.3)",
    "R1": "as A1a", "R2": "as A1a + A3",
}


def provenance() -> dict:
    """The real git HEAD the code runs from, and whether the working tree is dirty (uncommitted changes)."""
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True)
    diff = subprocess.check_output(["git", "-C", str(ROOT), "diff", "HEAD"], text=True)
    import hashlib
    return {"git HEAD": head, "dirty": bool(status.strip()),
            "dirty files": "; ".join(l[3:] for l in status.splitlines() if l.strip()),
            "diff sha1": hashlib.sha1(diff.encode()).hexdigest() if diff else "-"}


def _replace_path(obj, path, value):
    if len(path) == 1:
        return replace(obj, **{path[0]: value})
    return replace(obj, **{path[0]: _replace_path(getattr(obj, path[0]), path[1:], value)})


def config_for(name: str, offset: int):
    from regime_config import committed_with_regime
    cfg, info = committed_with_regime(REGIME)
    for k, v in ABLATIONS[name][2].items():
        cfg = _replace_path(cfg, k.split("."), v)
    cfg = replace(cfg, online=replace(cfg.online, thinning_seed_offset=int(offset)))
    return cfg.validate(), info


def constants(cfg, ds, st):
    from dlent_rt.models.core.error_params import make_error_params
    ep = make_error_params(st, ds, cfg)
    d = int(ds.d_max_steps)
    c = np.asarray(ds.capacity, float)
    return {"xi": ep.xi, "gamma": ep.gamma, "eps_D (tests)": ep.eps_D, "eps_D (thinning)": ep.eps_D_thinning,
            "p_U cap": 1.0 if ep.thin_off else (1 - ep.xi) / (1 + ep.eps_D_thinning),
            "p_U at phase-1 start": ep.p_U(d, 1), "p_U at t - tau = d_max": ep.p_U(d, d),
            "eta(d_max)": float(ep.eta(d)), "Test 2 factor 1-eps_D-xi": 1 - ep.eps_D - ep.xi,
            "eps_Test2(d_max)": float(ep.eps_test2(d, d)), "eps_Test3(d_max)": float(ep.eps_test3(d)),
            "Test 1 margin xi*c CPU": ep.xi * c[0], "Test 1 margin xi*c RAM": ep.xi * c[1]}


def one_run(name: str, offset: int) -> dict:
    warnings.simplefilter("ignore")
    from dlent_rt.data import load_dataset
    from dlent_rt.models.theory import TheoryModel
    from dlent_rt.phase0 import run_phase0
    from dlent_rt.reward import job_reward
    path = OUT / "runs" / f"{name}__s{offset}.json"
    if path.is_file():
        return json.loads(path.read_text())
    t0 = time.time()
    cfg, info = config_for(name, offset)
    ds = load_dataset(cfg)
    st = run_phase0(ds, cfg)
    model = TheoryModel(ds, st, cfg)
    res = model.run()
    S = model.stream
    adm = np.zeros(S.n, bool)
    adm[[a["i"] for a in res.arrivals if a["admit"]]] = True
    J = job_reward(S, st, cfg.objective)
    lam = cfg.objective
    ph = res.phases
    reasons = res.summary["phase_end_reasons"]
    tau = [(float(p["tau_cpu"]), float(p["tau_ram"])) for p in ph]
    ep = model.ep
    phase_table = [{"phase": m, "end reason": p["reason"], "start tick": int(p["tau_start"]), "end tick": int(p["tau_end"]),
                    "duration (ticks)": int(p["W"]), "W_prev": int(p["W_prev"]), "arrivals": int(p["n_arrivals"]),
                    "admitted": int(p["n_admitted"]), "thinning rejects": int(p["n_thinned"]),
                    "physical rejects": int(p["n_phys_reject"]), "reward $": float(p["R"]),
                    "tau > 0": bool(p["tau_cpu"] > 0 or p["tau_ram"] > 0),
                    "tau (cpu, ram)": f"({p['tau_cpu']:.4g}, {p['tau_ram']:.4g})",
                    "p_U at start": float(ep.p_U(int(p["W_prev"]), 1))}
                   for m, p in enumerate(ph, start=1)]
    o = {"ablation": name, "family": ABLATIONS[name][0], "level": ABLATIONS[name][1], "seed offset": int(offset),
         "R $": float(res.summary["R_total"]),
         "customer value $": float(lam.lambda1 * J.cv[adm].sum()),
         "profit $": float(lam.lambda2 * J.profit[adm].sum()),
         "carbon $": float(-lam.lambda3 * J.carbon_usd[adm].sum()),
         "admitted": int(adm.sum()), "thinning rejects": int(sum(p["n_thinned"] for p in ph)),
         "physical rejects": int(sum(p["n_phys_reject"] for p in ph)),
         "phases": len(ph), **{f"end {r}": int(reasons[r]) for r in ("timeout", "test1", "test2", "test3", "horizon")},
         "phases with tau > 0": int(sum(a > 0 or b > 0 for a, b in tau)),
         "Test 1 fires": bool(reasons["test1"] > 0), "tau per phase": "; ".join(f"({a:.3g},{b:.3g})" for a, b in tau),
         "s_C": float(st.carbon_price), "tau_0": str(np.round(st.tau, 6).tolist()),
         "capacity": str(np.asarray(ds.capacity, float).tolist()), "commit": info["commit"],
         "default blob": info["default_blob"], "regime blob": info["regime_blob"], **constants(cfg, ds, st),
         **{f"run {k}": v for k, v in provenance().items()}, "phase table": phase_table,
         "runtime s": time.time() - t0}
    assert np.isclose(o["customer value $"] + o["profit $"] + o["carbon $"], o["R $"], rtol=1e-9)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(o, indent=1, default=float))
    return o


def references() -> dict:
    """Job MILP, feasible ceiling, static Phase 0 and static + thinning per seed, on the baseline config."""
    path = OUT / "references.json"
    if path.is_file():
        return json.loads(path.read_text())
    warnings.simplefilter("ignore")
    from dlent_rt.benchmarks import Benchmarks
    from dlent_rt.data import load_dataset
    from dlent_rt.models.policy import run_static
    from dlent_rt.models.theory import TheoryModel
    from dlent_rt.phase0 import run_phase0
    cfg, info = config_for("baseline", SEEDS[0])
    ds = load_dataset(cfg)
    st = run_phase0(ds, cfg)
    m = TheoryModel(ds, st, cfg)
    B = Benchmarks.from_model(m)
    job = B.job_level_opt(time_limit=600.0)
    fits = ~np.any(B.S.A > B.c[None, :], axis=1)
    ref = {"job MILP $": job["R_ip"], "job MILP gap": job["gap"], "job MILP status": job["status"],
           "feasible ceiling $": float(np.maximum(B.R, 0)[fits].sum()),
           "static Phase 0 $": B.score(run_static(B.S, B.c, st).admit), **info}
    p_cap = (1 - m.ep.xi) / (1 + m.ep.eps_D)
    for off in SEEDS:
        u = TheoryModel(ds, st, replace(cfg, online=replace(cfg.online, thinning_seed_offset=off))).u_rand
        ref[f"static + thinning $ (offset {off})"] = B.score(run_static(B.S, B.c, st, thin_p=p_cap, u_rand=u).admit)
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ref, indent=1, default=float))
    return ref


def summarise():
    ref = references()
    rows = [json.loads(p.read_text()) for p in sorted((OUT / "runs").glob("*.json"))]
    df = pd.DataFrame([r for r in rows if r["ablation"] in ROUND1]).drop(columns=["phase table"], errors="ignore")
    jm, ceil, sp = ref["job MILP $"], ref["feasible ceiling $"], ref["static Phase 0 $"]
    df["% of job MILP"] = 100 * df["R $"] / jm
    df["% of feasible ceiling"] = 100 * df["R $"] / ceil
    df["% of static Phase 0"] = 100 * df["R $"] / sp
    order = ROUND1
    df["order"] = df["ablation"].map(order.index)
    df = df.sort_values(["order", "seed offset"]).drop(columns="order")
    df.to_csv(OUT / "runs.csv", index=False)
    base = df[df.ablation == "baseline"].set_index("seed offset")["R $"]
    summ = []
    for name in order:
        g = df[df.ablation == name].set_index("seed offset")
        if g.empty:
            continue
        seeds = sorted(set(g.index) & set(base.index))
        d = (g.loc[seeds, "R $"] - base.loc[seeds]).to_numpy()
        n = len(d)
        half = T_975_4 * d.std(ddof=1) / np.sqrt(n) if n == 5 and name != "baseline" else np.nan
        num = ["R $", "customer value $", "profit $", "carbon $", "admitted", "thinning rejects", "physical rejects",
               "phases", "end timeout", "end test1", "end test2", "end test3", "end horizon", "phases with tau > 0"]
        r = {"ablation": name, "family": ABLATIONS[name][0], "level": ABLATIONS[name][1], "seeds": n,
             "R mean $": g["R $"].mean(), "R std $": g["R $"].std(ddof=1),
             "paired delta mean $": d.mean() if name != "baseline" else 0.0,
             "paired delta 95% lo $": d.mean() - half, "paired delta 95% hi $": d.mean() + half,
             "paired delta % of job MILP": 100 * d.mean() / jm if name != "baseline" else 0.0,
             "beyond seed noise (95% CI excludes 0)": bool(name != "baseline" and (d.mean() - half > 0 or d.mean() + half < 0)),
             "% of job MILP (mean)": 100 * g["R $"].mean() / jm, "% of feasible ceiling (mean)": 100 * g["R $"].mean() / ceil,
             "% of static Phase 0 (mean)": 100 * g["R $"].mean() / sp,
             "Test 1 fires (runs)": int(g["Test 1 fires"].sum()),
             **{f"{k} (mean)": g[k].mean() for k in num[1:]},
             **{k: g[k].iloc[0] for k in ("xi", "gamma", "eps_D (tests)", "eps_D (thinning)", "p_U cap",
                                         "p_U at phase-1 start", "p_U at t - tau = d_max", "eta(d_max)",
                                         "Test 2 factor 1-eps_D-xi", "eps_Test2(d_max)", "eps_Test3(d_max)",
                                         "Test 1 margin xi*c CPU", "Test 1 margin xi*c RAM")},
             "voids": VOIDS[name]}
        summ.append(r)
    sdf = pd.DataFrame(summ).set_index("ablation")
    sdf.to_csv(OUT / "summary.csv")
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    prov = provenance()
    pd.Series({**{k: v for k, v in ref.items()}, "runner commit (HEAD at summary)": commit,
               "summary git dirty": prov["dirty"], "summary diff sha1": prov["diff sha1"],
               "runs' git HEAD (distinct)": "; ".join(sorted(df["run git HEAD"].dropna().unique()))
               if "run git HEAD" in df else "not recorded (round-1 runs; HEAD eba0c63 with a dirty tree)",
               "runs' dirty flag (distinct)": "; ".join(map(str, sorted(df["run dirty"].dropna().unique())))
               if "run dirty" in df else "not recorded",
               "seeds": str(SEEDS), "regime": REGIME}).to_csv(OUT / "meta.csv", header=["value"])
    return df, sdf, ref


def _paired(a: pd.Series, b: pd.Series, deterministic: bool):
    """Paired difference a - b over the common seeds: mean, 95% t-interval, beyond-noise flag."""
    seeds = sorted(set(a.index) & set(b.index))
    d = (a.loc[seeds] - b.loc[seeds]).to_numpy()
    half = T_975_4 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) == 5 else np.nan
    return {"mean": d.mean(), "lo": d.mean() - half, "hi": d.mean() + half,
            "beyond noise": bool(np.isfinite(half) and (d.mean() - half > 0 or d.mean() + half < 0)), "n": len(d)}


def cumulative_summary():
    """Cumulative round: C0..C3 over 5 seeds, R1 / R2 deterministic (one run, broadcast to the seeds for the
    pairing only). Paired differences vs C0 and vs the previous step; phase tables; round-1 identity checks."""
    ref = references()
    jm, ceil, sp = ref["job MILP $"], ref["feasible ceiling $"], ref["static Phase 0 $"]
    runs = {}
    for name in CUMULATIVE:
        offs = [SEEDS[0]] if name in DETERMINISTIC else list(SEEDS)
        runs[name] = {o: json.loads((OUT / "runs" / f"{name}__s{o}.json").read_text()) for o in offs}
    seriesR = {n: pd.Series({o: r["R $"] for o, r in v.items()}) for n, v in runs.items()}
    # deterministic references: the same value for every seed (no thinning randomness), used only to pair with C0
    for n in DETERMINISTIC:
        seriesR[n] = pd.Series({o: seriesR[n].iloc[0] for o in SEEDS})
    rows = []
    num = ["customer value $", "profit $", "carbon $", "admitted", "thinning rejects", "physical rejects", "phases",
           "end timeout", "end test1", "end test2", "end test3", "end horizon", "phases with tau > 0"]
    for n, v in runs.items():
        g = pd.DataFrame(list(v.values()))
        det = n in DETERMINISTIC
        vs0 = _paired(seriesR[n], seriesR["C0"], det) if n != "C0" else None
        prev = CUM_PREV.get(n)
        vsp = _paired(seriesR[n], seriesR[prev], det) if prev else None
        if vsp and det and prev in DETERMINISTIC:      # both single deterministic runs: one exact difference, no CI
            vsp = {**vsp, "lo": np.nan, "hi": np.nan, "beyond noise": False}
        r = {"config": n, "description": CUMULATIVE[n][1], "runs": len(g),
             "seed note": ("deterministic: thinning off, no online randomness; 1 run, seed std = 0; its CI comes "
                           "only from the other configuration's seed noise, NOT a replication") if det else "5 thinning seeds",
             "R mean $": g["R $"].mean(), "R std $": 0.0 if det else g["R $"].std(ddof=1),
             "% of job MILP": 100 * g["R $"].mean() / jm, "% of feasible ceiling": 100 * g["R $"].mean() / ceil,
             "% of static Phase 0": 100 * g["R $"].mean() / sp,
             "Test 1 fires (runs)": f"{int(g['Test 1 fires'].sum())}/{len(g)}"}
        for tag, d in (("vs C0", vs0), (f"vs previous ({prev})", vsp)):
            key = "vs C0" if tag == "vs C0" else "vs previous step"
            r[f"{key}: previous"] = (prev if key != "vs C0" else "C0")
            r[f"{key}: delta mean $"] = d["mean"] if d else np.nan
            r[f"{key}: 95% lo $"] = d["lo"] if d else np.nan
            r[f"{key}: 95% hi $"] = d["hi"] if d else np.nan
            r[f"{key}: beyond seed noise"] = (("n/a: both deterministic, one exact difference" if det and
                                                (key == "vs previous step" and prev in DETERMINISTIC) else
                                                d["beyond noise"]) if d else False)
        r.update({f"{k} (mean)": g[k].mean() for k in num})
        rows.append(r)
    sdf = pd.DataFrame(rows).set_index("config")
    sdf.to_csv(OUT / "cumulative_summary.csv")
    flat = pd.DataFrame([{k: v for k, v in r.items() if k != "phase table"} for v in runs.values() for r in v.values()])
    flat.to_csv(OUT / "cumulative_runs.csv", index=False)
    pt = [dict(config=n, **row) for n in ("C0", "C1", "R1") for row in runs[n][SEEDS[0]]["phase table"]]
    pd.DataFrame(pt).to_csv(OUT / "cumulative_phase_tables.csv", index=False)
    # round-1 identity checks: C0 == baseline, C1 == A1c, R1 == A1a (same code, same seeds), bit for bit
    checks = []
    for n, r1 in CUM_SAME_AS_ROUND1.items():
        for o, r in runs[n].items():
            old = json.loads((OUT / "runs" / f"{r1}__s{o}.json").read_text())
            checks.append({"config": n, "round-1 name": r1, "seed offset": o, "R new": r["R $"], "R round 1": old["R $"],
                           "identical": r["R $"] == old["R $"] and r["admitted"] == old["admitted"]
                           and r["phases"] == old["phases"]})
    pd.DataFrame(checks).to_csv(OUT / "cumulative_checks.csv", index=False)
    return sdf, pd.DataFrame(pt), pd.DataFrame(checks)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "cumulative":
        OUT.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        jobs = [(n, s) for n in CUMULATIVE for s in ([SEEDS[0]] if n in DETERMINISTIC else SEEDS)]
        todo = [j for j in jobs if not (OUT / "runs" / f"{j[0]}__s{j[1]}.json").is_file()]
        print(f"cumulative: {len(jobs)} runs, {len(jobs) - len(todo)} cached, {len(todo)} to run", flush=True)
        with ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=mp.get_context("spawn")) as pool:
            futs = {pool.submit(one_run, n, s): (n, s) for n, s in todo}
            for f in futs:
                o = f.result()
                print(f"{o['ablation']:<4} s{o['seed offset']}  R = {o['R $']:,.2f}  ({o['runtime s']:.0f} s, "
                      f"{time.time() - t0:.0f} s elapsed)", flush=True)
        sdf, pt, checks = cumulative_summary()
        pd.set_option("display.width", 250)
        print(checks.to_string())
        print(sdf[["description", "R mean $", "R std $", "vs C0: delta mean $", "vs previous step: delta mean $",
                   "vs previous step: beyond seed noise"]].to_string(float_format=lambda x: f"{x:,.1f}"))
        print(f"total {time.time() - t0:.0f} s")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if mode == "run":
        jobs = [(n, s) for n in ROUND1 for s in SEEDS]
        todo = [j for j in jobs if not (OUT / "runs" / f"{j[0]}__s{j[1]}.json").is_file()]
        print(f"{len(jobs)} runs, {len(jobs) - len(todo)} cached, {len(todo)} to run", flush=True)
        with ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=mp.get_context("spawn")) as pool:
            futs = {pool.submit(one_run, n, s): (n, s) for n, s in todo}
            ref_f = pool.submit(references)
            for f in futs:
                o = f.result()
                print(f"{o['ablation']:<18} s{o['seed offset']}  R = {o['R $']:,.2f}  ({o['runtime s']:.0f} s, "
                      f"{time.time() - t0:.0f} s elapsed)", flush=True)
            ref_f.result()
    df, sdf, ref = summarise()
    b2 = df[(df.ablation == "baseline") & (df["seed offset"] == 2)]
    if len(b2):
        print(f"baseline offset 2: R = {b2['R $'].iloc[0]:,.2f} (reference 40,892.14)")
    pd.set_option("display.width", 250)
    print(sdf[["level", "R mean $", "R std $", "paired delta mean $", "paired delta 95% lo $", "paired delta 95% hi $",
               "beyond seed noise (95% CI excludes 0)"]].to_string(float_format=lambda x: f"{x:,.1f}"))
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

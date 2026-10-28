"""
Clean type-grid sweep (week 30, read-only): n_profiles = 32 fixed (so every job's snapped A, c and the job set
are identical), k* in {128, 64, 32, 16}; capacity fraction_of_concurrency_quantile q = 0.7, fraction 1.0,
drop_jobs_exceeding_capacity ON (drops nothing at f = 1.0, checked). Base: the committed default config.

    dlent_venv/bin/python notebooks/src/grid_sweep_clean.py

Per k*: capacity_grid_sweep.run_one (DLENT-RT, StaticPolicy, G1, G2, fluid OPT, job OPT MILP 600 s, Phase 0
decision errors, populated types) with the default thinning seed, plus DLENT-RT with the thinning seed set to
42, 43, 44 (online.thinning_seed_offset 0, 1, 2; 44 is the default run). Only DLENT-RT uses random numbers
(the thinning draws); run.seed (the synthetic bids, i.e. the world) stays 42 for every run.
The grid still changes s_C (Phase 0 quantiles on the types), hence R_job; s_C is reported.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks" / "src"))

from capacity_grid_sweep import run_one
from capacity_scan import scan_config

warnings.simplefilter("ignore")
K_STARS = (128, 64, 32, 16)
THIN_SEEDS = (42, 43, 44)
TAB = ROOT / "reports" / "week30" / "tables"


def cfg_grid(k_star: int, thin_seed: int = None):
    cfg = scan_config(1.0, drop=True)
    cfg = replace(cfg, grid=replace(cfg.grid, k_star=k_star),
                  hardware=replace(cfg.hardware, n_profiles=32))
    if thin_seed is not None:
        cfg = replace(cfg, online=replace(cfg.online, thinning_seed_offset=thin_seed - cfg.run.seed))
    return cfg.validate()


def run_dlent(cfg) -> dict:
    """DLENT-RT only: R, phases, endings, online decision-error rate."""
    warnings.simplefilter("ignore")
    from dlent_rt.data import load_dataset
    from dlent_rt.models.theory import TheoryModel
    from dlent_rt.phase0 import run_phase0
    from dlent_rt.reward import job_reward
    ds = load_dataset(cfg)
    st = run_phase0(ds, cfg)
    model = TheoryModel(ds, st, cfg)
    res = model.run()
    R = job_reward(model.stream, st, cfg.objective).R
    arr = res.arrivals
    r_used = np.array([a["r_tilde"] for a in arr], float)
    i_arr = np.array([a["i"] for a in arr], np.int64)
    ok = np.isfinite(r_used)
    ph = res.phases
    return dict(R=res.summary["R_total"], n_phases=len(ph),
                reasons=res.summary["phase_end_reasons"],
                phases_tau_pos=int(sum((p["tau_cpu"] > 0) or (p["tau_ram"] > 0) for p in ph)),
                dec_err=100 * np.mean(np.sign(R[i_arr[ok]]) != np.sign(r_used[ok])),
                thin_seed=cfg.run.seed + cfg.online.thinning_seed_offset)


def main():
    with ProcessPoolExecutor(max_workers=len(K_STARS) * len(THIN_SEEDS), mp_context=mp.get_context("spawn")) as pool:
        full = {k: pool.submit(run_one, cfg_grid(k), f"k*={k}", 600.0) for k in K_STARS}
        dl = {(k, s): pool.submit(run_dlent, cfg_grid(k, s)) for k in K_STARS for s in THIN_SEEDS}
        FULL = {k: f.result() for k, f in full.items()}
        DL = {key: f.result() for key, f in dl.items()}
    (TAB / "grid_sweep_clean.json").write_text(json.dumps(
        {"full": {str(k): v for k, v in FULL.items()}, "dlent": {f"{k}_{s}": v for (k, s), v in DL.items()}},
        indent=1, default=str))
    rows = []
    for k in K_STARS:
        o = FULL[k]
        Rs = np.array([DL[(k, s)]["R"] for s in THIN_SEEDS])
        de = np.array([DL[(k, s)]["dec_err"] for s in THIN_SEEDS])
        # the default-seed DLENT run inside run_one must equal the seed-44 run
        # (same sum of R_job over the same admitted set, summed in a different order)
        assert np.isclose(o["R_DLENT"], DL[(k, 44)]["R"], rtol=1e-10)
        opt = o["R_job_opt"]
        rows.append({
            "k_star": k, "c CPU": o["c_cpu"], "c RAM": o["c_ram"], "dropped (H, stream)": f"{o['dropped_H']}, {o['dropped_stream']}",
            "s_C": o["s_C"], "types": o["n_types"], "populated types": o["n_types_populated"],
            "H jobs per populated type": o["jobs_per_populated_type"],
            "R fluid OPT": o["R_fluid"], "R job OPT": opt, "MILP gap": o["milp_gap"], "R G2": o["R_G2"],
            "discretization loss (job OPT - fluid)": opt - o["R_fluid"],
            "R DLENT mean (3 seeds)": Rs.mean(), "R DLENT std (3 seeds, ddof=1)": Rs.std(ddof=1),
            "R DLENT seeds 42/43/44": " / ".join(f"{x:,.2f}" for x in Rs),
            "R StaticPolicy": o["R_static"], "R G1": o["R_G1"],
            "DLENT % job OPT (mean)": 100 * Rs.mean() / opt, "DLENT % job OPT (std)": 100 * Rs.std(ddof=1) / opt,
            "StaticPolicy % job OPT": 100 * o["R_static"] / opt, "G1 % job OPT": 100 * o["R_G1"] / opt,
            "G2 % job OPT": 100 * o["R_G2"] / opt,
            "decision errors P0 types %": o["dec_err_P0_pct"], "decision errors G1 %": o["dec_err_G1_pct"],
            "decision errors DLENT online % (mean)": de.mean(), "decision errors DLENT online % (std)": de.std(ddof=1),
            "DLENT phases (seeds)": " / ".join(str(DL[(k, s)]["n_phases"]) for s in THIN_SEEDS),
            "DLENT endings (seeds)": " / ".join(
                ",".join(f"{r}={n}" for r, n in DL[(k, s)]["reasons"].items() if n) for s in THIN_SEEDS),
            "DLENT phases tau>0 (seeds)": " / ".join(str(DL[(k, s)]["phases_tau_pos"]) for s in THIN_SEEDS),
        })
    df = pd.DataFrame(rows).set_index("k_star")
    df.to_csv(TAB / "grid_sweep_clean.csv")
    print(df.T.to_string())


if __name__ == "__main__":
    main()

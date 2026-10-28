"""
Week 32: the theoretical models at f = 0.5 (configs/v5_theory.yaml, dual rule), the MILP fixes and the
discretization cost on one week. No online model, no job-level benchmark. Logic in dlent_rt/theory_eval.py.

    dlent_venv/bin/python scripts/week32/theory_f050.py      # ~1 hour (the 1-week MILP at f = 0.5 has 3,600 s)

Each MILP runs in its own process. Writes reports/week32/theory_f050_numbers.md and the cache
reports/week32/theory_f050_cache.pkl that notebook 05 shows in its first section.
"""
from __future__ import annotations

import multiprocessing as mp
import pickle
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "week32"))

from limits import lines, out, pct, table                    # noqa: E402
from dlent_rt import theory_eval as te                       # noqa: E402
from dlent_rt.benchmarks import Benchmarks                   # noqa: E402
from dlent_rt.config import load_config                      # noqa: E402
from dlent_rt.data import load_dataset                       # noqa: E402
from dlent_rt.phase0 import run_phase0                       # noqa: E402

OUT = ROOT / "reports" / "week32"
CFG = ROOT / "configs" / "v5_theory.yaml"
WEEK, DAY = (0.0, 7.0), (0.0, 1.0)
# (name, f, days, MILP limit s, tail checks every hour or None = the old behaviour)
JOBS = [("day, f 0.5, tail checks", 0.5, DAY, 900.0, 1.0), ("day, f 0.5, old", 0.5, DAY, 900.0, None),
        ("week, f 0.5, tail checks", 0.5, WEEK, 3600.0, 1.0), ("week, f 0.5, old", 0.5, WEEK, 3600.0, None),
        ("week, f 0.3, tail checks", 0.3, WEEK, 900.0, 1.0), ("week, f 0.7, tail checks", 0.7, WEEK, 900.0, 1.0)]


def _job(spec):
    warnings.simplefilter("ignore")
    name, f, days, tl, tail = spec
    return name, te.window_models(load_config(CFG), f, days=days, seed=42, milp_time_limit=tl, tail_hours=tail)


def milp_table(W):
    rows = []
    for name, w in W:
        best = w["bucketed_believed"]
        bound = min(w["bucketed_dual_bound"], w["bucketed_lp_believed"])
        rows.append({"run": name, "jobs": f"{w['n_jobs']:,}", "candidates": f"{w['n_candidates']:,}",
                     "nonzeros": f"{w['nonzeros']:,}", "tail checks": w["n_tail_checks"],
                     "fluid OPT $": f"{w['fluid']:,.2f}", "fluid feasible $": f"{w['fluid_feasible']:,.2f}",
                     "fluid, volume inside the window $": f"{w['fluid_window']:,.2f}",
                     "MILP best $": f"{best:,.2f}", "MILP bound $": f"{bound:,.2f}", "gap": pct(w["gap"], 2),
                     "discretization cost $ (best .. bound)": f"{w['fluid_feasible'] - best:,.2f} .. {w['fluid_feasible'] - bound:,.2f}",
                     "cost %": f"{100 * (w['fluid_feasible'] - best) / w['fluid_feasible']:.2f} .. "
                               f"{100 * (w['fluid_feasible'] - bound) / w['fluid_feasible']:.2f}",
                     "fluid >= MILP best": bool(w["fluid"] >= best - 1e-6),
                     "fluid (inside window) >= MILP bound": bool(w["fluid_window"] >= bound - 1e-6),
                     "MILP status": w["status"], "MILP s": f"{w['milp_runtime_s']:.0f}",
                     "LP relaxation s": f"{w['milp_lp_runtime_s']:.0f}", "MILP realised (job rewards) $": f"{w['bucketed_realised']:,.2f}"})
    return pd.DataFrame(rows)


def summary(cfg, week):
    r = te.evaluate_f(cfg, cfg.capacity.f, seed=cfg.run.seed)
    ds = load_dataset(cfg)
    st0 = run_phase0(ds, cfg)
    h = te.h_vs_stream(Benchmarks(ds, st0, cfg), st0, cfg)
    s = r["static"]
    w24, w6 = r["windows"][24], r["windows"][6]
    em = cfg.energy_model
    best = week["bucketed_believed"]
    bound = min(week["bucketed_dual_bound"], week["bucketed_lp_believed"])
    rows = [
        ("f", f"{cfg.capacity.f}"),
        ("C, CPU / RAM (machines)", f"{r['C'][0]:.3f} / {r['C'][1]:.3f} ({r['C'][0] * em.machine_cores:,.0f} cores / "
                                    f"{r['C'][1] * em.machine_ram_gb:,.0f} GB, declared 64 / 256)"),
        ("tau, Phase 0 LP (CPU, RAM)", f"{r['phase0_lp']['tau'][0]:.5f}, {r['phase0_lp']['tau'][1]:.5f}"),
        ("tau, fluid OPT (CPU, RAM)", f"{r['fluid']['tau'][0]:.5f}, {r['fluid']['tau'][1]:.5f}"),
        ("windows with tau > 0, 1 day (CPU, RAM)", f"{pct((w24[:, 0] > 0).mean())}, {pct((w24[:, 1] > 0).mean())}"),
        ("windows with tau > 0, 6 h (CPU, RAM)", f"{pct((w6[:, 0] > 0).mean())}, {pct((w6[:, 1] > 0).mean())}"),
        ("static policy (dual rule): time >= 95% full", pct(s["full_time"])),
        ("static policy: mean occupancy / C (CPU, RAM)", f"{pct(s['mean_occupancy_over_C'][0])}, {pct(s['mean_occupancy_over_C'][1])}"),
        ("static policy: jobs rejected for lack of space", pct(s["rejected_space"], 2)),
        ("static policy: realised reward $ (CV / profit / carbon)",
         f"{s['split']['reward']:,.0f} ({s['split']['customer value']:,.0f} / {s['split']['profit']:,.0f} / {s['split']['carbon']:,.0f})"),
        ("fluid OPT $ (CV / profit / carbon)", f"{r['fluid']['split']['reward']:,.0f} ({r['fluid']['split']['customer value']:,.0f} / "
                                               f"{r['fluid']['split']['profit']:,.0f} / {r['fluid']['split']['carbon']:,.0f})"),
        ("Phase 0 LP expected $", f"{r['phase0_lp']['expected']['reward']:,.0f}"),
        ("decomposition: Phase 0 LP -> stream statistics (Phase 0 grid) -> fluid OPT $",
         f"{h['phase0_expected']:,.0f} -> {h['stream_stats_phase0_grid_opt']:,.0f} -> {h['fluid_opt']:,.0f}"),
        ("static policy / fluid OPT", pct(s["split"]["reward"] / r["fluid"]["split"]["reward"])),
        ("discretization cost, one week (fluid feasible - bucketed MILP)",
         f"{week['fluid_feasible'] - best:,.0f} .. {week['fluid_feasible'] - bound:,.0f} $ "
         f"({100 * (week['fluid_feasible'] - best) / week['fluid_feasible']:.1f}% .. "
         f"{100 * (week['fluid_feasible'] - bound) / week['fluid_feasible']:.1f}%; MILP gap {pct(week['gap'], 2)})"),
    ]
    return pd.DataFrame(rows, columns=["quantity", "value"])


def main() -> None:
    warnings.simplefilter("ignore")
    cfg = load_config(CFG)
    out("# Week 32: theoretical models at f = 0.5, all numbers\n")
    out(f"Config configs/v5_theory.yaml: f = {cfg.capacity.f}, policy.unseen_types = {cfg.policy.unseen_types}, "
        f"quota at f = {cfg.limits.quota_from_f}.")
    with ProcessPoolExecutor(max_workers=len(JOBS), mp_context=mp.get_context("spawn")) as pool:
        W = list(pool.map(_job, JOBS))
    n1 = milp_table(W)
    table(n1, "N1. Bucketed MILP on a window (reference buckets, true durations; each MILP in its own process)")
    week = dict(W)["week, f 0.5, tail checks"]
    n2 = summary(cfg, week)
    table(n2, "N2. The theoretical models at f = 0.5 (v5_theory, dual rule)")
    (OUT / "theory_f050_numbers.md").write_text("\n".join(lines) + "\n")
    with open(OUT / "theory_f050_cache.pkl", "wb") as fh:
        pickle.dump(dict(milp=n1, summary=n2, raw=dict(W)), fh)


if __name__ == "__main__":
    main()

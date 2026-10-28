"""
Week 32, theory models part 2 on configs/v5_theory.yaml (no online model, no job-level benchmark):
the static Phase 0 policy with unseen types rejected vs the dual rule, H vs stream per type, and the bucketed MILP on
one window of the stream. Logic in dlent_rt/theory_eval.py.

    dlent_venv/bin/python scripts/week32/theory_models_2.py          # everything (~1 hour: the 1-week MILPs)
    dlent_venv/bin/python scripts/week32/theory_models_2.py --day    # only the 1-day MILPs; appends N6 to the numbers

Writes reports/week32/theory_models_2_numbers.md, reports/week32/figs/fig21_occupancy_both_options.png and the
cache reports/week32/theory_models_2_cache.pkl that notebook 05 reads (the bucketed MILP takes ~15 min per f).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
FIGS = OUT / "figs"
F_SCAN = (0.10, 0.20, 0.30, 0.50, 0.70, 1.25)
OPTIONS = ("reject", "dual_rule")
MILP_F = (0.30, 0.50, 0.70)
MILP_DAYS = (0.0, 7.0)          # the first 7 days of the stream (days 8-14 of the month)
MILP_TL = 900.0


MILP_DAYS_SHORT = (0.0, 1.0)    # the first day of the stream (day 8 of the month)


def _window(f, days=MILP_DAYS):
    """One window MILP in its own process: the HiGHS time limit also counts earlier solves of the same process."""
    warnings.simplefilter("ignore")
    return te.window_models(load_config(ROOT / "configs" / "v5_theory.yaml"), f, days=days, seed=42,
                            milp_time_limit=MILP_TL)


def _window_day(f):
    return _window(f, MILP_DAYS_SHORT)


def milp_rows(W):
    rows = []
    for f, w in zip(MILP_F, W):
        dc = w["fluid_feasible"] - w["bucketed_believed"]
        rows.append({"f": f, "jobs in the window": f"{w['n_jobs']:,}", "MILP candidates / nonzeros": f"{w['n_candidates']:,} / {w['nonzeros']:,}",
                     "fluid OPT $": f"{w['fluid']:,.2f}", "fluid feasible $": f"{w['fluid_feasible']:,.2f}",
                     "bucketed MILP (type rewards) $": f"{w['bucketed_believed']:,.2f}",
                     "bucketed MILP realised (job rewards) $": f"{w['bucketed_realised']:,.2f}",
                     "discretization cost $": f"{dc:,.2f}", "discretization cost %": pct(dc / w["fluid_feasible"], 2),
                     "fluid >= feasible >= MILP": bool(w["fluid"] >= w["fluid_feasible"] - 1e-6 >= w["bucketed_believed"] - 2e-6),
                     "MILP status, gap": f"{w['status']}, {w['gap']:.2e}", "MILP runtime s": f"{w['milp_runtime_s']:.0f}",
                     "MILP LP-relaxation bound $": f"{w['bucketed_lp_believed']:,.2f}",
                     "cost range (LP bound .. incumbent) $": f"{w['fluid_feasible'] - w['bucketed_lp_believed']:,.2f} .. {dc:,.2f}"})
    return pd.DataFrame(rows)


def day_only() -> None:
    import multiprocessing as mp
    import pickle
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=len(MILP_F), mp_context=mp.get_context("spawn")) as pool:
        W = list(pool.map(_window_day, MILP_F))
    n6 = milp_rows(W)
    lines.clear()
    table(n6, f"N6. One DAY of the stream (days {MILP_DAYS_SHORT[0]:g}-{MILP_DAYS_SHORT[1]:g} after H), same models")
    with open(OUT / "theory_models_2_numbers.md", "a") as fh:
        fh.write("\n".join(lines) + "\n")
    cache = OUT / "theory_models_2_cache.pkl"
    with open(cache, "rb") as fh:
        c = pickle.load(fh)
    c["n6"] = n6
    with open(cache, "wb") as fh:
        pickle.dump(c, fh)


def main() -> None:
    warnings.simplefilter("ignore")
    cfg = load_config(ROOT / "configs" / "v5_theory.yaml")
    out("# Week 32: theory models part 2 (configs/v5_theory.yaml), all numbers\n")

    # ---- Part 1: unseen types ---------------------------------------------------------------------
    R = [te.evaluate_f(cfg, f, seed=cfg.run.seed, windows_h=(), unseen_options=OPTIONS) for f in F_SCAN]
    rows = []
    for r in R:
        fl = r["fluid"]["split"]["reward"]
        for opt in OPTIONS:
            s = r["static_by_option"][opt]
            rows.append({"f": r["f"], "unseen types": opt, "% time >= 95% full (CPU or RAM)": pct(s["full_time"]),
                         "mean occupancy / C (CPU, RAM)": f"{pct(s['mean_occupancy_over_C'][0])}, {pct(s['mean_occupancy_over_C'][1])}",
                         "% jobs rejected for lack of space": pct(s["rejected_space"], 2),
                         "jobs admitted": f"{s['n_admitted']:,}", "unseen-type jobs admitted": f"{s['unseen_admitted']:,}",
                         "realised reward $": f"{s['split']['reward']:,.1f}",
                         "split CV / profit / carbon $": f"{s['split']['customer value']:,.0f} / {s['split']['profit']:,.0f} / {s['split']['carbon']:,.0f}",
                         "static / fluid OPT": pct(s["split"]["reward"] / fl)})
    n1 = pd.DataFrame(rows)
    table(n1, "N1. Static Phase 0 policy on the real stream: unseen types rejected vs dual rule (seed 42)")
    out(f"\nStream jobs in types unseen in H: {pct(R[0]['static_by_option']['reject']['unseen_type_share'], 2)}, "
        f"holding {pct(R[0]['static_by_option']['reject']['unseen_type_cpu_hours'])} of the stream's CPU-hours.")
    sel = {}
    for opt in OPTIONS:
        sel[opt] = te.select_f(R, option=opt)
        s = next(r for r in R if r["f"] == sel[opt])["static_by_option"][opt]
        out(f"Fixed rule (closest to 20-30%, ties to 25%), {opt}: f = {sel[opt]:g} ({pct(s['full_time'])}).")

    fsel = sel["dual_rule"]
    rs = next(r for r in R if r["f"] == fsel)
    fig, axs = plt.subplots(4, 1, figsize=(11, 10.5), sharex=True)
    te.fig_occupancy(rs, axs[:2], cfg.time.tick_seconds, option="reject")
    te.fig_occupancy(rs, axs[2:], cfg.time.tick_seconds, option="dual_rule")
    fig.tight_layout()
    fig.savefig(FIGS / "fig21_occupancy_both_options.png", dpi=150)
    plt.close(fig)

    # ---- Part 2: H vs stream ----------------------------------------------------------------------------
    from dataclasses import replace
    rows = []
    hv = None
    for f in (0.30, 0.50, 0.70):
        c = replace(cfg, capacity=replace(cfg.capacity, f=f)).validate()
        ds = load_dataset(c)
        st0 = run_phase0(ds, c)
        B = Benchmarks(ds, st0, c)
        h = te.h_vs_stream(B, st0, c)
        hv = h if f == 0.50 else hv
        rows.append({"f": f, "Phase 0 LP expected $": f"{h['phase0_expected']:,.1f}",
                     "LP on stream statistics, Phase 0 grid $": f"{h['stream_stats_phase0_grid_opt']:,.1f}",
                     "Phase 0 x on stream statistics $": f"{h['phase0_x_on_stream_stats']:,.1f}",
                     "fluid OPT (refit grid) $": f"{h['fluid_opt']:,.1f}"})
    out(f"\nN2. Types (Phase 0 grid): with jobs in H {hv['n_types_H']:,}, in the stream {hv['n_types_stream']:,}, in both "
        f"{hv['n_types_both']:,}. Seen in both, weighted by H's rate: mean ratio stream / H of the arrival rate "
        f"{hv['lambda_ratio']:.3f}, of the mean r*d {hv['rd_ratio']:.3f}, of r~ {hv['r_tilde_ratio']:.3f}. "
        f"CPU load (machines = sum of rate x mean r*d per hour): H {hv['load_H']:.3f}, stream {hv['load_stream']:.3f}, "
        f"of which in types unseen in H {hv['load_stream_unseen']:.3f}.")
    top = hv["top"].copy()
    for col in ("lambda H /h", "lambda stream /h", "mean r*d H", "mean r*d stream", "load H", "load stream", "load gap"):
        top[col] = top[col].map(lambda x: f"{x:.4g}")
    table(top, "N3. The 10 types (seen in both) whose CPU load differs most between H and the stream (f-independent)")
    n4 = pd.DataFrame(rows)
    table(n4, "N4. Phase 0 LP expected vs fluid OPT: decomposition (stream T)")

    # ---- Part 3: bucketed MILP on one window --------------------------------------------------------------
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=len(MILP_F), mp_context=mp.get_context("spawn")) as pool:
        W = list(pool.map(_window, MILP_F))
    rows = []
    for f, w in zip(MILP_F, W):
        dc = w["fluid_feasible"] - w["bucketed_believed"]
        rows.append({"f": f, "jobs in the window": f"{w['n_jobs']:,}", "MILP candidates / nonzeros": f"{w['n_candidates']:,} / {w['nonzeros']:,}",
                     "fluid OPT $": f"{w['fluid']:,.2f}", "fluid feasible $": f"{w['fluid_feasible']:,.2f}",
                     "bucketed MILP (type rewards) $": f"{w['bucketed_believed']:,.2f}",
                     "bucketed MILP realised (job rewards) $": f"{w['bucketed_realised']:,.2f}",
                     "discretization cost $": f"{dc:,.2f}", "discretization cost %": pct(dc / w["fluid_feasible"], 2),
                     "fluid >= feasible >= MILP": bool(w["fluid"] >= w["fluid_feasible"] - 1e-6 >= w["bucketed_believed"] - 2e-6),
                     "MILP status, gap": f"{w['status']}, {w['gap']:.2e}", "MILP runtime s": f"{w['milp_runtime_s']:.0f}",
                     "LPs runtime s": f"{w['lp_runtime_s']:.1f}", "true-duration feasible": w["feasible_true"],
                     "MILP LP-relaxation bound $": f"{w['bucketed_lp_believed']:,.2f}",
                     "cost range (LP bound .. incumbent) $": f"{w['fluid_feasible'] - w['bucketed_lp_believed']:,.2f} .. {dc:,.2f}"})
    n5 = pd.DataFrame(rows)
    table(n5, f"N5. One window of the stream (days {MILP_DAYS[0]:g}-{MILP_DAYS[1]:g} after H), "
                              f"reference buckets, true durations, MILP time limit {MILP_TL:g} s")
    (OUT / "theory_models_2_numbers.md").write_text("\n".join(lines) + "\n")
    import pickle
    with open(OUT / "theory_models_2_cache.pkl", "wb") as fh:
        pickle.dump(dict(n1=n1, sel=sel, selected=rs, hv=hv, top=top, n4=n4, n5=n5), fh)


if __name__ == "__main__":
    day_only() if "--day" in sys.argv else main()

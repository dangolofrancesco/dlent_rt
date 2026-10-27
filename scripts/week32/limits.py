"""
Week 32, step 3: effective duration, request / duration distributions, the two limits R_max and d_max, and the
Phase 0 type volume. No model is run: load_dataset, counting, the FCFS replay, and the Phase 0 type assignment
(bids -> grid -> type ids; no LP).

    dlent_venv/bin/python scripts/week32/limits.py

Base: configs/regime_v2.yaml with time.duration_mode = effective. Writes reports/week32/limits_numbers.md and
reports/week32/figs/fig4_request_size.png, fig5_duration.png, fig6_demand_limits.png.
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "week32"))

from data_audit import BLUE, INK, INK2, demand, fcfs, pct   # noqa: E402
from dlent_rt.bid import bid_model                          # noqa: E402
from dlent_rt.config import load_config                     # noqa: E402
from dlent_rt.data import _assign_real_tick_steps, load_dataset   # noqa: E402
from dlent_rt.grid import build_valuation_grid              # noqa: E402
from dlent_rt.phase0 import (                               # noqa: E402
    _synthesize_bids, assign_type_ids, build_type_space, type_arrays, type_request,
)

OUT = ROOT / "reports" / "week32"
FIGS = OUT / "figs"
R_MAX = (1.0, 2.0, 5.0, 10.0, 20.0, None)
R_EXTRA = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5)   # extra scan below 1 machine: where the recommendation rule can hold
D_MAX = (4.0, 12.0, None)              # None = 0.05 T (the code's ceiling)
ALMOST_NONE = 0.01                     # "(almost) no job larger than the cluster": they hold < 1% of the kept work
ORANGE = "#eb6834"
lines: list[str] = []


def out(s: str = "") -> None:
    print(s)
    lines.append(s)


def table(df: pd.DataFrame, title: str) -> None:
    out(f"\n### {title}\n")
    out("| " + " | ".join(map(str, df.columns)) + " |")
    out("|" + "---|" * len(df.columns))
    for r in df.itertuples(index=False):
        out("| " + " | ".join(map(str, r)) + " |")


def base_cfg():
    cfg = load_config(ROOT / "configs" / "regime_v2.yaml")
    return replace(cfg, time=replace(cfg.time, duration_mode="effective")).validate()


def cfg_for(base, r_max, d_max):
    o = replace(base.outliers, policy="none" if r_max is None else "r_max", r_max_machines=r_max)
    return replace(base, outliers=o, time=replace(base.time, d_max_hours=d_max)).validate()


def timeline(ds, cfg):
    """Absolute arrival steps, capped duration steps, A and capped hours of H + stream (one shared timeline)."""
    J = (ds.H, ds.stream)
    dts = np.concatenate([j.datetime for j in J])
    steps = _assign_real_tick_steps(dts, dts[0], cfg.time.tick_seconds)
    arr = np.concatenate([ds.H.arrival_step, ds.stream.arrival_step + steps[cfg.data.n0]]).astype(float)
    assert np.array_equal(arr, steps)
    return (arr, np.concatenate([j.duration_steps for j in J]).astype(float), np.concatenate([j.A for j in J]),
            np.concatenate([j.duration_hours for j in J]), dts)


def dist_figure(x, w, edges, marks, xlabel, title, path):
    """Two panels on one log x axis: jobs per bin; share of CPU-hours per bin with the cumulative share line."""
    n, _ = np.histogram(x, bins=edges)
    s, _ = np.histogram(x, bins=edges, weights=w)
    s = s / w.sum()
    order = np.argsort(x)
    cx, cw = x[order], np.cumsum(w[order]) / w.sum()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6.6), sharex=True)
    widths = np.diff(edges)
    a1.bar(edges[:-1], n, width=widths * 0.92, align="edge", color=BLUE)
    a1.set_ylabel("number of jobs", color=INK)
    a1.set_title(f"{title}: number of jobs", loc="left", fontsize=10, color=INK)
    a2.bar(edges[:-1], 100 * s, width=widths * 0.92, align="edge", color=BLUE, label="share of CPU-hours in the bin")
    a2.step(cx, 100 * cw, where="post", color=ORANGE, lw=2, label="cumulative share of work (jobs up to x)")
    a2.set_ylabel("share of CPU-hours (%)", color=INK)
    a2.set_ylim(0, 102)
    a2.set_title(f"{title}: share of the work (CPU-hours)", loc="left", fontsize=10, color=INK)
    a2.legend(loc="upper left", frameon=False, fontsize=8)
    for ax in (a1, a2):
        ax.set_xscale("log")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color="#e5e4e0", lw=0.6)
        for m, lbl in marks:
            ax.axvline(m, color=INK2, lw=1.0, ls=":")
    top = a1.get_ylim()[1]
    for m, lbl in marks:
        a1.text(m, 0.97 * top, f" {lbl} ", rotation=90, ha="right", va="top", fontsize=8, color=INK2)
    a2.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    warnings.simplefilter("ignore")
    FIGS.mkdir(parents=True, exist_ok=True)
    base = base_cfg()
    em = base.energy_model
    job = pd.read_csv(ROOT / base.data.batch_csv)
    span = job.duration.to_numpy(float)
    cpu, ram = job.CPU.to_numpy(float), job.RAM.to_numpy(float)
    eff = np.where(cpu > 0, job.CPU_hours_real / np.where(cpu > 0, cpu, 1), span)
    out("# Week 32 step 3: all numbers (limits.md)\n")
    out(f"Jobs: {len(job)} (data/batch_v2_30k.csv). Jobs with a zero CPU request (keep the span): {int((cpu == 0).sum())}.")

    # ---- A1: effective duration --------------------------------------------------
    ch = np.abs(eff - span) > 0.10 * span
    table(pd.DataFrame([
        {"duration": "span (first start to last end)", "CPU-hours": f"{(cpu * span).sum():,.0f}",
         "RAM-hours": f"{(ram * span).sum():,.0f}", "mean duration (h)": f"{span.mean():.3f}"},
        {"duration": "effective (real CPU-hours / request)", "CPU-hours": f"{(cpu * eff).sum():,.0f}",
         "RAM-hours": f"{(ram * eff).sum():,.0f}", "mean duration (h)": f"{eff.mean():.3f}"},
        {"duration": "real, each instance on its own span (reference)", "CPU-hours": f"{job.CPU_hours_real.sum():,.0f}",
         "RAM-hours": f"{job.RAM_hours_real.sum():,.0f}", "mean duration (h)": "-"},
    ]), "N1. Span versus effective duration (no cap, no limit)")
    out(f"\nJobs whose duration changes by more than 10%: {int(ch.sum())} ({pct(ch.mean())}); "
        f"they hold {pct((cpu * span)[ch].sum() / (cpu * span).sum())} of the span CPU-hours. "
        f"Effective / span: median {np.median(eff / span):.3f}, q0.10 {np.quantile(eff / span, .1):.3f}.")

    # ---- A3: the two limits -------------------------------------------------------
    total_cpuh = (cpu * eff).sum()
    rows, rows_x, res = [], [], {}
    for r in R_EXTRA + R_MAX:
        for d in D_MAX:
            cfg = cfg_for(base, r, d)
            ds = load_dataset(cfg)
            arr, dst, A, dh, dts = timeline(ds, cfg)
            c = ds.capacity
            big = (A > c[None, :]).any(1)
            work = A[:, 0] * dst
            fc = fcfs(arr, dst, A, c, 0.0, arr.max())
            key = (r, d)
            res[key] = dict(ds=ds, cfg=cfg, arr=arr, dst=dst, A=A, c=c, dts=dts,
                            kept_jobs=len(A) / len(job), kept_cpuh=(A[:, 0] * dh).sum() / total_cpuh,
                            big_share=work[big].sum() / work.sum(), n_big=int(big.sum()), fc=fc)
            (rows if r in R_MAX else rows_x).append({"R_max": "no limit" if r is None else f"{r:g}",
                         "d_max": f"{ds.d_max_steps / 3600:.2f} h" + (" (0.05 T)" if d is None else ""),
                         "T (h)": f"{ds.T / 3600:.1f}", "jobs kept": pct(res[key]["kept_jobs"]),
                         "CPU-hours kept": pct(res[key]["kept_cpuh"]),
                         "cluster CPU": f"{c[0]:.2f}", "cluster RAM": f"{c[1]:.2f}",
                         "jobs larger than cluster": res[key]["n_big"], "their work": pct(res[key]["big_share"]),
                         "FCFS jobs accepted": pct(fc["acc_jobs"]), "FCFS CPU-hours served": pct(fc["acc_cpuh"]),
                         "time >= 95% full": pct(fc["full_time"])})
    table(pd.DataFrame(rows), "N2. R_max x d_max (machines). CPU-hours kept: of all 30,000 jobs, effective duration, no cap")
    table(pd.DataFrame(rows_x), "N2b. Extra scan below 1 machine (same columns)")
    ok = [k for k, v in res.items() if v["big_share"] < ALMOST_NONE]
    if not ok:
        raise SystemExit("no (R_max, d_max) pair meets the rule; widen R_EXTRA")
    r_best = max((k[0] for k in ok), key=lambda r: np.inf if r is None else r)
    d_best = max((k for k in ok if k[0] == r_best), key=lambda k: res[k]["kept_cpuh"])[1]
    best = res[(r_best, d_best)]
    out(f"\nRule: largest R_max with the jobs larger than the cluster holding < {pct(ALMOST_NONE, 0)} of the kept work; "
        f"among those, the d_max that keeps the most work. -> R_max = {r_best}, d_max = "
        f"{best['ds'].d_max_steps / 3600:.2f} h ({'0.05 T' if d_best is None else f'{d_best:g} h'}). "
        f"Keeps {pct(best['kept_jobs'])} of the jobs and {pct(best['kept_cpuh'])} of the CPU-hours. Cluster "
        f"{best['c'][0]:.2f} CPU / {best['c'][1]:.2f} RAM machines ({best['c'][0] * em.machine_cores:,.0f} cores / "
        f"{best['c'][1] * em.machine_ram_gb:,.0f} GB under 64 / 256). Pairs that pass: "
        + ", ".join(f"({k[0]}, {'0.05T' if k[1] is None else k[1]})" for k in ok) + ".")
    ds_b, cfg_b = best["ds"], best["cfg"]
    out(f"T at the recommended pair: {ds_b.T / 3600:.1f} h; 0.05 T = {0.05 * ds_b.T / 3600:.2f} h; "
        f"xi on the data = {ds_b.xi:.3f} (forced xi_override = {cfg_b.constants.xi_override}).")
    t_end_0 = res[(None, None)]["ds"].T / 3600

    # option for the open decision: the cluster is never smaller than R_max (cluster = max(25% level, R_max))
    rows_o = []
    for r in (1.0, 2.0, 5.0, 10.0, 20.0):
        v = res[(r, None)]
        cc = np.maximum(v["c"], r)
        big = (v["A"] > cc[None, :]).any(1)
        fc = fcfs(v["arr"], v["dst"], v["A"], cc, 0.0, v["arr"].max())
        rows_o.append({"R_max": f"{r:g}", "d_max": f"{v['ds'].d_max_steps / 3600:.2f} h (0.05 T)",
                       "CPU-hours kept": pct(v["kept_cpuh"]), "cluster CPU": f"{cc[0]:.2f}", "cluster RAM": f"{cc[1]:.2f}",
                       "jobs larger than cluster": int(big.sum()), "FCFS jobs accepted": pct(fc["acc_jobs"]),
                       "FCFS CPU-hours served": pct(fc["acc_cpuh"]), "time >= 95% full": pct(fc["full_time"])})
    table(pd.DataFrame(rows_o), "N4. Option (not a current rule): cluster = max(level exceeded 25% of the time, R_max)")

    # ---- A2: distributions ----------------------------------------------------------
    pos = cpu > 0
    w = (cpu * eff)[pos]
    dist_figure(cpu[pos], w, np.logspace(-5, 3, 33),
                [(r, f"R_max {r:g}") for r in R_MAX if r is not None],
                "CPU request of the job (machines, log scale)", "Request size", FIGS / "fig4_request_size.png")
    dist_figure(eff[pos], w, np.logspace(-4, 3, 29),
                [(4.0, "d_max 4 h"), (12.0, "d_max 12 h"), (0.05 * t_end_0, f"d_max 0.05 T = {0.05 * t_end_0:.1f} h")],
                "effective duration of the job (hours, log scale)", "Duration", FIGS / "fig5_duration.png")
    out(f"\nZero-CPU jobs left out of the log-scale figures: {int((~pos).sum())}.")
    for r in (1.0, 2.0, 5.0, 10.0, 20.0):
        out(f"CPU request <= {r:g} machines: {pct((cpu <= r).mean(), 2)} of jobs, "
            f"{pct((cpu * eff)[cpu <= r].sum() / total_cpuh)} of the CPU-hours.")
    for d in (4.0, 12.0, 0.05 * t_end_0):
        out(f"Effective duration <= {d:.2f} h: {pct((eff <= d).mean(), 2)} of jobs, "
            f"{pct((cpu * eff)[eff <= d].sum() / total_cpuh)} of the CPU-hours.")

    # figure: demand over the month at the recommended pair
    _, lev, ww = demand(best["arr"], best["dst"], best["A"], 0.0, best["arr"].max())
    tt = np.concatenate([[0.0], np.cumsum(ww)])[:-1] / 86400
    fig, axs = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
    for i, (ax, rs) in enumerate(zip(axs, ["CPU", "RAM"])):
        ax.step(tt, lev[:, i], where="post", color=BLUE, lw=1.0)
        ax.axhline(best["c"][i], color=INK2, lw=1.0, ls="--")
        ax.text(tt[-1] + 0.3, best["c"][i], f"exceeded 25% of time:\n{best['c'][i]:,.2f} = cluster", va="center",
                fontsize=8, color=INK2)
        ax.set_ylabel(f"{rs} demand (machines)", color=INK)
        ax.set_title(f"{rs}: total request of running jobs, R_max = {r_best:g}, d_max = {ds_b.d_max_steps / 3600:.1f} h "
                     f"(peak {lev[:, i].max():,.1f})", loc="left", fontsize=10, color=INK)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color="#e5e4e0", lw=0.6)
    axs[-1].set_xlabel(f"days since the first arrival ({pd.Timestamp(best['dts'][0])} UTC)")
    axs[-1].set_xlim(0, tt[-1])
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    fig.savefig(FIGS / "fig6_demand_limits.png", dpi=150)
    plt.close(fig)

    # ---- A4: type volume, profile versus the mean real request ----------------------
    H = ds_b.H
    bids = _synthesize_bids(H, cfg_b, seed=cfg_b.run.seed)
    my = bid_model(cfg_b, bids.v_rate)
    grid = build_valuation_grid(bids.v_rate, my, cfg_b.grid.k_star, cfg_b.grid.space, cfg_b.grid.phi_floor)
    ts = build_type_space(H, ds_b, cfg_b)
    tid = assign_type_ids(ts, np.asarray(grid.assign(bids.v_rate, my)), H.priority, H.hw_idx)
    okt = tid >= 0
    _, _, A_prof = type_arrays(ts, grid)
    A_real = type_request(tid[okt], H.A[okt], A_prof, "real_mean")
    N = np.bincount(tid[okt], minlength=ts.n_types)
    pop = N > 0
    rows = []
    for i, rs in enumerate(("CPU", "RAM")):
        ratio = A_real[pop, i] / A_prof[pop, i]
        wq = np.repeat(ratio, N[pop])
        rows.append({"resource": rs, "types with jobs": int(pop.sum()),
                     "median real / profile": f"{np.median(ratio):.3f}", "q0.10": f"{np.quantile(ratio, .1):.3f}",
                     "q0.90": f"{np.quantile(ratio, .9):.3f}", "job-weighted median": f"{np.median(wq):.3f}",
                     "total volume real / profile": f"{(A_real[pop, i] * N[pop]).sum() / (A_prof[pop, i] * N[pop]).sum():.3f}"})
    table(pd.DataFrame(rows), "N3. Phase 0 type volume: mean real request of the type / hardware profile "
                              "(H jobs at the recommended pair; same d_bar, so the volume ratio = the request ratio)")
    (OUT / "limits_numbers.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

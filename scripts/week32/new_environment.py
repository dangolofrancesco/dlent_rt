"""
Week 32: the v3 environment (configs/v3_mean_demand.yaml): filled grid hours, Little's-law check of D_bar, the f scan
(FCFS, no prices), the distributions before / after the cap and the quota. No online model is run.

    dlent_venv/bin/python scripts/week32/new_environment.py            # numbers + figures
    dlent_venv/bin/python scripts/week32/new_environment.py --select   # also print the f the fixed rule selects

Writes reports/week32/new_environment_numbers.md and reports/week32/figs/fig7_cdf_request.png,
fig8_cdf_duration.png, fig9_demand_v3.png.
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
sys.path.insert(0, str(ROOT / "scripts" / "week32"))

from limits import BLUE, INK, INK2, ORANGE, demand, fcfs, pct, table, out, lines, timeline   # noqa: E402
from dlent_rt.config import load_config                                                       # noqa: E402
from dlent_rt.data import load_dataset                                                        # noqa: E402
from dlent_rt.phase0 import _compute_job_energy, _lookup_grid                                 # noqa: E402

OUT = ROOT / "reports" / "week32"
FIGS = OUT / "figs"
CFG = ROOT / "configs" / "v3_mean_demand.yaml"
F_SCAN = (0.10, 0.25, 0.50, 0.75, 1.00)
BAND, TARGET = (0.25, 0.50), 0.35          # selection rule, fixed in advance
FULL = 0.95
FILL_START = np.datetime64("2019-06-01T04:00")
QS = (0.50, 0.75, 0.90, 0.95, 0.99, 0.999, 1.0)


def wq(x, w, q):
    """Weighted quantile (step): the smallest x whose cumulative weight share reaches q."""
    o = np.argsort(x, kind="stable")
    cw = np.cumsum(np.asarray(w, float)[o]) / np.sum(w)
    return float(np.asarray(x)[o][min(np.searchsorted(cw, q - 1e-12), len(o) - 1)])


def select_f(rows):
    """Distance of '% time >= 95% of C' to the 25-50% band (0 inside); ties -> closest to 35%."""
    def key(r):
        x = r["full"]
        return (max(BAND[0] - x, 0.0, x - BAND[1]), abs(x - TARGET))
    return min(rows, key=key)["f"]


def main(select: bool) -> None:
    warnings.simplefilter("ignore")
    cfg = load_config(CFG)
    job = pd.read_csv(ROOT / cfg.data.batch_csv)
    out("# Week 32: the v3 environment, all numbers\n")
    out(f"Config: configs/v3_mean_demand.yaml. Jobs in the file: {len(job):,}.")

    ds = load_dataset(cfg)
    arr, dst, A, dh, dts = timeline(ds, cfg)
    lr, cd = ds.limit_report, ds.capacity_diag
    out(f"Quota (p99 of H, machines): CPU {lr.quota[0]:.4f}, RAM {lr.quota[1]:.4f}. Rejected: {lr.n_quota_rejected} jobs "
        f"({pct(lr.n_quota_rejected / lr.n_before, 2)}). Kept: {lr.n_after:,}. d_max = d_cap = {ds.d_max_steps / 3600:g} h; "
        f"T = {ds.T / 3600:.1f} h; |H| = {cd['span_hours']:.1f} h, n_H = {cd['n']}.")

    # ---- Part 1: filled grid hours ------------------------------------------------
    J = (ds.H, ds.stream)
    E = np.concatenate([_compute_job_energy(j, cfg) for j in J])
    price = np.concatenate([np.asarray(_lookup_grid(j, ds)[0]) for j in J])
    cost = E * price
    end = dts + (dh * 3600).astype("timedelta64[s]")
    in_fill = end > FILL_START
    arr_fill = dts >= FILL_START
    out(f"\nP1. Grid filled from {FILL_START} to 2019-06-03T00:00 UTC (45 hours). Jobs running at least partly in filled "
        f"hours: {int(in_fill.sum())} ({pct(in_fill.mean(), 2)}); of them arriving in filled hours (so priced with a "
        f"filled hour): {int(arr_fill.sum())}. Their share of the total energy cost: running {pct(cost[in_fill].sum() / cost.sum(), 2)}, "
        f"arriving {pct(cost[arr_fill].sum() / cost.sum(), 2)}. (The model prices a job at its arrival hour.)")

    # ---- Part 4: D_bar, Little's law, largest request / C ---------------------------
    n0 = ds.H.n
    _, levH, wH = demand(arr[:n0], dst[:n0], A[:n0], arr[0], arr[n0 - 1])
    occ_H = (levH * wH[:, None]).sum(0) / wH.sum()
    gap = occ_H / cd["D_bar"] - 1
    out(f"\nP4. H: lambda_hat = {cd['lambda_hat']:.3f} jobs/h; mean r*d = {cd['mean_rd'][0]:.5f} CPU, {cd['mean_rd'][1]:.5f} RAM "
        f"machine-hours/job; D_bar = {cd['D_bar'][0]:.4f} CPU, {cd['D_bar'][1]:.4f} RAM machines. Little's law: "
        f"time-average occupancy of H with no capacity over H's span = {occ_H[0]:.4f} CPU, {occ_H[1]:.4f} RAM; "
        f"gap {100 * gap[0]:+.1f}% CPU, {100 * gap[1]:+.1f}% RAM.")
    amax = A.max(0)

    # ---- Part 5: f scan on the full month ---------------------------------------------
    _, lev, w = demand(arr, dst, A, arr[0], arr[-1])
    rows, scan = [], []
    for f in F_SCAN:
        C = f * cd["D_bar"]
        full_r = [(w * (lev[:, i] >= FULL * C[i])).sum() / w.sum() for i in range(2)]
        full_any = (w * (lev >= FULL * C[None, :]).any(1)).sum() / w.sum()
        fc = fcfs(arr, dst, A, C, arr[0], arr[-1])
        scan.append(dict(f=f, full=full_any))
        rows.append({"f": f"{f:.2f}", "C CPU": f"{C[0]:.4f}", "C RAM": f"{C[1]:.4f}",
                     "% time demand >= 95% C (CPU)": pct(full_r[0]), "(RAM)": pct(full_r[1], 1),
                     "(CPU or RAM)": pct(full_any), "FCFS jobs accepted": pct(fc["acc_jobs"]),
                     "FCFS CPU-hours served": pct(fc["acc_cpuh"]),
                     "largest job / C (CPU)": f"{amax[0] / C[0]:.2f}", "largest job / C (RAM)": f"{amax[1] / C[1]:.2f}"})
    table(pd.DataFrame(rows), "N1. f scan: full month, FCFS, no prices (C = f x D_bar from H)")
    f_sel = select_f(scan)
    out(f"\nSelection rule (fixed in advance): '% time demand >= 95% of C' (CPU or RAM) closest to the 25-50% band, "
        f"ties to 35%. -> f = {f_sel:.2f}. Configured f = {cfg.capacity.f}.")
    if select:
        print(f"SELECTED f = {f_sel}")
    C = cfg.capacity.f * cd["D_bar"]
    out(f"At the configured f: C = {C[0]:.4f} CPU, {C[1]:.4f} RAM machines; largest admitted request / C = "
        f"{amax[0] / C[0]:.2f} CPU, {amax[1] / C[1]:.2f} RAM (flag if > 0.2: {bool((amax / C > 0.2).any())}).")

    # ---- Part 6: distributions ----------------------------------------------------------
    cpu, ram = job.CPU.to_numpy(float), job.RAM.to_numpy(float)
    span = job.duration.to_numpy(float)
    eff = np.divide(job.CPU_hours_real.to_numpy(float), cpu, out=span.copy(), where=cpu > 0)
    d_cap = np.minimum(np.maximum(eff, cfg.time.duration_floor_hours), cfg.limits.d_cap_hours)
    quota = (cpu > lr.quota[0]) | (ram > lr.quota[1])
    w_all = cpu * eff
    kept = ~quota
    w_kept = (cpu * d_cap)[kept]
    def prow(name, x, wgt):
        return {"quantity": name, **{f"p{100 * q:g}" if q < 1 else "max": f"{np.quantile(x, q):.4g}" for q in QS},
                **{f"w p{100 * q:g}" if q < 1 else "w max": f"{wq(x, wgt, q):.4g}" for q in QS}}
    rows = [prow("CPU request (machines), all", cpu, w_all), prow("RAM request (machines), all", ram, w_all),
            prow("effective duration (h), all", eff, w_all), prow("CPU request x eff. duration, all", cpu * eff, w_all),
            prow("CPU request, after cap + quota", cpu[kept], w_kept), prow("RAM request, after cap + quota", ram[kept], w_kept),
            prow("duration after 24 h cap, after quota", d_cap[kept], w_kept),
            prow("CPU request x capped duration, after cap + quota", (cpu * d_cap)[kept], w_kept)]
    table(pd.DataFrame(rows), "N2. Percentiles: by jobs (p..) and weighted by CPU-hours (w p..)")
    trunc = eff > cfg.limits.d_cap_hours
    out(f"\nCap 24 h: {int(trunc.sum())} jobs truncated ({pct(trunc.mean(), 2)}); CPU-hours lost to truncation: "
        f"{pct((cpu * (eff - np.minimum(eff, 24.0))).sum() / w_all.sum())} of all CPU-hours. Quota: {int(quota.sum())} "
        f"jobs rejected ({pct(quota.mean(), 2)}), {pct(w_all[quota].sum() / w_all.sum())} of all CPU-hours. Work left "
        f"after cap + quota: {pct((cpu * np.minimum(eff, 24.0))[kept].sum() / w_all.sum())} of all CPU-hours.")
    assert quota.sum() == lr.n_quota_rejected
    fl = eff < cfg.time.duration_floor_hours
    out(f"1-minute duration floor (time.duration_floor_hours): {int(fl.sum())} jobs lifted ({pct(fl.mean())}); "
        f"CPU-hours added: {pct(((cfg.time.duration_floor_hours - eff[fl]) * cpu[fl]).sum() / w_all.sum(), 3)}.")

    # ---- figures ---------------------------------------------------------------------
    def cdf(ax, x, label, color, ls="-"):
        xs = np.sort(x[x > 0])
        ax.step(xs, np.arange(1, xs.size + 1) / xs.size, where="post", color=color, lw=1.6, ls=ls, label=label)
    def style(ax, xlabel):
        ax.set_xscale("log")
        ax.set_ylabel("share of jobs at or below x", color=INK)
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", color="#e5e4e0", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, x, qv, nm in ((axs[0], cpu, lr.quota[0], "CPU"), (axs[1], ram, lr.quota[1], "RAM")):
        cdf(ax, x, f"{nm} request, all {len(job):,} jobs", BLUE)
        ax.axvline(qv, color=INK2, ls="--", lw=1)
        ax.text(qv, 0.45, f" quota {qv:.2f}", fontsize=8, color=INK2)
        ax.set_title(f"{nm} request per job", loc="left", fontsize=10, color=INK)
        style(ax, f"{nm} request (machines, log scale)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig7_cdf_request.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    cdf(ax, eff, "effective duration, all jobs", BLUE)
    cdf(ax, d_cap[kept], "after the 24 h cap and the quota", ORANGE, ls="--")
    ax.axvline(24.0, color=INK2, ls=":", lw=1)
    ax.text(24.0, 0.45, " cap 24 h", fontsize=8, color=INK2)
    ax.set_title("Duration per job", loc="left", fontsize=10, color=INK)
    style(ax, "duration (hours, log scale)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig8_cdf_duration.png", dpi=150)
    plt.close(fig)
    tt = (np.concatenate([[0.0], np.cumsum(w)])[:-1]) / 86400
    fig, axs = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
    for i, (ax, nm) in enumerate(zip(axs, ("CPU", "RAM"))):
        ax.step(tt, lev[:, i], where="post", color=BLUE, lw=1.0)
        ax.axhline(C[i], color=INK2, ls="--", lw=1)
        ax.text(tt[-1] + 0.3, C[i], f"C = {C[i]:.3f}\n(f = {cfg.capacity.f:g})", va="center", fontsize=8, color=INK2)
        ax.set_ylabel(f"{nm} demand (machines)", color=INK)
        ax.set_title(f"{nm}: total request of running jobs, every job admitted (peak {lev[:, i].max():.2f})",
                     loc="left", fontsize=10, color=INK)
        ax.grid(axis="y", color="#e5e4e0", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[-1].set_xlabel(f"days since the first arrival ({pd.Timestamp(dts[0])} UTC)")
    axs[-1].set_xlim(0, tt[-1])
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    fig.savefig(FIGS / "fig9_demand_v3.png", dpi=150)
    plt.close(fig)
    (OUT / "new_environment_numbers.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main("--select" in sys.argv)

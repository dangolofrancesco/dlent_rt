"""
Week 32, step 2: checks of the new data (tasks 3, 4, 6). No model is run: only load_dataset and counting.

    dlent_venv/bin/python scripts/week32/data_v2.py

Reads data/batch_v2_30k.csv, data/raw/v2/{population.csv.gz, instances.csv.gz}, reports/week32/bq/p0.csv
(step 1: instances per collection over all of cell a, from instance_events) and configs/regime_v2.yaml.
Writes reports/week32/data_v2_numbers.md and reports/week32/figs/fig3_demand_v2.png.
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

from data_audit import BLUE, INK, INK2, demand, fcfs, pct, wquant   # noqa: E402
from dlent_rt.config import load_config                           # noqa: E402
from dlent_rt.data import _assign_real_tick_steps, load_dataset   # noqa: E402

OUT = ROOT / "reports" / "week32"
RAW = ROOT / "data" / "raw" / "v2"
D_MAX = (4.0, 12.0, 23.0)
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


def inst_stats(n: pd.Series) -> dict:
    return {"jobs": f"{len(n):,}", "with 1 instance": pct((n == 1).mean()), "mean instances": f"{n.mean():.2f}",
            "median": f"{n.median():.0f}", "with > 100": pct((n > 100).mean(), 2)}


def main() -> None:
    warnings.simplefilter("ignore")
    job = pd.read_csv(ROOT / "data" / "batch_v2_30k.csv")
    pop = pd.read_csv(RAW / "population.csv.gz")
    inst = pd.read_csv(RAW / "instances.csv.gz")
    out("# Week 32 step 2: numbers of the new data\n")
    out(f"Sample: {len(job)} jobs, {len(inst)} instances. Population (cell a, type 0, with usage): {len(pop):,} jobs.")

    # ---- Task 3: the sample against the cell ---------------------------------
    nb = pop[~pop.border]
    elig = nb[nb.n_inst_with_req > 0]
    p0 = pd.read_csv(OUT / "bq" / "p0.csv")
    rows = [{"set": "our new sample", **inst_stats(job.n_instances)},
            {"set": "its population (no border, with a request)", **inst_stats(elig.n_inst_ran)},
            {"set": "all jobs of cell a with usage (border included)", **inst_stats(pop.n_inst_ran)},
            {"set": "all collections of cell a, instance_events (step 1)", **inst_stats(p0.n_instances)},
            {"set": "old sample (batch_may2019_30k), instance_events", **inst_stats(
                p0[p0.collection_id.isin(pd.read_csv(ROOT / "data" / "batch_may2019_30k.csv").collection_id)].n_instances)}]
    table(pd.DataFrame(rows), "T1. Instances per job (instances that ran, unless said otherwise)")
    cpu_all, ram_all = pop.cpu_sum.sum(), pop.ram_sum.sum()
    out(f"\nBorder jobs in the population: {int(pop.border.sum()):,} ({pct(pop.border.mean(), 2)}), holding "
        f"{pct(pop.cpu_sum[pop.border].sum() / cpu_all)} of the summed CPU request. Non-border jobs with no request: "
        f"{int((nb.n_inst_with_req == 0).sum())}.")
    out(f"Share we hold: {pct(len(job) / len(pop), 3)} of the jobs with usage, {pct(len(job) / len(elig), 3)} of the "
        f"eligible jobs; {pct(job.CPU.sum() / cpu_all, 3)} of their summed CPU request "
        f"({pct(job.CPU.sum() / elig.cpu_sum.sum(), 3)} of the eligible jobs'), {pct(job.RAM.sum() / ram_all, 3)} of RAM.")
    assert set(job.collection_id) <= set(elig.collection_id)
    srt = elig.sort_values(["fp", "collection_id"]).collection_id.head(len(job))
    out(f"Check: the sample = the {len(job)} eligible jobs with the smallest fingerprint: {set(srt) == set(job.collection_id)}.")
    out(f"Request changed (an instance shows > 1 request value): {pct(job.request_changed.mean())} of jobs; "
        f"jobs with UPDATE events: {pct((job.n_update_events > 0).mean())}.")

    # ---- Task 4: SUM versus the real peak ------------------------------------
    r = (job.CPU / job.CPU_peak).where(job.CPU_peak > 0)
    multi = job.n_instances > 1
    vol_sum = (job.CPU * job.duration).sum()
    vol_peak = (job.CPU_peak * job.duration).sum()
    vol_real = job.CPU_hours_real.sum()
    out(f"\nT2a. Per job, SUM / real peak (CPU): equal (ratio 1) for {pct((r <= 1 + 1e-9).mean())} of jobs; "
        f"among multi-instance jobs ({int(multi.sum())}): median {r[multi].median():.2f}, q0.90 {r[multi].quantile(.9):.2f}, "
        f"q0.99 {r[multi].quantile(.99):.2f}, max {r.max():.1f}.")
    rows = [{"what": "total request (sum over jobs)", "SUM": f"{job.CPU.sum():.1f}", "real peak": f"{job.CPU_peak.sum():.1f}",
             "real (each instance its own span)": "-", "SUM / real": f"{job.CPU.sum() / job.CPU_peak.sum():.3f}"},
            {"what": "CPU-hours (request x job span)", "SUM": f"{vol_sum:,.0f}", "real peak": f"{vol_peak:,.0f}",
             "real (each instance its own span)": f"{vol_real:,.0f}", "SUM / real": f"{vol_sum / vol_real:.3f}"}]
    # concurrent demand over the month: model view (job SUM over its span) vs real (instances on their own spans)
    t0 = job.first_start_us.min()
    js, jd = (job.first_start_us - t0) / 3.6e9, job.duration.to_numpy()
    w1 = (job.first_start_us.max() - t0) / 3.6e9
    lv = {}
    for k, (s, d, A) in {"SUM": (js, jd, job[["CPU", "RAM"]].to_numpy()),
                         "peak": (js, jd, job[["CPU_peak", "RAM_peak"]].to_numpy()),
                         "real": ((inst.first_start - t0) / 3.6e9, (inst.last_end - inst.first_start) / 3.6e9,
                                  inst[["cpus", "memory"]].fillna(0).to_numpy())}.items():
        _, l, w = demand(np.asarray(s, float), np.asarray(d, float), np.asarray(A, float), 0.0, w1)
        lv[k] = ((l * w[:, None]).sum(0) / w.sum(), l.max(0), wquant(l, w, 0.75))
    for name, i in (("mean", 0), ("peak", 1), ("level exceeded 25%", 2)):
        rows.append({"what": f"concurrent CPU demand, {name}", "SUM": f"{lv['SUM'][i][0]:.2f}",
                     "real peak": f"{lv['peak'][i][0]:.2f}", "real (each instance its own span)": f"{lv['real'][i][0]:.2f}",
                     "SUM / real": f"{lv['SUM'][i][0] / lv['real'][i][0]:.3f}"})
    table(pd.DataFrame(rows), "T2b. SUM versus the real peak and the real instance spans (CPU, machines)")

    # ---- Task 6: d_max --------------------------------------------------------
    base = load_config(ROOT / "configs" / "regime_v2.yaml")
    em = base.energy_model
    rows_a, rows_b, rows_c, keep = [], [], [], {}
    for dmx in D_MAX:
        cfg = replace(base, time=replace(base.time, d_max_hours=dmx)).validate()
        ds = load_dataset(cfg)
        n0 = cfg.data.n0
        J = (ds.H, ds.stream)
        dts = np.concatenate([ds.H.datetime, ds.stream.datetime])
        steps = _assign_real_tick_steps(dts, dts[0], cfg.time.tick_seconds)
        arr = np.concatenate([ds.H.arrival_step, ds.stream.arrival_step + steps[n0]]).astype(float)
        assert np.array_equal(arr, steps)
        A = np.concatenate([j.A for j in J])
        dsteps = np.concatenate([j.duration_steps for j in J]).astype(float)
        dh = np.concatenate([j.duration_hours for j in J])
        raw = np.maximum(job.set_index("collection_id").loc[np.concatenate([j.collection_id for j in J]), "duration"]
                         .to_numpy(float), cfg.time.duration_floor_hours)
        assert np.allclose(dh, np.minimum(raw, ds.d_max_steps / 3600))       # one duration everywhere
        c = ds.capacity
        _, lev, w = demand(arr, dsteps, A, 0.0, arr.max())
        peak = lev.max(0)
        big = (A > c[None, :]).any(1)
        d_h = ds.d_max_steps / 3600
        rows_a.append({"d_max asked": f"{dmx:g} h", "d_max used": f"{d_h:.2f} h", "T": f"{ds.T / 3600:.1f} h",
                       "d_max / T": f"{ds.d_max_steps / ds.T:.4f}", "jobs cut": int((raw > d_h).sum()),
                       "CPU-hours kept": pct((A[:, 0] * dh).sum() / (A[:, 0] * raw).sum())})
        rows_b.append({"d_max": f"{dmx:g} h", "cluster CPU (machines)": f"{c[0]:.2f}", "cluster RAM (machines)": f"{c[1]:.2f}",
                       "CPU cores (assumed 64 / machine)": f"{c[0] * em.machine_cores:,.0f}",
                       "RAM GB (assumed 256 / machine)": f"{c[1] * em.machine_ram_gb:,.0f}",
                       "f CPU = c / peak": f"{c[0] / peak[0]:.3f}", "f RAM": f"{c[1] / peak[1]:.3f}",
                       "jobs larger than cluster": int(big.sum()),
                       "their CPU-hours": pct((A[big, 0] * dsteps[big]).sum() / (A[:, 0] * dsteps).sum())})
        for lbl, cc in (("25% rule", c), ("f = 1 (cluster = peak)", peak)):
            res = fcfs(arr, dsteps, A, cc, 0.0, arr.max())
            rows_c.append({"d_max": f"{dmx:g} h", "cluster": lbl, "jobs accepted": pct(res["acc_jobs"]),
                           "CPU-hours served": pct(res["acc_cpuh"]), "time >= 95% full": pct(res["full_time"])})
        keep[dmx] = (arr, lev, w, c, dts, ds)
    table(pd.DataFrame(rows_a), "T3. d_max: horizon, jobs cut, CPU-hours kept")
    table(pd.DataFrame(rows_b), "T4. Cluster by the 25% rule (machines; cores and GB under the declared 64 / 256)")
    table(pd.DataFrame(rows_c), "T5. First come, first served (admit a job if it fits now)")
    ds = keep[23.0][5]
    late = (np.concatenate([ds.H.datetime, ds.stream.datetime]) > ds.grid_data.timestamps.max() + np.timedelta64(1, "h")).sum()
    out(f"\nGrid file ends at {ds.grid_data.timestamps.max()} UTC; jobs arriving after its last hour: {int(late)} "
        f"(they get the last hour's price, energy.GridData.lookup clips the index).")
    out(f"xi on the data (max a_max / c) at d_max 23 h: {ds.xi:.2f} (forced xi_override = {base.constants.xi_override}).")

    # figure: demand over the month at d_max = 23 h, with the 25% level
    arr, lev, w, c, dts, ds23 = keep[23.0]
    dmax_used = ds23.d_max_steps / 3600
    tt = np.concatenate([[0.0], np.cumsum(w)])[:-1] / 86400
    fig, axs = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
    for i, (ax, res) in enumerate(zip(axs, ["CPU", "RAM"])):
        ax.step(tt, lev[:, i], where="post", color=BLUE, lw=1.0)
        ax.axhline(c[i], color=INK2, lw=1.0, ls="--")
        ax.text(tt[-1] + 0.3, c[i], f"exceeded 25% of time:\n{c[i]:,.1f} = cluster", va="center", fontsize=8, color=INK2)
        ax.set_ylabel(f"{res} demand (machines)", color=INK)
        ax.set_title(f"{res}: total request of running jobs, d_max = {dmax_used:.1f} h (peak {lev[:, i].max():,.1f})",
                     loc="left", fontsize=10, color=INK)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color="#e5e4e0", lw=0.6)
    axs[-1].set_xlabel(f"days since the first arrival ({pd.Timestamp(dts[0])} UTC)")
    axs[-1].set_xlim(0, tt[-1])
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    fig.savefig(OUT / "figs" / "fig3_demand_v2.png", dpi=150)
    plt.close(fig)
    (OUT / "data_v2_numbers.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

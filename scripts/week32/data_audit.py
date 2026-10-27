"""
Week 32 data audit, step 1 (diagnosis only: no model is run, no config is changed).

    dlent_venv/bin/python scripts/week32/data_audit.py

Reads data/batch_may2019_30k.csv (29,670 jobs) and, when present, the BigQuery outputs of
scripts/week32/bq_checks.py (reports/week32/bq/*.csv). Writes:
    reports/week32/audit_numbers.md     every number of reports/week32/data_audit.md, as printed tables
    reports/week32/figs/*.png           figures

Units: a request is in "machines": 1.0 = the largest machine of the trace (Google rescales CPU and RAM by the
largest machine capacity). A job's request = SUM of resource_request over its instances (scripts/data/fetch_data.py).
Demand at time t = sum of the requests of the jobs running at t (job j runs in [request_time_j, + duration_j)).
All time statistics are time-weighted over the window W = [first arrival, last arrival] (31 days).
"Level exceeded X% of the time" = the time-weighted (1 - X) quantile of the demand.
"""
from __future__ import annotations

import heapq
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

from capacity_regime import committed                                   # noqa: E402
from dlent_rt.data import (                                             # noqa: E402
    _assign_real_tick_steps, _OccupancyBasis, _size_capacity, load_dataset,
)
from dlent_rt.preprocess import fit_hardware_catalogue, handle_outliers  # noqa: E402

OUT = ROOT / "reports" / "week32"
FIGS = OUT / "figs"
BQ = OUT / "bq"
D_MAX = (4.0, 24.0, 72.0, None)          # hours; None = no cap
X_LEVELS = (0.10, 0.25, 0.50)            # cluster = demand level exceeded X of the time
FULL = 0.95                              # "cluster at least 95% full"
REGIME_C = np.array([0.4014237441083392, 0.2616208305922613])  # configs/regime_small_cluster.yaml (machines)
INK, INK2, BLUE, ORANGE = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"

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


# --------------------------------------------------------------------------- #
# Demand over time and first-come-first-served
# --------------------------------------------------------------------------- #
def demand(start, dur, A, w0, w1):
    """Step function of the summed request of running jobs, clipped to [w0, w1]: (times, levels, weights)."""
    s = np.clip(start, w0, w1)
    e = np.clip(start + dur, w0, w1)
    t = np.concatenate([s, e, [w0, w1]])
    d = np.concatenate([A, -A, np.zeros((2, A.shape[1]))])
    o = np.argsort(t, kind="stable")
    ut, inv = np.unique(t[o], return_inverse=True)
    agg = np.zeros((len(ut), A.shape[1]))
    np.add.at(agg, inv, d[o])
    lev = np.cumsum(agg, axis=0)[:-1]
    return ut, lev, np.diff(ut)


def wquant(lev, w, q):
    outv = []
    for i in range(lev.shape[1]):
        o = np.argsort(lev[:, i], kind="stable")
        cw = np.cumsum(w[o]) / w.sum()
        outv.append(lev[o, i][min(np.searchsorted(cw, q), len(o) - 1)])
    return np.array(outv)


def stats(lev, w):
    r = dict(mean=(lev * w[:, None]).sum(0) / w.sum(), peak=lev.max(0))
    for x in X_LEVELS:
        r[f"exc{int(x * 100)}"] = wquant(lev, w, 1 - x)
    return r


def fcfs(start, dur, A, c, w0, w1):
    """Admit each job, in arrival order, if it fits now on every resource (no queue, no reward)."""
    order = np.argsort(start, kind="stable")
    occ = np.zeros(A.shape[1])
    heap: list = []
    acc = np.zeros(len(start), bool)
    tol = 1e-12
    for j in order:
        while heap and heap[0][0] <= start[j]:
            _, k = heapq.heappop(heap)
            occ -= A[k]
        if np.all(occ + A[j] <= c + tol):
            acc[j] = True
            occ += A[j]
            heapq.heappush(heap, (start[j] + dur[j], j))
    _, lev, w = demand(start[acc], dur[acc], A[acc], w0, w1)
    full = (lev >= FULL * c[None, :] - tol).any(axis=1)
    return dict(acc_jobs=acc.mean(), acc_cpuh=(A[acc, 0] * dur[acc]).sum() / (A[:, 0] * dur).sum(),
                full_time=(w * full).sum() / w.sum())


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def main() -> None:
    warnings.simplefilter("ignore")
    FIGS.mkdir(parents=True, exist_ok=True)
    cfg = committed()
    df = pd.read_csv(ROOT / "data" / "batch_may2019_30k.csv", parse_dates=["request_time"])
    df = df.sort_values("request_time", kind="stable").reset_index(drop=True)
    n = len(df)
    t0 = df.request_time.iloc[0]
    start = (df.request_time - t0).dt.total_seconds().to_numpy() / 3600
    dur = df.duration.to_numpy(float)
    A = df[["CPU", "RAM"]].to_numpy(float)
    w0, w1 = 0.0, start.max()
    vol = A * dur[:, None]
    out("# Week 32 data audit: all numbers\n")
    out(f"Jobs: {n}. Window W: {t0} to {df.request_time.iloc[-1]} = {w1:.1f} h. Units: machines.")

    # ---- Task 1: outlier filters ------------------------------------------------
    o = cfg.outliers
    a_thr = np.array([df.CPU.quantile(o.a_max_quantile), df.RAM.quantile(o.a_max_quantile)])
    d_thr = float(df.duration.quantile(o.d_max_quantile))
    over_cpu, over_ram = A[:, 0] > a_thr[0], A[:, 1] > a_thr[1]
    over_res, over_dur = over_cpu | over_ram, dur > d_thr
    rows = []
    for name, m in [("CPU request > q0.90", over_cpu), ("RAM request > q0.90", over_ram),
                    ("either request > q0.90 (resource filter)", over_res),
                    (f"duration > q0.95 = {d_thr:.2f} h (duration filter)", over_dur),
                    ("removed by both filters together", over_res | over_dur)]:
        rows.append({"filter": name, "jobs removed": int(m.sum()), "share of jobs": pct(m.mean()),
                     "share of CPU-hours": pct(vol[m, 0].sum() / vol[:, 0].sum(), 2),
                     "share of RAM-hours": pct(vol[m, 1].sum() / vol[:, 1].sum(), 2)})
    out(f"\nThresholds: CPU q{o.a_max_quantile} = {a_thr[0]:.4f}, RAM q{o.a_max_quantile} = {a_thr[1]:.4f} machines, "
        f"duration q{o.d_max_quantile} = {d_thr:.4f} h. policy = {o.policy}.")
    table(pd.DataFrame(rows), "T1a. What the outlier filters remove")
    kept, rep = handle_outliers(df, cfg, ["CPU", "RAM"], "duration")
    assert rep.n_after == int((~(over_res | over_dur)).sum()) and np.allclose(rep.a_max_thresholds, a_thr)
    out(f"\nCheck: handle_outliers keeps {rep.n_after} jobs ({pct(rep.n_after / n)}), same as above.")

    # snapping
    rows = []
    for label, sub in [("after filters (current model)", ~(over_res | over_dur)), ("all jobs", np.ones(n, bool))]:
        Ar = A[sub]
        cat = fit_hardware_catalogue(Ar, cfg)
        As = cat.centroids[cat.assign(Ar)]
        rel = np.abs(As - Ar) / np.clip(Ar, 1e-12, None)
        rows.append({"jobs": f"{label}, n = {sub.sum()}", "moved > 5% (CPU or RAM)": pct((rel > 0.05).any(1).mean()),
                     "total CPU request": f"{(As[:, 0].sum() / Ar[:, 0].sum() - 1) * 100:+.1f}%",
                     "total RAM request": f"{(As[:, 1].sum() / Ar[:, 1].sum() - 1) * 100:+.1f}%",
                     "CPU-hours": f"{((As[:, 0] * dur[sub]).sum() / (Ar[:, 0] * dur[sub]).sum() - 1) * 100:+.1f}%",
                     "RAM-hours": f"{((As[:, 1] * dur[sub]).sum() / (Ar[:, 1] * dur[sub]).sum() - 1) * 100:+.1f}%"})
    table(pd.DataFrame(rows), "T1b. Hardware snapping (32 profiles): change of the requests")

    # ---- Task 2: raw demand ------------------------------------------------------
    times, lev, w = demand(start, dur, A, w0, w1)
    s_raw = stats(lev, w)
    def srow(lbl, s):
        r = {"case": lbl}
        for i, res in enumerate(["CPU", "RAM"]):
            r.update({f"{res} mean": f"{s['mean'][i]:.1f}", f"{res} peak": f"{s['peak'][i]:.1f}",
                      **{f"{res} >{int(x*100)}%": f"{s[f'exc{int(x*100)}'][i]:.1f}" for x in X_LEVELS}})
        return r
    table(pd.DataFrame([srow("all jobs, real durations", s_raw)]),
          "T2. Demand of running jobs (machines). '>X%' = level exceeded X% of the time")

    # filtered model arrays, for the "2.77" check and the current regime
    ds = load_dataset(replace(cfg, capacity=replace(cfg.capacity, mode="absolute",
                              absolute=replace(cfg.capacity.absolute, cpu=float(REGIME_C[0]), ram=float(REGIME_C[1])))))
    steps = _assign_real_tick_steps(kept.request_time.to_numpy(), kept.request_time.to_numpy()[0], 1.0)
    n0 = cfg.data.n0
    arr = np.concatenate([ds.H.arrival_step, ds.stream.arrival_step + steps[n0]]).astype(float)
    dsteps = np.concatenate([ds.H.duration_steps, ds.stream.duration_steps]).astype(float)
    Am = np.concatenate([ds.H.A, ds.stream.A])
    assert np.array_equal(arr, steps)
    basis = _OccupancyBasis(arrival_step=arr.astype(np.int64), duration_steps=dsteps.astype(np.int64), A=Am)
    cq = replace(cfg, capacity=replace(cfg.capacity, mode="fraction_of_concurrency_quantile",
                                       concurrency_quantile=0.7, fraction=1.0))
    c_q07 = _size_capacity(basis, cq)
    assert np.allclose(0.15 * c_q07, REGIME_C), (0.15 * c_q07, REGIME_C)
    sm, dm = arr / 3600, dsteps / 3600
    _, levm, wm = demand(sm, dm, Am, 0.0, sm.max())
    s_mod = stats(levm, wm)
    _, levr, wr = demand(sm, dm, kept[["CPU", "RAM"]].to_numpy(float), 0.0, sm.max())
    table(pd.DataFrame([srow("after filters, raw requests", stats(levr, wr)),
                        srow("after filters, snapped (what the model sees)", s_mod)]),
          "T2b. The same after the outlier filters (machines)")
    out(f"\nd_max of the model after filters: {ds.d_max_steps / 3600:.2f} h; T = {ds.T / 3600:.1f} h.")

    fig, axs = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
    for i, (ax, res) in enumerate(zip(axs, ["CPU", "RAM"])):
        ax.step(times[:-1] / 24, lev[:, i], where="post", color=BLUE, lw=1.0)
        marks = sorted([(s_raw[key][i], lbl, ls) for key, lbl, ls in
                        [("mean", "mean", "--"), ("exc10", "exceeded 10% of time", ":"),
                         ("exc25", "exceeded 25% of time", ":"), ("exc50", "exceeded 50% of time", ":")]])
        gap, y_prev = 0.05 * (lev[:, i].max() - lev[:, i].min()), -np.inf
        for v, lbl, ls in marks:
            ax.axhline(v, color=INK2, lw=1.0, ls=ls)
            y_prev = max(v, y_prev + gap)
            ax.text(w1 / 24 + 0.3, y_prev, f"{lbl}: {v:,.0f}", va="center", fontsize=8, color=INK2)
        ax.set_ylabel(f"{res} demand (machines)", color=INK)
        ax.set_title(f"{res}: total request of running jobs (peak {s_raw['peak'][i]:,.0f})", loc="left",
                     fontsize=10, color=INK)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color="#e5e4e0", lw=0.6)
    axs[-1].set_xlabel("day of May 2019 (trace time)")
    axs[-1].set_xlim(0, w1 / 24)
    fig.tight_layout(rect=(0, 0, 0.80, 1))
    fig.savefig(FIGS / "fig1_demand_all_jobs.png", dpi=150)
    plt.close(fig)

    # ---- Task 3: duration cap ----------------------------------------------------
    T_all = None
    st_all = _assign_real_tick_steps(df.request_time.to_numpy(), df.request_time.to_numpy()[0], 1.0)
    T_all = (st_all[-1] - st_all[n0]) + 1           # stream horizon the code would use on all jobs (seconds)
    out(f"\nT (stream horizon of the code, all jobs, n0 = {n0}) = {T_all / 3600:.1f} h. Code ceiling "
        f"d_max <= {cfg.time.d_max_horizon_fraction} T = {cfg.time.d_max_horizon_fraction * T_all / 3600:.1f} h; "
        f"abort if d_max/T > {cfg.time.d_max_ratio_abort}.")
    rows3, caps = [], {}
    for dmx in D_MAX:
        dc = dur if dmx is None else np.minimum(dur, dmx)
        _, lv, ww = demand(start, dc, A, w0, w1)
        s = stats(lv, ww)
        caps[dmx] = (dc, s, lv, ww)
        lbl = "no cap" if dmx is None else f"{dmx:g} h"
        ratio = (dur.max() if dmx is None else dmx) * 3600 / T_all
        rows3.append({"d_max": lbl, "d_max / T": f"{ratio:.3f}",
                      "CPU-hours kept": pct((A[:, 0] * dc).sum() / vol[:, 0].sum()),
                      "jobs cut": int((dur > (np.inf if dmx is None else dmx)).sum()),
                      **{k: v for k, v in srow(lbl, s).items() if k != "case"}})
    table(pd.DataFrame(rows3), "T3. Duration cap: CPU-hours kept and demand (machines)")

    fig, axs = plt.subplots(len(D_MAX), 1, figsize=(10, 8), sharex=True)
    for ax, dmx in zip(axs, D_MAX):
        _, s, lv, ww = caps[dmx]
        tt = np.concatenate([[w0], np.cumsum(ww)])[:-1]
        ax.step(tt / 24, lv[:, 0], where="post", color=BLUE, lw=1.0)
        ax.axhline(s["mean"][0], color=INK2, lw=1.0, ls="--")
        ax.set_title(f"CPU demand, d_max = {'no cap' if dmx is None else f'{dmx:g} h'} "
                     f"(mean {s['mean'][0]:,.0f}, peak {s['peak'][0]:,.0f} machines; dashed = mean)",
                     loc="left", fontsize=9, color=INK)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color="#e5e4e0", lw=0.6)
    axs[-1].set_xlabel("day of May 2019 (trace time)")
    axs[-1].set_xlim(0, w1 / 24)
    fig.supylabel("CPU demand (machines)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_demand_by_duration_cap.png", dpi=150)
    plt.close(fig)

    # ---- Task 4: new cluster rule + FCFS -----------------------------------------
    rows4 = []
    for dmx in D_MAX:
        dc, s, _, _ = caps[dmx]
        for x in (*X_LEVELS, 0.0):
            c = s["peak"] if x == 0.0 else s[f"exc{int(x * 100)}"]
            big = (A > c[None, :]).any(1)
            r = fcfs(start, dc, A, c, w0, w1)
            rows4.append({"d_max": "no cap" if dmx is None else f"{dmx:g} h",
                          "X": "peak (f = 1)" if x == 0.0 else f"{int(x * 100)}%",
                          "cluster CPU": f"{c[0]:.1f}", "cluster RAM": f"{c[1]:.1f}",
                          "f CPU": f"{c[0] / s['peak'][0]:.3f}", "f RAM": f"{c[1] / s['peak'][1]:.3f}",
                          "jobs larger than cluster": int(big.sum()),
                          "their CPU-hours": pct((A[big, 0] * dc[big]).sum() / (A[:, 0] * dc).sum()),
                          "FCFS jobs accepted": pct(r["acc_jobs"]), "FCFS CPU-hours served": pct(r["acc_cpuh"]),
                          "time >= 95% full": pct(r["full_time"])})
    table(pd.DataFrame(rows4), "T4. New rule: cluster = demand level exceeded X% of the time; FCFS result")

    rows5 = []
    for lbl, c in [("current rule at f = 1 (c = q0.70 level after filters)", c_q07),
                   ("current regime, f = 0.15 (regime_small_cluster.yaml)", REGIME_C)]:
        big = (Am > c[None, :]).any(1)
        r = fcfs(sm, dm, Am, c, 0.0, sm.max())
        rows5.append({"case": lbl, "cluster CPU": f"{c[0]:.3f}", "cluster RAM": f"{c[1]:.3f}",
                      "f CPU (c / peak)": f"{c[0] / s_mod['peak'][0]:.3f}",
                      "f RAM (c / peak)": f"{c[1] / s_mod['peak'][1]:.3f}",
                      "jobs larger than cluster": int(big.sum()),
                      "their CPU-hours": pct((Am[big, 0] * dm[big]).sum() / (Am[:, 0] * dm).sum()),
                      "FCFS jobs accepted": pct(r["acc_jobs"]), "FCFS CPU-hours served": pct(r["acc_cpuh"]),
                      "time >= 95% full": pct(r["full_time"])})
    table(pd.DataFrame(rows5), f"T5. Current regime, same FCFS (filtered jobs, snapped requests; n = {len(Am)})")

    # ---- Task 5: BigQuery outputs ------------------------------------------------
    if (BQ / "a.csv").is_file():
        a = pd.read_csv(BQ / "a.csv").merge(df[["collection_id", "CPU", "RAM"]], on="collection_id", how="right")
        rows = []
        for lbl, m in [("job (collection_type 0)", a.collection_type == 0),
                       ("alloc set (collection_type 1)", a.collection_type == 1),
                       ("job that runs inside an alloc set", (a.collection_type == 0) & (a.alloc_collection_id > 0)),
                       ("not found in collection_events", a.collection_type.isna())]:
            rows.append({"kind": lbl, "collections": int(m.sum()), "share": pct(m.mean()),
                         "share of requested CPU": pct(a.CPU[m].sum() / a.CPU.sum()),
                         "share of requested RAM": pct(a.RAM[m].sum() / a.RAM.sum())})
        table(pd.DataFrame(rows), "T6. collection_type of the 29,670 collections")
    if (BQ / "b0.csv").is_file():
        b = pd.read_csv(BQ / "b0.csv").merge(df[["collection_id", "CPU"]], on="collection_id", how="right")
        q = b.n_instances.quantile([0.5, 0.75, 0.9, 0.99]).to_numpy()
        bins = [0, 1, 10, 100, 1000, np.inf]
        lab = ["1", "2-10", "11-100", "101-1000", "> 1000"]
        cut = pd.cut(b.n_instances, bins=bins, labels=lab)
        g = b.groupby(cut, observed=False).agg(jobs=("collection_id", "size"), cpu=("CPU", "sum"))
        table(pd.DataFrame({"instances per job": lab, "jobs": g.jobs.to_numpy(),
                            "share of jobs": [pct(v) for v in g.jobs / len(b)],
                            "share of requested CPU": [pct(v) for v in g.cpu / b.CPU.sum()]}),
              "T7. Number of instances per job")
        out(f"\nInstances per job: median {q[0]:.0f}, q0.75 {q[1]:.0f}, q0.90 {q[2]:.0f}, q0.99 {q[3]:.0f}, "
            f"max {b.n_instances.max()}. Event rows per instance: median "
            f"{(b.n_event_rows / b.n_instances).median():.1f}.")
        # sampling bias of fetch step 1: a job enters if ANY of its event rows passes RAND() < 0.05
        p_in = 1 - 0.95 ** b.n_event_rows
        wt = 1 / p_in
        out(f"Step-1 sampling: inclusion chance = 1 - 0.95^(event rows); in our sample it is "
            f"{pct(p_in.median())} for the median job, {pct((p_in < 0.2).mean())} of jobs have a chance < 20%. "
            f"Mean request per job: {b.CPU.mean():.3f} (sample) vs {(wt * b.CPU).sum() / wt.sum():.3f} "
            f"(re-weighted by 1 / inclusion chance) machines CPU.")
        # task 6: old query = SUM over all event rows
        old = pd.read_csv(ROOT / "data" / "batch_may2019_300k.csv", usecols=["collection_id", "A_cpu", "A_ram"])
        m = b.merge(old, on="collection_id").merge(df[["collection_id", "RAM"]], on="collection_id")
        m = m[(m.CPU > 0)]
        ratio = m.A_cpu / m.CPU
        epi = m.n_event_rows / m.n_instances
        out(f"\nT8 (task 6). Jobs in both files: {len(m)}. Ratio 300k / 30k request (CPU): q0.10 "
            f"{ratio.quantile(.1):.2f}, median {ratio.median():.2f}, q0.90 {ratio.quantile(.9):.2f}. "
            f"Event rows per instance on the same jobs: q0.10 {epi.quantile(.1):.2f}, median {epi.median():.2f}, "
            f"q0.90 {epi.quantile(.9):.2f}. Spearman(ratio, rows per instance) = "
            f"{ratio.rank().corr(epi.rank()):.3f}; share with ratio within 10% of rows per instance: "
            f"{pct((np.abs(ratio / epi - 1) < 0.10).mean())}; ratio <= rows per instance: "
            f"{pct((ratio <= epi * 1.001).mean())}.")

    # ---- Task 8: fetch step audit -------------------------------------------------
    ev = pd.read_csv(ROOT / "data" / "raw" / "raw_events_30k.csv")
    us = pd.read_csv(ROOT / "data" / "raw" / "raw_usage_30k.csv")
    out(f"\nT9 (8b). raw_events_30k: {len(ev)} collections; raw_usage_30k: {len(us)}; lost in the inner merge: "
        f"{len(set(ev.collection_id) - set(us.collection_id))} (all have no usage row).")
    if (BQ / "lost.csv").is_file():
        lo = pd.read_csv(BQ / "lost.csv")
        names = {0: "SUBMIT", 1: "QUEUE", 2: "ENABLE", 3: "SCHEDULE", 4: "EVICT", 5: "FAIL", 6: "FINISH", 7: "KILL",
                 8: "LOST", 9: "UPDATE_PENDING", 10: "UPDATE_RUNNING"}
        last = lo.last_type.map(names).value_counts().to_dict()
        out(f"Lost: {len(lo)}; alloc sets {int((lo.collection_type == 1).sum())}; never scheduled "
            f"{int((lo.ever_scheduled == 0).sum())}; last collection event: {last}.")
    if (BQ / "pop.csv").is_file() and (BQ / "p0.csv").is_file():
        pop = pd.read_csv(BQ / "pop.csv", usecols=["collection_id", "collection_type"])
        p0 = pd.read_csv(BQ / "p0.csv")
        fair = p0.sample(n=len(df), random_state=42)
        ours = p0[p0.collection_id.isin(df.collection_id)]
        rows = []
        for lbl, x in [("all of cell a", p0), (f"fair random sample ({len(df)} ids)", fair), ("our sample", ours)]:
            ni = x.n_instances
            rows.append({"set": lbl, "collections": len(x), "median instances": f"{ni.median():.0f}",
                         "mean instances": f"{ni.mean():.1f}", "share with 1 instance": pct((ni == 1).mean()),
                         "share with > 100 instances": pct((ni > 100).mean())})
        table(pd.DataFrame(rows), "T10 (8a). Instances per job: our sample vs a fair sample")
        out(f"\nCell a: {len(pop)} collections in collection_events ({int((pop.collection_type == 0).sum())} jobs, "
            f"{int((pop.collection_type == 1).sum())} alloc sets); {len(p0)} collections in instance_events. "
            f"Our sample = {pct(len(df) / len(p0), 2)} of the collections and "
            f"{pct(ours.n_instances.sum() / p0.n_instances.sum(), 1)} of the instances.")
    pr = ev.priority
    out(f"\nT11 (8f). Raw priorities in 351-359 (code: tier 5, doc: production): {int(((pr > 350) & (pr < 360)).sum())}; "
        f">= 360: {int((pr >= 360).sum())}.")
    x = ev.merge(us, on="collection_id").merge(df[["collection_id", "CPU", "RAM", "duration"]], on="collection_id")
    s0, s1 = 600e6, 600e6 + 31 * 86400e6
    v_c, v_r = x.CPU * x.duration, x.RAM * x.duration
    rows = []
    for lbl, m in [("started before the trace (start < 600 s)", x.start_time < s0),
                   ("still running at the end (last window ends < 5 min before the end)", x.end_time >= s1 - 300e6),
                   ("either", (x.start_time < s0) | (x.end_time >= s1 - 300e6))]:
        rows.append({"jobs": lbl, "count": int(m.sum()), "share of jobs": pct(m.mean(), 2),
                     "share of CPU-hours": pct(v_c[m].sum() / v_c.sum()), "share of RAM-hours": pct(v_r[m].sum() / v_r.sum())})
    table(pd.DataFrame(rows), "T12 (8e). Jobs cut at the borders of the trace")
    r = x.actual_cpu_hours / (x.CPU * x.duration)
    out(f"\nT13 (8g). actual_cpu_hours / (request x duration): median {r.median():.2f}; share > 1: {pct((r > 1).mean())}.")

    (OUT / "audit_numbers.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

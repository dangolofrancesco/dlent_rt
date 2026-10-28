"""
Week 32: the uniform 300k sample and configs/v4_300k.yaml. Data checks (pass / fail), the quota fixed point, the
choice of H, the f scan (FCFS, no prices). No online model is run.

    dlent_venv/bin/python scripts/week32/v4_300k.py

Writes reports/week32/v4_300k_numbers.md and reports/week32/figs/fig10_cdf_30k_300k.png, fig11_demand_v4.png.
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

from limits import BLUE, INK, INK2, ORANGE, demand, fcfs, lines, out, pct, table, timeline   # noqa: E402
from new_environment import select_f                                                          # noqa: E402
from dlent_rt.config import load_config                                                       # noqa: E402
from dlent_rt.data import load_dataset, mean_demand                                           # noqa: E402

OUT = ROOT / "reports" / "week32"
FIGS = OUT / "figs"
CFG = ROOT / "configs" / "v4_300k.yaml"
F_SCAN = (0.10, 0.25, 0.50, 0.75, 1.00)
FULL = 0.95
START_US, END_US, WIN_US = 600 * 10**6, 600 * 10**6 + 31 * 86400 * 10**6, 300 * 10**6
QS = (0.50, 0.75, 0.90, 0.95, 0.99, 0.999, 1.0)


def eff_dur(j):
    cpu, span = j.CPU.to_numpy(float), j.duration.to_numpy(float)
    return np.divide(j.CPU_hours_real.to_numpy(float), cpu, out=span.copy(), where=cpu > 0)


def verdict(ok):
    return "PASS" if ok else "FAIL"


def main() -> None:
    warnings.simplefilter("ignore")
    base = load_config(CFG)
    big = pd.read_csv(ROOT / "data" / "batch_v2_300k.csv")
    small = pd.read_csv(ROOT / "data" / "batch_v2_30k.csv")
    pop = pd.read_csv(ROOT / "data" / "raw" / "v2_300k" / "population.csv.gz")
    elig = pop[~pop.border & (pop.n_inst_with_req > 0)]
    out("# Week 32: v4 (uniform 300k sample), all numbers\n")
    out(f"300k file: {len(big):,} jobs; 30k file: {len(small):,}; eligible population (cell a, jobs that ran, no border, "
        f"with a request): {len(elig):,}; sample rate {len(big) / len(elig):.4%}.")

    # ---- Part 3: checks -------------------------------------------------------------------
    checks = []
    s1, p1 = (big.n_instances == 1).mean(), (elig.n_inst_ran == 1).mean()
    checks.append(("single-instance share vs the eligible cell (within 1 point)", f"{pct(s1)} vs {pct(p1)}",
                   verdict(abs(s1 - p1) < 0.01)))
    mix_b = big.priority_tier.value_counts(normalize=True).sort_index()
    mix_s = small.priority_tier.value_counts(normalize=True).reindex(mix_b.index, fill_value=0)
    checks.append(("priority-tier mix vs the 30k sample (max gap < 1 point; the cell's priorities are not fetched)",
                   ", ".join(f"tier {int(k)} {pct(mix_b[k])} / {pct(mix_s[k])}" for k in mix_b.index),
                   verdict((mix_b - mix_s).abs().max() < 0.01)))
    hb = np.floor((big.first_start_us - START_US) / 3.6e9).astype(int).value_counts().sort_index()
    hp = np.floor((elig.first_start - START_US) / 3.6e9).astype(int).value_counts().sort_index()
    hrs = hp.index.union(hb.index)
    hb, hp = hb.reindex(hrs, fill_value=0), hp.reindex(hrs, fill_value=0) * len(big) / len(elig)
    corr = np.corrcoef(hb, hp)[0, 1]
    checks.append(("arrivals per hour vs the cell x the sample rate (correlation > 0.95)",
                   f"corr {corr:.4f}; mean {hb.mean():.1f} vs {hp.mean():.1f} per hour", verdict(corr > 0.95)))
    m = small.merge(big, on="collection_id", how="left", suffixes=("", "_b"), indicator=True)
    found = (m["_merge"] == "both").sum()
    cols = [c for c in small.columns if c != "collection_id"]
    exact = [c for c in cols if m[c].dtype.kind not in "f"]
    flt = [c for c in cols if m[c].dtype.kind == "f"]
    same_exact = all((m[c] == m[f"{c}_b"]).all() for c in exact)
    rel = max(float(np.nanmax(np.abs(m[c] - m[f"{c}_b"]) / np.maximum(np.abs(m[c]), 1e-300))) for c in flt)
    checks.append(("the 30k jobs inside the 300k, identical rows (integers / text exact; floats to 1e-12 relative: "
                   "the instance sums run in a different order)",
                   f"{found:,} of {len(small):,} found; exact columns equal: {same_exact}; largest float gap {rel:.1e}",
                   verdict(found == len(small) and same_exact and rel < 1e-12)))
    eq = (big.CPU_peak >= big.CPU * (1 - 1e-9)) & (big.RAM_peak >= big.RAM * (1 - 1e-9))
    checks.append(("SUM of first requests = real peak for >= 99% of jobs", pct(eq.mean(), 2), verdict(eq.mean() >= 0.99)))
    e = eff_dur(big)
    ok_e = np.isfinite(e).all() and (e >= 0).all() and (e <= big.duration * (1 + 1e-9)).all() \
        and not big[["CPU", "RAM", "duration", "CPU_hours_real"]].isna().any().any()
    checks.append(("effective duration = real CPU-hours / request: finite, >= 0, <= span; no NaN", f"min {e.min():.2e} h",
                   verdict(ok_e)))
    first = pd.to_datetime(big.arrival_utc).min()
    checks.append(("time base UTC: first arrival at or after 2019-05-01 07:00 UTC", str(first),
                   verdict(first >= pd.Timestamp("2019-05-01 07:00", tz="UTC") and
                           first < pd.Timestamp("2019-05-01 07:10", tz="UTC"))))
    nb = (big.first_start_us >= START_US).all() and (big.last_end_us <= END_US - WIN_US).all()
    checks.append(("no border jobs", f"first start >= 600 s and last end <= end - 300 s for all: {nb}", verdict(nb)))
    # distributions: is each 30k percentile (p50..p99) inside the 95% band of the same percentile over 200 random
    # 30,000-job subsamples of the 300k? (a fixed % threshold is too strict at p99, where 30k has ~300 jobs)
    rng = np.random.default_rng(42)
    subs = [rng.choice(len(big), size=len(small), replace=False) for _ in range(200)]
    rows, n_out = [], 0
    for name, xb, xs in (("CPU request", big.CPU, small.CPU), ("RAM request", big.RAM, small.RAM),
                         ("effective duration", e, eff_dur(small)), ("CPU request x eff. duration", big.CPU * e,
                                                                       small.CPU * eff_dur(small))):
        xb, xs = np.asarray(xb, float), np.asarray(xs, float)
        qb, qs = np.quantile(xb, QS), np.quantile(xs, QS)
        boot = np.array([np.quantile(xb[i], QS[:5]) for i in subs])
        lo, hi = np.quantile(boot, 0.025, axis=0), np.quantile(boot, 0.975, axis=0)
        inside = (qs[:5] >= lo) & (qs[:5] <= hi)
        n_out += int((~inside).sum())
        rows.append({"quantity": name, **{f"p{100 * q:g}" if q < 1 else "max": f"{b:.4g} / {s:.4g}"
                                          for q, b, s in zip(QS, qb, qs)},
                     "30k inside the 95% band (p50-p99)": f"{int(inside.sum())} of 5"})
    checks.append(("distributions close to the 30k: each 30k percentile p50-p99 inside the 95% band of 30k-sized "
                   "subsamples of the 300k", f"{20 - n_out} of 20 inside", verdict(n_out <= 1)))
    table(pd.DataFrame(checks, columns=["check", "value", "result"]), "N1. 300k data checks")
    table(pd.DataFrame(rows), "N2. Percentiles, 300k / 30k (p99.9 and max are noisy on 30k)")

    # ---- Part 4 + 5: the quota fixed point and the choice of H (at f = 1.0) -----------------------
    def H_cfg(h_days, n0=None):
        d = replace(base.data, h_days=h_days, **({} if n0 is None else {"n0": n0}))
        return replace(base, data=d, capacity=replace(base.capacity, f=1.0)).validate()
    choices = {"first 10% of jobs": H_cfg(None, int(round(0.10 * len(big)))), "first 7 days": H_cfg(7.0)}
    rows, its_7 = [], None
    for name, cfg in choices.items():
        ds = load_dataset(cfg)
        arr, dst, A, dh, dts = timeline(ds, cfg)
        t_h = (dts - dts[0]) / np.timedelta64(1, "h")
        span_m = (arr[-1] - arr[0] + 1) / 3600
        r = {"H": name, "jobs in H": ds.H.n, "H span (h)": f"{ds.capacity_diag['span_hours']:.1f}",
             "Q CPU / RAM": f"{ds.limit_report.quota[0]:.4f} / {ds.limit_report.quota[1]:.4f}",
             "D_bar of H, CPU / RAM": f"{ds.capacity_diag['D_bar'][0]:.3f} / {ds.capacity_diag['D_bar'][1]:.3f}"}
        wk = []
        for k in range(5):
            sel = (t_h >= 168 * k) & (t_h < 168 * (k + 1))
            span_k = min(168.0, t_h.max() - 168 * k)
            wk.append(mean_demand(A[sel], dh[sel], span_k)["D_bar"])
            r[f"week {k + 1} CPU"] = f"{wk[-1][0]:.3f}"
        month = mean_demand(A, dh, span_m)["D_bar"]
        r["month CPU / RAM"] = f"{month[0]:.3f} / {month[1]:.3f}"
        r["H vs month (CPU)"] = f"{100 * (ds.capacity_diag['D_bar'][0] / month[0] - 1):+.1f}%"
        r["H vs month (RAM)"] = f"{100 * (ds.capacity_diag['D_bar'][1] / month[1] - 1):+.1f}%"
        rows.append(r)
        if name == "first 7 days":
            its_7 = ds.limit_report.quota_iterations
    table(pd.DataFrame(rows), "N3. Choice of H (f = 1.0; weeks = 7-day blocks from the first arrival, week 5 = last 2-3 days; "
                              "D_bar of a week / the month = its work / its length, same quota)")
    gaps = {r["H"]: abs(float(r["H vs month (CPU)"].rstrip("%"))) + abs(float(r["H vs month (RAM)"].rstrip("%")))
            for r in rows}
    out(f"\nH closest to the month (sum of the CPU and RAM gaps): {min(gaps, key=gaps.get)} ({gaps}).")
    table(pd.DataFrame([{"iteration": i["iteration"], "Q CPU": f"{i['Q'][0]:.4f}", "Q RAM": f"{i['Q'][1]:.4f}",
                         "jobs kept": f"{i['n_kept']:,}", "jobs in H": f"{i['n_H']:,}",
                         "D_bar CPU": f"{i['D_bar'][0]:.4f}", "D_bar RAM": f"{i['D_bar'][1]:.4f}",
                         "next Q CPU": f"{i['Q_next'][0]:.4f}", "next Q RAM": f"{i['Q_next'][1]:.4f}",
                         "change CPU / RAM": f"{pct(i['change'][0])} / {pct(i['change'][1])}"} for i in its_7]),
          "N4. Quota fixed point Q = 0.2 x C (H = 7 days, f = 1.0)")

    # ---- Part 6: f scan (H = 7 days, the config) ---------------------------------------------------
    total_jobs = len(big)
    d_all = np.minimum(np.maximum(e, base.time.duration_floor_hours), base.limits.d_cap_hours)
    total_cpuh = (big.CPU.to_numpy() * d_all).sum()
    rows, scan, keep = [], [], {}
    for f in F_SCAN:
        cfg = replace(base, capacity=replace(base.capacity, f=f)).validate()
        try:
            ds = load_dataset(cfg)
        except ValueError as err:
            if "collapsed" not in str(err):
                raise
            rows.append({"f": f"{f:.2f}", "C CPU": "collapsed: Q -> 0 (no positive fixed point)"})
            scan.append(dict(f=f, full=np.inf))
            continue
        arr, dst, A, dh, dts = timeline(ds, cfg)
        C = ds.capacity
        _, lev, w = demand(arr, dst, A, arr[0], arr[-1])
        full_any = (w * (lev >= FULL * C[None, :]).any(1)).sum() / w.sum()
        fc = fcfs(arr, dst, A, C, arr[0], arr[-1])
        amax = A.max(0)
        scan.append(dict(f=f, full=full_any))
        keep[f] = (arr, lev, w, C, dts, ds)
        rows.append({"f": f"{f:.2f}", "C CPU": f"{C[0]:.3f}", "C RAM": f"{C[1]:.3f}",
                     "Q CPU": f"{ds.limit_report.quota[0]:.4f}", "Q RAM": f"{ds.limit_report.quota[1]:.4f}",
                     "iterations": len(ds.limit_report.quota_iterations),
                     "% time demand >= 95% C (CPU or RAM)": pct(full_any),
                     "FCFS jobs accepted": pct(fc["acc_jobs"]), "FCFS CPU-hours served": pct(fc["acc_cpuh"]),
                     "largest admitted / C CPU": f"{amax[0] / C[0]:.3f}", "largest admitted / C RAM": f"{amax[1] / C[1]:.3f}",
                     "quota keeps: jobs": pct(len(A) / total_jobs), "quota keeps: CPU-hours": pct((A[:, 0] * dh).sum() / total_cpuh)})
    table(pd.DataFrame(rows), "N5. f scan (H = 7 days; full month, FCFS, no prices; capped durations)")
    f_sel = select_f(scan)
    out(f"\nSelection rule (fixed in advance): '% time demand >= 95% of C' (CPU or RAM) closest to 25-50%, ties to 35%. "
        f"-> f = {f_sel:.2f}. Configured f = {base.capacity.f}.")
    arr, lev, w, C, dts, ds = keep[f_sel]
    out(f"At f = {f_sel:.2f}: T = {ds.T / 3600:.1f} h, H = {ds.H.n:,} jobs, stream = {ds.stream.n:,}, d_max = "
        f"{ds.d_max_steps / 3600:g} h; quota rejected {ds.limit_report.n_quota_rejected:,} jobs.")

    # ---- figures ----------------------------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (nm, xb, xs, xl) in zip(axs, (("CPU request", big.CPU, small.CPU, "machines"),
                                          ("RAM request", big.RAM, small.RAM, "machines"),
                                          ("effective duration", e, eff_dur(small), "hours"))):
        for x, lab, col, ls in ((xs, "30k", ORANGE, "--"), (xb, "300k", BLUE, "-")):
            v = np.sort(np.asarray(x)[np.asarray(x) > 0])
            ax.step(v, np.arange(1, v.size + 1) / v.size, where="post", color=col, ls=ls, lw=1.6, label=lab)
        ax.set_xscale("log")
        ax.set_title(nm, loc="left", fontsize=10, color=INK)
        ax.set_xlabel(f"{xl} (log scale)")
        ax.grid(axis="y", color="#e5e4e0", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    axs[0].set_ylabel("share of jobs at or below x", color=INK)
    fig.tight_layout()
    fig.savefig(FIGS / "fig10_cdf_30k_300k.png", dpi=150)
    plt.close(fig)
    tt = np.concatenate([[0.0], np.cumsum(w)])[:-1] / 86400
    fig, axs = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
    for i, (ax, nm) in enumerate(zip(axs, ("CPU", "RAM"))):
        ax.step(tt, lev[:, i], where="post", color=BLUE, lw=0.8)
        ax.axhline(C[i], color=INK2, ls="--", lw=1)
        ax.text(tt[-1] + 0.3, C[i], f"C = {C[i]:.2f}\n(f = {f_sel:g})", va="center", fontsize=8, color=INK2)
        ax.set_ylabel(f"{nm} demand (machines)", color=INK)
        ax.set_title(f"{nm}: total request of running jobs, every job admitted (300k; peak {lev[:, i].max():.1f})",
                     loc="left", fontsize=10, color=INK)
        ax.grid(axis="y", color="#e5e4e0", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[-1].set_xlabel(f"days since the first arrival ({pd.Timestamp(dts[0])} UTC)")
    axs[-1].set_xlim(0, tt[-1])
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    fig.savefig(FIGS / "fig11_demand_v4.png", dpi=150)
    plt.close(fig)
    (OUT / "v4_300k_numbers.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

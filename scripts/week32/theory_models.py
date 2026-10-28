"""
Week 32: job distributions and the type / average models over f on configs/v5_theory.yaml (no online model, no
job-level benchmark). The logic is in dlent_rt/theory_eval.py; this script writes the numbers and the figures.

    dlent_venv/bin/python scripts/week32/theory_models.py

Writes reports/week32/theory_models_numbers.md and reports/week32/figs/fig12_..fig20_*.png.
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

from limits import lines, out, pct, table                                            # noqa: E402
from dlent_rt import theory_eval as te                                               # noqa: E402
from dlent_rt.config import load_config                                              # noqa: E402

OUT = ROOT / "reports" / "week32"
FIGS = OUT / "figs"
F_SCAN = (0.10, 0.20, 0.30, 0.50, 0.70, 1.25)


def g(x, d=4):
    return f"{x:.{d}g}"


def main() -> None:
    warnings.simplefilter("ignore")
    cfg = load_config(ROOT / "configs" / "v5_theory.yaml")
    out("# Week 32: theoretical models over f (configs/v5_theory.yaml), all numbers\n")

    # ---- Part 1: job distributions ---------------------------------------------------------------
    J = te.job_distributions(cfg)
    d = J["dist"].copy()
    for c in d.columns[4:]:
        d[c] = d[c].map(lambda x: g(x))
    table(d, "N1. Distributions: count, mean, percentiles by jobs (p..) and weighted by CPU-hours (w p..)")
    imp = J["impact"].copy()
    for c in ("share of jobs cut", "CPU-hours cut by the cap", "share of jobs removed", "CPU-hours removed by the quota",
              "CPU-hours left after both"):
        imp[c] = imp[c].map(lambda x: pct(x, 2))
    imp["CPU-hours (effective)"] = imp["CPU-hours (effective)"].map(lambda x: f"{x:,.0f}")
    table(imp, f"N2. What the 24 h cap and the quota (Q = {J['Q'][0]:.4f} CPU, {J['Q'][1]:.4f} RAM) do")
    for which, name in (("cpu", "fig12_cdf_cpu"), ("ram", "fig13_cdf_ram"), ("duration", "fig14_cdf_duration")):
        fig, ax = plt.subplots(figsize=(8, 4.2))
        te.fig_cdf_jobs_work(J, which, ax)
        fig.tight_layout()
        fig.savefig(FIGS / f"{name}.png", dpi=150)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    h = te.fig_hist2d(J, ax)
    fig.colorbar(h[3], ax=ax, label="jobs per cell")
    fig.tight_layout()
    fig.savefig(FIGS / "fig15_hist2d_cpu_duration.png", dpi=150)
    plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    te.fig_H_vs_month(J, axs)
    fig.tight_layout()
    fig.savefig(FIGS / "fig16_H_vs_month.png", dpi=150)
    plt.close(fig)

    # ---- Part 3: models over f ------------------------------------------------------------------------
    R = [te.evaluate_f(cfg, f, seed=cfg.run.seed) for f in F_SCAN]
    em = cfg.energy_model
    out(f"\nSame jobs for every f: quota {np.round(R[0]['quota'], 4).tolist()} machines (fixed point at f = "
        f"{cfg.limits.quota_from_f}); H {R[0]['n_H']:,} jobs; stream {R[0]['n_stream']:,} jobs; T = {R[0]['T_h']:.1f} h.")
    rows = []
    for r in R:
        rows.append({"f": r["f"], "C CPU / RAM (machines)": f"{r['C'][0]:.3f} / {r['C'][1]:.3f}",
                     "C cores / GB (64 / 256 per machine)": f"{r['C_cores']:,.0f} / {r['C_gb']:,.0f}",
                     "stream load / C (CPU, RAM)": f"{r['static']['stream_load_over_C'][0]:.2f}, {r['static']['stream_load_over_C'][1]:.2f}",
                     "jobs > C: stream / H": f"{r['oversize']['stream']} / {r['oversize']['H']}",
                     "their CPU-hours (stream)": pct(r["oversize"]["cpu_hours_share"], 2)})
    table(pd.DataFrame(rows), "N3. Cluster size by f")
    rows = []
    for r in R:
        for key, lab in (("phase0_lp", "Phase 0 LP (H)"), ("fluid", "fluid OPT (stream, hindsight)"),
                         ("fluid_feasible", "fluid on feasible types")):
            m = r[key]
            rows.append({"f": r["f"], "LP": lab, "tau CPU": g(m["tau"][0]), "tau RAM": g(m["tau"][1]),
                         "types fully / partly / rejected": f"{m['counts']['fully']:,} / {m['counts']['partly']:,} / {m['counts']['rejected']:,}",
                         "LP fill CPU / RAM": f"{pct(m['fill'][0])} / {pct(m['fill'][1])}"})
    table(pd.DataFrame(rows), "N4. LPs by f: tau, types admitted, how full the LP keeps the cluster")
    rows = []
    for r in R:
        for key, lab in te.MODELS:
            sp = r["phase0_lp"]["expected"] if key == "phase0_expected" else r[key]["split"]
            rows.append({"f": r["f"], "model": lab, "reward $": f"{sp['reward']:,.1f}",
                         "customer value $": f"{sp['customer value']:,.1f}", "profit $": f"{sp['profit']:,.1f}",
                         "carbon $": f"{sp['carbon']:,.1f}"})
        b = r["bucketed"]
        rows.append({"f": r["f"], "model": "bucketed MILP (reference buckets)",
                     "reward $": (f"{b['split']['reward']:,.1f}" if b["run"] else
                                  f"skipped: {b['n_candidates']:,} jobs, {b['nonzeros']:,} nonzeros")})
    table(pd.DataFrame(rows).fillna(""), "N5. Reward and split per model, over the stream (T as above)")
    rows = []
    for r in R:
        fl, ff, st = r["fluid"]["split"]["reward"], r["fluid_feasible"]["split"]["reward"], r["static"]["split"]["reward"]
        b = r["bucketed"]
        rows.append({"f": r["f"], "fluid OPT - fluid feasible $": f"{fl - ff:,.2f}",
                     "discretization cost (fluid feasible - bucketed) $": (f"{ff - b['split']['reward']:,.1f}" if b["run"] else "n/a (MILP skipped)"),
                     "gap static Phase 0 to fluid OPT $": f"{fl - st:,.1f}", "static / fluid OPT": pct(st / fl)})
    table(pd.DataFrame(rows), "N6. Gaps between models")
    rows = []
    for r in R:
        s = r["static"]
        rows.append({"f": r["f"], "% time >= 95% full (CPU or RAM)": pct(s["full_time"]),
                     "mean occupancy / C (CPU, RAM)": f"{pct(s['mean_occupancy_over_C'][0])}, {pct(s['mean_occupancy_over_C'][1])}",
                     "jobs admitted": f"{s['n_admitted']:,}", "% jobs rejected for lack of space": pct(s["rejected_space"], 2),
                     "realised reward $": f"{s['split']['reward']:,.1f}",
                     "stream jobs in types unseen in H": pct(s["unseen_type_share"], 2),
                     "their CPU-hours": pct(s["unseen_type_cpu_hours"])})
    table(pd.DataFrame(rows), "N7. (a) Static Phase 0 policy on the real stream (fraction thinning, seed 42, real gate)")
    rows = []
    for r in R:
        for h, taus in r["windows"].items():
            row = {"f": r["f"], "window": f"{h} h", "windows": len(taus)}
            for i, nm in ((0, "CPU"), (1, "RAM")):
                t = taus[:, i]
                row[f"% windows tau > 0 ({nm})"] = pct((t > 0).mean())
                row[f"tau p50 / p90 / max ({nm})"] = f"{g(np.quantile(t, .5), 3)} / {g(np.quantile(t, .9), 3)} / {g(t.max(), 3)}"
            rows.append(row)
    table(pd.DataFrame(rows), "N8. (b) Fluid LP per window (the window's own type rates)")
    f_sel = te.select_f(R)
    rs = next(r for r in R if r["f"] == f_sel)
    out(f"\nSelection rule (fixed in advance, report only): '% time >= 95% full' of (a) closest to 20-30%, ties to 25% "
        f"-> f = {f_sel:g} ({pct(rs['static']['full_time'])}). Not set in any config.")

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    te.fig_tau_vs_f(R, axs)
    fig.tight_layout()
    fig.savefig(FIGS / "fig17_tau_vs_f.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    te.fig_reward_vs_f(R, ax)
    fig.tight_layout()
    fig.savefig(FIGS / "fig18_reward_vs_f.png", dpi=150)
    plt.close(fig)
    fig, axs = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True)
    te.fig_occupancy(rs, axs, cfg.time.tick_seconds)
    fig.tight_layout()
    fig.savefig(FIGS / "fig19_occupancy_selected.png", dpi=150)
    plt.close(fig)
    fig, axs = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True)
    te.fig_tau_windows(rs, axs, 24)
    fig.tight_layout()
    fig.savefig(FIGS / "fig20_tau_per_day_selected.png", dpi=150)
    plt.close(fig)
    (OUT / "theory_models_numbers.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

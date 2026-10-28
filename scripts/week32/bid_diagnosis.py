"""
Week 32 bid-size diagnosis on configs/v4_300k.yaml (diagnosis only: no recalibration, no config change, no online run).

    dlent_venv/bin/python scripts/week32/bid_diagnosis.py

Per-job reward terms with Phase 0's bid model and frozen s_C (reward.job_terms), H and stream; deciles of
CPU request x duration (r*d, capped duration); two worked examples; the volume of positive / negative types in the
LP; an extended f scan with the fixed selection rule. Writes reports/week32/bid_diagnosis_numbers.md.
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "week32"))

from limits import demand, fcfs, lines, out, pct, table, timeline        # noqa: E402
from new_environment import select_f                                     # noqa: E402
from dlent_rt.config import load_config                                  # noqa: E402
from dlent_rt.data import load_dataset                                   # noqa: E402
from dlent_rt.models.core.stream import build_stream                     # noqa: E402
from dlent_rt.phase0 import (_compute_job_energy, _lookup_grid, _synthesize_bids, run_phase0,   # noqa: E402
                             type_arrays)
from dlent_rt.reward import carbon_kg, electricity_cost, job_reward, job_terms                 # noqa: E402

OUT = ROOT / "reports" / "week32"
CFG = ROOT / "configs" / "v4_300k.yaml"
F_SCAN = (0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 4.00, 5.00)
FULL = 0.95


def main() -> None:
    warnings.simplefilter("ignore")
    cfg = load_config(CFG)
    ds = load_dataset(cfg)
    st = run_phase0(ds, cfg)
    lam = cfg.objective
    out("# Week 32: bid-size diagnosis on v4, all numbers\n")
    out(f"Config configs/v4_300k.yaml (f = {cfg.capacity.f}); s_C = {st.carbon_price:.2f} $/kg; jobs H {ds.H.n:,}, "
        f"stream {ds.stream.n:,}; u0 = {cfg.bid.base_utility_u0}, mode {cfg.bid.base_utility_mode}, gamma1 = "
        f"{cfg.bid.gamma1}, gamma2 = {cfg.bid.gamma2}, sigma = {cfg.bid.sigma}, machine {cfg.energy_model.machine_cores} "
        f"cores / {cfg.energy_model.machine_ram_gb} GB.")

    # ---- per-job terms, H (Phase 0 bids) + stream (online bids), Phase 0's model and s_C ------------
    H = ds.H
    bH = _synthesize_bids(H, cfg, seed=cfg.run.seed)
    EH = _compute_job_energy(H, cfg)
    pr, ci = _lookup_grid(H, ds)
    tH = job_terms(H.priority, bH.v_rate, bH.cost_base, electricity_cost(EH, pr), carbon_kg(EH, ci),
                   st.myerson, st.carbon_price, lam)
    S = build_stream(ds, cfg)
    tS = job_reward(S, st, lam)
    A = np.concatenate([H.A, S.A])
    d = np.concatenate([H.duration_hours, S.duration_hours])
    q = np.concatenate([H.priority, S.priority])
    v = np.concatenate([bH.v_rate, S.v])
    cost = np.concatenate([bH.cost_base, S.cost_base])
    T = {k: np.concatenate([getattr(tH, k), getattr(tS, k)]) for k in ("cv", "phi", "profit", "carbon_usd", "carbon_kg", "R")}
    E = np.concatenate([EH, S.energy_kwh])
    rd = A[:, 0] * d
    edges = np.quantile(rd, np.linspace(0, 1, 11))
    dec = np.clip(np.searchsorted(edges, rd, side="right") - 1, 0, 9)
    rows = []
    for k in range(10):
        m = dec == k
        rows.append({"decile": k + 1, "r*d range (machine-h)": f"{rd[m].min():.2g} - {rd[m].max():.2g}",
                     "jobs": int(m.sum()), "mean CPU": f"{A[m, 0].mean():.4f}", "mean d (h)": f"{d[m].mean():.3f}",
                     "mean bid $": f"{v[m].mean():.3f}", "mean customer value $": f"{T['cv'][m].mean():.3f}",
                     "mean profit $": f"{T['profit'][m].mean():.4f}", "mean carbon $": f"{T['carbon_usd'][m].mean():.3f}",
                     "mean reward $": f"{T['R'][m].mean():.3f}",
                     "reward per machine-h $": f"{T['R'][m].sum() / rd[m].sum():.4g}",
                     "% positive reward": pct((T["R"][m] > 0).mean()), "share of CPU-hours": pct(rd[m].sum() / rd.sum(), 2)})
    table(pd.DataFrame(rows), "N1. Per-job reward terms by decile of CPU request x capped duration (H + stream, v4)")
    out(f"\nAll jobs: positive reward {pct((T['R'] > 0).mean())}; CPU-hours of jobs with positive reward "
        f"{pct(rd[T['R'] > 0].sum() / rd.sum())}. Mean bid / cost base (markup) {np.mean(v / cost):.4f}; "
        f"share of the cost base from the base utility (median over jobs) "
        f"{np.median(1 - (cfg.bid.gamma1 * A[:, 0] * cfg.energy_model.machine_cores + cfg.bid.gamma2 * A[:, 1] * cfg.energy_model.machine_ram_gb) * d / cost):.4f}.")

    # ---- worked examples ------------------------------------------------------------------------------
    i_med = int(np.argsort(rd)[len(rd) // 2])
    Q = ds.limit_report.quota
    i_q = int(np.argmin(np.abs(A[:, 0] - Q[0])))
    em, bd = cfg.energy_model, cfg.bid
    ex = []
    for name, i in (("median job (by r*d)", i_med), (f"job nearest the CPU quota Q = {Q[0]:.3f}", i_q)):
        base_rate = bd.base_utility_u0 * (1 + np.log(q[i]))
        size_rate = bd.gamma1 * A[i, 0] * em.machine_cores + bd.gamma2 * A[i, 1] * em.machine_ram_gb
        ex.append({"job": name, "CPU (machines)": f"{A[i, 0]:.4f}", "RAM (machines)": f"{A[i, 1]:.4f}",
                   "priority tier": int(q[i]), "duration (h)": f"{d[i]:.4f}", "r*d (machine-h)": f"{rd[i]:.4g}",
                   "base utility $/h u0(1+ln q)": f"{base_rate:.3f}", "size term $/h g1 cores + g2 GB": f"{size_rate:.4f}",
                   "cost base $ (rate x d)": f"{cost[i]:.4f}", "bid $": f"{v[i]:.4f}", "phi $": f"{T['phi'][i]:.4f}",
                   "customer value $ (q x bid)": f"{T['cv'][i]:.4f}", "energy kWh": f"{E[i]:.4g}",
                   "carbon kg": f"{T['carbon_kg'][i]:.4g}", "carbon $ (s_C x kg)": f"{T['carbon_usd'][i]:.4f}",
                   "profit $ (phi - elec)": f"{T['profit'][i]:.4f}", "reward $": f"{T['R'][i]:.4f}"})
    exdf = pd.DataFrame(ex).set_index("job").T.reset_index().rename(columns={"index": "term"})
    table(exdf, "N2. Two worked examples")

    # ---- why tau = 0: volume of positive / negative types ------------------------------------------
    pos = st.r_tilde > 0
    vol = st.p[:, None] * st.v
    tot = vol.sum(0)
    _, prio_t, _ = type_arrays(st.type_space, st.grid)
    rd_t = st.A_type[:, 0] * st.d_bar / 3600.0
    popm = st.p > 0
    out(f"\nN3. LP load p*v / C by reward sign (types with jobs in H): positive types {np.round(vol[pos].sum(0) / ds.capacity, 4).tolist()} "
        f"(CPU, RAM), negative types {np.round(vol[~pos & popm].sum(0) / ds.capacity, 4).tolist()}; all {np.round(tot / ds.capacity, 4).tolist()}. "
        f"Mass p: positive {pct(st.p[pos].sum() / st.p.sum())}. Mean type r*d (machine-h, p-weighted): positive "
        f"{np.average(rd_t[pos & popm], weights=st.p[pos & popm]):.4g}, negative "
        f"{np.average(rd_t[~pos & popm], weights=st.p[~pos & popm]):.4g}. tau = {np.asarray(st.tau).tolist()}.")

    # ---- extended f scan -----------------------------------------------------------------------------
    job = pd.read_csv(ROOT / cfg.data.batch_csv)
    cpu, span = job.CPU.to_numpy(float), job.duration.to_numpy(float)
    eff = np.divide(job.CPU_hours_real.to_numpy(float), cpu, out=span.copy(), where=cpu > 0)
    d_all = np.minimum(np.maximum(eff, cfg.time.duration_floor_hours), cfg.limits.d_cap_hours)
    total_cpuh = (cpu * d_all).sum()
    rows, scan = [], []
    for f in F_SCAN:
        c = replace(cfg, capacity=replace(cfg.capacity, f=f)).validate()
        try:
            dsf = load_dataset(c)
        except ValueError as err:
            if "collapsed" not in str(err):
                raise
            rows.append({"f": f"{f:.2f}", "C CPU": "collapsed"})
            scan.append(dict(f=f, full=np.inf))
            continue
        arr, dst, Af, dh, dts = timeline(dsf, c)
        C = dsf.capacity
        _, lev, w = demand(arr, dst, Af, arr[0], arr[-1])
        full_any = (w * (lev >= FULL * C[None, :]).any(1)).sum() / w.sum()
        fc = fcfs(arr, dst, Af, C, arr[0], arr[-1])
        amax = Af.max(0)
        scan.append(dict(f=f, full=full_any))
        rows.append({"f": f"{f:.2f}", "C CPU": f"{C[0]:.3f}", "C RAM": f"{C[1]:.3f}",
                     "Q CPU": f"{dsf.limit_report.quota[0]:.4f}", "Q RAM": f"{dsf.limit_report.quota[1]:.4f}",
                     "iterations": len(dsf.limit_report.quota_iterations),
                     "% time demand >= 95% C (CPU or RAM)": pct(full_any),
                     "FCFS jobs accepted": pct(fc["acc_jobs"]), "FCFS CPU-hours served": pct(fc["acc_cpuh"]),
                     "largest admitted / C CPU": f"{amax[0] / C[0]:.3f}", "largest admitted / C RAM": f"{amax[1] / C[1]:.3f}",
                     "jobs kept by the quota": pct(len(Af) / len(job)),
                     "CPU-hours kept by the quota": pct((Af[:, 0] * dh).sum() / total_cpuh)})
    table(pd.DataFrame(rows), "N4. Extended f scan (v4: H = 7 days, quota = 0.2 x C; full month, FCFS, no prices)")
    f_sel = select_f(scan)
    share = next(s["full"] for s in scan if s["f"] == f_sel)
    out(f"\nFixed rule ('% time demand >= 95% of C' closest to 25-50%, ties to 35%) -> f = {f_sel:.2f} with {pct(share)}. "
        f"Not set in the config (configured f = {cfg.capacity.f}).")
    (OUT / "bid_diagnosis_numbers.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

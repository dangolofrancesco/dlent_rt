"""
Week 32: the size-based bid (configs/v5_size_bid.yaml) against v4. Phase 0 only (no online run, no recalibration).

    dlent_venv/bin/python scripts/week32/v5_size_bid.py

Phase 0 summaries of v4 and v5; the scale check (gamma1, gamma2 x 0.5 and x 2 on v5); per-job reward terms by
decile of r*d and two worked examples on v5; bid vs electricity cost per machine-hour; LP load by reward sign; the
quota share 0.2 vs 0.5 at f = 1.25. Writes reports/week32/v5_size_bid_numbers.md.
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

from limits import demand, lines, out, pct, table, timeline                                    # noqa: E402
from dlent_rt.config import load_config                                                         # noqa: E402
from dlent_rt.data import load_dataset                                                          # noqa: E402
from dlent_rt.diagnostics import phase0_summary                                                 # noqa: E402
from dlent_rt.models.core.stream import build_stream                                            # noqa: E402
from dlent_rt.phase0 import _compute_job_energy, _lookup_grid, _synthesize_bids, run_phase0     # noqa: E402
from dlent_rt.reward import carbon_kg, electricity_cost, job_reward, job_terms                 # noqa: E402

OUT = ROOT / "reports" / "week32"
V4, V5 = ROOT / "configs" / "v4_300k.yaml", ROOT / "configs" / "v5_size_bid.yaml"
FULL = 0.95


def per_job(ds, st, cfg):
    """Reward terms of every job (H with Phase 0's bids, stream with the online bids), Phase 0's model and s_C."""
    H, lam = ds.H, cfg.objective
    bH = _synthesize_bids(H, cfg, seed=cfg.run.seed)
    EH = _compute_job_energy(H, cfg)
    pr, ci = _lookup_grid(H, ds)
    tH = job_terms(H.priority, bH.v_rate, bH.cost_base, electricity_cost(EH, pr), carbon_kg(EH, ci),
                   st.myerson, st.carbon_price, lam)
    S = build_stream(ds, cfg)
    tS = job_reward(S, st, lam)
    J = {k: np.concatenate([getattr(tH, k), getattr(tS, k)]) for k in ("cv", "phi", "profit", "carbon_usd", "carbon_kg", "R")}
    J.update(A=np.concatenate([H.A, S.A]), d=np.concatenate([H.duration_hours, S.duration_hours]),
             q=np.concatenate([H.priority, S.priority]), v=np.concatenate([bH.v_rate, S.v]),
             cost=np.concatenate([bH.cost_base, S.cost_base]), E=np.concatenate([EH, S.energy_kwh]),
             elec=np.concatenate([electricity_cost(EH, pr), S.C_elec]))
    J["rd"] = J["A"][:, 0] * J["d"]
    return J


def summary_row(name, s):
    sa, sm = s["reward_split_per_h_all"], s["reward_split_per_h_admitted"]
    return {"config": name, "s_C $/kg": f"{s['s_C']:,.4g}", "types with jobs in H": f"{s['n_types_populated']:,}",
            "tau CPU, RAM": str([round(x, 6) for x in s["tau"]]),
            "types admitted fully / partly": f"{s['n_types_fully_admitted']:,} / {s['n_types_admitted'] - s['n_types_fully_admitted']:,}",
            "LP fill CPU / RAM": f"{pct(s['lp_fill'][0])} / {pct(s['lp_fill'][1])}",
            "positive types / arrival mass": f"{pct(s['share_positive_reward_types'])} / {pct(s['share_positive_reward_mass'])}",
            "split all CV / profit / carbon $/h": f"{sa['customer value']:,.0f} / {sa['profit']:,.0f} / {sa['carbon']:,.0f}",
            "split admitted $/h": f"{sm['customer value']:,.0f} / {sm['profit']:,.0f} / {sm['carbon']:,.0f}",
            "C CPU / RAM": f"{s['capacity'][0]:.3f} / {s['capacity'][1]:.3f}"}


def load_by_sign(ds, st):
    pos, popm = st.r_tilde > 0, st.p > 0
    vol = st.p[:, None] * st.v
    rd_t = st.A_type[:, 0] * st.d_bar / 3600.0
    return (f"LP load p*v / C: positive types {np.round(vol[pos].sum(0) / ds.capacity, 4).tolist()} (CPU, RAM), negative "
            f"types {np.round(vol[~pos & popm].sum(0) / ds.capacity, 4).tolist()}; arrival mass positive "
            f"{pct(st.p[pos].sum() / st.p.sum())}; mean type r*d (machine-h, p-weighted) positive "
            f"{np.average(rd_t[pos & popm], weights=st.p[pos & popm]):.4g}, negative "
            f"{np.average(rd_t[~pos & popm], weights=st.p[~pos & popm]):.4g}; tau {np.asarray(st.tau).tolist()}.")


def main() -> None:
    warnings.simplefilter("ignore")
    out("# Week 32: size-based bid (v5) vs v4, all numbers\n")
    runs = {}
    for name, path in (("v4 (base-utility bid, f 1.00)", V4), ("v5 (size bid, f 1.25)", V5)):
        cfg = load_config(path)
        ds = load_dataset(cfg)
        st = run_phase0(ds, cfg)
        runs[name] = (cfg, ds, st, phase0_summary(ds, st, cfg))
    table(pd.DataFrame([summary_row(k, v[3]) for k, v in runs.items()]), "N1. Phase 0, v4 vs v5")
    cfg5, ds5, st5, s5 = runs["v5 (size bid, f 1.25)"]
    out(f"\nv5 quota (fixed point): Q = {np.round(ds5.limit_report.quota, 4).tolist()}; jobs H {ds5.H.n:,}, stream "
        f"{ds5.stream.n:,}; flags: {s5['flags']}.")

    # ---- scale check: gamma1, gamma2 x k on v5 -------------------------------------------------
    rows = []
    for k in (0.5, 1.0, 2.0):
        c = replace(cfg5, bid=replace(cfg5.bid, gamma1=cfg5.bid.gamma1 * k, gamma2=cfg5.bid.gamma2 * k)).validate()
        st = run_phase0(ds5, c)
        s = phase0_summary(ds5, st, c)
        rows.append({"gamma factor": k, "positive types": pct(s["share_positive_reward_types"], 2),
                     "positive arrival mass": pct(s["share_positive_reward_mass"], 2), "s_C $/kg": f"{s['s_C']:.6g}",
                     "s_C / s_C(1)": None, "types with jobs in H": s["n_types_populated"],
                     "tau": str([round(x, 6) for x in s["tau"]]), "_s": s["s_C"]})
    s1 = rows[1]["_s"]
    for r in rows:
        r["s_C / s_C(1)"] = f"{r.pop('_s') / s1:.6f}"
    table(pd.DataFrame(rows), "N2. Scale check on v5: gamma1 and gamma2 both x k (same data, same random numbers)")

    # ---- deciles and worked examples on v5 -------------------------------------------------------
    J = per_job(ds5, st5, cfg5)
    rd = J["rd"]
    edges = np.quantile(rd, np.linspace(0, 1, 11))
    dec = np.clip(np.searchsorted(edges, rd, side="right") - 1, 0, 9)
    rows = []
    for k in range(10):
        m = dec == k
        rows.append({"decile": k + 1, "mean CPU": f"{J['A'][m, 0].mean():.4f}", "mean d (h)": f"{J['d'][m].mean():.3f}",
                     "mean bid $": f"{J['v'][m].mean():.4g}", "mean customer value $": f"{J['cv'][m].mean():.4g}",
                     "mean profit $": f"{J['profit'][m].mean():.4g}", "mean carbon $": f"{J['carbon_usd'][m].mean():.4g}",
                     "mean reward $": f"{J['R'][m].mean():.4g}", "reward per machine-h $": f"{J['R'][m].sum() / rd[m].sum():.4g}",
                     "% positive reward": pct((J["R"][m] > 0).mean()), "share of CPU-hours": pct(rd[m].sum() / rd.sum(), 2)})
    table(pd.DataFrame(rows), "N3. v5: per-job reward terms by decile of CPU request x capped duration (H + stream)")
    out(f"\nv5 all jobs: positive reward {pct((J['R'] > 0).mean())}; CPU-hours of jobs with positive reward "
        f"{pct(rd[J['R'] > 0].sum() / rd.sum())}; markup bid / cost base {np.mean(J['v'] / J['cost']):.4f}.")
    i_med = int(np.argsort(rd)[len(rd) // 2])
    Q = ds5.limit_report.quota
    i_q = int(np.argmin(np.abs(J["A"][:, 0] - Q[0])))
    em, bd = cfg5.energy_model, cfg5.bid
    ex = []
    for name, i in (("median job (by r*d)", i_med), (f"job nearest the CPU quota Q = {Q[0]:.3f}", i_q)):
        size_rate = bd.gamma1 * J["A"][i, 0] * em.machine_cores + bd.gamma2 * J["A"][i, 1] * em.machine_ram_gb
        ex.append({"job": name, "CPU (machines)": f"{J['A'][i, 0]:.4f}", "RAM (machines)": f"{J['A'][i, 1]:.4f}",
                   "priority tier": int(J["q"][i]), "duration (h)": f"{J['d'][i]:.4f}", "r*d (machine-h)": f"{rd[i]:.4g}",
                   "(1 + ln q)": f"{1 + np.log(J['q'][i]):.4f}", "size rate $/h g1 cores + g2 GB": f"{size_rate:.4f}",
                   "cost base $": f"{J['cost'][i]:.5g}", "bid $": f"{J['v'][i]:.5g}", "phi $": f"{J['phi'][i]:.5g}",
                   "customer value $": f"{J['cv'][i]:.5g}", "electricity $": f"{J['elec'][i]:.5g}",
                   "energy kWh": f"{J['E'][i]:.4g}", "carbon kg": f"{J['carbon_kg'][i]:.4g}",
                   "carbon $ (s_C x kg)": f"{J['carbon_usd'][i]:.5g}", "profit $": f"{J['profit'][i]:.5g}",
                   "reward $": f"{J['R'][i]:.5g}"})
    table(pd.DataFrame(ex).set_index("job").T.reset_index().rename(columns={"index": "term"}), "N4. v5: two worked examples")

    # ---- profit sanity: bid and electricity per machine-hour ----------------------------------------
    m = rd > 0
    bph, eph = J["v"][m] / rd[m], J["elec"][m] / rd[m]
    out(f"\nN5. Per CPU machine-hour (v5 jobs with r*d > 0): bid mean {bph.mean():.4g} $, p10 {np.quantile(bph, .1):.4g}, "
        f"p90 {np.quantile(bph, .9):.4g}; electricity mean {eph.mean():.4g} $, p10 {np.quantile(eph, .1):.4g}, "
        f"p90 {np.quantile(eph, .9):.4g}; bid / electricity: median {np.median(J['v'][m] / J['elec'][m]):.4g}.")
    out("N6. " + load_by_sign(ds5, st5))
    out("N6 (v4). " + load_by_sign(runs["v4 (base-utility bid, f 1.00)"][1], runs["v4 (base-utility bid, f 1.00)"][2]))

    # ---- quota share 0.2 vs 0.5 at f = 1.25 ------------------------------------------------------------
    job = pd.read_csv(ROOT / cfg5.data.batch_csv)
    cpu, span = job.CPU.to_numpy(float), job.duration.to_numpy(float)
    eff = np.divide(job.CPU_hours_real.to_numpy(float), cpu, out=span.copy(), where=cpu > 0)
    total_cpuh = (cpu * np.minimum(np.maximum(eff, cfg5.time.duration_floor_hours), cfg5.limits.d_cap_hours)).sum()
    rows = []
    for share in (0.2, 0.5):
        c = replace(cfg5, limits=replace(cfg5.limits, quota_share=share)).validate()
        ds = load_dataset(c)
        st = run_phase0(ds, c)
        s = phase0_summary(ds, st, c)
        arr, dst, A, dh, dts = timeline(ds, c)
        _, lev, w = demand(arr, dst, A, arr[0], arr[-1])
        full_any = (w * (lev >= FULL * ds.capacity[None, :]).any(1)).sum() / w.sum()
        rows.append({"quota share": share, "C CPU / RAM": f"{ds.capacity[0]:.3f} / {ds.capacity[1]:.3f}",
                     "Q CPU / RAM": f"{ds.limit_report.quota[0]:.4f} / {ds.limit_report.quota[1]:.4f}",
                     "fixed-point steps": len(ds.limit_report.quota_iterations),
                     "% time demand >= 95% C": pct(full_any), "jobs kept": pct(len(A) / len(job)),
                     "CPU-hours kept": pct((A[:, 0] * dh).sum() / total_cpuh),
                     "tau": str([round(x, 6) for x in s["tau"]]),
                     "LP fill CPU / RAM": f"{pct(s['lp_fill'][0])} / {pct(s['lp_fill'][1])}",
                     "positive types / mass": f"{pct(s['share_positive_reward_types'])} / {pct(s['share_positive_reward_mass'])}",
                     "s_C $/kg": f"{s['s_C']:.4g}"})
    table(pd.DataFrame(rows), "N7. v5 at f = 1.25: quota share 0.2 vs 0.5 (each with its fixed point)")
    (OUT / "v5_size_bid_numbers.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

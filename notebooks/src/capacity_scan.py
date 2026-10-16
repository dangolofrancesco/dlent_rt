"""
Capacity scan (week 30, read-only, Phase 0 + full-stream reference only; no online run).

    dlent_venv/bin/python notebooks/src/capacity_scan.py

Base: the COMMITTED configs/default.yaml (git HEAD), default grid (k* 128, 32 profiles), capacity
fraction_of_concurrency_quantile with q = 0.7, drop_jobs_exceeding_capacity ON, fraction f in
FRACTIONS. For each f, for H and for the stream:
  * c (cores, GB);
  * jobs removed by the filter (count, %) and their share of the total positive R_job, computed
    with the flag OFF on the same c (R_job under that Phase 0 state: its bid model and s_C);
  * post-filter profitable load sum_{r~_j>0} p_j v_ij / c_i (H: Phase 0 state; stream: the
    full-stream reference state of Benchmarks, i.e. the fluid OPT's state);
  * tau (H: tau_0 of Phase 0; stream: tau* of the reference) and the fluid LP objective
    lambda* * T (H: Phase 0's lambda*; stream: fluid OPT = lambda_ref * T).
Then the selection of item 2: the smallest f with profitable load >= c/0.6 (166.7%) on at least
one resource and reward removed <= 10%, with H (primary) and with the stream.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks" / "src"))

from dlent_rt.benchmarks import Benchmarks
from dlent_rt.config import load_config
from dlent_rt.data import load_dataset
from dlent_rt.models.core.stream import build_stream
from dlent_rt.phase0 import run_phase0
from dlent_rt.reward import job_reward, job_terms
from lambda_sensitivity_check import h_jobs

warnings.simplefilter("ignore")
FRACTIONS = (0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3)
Q = 0.7
RHO = 0.6
MAX_REMOVED_PCT = 10.0
REPORT = ROOT / "reports" / "week30"


def committed_default():
    txt = subprocess.check_output(["git", "-C", str(ROOT), "show", "HEAD:configs/default.yaml"], text=True)
    p = Path(tempfile.mkdtemp()) / "default_head.yaml"
    p.write_text(txt)
    return load_config(p)


def scan_config(f: float, drop: bool = True):
    base = committed_default()
    cap = replace(base.capacity, mode="fraction_of_concurrency_quantile", concurrency_quantile=Q,
                  fraction=f, drop_jobs_exceeding_capacity=drop)
    return replace(base, capacity=cap).validate()


def prof_load(st, c):
    return (st.p * (st.r_tilde > 0)) @ st.v / c


def one(f: float) -> dict:
    # flag OFF, same c: what the filter removes
    cfg0 = scan_config(f, drop=False)
    ds0 = load_dataset(cfg0)
    st0 = run_phase0(ds0, cfg0)
    c = np.asarray(ds0.capacity, float)
    big_H = np.any(ds0.H.A > c, axis=1)
    big_S = np.any(ds0.stream.A > c, axis=1)
    J = h_jobs(ds0, st0, cfg0)
    R_H = job_terms(J["q"], J["v"], J["c"], J["C_elec"], J["K"], st0.myerson, st0.carbon_price,
                    cfg0.objective).R
    R_S = job_reward(build_stream(ds0, cfg0), st0, cfg0.objective).R
    pos_H, pos_S = np.maximum(R_H, 0), np.maximum(R_S, 0)

    # flag ON
    cfg = scan_config(f, drop=True)
    ds = load_dataset(cfg)
    assert np.array_equal(ds.capacity, c)
    st = run_phase0(ds, cfg)
    B = Benchmarks(ds, st, cfg)
    ref = B.reference_state
    em = cfg.energy_model
    lH, lS = prof_load(st, c), prof_load(ref, c)
    return {
        "fraction": f, "c CPU (cores)": c[0] * em.spec_cpu_core, "c RAM (GB)": c[1] * em.spec_ram_gb,
        "H removed": int(big_H.sum()), "H removed %": 100 * big_H.mean(),
        "stream removed": int(big_S.sum()), "stream removed %": 100 * big_S.mean(),
        "H R_job+ removed %": 100 * pos_H[big_H].sum() / pos_H.sum(),
        "stream R_job+ removed %": 100 * pos_S[big_S].sum() / pos_S.sum(),
        "H prof load CPU %": 100 * lH[0], "H prof load RAM %": 100 * lH[1],
        "stream prof load CPU %": 100 * lS[0], "stream prof load RAM %": 100 * lS[1],
        "tau_0 (H) CPU": float(st.tau[0]), "tau_0 (H) RAM": float(st.tau[1]),
        "tau* (stream) CPU": float(ref.tau[0]), "tau* (stream) RAM": float(ref.tau[1]),
        "fluid LP H: lambda*_0 * T $": float(st.lam_star) * ds.T,
        "fluid LP stream: fluid OPT $": B.fluid_opt(),
        "s_C $/kgCO2": st.carbon_price,
    }


def select(df: pd.DataFrame, pop: str) -> str:
    load = df[[f"{pop} prof load CPU %", f"{pop} prof load RAM %"]].max(axis=1)
    removed = df[f"{pop} R_job+ removed %"]
    ok = (load >= 100 / RHO) & (removed <= MAX_REMOVED_PCT)
    if ok.any():
        f = float(df.index[ok][0])
        return (f"(a) smallest f with max-resource profitable load >= {100 / RHO:.1f}% of c and R_job+ removed "
                f"<= {MAX_REMOVED_PCT:.0f}% ({pop}): **f = {f}** (load {load[f]:.1f}%, removed {removed[f]:.2f}%)"), f
    k = float(load.idxmax())
    return (f"(a) no f satisfies both ({pop}). (b) maximum post-filter profitable load = {load[k]:.1f}% of c "
            f"at f = {k}, with {removed[k]:.2f}% of R_job+ removed"), k


def fmt(v):
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, (float, np.floating)):
        return f"{v:,.2f}" if abs(v) >= 1 or v == 0 else f"{v:.3g}"
    return str(v)


def main():
    rows = [one(f) for f in FRACTIONS]
    df = pd.DataFrame(rows).set_index("fraction")
    df.to_csv(REPORT / "tables" / "capacity_scan.csv")
    print(df.T.to_string(float_format=lambda x: f"{x:.4g}"))
    for pop in ("H", "stream"):
        print(select(df, pop)[0])
    md = ["| quantity | " + " | ".join(f"f = {f}" for f in df.index) + " |",
          "|---|" + "---|" * len(df.index)]
    for col in df.columns:
        md.append(f"| {col} | " + " | ".join(fmt(df.loc[f, col]) for f in df.index) + " |")
    (REPORT / "tables" / "capacity_scan_table.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()

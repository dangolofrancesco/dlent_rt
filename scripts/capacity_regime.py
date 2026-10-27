"""
Cluster-capacity regime comparison (week 30, report only). One row per candidate regime, seed 42, on the COMMITTED
configs/default.yaml (`git show HEAD:configs/default.yaml` into a temporary file).

    dlent_venv/bin/python scripts/capacity_regime.py      # -> reports/week30/cluster_regime/*.csv

Regimes: default (from_amax, no filter); capacity fraction_of_concurrency_quantile q = 0.7 with
drop_jobs_exceeding_capacity ON at f in {1.0, 0.5, 0.3, 0.2}; at f = 0.2 arrival-time compression k in {2, 3, 4}.

Computed here (cheap): c, jobs / reward removed by the filter (flag OFF, same c), Sigma R_job+ (no capacity), the
job-level MILP (re-solved where it takes seconds: every regime except k >= 2, whose 600 s MILPs come from
reports/week30/tables/intensity_online.csv), fluid OPT, G2 / G1 / StaticPolicy (same simulate loop and gate),
G2 physical rejects and the share of time with occupancy >= 95% of c, tau_0 and tau*, jobs per populated type,
the theory constants (ErrorParams with the forced xi / gamma and with the data values), and the windowed
"profitable load" (all jobs with R_job > 0 admitted, no capacity) over 1 h / 6 h / 1 day windows.
DLENT-RT comes from the existing runs (cited per row). No maths, defaults or configs are changed.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dlent_rt.benchmarks import Benchmarks
from dlent_rt.config import load_config
from dlent_rt.data import load_dataset
from dlent_rt.models.core.error_params import make_error_params
from dlent_rt.models.core.occupancy import Occupancy
from dlent_rt.models.core.stream import build_stream
from dlent_rt.models.policy import JobGreedyPolicy, StaticPolicy, TypeGreedyPolicy, run_policy, simulate
from dlent_rt.phase0 import assign_type_ids, run_phase0

TAB = ROOT / "reports" / "week30" / "tables"
OUT = ROOT / "reports" / "week30" / "cluster_regime"
WINDOWS = {"1 h": 3600, "6 h": 6 * 3600, "1 day": 24 * 3600}      # ticks of 1 s (compressed ticks under k)


def committed():
    txt = subprocess.check_output(["git", "-C", str(ROOT), "show", "HEAD:configs/default.yaml"], text=True)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "default_committed.yaml"
        p.write_text(txt)
        return load_config(p)


def regime_cfg(base, f=None, k=1.0, drop=True):
    cfg = base
    if f is not None:
        cfg = replace(cfg, capacity=replace(cfg.capacity, mode="fraction_of_concurrency_quantile",
                                            concurrency_quantile=0.7, fraction=f,
                                            drop_jobs_exceeding_capacity=drop))
    if k != 1.0:
        cfg = replace(cfg, time=replace(cfg.time, arrival_time_scale=float(k)))
    return cfg.validate()


def step_from_events(events, T):
    """(breakpoints t, level U on [t_k, t_{k+1})) of the occupancy step function on [0, T]."""
    ev = np.array([(e[0], *e[3:]) for e in events], float)
    te, Ue = ev[:, 0], ev[:, 1:]
    keep = np.r_[te[1:] != te[:-1], True]
    te, Ue = te[keep], Ue[keep]
    return np.r_[0.0, te], np.vstack([np.zeros(Ue.shape[1]), Ue])


def step_from_jobs(t, D, A):
    times = np.r_[t, t + D].astype(float)
    d = np.vstack([A, -A])
    o = np.argsort(times, kind="stable")
    uniq, inv = np.unique(times[o], return_inverse=True)
    lev = np.zeros((uniq.size, A.shape[1]))
    np.add.at(lev, inv, d[o])
    return np.r_[0.0, uniq], np.vstack([np.zeros(A.shape[1]), np.cumsum(lev, axis=0)])


def time_share_at_least(bp, lev, T, thr):
    """Share of [0, T) during which any resource has U >= thr."""
    dur = np.diff(np.r_[np.clip(bp, 0, T), T])
    return float(dur[(lev >= thr[None, :] - 1e-12).any(axis=1)].sum() / T)


def windowed(bp, lev, T, c, step=60):
    """Average, instantaneous peak and the peak of the w-window average of U, as % of c."""
    dur = np.diff(np.r_[bp, max(T, bp[-1])])
    I = np.vstack([np.zeros(lev.shape[1]), np.cumsum(lev * dur[:, None], axis=0)])   # integral at bp, end
    xs = np.r_[bp, max(T, bp[-1])]
    grid = np.arange(0, T + 1, step, dtype=float)
    Ig = np.column_stack([np.interp(grid, xs, I[:, i]) for i in range(lev.shape[1])])
    out = {"avg": I[-1] / T, "inst peak": lev.max(axis=0)}
    for name, w in WINDOWS.items():
        n = w // step
        if n >= len(grid):
            out[name] = out["avg"]
            continue
        out[name] = ((Ig[n:] - Ig[:-n]) / w).max(axis=0)
    return {k: 100 * v / c for k, v in out.items()}


def run_regime(label, cfg, cached, base_cfg):
    t0 = time.time()
    ds = load_dataset(cfg)
    st = run_phase0(ds, cfg)
    S = build_stream(ds, cfg)
    B = Benchmarks(ds, st, cfg, stream=S)
    R, c, T = B.R, np.asarray(ds.capacity, float), int(ds.T)
    em = cfg.energy_model
    o = {"regime": label, "c CPU (cores)": c[0] * em.machine_cores, "c RAM (GB)": c[1] * em.machine_ram_gb,
         "c CPU (norm)": c[0], "c RAM (norm)": c[1], "|H|": ds.H.n, "|stream|": S.n, "T (ticks)": T,
         "s_C": st.carbon_price}
    # filter: what it removes (flag OFF on the same c)
    if cfg.capacity.drop_jobs_exceeding_capacity:
        cfg0 = replace(cfg, capacity=replace(cfg.capacity, drop_jobs_exceeding_capacity=False))
        ds0 = load_dataset(cfg0)
        st0 = run_phase0(ds0, cfg0)
        S0 = build_stream(ds0, cfg0)
        R0 = Benchmarks(ds0, st0, cfg0, stream=S0).R
        big = np.any(S0.A > c[None, :], axis=1)
        o.update({"stream jobs dropped": int(big.sum()), "stream jobs dropped %": 100 * big.mean(),
                  "H jobs dropped": ds.capacity_drop["n_H"],
                  "stream R_job+ removed %": 100 * np.maximum(R0, 0)[big].sum() / np.maximum(R0, 0).sum(),
                  "stream R_job+ before filter $": float(np.maximum(R0, 0).sum())})
    else:
        o.update({"stream jobs dropped": 0, "stream jobs dropped %": 0.0, "H jobs dropped": 0,
                  "stream R_job+ removed %": 0.0, "stream R_job+ before filter $": float(np.maximum(R, 0).sum())})
    R_nocap = float(np.maximum(R, 0).sum())
    o["R no capacity $"] = R_nocap
    if cached.get("job_opt") is None:
        job = B.job_level_opt(time_limit=600.0)
        o.update({"R job MILP $": job["R_ip"], "job MILP status": job["status"], "job MILP gap": job["gap"],
                  "job MILP runtime s": job["runtime_s"], "job MILP source": "computed now"})
    else:
        o.update({"R job MILP $": cached["job_opt"], "job MILP status": cached["job_status"],
                  "job MILP gap": cached["job_gap"], "job MILP runtime s": np.nan,
                  "job MILP source": cached["job_src"]})
    jm = o["R job MILP $"]
    o["capacity cost % ((no-cap - job MILP)/no-cap)"] = 100 * (R_nocap - jm) / R_nocap
    o["R fluid OPT $"] = B.fluid_opt()
    o["job MILP - fluid OPT $"] = jm - o["R fluid OPT $"]
    # G2 with the occupancy log
    occ = Occupancy(c, record=True)
    g2 = simulate(S, occ, JobGreedyPolicy(R))
    bp, lev = step_from_events(occ.events, T)
    o.update({"R G2 $": B.score(g2.admit), "G2 phys rejects": int(g2.phys_reject.sum()),
              "G2 phys rejects % of jobs": 100 * g2.phys_reject.mean(),
              "G2 phys rejects % of wanted": 100 * g2.phys_reject.sum() / max(int(g2.want.sum()), 1),
              "G2 time with U >= 95% c (%)": 100 * time_share_at_least(bp, lev, T, 0.95 * c)})
    g1 = run_policy(S, c, TypeGreedyPolicy.from_state(st, S, R, ds.H_span_steps))
    sp = run_policy(S, c, StaticPolicy(st, S))
    o.update({"R G1 $": B.score(g1.admit), "R static P0 $": B.score(sp.admit)})
    for k_ in ("G2", "G1", "static P0", "fluid OPT"):
        o[f"{k_} % of job MILP"] = 100 * o[f"R {k_} $"] / jm
    o.update({"R DLENT-RT $": cached["dlent_R"], "DLENT-RT % of job MILP": 100 * cached["dlent_R"] / jm,
              "DLENT-RT end reasons": cached["dlent_reasons"], "DLENT-RT source": cached["dlent_src"]})
    ref = B.reference_state
    k = st.grid.assign(st.bids.v_rate, st.myerson)
    tid = assign_type_ids(st.type_space, np.asarray(k), ds.H.priority, ds.H.hw_idx)
    o.update({"tau_0 (Phase 0)": str(np.round(st.tau, 5).tolist()), "tau* (hindsight)": str(np.round(ref.tau, 5).tolist()),
              "populated types": int((st.p > 0).sum()),
              "H jobs per populated type": int((tid >= 0).sum()) / max(int((st.p > 0).sum()), 1)})
    # theory constants: forced (the run) and data (overrides off)
    ep = make_error_params(st, ds, cfg)
    cfg_no = replace(cfg, constants=replace(cfg.constants, xi_override=None, gamma_override=None))
    ep_no = make_error_params(st, ds, cfg_no)
    d = int(ds.d_max_steps)
    for tag, e in (("forced", ep), ("data", ep_no)):
        o.update({f"xi ({tag})": e.xi, f"gamma ({tag})": e.gamma, f"eps_D ({tag})": e.eps_D,
                  f"1-eps_D-xi ({tag})": 1 - e.eps_D - e.xi, f"p_U cap ({tag})": (1 - e.xi) / (1 + e.eps_D),
                  f"p_U at phase-1 start ({tag})": e.p_U(d, 1), f"eta(d_max) ({tag})": float(e.eta(d)),
                  f"eps_A'(d_max) ({tag})": float(e.eps_Ap(d)), f"eps_Test2(d_max) ({tag})": float(e.eps_test2(d, d)),
                  f"eps_Test3(d_max) ({tag})": float(e.eps_test3(d))})
    o.update({"xi data = max a_max/c": float(np.max(np.asarray(ds.a_max) / c)), "c_min": float(c.min()),
              "Test1 margin forced (xi*c) CPU,RAM": str(np.round(ep.xi * c, 6).tolist()),
              "Test1 threshold data (c - a_max) CPU,RAM": str(np.round(c - np.asarray(ds.a_max), 5).tolist()),
              "d_max": d, "d_max / T": d / T})
    # windowed profitable load (every job with R_job > 0 admitted, no capacity)
    pos = R > 0
    bp2, lev2 = step_from_jobs(S.tick[pos], S.D[pos], S.A[pos])
    wl = windowed(bp2, lev2, T, c)
    for name, v in wl.items():
        o[f"profitable load {name} CPU % c"] = v[0]
        o[f"profitable load {name} RAM % c"] = v[1]
    for name in ("inst peak", *WINDOWS):
        o[f"peak/avg {name} (max resource)"] = float(np.max(wl[name] / wl["avg"]))
    o["time with profitable load > c (%)"] = 100 * time_share_at_least(bp2, lev2, T, c * (1 + 1e-12))
    o["runtime s"] = time.time() - t0
    print(f"{label}: done in {o['runtime s']:.0f} s", flush=True)
    return o


def cached_values():
    """DLENT-RT (and the k >= 2 job MILPs) from the existing runs, with their source."""
    sw = pd.read_csv(TAB / "table08_capacity_grid_sweep_full.csv", index_col=[0, 1])
    it = json.loads((TAB / "intensity_online.json").read_text())
    f02 = json.loads((TAB / "capacity_scan_online_f0.2.json").read_text())

    def reasons(d):
        return ", ".join(f"{r}={int(d[f'end_{r}'])}" for r in ("timeout", "test1", "test2", "test3", "horizon"))
    out = {"default": dict(dlent_R=48391.61142793904, dlent_src="tests/data/snapshot_real.npz / nb03_numbers.md",
                           dlent_reasons="timeout=9, test1=0, test2=1, test3=0, horizon=1")}
    for f in (1.0, 0.5, 0.3):
        r = sw.loc[(f, "default (k*=128, hw=32)")]
        out[f"f={f}"] = dict(dlent_R=float(r["R DLENT-RT $"]), dlent_src="tables/table08_capacity_grid_sweep_full.csv "
                             "(nb03 Section 8, capacity_grid_sweep.md)",
                             dlent_reasons=", ".join(f"{x}={int(r[f'end {x}'])}"
                                                     for x in ("timeout", "test1", "test2", "test3", "horizon")))
    out["f=0.2"] = dict(dlent_R=f02["R_DLENT"], dlent_src="tables/capacity_scan_online_f0.2.json (capacity_scan.md)",
                        dlent_reasons=reasons(f02))
    for k in (2, 3, 4):
        o = it[str(k)]
        out[f"f=0.2, k={k}"] = dict(dlent_R=o["R_DLENT"], dlent_src="tables/intensity_online.json (intensity_experiment.md)",
                                    dlent_reasons=reasons(o), job_opt=o["R_job_opt"], job_status=o["milp_status"],
                                    job_gap=o["milp_gap"],
                                    job_src="tables/intensity_online.json (600 s limit; incumbent)")
    return out


def main():
    warnings.simplefilter("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    base = committed()
    blob = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD:configs/default.yaml"], text=True).strip()
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    cache = cached_values()
    regimes = [("default", regime_cfg(base)), *[(f"f={f}", regime_cfg(base, f)) for f in (1.0, 0.5, 0.3, 0.2)],
               *[(f"f=0.2, k={k}", regime_cfg(base, 0.2, k)) for k in (2, 3, 4)]]
    rows = [run_regime(lab, cfg, cache[lab], base) for lab, cfg in regimes]
    df = pd.DataFrame(rows).set_index("regime")
    df.to_csv(OUT / "regimes.csv")
    pd.Series({"commit": commit, "configs/default.yaml blob": blob,
               "config": "git show HEAD:configs/default.yaml -> temporary file (deleted)"}).to_csv(
        OUT / "meta.csv", header=["value"])
    pd.set_option("display.width", 250)
    print(df.T.to_string())


if __name__ == "__main__":
    main()

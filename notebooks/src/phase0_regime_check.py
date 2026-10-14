"""
Phase-0 regime check (READ-ONLY): capacity / load / LP outcome, and how the error parameters
of error_params.py react to xi and gamma. Nothing in dlent_rt/ or the constants is changed: the
sensitivity tables call the existing ErrorParams methods on copies with xi or gamma replaced.

    dlent_venv/bin/python notebooks/src/phase0_regime_check.py         # real trace
    DLENT_SYNTHETIC=1 dlent_venv/bin/python notebooks/src/phase0_regime_check.py

Config: default + bid.strategy=lognormal, bid.sigma=0.3, data.n0=10000.
"""
from __future__ import annotations

import dataclasses, os, sys, tempfile, warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dlent_rt.config import default_config, resolve_path
from dlent_rt.data import load_dataset
from dlent_rt.phase0 import run_phase0, assign_type_ids, _weighted_quantile
from dlent_rt.lp_oracle import solve_lp_rs
from dlent_rt.models.core.error_params import make_error_params

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda x: f"{x:.6g}")


def _replace_path(obj, path, value):
    if len(path) == 1:
        return dataclasses.replace(obj, **{path[0]: value})
    return dataclasses.replace(obj, **{path[0]: _replace_path(getattr(obj, path[0]), path[1:], value)})


def apply_overrides(cfg, overrides):
    for k, v in overrides.items():
        cfg = _replace_path(cfg, k.split("."), v)
    return cfg


cfg = default_config()
cfg = apply_overrides(cfg, {"bid.strategy": "lognormal", "bid.sigma": 0.3, "data.n0": 10000})
if os.environ.get("DLENT_SYNTHETIC") == "1" or not resolve_path(cfg.data.batch_csv).is_file():
    from tests.synthetic import write_synthetic
    b, g = write_synthetic(Path(tempfile.mkdtemp()), n_jobs=3000, drift=True)
    cfg = apply_overrides(cfg, {"data.batch_csv": str(b), "data.grid_csv": str(g), "data.n0": 1000,
                                "grid.k_star": 16, "hardware.n_profiles": 6,
                                "time.d_max_horizon_fraction": 0.10, "time.d_max_ratio_abort": 0.2})
    print("USING SYNTHETIC FALLBACK DATA: numbers are NOT meaningful")
cfg = cfg.validate()
CORES, GB = cfg.energy_model.spec_cpu_core, cfg.energy_model.spec_ram_gb
UNIT = np.array([CORES, GB])

with warnings.catch_warnings(record=True) as W:
    warnings.simplefilter("always")
    ds = load_dataset(cfg)
    st = run_phase0(ds, cfg)
for w in W:
    if "xi" in str(w.message):
        print(f"WARNING ({Path(w.filename).name}:{w.lineno}): {w.message}")
ep = make_error_params(st, ds, cfg)
print(f"seed={cfg.run.seed}  |H|={ds.H.n:,}  |stream|={ds.stream.n:,}  T={ds.T:,}  d_max={ds.d_max_steps:,} steps  "
      f"capacity.mode={cfg.capacity.mode} xi_target={cfg.capacity.xi_target}")


def section(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


# --------------------------------------------------------------------------- 1
section("1. Capacity, a_max, xi_data, jobs with A > c")
c = ds.capacity
A_all = np.concatenate([ds.H.A, ds.stream.A])
amax_all = A_all.max(axis=0)
r = ds.outlier_report
t1 = pd.DataFrame({
    "capacity (norm.)": c, "capacity (cores | GB)": c * UNIT,
    "a_max stream (norm.)  [ds.a_max]": ds.a_max, "a_max H+stream (norm.)": amax_all,
    "a_max H+stream (cores | GB)": amax_all * UNIT,
    "outlier cutoff, raw request (cores | GB)": r.a_max_thresholds * UNIT,
    "xi_data = a_max(stream) / c": ds.a_max / c,
    "jobs with A > c (H+stream)": (A_all > c).sum(axis=0),
}, index=["CPU", "RAM"]).T
print(t1.to_string())
print(f"ds.xi (loader) = {ds.xi:.6g};  state.xi (Phase 0) = {st.xi:.6g};  xi used by the loop = {ep.xi} "
      f"({'forced, constants.xi_override' if cfg.constants.xi_override is not None else 'data'})")

# --------------------------------------------------------------------------- 2
section("2. Expected consumption (time average)")
pos = st.r_tilde > 0
load_all = st.p @ st.v                        # sum_j p_j v_ij  (H, per tick = LP left-hand side at x = 1)
load_pos = st.p[pos] @ st.v[pos]
S = ds.stream
load_stream = (S.A * S.duration_steps[:, None]).sum(axis=0) / ds.T
from dlent_rt.models.core.stream import build_stream
SD = build_stream(ds, cfg)
tid = assign_type_ids(st.type_space, st.grid.assign(SD.v, st.myerson), SD.priority, SD.hw_idx)
pos_s = (tid >= 0) & pos[np.maximum(tid, 0)]
load_stream_pos = (S.A[pos_s] * S.duration_steps[pos_s, None]).sum(axis=0) / ds.T
rows = {}
for name, L in [("all jobs, H (sum p v)", load_all), ("r~ > 0, H (sum p v over r~>0 types)", load_pos),
                ("all jobs, stream (sum A D / T)", load_stream),
                ("r~ > 0, stream (Phase-0 types)", load_stream_pos)]:
    rows[name] = {"CPU cores": L[0] * CORES, "RAM GB": L[1] * GB, "CPU % of c": 100 * L[0] / c[0],
                  "RAM % of c": 100 * L[1] / c[1]}
print(pd.DataFrame(rows).T.to_string())
print(f"populated types: {int((st.p > 0).sum())}, with r~ > 0: {int((pos & (st.p > 0)).sum())} "
      f"(p-mass {100 * st.p[pos].sum() / st.p.sum():.1f}%)")
fac = pd.DataFrame({
    "profitable load / c (H)": load_pos / c,
    "factor to reach 1.5 c (H)": 1.5 * c / load_pos, "factor to reach 2 c (H)": 2.0 * c / load_pos,
    "profitable load / c (stream)": load_stream_pos / c,
    "factor to reach 1.5 c (stream)": 1.5 * c / load_stream_pos,
    "factor to reach 2 c (stream)": 2.0 * c / load_stream_pos,
}, index=["CPU", "RAM"]).T
print(fac.to_string())

# --------------------------------------------------------------------------- 3
section("3. LP-RS outcome (Phase 0)")
lp = solve_lp_rs(p=st.p, r=st.r_tilde, v=st.v, c=c, presolve=cfg.lp.presolve)
x = np.zeros(st.n_types)
x[lp.support_idx] = lp.x
util = (st.p * x) @ st.v / c
print(f"status={lp.status}  lambda* = {lp.optimum:.6g} $/tick (state.lam_star = {st.lam_star:.6g})")
print(pd.DataFrame({"tau ($ per unit/tick)": lp.tau, "util = sum p v x / c": util}, index=["CPU", "RAM"]).T.to_string())
# jobs of H per type; a type with x in (0,1) splits its jobs x admitted / (1-x) rejected by capacity
N = np.round(st.p * ds.H_span_steps).astype(np.int64)    # p = N / H_span (real_tick)
n_tot = N.sum()
adm = float((N * x).sum())
rej_rew = float(N[~pos].sum())
rej_cap = float((N[pos] * (1 - x[pos])).sum())
out = pd.DataFrame({"jobs": [adm, rej_rew, rej_cap, n_tot],
                    "%": [100 * adm / n_tot, 100 * rej_rew / n_tot, 100 * rej_cap / n_tot, 100.0]},
                   index=["admitted (sum N x)", "rejected by reward (r~ <= 0)",
                          "rejected by capacity (r~ > 0, 1 - x)", "total assigned H jobs"])
print(out.to_string())
print(f"fractional types (0 < x < 1): {int(((x > 1e-9) & (x < 1 - 1e-9)).sum())};  "
      f"r~>0 types with x = 0: {int((pos & (st.p > 0) & (x <= 1e-9)).sum())}")

# --------------------------------------------------------------------------- 4
section("4. xi sensitivity of the constants (ErrorParams with xi replaced; gamma as used)")
rows = []
for xv in [0.001, 0.01, 0.05, 0.1, 0.5]:
    e = dataclasses.replace(ep, xi=xv)
    rows.append({"xi": xv, "eps_D": e.eps_D, "thinning cap (1-xi)/(1+eps_D)": (1 - xv) / (1 + e.eps_D),
                 "1 - eps_D - xi": 1 - e.eps_D - xv,
                 "Test-1 margin xi*c CPU cores": xv * c[0] * CORES, "Test-1 margin xi*c RAM GB": xv * c[1] * GB})
print(pd.DataFrame(rows).set_index("xi").to_string())
print(f"loop uses xi = {ep.xi} (forced, constants.xi_override); xi_data = {ep.xi_data:.6g}; "
      f"log(|I|/delta) = {ep.log_I:.4g}, delta = {ep.delta:.3g}")

# --------------------------------------------------------------------------- 5
section("5. gamma sensitivity (ErrorParams with gamma replaced; xi as used)")
m = st.p > 0
abs_r = np.abs(st.r_tilde[m])
w = st.p[m]
W_prev1 = int(ds.d_max_steps)        # phase 1: TheoryModel._new_phase(state0, tau_prev=0, W_prev=d_max)
cands = {"1 (forced)": 1.0, "max|r~|": float(abs_r.max()),
         "Q99(|r~|) p-weighted": _weighted_quantile(abs_r, w, 0.99),
         "Q90(|r~|) p-weighted": _weighted_quantile(abs_r, w, 0.90),
         "data-derived gamma (state.gamma)": float(st.gamma)}
rows = []
for name, gv in cands.items():
    e = dataclasses.replace(ep, gamma=gv)
    for n in (W_prev1, 2 * W_prev1):
        rows.append({
            "gamma": name, "value": gv, "n = t - tau_0": n,
            "eta(1)": float(e.eta(1)),
            "p_U(t - tau = 1)": e.p_U(W_prev1, 1), "p_U(t - tau = d_max)": e.p_U(W_prev1, W_prev1),
            "gamma*eta(W_prev)": float(gv * e.eta(W_prev1)), "4 gamma/sqrt(W_prev)": 4 * gv / np.sqrt(W_prev1),
            "CI1(n)": float(e.CI1(n)), "eps_test2": float(e.eps_test2(W_prev1, n)),
            "eps_test2 / lambda*": float(e.eps_test2(W_prev1, n)) / max(st.lam_star, 1e-300),
        })
print(pd.DataFrame(rows).set_index(["gamma", "n = t - tau_0"]).to_string())
print(f"phase 1: W_prev = d_max = {W_prev1:,} ticks (theory.py run(): _new_phase(..., W_prev=self.d_max)); "
      f"its timeout is n = 2 W_prev = {2 * W_prev1:,}, so CI1 is shown at n = W_prev and n = 2 W_prev. "
      f"c_min = {ep.c_min:.6g} (normalized). The first realised phase length is only known after a run.")
print(f"data-derived gamma = max(r_max, v_rate_max, d_max, c_max, r_max/v_min) (phase0._system_constants) "
      f"= {st.gamma:.6g}; d_max = {ds.d_max_steps}, v_rate_max = {st.bids.v_rate.max():.6g}")

# --------------------------------------------------------------------------- 6
section("6. Where xi and gamma enter the code (read-only)")
print("""\
xi:
  data.py:432               ds.xi = max_i a_max,i(stream) / c_i  (warning at data.py:434 if > outliers.xi_warn_threshold)
  phase0.py:273             state.xi = max_i ds.a_max / ds.capacity  (data-derived, logged as xi_data)
  error_params.py:131       ep.xi = constants.xi_override if set, else state.xi
  error_params.py:73        eps_D = 2 sqrt(xi log(|I|/delta)) + 2 xi log(|I|/delta)
  error_params.py:94        p_U = (1 - xi) / (1 + eps_U)  (thinning; eps_U contains eps_D)
  theory.py:61              a_max_eff = xi * c when xi is forced (else ds.a_max) -> Test 1 at monitor.py:144 (c - a_max_eff)
  monitor.py:65             fac = 1 - eps_D - xi  (Test-2 / Test-3 running sums, monitor.py:119)
  theory.py:109,229         target_paper = (1 - eps_D - xi) * bench, logged
gamma:
  phase0.py:272             state.gamma = max(r_max, v_rate_max, d_max, c_max, r_max / v_min)
  error_params.py:130       ep.gamma = constants.gamma_override if set, else state.gamma
  error_params.py:50-68     eps_A', eps_B', eps_C', eps_G, eta = gamma / (c_min sqrt(n))
  error_params.py:91,94     eps_U -> p_U (thinning) through eta(W_prev) + eta(t - tau)
  error_params.py:97-105    CI1 and eps_test2 (Test 2 threshold, monitor.py:137)
  error_params.py:108-115   eps_test3 and lam_tilde_term (Test 3, monitor.py:122,183)""")

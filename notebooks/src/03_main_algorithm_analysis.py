# %% [markdown]
# # DLENT-RT — Main algorithm analysis (theory model)
#
# One configuration only: `BASE = apply_overrides(cfg, {"bid.strategy": "lognormal", "bid.sigma": 0.3})` on top of
# `configs/default.yaml`, with the default normalization (`quantile_linear`, q = 0.90, no common divisor,
# s_C frozen in Phase 0). Reward in dollars, unbounded.
# Theoretical setting: the true duration of an admitted job is known at arrival (`model.kind = "theory"`).
#
# Notebook-only knob: `capacity_scale` multiplies the initial cluster capacity c_i (CPU and RAM) **before Phase 0**.
# It is an override inside this notebook, not a change of the default. Sweep: capacity_scale in {0.5, 1.0} (`DLENT_SCALES` to change);
# every section below is for the baseline 1.0, the sweep is the compact table + occupancy figure of Section 6.
#
# Sections: 0. Setup &middot; 1. Distributions (C.1) &middot; 2. Per-phase margin (C.2) &middot;
# 3. Results per phase (C.3) &middot; 4. Accuracy per phase (C.4) &middot; 5. Baselines and regret (C.5) &middot;
# 6. capacity_scale sweep &middot; 7. Anomalies.
#
# Rebuild/execute: `python notebooks/src/build.py 03_main_algorithm_analysis --execute`.
# Data: real trace only, `data/batch_may2019_30k.csv` + `data/grid_data_may_2019.csv` (no synthetic data).
# Fewer scales for a quick run: `DLENT_SCALES=1.0`.

# %% [markdown]
# ## Section 0: Setup and configuration

# %%
import os, sys, time, warnings, dataclasses
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.optimize import milp, linprog, LinearConstraint, Bounds
from scipy.stats import lognorm
from IPython.display import display
warnings.simplefilter("ignore")

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "dlent_rt").is_dir())
sys.path.insert(0, str(ROOT))

from dlent_rt.config import default_config
from dlent_rt.data import load_dataset
from dlent_rt.phase0 import (run_phase0, assign_type_ids, type_arrays, scalarized_reward,
                             _weighted_quantile)
from dlent_rt.models.theory import TheoryModel
from dlent_rt.models.core.occupancy import Occupancy
from dlent_rt.models.core.reinit import build_window_state
from dlent_rt.models.core.error_params import make_error_params

plt.rcParams["figure.figsize"] = (13, 3.6)
plt.rcParams["figure.dpi"] = 110
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)

# palette (notebook 01 conventions: named matplotlib colours)
C_SW, C_PI, C_C, C_BID = "steelblue", "darkorange", "gray", "steelblue"
C_CPU, C_RAM = "steelblue", "darkorange"
DEC_ORDER = ["accepted", "oracle reject", "thinning reject", "physical reject", "IR reject"]
DEC_STYLE = {"accepted": ("seagreen", "o"), "oracle reject": ("gray", "x"),
             "thinning reject": ("darkorange", "v"), "physical reject": ("indianred", "s"),
             "IR reject": ("purple", "D")}

# %%
def _replace_path(obj, path, value):
    if len(path) == 1:
        return dataclasses.replace(obj, **{path[0]: value})
    head, rest = path[0], path[1:]
    return dataclasses.replace(obj, **{head: _replace_path(getattr(obj, head), rest, value)})

def apply_overrides(cfg, overrides: dict):
    """Apply {'a.b.c': value, ...} on top of cfg via recursive dataclasses.replace."""
    for dotted_key, value in overrides.items():
        cfg = _replace_path(cfg, dotted_key.split("."), value)
    return cfg

# Real trace only (no synthetic fallback): the full 30k batch trace and the May-2019 grid data.
BATCH_CSV = ROOT / "data" / "batch_may2019_30k.csv"
GRID_CSV = ROOT / "data" / "grid_data_may_2019.csv"
for _p in (BATCH_CSV, GRID_CSV):
    if not _p.is_file():
        raise FileNotFoundError(f"required input not found: {_p}")

cfg = apply_overrides(default_config(), {"data.batch_csv": str(BATCH_CSV), "data.grid_csv": str(GRID_CSV)})
BASE = apply_overrides(cfg, {"bid.strategy": "lognormal", "bid.sigma": 0.3})
# Notebook-only guard: the analysis is run with n0 = 10,000 (|stream| = 15,308), the value configs/default.yaml
# currently has; if the YAML changes, this override keeps the notebook on the same H/stream split.
N0_OVERRIDE = 10_000
if BASE.data.n0 != N0_OVERRIDE:
    print(f"OVERRIDE: data.n0 {BASE.data.n0:,} (configs/default.yaml) -> {N0_OVERRIDE:,} (this notebook only)")
    BASE = apply_overrides(BASE, {"data.n0": N0_OVERRIDE})
BASE = BASE.validate()
print(f"batch trace: {BATCH_CSV}\ngrid data:   {GRID_CSV}")

CAPACITY_SCALES = [float(x) for x in os.environ.get("DLENT_SCALES", "0.5,1.0").split(",")]
BASELINE_SCALE = 1.0
assert BASELINE_SCALE in CAPACITY_SCALES
L1, L2, L3 = BASE.objective.lambda1, BASE.objective.lambda2, BASE.objective.lambda3
print(f"model={BASE.model.kind}  oracle={BASE.oracle.kind}  scalarization={BASE.scalarization.method}")
print(f"bid: strategy={BASE.bid.strategy} sigma={BASE.bid.sigma} u0={BASE.bid.base_utility_u0} "
      f"mode={BASE.bid.base_utility_mode}")
print(f"normalization: quantile_linear  q={BASE.normalization.quantile}")
print(f"lambda = ({L1}, {L2}, {L3})   k_star={BASE.grid.k_star}  hw profiles={BASE.hardware.n_profiles}")
print(f"constants: gamma_override={BASE.constants.gamma_override}  xi_override={BASE.constants.xi_override}  "
      f"delta={BASE.confidence.delta} (1/T^{BASE.confidence.delta_power} if None)")
print(f"capacity: mode={BASE.capacity.mode} fraction={BASE.capacity.fraction} "
      f"q={BASE.capacity.concurrency_quantile}   capacity_scale sweep = {CAPACITY_SCALES}")

# %%
t_load = time.time()
ds0 = load_dataset(BASE)
print(f"seeds: run.seed={BASE.run.seed} (Phase-0 bids)  stream bids={BASE.run.seed + BASE.online.bid_seed_offset}  "
      f"thinning U(t)={BASE.run.seed + BASE.online.thinning_seed_offset}")
print(f"|H| = {ds0.H.n:,}  (config data.n0 = {BASE.data.n0:,}: "
      f"{'matches' if ds0.H.n == BASE.data.n0 else 'DIFFERS from the config value'})")
print(f"|stream| = {ds0.stream.n:,}   T = {ds0.T:,} ticks of {BASE.time.tick_seconds:g} s   "
      f"d_max = {ds0.d_max_steps:,} ticks   H spans {ds0.H_span_steps:,} ticks")
print(f"capacity c (scale 1.0) = {np.round(ds0.capacity, 5)}   a_max (stream) = {np.round(ds0.a_max, 5)}   "
      f"xi_data = max a_max/c = {ds0.xi:.4f}")
print(f"loaded in {time.time() - t_load:.1f} s")

# %% [markdown]
# ### 0.1 What `capacity_scale` changes
#
# `capacity_scale` multiplies c before Phase 0, so it changes:
# * **ξ** — the data value ξ_data = max_i a_max,i / c_i scales as 1/scale. The value *used* by the loop is the forced
#   `constants.xi_override` (unchanged by the scale); with a forced ξ, Test 1 uses a_max,eff = ξ·c, which scales with c.
# * **the thinning cap** — p_U(t) = (1 − ξ) / (1 + ε_D + η(W_prev) + η(t − τ_prev)) with η(n) = γ / (c_min √n);
#   c_min scales, so η shrinks and p_U rises toward its cap (1 − ξ)/(1 + ε_D).
# * **the physical gate** — admission requires U + A_j ≤ c; jobs with A_j > c can never be admitted.
# * the LP-RS (τ, λ̂) and the Test 2/3 radii through c_min (ε_A', η).

# %%
def scaled_dataset(ds, s):
    c = np.asarray(ds.capacity, float) * s
    return dataclasses.replace(ds, capacity=c, xi=float(np.max(np.asarray(ds.a_max) / c)))

class RecordingModel(TheoryModel):
    """TheoryModel that also keeps the PhaseState used in each phase (analysis only, no logic change)."""
    def run(self):
        self.states = []
        return super().run()

    def _new_phase(self, state, tau_prev, W_prev):
        self.states.append(state)
        super()._new_phase(state, tau_prev, W_prev)

rows = []
for s in CAPACITY_SCALES:
    ds_s = scaled_dataset(ds0, s)
    st_s = run_phase0(ds_s, BASE)
    ep = make_error_params(st_s, ds_s, BASE)
    a_eff = ep.xi * ds_s.capacity if BASE.constants.xi_override is not None else ds_s.a_max
    too_big = np.any(ds_s.stream.A > ds_s.capacity[None, :], axis=1)
    rows.append({
        "capacity_scale": s, "c_cpu": ds_s.capacity[0], "c_ram": ds_s.capacity[1],
        "xi_data": ds_s.xi, "xi_used": ep.xi, "eps_D": ep.eps_D, "gamma_used": ep.gamma,
        "p_U cap (1-xi)/(1+eps_D)": (1 - ep.xi) / (1 + ep.eps_D),
        "eta(1) = gamma/c_min": ep.eta(1),
        "p_U at phase-1 start": ep.p_U(ds_s.d_max_steps, 1),
        "p_U at t-tau=d_max": ep.p_U(ds_s.d_max_steps, ds_s.d_max_steps),
        "Test1 thr c-a_eff (cpu)": ds_s.capacity[0] - a_eff[0],
        "Test1 thr c-a_eff (ram)": ds_s.capacity[1] - a_eff[1],
        "stream jobs with A>c (never fit)": int(too_big.sum()),
        "share A>c %": 100 * too_big.mean(),
        "s_C $/kg": st_s.carbon_price, "tau*_P0": np.round(st_s.tau, 6),
    })
const_df = pd.DataFrame(rows).set_index("capacity_scale")
display(const_df.T.style.format(precision=5))

# %% [markdown]
# ### 0.2 Online loop, once per capacity_scale (fixed seeds)

# %%
RUNS = {}
for s in CAPACITY_SCALES:
    t1 = time.time()
    ds_s = scaled_dataset(ds0, s)
    st_s = run_phase0(ds_s, BASE)
    model = RecordingModel(ds_s, st_s, BASE)
    res = model.run()
    RUNS[s] = dict(scale=s, ds=ds_s, st0=st_s, model=model, res=res)
    sm = res.summary
    print(f"scale={s:<4} seed={BASE.run.seed} |H|={ds_s.H.n:,} |stream|={model.stream.n:,} T={sm['T']:,}  "
          f"phases={sm['n_phases']} {sm['phase_end_reasons']}  admitted={sm['n_admitted']:,}  "
          f"R_total=${sm['R_total']:,.2f}  ({time.time() - t1:.0f} s)")

# %% [markdown]
# ### 0.3 Shared analysis helpers
#
# **Reference (ORACLE, not available online).** The full stream with true labels: every arrival, true D, F̂ refit on
# the whole stream, no fallback — the same construction as the code's hindsight benchmark
# (`reinit.hindsight_lambda`). It yields τ*, r_ref and λ̂_ref; R_OPT (fluid) = λ̂_ref · T is `summary["bench_global"]`.
#
# **One job-level reward.** The bid distribution is known (`bid.known_distribution`), so there is a single virtual
# value φ_j = `phi_job(v_j, c_j)` = c_j · φ_m(v_j / c_j) and a single job reward
# R_job = λ1·q·v + λ2·(φ_j − C_elec) − λ3·s_C·K (s_C frozen in Phase 0). It is the online loop's formula: the logged R
# of every admitted job equals R_job (asserted below), so every algorithm and benchmark is scored with R_job.
#
# **Offline job-level OPT (ORACLE).** max Σ R_job,j x_j s.t. at every arrival tick the occupancy of the chosen jobs
# is ≤ c, x ∈ {0,1} (HiGHS MILP; LP relaxation reported as an upper bound). Only jobs with R_job > 0 can be chosen.

# %%
def types_under(st, S, idx=None):
    idx = np.arange(S.n) if idx is None else np.asarray(idx)
    k = st.grid.assign(S.v[idx], st.myerson)
    return assign_type_ids(st.type_space, k, S.priority[idx], S.hw_idx[idx])

def reference_state(model):
    S, ds = model.stream, model.ds
    cfg_h = dataclasses.replace(BASE, phase_transition=dataclasses.replace(BASE.phase_transition, n_min=2))
    idx = np.arange(S.n)
    st, _ = build_window_state(S, idx, idx, int(ds.T), model.state0, ds, cfg_h, fallback=False)
    return st

def job_terms(model):
    """THE job-level reward of every stream job (the online loop's formula, theory.py), vectorised."""
    S, st0 = model.stream, model.state0
    phi = np.asarray(st0.myerson.phi_job(S.v, S.cost_base))
    SW, PI, K = S.priority * S.v, phi - S.C_elec, S.C_carbon
    R_nocarb = L1 * SW + L2 * PI
    return dict(phi=phi, SW=SW, PI=PI, K=K, Cusd=st0.carbon_price * K, R_nocarb=R_nocarb,
                R=R_nocarb - L3 * st0.carbon_price * K)

def decision_of(df):
    rules = [(df["admit"], "accepted"), (df["ir_reject"], "IR reject"), (~df["k_hat"], "oracle reject"),
             (df["k_hat"] & ~df["U"], "thinning reject"), (df["phys_reject"], "physical reject")]
    return np.select([c for c, _ in rules], [d for _, d in rules], "?")

def arrivals_frame(model, res, st_ref, J):
    S = model.stream
    a = pd.DataFrame(res.arrivals)
    i = a["i"].to_numpy()
    for k in ("SW", "PI", "K", "Cusd", "R_nocarb"):
        a[k] = J[k][i]
    a["R_job"] = J["R"][i]                                         # what the loop collects if admitted
    a["decision"] = decision_of(a)
    j_ref = types_under(st_ref, S, i)
    a["k_hat_ref"] = np.where(j_ref >= 0, st_ref.r_tilde[j_ref] - st_ref.v[j_ref] @ st_ref.tau, -np.inf) > 0
    # the one reward check: logged R == re-scored R_job on the admitted set (same function, same phi)
    assert np.allclose(a.loc[a["admit"], "R"], a.loc[a["admit"], "R_job"], rtol=1e-12, atol=1e-12)
    assert np.allclose(a["phi_mech"], J["phi"][i], rtol=1e-12, atol=1e-12)
    assert (a["decision"] != "?").all()
    return a

def offline_opt(model, R, time_limit=300):
    S, c = model.stream, model.c
    cand = np.flatnonzero(R > 0)
    t, D, A = S.tick[cand], S.D[cand], S.A[cand]
    last = np.searchsorted(t, t + D, side="left") - 1              # last row with t_k < t_j + D_j
    cols = np.repeat(np.arange(cand.size), last - np.arange(cand.size) + 1)
    rows = np.concatenate([np.arange(j, last[j] + 1) for j in range(cand.size)])
    M = sp.vstack([sp.csr_matrix((A[cols, r], (rows, cols)), shape=(cand.size, cand.size))
                   for r in range(A.shape[1])]).tocsr()
    ub = np.repeat(c, cand.size)
    lp = linprog(-R[cand], A_ub=M, b_ub=ub, bounds=(0, 1), method="highs")
    ip = milp(-R[cand], constraints=LinearConstraint(M, -np.inf, ub), integrality=np.ones(cand.size),
              bounds=Bounds(0, 1), options=dict(time_limit=time_limit))
    x = np.zeros(S.n)
    x[cand] = np.round(ip.x)
    return dict(x=x.astype(bool), R_ip=float(R @ np.round(x)), R_lp=-float(lp.fun),
                R_nocap=float(R[cand].sum()), gap=getattr(ip, "mip_gap", np.nan), msg=ip.message)

def run_static(model, st, thin_p=None):
    """Same event-driven loop (heap Occupancy, physical gate) with the FIXED (r~, tau, v) of `st`:
    no learning, no tests, no restarts. thin_p = None -> no thinning; else U = u_rand < thin_p
    (same common random numbers as the online run)."""
    S = model.stream
    j = types_under(st, S)
    score = np.where(j >= 0, st.r_tilde[np.maximum(j, 0)] - st.v[np.maximum(j, 0)] @ st.tau, -np.inf)
    khat = score > 0
    U = np.ones(S.n, bool) if thin_p is None else model.u_rand < thin_p
    occ = Occupancy(model.c, record=False)
    adm, phys = np.zeros(S.n, bool), np.zeros(S.n, bool)
    for i in range(S.n):
        t = int(S.tick[i])
        occ.release_until(t)
        if khat[i] and U[i]:
            if occ.fits(S.A[i]):
                occ.add(i, S.A[i], t, int(S.D[i]))
                adm[i] = True
            else:
                phys[i] = True
    return dict(admit=adm, khat=khat, phys=phys)

def occ_series(res, T):
    ev = np.array([(e[0], *e[3:]) for e in res.occupancy_events], float)
    if ev.size == 0:
        return np.array([1]), np.zeros((1, 2))
    te, Ue = ev[:, 0].astype(np.int64), ev[:, 1:]
    assert np.all(np.diff(te) >= 0), "occupancy events must be time-ordered"
    keep = np.r_[te[1:] != te[:-1], True]                         # last level at each tick
    return te[keep], Ue[keep]

def occ_stats(te, Ue, a, b):
    """Time-weighted mean and peak of the occupancy step function on ticks [a, b]."""
    lo, hi = np.searchsorted(te, a, "right"), np.searchsorted(te, b, "right")
    start = Ue[lo - 1] if lo > 0 else np.zeros(Ue.shape[1])
    pts = np.r_[a, te[lo:hi]]
    lev = np.vstack([start, Ue[lo:hi]])
    dur = np.diff(np.r_[pts, b + 1])
    return (lev * dur[:, None]).sum(0) / (b - a + 1), lev.max(0)

def analyse(run):
    model, res = run["model"], run["res"]
    S = model.stream
    st_ref = reference_state(model)
    J = job_terms(model)
    R_job = J["R"]
    a = arrivals_frame(model, res, st_ref, J)
    opt = offline_opt(model, R_job)
    ep = model.ep
    p_cap = (1 - ep.xi) / (1 + ep.eps_D)
    base = {
        "static tau* (full stream, ORACLE)": run_static(model, st_ref),
        "static Phase 0 (H)": run_static(model, model.state0),
        "static Phase 0 (H) + thinning at cap": run_static(model, model.state0, thin_p=p_cap),
    }
    te, Ue = occ_series(res, int(model.ds.T))
    run.update(st_ref=st_ref, J=J, R_job=R_job, A=a, opt=opt, base=base, te=te, Ue=Ue,
               ph=pd.DataFrame(res.phases), p_cap=p_cap)
    assert np.isclose(st_ref.lam_star * model.ds.T, res.summary["bench_global"], rtol=1e-9)
    return run

for s in CAPACITY_SCALES:
    t1 = time.time()
    analyse(RUNS[s])
    o = RUNS[s]["opt"]
    print(f"scale={s:<4} reference + offline OPT + static baselines in {time.time() - t1:.0f} s   "
          f"MILP: {o['msg']} (gap {o['gap']:.2e})")

R0 = RUNS[BASELINE_SCALE]
A0, PH0, M0 = R0["A"], R0["ph"], R0["model"]
S0, SC0 = M0.stream, M0.state0.carbon_price
print(f"\nbaseline scale {BASELINE_SCALE}: s_C = {SC0:.4f} $/kgCO2 (frozen), reward in $ (no divisor)")
print(f"tau per phase (cpu, ram): " + ", ".join(f"m{k + 1}={np.round(st.tau, 6)}" for k, st in enumerate(M0.states)))
print(f"tau* (full-stream hindsight LP) = {R0['st_ref'].tau}")

# %% [markdown]
# ## Section 1: Distributions (C.1, baseline scale)
#
# ### 1.a Bids and virtual values on the stream
# φ_mech = φ under the F̂ of the phase in which the job arrived (what the loop uses); φ_ref = full-stream fit;
# φ_gen = generator's per-job φ (diagnostic only). Histograms clipped to [p1, p99] and binned inside that window;
# dashed lines = p10 / p50 / p90.

# %%
def qhist(ax, x, color, title, w=None, bins=60, logx=False):
    x = np.asarray(x, float)
    w = np.ones_like(x) if w is None else np.asarray(w, float)
    xv = np.log10(np.clip(x, 1e-12, None)) if logx else x
    lo, hi = _weighted_quantile(xv, w, 0.01), _weighted_quantile(xv, w, 0.99)
    ax.hist(xv, bins=bins, range=(lo, hi), weights=w / w.sum(), color=color, alpha=0.85)
    for q, ls in ((0.10, ":"), (0.50, "--"), (0.90, ":")):
        ax.axvline(_weighted_quantile(xv, w, q), color="k", ls=ls, lw=0.9)
    if not logx and lo < 0 < hi:
        ax.axvline(0, color="r", lw=0.8)
    neg = np.average(x < 0, weights=w)
    title = title.replace("$", r"\$")                     # no mathtext between two '$'
    ax.set_title(f"{title}\np50={_weighted_quantile(x, w, .5):.3g}  p90={_weighted_quantile(x, w, .9):.3g}"
                 f"  share<0={100 * neg:.1f}%", fontsize=9)

fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
qhist(ax[0], A0["v"], C_BID, "log10(bid v) [$]", logx=True)
qhist(ax[1], A0["phi_mech"], C_PI, "phi_mech = phi_job(v, c_j) [$] (logged)")
qhist(ax[2], A0["v"] / S0.cost_base[A0["i"]], "teal", "markup v / c_j (known: LogNormal(-s^2/2, s))")
qhist(ax[3], A0["phi_gen"], "gray", "phi_gen(v) [$] (generator, diagnostic)")
plt.tight_layout(); plt.show()

# %% [markdown]
# ### 1.b SW, Π, C$ = s_C·K: job level (stream) and type level (full-stream reference types, weights p_j)

# %%
st_ref0 = R0["st_ref"]
_, prio_ref, _ = type_arrays(st_ref0.type_space, st_ref0.grid)
pop = st_ref0.p > 0
wt = st_ref0.p[pop]
SW_t = (prio_ref * st_ref0.v_bar)[pop]
PI_t = (st_ref0.phi_bar - st_ref0.C_elec_bar)[pop]
C_t = SC0 * st_ref0.C_carbon_bar[pop]

fig, ax = plt.subplots(2, 3, figsize=(14, 6))
qhist(ax[0, 0], A0["SW"], C_SW, "JOB  SW = q*v [$]")
qhist(ax[0, 1], A0["PI"], C_PI, "JOB  Pi = phi_mech - C_elec [$]")
qhist(ax[0, 2], A0["Cusd"], C_C, "JOB  C$ = s_C*K [$]")
qhist(ax[1, 0], SW_t, C_SW, "TYPE SW [$] (w = p_j)", w=wt)
qhist(ax[1, 1], PI_t, C_PI, "TYPE Pi [$] (w = p_j)", w=wt)
qhist(ax[1, 2], C_t, C_C, "TYPE C$ [$] (w = p_j)", w=wt)
plt.tight_layout(); plt.show()

desc = []
for lvl, (sw, pi, cu, w) in {"job": (A0["SW"], A0["PI"], A0["Cusd"], None),
                             "type": (SW_t, PI_t, C_t, wt)}.items():
    for name, x in (("SW", sw), ("Pi", pi), ("C$", cu)):
        x = np.asarray(x, float); ww = np.ones_like(x) if w is None else w
        desc.append({"level": lvl, "objective": name, "min": x.min(),
                     **{f"p{int(q * 100)}": _weighted_quantile(x, ww, q) for q in (.01, .1, .5, .9, .99)},
                     "max": x.max(), "share<0 %": 100 * np.average(x < 0, weights=ww)})
display(pd.DataFrame(desc).set_index(["level", "objective"]).style.format(precision=4))

# %% [markdown]
# ### 1.c Negative base reward R = λ1·SW + λ2·Π − λ3·s_C·K
#
# Job level: R_job of every stream arrival (φ of its phase). Type level: r̃ of the state used in each phase, weighted
# by that state's p̂ (populated types); "reference" = full-stream types weighted by p_ref. "no carbon" drops −λ3·s_C·K.

# %%
neg_rows = [{"level": "job", "scope": "all phases", "n": len(A0),
             "R<0 %": 100 * (A0["R_job"] < 0).mean(), "R<0 no carbon %": 100 * (A0["R_nocarb"] < 0).mean()}]
for m, g in A0.groupby("m"):
    neg_rows.append({"level": "job", "scope": f"phase {m}", "n": len(g),
                     "R<0 %": 100 * (g["R_job"] < 0).mean(),
                     "R<0 no carbon %": 100 * (g["R_nocarb"] < 0).mean()})
for m, st in enumerate(M0.states, start=1):
    pp = st.p > 0
    neg_rows.append({"level": "type", "scope": f"phase {m} state", "n": int(pp.sum()),
                     "R<0 %": 100 * np.average(st.r_tilde[pp] < 0, weights=st.p[pp]),
                     "R<0 no carbon %": 100 * np.average(st.r_nocarbon[pp] < 0, weights=st.p[pp])})
neg_rows.append({"level": "type", "scope": "reference (full stream)", "n": int(pop.sum()),
                 "R<0 %": 100 * np.average(st_ref0.r_tilde[pop] < 0, weights=wt),
                 "R<0 no carbon %": 100 * np.average(st_ref0.r_nocarbon[pop] < 0, weights=wt)})
display(pd.DataFrame(neg_rows).set_index(["level", "scope"]).style.format(precision=2))

neg = A0[A0["R_job"] < 0]
na = neg[neg["admit"]]
print(f"jobs with R_job < 0: {len(neg):,} / {len(A0):,} ({100 * len(neg) / len(A0):.1f}%)  -> admitted: "
      f"{len(na):,} ({100 * len(na) / max(len(neg), 1):.1f}% of the negative ones)")
print("decision of negative-R jobs:", neg["decision"].value_counts().to_dict())
if len(na):
    print("admitted negative-R jobs (the oracle sees the TYPE reward r~, not R_job):")
    print(f"  share with type score r~ - tau^T v > 0      : {100 * (na['score'] > 0).mean():.1f}%")
    print(f"  mean r~_type = {na['r_tilde'].mean():.3f} $   mean R_job = {na['R_job'].mean():.3f} $   "
          f"mean gap R_job - r~ = {(na['R_job'] - na['r_tilde']).mean():.3f} $")
    print(f"  share with Pi_job < 0                        : {100 * (na['PI'] < 0).mean():.1f}%")
    print(f"  share with phi_mech < 0                      : {100 * (na['phi_mech'] < 0).mean():.1f}%")
    print(f"  share with R_job >= 0 without carbon (carbon flips the sign): "
          f"{100 * (na['R_nocarb'] >= 0).mean():.1f}%")

# %% [markdown]
# ## Section 2: Per-phase margin (C.2, baseline scale)
#
# One figure per phase, x = arrival tick (jobs arriving in that phase only). Grey dots: job reward R_job.
# Blue line: rolling median of the type reward r̃_type. Purple line: rolling median of the shadow-price cost τᵀv_type.
# Markers: margin r̃_type − τᵀv_type by final decision. Black line: rolling median of the margin, shaded green
# where > 0 and red where < 0. y axis: symlog (linear within ±1 $). Phases with more than `N_PLOT` arrivals are
# randomly subsampled to `N_PLOT` points (fixed seed); rolling medians use every arrival of the phase.

# %%
N_PLOT = 800
for m, g_all in A0.groupby("m"):
    p = PH0[PH0["m"] == m].iloc[0]
    st = M0.states[m - 1]
    g_all = g_all.sort_values("t")
    w = max(5, len(g_all) // 25) | 1
    roll = lambda col: g_all[col].rolling(w, center=True, min_periods=1).median()
    rm_r, rm_s, rm_m = roll("r_tilde"), roll("shadow_cost"), roll("score")
    g = g_all.sample(N_PLOT, random_state=0).sort_values("t") if len(g_all) > N_PLOT else g_all
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.scatter(g["t"], g["R_job"], s=5, color="lightgray", label="R_job", zorder=1)
    ax.plot(g_all["t"], rm_r, color="steelblue", lw=1.2, label=f"r~_type (rolling median, w={w})")
    ax.plot(g_all["t"], rm_s, color="purple", lw=1.2, label="tau^T v_type (rolling median)")
    ax.plot(g_all["t"], rm_m, color="k", lw=1.6, label="margin (rolling median)")
    mm = rm_m.to_numpy()
    ax.fill_between(g_all["t"], 0, mm, where=mm > 0, color="green", alpha=0.18, interpolate=True)
    ax.fill_between(g_all["t"], 0, mm, where=mm < 0, color="red", alpha=0.18, interpolate=True)
    for d in DEC_ORDER:
        h = g[g["decision"] == d]
        if len(h):
            col, mk = DEC_STYLE[d]
            ax.scatter(h["t"], h["score"].replace(-np.inf, np.nan), s=14, color=col, marker=mk,
                       label=f"margin: {d} ({(g_all['decision'] == d).sum()})", zorder=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("arrival tick"); ax.set_ylabel("$")
    sub = f"  [subsample {len(g)}/{len(g_all)}]" if len(g) < len(g_all) else ""
    ax.set_title(f"Phase {m}: ticks {p['tau_start']:,}-{p['tau_end']:,} ({p['reason']}), {len(g_all)} arrivals{sub}"
                 f"   tau = (cpu {st.tau[0]:.4g}, ram {st.tau[1]:.4g})", fontsize=10)
    ax.legend(fontsize=7, ncol=3, loc="lower left")
    plt.tight_layout(); plt.show()
    zero_tau = np.allclose(st.tau, 0)
    print(f"phase {m}: share margin>0 = {100 * (g_all['score'] > 0).mean():.1f}%   "
          f"median margin = {np.nanmedian(g_all['score'].replace(-np.inf, np.nan)):.3f} $"
          + ("   (tau = 0: shadow cost = 0, margin = r~_type)" if zero_tau else ""))

# %% [markdown]
# ## Section 3: Results per phase (C.3, baseline scale)
#
# Rejections by reason: IR (no type), oracle (r̃ − τᵀv ≤ 0), thinning (oracle yes, U(t) = 0), physical
# (oracle yes, U(t) = 1, U + A > c). Reward = logged dollars; its three parts are λ1·SW, λ2·Π and −λ3·s_C·K of the
# admitted jobs (they sum to the total). Occupancy = exact step function from the event log (heap releases), as a
# fraction of the current capacity, time-weighted over the phase's ticks.

# %%
def phase_table(run):
    a, ph, model = run["A"], run["ph"], run["model"]
    c = model.c
    rows = []
    for _, p in ph.iterrows():
        m = int(p["m"]); g = a[a["m"] == m]; adm = g[g["admit"]]
        mean, peak = occ_stats(run["te"], run["Ue"], int(p["tau_start"]), int(p["tau_end"]))
        st = model.states[m - 1]
        # cross-check against the loop's own per-phase counters
        assert (g["decision"] == "thinning reject").sum() == p["n_thinned"]
        assert (g["decision"] == "physical reject").sum() == p["n_phys_reject"]
        assert len(adm) == p["n_admitted"] and np.isclose(adm["R"].sum(), p["R"])
        rows.append({
            "phase": m, "start": int(p["tau_start"]), "end": int(p["tau_end"]), "steps": int(p["W"]),
            "jobs": len(g), "end reason": p["reason"], "admitted": len(adm),
            "rej IR": int((g["decision"] == "IR reject").sum()),
            "rej oracle": int((g["decision"] == "oracle reject").sum()),
            "rej thinning": int((g["decision"] == "thinning reject").sum()),
            "rej physical": int((g["decision"] == "physical reject").sum()),
            "adm rate %": 100 * len(adm) / max(len(g), 1),
            "R $": adm["R"].sum(), "l1*SW $": L1 * adm["SW"].sum(), "l2*Pi $": L2 * adm["PI"].sum(),
            "-l3*s_C*K $": -L3 * adm["Cusd"].sum(),
            "adm R<0": int((adm["R"] < 0).sum()), "adm R<0 %": 100 * (adm["R"] < 0).mean() if len(adm) else 0.0,
            "occ cpu mean %": 100 * mean[0] / c[0], "occ ram mean %": 100 * mean[1] / c[1],
            "occ cpu peak %": 100 * peak[0] / c[0], "occ ram peak %": 100 * peak[1] / c[1],
            "tau_cpu start": st.tau[0], "tau_ram start": st.tau[1],
        })
    df = pd.DataFrame(rows).set_index("phase")
    tot = df.sum(numeric_only=True)
    g = run["A"]; adm = g[g["admit"]]
    mean, peak = occ_stats(run["te"], run["Ue"], 1, int(model.ds.T))
    tot.update({"start": 1, "end": int(model.ds.T), "adm rate %": 100 * len(adm) / len(g),
                "adm R<0 %": 100 * (adm["R"] < 0).mean(), "occ cpu mean %": 100 * mean[0] / c[0],
                "occ ram mean %": 100 * mean[1] / c[1], "occ cpu peak %": 100 * peak[0] / c[0],
                "occ ram peak %": 100 * peak[1] / c[1], "tau_cpu start": np.nan, "tau_ram start": np.nan})
    df.loc["total"] = tot
    df.loc["total", "end reason"] = ""
    return df

PT0 = phase_table(R0)
display(PT0.style.format(precision=3))
assert np.isclose(PT0.loc["total", "R $"], R0["res"].summary["R_total"])

# %%
def plot_occupancy(ax_cpu, ax_ram, run, title=""):
    model = run["model"]; c = model.c; T = int(model.ds.T)
    te, Ue = run["te"], run["Ue"]
    x = np.r_[1, te, T]
    for ax, i, col, name in ((ax_cpu, 0, C_CPU, "CPU"), (ax_ram, 1, C_RAM, "RAM")):
        y = 100 * np.r_[0.0, Ue[:, i], Ue[-1, i]] / c[i]
        ax.step(x, y, where="post", color=col, lw=0.7)
        ax.axhline(100, color="red", ls="--", lw=0.9)
        for b in run["ph"]["tau_start"].iloc[1:]:
            ax.axvline(b, color="k", ls=":", lw=0.8)
        ax.set_ylim(0, 105); ax.set_xlim(1, T)
        ax.set_ylabel(f"{name} occupancy % of c")
        ax.set_title(f"{title}{name}: c={c[i]:.4g}, mean={100 * np.average(Ue[:-1, i], weights=np.diff(te)) / c[i] if len(te) > 1 else 0:.1f}%"
                     f", peak={100 * Ue[:, i].max() / c[i]:.1f}%", fontsize=9)

fig, ax = plt.subplots(2, 1, figsize=(13, 5.5), sharex=True)
plot_occupancy(ax[0], ax[1], R0, title=f"scale {BASELINE_SCALE} - ")
ax[1].set_xlabel("tick (dotted = phase boundaries, dashed = 100% of capacity)")
plt.tight_layout(); plt.show()

# %%
body = PT0.drop(index="total")
fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
idx = body.index.astype(int)
ax[0].bar(idx, body["admitted"], color="seagreen")
ax[0].set_title("admitted per phase"); ax[0].set_xlabel("phase")
bottom = np.zeros(len(body))
for col, d in (("rej oracle", "oracle reject"), ("rej thinning", "thinning reject"),
               ("rej physical", "physical reject"), ("rej IR", "IR reject")):
    ax[1].bar(idx, body[col], bottom=bottom, color=DEC_STYLE[d][0], label=d)
    bottom += body[col].to_numpy(float)
ax[1].legend(fontsize=8); ax[1].set_title("rejected per phase, by reason"); ax[1].set_xlabel("phase")
wb = 0.2
for k, (col, cc) in enumerate((("R $", "k"), ("l1*SW $", C_SW), ("l2*Pi $", C_PI), ("-l3*s_C*K $", C_C))):
    ax[2].bar(idx + (k - 1.5) * wb, body[col], width=wb, color=cc, label=col)
ax[2].axhline(0, color="k", lw=0.7); ax[2].legend(fontsize=8); ax[2].set_title("reward per phase [$]")
ax[2].set_xlabel("phase")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Section 4: Accuracy per phase (C.4, baseline scale)
#
# **All references are ORACLE quantities** (full stream, true labels; not available online). Phase k runs with the
# state built at the end of phase k−1 (phase 1: Phase 0 on H). Because the valuation grid is refit at every phase,
# each comparison is done **on phase k's own type space**: the full stream is re-assigned to phase k's types to
# build p_ref, d_ref and r_ref (with φ_ref, s_C frozen).
#
# * TV(p̂_k, p_ref): both normalised to sum 1 over types; thin mass = p_ref mass of types with N_j < n_type_min in the
#   data that built the state.
# * d̂ error: |d̂ − d_ref| / d_ref over types with p_ref > 0 (p_ref-weighted mean and median). CI: 95% normal half-width
#   1.96·sd/√M of the observed durations that built the state, relative to their mean, over types with M ≥ 2.
# * KS(F̂_k, F_emp): the phase's bid model F̂_k vs the empirical CDF of all stream MARKUPS m = v / c_j. Under the known
#   distribution F̂_k = LogNormal(−σ²/2, σ) in every phase (never refit), so this measures the fit of the configured
#   markup distribution, not learning. φ error: mean |φ_k(v) − φ(v)|, 0 by construction with one known φ.
# * reward error: p_ref-weighted mean |r̃_k − r_ref| ($ and relative to mean |r_ref|) and sign-mismatch rate.
# * ‖τ̂_k − τ*‖ with τ* the dual of the full-stream hindsight LP.
# * decision agreement: share of phase-k arrivals where k̂ (phase oracle) equals the hindsight oracle
#   1[r_ref − τ*ᵀv_ref > 0] on the reference types; also vs the offline job-level OPT x*.
# * theory (error_params.py, Appendix A): ε_D, η(W_prev), p_U at the phase start/end, CI¹(W), ε̃_Test2(W), ε_Test3(W).

# %%
def accuracy_table(run):
    model, a, ph, st_ref = run["model"], run["A"], run["ph"], run["st_ref"]
    S, ds, ep = model.stream, model.ds, model.ep
    nmin = BASE.phase_transition.n_type_min
    phi_ref = run["J"]["phi"]                                     # the one known phi
    m_sorted = np.sort(S.v / S.cost_base); n = S.n                # markups, the known model's variable
    e_hi, e_lo = np.arange(1, n + 1) / n, np.arange(n) / n
    rows = []
    for k, st in enumerate(model.states, start=1):
        nT = st.n_types
        tid = types_under(st, S); ok = tid >= 0
        N_ref = np.bincount(tid[ok], minlength=nT).astype(float)
        p_ref = N_ref / N_ref.sum()
        p_hat = st.p / st.p.sum()
        if k == 1:
            W_used, n_used = ds.H_span_steps, ds.H.n
            H = ds.H
            tid_o = assign_type_ids(st.type_space, st.grid.assign(st.bids.v_rate, st.myerson), H.priority, H.hw_idx)
            D_o = H.duration_steps
        else:
            prev = ph[ph["m"] == k - 1].iloc[0]
            W_used, n_used = int(prev["W"]), int(prev["n_arrivals"])
            obs = a.loc[(a["m"] == k - 1) & a["admit"], "i"].to_numpy()
            tid_o, D_o = types_under(st, S, obs), S.D[obs]
        N_used = np.rint(st.p * W_used)
        mean_by = lambda x: np.divide(np.bincount(tid[ok], weights=np.asarray(x, float)[ok], minlength=nT),
                                      N_ref, out=np.zeros(nT), where=N_ref > 0)
        d_ref = mean_by(S.D)
        mk = p_ref > 0
        rel_d = np.abs(st.d_bar[mk] - d_ref[mk]) / d_ref[mk]
        oo = tid_o >= 0
        M = np.bincount(tid_o[oo], minlength=nT).astype(float)
        s1 = np.bincount(tid_o[oo], weights=D_o[oo].astype(float), minlength=nT)
        s2 = np.bincount(tid_o[oo], weights=D_o[oo].astype(float) ** 2, minlength=nT)
        mu = np.divide(s1, M, out=np.zeros(nT), where=M > 0)
        var = np.divide(s2 - M * mu ** 2, M - 1, out=np.zeros(nT), where=M > 1)
        ci = np.divide(1.96 * np.sqrt(np.clip(var, 0, None) / np.maximum(M, 1)), mu,
                       out=np.full(nT, np.nan), where=(M > 1) & (mu > 0))
        mc = mk & np.isfinite(ci)
        assert st.myerson.known, "KS on markups assumes bid.known_distribution"
        F_hat = lognorm.cdf(m_sorted, s=st.myerson.sigma, scale=np.exp(st.myerson.mu))
        ks = max(np.abs(F_hat - e_hi).max(), np.abs(F_hat - e_lo).max())
        phi_k = np.asarray(st.myerson.phi_job(S.v, S.cost_base))
        _, prio, A_t = type_arrays(st.type_space, st.grid)
        r_ref, *_ = scalarized_reward(prio, mean_by(S.v), mean_by(phi_ref), mean_by(S.C_elec), mean_by(S.C_carbon),
                                      p_ref, A_t * d_ref[:, None], ds, BASE,
                                      carbon_price=model.state0.carbon_price)
        w = p_ref[mk]
        r_err = np.average(np.abs(st.r_tilde[mk] - r_ref[mk]), weights=w)
        g = a[a["m"] == k]
        W = int(ph.loc[ph["m"] == k, "W"].iloc[0]); W_prev = int(ph.loc[ph["m"] == k, "W_prev"].iloc[0])
        rows.append({
            "phase": k, "n used (arrivals)": n_used, "built from": "H (Phase 0)" if k == 1 else f"phase {k - 1}",
            "TV(p_hat,p_ref)": 0.5 * np.abs(p_hat - p_ref).sum(),
            "thin mass (N_j<n_type_min)": p_ref[N_used < nmin].sum(),
            "d rel err mean": np.average(rel_d, weights=w),
            "d rel err median": _weighted_quantile(rel_d, w, 0.5),
            "d CI rel half-width": np.average(ci[mc], weights=p_ref[mc]) if mc.any() else np.nan,
            "d CI mass covered": p_ref[mc].sum(),
            "KS(F_hat,F_emp)": ks,
            "phi mean abs err $": np.mean(np.abs(phi_k - phi_ref)),
            "phi rel err": np.mean(np.abs(phi_k - phi_ref)) / np.mean(np.abs(phi_ref)),
            "r mean abs err $": r_err,
            "r rel err": r_err / np.average(np.abs(r_ref[mk]), weights=w),
            "r sign mismatch": np.average(np.sign(st.r_tilde[mk]) != np.sign(r_ref[mk]), weights=w),
            "||tau_k - tau*||": float(np.linalg.norm(st.tau - st_ref.tau)),
            "agree hindsight oracle": (g["k_hat"] == g["k_hat_ref"]).mean(),
            "agree offline x*": (g["k_hat"].to_numpy() == run["opt"]["x"][g["i"].to_numpy()]).mean(),
            "eps_D": ep.eps_D, "eta(W_prev)": float(ep.eta(W_prev)),
            "p_U start": ep.p_U(W_prev, 1), "p_U end": ep.p_U(W_prev, W),
            "CI1(W)": float(ep.CI1(W)), "eps_Test2(W)": float(ep.eps_test2(W_prev, W)),
            "eps_Test3(W)": float(ep.eps_test3(W)),
            "max(Test2 lhs - thr)": float((g["test2_lhs"] - g["test2_thr"]).max()) if "test2_lhs" in g else np.nan,
            "min Test3 screen margin": float(g["test3_screen_margin"].min()) if "test3_screen_margin" in g else np.nan,
        })
    return pd.DataFrame(rows).set_index("phase")

ACC0 = accuracy_table(R0)
display(ACC0.T.style.format(precision=4))

# %%
METRICS = ["TV(p_hat,p_ref)", "thin mass (N_j<n_type_min)", "d rel err mean", "d CI rel half-width",
           "KS(F_hat,F_emp)", "phi mean abs err $", "r mean abs err $", "r sign mismatch",
           "||tau_k - tau*||", "agree hindsight oracle", "agree offline x*", "p_U start"]
fig, axes = plt.subplots(3, 4, figsize=(16, 9))
for ax, met in zip(axes.ravel(), METRICS):
    ax.plot(ACC0.index, ACC0[met], "o-", color="steelblue", label=met)
    ax.set_title(met, fontsize=9); ax.set_xlabel("phase")
    tw = ax.twinx()
    tw.bar(ACC0.index, ACC0["n used (arrivals)"], color="gray", alpha=0.2, width=0.6)
    tw.set_yscale("log"); tw.tick_params(labelsize=7, colors="gray")
    if met == "p_U start":
        ax.plot(ACC0.index, ACC0["p_U end"], "s--", color="darkorange", label="p_U end")
        ax.legend(fontsize=7)
    ax.set_zorder(tw.get_zorder() + 1); ax.patch.set_visible(False)
fig.suptitle("Accuracy per phase (twin axis, grey bars: arrivals used to build the phase's state, log scale)",
             fontsize=10)
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(1, 2, figsize=(13, 3.4))
for col, cc in (("eps_Test2(W)", "indianred"), ("eps_Test3(W)", "purple"), ("CI1(W)", "teal"),
                ("eta(W_prev)", "gray")):
    ax[0].plot(ACC0.index, ACC0[col], "o-", color=cc, label=col)
ax[0].set_yscale("log"); ax[0].legend(fontsize=8); ax[0].set_title("theoretical radii (error_params.py)")
ax[0].set_xlabel("phase")
ax[1].plot(ACC0.index, ACC0["max(Test2 lhs - thr)"], "o-", color="indianred", label="max(Test2 lhs - thr)  (>0 fails)")
ax[1].plot(ACC0.index, ACC0["min Test3 screen margin"], "s-", color="purple", label="min Test3 screen margin (<0 screens fail)")
ax[1].axhline(0, color="k", lw=0.8); ax[1].set_yscale("symlog", linthresh=1e-3); ax[1].legend(fontsize=8)
ax[1].set_title("empirical test statistics vs thresholds"); ax[1].set_xlabel("phase")
plt.tight_layout(); plt.show()

# %%
def verdict(df, cols, lower_is_better=True):
    out = []
    for c in cols:
        x = df[c].to_numpy(float)
        d = np.diff(x) if lower_is_better else -np.diff(x)
        d2 = d[1:]
        out.append({"metric": c, "direction checked": "decrease" if lower_is_better else "increase",
                    "phase 1": x[0], "last phase": x[-1],
                    "monotone (all phases)": "yes" if np.all(d <= 1e-12) else "no",
                    "monotone (phases >= 2)": ("yes" if np.all(d2 <= 1e-12) else "no") if len(d2) else "n/a",
                    "last < first": "yes" if (x[-1] < x[0] if lower_is_better else x[-1] > x[0]) else "no"})
    return pd.DataFrame(out).set_index("metric")

V0 = pd.concat([
    verdict(ACC0, ["TV(p_hat,p_ref)", "thin mass (N_j<n_type_min)", "d rel err mean", "d rel err median",
                   "KS(F_hat,F_emp)", "phi mean abs err $", "r mean abs err $", "r rel err", "r sign mismatch",
                   "||tau_k - tau*||"]),
    verdict(ACC0, ["agree hindsight oracle", "agree offline x*"], lower_is_better=False),
])
display(V0.style.format(precision=4))
print(f"metrics monotone over all phases: {(V0['monotone (all phases)'] == 'yes').sum()} / {len(V0)};  "
      f"over phases >= 2: {(V0['monotone (phases >= 2)'] == 'yes').sum()} / {len(V0)};  "
      f"last better than phase 1: {(V0['last < first'] == 'yes').sum()} / {len(V0)}")

# %% [markdown]
# ## Section 5: Baselines and regret (C.5, theoretical setting: true duration known at arrival)
#
# * **DLENT-RT (online)**: the run above. R = Σ R_job over its admitted set (= the logged reward, asserted).
# * **R_OPT fluid (existing benchmark code)**: λ̂_ref · T, the full-stream LP-RS (`summary["bench_global"]`). It is a
#   per-tick fluid quantity on TYPE means, not a feasible schedule.
# * **R_OPT offline job-level (ORACLE)**: MILP above (LP relaxation and no-capacity sum shown as upper bounds).
# * **static τ* (full stream, ORACLE)**: (r_ref, τ*, v_ref) of the full stream, computed once; same event-driven
#   loop, oracle and physical gate; no thinning, no learning, no restarts.
# * **static Phase 0 (H)**: (r̃, τ, v) of Phase 0 (from H), never updated; no thinning.
# * **static Phase 0 (H) + thinning at cap**: same, with U(t) ~ Bern((1 − ξ)/(1 + ε_D)), same random numbers as the
#   online run (isolates the thinning effect from learning).

# %%
def baseline_table(run):
    model, a, opt = run["model"], run["A"], run["opt"]
    S = model.stream; R_job = run["R_job"]
    R_fluid = run["res"].summary["bench_global"]
    alg_adm = np.zeros(S.n, bool); alg_adm[a.loc[a["admit"], "i"].to_numpy()] = True
    assert np.isclose(float(R_job[alg_adm].sum()), run["res"].summary["R_total"], rtol=1e-10)   # logged == re-scored
    algs = {"DLENT-RT (online)": dict(admit=alg_adm, phys=np.isin(np.arange(S.n), a.loc[a["phys_reject"], "i"]))}
    for name, b in run["base"].items():
        algs[name] = dict(admit=b["admit"], phys=b["phys"])
    algs["offline job-level OPT (ORACLE)"] = dict(admit=opt["x"], phys=np.zeros(S.n, bool))
    R_alg = float(R_job[alg_adm].sum())
    rows = []
    for name, d in algs.items():
        adm = d["admit"]; Ry = float(R_job[adm].sum())
        rows.append({"algorithm": name, "admitted": int(adm.sum()), "adm rate %": 100 * adm.mean(),
                     "phys rejects": int(d["phys"].sum()),
                     "R $": Ry, "adm with R_job<0 %": 100 * (R_job[adm] < 0).mean() if adm.any() else 0,
                     "R / R_OPT fluid": Ry / R_fluid, "R / R_OPT job": Ry / opt["R_ip"],
                     "R_ALG / R": R_alg / Ry if Ry else np.nan,
                     "regret vs fluid $": R_fluid - Ry, "regret vs job OPT $": opt["R_ip"] - Ry})
    df = pd.DataFrame(rows).set_index("algorithm")
    bounds = {"R_OPT fluid (bench_global)": R_fluid, "R_OPT job MILP": opt["R_ip"],
              "R_OPT job LP relaxation": opt["R_lp"], "sum of R_job>0 (no capacity)": opt["R_nocap"],
              "R_ALG logged (summary R_total)": run["res"].summary["R_total"],
              "fluid regret (summary regret_global)": run["res"].summary["regret_global"],
              "per-phase fluid bench sum (summary)": run["res"].summary["bench_per_phase"]}
    return df, algs, bounds

BT0, ALGS0, BOUNDS0 = baseline_table(R0)
display(BT0.style.format(precision=4))
for k, v in BOUNDS0.items():
    print(f"{k:<40} {v:14,.3f} $")

# %%
order = np.argsort(S0.tick)
tk = S0.tick[order]
Rr = R0["R_job"][order]
lam_ref = R0["st_ref"].lam_star
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
cum_opt = np.cumsum(Rr * R0["opt"]["x"][order])
for (name, d), cc in zip(ALGS0.items(), ["k", "teal", "steelblue", "darkorange", "seagreen"]):
    if name.startswith("offline"):
        continue
    cum = np.cumsum(Rr * d["admit"][order])
    ax[0].plot(tk, cum_opt - cum, color=cc, lw=1.4 if name.startswith("DLENT") else 0.9, label=name)
    if name.startswith("DLENT"):
        ax[1].plot(tk, lam_ref * tk - cum, color="k", lw=1.4, label="DLENT-RT vs fluid lam_ref*t")
        ax[1].plot(tk, cum_opt - cum, color="k", ls="--", lw=1.0, label="DLENT-RT vs offline job OPT")
for a_ in ax:
    for b in PH0["tau_start"].iloc[1:]:
        a_.axvline(b, color="gray", ls=":", lw=0.8)
    a_.axhline(0, color="k", lw=0.6); a_.set_xlabel("tick (dotted = DLENT phase boundaries)"); a_.set_ylabel("$")
    a_.legend(fontsize=7)
ax[0].set_title("cumulative regret vs offline job-level OPT (R_job)", fontsize=10)
ax[1].set_title("DLENT-RT cumulative regret: fluid vs job-level benchmark", fontsize=10)
plt.tight_layout(); plt.show()

# %%
def per_phase_regret(run, algs):
    a, ph = run["A"], run["ph"]
    R_job, x = run["R_job"], run["opt"]["x"]
    rows = []
    for _, p in ph.iterrows():
        m = int(p["m"]); ii = a.loc[a["m"] == m, "i"].to_numpy()
        opt_m = float(R_job[ii][x[ii]].sum())
        row = {"phase": m, "R_OPT job (phase) $": opt_m,
               "DLENT fluid regret (code: bench_hindsight - R) $": float(p["regret_hindsight"])}
        for name, d in algs.items():
            if name.startswith("offline"):
                continue
            row[f"regret {name} $"] = opt_m - float(R_job[ii][d["admit"][ii]].sum())
        rows.append(row)
    df = pd.DataFrame(rows).set_index("phase")
    df.loc["total"] = df.sum()
    return df

PR0 = per_phase_regret(R0, ALGS0)
display(PR0.style.format(precision=3))

# %% [markdown]
# ## Section 6: capacity_scale sweep
#
# Same seeds, same stream, same BASE config; only c is multiplied before Phase 0. ξ_used is the forced
# `constants.xi_override`; ξ_data, the thinning cap and the gate change with the scale (Section 0.1).

# %%
sweep = []
for s, run in RUNS.items():
    sm, ep, c = run["res"].summary, run["model"].ep, run["model"].c
    a = run["A"]
    bt, _, bd = baseline_table(run)
    mean, peak = occ_stats(run["te"], run["Ue"], 1, int(run["model"].ds.T))
    sweep.append({
        "capacity_scale": s, "c_cpu": c[0], "c_ram": c[1], "xi_data": run["ds"].xi, "xi_used": ep.xi,
        "p_U cap": run["p_cap"], "p_U min": a["p_U"].min(), "p_U mean": a["p_U"].mean(),
        "phys gate: jobs A>c": int(np.any(run["model"].stream.A > c, axis=1).sum()),
        "phases": sm["n_phases"], "end reasons": ", ".join(f"{k}:{v}" for k, v in sm["phase_end_reasons"].items() if v),
        "admitted": sm["n_admitted"], "adm rate %": 100 * sm["n_admitted"] / sm["n_arrivals"],
        "rej oracle": int((a["decision"] == "oracle reject").sum()),
        "rej thinning": int((a["decision"] == "thinning reject").sum()),
        "rej physical": int((a["decision"] == "physical reject").sum()),
        "occ cpu mean %": 100 * mean[0] / c[0], "occ ram mean %": 100 * mean[1] / c[1],
        "occ cpu peak %": 100 * peak[0] / c[0], "occ ram peak %": 100 * peak[1] / c[1],
        "R_ALG $": sm["R_total"],
        "R_OPT fluid $": bd["R_OPT fluid (bench_global)"], "R_OPT job $": bd["R_OPT job MILP"],
        "R_ALG/R_OPT fluid": bt.loc["DLENT-RT (online)", "R / R_OPT fluid"],
        "R_ALG/R_OPT job": bt.loc["DLENT-RT (online)", "R / R_OPT job"],
        "R_ALG/R_static tau*": bt.loc["static tau* (full stream, ORACLE)", "R_ALG / R"],
        "R_ALG/R_static P0": bt.loc["static Phase 0 (H)", "R_ALG / R"],
        "regret vs job OPT $": bt.loc["DLENT-RT (online)", "regret vs job OPT $"],
        "regret vs fluid $": bt.loc["DLENT-RT (online)", "regret vs fluid $"],
        "adm R_job<0 % (ALG)": bt.loc["DLENT-RT (online)", "adm with R_job<0 %"],
        "tau* (hindsight)": np.round(run["st_ref"].tau, 6),
    })
SWEEP = pd.DataFrame(sweep).set_index("capacity_scale")
display(SWEEP.T.style.format(precision=4))

# %%
fig, axes = plt.subplots(len(RUNS), 2, figsize=(14, 2.3 * len(RUNS)), sharex=True, squeeze=False)
for (s, run), (a1, a2) in zip(RUNS.items(), axes):
    plot_occupancy(a1, a2, run, title=f"scale {s} - ")
for a_ in axes[-1]:
    a_.set_xlabel("tick")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Section 7: Anomalies (numbers only)

# %%
an = []
# 1. |H| and s_C depend on data.n0 (this notebook and default.yaml: n0 = 10,000)
for n0 in sorted({10_000, BASE.data.n0}):
    c_n = apply_overrides(BASE, {"data.n0": n0})
    ds_n = load_dataset(c_n); st_n = run_phase0(ds_n, c_n)
    an.append({"item": f"s_C with data.n0={n0:,}", "value": f"{st_n.carbon_price:.3f} $/kgCO2 "
               f"(|H|={ds_n.H.n:,}, |stream|={ds_n.stream.n:,}, K_q={st_n.norm_info['K_q']:.4g}, "
               f"D_q={st_n.norm_info['D_q']:.4g})"})
# 2. negative payments (payment = phi_tilde of the type's bin floor; lognormal: phi(L) can be < 0)
adm = A0[A0["admit"]]
negp = adm[adm["payment"] < 0]
an.append({"item": "admitted jobs with payment < 0",
           "value": f"{len(negp):,} / {len(adm):,} ({100 * len(negp) / max(len(adm), 1):.1f}%), sum = "
                    f"{negp['payment'].sum():,.2f} $; per phase {negp.groupby('m').size().to_dict()}; "
                    f"bins k {sorted(negp['k_bin'].unique().tolist())[:12]}{'...' if negp['k_bin'].nunique() > 12 else ''}"})
an.append({"item": "admitted jobs with payment > bid v", "value": f"{int((adm['payment'] > adm['v']).sum()):,}"})
for m, st in enumerate(M0.states, start=1):
    pp = st.p > 0
    an.append({"item": f"phase {m} state: populated types with phi_tilde < 0",
               "value": f"{int((st.phi_tilde[pp] < 0).sum())} / {int(pp.sum())} "
                        f"(p-mass {100 * st.p[pp][st.phi_tilde[pp] < 0].sum() / st.p[pp].sum():.1f}%)"})
# 3. jobs bigger than the cluster
big = np.any(S0.A > M0.c, axis=1)
an.append({"item": "xi_data = max a_max/c (scale 1)", "value": f"{R0['ds'].xi:.4f}  (forced xi_used = {M0.ep.xi})"})
an.append({"item": "stream jobs with A_i > c_i (never admissible)", "value": f"{int(big.sum()):,} ({100 * big.mean():.2f}%)"})
# 4. shadow prices and fluid LP load (only r~ > 0 types can enter the LP-RS optimum)
for m, st in enumerate(M0.states + [R0["st_ref"]], start=1):
    pos = st.r_tilde > 0
    load = (st.p[pos, None] * st.v[pos]).sum(0) / M0.c
    name = f"phase {m} state" if m <= len(M0.states) else "reference (full stream)"
    an.append({"item": f"{name}: tau, fluid load sum_(r~>0) p_j v_j / c",
               "value": f"tau={np.round(st.tau, 6)}, load={np.round(load, 4)}"})
pr = R0["st_ref"].p > 0
an.append({"item": "reference types: populated / stream jobs",
           "value": f"{int(pr.sum()):,} / {S0.n:,} ({S0.n / pr.sum():.2f} jobs per populated type)"})
# 5. tests and gamma
mx = max(abs(st.r_tilde[st.p > 0]).max() for st in M0.states)
an.append({"item": "max |r~| over populated types (all phase states)", "value": f"{mx:.3f} $ vs gamma_used = {M0.ep.gamma}"})
an.append({"item": "phase end reasons", "value": str(R0["res"].summary["phase_end_reasons"])})
an.append({"item": "max over phases of (Test2 lhs - thr)", "value": f"{ACC0['max(Test2 lhs - thr)'].max():.4g}"})
an.append({"item": "min over phases of Test3 screen margin", "value": f"{ACC0['min Test3 screen margin'].min():.4g}"})
# 6. reward bases
an.append({"item": "R_ALG logged (summary R_total) vs re-scored with R_job",
           "value": f"{R0['res'].summary['R_total']:,.2f} vs {BT0.loc['DLENT-RT (online)', 'R $']:,.2f} $ (asserted equal)"})
an.append({"item": "R_OPT fluid vs offline job-level MILP",
           "value": f"{BOUNDS0['R_OPT fluid (bench_global)']:,.2f} vs {BOUNDS0['R_OPT job MILP']:,.2f} $"})
with pd.option_context("display.max_colwidth", None):
    display(pd.DataFrame(an).set_index("item"))

# %% [markdown]
# ## Export (week-29 report)
#
# Writes figures (PNG, dpi 200) to `reports/week30/figures/` and tables (CSV) to `reports/week30/tables/`.
# Figures are re-rendered by re-running the plotting cells above on the results already in memory, with
# `plt.show` redirected to `savefig` (nothing is recomputed). Tables are the DataFrames built above; Tables 3, 7 and
# the admitted-R<0 summary are arithmetic on those results.

# %%
import io, contextlib
REPORT_DIR = ROOT / "reports" / "week30"
FIG_DIR, TAB_DIR = REPORT_DIR / "figures", REPORT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "notebooks" / "src"))
from build import parse as _parse_src
_SRC = [c.source for c in _parse_src((ROOT / "notebooks" / "src" / "03_main_algorithm_analysis.py").read_text())
        if c.cell_type == "code"]

# (file stem, unique marker of the plotting cell, names for successive plt.show calls or None = numbered)
FIG_SPECS = [
    ("fig01_bids_phi", 'qhist(ax[0], A0["v"]', ["fig01_bids_phi"]),
    ("fig02_objectives_job_type", 'qhist(ax[0, 0], A0["SW"]', ["fig02_objectives_job_type"]),
    ("fig03_margin_phase", "N_PLOT = 800", None),
    ("fig04_occupancy", "plot_occupancy(ax[0], ax[1], R0", ["fig04_occupancy"]),
    ("fig05_phase_bars", "body = PT0.drop", ["fig05_phase_bars"]),
    ("fig06_accuracy", "METRICS = [", ["fig06_accuracy", "fig06b_theory_radii"]),
    ("fig07_cumulative_regret", "cum_opt = np.cumsum", ["fig07_cumulative_regret"]),
    ("fig08_occupancy_sweep", "zip(RUNS.items(), axes)", ["fig08_occupancy_sweep"]),
]
_real_show, _real_display = plt.show, display
saved_figs = []
for base, marker, names in FIG_SPECS:
    src = next(s for s in _SRC if marker in s and "FIG_SPECS" not in s)
    count = [0]
    def _save_show(*_a, _base=base, _names=names, _count=count, **_k):
        _count[0] += 1
        name = _names[_count[0] - 1] if _names else f"{_base}{_count[0]}"
        fig = plt.gcf()
        fig.savefig(FIG_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_figs.append(name)
    plt.show = _save_show
    globals()["display"] = lambda *_a, **_k: None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(src, globals())
    finally:
        plt.show = _real_show
        globals()["display"] = _real_display
plt.close("all")

def _csv(df, name, **kw):
    df.to_csv(TAB_DIR / f"{name}.csv", **kw)
    return name

ep0, sm0 = M0.ep, R0["res"].summary
setup = pd.Series({
    "trace": Path(BASE.data.batch_csv).name, "grid data": Path(BASE.data.grid_csv).name,
    "data.n0 (notebook override)": BASE.data.n0,
    "seed run (Phase-0 bids)": BASE.run.seed, "seed stream bids": BASE.run.seed + BASE.online.bid_seed_offset,
    "seed thinning": BASE.run.seed + BASE.online.thinning_seed_offset,
    "|H|": ds0.H.n, "|stream|": ds0.stream.n, "T (ticks)": ds0.T, "tick seconds": BASE.time.tick_seconds,
    "d_max (ticks)": ds0.d_max_steps, "c_cpu": ds0.capacity[0], "c_ram": ds0.capacity[1],
    "capacity mode": BASE.capacity.mode, "capacity fraction": BASE.capacity.fraction,
    "capacity concurrency_quantile": BASE.capacity.concurrency_quantile,
    "a_max_cpu": ds0.a_max[0], "a_max_ram": ds0.a_max[1], "xi_data": ds0.xi, "xi_used": ep0.xi,
    "gamma_used": ep0.gamma, "gamma_data": ep0.gamma_data, "delta": ep0.delta, "eps_D": ep0.eps_D,
    "thinning cap (1-xi)/(1+eps_D)": R0["p_cap"], "p_U min": A0["p_U"].min(), "p_U max": A0["p_U"].max(),
    "lambda1": L1, "lambda2": L2, "lambda3": L3, "lambda sum": L1 + L2 + L3,
    "s_C ($/kgCO2, frozen)": SC0, "normalization": "quantile_linear",
    "quantile q": BASE.normalization.quantile,
    "bid strategy": BASE.bid.strategy, "bid sigma": BASE.bid.sigma, "u0": BASE.bid.base_utility_u0,
    "k_star": BASE.grid.k_star, "hw profiles": BASE.hardware.n_profiles, "model": BASE.model.kind,
    "n_type_min": BASE.phase_transition.n_type_min, "n_min": BASE.phase_transition.n_min,
}, name="value")

t3 = pd.DataFrame([{"phase": m, "tau_cpu": st.tau[0], "tau_ram": st.tau[1],
                    "arrivals": int((A0["m"] == m).sum()),
                    "points plotted": int(min(N_PLOT, (A0["m"] == m).sum())),
                    "subsampled": bool((A0["m"] == m).sum() > N_PLOT)}
                   for m, st in enumerate(M0.states, start=1)]).set_index("phase")

_neg = A0[A0["R_job"] < 0]
_na = _neg[_neg["admit"]]
adm_neg = pd.Series({
    "arrivals": len(A0), "R_job<0": len(_neg), "R_job<0 %": 100 * len(_neg) / len(A0),
    "R_nocarb<0 %": 100 * (A0["R_nocarb"] < 0).mean(),
    "admitted": int(A0["admit"].sum()), "admitted with R_job<0": len(_na),
    "admitted R_job<0 % of admitted": 100 * len(_na) / A0["admit"].sum(),
    "admitted R_job<0 % of negative jobs": 100 * len(_na) / max(len(_neg), 1),
    "of admitted R_job<0: score>0 %": 100 * (_na["score"] > 0).mean(),
    "of admitted R_job<0: Pi_job<0 %": 100 * (_na["PI"] < 0).mean(),
    "of admitted R_job<0: phi_mech<0 %": 100 * (_na["phi_mech"] < 0).mean(),
    "of admitted R_job<0: R_nocarb>=0 % (carbon flips sign)": 100 * (_na["R_nocarb"] >= 0).mean(),
    "of admitted R_job<0: mean r~_type $": _na["r_tilde"].mean(),
    "of admitted R_job<0: mean R_job $": _na["R_job"].mean(),
    "admitted with payment<0": int((A0.loc[A0["admit"], "payment"] < 0).sum()),
}, name="value")

_Ry = BT0["R $"]
_opt, _p0 = _Ry["offline job-level OPT (ORACLE)"], _Ry["static Phase 0 (H)"]
_p0t, _alg = _Ry["static Phase 0 (H) + thinning at cap"], _Ry["DLENT-RT (online)"]
_tot = _opt - _alg
t7 = pd.DataFrame([
    {"step": "OPT (job-level) -> static Phase 0", "attributed to": "discretization / type aggregation (Phase-0 types, no learning)", "$": _opt - _p0},
    {"step": "static Phase 0 -> + thinning at cap", "attributed to": "thinning", "$": _p0 - _p0t},
    {"step": "+ thinning at cap -> DLENT-RT online", "attributed to": "learning (ReInit) + early-phase thinning below the cap", "$": _p0t - _alg},
    {"step": "total regret (OPT - online)", "attributed to": "", "$": _tot},
]).set_index("step")
t7["% of total"] = 100 * t7["$"] / _tot
t7b = PR0.drop(index="total").copy()
t7b.insert(0, "arrivals", A0.groupby("m").size())
t7b["DLENT regret per arrival $"] = t7b["regret DLENT-RT (online) $"] / t7b["arrivals"]
t7b.loc["total"] = t7b.sum()
t7b.loc["total", "DLENT regret per arrival $"] = t7b.loc["total", "regret DLENT-RT (online) $"] / t7b.loc["total", "arrivals"]

t8 = SWEEP[["c_cpu", "c_ram", "xi_data", "phys gate: jobs A>c", "p_U min", "phases", "end reasons", "admitted",
            "adm rate %", "rej physical", "rej thinning", "occ cpu mean %", "occ ram mean %", "occ cpu peak %",
            "occ ram peak %", "R_ALG $", "R_OPT job $", "R_ALG/R_OPT job",
            "R_OPT fluid $", "R_ALG/R_OPT fluid", "tau* (hindsight)"]].copy()
t8["tau* (hindsight)"] = t8["tau* (hindsight)"].astype(str)

saved_tabs = [
    _csv(setup.to_frame(), "table00_setup"),
    _csv(const_df.T, "table00b_constants_per_scale"),
    _csv(pd.DataFrame(desc).set_index(["level", "objective"]), "table01_objective_quantiles"),
    _csv(pd.DataFrame(neg_rows).set_index(["level", "scope"]), "table02_negative_reward"),
    _csv(t3, "table03_phase_tau_points"),
    _csv(PT0, "table04_phase_results"),
    _csv(adm_neg.to_frame(), "table04b_admitted_negative"),
    _csv(ACC0, "table05_accuracy"),
    _csv(V0, "table05b_monotonicity"),
    _csv(BT0, "table06_baselines"),
    _csv(pd.Series(BOUNDS0, name="$").to_frame(), "table06b_benchmarks"),
    _csv(t7, "table07_regret_decomposition"),
    _csv(t7b, "table07b_regret_per_phase"),
    _csv(t8, "table08_capacity_sweep"),
    _csv(pd.DataFrame(an).set_index("item"), "table10_anomalies"),
]
print(f"figures -> {FIG_DIR}:")
print("  " + "\n  ".join(saved_figs))
print(f"tables -> {TAB_DIR}:")
print("  " + "\n  ".join(saved_tabs))

# %% [markdown]
# # DLENT-RT — Phase 0 Diagnostics
#
# Understand the historical trace's distributions, validate the on-the-fly bid
# synthesis and the energy/grid cost model, and inspect Phase 0's per-type
# statistics, normalized objectives, and LP-RS admission oracle for the
# configuration set in `OVERRIDES` below.
#
# Sections: 0. Setup and configuration &middot; 1. Bid synthesis &middot;
# 2. Energy model and grid data &middot; 3. Phase 0 full output &middot;
# 4. Objective analysis &middot; 5. Order of magnitude across 20 seeds.

# %% [markdown]
# ## Section 0: Setup and Configuration

# %%
import sys, warnings, dataclasses, random
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from dlent_rt.data import load_dataset
warnings.simplefilter("ignore")
sys.path.insert(0, str(Path.cwd().parent))

from dlent_rt.config import default_config
from dlent_rt.data import load_dataset
from dlent_rt.phase0 import (run_phase0, _compute_job_energy,
    _compute_job_power, _synthesize_bids, _lookup_grid)
from dlent_rt.bid import (synthesize_lognormal, LognormalBidParams,
    compute_phi_tilde, MyersonModel)
from dlent_rt.lp_oracle import solve_lp_rs
from dlent_rt.energy import load_grid_data
from dlent_rt.grid import ValuationGrid, build_valuation_grid

plt.rcParams["figure.figsize"] = (13, 3.6)
plt.rcParams["figure.dpi"] = 110

cfg = default_config()
print("=== active configuration ===")

def print_config(c):
    """Print every config field, section by section, up to two levels deep."""
    for f in dataclasses.fields(c):
        val = getattr(c, f.name)
        if dataclasses.is_dataclass(val):
            print(f"[{f.name}]")
            for sf in dataclasses.fields(val):
                sval = getattr(val, sf.name)
                if dataclasses.is_dataclass(sval):
                    print(f"  {sf.name}:")
                    for ssf in dataclasses.fields(sval):
                        print(f"    {ssf.name} = {getattr(sval, ssf.name)}")
                else:
                    print(f"  {sf.name} = {sval}")
        else:
            print(f"{f.name} = {val}")
    print()
    
print_config(cfg)

ds = load_dataset(cfg)


print(f"H jobs:      {ds.H.n}")
print(f"stream jobs: {ds.stream.n}")
print(f"T (ticks):   {ds.T:,}")
print(f"H time range:     {ds.H.datetime.min()} -> {ds.H.datetime.max()}")
print(f"stream starts at: {ds.stream.datetime.min()}")
print(f"capacity: {np.round(ds.capacity, 3)}  (resources={ds.resources})")

# %% [markdown]
# ### 0.1 Cluster occupancy over time and initial cluster size
#
# Occupancy of the cluster at every tick of the trace (H + stream on one shared absolute timeline, the same basis
# `load_dataset` uses to size the capacity). A job arriving at tick $t_j$ with duration $D_j$ occupies ticks
# $t_j, \dots, t_j + D_j - 1$; the occupancy is $U_i(t) = \sum_{j:\, t_j \le t < t_j + D_j} A_{ij}$, computed exactly
# from the arrival/release events (no per-tick scan) and shown as a percentage of the initial capacity $c_i$.
#
# Phase 0 makes no admission decisions, so this is the load the trace *offers* to the cluster (every job counted):
# values above 100% are times when the requested resources exceed $c_i$. The capacity itself is
# `capacity.fraction` × the `capacity.concurrency_quantile` time-weighted quantile of this same series.

# %%
from dlent_rt.data import _assign_real_tick_steps, _occupancy_events, _time_weighted_quantile

# absolute tick of every job on the shared H + stream timeline (t0 = first H arrival), as in load_dataset
all_dt = np.concatenate([ds.H.datetime, ds.stream.datetime])
all_steps = _assign_real_tick_steps(all_dt, all_dt[0], cfg.time.tick_seconds)
assert np.array_equal(all_steps[:ds.H.n], ds.H.arrival_step), "H timeline mismatch"
assert np.array_equal(all_steps[ds.H.n:] - all_steps[ds.H.n], ds.stream.arrival_step), "stream timeline mismatch"

basis = dataclasses.make_dataclass("Basis", ["arrival_step", "duration_steps", "A"])(
    all_steps, np.concatenate([ds.H.duration_steps, ds.stream.duration_steps]),
    np.concatenate([ds.H.A, ds.stream.A]))
occ_t, occ_U = _occupancy_events(basis)                 # occ_U[k] holds on [occ_t[k], occ_t[k+1])
c0 = np.asarray(ds.capacity, float)
pct = 100 * occ_U / c0
ts = all_dt[0] + (occ_t * cfg.time.tick_seconds * 1e3).astype("timedelta64[ms]")   # tick -> timestamp
hold = np.diff(occ_t).astype(float)                     # ticks each level is held

# --- initial cluster size ---
spec = cfg.energy_model
levels_q = _time_weighted_quantile(occ_U[:-1], hold, cfg.capacity.concurrency_quantile)
print("=== initial cluster size c (Phase 0, before any online decision) ===")
print(f"sizing: mode={cfg.capacity.mode}  fraction={cfg.capacity.fraction}  "
      f"concurrency_quantile={cfg.capacity.concurrency_quantile}")
for i, (name, unit, k) in enumerate((("CPU", "cores", spec.spec_cpu_core), ("RAM", "GB", spec.spec_ram_gb))):
    print(f"  {name}: c = {c0[i]:.5f} normalized  = {c0[i] * k:,.1f} {unit}   "
          f"(q{cfg.capacity.concurrency_quantile:g} of the occupancy = {levels_q[i]:.5f}; "
          f"peak = {occ_U[:, i].max():.5f} = {100 * occ_U[:, i].max() / c0[i]:.0f}% of c)")
print(f"  ({spec.spec_cpu_core:g} cores per 1.0 normalized CPU, {spec.spec_ram_gb:g} GB per 1.0 normalized RAM)")

occ_summary = pd.DataFrame({
    "c (normalized)": c0, "c (physical)": [f"{c0[0] * spec.spec_cpu_core:,.1f} cores", f"{c0[1] * spec.spec_ram_gb:,.1f} GB"],
    "mean occupancy %": [100 * np.average(occ_U[:-1, i], weights=hold) / c0[i] for i in range(2)],
    "median occupancy %": [100 * _time_weighted_quantile(occ_U[:-1, [i]], hold, 0.5)[0] / c0[i] for i in range(2)],
    "peak occupancy %": pct.max(axis=0),
    "time above 100% of c": [f"{100 * hold[pct[:-1, i] > 100].sum() / hold.sum():.1f}%" for i in range(2)],
}, index=["CPU", "RAM"])
display(occ_summary.round(2))

# --- occupancy over time ---
t_split = ds.stream.datetime.min()
fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
for ax, i, col, name in ((axes[0], 0, "steelblue", "CPU"), (axes[1], 1, "darkorange", "RAM")):
    ax.step(ts, pct[:, i], where="post", color=col, lw=0.6)
    ax.axhline(100, color="red", ls="--", lw=1.0, label=f"100% of c = {c0[i]:.4f}")
    ax.axvline(t_split, color="k", ls=":", lw=1.0, label="H | stream boundary")
    ax.set_ylim(bottom=0)
    ax.set_ylabel(f"{name} occupancy [% of c]")
    ax.set_title(f"{name}: c = {c0[i]:.4f} normalized, mean {occ_summary.loc[name, 'mean occupancy %']:.0f}%, "
                 f"peak {pct[:, i].max():.0f}%", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
axes[1].set_xlabel("timestamp (request_time)")
fig.autofmt_xdate()
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Section 1: Bid Synthesis

# %%
# Lognormal bid synthesis across a small grid of (sigma, base_utility)
LOGNORMAL_CONFIGS = [
    {"sigma": 0.3,  "base_utility_u0": 23.93},
    {"sigma": 0.5,  "base_utility_u0": 23.93},
]

rows = []
lognormal_bids = []
for lc in LOGNORMAL_CONFIGS:
    params = LognormalBidParams(
        sigma=lc["sigma"], base_utility_u0=lc["base_utility_u0"],
        base_utility_mode=cfg.bid.base_utility_mode,
        gamma1=cfg.bid.gamma1, gamma2=cfg.bid.gamma2,
        spec_cpu_core=cfg.energy_model.spec_cpu_core,
        spec_ram_gb=cfg.energy_model.spec_ram_gb,
    )
    b = synthesize_lognormal(
        ds.H.A[:, 0], ds.H.A[:, 1], ds.H.duration_hours, ds.H.priority,
        params, np.random.default_rng(42),
    )
    lognormal_bids.append((lc, b))
    rows.append({
        "sigma": lc["sigma"], "base_utility_u0": lc["base_utility_u0"],
        "bid_min": b.v_rate.min(), "bid_max": b.v_rate.max(),
        "bid_mean": b.v_rate.mean(),
        "phi<0%": 100 * (b.phi_rate < 0).mean(), "phi_mean": b.phi_rate.mean(),
    })
lognorm_df = pd.DataFrame(rows).round(4)
display(lognorm_df)

PHI_XLIM = (-150, 150)

fig, axes = plt.subplots(len(LOGNORMAL_CONFIGS), 2,
                          figsize=(13, 3.2 * len(LOGNORMAL_CONFIGS)))
for row, (lc, b) in enumerate(lognormal_bids):
    a0, a1 = axes[row]

    log_bid = np.log10(b.v_rate.clip(1e-12))
    a0.hist(log_bid, bins=50, color="steelblue")
    lo0, hi0 = np.percentile(log_bid, [1, 99])
    margin0 = (hi0 - lo0) * 0.1
    a0.set_xlim(lo0 - margin0, hi0 + margin0)
    a0.set_title(f"sigma={lc['sigma']}, base_utility_u0={lc['base_utility_u0']}\nlog10(bid)")

    # Bin WITHIN the visible window (range=PHI_XLIM), not over the full
    # phi_rate range: with fixed bins over the full range (which extends far
    # beyond +-150), only 2-3 wide bins would ever land inside the zoomed
    # window, hiding the actual shape of the central mass.
    a1.hist(b.phi_rate, bins=80, range=PHI_XLIM, color="darkorange")
    a1.axvline(0, color="r", ls="--")
    a1.set_xlim(-40, 80)
    a1.set_title(f"phi(bid), {100*(b.phi_rate<0).mean():.1f}% negative")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Section 2: Energy Model and Grid Data

# %% [markdown]
# Convert each job's normalized CPU/RAM request (`ds.H.A[:, 0]`, `ds.H.A[:, 1]`)
# into physical power (kW) and total energy (kWh) via the reference machine
# spec, then look up the electricity price and carbon intensity in effect at
# each job's `request_time` from the historical grid CSV
# (`cfg.data.grid_csv`). These feed directly into `C_elec_bar` and
# `C_carbon_bar` in Phase 0.

# %%
energy = _compute_job_energy(ds.H, cfg)
power = _compute_job_power(ds.H, cfg)

fig, ax = plt.subplots(1, 3)
ax[0].hist(np.log10(energy.clip(1e-12)), bins=60, color="indianred")
ax[0].set_title(f"log10(energy_kWh)\nmean={energy.mean():.4f}  max={energy.max():.2f}")
ax[1].hist(np.log10(power.clip(1e-12)), bins=60, color="teal")
ax[1].set_title(f"log10(power_kW)\nmean={power.mean():.6f}  max={power.max():.4f}")
ax[2].scatter(ds.H.A[:500, 0], power[:500], s=2, alpha=0.5)
ax[2].set_xlabel("CPU (normalized, A[:,0])"); ax[2].set_ylabel("power (kW)")
ax[2].set_title("Power vs CPU request")
plt.tight_layout(); plt.show()

print(f"energy kWh: min={energy.min():.8f}  max={energy.max():.4f}  mean={energy.mean():.4f}")
print(f"power kW:   min={power.min():.8f}  max={power.max():.6f}  mean={power.mean():.6f}")

# %%
# grid_data is loaded inside load_dataset() from cfg.data.grid_csv; re-load it
# here explicitly to confirm the path is not hardcoded anywhere downstream.
from dlent_rt.config import resolve_path
grid_data_check = load_grid_data(str(resolve_path(cfg.data.grid_csv)))   # repo-relative path
assert grid_data_check.timestamps.shape == ds.grid_data.timestamps.shape

elec_price, carbon_int = _lookup_grid(ds.H, ds)

fig, ax = plt.subplots(1, 2)
ax[0].hist(elec_price, bins=40, color="gold")
ax[0].set_title(f"Electricity price at request_time\nmean={elec_price.mean():.4f} $/kWh")
ax[1].hist(carbon_int, bins=40, color="gray")
ax[1].set_title(f"Carbon intensity at request_time\nmean={carbon_int.mean():.4f} kgCO2/kWh")
plt.tight_layout(); plt.show()

print(f"grid_csv: {cfg.data.grid_csv}")
print(f"grid data time range: {ds.grid_data.timestamps.min()} -> {ds.grid_data.timestamps.max()}")
print(f"H request_time range: {ds.H.datetime.min()} -> {ds.H.datetime.max()}")
print(f"price range: [{ds.grid_data.price_min:.4f}, {ds.grid_data.price_max:.4f}] $/kWh")
print(f"CI range:    [{ds.grid_data.ci_min:.4f}, {ds.grid_data.ci_max:.4f}] kgCO2/kWh")

# %% [markdown]
# ### 2.5 quantile_linear normalization (Phase 0)
#
# Per populated type $j$ (weights $p_j = N_j/N$): $CV_j = q_j \bar v_j$ [\$], $\Pi_j = \bar\phi_j - \bar C^{elec}_j$ [\$],
# $K_j = \overline{E \cdot CI}_j$ [kgCO2]. Carbon is mapped to dollars linearly through the origin, $C\$_j = s_C K_j$, with
#
# $$D_q = \tfrac12\big(Q_q(CV) + Q_q(|\Pi|)\big),\qquad K_q = Q_q(K),\qquad s_C = D_q / K_q \;[\$/\text{kgCO2}],$$
#
# weighted quantiles (weights $p_j$, same $q$ for the three objectives). Reward in dollars, no common divisor:
# $\tilde R_j = \lambda_1 CV_j + \lambda_2 \Pi_j - \lambda_3 s_C K_j$. Oracle: $\hat k_j = 1[\tilde R_j - \tau^\top v_j > 0]$.
# $s_C$ is computed once here and frozen (ReInit / online loop reuse `state0.carbon_price`).
#
# This section only reports the evidence for the choice of $q$; the configured default is `cfg.normalization.quantile`.

# %%
from dlent_rt.phase0 import type_arrays
from dlent_rt.reward import carbon_price as carbon_scale_from_quantile, weighted_quantile as _weighted_quantile

state0 = run_phase0(ds, cfg)
m0 = state0.p > 0
_, prio0, A0 = type_arrays(state0.type_space, state0.grid)
CV0 = (prio0 * state0.v_bar)[m0]                         # $
PI0 = (state0.phi_bar - state0.C_elec_bar)[m0]           # $
K0  = state0.C_carbon_bar[m0]                            # kgCO2
w0  = state0.p[m0]
lam = cfg.objective
# $/kgCO2 reference: $190/ton, US EPA 2023 Social Cost of Carbon (EPA-HQ-OAR-2021-0317,
# Rennert et al. Nature 2022). Comparison only, not used by the model.
SCC = 0.19

Q_DEFAULT = cfg.normalization.quantile
Q_LIST = sorted({0.50, 0.75, 0.90, 0.95, 0.99, Q_DEFAULT})
scales = {q: carbon_scale_from_quantile(CV0, PI0, K0, w0, q) for q in Q_LIST}
assert np.isclose(scales[Q_DEFAULT]["s_C"], state0.carbon_price)

def wq(x, q, w=w0):
    return _weighted_quantile(x, w, q)

print(f"strategy={state0.norm_strategy}  q={Q_DEFAULT}  s_C={state0.carbon_price:.4g} $/kgCO2  (reward in $, no divisor)")
print(f"populated types: {m0.sum()} / {state0.n_types}   (H jobs assigned: {state0.n_assigned})")

# %% [markdown]
# #### 2.5.a Quantiles and carbon price per $q$
#
# `median(C$)/median(CV)` and `median(C$)/median(Pi+)` use p-weighted medians over populated types; `Pi+` is $\Pi$
# restricted to types with $\Pi > 0$. The two `r<0` columns are the share of populated types (unweighted, as
# `pos_reward_types` in Section 3) with $\lambda_1 CV + \lambda_2 \Pi < 0$ (no carbon) and with
# $\lambda_1 CV + \lambda_2 \Pi - \lambda_3 s_C K < 0$ (config lambdas).

# %%
r_nocarb = lam.lambda1 * CV0 + lam.lambda2 * PI0
pi_pos = PI0 > 0
med_cv = wq(CV0, 0.5)
med_pipos = _weighted_quantile(PI0[pi_pos], w0[pi_pos], 0.5) if pi_pos.any() else np.nan

rows_a = []
for q in Q_LIST:
    s = scales[q]
    C_usd = s["s_C"] * K0
    r_carb = r_nocarb - lam.lambda3 * C_usd
    rows_a.append({
        "q": q, "default": "<-- config" if q == Q_DEFAULT else "",
        "Q_q(CV) $": s["Q_cv"], "Q_q(|Pi|) $": s["Q_abs_pi"], "D_q $": s["D_q"],
        "K_q kgCO2": s["K_q"], "s_C $/kgCO2": s["s_C"], 
        "med(C$)/med(CV)": wq(C_usd, 0.5) / med_cv,
        "med(C$)/med(Pi+)": wq(C_usd, 0.5) / med_pipos,
        "r<0 no carbon %": 100 * (r_nocarb < 0).mean(),
        "r<0 with carbon %": 100 * (r_carb < 0).mean(),
    })
table_a = pd.DataFrame(rows_a).set_index("q")
display(table_a.style.format(precision=4).apply(
    lambda r: ["font-weight: bold; background-color: #fff3cd" if r.name == Q_DEFAULT else "" for _ in r],
    axis=1))

# %% [markdown]
# #### 2.5.b Ranges of each objective, in dollars (no divisor)
#
# p-weighted percentiles over populated types; `min`/`max` over populated types. `coverage [0, D_q]` = p-weighted share
# of populated types whose value lies in $[0, D_q]$. $CV$ and $\Pi$ do not depend on $q$; only their coverage does.

# %%
PCTS = [("p5", 0.05), ("p50", 0.50), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99)]
rows_b = []
for q in Q_LIST:
    s = scales[q]
    for name, x in (("CV", CV0), ("Pi", PI0), ("C$ = s_C*K", s["s_C"] * K0)):
        row = {"q": q, "objective": name, "min": x.min()}
        row.update({k: wq(x, v) for k, v in PCTS})
        row["max"] = x.max()
        row["D_q"] = s["D_q"]
        row["coverage [0, D_q] %"] = 100 * w0[(x >= 0) & (x <= s["D_q"])].sum() / w0.sum()
        rows_b.append(row)
table_b = pd.DataFrame(rows_b).set_index(["q", "objective"])
display(table_b.style.format(precision=4).apply(
    lambda r: ["background-color: #fff3cd" if r.name[0] == Q_DEFAULT else "" for _ in r], axis=1))

# %% [markdown]
# #### 2.5.c Histograms per $q$
#
# One row per $q$; histograms over populated types weighted by $p_j$; x range clipped to the p-weighted [p1, p99] and
# shared per column across rows. Dashed lines: $Q_q$ of the objective (for $\Pi$: $Q_q(|\Pi|)$, the term entering $D_q$)
# and $D_q$. For $C\$$ the two coincide by construction ($s_C K_q = D_q$).

# %%
cols_c = [("CV [\\$]", lambda s: CV0, "steelblue", lambda s: s["Q_cv"], "Q_q(CV)"),
          ("Pi [\\$]", lambda s: PI0, "darkorange", lambda s: s["Q_abs_pi"], "Q_q(|Pi|)"),
          ("C\\$ = s_C*K [\\$]", lambda s: s["s_C"] * K0, "gray", lambda s: s["s_C"] * s["K_q"], "Q_q(C\\$)")]
xr = []
for _, f, _, _, _ in cols_c:
    xs = [f(scales[q]) for q in Q_LIST]
    xr.append((min(wq(x, 0.01) for x in xs), max(wq(x, 0.99) for x in xs)))

fig, axes = plt.subplots(len(Q_LIST), 3, figsize=(13, 2.7 * len(Q_LIST)), squeeze=False)
for r, q in enumerate(Q_LIST):
    s = scales[q]
    for c, (name, f, color, fq, qlab) in enumerate(cols_c):
        ax, x, (lo, hi) = axes[r, c], f(s), xr[c]
        ax.hist(x, bins=60, range=(lo, hi), weights=w0 / w0.sum(), color=color)
        ax.axvline(fq(s), color="k", ls="--", lw=1, label=f"{qlab}={fq(s):.3g}")
        ax.axvline(s["D_q"], color="purple", ls="--", lw=1, label=f"D_q={s['D_q']:.3g}")
        ax.set_xlim(lo, hi)
        title = f"q={q}{'  (config)' if q == Q_DEFAULT else ''}  s_C={s['s_C']:.3g} \\$/kg\n{name}"
        if c == 1:
            ax.axvline(0, color="r", ls="--")
            title += f" - {100 * w0[x < 0].sum() / w0.sum():.0f}% negative (p-weighted)"
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
        if c == 0:
            ax.set_ylabel("p-weighted share")
plt.tight_layout(); plt.show()

# compact overlay at q = 0.90
s90 = scales[0.90]
ov = [("CV", CV0, "steelblue"), ("Pi", PI0, "darkorange"), ("C\\$ = s_C*K", s90["s_C"] * K0, "gray")]
lo = min(wq(x, 0.01) for _, x, _ in ov); hi = max(wq(x, 0.99) for _, x, _ in ov)
fig, ax = plt.subplots(figsize=(9, 3.6))
for name, x, color in ov:
    ax.hist(x, bins=80, range=(lo, hi), weights=w0 / w0.sum(), histtype="step", lw=1.6, color=color, label=name)
ax.axvline(s90["D_q"], color="purple", ls="--", lw=1, label=f"D_q={s90['D_q']:.3g}")
ax.axvline(0, color="r", ls="--", lw=1)
ax.set_xlim(lo, hi); ax.set_xlabel("\\$"); ax.set_ylabel("p-weighted share")
ax.set_title(f"q=0.90: CV, Pi, C\\$ overlaid (s_C={s90['s_C']:.3g} \\$/kgCO2), x in p-weighted [p1, p99]")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()

# %% [markdown]
# #### 2.5.d 15 random Phase-0 jobs (default $q$)
#
# Job-level objectives in dollars use the job's own bid $v$, $\phi(v)$ (generator $\phi$, as in $\bar\phi_j$), $E$, $CI$
# and price. `pct_*` = p-weighted percentile of the job's value within the type-level distribution of that objective
# over populated types of H (the distribution that defines $Q_q$). Decision = oracle only (static: no occupancy,
# no thinning, no replay): `rejected: IR` if the bid is below the first grid bin, else `accepted` if
# margin $= \tilde r_{type} - \tau^\top v_{type} > 0$, else `rejected: oracle`. `decision error (discretization)` flags rows where the
# job-level $R$ and the type-level margin disagree in sign.

# %%
from dlent_rt.phase0 import assign_type_ids

H = ds.H
s_C = state0.carbon_price
k_H = np.asarray(state0.grid.assign(state0.bids.v_rate, state0.myerson))
tid_H = assign_type_ids(state0.type_space, k_H, H.priority, H.hw_idx)
N_type = np.bincount(tid_H[tid_H >= 0], minlength=state0.n_types)
E_H = _compute_job_energy(H, cfg)
price_H, ci_H = _lookup_grid(H, ds)
# job-level reward terms of every H job from the single reward module (dlent_rt.reward)
from dlent_rt import reward
JT = reward.job_terms(H.priority, state0.bids.v_rate, state0.bids.cost_base,
                      reward.electricity_cost(E_H, price_H), reward.carbon_kg(E_H, ci_H),
                      state0.myerson, s_C, lam)

def wpct(x, ref):
    """p-weighted percentile of x within ref (populated types), midpoint convention."""
    o = np.argsort(ref, kind="stable")
    cw = (np.cumsum(w0[o]) - 0.5 * w0[o]) / w0.sum()
    return 100 * float(np.interp(x, ref[o], cw))

rng25 = np.random.default_rng(2025)
pick = np.sort(rng25.choice(H.n, size=15, replace=False))
tau0 = state0.tau
rows_d = []
for i in pick:
    j = int(tid_H[i]); v_i = float(state0.bids.v_rate[i]); phi_i = float(JT.phi[i])
    C_elec_i = float(E_H[i] * price_H[i])
    sw, pi, cd, R = float(JT.cv[i]), float(JT.profit[i]), float(JT.carbon_usd[i]), float(JT.R[i])
    if j < 0:
        r_t = margin = np.nan; dec = "rejected: IR"; v_type = np.full(len(tau0), np.nan); n_j = d_j = np.nan
    else:
        r_t = float(state0.r_tilde[j]); v_type = state0.v[j]; margin = r_t - float(tau0 @ v_type)
        dec = "accepted" if margin > 0 else "rejected: oracle"; n_j = int(N_type[j]); d_j = float(state0.d_bar[j])
    rows_d.append({
        "job": int(H.collection_id[i]), "prio": int(H.priority[i]), "hw": int(H.hw_idx[i]),
        "type": j, "N_j": n_j, "d_bar_j [steps]": d_j,
        "v $": v_i, "phi(v) $": phi_i, "dur [h]": float(H.duration_hours[i]),
        "E kWh": float(E_H[i]), "CI kg/kWh": float(ci_H[i]), "C_elec $": C_elec_i,
        "CV $": sw, "Pi $": pi, "s_C*K $": cd, "R $": R, "r~_type $": r_t, "margin $": margin,
        "pct_CV": wpct(sw, CV0), "pct_Pi": wpct(pi, PI0), "pct_C$": wpct(cd, s_C * K0),
        "A_cpu": float(H.A[i, 0]), "A_ram": float(H.A[i, 1]),
        "v_cpu": float(v_type[0]), "v_ram": float(v_type[1]),
        "A_cpu/c_cpu %": 100 * float(H.A[i, 0] / ds.capacity[0]),
        "A_ram/c_ram %": 100 * float(H.A[i, 1] / ds.capacity[1]),
        "decision": dec,
        # job reward and type decision disagree: a decision error due to the type discretization
        "decision error (discretization)": bool(np.isfinite(margin) and np.sign(R) != np.sign(margin)),
    })
jobs_d = pd.DataFrame(rows_d)

print(f"tau (shadow prices, {ds.resources}) = {tau0}   (0 => capacity not binding)")
print(f"s_C = {s_C:.4g} $/kgCO2 (q={Q_DEFAULT});  lambdas = {lam.lambda1}, {lam.lambda2}, {lam.lambda3}")
print(jobs_d["decision"].value_counts().to_string())
print(f"decision errors due to discretization (job R vs type margin): "
      f"{int(jobs_d['decision error (discretization)'].sum())}")

DEC_COL = {"accepted": "#d4edda", "rejected: oracle": "#f8d7da", "rejected: IR": "#e2d9f3"}
display(jobs_d.style.format(precision=4)
        .apply(lambda r: [f"background-color: {DEC_COL[r['decision']]}" for _ in r], axis=1)
        .apply(lambda r: ["font-weight: bold; color: #b00" if r["decision error (discretization)"]
                          and c in ("R $", "r~_type $", "margin $", "decision error (discretization)")
                          else "" for c in r.index], axis=1))

# %%
DEC_MK = {"accepted": "green", "rejected: oracle": "red", "rejected: IR": "purple"}
parts = np.column_stack([lam.lambda1 * jobs_d["CV $"], lam.lambda2 * jobs_d["Pi $"], -lam.lambda3 * jobs_d["s_C*K $"]])
labels = ["lam1*CV", "lam2*Pi", "-lam3*s_C*K"]
colors = ["steelblue", "darkorange", "gray"]
x = np.arange(len(jobs_d))
fig, ax = plt.subplots(figsize=(13, 4.2))
pos_base = np.zeros(len(x)); neg_base = np.zeros(len(x))
for k in range(3):
    val = parts[:, k]
    base = np.where(val >= 0, pos_base, neg_base)
    ax.bar(x, val, bottom=base, color=colors[k], label=labels[k], width=0.7)
    pos_base += np.clip(val, 0, None); neg_base += np.clip(val, None, 0)
for d, col in DEC_MK.items():
    sel = (jobs_d["decision"] == d).to_numpy()
    if sel.any():
        ax.scatter(x[sel], jobs_d["R $"][sel], marker="D", s=45, color=col, edgecolor="k", zorder=3,
                   label=f"net R ({d})")
mm = jobs_d["decision error (discretization)"].to_numpy()
if mm.any():
    ax.scatter(x[mm], jobs_d["R $"][mm], marker="o", s=160, facecolor="none", edgecolor="k", zorder=4,
               label="decision error (discretization): R vs margin")
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([f"{j}\nprio={p}" for j, p in zip(jobs_d["job"], jobs_d["prio"])],
                   fontsize=7, rotation=45, ha="right")
for t, d in zip(ax.get_xticklabels(), jobs_d["decision"]):
    t.set_color(DEC_MK[d])
ax.set_ylabel("\\$"); ax.set_title(f"15 Phase-0 jobs: reward components (q={Q_DEFAULT}, s_C={s_C:.3g} \\$/kgCO2); "
                                 f"tick colour = decision")
ax.legend(fontsize=7, ncol=3); plt.tight_layout(); plt.show()

# %% [markdown]
# #### 2.5.e Weight of each objective in the reward
#
# A decision error due to discretization occurs when the job's own reward $R$ and the margin of its type have opposite signs: the decision is taken on the type average, so an individually unprofitable job can be accepted (and vice versa), a discretization effect that shrinks with a finer valuation grid.
#
# *Note (checked on the data):* the within-type spread is not only in $v$. In the mismatch type of 2.5.d all jobs sit in the
# same bid bin (bids within ~2%), yet the generator $\phi(v)$ ranges over both signs (it depends on each job's own bid
# distribution), and $C^{elec}$, $K$ depend on duration and CI, which are not part of the type. A finer valuation grid
# reduces only the $v$ part of the spread.
#
# Weight of objective $i$ = $|c_i| / \sum_k |c_k|$ with $c = (\lambda_1 CV,\ \lambda_2 \Pi,\ -\lambda_3 s_C K)$
# (absolute contribution share). Type level: p-weighted mean share over populated types, and p-weighted share of types
# where each objective has the largest $|c_i|$.

# %%
OBJ = ["CV", "Pi", "C$"]

def contrib(sw, pi, k, s):
    c = np.column_stack([lam.lambda1 * sw, lam.lambda2 * pi, -lam.lambda3 * s * k])
    tot = np.abs(c).sum(axis=1, keepdims=True)
    return c, np.divide(np.abs(c), tot, out=np.zeros_like(c), where=tot > 0)

# job level (the 15 jobs of 2.5.d)
c_j, sh_j = contrib(jobs_d["CV $"].to_numpy(), jobs_d["Pi $"].to_numpy(), jobs_d["s_C*K $"].to_numpy() / s_C, s_C)
weights_jobs = pd.DataFrame({
    "job": jobs_d["job"], "decision": jobs_d["decision"], "R $": jobs_d["R $"],
    **{f"w_{o} %": 100 * sh_j[:, k] for k, o in enumerate(OBJ)},
    "dominant": [OBJ[k] for k in np.abs(c_j).argmax(axis=1)],
    "decision error (discretization)": jobs_d["decision error (discretization)"],
})
display(weights_jobs.style.format(precision=2)
        .apply(lambda r: [f"background-color: {DEC_COL[r['decision']]}" for _ in r], axis=1))

# type level, per q (CV, Pi do not depend on q; only the carbon share does)
rows_e = []
for q in Q_LIST:
    s = scales[q]["s_C"]
    c_t, sh_t = contrib(CV0, PI0, K0, s)
    dom = np.abs(c_t).argmax(axis=1)
    row = {"q": q, "default": "<-- config" if q == Q_DEFAULT else ""}
    row.update({f"mean w_{o} %": 100 * np.average(sh_t[:, k], weights=w0) for k, o in enumerate(OBJ)})
    row.update({f"dominant {o} %": 100 * w0[dom == k].sum() / w0.sum() for k, o in enumerate(OBJ)})
    rows_e.append(row)
display(pd.DataFrame(rows_e).set_index("q").style.format(precision=2))

# job-level mismatch rate over the whole H
ok_H = tid_H >= 0
R_H = JT.R
print(f"decision errors due to discretization (R_job vs r~_type) over H: {100 * np.mean(np.sign(R_H[ok_H]) != np.sign(state0.r_tilde[tid_H[ok_H]])):.1f}% "
      f"of {ok_H.sum()} jobs")

# 100% stacked bars: 15 jobs (left), type level per q (right)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [3, 1.4]})
xj = np.arange(len(weights_jobs)); left = np.zeros(len(xj))
for k, (o, col) in enumerate(zip(OBJ, ["steelblue", "darkorange", "gray"])):
    ax[0].bar(xj, 100 * sh_j[:, k], bottom=left, color=col, label=o.replace("$", "\\$"), width=0.7); left += 100 * sh_j[:, k]
ax[0].set_xticks(xj); ax[0].set_xticklabels(weights_jobs["job"], fontsize=7, rotation=45, ha="right")
for t, d in zip(ax[0].get_xticklabels(), weights_jobs["decision"]):
    t.set_color(DEC_MK[d])
ax[0].set_ylabel("share of |contribution| %"); ax[0].set_title("15 jobs (tick colour = decision)")
ax[0].legend(fontsize=8)
tl = pd.DataFrame(rows_e).set_index("q"); xq = np.arange(len(tl)); left = np.zeros(len(xq))
for o, col in zip(OBJ, ["steelblue", "darkorange", "gray"]):
    ax[1].bar(xq, tl[f"mean w_{o} %"], bottom=left, color=col, width=0.7); left += tl[f"mean w_{o} %"].to_numpy()
ax[1].set_xticks(xq); ax[1].set_xticklabels([f"{q}{'*' if q == Q_DEFAULT else ''}" for q in tl.index])
ax[1].set_xlabel("q (* = config)"); ax[1].set_title("types: p-weighted mean share")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Section 3: Phase 0 — Full Output

# %% [markdown]
# Phase 0 turns the historical batch `H` into a discretized type space
# (`K* x |priorities| x |hw profiles|`), synthesizes bids, computes energy and
# carbon costs per job, aggregates everything into per-type statistics,
# normalizes and scalarizes the three objectives into `r_tilde`, and solves the
# fluid LP-RS relaxation for the admission oracle's shadow prices. `PhaseState`
# (`st`) holds every one of these per-type arrays; the block below is the
# summary read after running it once with the active configuration.

# %%
st = run_phase0(ds, cfg)
m = st.p > 0

print("--- Phase 0 results ---")
print(f"n_types={st.n_types}  populated={m.sum()}  assigned={st.n_assigned}/{ds.H.n}")
print(f"IR rejected: {st.n_ir_rejected} ({100*st.n_ir_rejected/ds.H.n:.1f}%)")
print(f"sum(p)={st.p.sum():.6f}  true_rate={ds.H.n/ds.H_span_steps:.6f}")
print(f"r_tilde: [{st.r_tilde[m].min():.4f}, {st.r_tilde[m].max():.4f}]")
pos_r = (st.r_tilde[m] > 0).sum()
print(f"pos_reward_types: {pos_r}/{m.sum()} ({100*pos_r/max(m.sum(),1):.1f}%)")
print(f"gamma={st.gamma:.4g}  xi={st.xi:.4g}")

lp = solve_lp_rs(st.p, st.r_tilde, st.v, ds.capacity)
cons_opt = ((st.p[lp.support_idx, None] * st.v[lp.support_idx] * lp.x[:, None]).sum(axis=0)
            if lp.x.size else np.zeros(len(ds.capacity)))
util_opt = cons_opt / ds.capacity
print(f"util_opt={np.round(util_opt, 4)}  tau={st.tau}  binds={st.tau.max()>0}")

print("\n--- LP breakdown ---")
cons_upper = (st.p[m, None] * st.v[m]).sum(axis=0)
print(f"capacity            = {np.round(ds.capacity, 2)}")
print(f"consumption (x=1)   = {np.round(cons_upper, 2)}")
print(f"consumption (opt)   = {np.round(cons_opt, 2)}")
print(f"util_opt            = {np.round(util_opt, 4)}")
print(f"tau (shadow prices) = {st.tau}")
print(f"lambda_star         = {lp.optimum:.4f}")
print(f"LP binds            = {bool(st.tau.max() > 0)}")

# %% [markdown]
# ## Section 4: Objective Analysis

# %% [markdown]
# Phase 0 keeps two different virtual-value quantities per type: `phi_bar`
# (the sample mean of `phi(v)` over the jobs actually assigned to that type)
# drives the reward `r_tilde` used for admission, while `phi(L)`
# (`st.phi_tilde`, evaluated at the type's discretization bin floor `L`) is
# reserved for the online payment rule. This section looks at both, and at why
# the reward uses `phi_bar`.

# %%
"""OBJ_STRATEGIES = [
    ("lognormal sigma=0.3", apply_overrides(cfg, {"bid.sigma": 0.3})),
]

obj_results = []
scale_rows = []
for label, cfg_i in OBJ_STRATEGIES:
    st_i = run_phase0(ds, cfg_i)
    m_i = st_i.p > 0
    priority_of_i = np.array([st_i.type_space.decode(t)[1] for t in range(st_i.n_types)])

    raw_sat_i  = priority_of_i[m_i] * st_i.v_bar[m_i]
    raw_prof_i = st_i.phi_bar[m_i] - st_i.C_elec_bar[m_i]
    raw_carb_i = st_i.C_carbon_bar[m_i] * st_i.carbon_price
    obj_results.append((label, raw_sat_i, raw_prof_i, raw_carb_i))

    scales_i = {"W": np.abs(raw_sat_i).max(), "Pi": np.abs(raw_prof_i).max(),
                "C": np.abs(raw_carb_i).max()}
    scale_rows.append({
        "strategy": label,
        "W_max": scales_i["W"], "Pi_max": scales_i["Pi"], "C_max": scales_i["C"],
        "ratio max/min": max(scales_i.values()) / max(min(scales_i.values()), 1e-12),
        "Pi<0 %": 100 * (raw_prof_i < 0).mean(),
    })

scale_df = pd.DataFrame(scale_rows).round(4)
display(scale_df)

fig, axes = plt.subplots(len(OBJ_STRATEGIES), 3, figsize=(13, 3.2 * len(OBJ_STRATEGIES)))
for row, (label, raw_sat_i, raw_prof_i, raw_carb_i) in enumerate(obj_results):
    a0, a1, a2 = axes[row]

    # W histogram: manual bins of uniform width 2, matching the Pi binning below.
    w_pos = raw_sat_i[raw_sat_i > 0]
    w_lo = np.floor(w_pos.min() / 2.0) * 2.0
    w_hi = np.ceil(w_pos.max() / 2.0) * 2.0
    w_bin_edges = np.arange(w_lo, w_hi + 2.0, 2.0)
    a0.hist(w_pos, bins=w_bin_edges, color="steelblue")
    a0.set_title(f"{label}\nW = priority x v_bar\nrange=[{raw_sat_i.min():.4f}, {raw_sat_i.max():.4f}]")
    a0.set_xlim(raw_sat_i.min(), 60)

    # Pi histogram: manual bins of uniform width 2 across the full range
    # (negative and non-negative), computed directly on raw_prof_i.
    pi_lo = np.floor(raw_prof_i.min() / 2.0) * 2.0
    pi_hi = np.ceil(raw_prof_i.max() / 2.0) * 2.0
    pi_bin_edges = np.arange(pi_lo, pi_hi + 2.0, 2.0)

    a1.hist(raw_prof_i, bins=pi_bin_edges, color="darkorange")
    a1.axvline(0, color="r", ls="--")
    a1.set_title(f"Pi = phi_bar - C_elec\n"
                 f"range=[{raw_prof_i.min():.4f}, {raw_prof_i.max():.4f}] - "
                 f"{100*(raw_prof_i<0).mean():.0f}% negative")
    a1.set_xlim(-50, 50)

    a2.hist(raw_carb_i[raw_carb_i > 0], bins=80, color="gray")
    a2.set_title(f"C = C_carbon_bar (kgCO2)\nrange=[{raw_carb_i.min():.4f}, {raw_carb_i.max():.4f}]")

plt.tight_layout(); plt.show()"""

# %%
# ============================================================
# GAMMA COMPONENT ANALYSIS — which term dominates gamma?
# gamma = max(r_max, v_rate_max, d_max, c_max, r_max/v_min)
# ============================================================
m = st.p > 0

r_max     = float(np.max(np.abs(st.r_tilde[m])))
v_rate_max = float(st.bids.v_rate.max())
d_max     = ds.d_max_steps
c_max     = float(ds.capacity.max())
v_min     = float(np.min(st.v[st.v > 0]))
r_over_v  = r_max / max(v_min, 1e-12)

components = {
    "r_max (max |reward|)":           r_max,
    "v_rate_max (max bid)":           v_rate_max,
    "d_max (max duration, steps)":    d_max,
    "c_max (max capacity)":           c_max,
    "r_max / v_min (reward/volume)":  r_over_v,
}

print("gamma components (gamma = max of these):")
print(f"{'component':40s} {'value':>14s}  {'dominates?':>10s}")
print("-" * 70)
dominant = max(components.values())
for name, val in sorted(components.items(), key=lambda x: -x[1]):
    flag = "  ◄── THIS" if abs(val - dominant) < 1e-6 else ""
    print(f"{name:40s} {val:14.2f}{flag}")

print(f"\ngamma = {st.gamma:.2f}")
print(f"\nInterpretation:")
dom_name = max(components, key=components.get)
if "d_max" in dom_name:
    print(f"  d_max ({d_max} steps = {d_max * cfg.time.tick_seconds / 3600:.1f} hours)")
    print(f"  dominates. Every test threshold is proportional to gamma,")
    print(f"  so tests need O(gamma^2) steps to fire. With gamma={st.gamma:.0f},")
    print(f"  tests need ~{st.gamma**2:.1e} steps, vs T={ds.T:,} available.")
    print(f"  Ratio gamma^2/T = {st.gamma**2 / ds.T:.1f}x")
    print(f"  If >> 1, tests will NEVER fire in the available horizon.")
elif "r_max" in dom_name and "v_min" not in dom_name:
    print(f"  r_max = {r_max:.4f} dominates — reward range is the bottleneck.")
    print(f"  Reducing the reward range (tighter normalization) would lower gamma.")
elif "v_min" in dom_name:
    print(f"  r_max/v_min = {r_over_v:.2f} dominates — some types have tiny")
    print(f"  resource volume v_ij. The smallest v is {v_min:.6f}.")
    print(f"  These are types with very short durations or tiny resource requests.")

# %%
# ============================================================
# Diagnose the v_min=0 type driving gamma
# ============================================================
m = st.p > 0
v_flat = st.v[m]  # (n_populated, n_res)
v_min_per_type = v_flat.min(axis=1)  # min across resources, per type
idx_min = np.argmin(v_min_per_type)

populated_ids = np.where(m)[0]
tid = populated_ids[idx_min]

k, priority, a_row = st.type_space.decode(tid)
print(f"Type with v_min: type_id={tid}")
print(f"  bin k={k}, priority={priority}, hw profile={a_row}")
print(f"  p_j = {st.p[tid]:.8f}  (N_j implied ~ {st.p[tid] * ds.H_span_steps:.2f} jobs)")
print(f"  d_bar[j] = {st.d_bar[tid]:.4f} steps")
print(f"  v[j] = {st.v[tid]}")
print(f"  v_bar (bid) = {st.v_bar[tid]:.4f}")
print(f"  r_tilde[j] = {st.r_tilde[tid]:.6f}")
print()

# how many types have near-zero volume?
near_zero = (v_flat.min(axis=1) < 1e-6).sum()
print(f"Types with v_min < 1e-6: {near_zero} / {m.sum()}")

# recompute gamma excluding near-zero volume types
v_safe = v_flat[v_flat.min(axis=1) >= 1e-6]
if len(v_safe) > 0:
    v_min_safe = v_safe[v_safe > 0].min()
    r_max = float(np.max(np.abs(st.r_tilde[m])))
    print(f"\nv_min (all)   = {v_flat[v_flat>0].min():.8f}  -> r_max/v_min = {r_max/v_flat[v_flat>0].min():.2e}")
    print(f"v_min (safe)  = {v_min_safe:.8f}  -> r_max/v_min = {r_max/v_min_safe:.2e}")
    print(f"gamma would drop by {(r_max/v_flat[v_flat>0].min())/(r_max/v_min_safe):.0f}x")

# %%
pop = st.p > 0
w = st.p[pop][:, None] * st.v[pop]          # consumo atteso per tipo e risorsa
pos = st.r_tilde[pop] > 0
print("c:", ds.capacity)
print("consumo r>0:", w[pos].sum(0), " r<0:", w[~pos].sum(0), " totale:", w.sum(0))
print("quota di volume con r<0:", w[~pos].sum(0) / w.sum(0))

# %%
# ---- Phase 0: where does the volume of r~<0 types come from? (read-only) ----
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from dlent_rt.phase0 import type_arrays
from dlent_rt.lp_oracle import solve_lp_rs

spec = cfg.energy_model
unit = np.array([spec.spec_cpu_core, spec.spec_ram_gb], float)       # cores / GB per 1.0 normalized
res_names = ["CPU (cores)", "RAM (GB)"]
c0 = np.asarray(ds.capacity, float)
lam1, lam2, lam3 = cfg.objective.lambda1, cfg.objective.lambda2, cfg.objective.lambda3

cap = pd.DataFrame({"normalized": c0, "physical": c0 * unit}, index=res_names)
print(f"Cluster capacity (1.0 normalized = {spec.spec_cpu_core:g} cores / {spec.spec_ram_gb:g} GB):")
display(cap.round(3))

pop = st.p > 0
p, v, r = st.p[pop], st.v[pop], st.r_tilde[pop]
_, prio_t, _ = type_arrays(st.type_space, st.grid)
CV = (prio_t * st.v_bar)[pop]
PI = (st.phi_bar - st.C_elec_bar)[pop]
K = st.C_carbon_bar[pop]

sC = getattr(st, "carbon_price", None)
if sC is None:                                    # fall back: infer the frozen carbon price from the reward
    m = K > 0
    sC = float(np.median((lam1 * CV[m] + lam2 * PI[m] - r[m]) / (lam3 * K[m])))
    print(f"s_C not stored on the state: inferred {sC:.2f} $/kgCO2")
T1, T2, T3 = lam1 * CV, lam2 * PI, -lam3 * sC * K       # reward components [$]
print(f"check r~ = lam1*CV + lam2*Pi - lam3*s_C*K : max |diff| = {np.abs(r - (T1 + T2 + T3)).max():.3g} $"
      "  (should be ~0 under quantile_linear)")

# ---- LP outcome ----
lp = solve_lp_rs(p=st.p, r=st.r_tilde, v=st.v, c=ds.capacity)
x = lp.x                                                  # aligned with the populated types
print(f"LP: tau = {st.tau}, lambda* = {st.lam_star:.4f} $/tick, status = {lp.status}")

# ---- 1. job-level outcome of Phase 0 (expected, fluid) ----
neg = r <= 0
n_ass = st.n_assigned
sh = p / p.sum()
admit = float((sh * x).sum())
rej_rew = float(sh[neg].sum())
rej_cap = float((sh[~neg] * (1 - x[~neg])).sum())
jobs = pd.DataFrame({"% of H jobs": 100 * np.array([admit, rej_rew, rej_cap]),
                     "jobs": n_ass * np.array([admit, rej_rew, rej_cap])},
                    index=["admitted (fluid LP)", "rejected: reward <= 0", "rejected: capacity (x < 1)"])
print(f"\nPhase-0 job outcome on H ({n_ass} assigned jobs, IR-rejected: {st.n_ir_rejected}):")
display(jobs.round(1))

# ---- 2. volume by class ----
cls = np.where(~neg, "r>0",
      np.where((T1 + T2) <= 0, "r<=0 even without carbon (profit term)", "r<=0 only because of carbon"))
w = p[:, None] * v                                        # expected consumption, normalized units
rows = []
for name in ["r>0", "r<=0 only because of carbon", "r<=0 even without carbon (profit term)"]:
    mk = cls == name
    if not mk.any():
        continue
    vol = w[mk].sum(0)
    wt = p[mk]
    rows.append({"class": name, "types": int(mk.sum()), "% of jobs": 100 * sh[mk].sum(),
                 "CPU cores": vol[0] * unit[0], "CPU % of c": 100 * vol[0] / c0[0],
                 "RAM GB": vol[1] * unit[1], "RAM % of c": 100 * vol[1] / c0[1],
                 "mean lam1*CV $": np.average(T1[mk], weights=wt),
                 "mean lam2*Pi $": np.average(T2[mk], weights=wt),
                 "mean -lam3*sC*K $": np.average(T3[mk], weights=wt)})
vol_tbl = pd.DataFrame(rows).set_index("class")
print("\nExpected consumption by class (time-average) and reward components (weighted by p):")
display(vol_tbl.round(2))
tot = w.sum(0)
print(f"total if everything is admitted: {tot[0]*unit[0]:.1f} cores ({100*tot[0]/c0[0]:.0f}% of c), "
      f"{tot[1]*unit[1]:.1f} GB ({100*tot[1]/c0[1]:.0f}% of c)")

# ---- 3. the 10 heaviest rejected types ----
ids = np.where(pop)[0]
order = np.argsort(-w[:, 0] * neg)[:10]
top = pd.DataFrame({"type id": ids[order], "% of jobs": 100 * sh[order],
                    "CPU cores": w[order, 0] * unit[0], "RAM GB": w[order, 1] * unit[1],
                    "lam1*CV $": T1[order], "lam2*Pi $": T2[order], "-lam3*sC*K $": T3[order], "r~ $": r[order]})
print("\n10 rejected types with the largest CPU volume:")
display(top.round(2))

# ---- 4. figure: volume by class, % of c ----
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
colors = {"r>0": "#4C9F70", "r<=0 only because of carbon": "#E0A030",
          "r<=0 even without carbon (profit term)": "#C0504D"}
for ax, i, nm in zip(axes, (0, 1), res_names):
    left = 0.0
    for name in colors:
        if name in vol_tbl.index:
            val = vol_tbl.loc[name, ["CPU % of c", "RAM % of c"][i]]
            ax.barh([0], [val], left=left, color=colors[name], label=name)
            left += val
    ax.axvline(100, color="k", ls="--", lw=1)
    ax.set_yticks([]); ax.set_xlabel(f"{nm}: expected consumption [% of c]")
axes[0].legend(fontsize=7, loc="upper center", bbox_to_anchor=(1.1, -0.35), ncol=3)
plt.tight_layout(); plt.show()

# %%
# ---- Dataset sanity check: what did the outlier step remove, and is the scale plausible? (read-only) ----
import numpy as np, pandas as pd
from dlent_rt.data import _read_csv

col = cfg.data.columns
spec = cfg.energy_model
k_cpu, k_ram = spec.spec_cpu_core, spec.spec_ram_gb
raw = _read_csv(cfg)
orr = ds.outlier_report
print(f"outlier policy = {orr.policy} | a_max_quantile = {cfg.outliers.a_max_quantile} | d_max_quantile = {cfg.outliers.d_max_quantile}")
print(f"jobs: raw {orr.n_before:,} -> kept {orr.n_after:,}  (dropped {orr.n_dropped_total:,} = "
      f"{100*orr.n_dropped_total/orr.n_before:.1f}%: resource {orr.n_dropped_resource:,}, duration {orr.n_dropped_duration:,})")
print(f"cutoffs: CPU {orr.a_max_thresholds[0]*k_cpu:.1f} cores, RAM {orr.a_max_thresholds[1]*k_ram:.1f} GB, "
      f"duration {orr.d_max_hours_threshold*60:.1f} min")

A = raw[[col.requested_cpu, col.requested_ram]].to_numpy(float) * np.array([k_cpu, k_ram])
D_h = raw[col.duration_hours].to_numpy(float)
keep = ~(((raw[[col.requested_cpu, col.requested_ram]].to_numpy(float) > orr.a_max_thresholds[None, :]).any(axis=1))
         | (D_h > orr.d_max_hours_threshold)) if orr.policy == "drop" else np.ones(len(raw), bool)

span_h = (raw[col.datetime].max() - raw[col.datetime].min()).total_seconds() / 3600
ch_cpu = A[:, 0] * D_h                                   # core-hours per job
ch_ram = A[:, 1] * D_h                                   # GB-hours per job
summary = pd.DataFrame({
    "jobs": [len(raw), int(keep.sum()), int((~keep).sum())],
    "core-hours": [ch_cpu.sum(), ch_cpu[keep].sum(), ch_cpu[~keep].sum()],
    "GB-hours": [ch_ram.sum(), ch_ram[keep].sum(), ch_ram[~keep].sum()],
}, index=["raw trace", "kept (used)", "dropped"])
summary["mean cores busy"] = summary["core-hours"] / span_h
summary["mean GB busy"] = summary["GB-hours"] / span_h
print(f"\ntrace span: {span_h/24:.1f} days  |  arrivals/hour: raw {len(raw)/span_h:.1f}, kept {keep.sum()/span_h:.1f}")
display(summary.round(1))
print(f"share of core-hours removed by the outlier step: {100*ch_cpu[~keep].sum()/ch_cpu.sum():.1f}% "
      f"(with {100*(~keep).mean():.1f}% of the jobs)")

qs = [0.5, 0.9, 0.99, 1.0]
def qtab(mask, name):
    return pd.DataFrame({"cores": np.quantile(A[mask, 0], qs), "GB": np.quantile(A[mask, 1], qs),
                         "duration [min]": np.quantile(D_h[mask] * 60, qs),
                         "core-hours/job": np.quantile(ch_cpu[mask], qs)},
                        index=[f"{name} p{int(q*100)}" for q in qs])
display(pd.concat([qtab(np.ones(len(raw), bool), "raw"), qtab(keep, "kept")]).round(2))

# %%
# ---- Outlier cutoff sweep (read-only, raw data only): which cutoff gives a coherent regime? ----
import numpy as np, pandas as pd
from dlent_rt.data import _read_csv

col = cfg.data.columns
spec = cfg.energy_model
unit = np.array([spec.spec_cpu_core, spec.spec_ram_gb], float)
raw = _read_csv(cfg)
A_n = raw[[col.requested_cpu, col.requested_ram]].to_numpy(float)          # normalized
A = A_n * unit                                                              # cores, GB
D_h = raw[col.duration_hours].to_numpy(float)
span_h = (raw[col.datetime].max() - raw[col.datetime].min()).total_seconds() / 3600
ch_raw = (A[:, 0] * D_h).sum()
c_now = np.asarray(ds.capacity, float) * unit                               # cores, GB (current sizing)

rows = []
for q in (0.75, 0.90, 0.95, 0.99):
    a_thr = np.quantile(A_n, q, axis=0)
    d_thr = np.quantile(D_h, q)
    keep = ~((A_n > a_thr[None, :]).any(axis=1) | (D_h > d_thr))
    Ak, Dk = A[keep], D_h[keep]
    load = (Ak * Dk[:, None]).sum(0) / span_h                               # mean cores / GB busy
    a_max = Ak.max(0)
    r = {"q": q, "kept jobs": int(keep.sum()), "% dropped": 100 * (1 - keep.mean()),
         "jobs/hour": keep.sum() / span_h,
         "% core-hours kept": 100 * (Ak[:, 0] * Dk).sum() / ch_raw,
         "d_max [min]": d_thr * 60, "a_max CPU [cores]": a_max[0], "a_max RAM [GB]": a_max[1],
         "mean load CPU [cores]": load[0], "mean load RAM [GB]": load[1],
         "xi with current c (CPU)": a_max[0] / c_now[0]}
    for xi in (0.1, 0.25, 0.5):
        c_need = a_max[0] / xi                                              # CPU cores for that xi
        r[f"c for xi={xi} [cores]"] = c_need
        r[f"load/c at xi={xi}"] = load[0] / c_need
    rows.append(r)
sweep = pd.DataFrame(rows).set_index("q")
print(f"current cluster: {c_now[0]:.1f} cores / {c_now[1]:.1f} GB | trace span {span_h/24:.1f} days | raw jobs {len(raw):,}")
with pd.option_context("display.width", 250, "display.max_columns", None, "display.float_format", "{:,.2f}".format):
    display(sweep.T)
print("\nHow to read: 'c for xi=...' is the CPU capacity needed for the largest kept job to be at most xi*c; "
      "'load/c' is the mean CPU load of the kept jobs divided by that capacity "
      "(>= ~1 means the capacity can bind, << 1 means the cluster is idle even if xi is small).")

# %%
# ---- 2D outlier sweep (read-only, raw data only): size cutoff x duration cutoff ----
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from dlent_rt.data import _read_csv

col = cfg.data.columns
spec = cfg.energy_model
unit = np.array([spec.spec_cpu_core, spec.spec_ram_gb], float)
raw = _read_csv(cfg)
A_n = raw[[col.requested_cpu, col.requested_ram]].to_numpy(float)
A = A_n * unit                                                   # cores, GB
D_h = raw[col.duration_hours].to_numpy(float)
span_h = (raw[col.datetime].max() - raw[col.datetime].min()).total_seconds() / 3600
Q_SIZE = (0.50, 0.75, 0.90, 0.95)
Q_DUR = (0.90, 0.95, 0.99)
RATIO_TARGET = 4.0            # load / a_max >= 4  ->  load/c ~ 2 at xi = 0.5 (c = a_max / xi)

rows = []
for qs in Q_SIZE:
    a_thr = np.quantile(A_n, qs, axis=0)
    over_a = (A_n > a_thr[None, :]).any(axis=1)
    for qd in Q_DUR:
        d_thr = np.quantile(D_h, qd)
        keep = ~(over_a | (D_h > d_thr))
        Ak, Dk = A[keep], D_h[keep]
        load = (Ak * Dk[:, None]).sum(0) / span_h
        a_max = Ak.max(0)
        rows.append({"q size": qs, "q dur": qd, "kept jobs": int(keep.sum()), "% dropped": 100 * (1 - keep.mean()),
                     "jobs/h": keep.sum() / span_h, "d_max [min]": d_thr * 60, "d_max/T": d_thr / span_h,
                     "a_max CPU [cores]": a_max[0], "a_max RAM [GB]": a_max[1],
                     "load CPU [cores]": load[0], "load RAM [GB]": load[1],
                     "load/a_max CPU": load[0] / a_max[0], "load/a_max RAM": load[1] / a_max[1],
                     "c for xi=0.5 [cores]": a_max[0] / 0.5,
                     "arrivals x for load/c=2 (xi=0.5)": 2 * (a_max[0] / 0.5) / load[0],
                     "arrivals x for load/c=2 (xi=0.25)": 2 * (a_max[0] / 0.25) / load[0]})
sw = pd.DataFrame(rows)
sw["d_max/T ok (<0.10)"] = sw["d_max/T"] < 0.10
print(f"trace span {span_h/24:.1f} days | raw jobs {len(raw):,} | target: load/a_max >= {RATIO_TARGET:g}")
with pd.option_context("display.width", 260, "display.max_columns", None, "display.float_format", "{:,.2f}".format):
    display(sw.set_index(["q size", "q dur"]))

# heatmaps: load/a_max (CPU) and kept jobs
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, key, title, fmt in ((axes[0], "load/a_max CPU", "load / a_max (CPU)  [target >= 4]", "{:.2f}"),
                            (axes[1], "kept jobs", "kept jobs", "{:,.0f}")):
    M = sw.pivot(index="q size", columns="q dur", values=key).loc[list(Q_SIZE), list(Q_DUR)]
    im = ax.imshow(M.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(Q_DUR))); ax.set_xticklabels(Q_DUR); ax.set_xlabel("duration cutoff quantile")
    ax.set_yticks(range(len(Q_SIZE))); ax.set_yticklabels(Q_SIZE); ax.set_ylabel("size cutoff quantile")
    ax.set_title(title, fontsize=10)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = M.values[i, j]
            ax.text(j, i, fmt.format(val), ha="center", va="center", color="white", fontsize=9,
                    fontweight="bold" if (key == "load/a_max CPU" and val >= RATIO_TARGET) else None)
    plt.colorbar(im, ax=ax)
plt.tight_layout(); plt.show()
print("\nHow to read: a combination is promising when load/a_max (CPU and RAM) is high, d_max/T < 0.10 and enough jobs are kept. "
      "'arrivals x' = how much the arrival intensity must grow so that load/c = 2 with c = a_max/xi.")


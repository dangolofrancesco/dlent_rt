# %% [markdown]
# # DLENT-RT — Phase 0 diagnostics (week-32 environment)
#
# What Phase 0 sees and produces in the new environment: the uniform 300k sample, the size-based bid, the 24 h
# duration cap, the quota tied to the cluster, and the cluster = f × mean demand of H. Phase 0 and the configs are
# not changed here. The logic is in `dlent_rt/diagnostics.py`; this notebook calls it and plots.
#
# `CONFIG_PATH` (default `configs/v6_theory.yaml`: size_u0 bid, f = 0.4). Sections marked **[both f]** also show
# `OTHER_PATH` (default `configs/v6_size_bid.yaml`, f = 1.25). The v5 run (size bid) is in git history and in
# `reports/week32/phase0_diagnostics.md`. Every plot of bids, grid and objectives uses H, the Phase 0 history.
# Build and run: `python notebooks/src/build.py 01_phase0_diagnostics --execute`.

# %%
import os, sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
from dlent_rt.config import load_config
from dlent_rt.data import load_dataset
from dlent_rt.phase0 import run_phase0
from dlent_rt import diagnostics as dg
warnings.simplefilter("ignore")
pd.set_option("display.max_columns", 30, "display.width", 250, "display.max_colwidth", 120)
BLUE, ORANGE, GREY, INK = "#2a78d6", "#eb6834", "#52514e", "#0b0b0b"

CONFIG_PATH = os.environ.get("DLENT_CONFIG", "configs/v6_theory.yaml")
OTHER_PATH = os.environ.get("DLENT_CONFIG_OTHER", "configs/v6_size_bid.yaml")
OUT = ROOT / "reports" / "week32"


def style(ax, xlabel=None, ylabel=None, title=None):
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if title: ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.grid(axis="y", color="#e5e4e0", lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


cfg = load_config(ROOT / CONFIG_PATH)
ds = load_dataset(cfg)
st = run_phase0(ds, cfg)
cfg2 = load_config(ROOT / OTHER_PATH)
ds2 = load_dataset(cfg2)
st2 = run_phase0(ds2, cfg2)
TAG = {CONFIG_PATH: f"f = {cfg.capacity.f:g}", OTHER_PATH: f"f = {cfg2.capacity.f:g}"}

# %% [markdown]
# ## 0. Setup and environment
#
# The settings that define the environment, the dataset Phase 0 works on, and how the cluster C and the quota Q
# follow from the history H. **[both f]** for the cluster derivation.

# %%
display(pd.Series(dg.settings(cfg), name=CONFIG_PATH).to_frame())
display(pd.Series(dg.dataset_summary(ds, cfg), name="dataset").to_frame())

# %%
# full config (collapse this cell)
import dataclasses
print(json.dumps(dataclasses.asdict(cfg), indent=1, default=str)[:20000])

# %% [markdown]
# **Cluster derivation, step by step.** Total work in H / |H| = D̄; C = f × D̄; Q = 0.2 × C (fixed point at the
# quota's f). Little's law: the time-average occupancy of H with no capacity should be close to D̄. **[both f]**

# %%
deriv = pd.DataFrame({TAG[CONFIG_PATH]: dg.cluster_derivation(ds, cfg), TAG[OTHER_PATH]: dg.cluster_derivation(ds2, cfg2)})
display(deriv)

# %% [markdown]
# **Arrivals over the day and the week.** Jobs per hour by hour of day (UTC) and by day of week, H against the stream.
# If H's pattern matches the stream, rates learned on H transfer.

# %%
ap = dg.arrival_profile(ds)
fig, axs = plt.subplots(1, 2, figsize=(13, 3.8))
for ax, by, lab in zip(axs, ("hour of day", "day of week"), ("hour of day (UTC)", "day of week (0 = Monday)")):
    for name, col, ls in (("H", BLUE, "-"), ("stream", ORANGE, "--")):
        d = ap[(ap.set == name) & (ap.by == by)]
        ax.plot(d.value, d["jobs per hour"], color=col, ls=ls, marker="o", ms=3, label=name)
    style(ax, lab, "jobs per hour", f"Arrivals by {by}")
    ax.legend(frameon=False)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 1. Jobs in H
#
# The request, the capped duration and the work of the jobs Phase 0 learns from. The lines show the quota Q and the
# 24 h cap: no job in H is above Q, and no duration is above 24 h.

# %%
J = dg.h_jobs(ds)
Q, cap = ds.limit_report.quota, cfg.limits.d_cap_hours
fig, axs = plt.subplots(1, 4, figsize=(17, 3.6))
for ax, (name, x), line in zip(axs, J.items(), (Q[0], Q[1], cap, None)):
    x = np.asarray(x)[np.asarray(x) > 0]
    ax.hist(x, bins=np.logspace(np.log10(x.min()), np.log10(x.max()), 50), color=BLUE)
    ax.set_xscale("log")
    if line:
        ax.axvline(line, color=ORANGE, ls="--"); ax.text(line, ax.get_ylim()[1] * 0.9, f" {line:.3g}", color=ORANGE, fontsize=8)
    style(ax, name + " (log)", "jobs")
plt.tight_layout(); plt.show()
display(dg.stats_table(J))

# %% [markdown]
# ## 2. Bids
#
# The bid grows with the requested resources and the duration. The markup bid / cost base is the known lognormal
# with σ = 0.3 (mean 1). Its virtual value φ is negative for low markups.

# %%
FORMULA = {"size": "cost_base = (1 + ln q) * (gamma1 * cores + gamma2 * GB) * d",
           "size_u0": f"cost_base = (u0 * r + gamma1 * cores + gamma2 * GB) * d, u0 = {cfg.bid.u0_per_machine_hour} $/machine-h, no priority",
           "base_utility": "cost_base = (u0 * (1 + ln q) + gamma1 * cores + gamma2 * GB) * d"}
print(FORMULA[cfg.bid.mode], f"   (dlent_rt/bid.py, synthesize_lognormal, mode '{cfg.bid.mode}')")
print("bid = lognormal(mu = ln(cost_base) - sigma^2 / 2, sigma)        so E[bid] = cost_base;  "
      f"sigma = {cfg.bid.sigma}, gamma1 = {cfg.bid.gamma1} $/core-h, gamma2 = {cfg.bid.gamma2} $/GB-h")
bf = dg.bid_frame(ds, st, cfg)
fig, axs = plt.subplots(2, 3, figsize=(16, 7.5))
axs[0, 0].hist(np.log10(bf.cost_base), bins=60, color=BLUE); style(axs[0, 0], "log10 cost base ($)", "jobs", "Cost base")
axs[0, 1].hist(np.log10(bf.bid), bins=60, color=BLUE); style(axs[0, 1], "log10 bid ($)", "jobs", "Bid")
m = np.linspace(0.2, 3, 400)
axs[0, 2].hist(bf.markup, bins=80, density=True, color=BLUE, label="H jobs")
axs[0, 2].plot(m, lognorm.pdf(m, s=cfg.bid.sigma, scale=np.exp(-cfg.bid.sigma ** 2 / 2)), color=ORANGE, label="lognormal pdf")
style(axs[0, 2], "markup bid / cost base", "density", f"Markup (mean {bf.markup.mean():.4f})"); axs[0, 2].legend(frameon=False)
axs[1, 0].hist(np.sign(bf.phi) * np.log10(1 + np.abs(bf.phi) * 1e4), bins=80, color=BLUE)
style(axs[1, 0], "sign(phi) log10(1 + |phi| x 1e4)", "jobs", f"phi(bid): {100 * (bf.phi < 0).mean():.1f}% below 0")
for q in sorted(bf.priority.unique()):
    d = bf[(bf.priority == q) & (bf.rd > 0)]
    axs[1, 1].scatter(d.rd, d.bid, s=2, alpha=0.4, label=f"tier {q:g}")
axs[1, 1].set_xscale("log"); axs[1, 1].set_yscale("log")
style(axs[1, 1], "work r*d (machine-h)", "bid ($)", "Bid vs work, by priority tier"); axs[1, 1].legend(frameon=False, markerscale=4, fontsize=7)
tiers = sorted(bf.priority.unique())
axs[1, 2].boxplot([bf.bid_per_machine_h[(bf.priority == q)].dropna() for q in tiers], tick_labels=[f"{q:g}" for q in tiers],
                  showfliers=False)
style(axs[1, 2], "priority tier", "bid per machine-hour ($)", "Bid per machine-hour")
plt.tight_layout(); plt.show()
display(dg.bid_tier_table(bf))
display(dg.sigma_sensitivity(ds, cfg))

# %% [markdown]
# ## 3. Energy and grid data
#
# Power is proportional to the request (no idle term). The grid gives the hourly price and carbon intensity (UTC;
# 1–2 June are copies of 30–31 May, marked `is_filled`). Prices and carbon are read at each job's arrival hour.

# %%
display(dg.power_check(ds, cfg))
ef = dg.energy_frame(ds, cfg)
fig, axs = plt.subplots(1, 3, figsize=(16, 3.6))
axs[0].hist(np.log10(ef.power_kW[ef.power_kW > 0]), bins=60, color=BLUE); style(axs[0], "log10 power (kW)", "jobs", "Power")
axs[1].scatter(ef.cpu, ef.power_kW, s=2, alpha=0.3, color=BLUE); axs[1].set_xscale("log"); axs[1].set_yscale("log")
style(axs[1], "CPU request (machines)", "power (kW)", "Power vs CPU request")
axs[2].hist(np.log10(ef.energy_kWh[ef.energy_kWh > 0]), bins=60, color=BLUE); style(axs[2], "log10 energy (kWh)", "jobs", "Energy per job")
plt.tight_layout(); plt.show()

# %%
g = dg.grid_frame(cfg)
h0, h1 = pd.Timestamp(ds.H.datetime[0]), pd.Timestamp(ds.H.datetime[-1])
fig, axs = plt.subplots(2, 1, figsize=(14, 5.5), sharex=True)
for ax, col, lab in zip(axs, ("elec_price_per_kWh", "carbon_intensity_kgCO2_per_kWh"), ("price ($/kWh)", "carbon (kg/kWh)")):
    ax.plot(g.timestamp_utc, g[col], color=BLUE, lw=0.8)
    ax.axvspan(h0, h1, color=GREY, alpha=0.12, label="H")
    f = g[g.is_filled]
    ax.scatter(f.timestamp_utc, f[col], s=6, color=ORANGE, label="filled hours", zorder=3)
    style(ax, None, lab); ax.legend(frameon=False, fontsize=8)
axs[0].set_title("Grid over the month (UTC)", loc="left", fontsize=10)
plt.tight_layout(); plt.show()

g["hour"] = g.timestamp_utc.dt.hour
fig, axs = plt.subplots(1, 3, figsize=(16, 3.6))
for ax, col, lab in zip(axs[:2], ("elec_price_per_kWh", "carbon_intensity_kgCO2_per_kWh"), ("price ($/kWh)", "carbon (kg/kWh)")):
    gr = g.groupby("hour")[col]
    ax.plot(gr.mean().index, gr.mean(), color=BLUE)
    ax.fill_between(gr.mean().index, gr.quantile(0.1), gr.quantile(0.9), color=BLUE, alpha=0.2)
    style(ax, "hour of day (UTC)", lab, f"{lab.split()[0]}: mean and p10-p90")
r = np.corrcoef(g.elec_price_per_kWh, g.carbon_intensity_kgCO2_per_kWh)[0, 1]
axs[2].scatter(g.elec_price_per_kWh, g.carbon_intensity_kgCO2_per_kWh, s=4, color=BLUE)
style(axs[2], "price ($/kWh)", "carbon (kg/kWh)", f"Hourly price vs carbon (corr {r:.2f})")
plt.tight_layout(); plt.show()

fig, axs = plt.subplots(1, 4, figsize=(17, 3.4))
for ax, col, lab in zip(axs, ("price", "ci", "C_elec", "kg"), ("price at arrival ($/kWh)", "carbon intensity (kg/kWh)",
                                                               "log10 electricity cost per job ($)", "log10 carbon per job (kg)")):
    x = ef[col].to_numpy()
    if col in ("C_elec", "kg"):
        x = np.log10(x[x > 0])
    ax.hist(x, bins=50, color=BLUE); style(ax, lab, "jobs")
plt.tight_layout(); plt.show()
print(json.dumps(dg.grid_coverage(ds, cfg), indent=1))

# %% [markdown]
# ## 4. Phase 0 output (what the online algorithm starts from) [both f]
#
# Every quantity of the Phase 0 state the online model uses, for both cluster sizes. The two runs differ only in C,
# so the types, the bids and s_C are the same; τ, the LP and its fill change with C.

# %%
tab1, tab2 = dg.phase0_table(ds, st, cfg), dg.phase0_table(ds2, st2, cfg2)
dg.save_table(tab1, OUT / f"phase0_summary_{cfg.run.name}.json")
dg.save_table(tab2, OUT / f"phase0_summary_{cfg2.run.name}.json")
show = lambda t: {k: (json.dumps(v, default=float) if isinstance(v, (dict, list)) else v) for k, v in t.items() if not k.startswith("_")}
display(pd.DataFrame({TAG[CONFIG_PATH]: show(tab1), TAG[OTHER_PATH]: show(tab2)}))

# %%
e = np.asarray(st.grid.bnd_v, float)
k = st.grid.assign(st.bids.v_rate, st.myerson)
fig, axs = plt.subplots(1, 3, figsize=(16, 3.6))
axs[0].plot(np.arange(e.size), e, color=BLUE); axs[0].set_yscale("log"); style(axs[0], "edge index", "bid ($, log)", "Valuation grid edges")
axs[1].plot(np.arange(1, e.size), np.diff(e), color=BLUE); axs[1].set_yscale("log"); style(axs[1], "bin k", "bin width ($, log)", "Bin width")
axs[2].bar(*np.unique(k, return_counts=True), color=BLUE); style(axs[2], "bin k", "H jobs", "H jobs per bin")
plt.tight_layout(); plt.show()

# %%
tt = dg.type_terms(st, cfg)
pop = tt[tt.p > 0]
jobs_per_type = np.rint(pop.p * ds.H_span_steps).astype(int)
fig, axs = plt.subplots(1, 2, figsize=(14, 3.8))
axs[0].hist(jobs_per_type, bins=np.logspace(0, np.log10(jobs_per_type.max() + 1), 40), color=BLUE); axs[0].set_xscale("log")
style(axs[0], "H jobs per populated type (log)", "types", f"Jobs per type ({100 * (jobs_per_type == 1).mean():.0f}% of types have 1 job)")
heat = pop.groupby(["priority", "hw"]).size().unstack(fill_value=0)
im = axs[1].imshow(heat.to_numpy(), aspect="auto", cmap="Blues")
axs[1].set_yticks(range(len(heat.index)), [f"{q:g}" for q in heat.index]); axs[1].set_xlabel("hardware profile"); axs[1].set_ylabel("priority tier")
axs[1].set_title("Populated types by priority x hardware profile", loc="left", fontsize=10); fig.colorbar(im, ax=axs[1])
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 5. Objective analysis
#
# The three terms of the type reward r̃ = customer value + profit − carbon $, each on its own, per type and per job;
# how they combine into the sign of r̃; and how they vary with size, hour of day and priority.

# %%
jt = dg.job_terms_H(ds, st, cfg)
fig, axs = plt.subplots(2, 3, figsize=(16, 7))
for j, term in enumerate(("customer value", "profit", "carbon $")):
    for i, (frame, lab) in enumerate(((pop, "per type"), (jt, "per job"))):
        x = frame[term].to_numpy()
        axs[i, j].hist(x, bins=np.concatenate([-np.logspace(3, -6, 40), np.logspace(-6, 3, 40)]), color=BLUE)
        axs[i, j].set_xscale("symlog", linthresh=1e-6)
        style(axs[i, j], f"{term} ($, symlog)", lab, f"{term}, {lab}")
plt.tight_layout(); plt.show()
rows = []
for lab, frame in (("per type", pop), ("per job", jt)):
    for term in ("customer value", "profit", "carbon $"):
        x = frame[term]
        rows.append({"level": lab, "term": term, "mean": x.mean(), **{f"p{q}": x.quantile(q / 100) for q in (10, 50, 90)}, "max": x.max()})
display(pd.DataFrame(rows))
x1 = tab1["_x"]
display(pd.DataFrame({"all populated types": dg.term_shares(tt), "admitted types (p x)": dg.term_shares(tt, x1)}))

# %%
fig, axs = plt.subplots(1, 3, figsize=(17, 3.8))
for ax, (s_, t_, tab_, lab) in zip(axs[:2], ((st, tt, tab1, TAG[CONFIG_PATH]), (st2, dg.type_terms(st2, cfg2), tab2, TAG[OTHER_PATH]))):
    pp = t_.p > 0
    r = t_.r_tilde[pp]
    adm = tab_["_x"][pp] > 1e-9
    bins = np.concatenate([-np.logspace(2, -6, 40), np.logspace(-6, 2, 40)])
    ax.hist(r[~adm], bins=bins, color=GREY, label="not admitted"); ax.hist(r[adm], bins=bins, color=BLUE, label="admitted", alpha=0.8)
    ax.axvline(0, color=ORANGE); ax.set_xscale("symlog", linthresh=1e-6)
    style(ax, "r_tilde ($, symlog)", "types", f"r_tilde, {lab}"); ax.legend(frameon=False, fontsize=8)
c = np.where(pop.r_tilde > 0, BLUE, ORANGE)
axs[2].scatter(pop["customer value"], pop["carbon $"], s=4, c=c); axs[2].set_xscale("log"); axs[2].set_yscale("log")
axs[2].plot([1e-5, 1e3], [1e-5, 1e3], color=GREY, lw=0.8)
style(axs[2], "customer value per type ($)", "carbon $ per type", "Customer value vs carbon (blue r > 0, orange r <= 0)")
plt.tight_layout(); plt.show()

# %%
display(dg.decile_table(jt))
by_hour = jt.groupby("hour")[["customer value", "profit", "carbon $", "reward"]].mean()
by_tier = jt.groupby("priority")[["customer value", "profit", "carbon $", "reward"]].mean()
fig, axs = plt.subplots(1, 2, figsize=(14, 3.8))
for col, cc in zip(by_hour.columns, (BLUE, GREY, ORANGE, INK)):
    axs[0].plot(by_hour.index, by_hour[col], color=cc, marker="o", ms=3, label=col)
style(axs[0], "hour of day of arrival (UTC)", "mean per job ($)", "Terms by hour of day"); axs[0].legend(frameon=False, fontsize=8)
by_tier.plot.bar(ax=axs[1], color=(BLUE, GREY, ORANGE, INK)); style(axs[1], "priority tier", "mean per job ($)", "Terms by priority tier")
plt.tight_layout(); plt.show()
display(by_tier)

# %%
display(dg.trace_types(ds, st, cfg))
display(dg.carbon_weight_sensitivity(ds, st, cfg))

# %% [markdown]
# ## 6. Stability over 20 seeds (bid noise only)
#
# Only the bid noise changes with the seed. If s_C, the sign of r̃ and the LP move little, Phase 0 does not depend on
# one lucky draw.

# %%
ss = dg.seed_stability(ds, cfg)
display(ss.drop(columns="seed").agg(["mean", "std"]).T)

# %% [markdown]
# ## 7. Checks
#
# PASS / FAIL on the identities Phase 0 must satisfy.

# %%
display(dg.checks(ds, st, cfg, tab1))
display(dg.checks(ds2, st2, cfg2, tab2).rename(columns={"result": f"result ({TAG[OTHER_PATH]})"}))

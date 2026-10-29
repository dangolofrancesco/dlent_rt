# %% [markdown]
# # DLENT-RT — model check (week 33)
#
# This notebook checks the final model decisions D1–D6 on `configs/default.yaml`, the one configuration. It
# replaces the old notebooks 01 (Phase 0 diagnostics) and 05 (theoretical models).
#
# Each section has a short introduction, then a PASS / FAIL table, then the results. The logic is in
# `dlent_rt/diagnostics.py`. Three long computations are cached by `scripts/model_check_cache.py` in
# `reports/week33/cache/`: the f scan, the one-week bucketed MILP (3,600 s, its own process) and the one
# theoretical DLENT-RT run. Without the cache, this notebook computes them itself (about 1.5 hours).
#
# Build: `dlent_venv/bin/python notebooks/src/build.py 01_model_check --execute`.

# %%
import pickle, subprocess, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "dlent_rt").is_dir())
sys.path.insert(0, str(ROOT))
warnings.simplefilter("ignore")
pd.set_option("display.width", 200, "display.max_columns", 30, "display.float_format", lambda x: f"{x:,.6g}")

from dlent_rt import diagnostics as dg
from dlent_rt import theory_eval as te
from dlent_rt.benchmarks import Benchmarks, enlarge_for_job_models
from dlent_rt.config import default_config
from dlent_rt.data import load_dataset
from dlent_rt.models.core.stream import build_stream
from dlent_rt.phase0 import run_phase0

CACHE = ROOT / "reports" / "week33" / "cache"
BLUE, ORANGE, INK, INK2, GRID = te.BLUE, te.ORANGE, te.INK, te.INK2, te.GRID
RESULTS = {}                      # (decision, item) -> PASS / FAIL, for the checklist of section 11


def show(tab, keys=None):
    """Display a PASS / FAIL table and record its results for the checklist."""
    display(tab)
    for (d, w), rows in (keys or {}).items():
        RESULTS[(d, w)] = "PASS" if all(tab.loc[tab.check.isin(rows), "result"] == "PASS") else "FAIL"


def style(ax, xlabel, ylabel=None, log=False):
    if log:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def cached(name, compute):
    path = CACHE / name
    if path.is_file():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    print(f"{name}: no cache, computing it now")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "model_check_cache.py"), compute], check=True, cwd=ROOT)
    with open(path, "rb") as fh:
        return pickle.load(fh)


CFG = default_config()
COMMIT = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
DS = load_dataset(CFG)
ST = run_phase0(DS, CFG)
B = Benchmarks(DS, ST, CFG)
S = B.S
print(f"commit {COMMIT}; config configs/default.yaml; f = {CFG.capacity.f}; |H| = {DS.H.n:,}; |stream| = {S.n:,}")

# %% [markdown]
# ## 1. Data and limits
#
# The jobs come from `data/batch_v2_300k.csv` (a uniform 300k sample of May 2019). Each job has one duration: the
# effective duration (CPU-hours / CPU request), floored at 1 minute and cut at 24 h. Jobs are cut, not removed.
# The only rule that removes jobs is the job quota: a job asks for more than Q = 0.2 C on some resource. There
# are no outlier filters. H is the first 7 days; the stream is the rest.

# %%
RAW = dg.raw_jobs(CFG)
tab, summ = dg.checks_data(DS, CFG, RAW)
show(tab, {("D3", "24 h cap and 1-minute floor"): ["durations inside [1 minute, 24 h]",
                                                    "duration = effective duration (CPU-hours / CPU), floored and capped"],
           ("D3", "H = first 7 days"): ["H = the jobs of the first 7 days"]})
display(pd.Series(summ, name="value").to_frame())
display(dg.stats_table({"CPU request (machines)": DS.H.A[:, 0], "RAM request (machines)": DS.H.A[:, 1],
                        "duration (h)": DS.H.duration_hours, "work r x d (CPU machine-h)": DS.H.A[:, 0] * DS.H.duration_hours}))

# %%
J = te.job_distributions(CFG)
fig, axs = plt.subplots(1, 3, figsize=(14, 3.6))
for ax, which in zip(axs, ("cpu", "ram", "duration")):
    te.fig_cdf_jobs_work(J, which, ax)
plt.tight_layout(); plt.show()

# %% [markdown]
# **Reading.** The cap and the quota change little of the work. The table below gives the shares of jobs and of
# CPU-hours they remove or cut.

# %%
display(J["impact"])

# %% [markdown]
# ## 1b. What the job limits leave out
#
# The quota and the 24 h cap, on ALL 300,000 jobs of the file. The duration here is the effective duration
# (CPU-hours / CPU request) with the 1-minute floor and NO cap. Q is the fixed point at the configured f.

# %% [markdown]
# **Quota: requests and work.** (a) CPU vs RAM request; the kept jobs are below both quota lines. (b) CPU-hours vs
# RAM-hours per job. Kept jobs are solid; jobs over the quota on either resource are faded.

# %%
fig, QTAB = dg.plot_quota(RAW, DS.limit_report.quota, CFG)
plt.show()
display(QTAB)

# %% [markdown]
# **Reading.**

# %% [markdown]
# **24 h cap: durations.** (a) The effective duration of every job, counted by jobs. (b) The same bins weighted by
# CPU-hours, i.e. where the work is. Bars above 24 h are faded: the cap cuts those jobs to 24 h.

# %%
fig, CTAB = dg.plot_cap(RAW, CFG)
plt.show()
display(CTAB)

# %% [markdown]
# **Reading.**

# %% [markdown]
# ## 2. Bid and U0 calibration
#
# D2: cost_base_j = (U0 (1 + ln q_j) + γ1 cores_j + γ2 GB_j) d_j, and the bid v_j is lognormal with mean
# cost_base_j and σ = 0.3. U0 is in $ per hour per job. It is the cost rate γ1 cores + γ2 GB of the median job
# of H, ranked by that rate. It is computed once in Phase 0 and then fixed for the stream. γ1 = 0.0332 $/vCPU-h,
# γ2 = 0.00445 $/GiB-h, cores = CPU × 64, GB = RAM × 256.

# %%
tab, summ = dg.checks_bid(DS, ST, CFG, S)
show(tab, {("D2", "cost_base = (U0 (1 + ln q) + g1 cores + g2 GB) d"): ["cost_base = (U0 (1 + ln q) + gamma1 cores + gamma2 GB) d, H",
                                                                        "cost_base formula, stream"],
           ("D2", "U0 = rate of the median H job"): ["U0 = gamma1 cores + gamma2 GB of the median H job",
                                                     "U0 is the same for H and the stream (fixed after Phase 0)"]})
display(pd.Series(summ, name="value").to_frame())
BF = dg.bid_frame(DS, ST, CFG)
display(dg.bid_tier_table(BF))

# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 3.6))
axs[0].hist(np.log(BF.markup), bins=80, color=BLUE)
style(axs[0], "ln(markup v / cost_base), H", "jobs")
axs[0].set_title("Markups of H: normal in log space", loc="left", fontsize=10)
axs[1].hist(np.log10(BF.cost_base), bins=80, color=BLUE)
style(axs[1], "log10 cost_base ($), H", "jobs")
axs[1].set_title("Cost bases of H", loc="left", fontsize=10)
plt.tight_layout(); plt.show()

# %% [markdown]
# **Reading.** U0 is small because the median job is small: most jobs ask for a few hundredths of a machine. The
# bid grows with priority through (1 + ln q) only.

# %% [markdown]
# ## 3. Virtual value
#
# φ_j = cost_base_j · φ_m(v_j / cost_base_j), with φ_m(m) = m − (1 − F(m)) / f(m) for the lognormal markup
# (μ = −σ²/2). Two facts must hold: E[φ_m] = 0 (exactly, by integration by parts) and φ_m increasing (a regular
# distribution). The theoretical model refits σ on the markups: on H in Phase 0, and on the previous phase's
# arrivals at every ReInit. The no-learning model keeps σ = 0.3.

# %%
RUN = cached("theory_run.pkl", "theory")
tab, summ = dg.checks_phi(ST, CFG, phase_sigmas=RUN["phases"].sigma.to_numpy())
show(tab, {("D2", "sigma refit in Phase 0"): ["sigma refit on H within 0.01 of 0.3"],
           ("D2", "sigma refit at ReInit"): ["sigma refit at every ReInit within 0.01 of 0.3"],
           ("D2", "phi = cost_base phi_m(v / cost_base)"): ["E[phi_m] = 0 by quadrature (sigma = 0.1, 0.3, refit, 0.5)",
                                                           "phi_m increasing (z in [-6, 6])"]})
display(pd.Series(summ, name="value").to_frame())
print(f"sample mean of phi / cost_base over H: {(BF.phi / BF.cost_base).mean():.3f} (positive: the negative mass that "
      f"balances E[phi_m] = 0 sits in markups below 1 in a million draws)")

# %%
m = np.linspace(0.3, 2.0, 400)
fig, ax = plt.subplots(figsize=(7, 3.6))
from dlent_rt.bid import MyersonModel
for s, col, ls in ((CFG.bid.sigma, BLUE, "-"), (ST.myerson.sigma, ORANGE, "--")):
    ax.plot(m, MyersonModel.known_markup(s).phi_markup(m), color=col, ls=ls, lw=1.6, label=f"sigma = {s:.4f}")
ax.axhline(0, color=INK2, lw=0.8)
style(ax, "markup m = v / cost_base", "phi_m(m)")
ax.set_title("Virtual value of the markup (generator sigma and refit sigma overlap)", loc="left", fontsize=10)
ax.legend(frameon=False); plt.tight_layout(); plt.show()

# %% [markdown]
# ## 4. Objectives
#
# D1: customer value V = v (no priority factor), profit Π = φ − C_elec, carbon = s_C · K, and the reward
# R = λ1 V + λ2 Π − λ3 s_C K with all λ = 1 (linear). s_C = D_q / K_q, D_q = (Q_0.9(V) + Q_0.9(|Π|)) / 2,
# K_q = Q_0.9(K), p-weighted over H's types, computed once in Phase 0. Social welfare is a metric only:
# SW = Σ_admitted (v − C_elec − s_C K). Priority enters only the bid.

# %%
tab, summ, TERMS = dg.checks_objectives(DS, ST, CFG, S)
show(tab, {("D1", "V = v (no q factor); Pi = phi - C_elec; carbon = s_C K"): ["customer value V = v (no priority factor), stream"],
           ("D1", "R = l1 V + l2 Pi - l3 s_C K, linear, all lambda = 1"): ["all lambda = 1", "r~ = V + Pi - s_C K (linear, per type)"],
           ("D1", "s_C = D_q / K_q over H's types, once in Phase 0"): ["s_C = D_q / K_q, D_q = (Q_0.9(V) + Q_0.9(|Pi|)) / 2, recomputed"],
           ("D1", "SW metric only"): ["SW = v - C_elec - s_C K per job (metric only)"]})
display(pd.Series(summ, name="value").to_frame())

# %% [markdown]
# **Reading.** The shares say how much of Σ|term| each term carries over H's types. The ratio ΣpV / Σp|Π| says how
# the customer value compares with the profit. s_C is in $ per kg CO2 and follows the dollar level of the bids.

# %% [markdown]
# ## 5. Grid
#
# D4: a geometric grid on bids, L_k = v_min ρ^k, ρ = (v_max / v_min)^(1/128), with v_min / v_max the smallest /
# largest bid. There is no fixed floor (only a guard against bids ≤ 0). The theoretical model rebuilds the grid at
# every ReInit; the no-learning model keeps Phase 0's grid. Rounding a bid down to its bin edge loses at most
# 1 − 1/ρ of it.

# %%
tab, summ = dg.checks_grid(ST, CFG)
show(tab, {("D4", "geometric grid between the smallest and largest bid, no floor"): list(tab.check)})
display(pd.Series(summ, name="value").to_frame())

# %% [markdown]
# ## 6. Cluster and quota
#
# D3: D̄_i = Σ_H r_ij d_j / |H|, C_i = f · D̄_i, and Q_i = 0.2 · C_i of the cluster actually used (a fixed point:
# Q sets which jobs are kept, the kept jobs set D̄). f follows a rule with two conditions, in priority order, over
# f ∈ {0.3, 0.35, 0.4, 0.45, 0.5}:
#
# * (a) REQUIRED: τ > 0 (CPU or RAM) in both the Phase 0 LP and fluid OPT;
# * (b) among the f that pass (a): the static Phase 0 policy's "% of the time ≥ 95% full" closest to the 20–30% band.
#
# The rule was made explicit after the preview of this notebook (week 33): the first version ranked by (b) only,
# over a scan up to f = 0.6, and picked f = 0.6, where τ = 0.

# %%
SCAN = cached("f_scan.pkl", "scan")
R0 = next(r for r in SCAN["results"] if r["f"] == CFG.capacity.f)
tab, ITERS = dg.checks_cluster(DS, ST, CFG, SCAN, R0["fluid"]["tau"])
show(tab, {("D3", "C = f D_bar on H"): ["C = f x D_bar (H only)"],
           ("D3", "Q = 0.2 C fixed point at the actual f"): ["Q = 0.2 C at the fixed point (within 2%)",
                                                             "fixed point stopped with a move < 1%"]})
display(dg.scan_table(SCAN))
display(Markdown(f"**Fixed point at f = {CFG.capacity.f}** (Q in machines):"))
display(ITERS)

# %%
SC = dg.scan_table(SCAN)
fig, axs = plt.subplots(1, 2, figsize=(13, 3.6))
axs[0].axhspan(20, 30, color=GRID, alpha=0.8, lw=0)
axs[0].plot(SC.f, SC["% time >= 95% full"], color=BLUE, marker="o", ms=5, lw=1.6)
for f, y in zip(SC.f, SC["% time >= 95% full"]):
    axs[0].annotate(f"{y:.1f}%", (f, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8, color=INK2)
style(axs[0], "f", "% of the time >= 95% full")
axs[0].set_title("Fixed rule: the band is 20-30% (grey)", loc="left", fontsize=10)
axs[1].plot(SC.f, SC["C CPU"], color=BLUE, marker="o", ms=5, lw=1.6, label="C, CPU")
axs[1].plot(SC.f, SC["C RAM"], color=ORANGE, marker="o", ms=5, lw=1.6, ls="--", label="C, RAM")
style(axs[1], "f", "C (machines)")
axs[1].set_title("C = f x D_bar: D_bar grows with f through the quota", loc="left", fontsize=10)
axs[1].legend(frameon=False)
plt.tight_layout(); plt.show()

# %%
fig, axs = plt.subplots(2, 1, figsize=(13, 5.5), sharex=True)
te.fig_occupancy(R0, axs, CFG.time.tick_seconds)
plt.tight_layout(); plt.show()

# %% [markdown]
# **Job-level enlargement (D3, off by default).** The check runs Greedy-Pos (admit every job with R > 0 that fits)
# only as a capacity probe. It reports the share of the positive-reward jobs it rejects for lack of space. The
# option would enlarge C only if that share were above 50%.

# %%
ENL = RUN["job_level_check"]
chk = ENL["checks"][0]
tab = dg.check_table([("job-level enlargement off by default", not CFG.capacity.job_level_enlarge, ""),
                      ("check computed: positive-reward jobs rejected for space <= 50%", not chk["exceeds"],
                       f"{chk['n_rejected_space']:,} of {chk['n_positive']:,} = {100 * chk['share_rejected_space']:.1f}%")])
show(tab, {("D3", "job-level enlargement option and check"): ["job-level enlargement off by default"]})

# %% [markdown]
# ## 7. ξ and γ
#
# The DLENT guarantee (Theorem 1) is (1 − ε_D − ξ) · OPT minus error terms, with ε_D = 2√(ξL) + 2ξL,
# L = ln(|I|/δ), δ = 1/T³. It is not empty only if 1 − ε_D − ξ > 0, i.e. ξ ≤ ξ_max. The data ξ = max_i a_max,i / C_i
# is far above ξ_max, so ξ = 0.001 and γ = 1 are forced (D5). Test 1 then uses the threshold (1 − ξ) c.

# %%
tab, summ = dg.checks_constants(DS, ST, CFG)
show(tab, {("D5", "xi = 0.001, gamma = 1 forced; Test 1 = (1 - xi) c"): ["xi forced = 0.001, gamma forced = 1",
                                                                         "Theorem 1 not empty at the forced xi: 1 - eps_D - xi > 0"]})
display(pd.Series(summ, name="value").to_frame())

# %% [markdown]
# **Reading.** With the quota at 0.2 C, the largest job is about 0.2 C, so the data ξ stays near 0.2 at every f.
# Enlarging C does not help while Q grows with C. The row "C needed" keeps today's jobs fixed.

# %% [markdown]
# ## 8. Phase 0 output and the 9 checks
#
# The row "r_max (Tests 2-3 reward unit, $)" is the paper's bound on the rewards of admissible actions: the LARGEST r̃
# over Phase 0's populated types with r̃ > 0 (negative types are never admitted). With γ forced to 1, Tests 2 and 3
# divide every reward by it. It is fixed for the whole run (not changed at ReInit).

# %%
TAB0 = dg.phase0_table(DS, ST, CFG)
show(dg.checks_phase0(DS, ST, CFG, TAB0))
display(pd.Series({k: v for k, v in TAB0.items() if not k.startswith("_")}, name="value").to_frame())

# %% [markdown]
# ## 9. Theoretical benchmarks at the chosen f
#
# Fluid OPT (the full-stream LP on the stream's own type statistics), fluid on feasible types, the Phase 0 LP's
# expected reward, the static Phase 0 policy (LP fractions, dual rule for unseen types, real gate) and the
# no-learning model (the same, then the theoretical model's thinning). Every realised number uses the same job
# reward. Then τ per window, the decomposition from H to the stream, and the one-week bucketed MILP.

# %%
display(dg.benchmark_table(R0))
w24, w6 = R0["windows"][24], R0["windows"][6]
display(pd.DataFrame({"windows": [len(w24), len(w6)], "tau CPU > 0 %": [100 * (w24[:, 0] > 0).mean(), 100 * (w6[:, 0] > 0).mean()],
                      "tau RAM > 0 %": [100 * (w24[:, 1] > 0).mean(), 100 * (w6[:, 1] > 0).mean()]}, index=["1 day", "6 h"]))
fig, axs = plt.subplots(2, 2, figsize=(13, 5.5))
te.fig_tau_windows(R0, axs[0], hours=24)
te.fig_tau_windows(R0, axs[1], hours=6)
plt.tight_layout(); plt.show()

# %%
HS = te.h_vs_stream(B, ST, CFG)
display(pd.Series({"Phase 0 LP, expected $": HS["phase0_expected"],
                   "LP on the stream's statistics (Phase 0 grid) $": HS["stream_stats_phase0_grid_opt"],
                   "Phase 0's x on the stream's statistics $": HS["phase0_x_on_stream_stats"],
                   "fluid OPT $": HS["fluid_opt"], "arrival rate, stream / H (weighted)": HS["lambda_ratio"],
                   "mean r x d, stream / H (weighted)": HS["rd_ratio"]}, name="value").to_frame())

# %% [markdown]
# **The one-week bucketed MILP** (the first 7 days of the stream; reference buckets, true durations; its own HiGHS
# instance with a 3,600 s limit). The fluid LP on a window counts only the volume inside the window (D6).

# %%
WEEK = cached("week_milp.pkl", "week")
show(dg.checks_week(WEEK))
best, bound = WEEK["bucketed_believed"], min(WEEK["bucketed_dual_bound"], WEEK["bucketed_lp_believed"])
display(pd.Series({"jobs": WEEK["n_jobs"], "MILP candidates": WEEK["n_candidates"], "nonzeros": WEEK["nonzeros"],
                   "fluid OPT, volume inside the window $": WEEK["fluid"],
                   "fluid on feasible types, inside the window $": WEEK["fluid_feasible"],
                   "fluid OPT, whole volume $": WEEK["fluid_whole_volume"], "MILP best $": best, "MILP bound $": bound,
                   "MILP gap %": 100 * WEEK["gap"], "MILP runtime s": WEEK["milp_runtime_s"],
                   "discretization cost $ (best .. bound)": f"{WEEK['fluid_feasible'] - best:,.2f} .. {WEEK['fluid_feasible'] - bound:,.2f}",
                   "discretization cost %": f"{100 * (WEEK['fluid_feasible'] - best) / WEEK['fluid_feasible']:.2f} .. "
                                            f"{100 * (WEEK['fluid_feasible'] - bound) / WEEK['fluid_feasible']:.2f}",
                   "MILP realised (job rewards) $": WEEK["bucketed_realised"]}, name="value").to_frame())

# %% [markdown]
# ## 10. One theoretical DLENT-RT run
#
# The theoretical model with the default config (seed 42): frozen oracle, thinning after the LP decision, physical
# gate, Tests 1–3, ReInit, threshold payments. The run is cached. The tables below check each decision on the logs.
# Tests 2 and 3 work in units of Phase 0's r_max = the largest positive r̃ (section 8), fixed for the whole run.

# %%
show(dg.checks_run(RUN, CFG),
     {("D5", "frozen oracle; live switch"): ["frozen oracle: every score uses the phase's r~ (no live price)",
                                            "frozen oracle: k_hat = 1[r~ - tau v > 0]"],
      ("D5", "thinning after the LP decision"): ["admit = k_hat and U and fits (thinning after the LP decision, then the gate)",
                                                 "thinned count = k_hat and not U"],
      ("D5", "ReInit: all arrivals vs admitted"): ["ReInit statistics: rates, bids, sigma, requests from ALL arrivals of the phase",
                                                   "ReInit statistics: duration, energy, costs from ADMITTED jobs only"],
      ("D4", "grid rebuilt at every ReInit"): ["grid rebuilt at every ReInit with >= n_min arrivals"],
      ("D6", "threshold payment = lowest admitted bin's lower edge"): ["payment = lower edge of the lowest admitted bin of (q, hw) at the phase's tau"],
      ("D6", "surplus, realised profit, IR"): ["IR: payment <= bid for every admitted job"],
      ("D5", "timeout at 2 x the previous phase length"): ["tests start d_max after the phase start; timeout at 2 x the previous phase length"]})
display(dg.phase_table(RUN["phases"], CFG.time.tick_seconds))

# %%
A = RUN["arrivals"]
adm = A[A.admit]
sm = RUN["summary"]
static, nl = R0["static"]["split"], R0["no_learning"]["split"]
tot = pd.DataFrame({
    "DLENT-RT (theoretical)": [sm["R_total"], adm.R.sum(), sm["SW_total"], sm["n_admitted"]],
    "static Phase 0 policy": [static["reward"], static["reward"], static["social welfare"], R0["static"]["n_admitted"]],
    "no-learning model": [nl["reward"], nl["reward"], nl["social welfare"], R0["no_learning"]["n_admitted"]],
    "fluid OPT": [R0["fluid"]["split"]["reward"], np.nan, np.nan, np.nan]},
    index=["reward $", "reward $ (sum of admitted R)", "social welfare $", "jobs admitted"])
display(tot)
split = pd.DataFrame({"customer value $": [float(TERMS.cv[adm.i].sum())],
                      "profit $ (phi - C_elec)": [float(TERMS.profit[adm.i].sum())],
                      "carbon $": [-float(TERMS.carbon_usd[adm.i].sum())],
                      "payments $": [sm["payments_total"]], "customer surplus v - p $": [sm["surplus_total"]],
                      "realised profit p - C_elec $": [sm["profit_realised_total"]],
                      "IR violations": [sm["n_ir_violations"]], "phases": [sm["n_phases"]],
                      "end reasons": [str(sm["phase_end_reasons"])], "Tests 2-3 reward unit r_max $": [RUN["r_unit"]],
                      "runtime s": [RUN["runtime_s"]]}, index=["DLENT-RT"]).T
display(split)

# %%
viol = adm[~adm.ir]
display(Markdown(f"**IR violations: {len(viol):,} of {len(adm):,} admitted jobs.**"))
if len(viol):
    display(viol[["i", "m", "k_bin", "q", "hw", "v", "payment", "threshold_k", "cost_base"]].head(10))
off = adm[adm.below_grid | adm.above_grid]
display(Markdown(f"Admitted bids outside the phase grid's [v_min, v_max]: {int(adm.below_grid.sum()):,} below, "
                 f"{int(adm.above_grid.sum()):,} above; the largest payment / bid among them is "
                 f"{(off.payment / off.v).max() if len(off) else float('nan'):.3f}."))

# %%
fig, axs = plt.subplots(1, 2, figsize=(13, 3.8))
ph = RUN["phases"]
h = CFG.time.tick_seconds / 3600
axs[0].step(ph.tau_start * h / 24, ph.tau_cpu, where="post", color=BLUE, lw=1.4, label="tau CPU")
axs[0].step(ph.tau_start * h / 24, ph.tau_ram, where="post", color=ORANGE, lw=1.4, ls="--", label="tau RAM")
style(axs[0], "days since the start of the stream", "tau ($ per machine-tick)")
axs[0].set_title("Shadow prices per phase", loc="left", fontsize=10)
axs[0].legend(frameon=False)
axs[1].scatter(adm.v, adm.payment, s=3, color=BLUE, alpha=0.4, lw=0)
lim = [adm.v.min(), adm.v.max()]
axs[1].plot(lim, lim, color=INK2, lw=0.8, ls=":")
axs[1].set_xscale("log"); axs[1].set_yscale("log")
style(axs[1], "bid v ($, log)", "payment p ($, log)")
axs[1].set_title("Threshold payment vs bid (IR: below the dotted line)", loc="left", fontsize=10)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 11. D1–D6 checklist
#
# Each row names the code (file and line, found by searching the source) and the result of the notebook's checks.
# An empty result means the item is checked by reading the code (and by the tests), not by a number here.

# %%
display(dg.checklist(RESULTS))

# %% [markdown]
# # DLENT-RT — theoretical (type / average) models over f
#
# Loads `configs/v5_theory.yaml` (f = 0.5 and the dual rule for unseen types; the quota is fixed at the f = 1.25
# fixed point, so every f has the same jobs),
# shows the job distributions, then evaluates the type / average models for each f: fluid OPT (hindsight), fluid on
# feasible types, the Phase 0 LP and its static policy on the real stream, and the fluid LP per window. The bucketed
# MILP is run only if its matrix is small enough. No online model. The logic is in `dlent_rt/theory_eval.py`.
# Build and run: `python notebooks/src/build.py 05_theory_models --execute`.

# %%
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
from dlent_rt.config import load_config
from dlent_rt import theory_eval as te
warnings.simplefilter("ignore")
pd.set_option("display.max_columns", 30, "display.width", 250)

cfg = load_config(ROOT / "configs" / "v5_theory.yaml")          # f = 0.5, policy.unseen_types = dual_rule
F_SCAN = (0.10, 0.20, 0.30, 0.50, 0.70, 1.25)

# %% [markdown]
# ## 0. The theoretical models at f = 0.5 (summary)
#
# From the cache of `scripts/week32/theory_f050.py` (`reports/week32/theory_f050_cache.pkl`; the 1-week bucketed
# MILP behind the discretization cost takes up to an hour). Run that script first.

# %%
import pickle
with open(ROOT / "reports" / "week32" / "theory_f050_cache.pkl", "rb") as fh:
    F050 = pickle.load(fh)
with pd.option_context("display.max_colwidth", 200):
    display(F050["summary"].set_index("quantity"))
display(F050["milp"])

# %% [markdown]
# ## 1. Job distributions (all 300k jobs and the jobs of H, before and after the 24 h cap and the quota)

# %%
J = te.job_distributions(cfg)
display(J["impact"])
display(J["dist"].round(4))

# %%
fig, axs = plt.subplots(1, 3, figsize=(16, 4.2))
for ax, which in zip(axs, ("cpu", "ram", "duration")):
    te.fig_cdf_jobs_work(J, which, ax)
plt.tight_layout(); plt.show()

# %%
fig, ax = plt.subplots(figsize=(8, 5.5))
h = te.fig_hist2d(J, ax)
fig.colorbar(h[3], ax=ax, label="jobs per cell")
plt.tight_layout(); plt.show()

fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
te.fig_H_vs_month(J, axs)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 2. Models for each f

# %%
R = [te.evaluate_f(cfg, f, seed=cfg.run.seed) for f in F_SCAN]
rows = []
for r in R:
    for key, lab in (("phase0_lp", "Phase 0 LP"), ("fluid", "fluid OPT"), ("fluid_feasible", "fluid feasible")):
        m = r[key]
        rows.append(dict(f=r["f"], LP=lab, C_cpu=r["C"][0], C_ram=r["C"][1], tau_cpu=m["tau"][0], tau_ram=m["tau"][1],
                         fully=m["counts"]["fully"], partly=m["counts"]["partly"], rejected=m["counts"]["rejected"],
                         fill_cpu=m["fill"][0], fill_ram=m["fill"][1]))
display(pd.DataFrame(rows).round(5))

# %%
rows = []
for r in R:
    for key, lab in te.MODELS:
        sp = r["phase0_lp"]["expected"] if key == "phase0_expected" else r[key]["split"]
        rows.append(dict(f=r["f"], model=lab, **{k: round(v, 1) for k, v in sp.items()}))
    b = r["bucketed"]
    rows.append(dict(f=r["f"], model="bucketed MILP", reward=b["split"]["reward"] if b["run"] else np.nan,
                     note="" if b["run"] else f"skipped: {b['nonzeros']:,} nonzeros"))
display(pd.DataFrame(rows))

# %%
rows = []
for r in R:
    s = r["static"]
    w24, w6 = r["windows"][24], r["windows"][6]
    rows.append(dict(f=r["f"], full_time=s["full_time"], mean_occ_cpu=s["mean_occupancy_over_C"][0],
                     rejected_space=s["rejected_space"], realised=s["split"]["reward"],
                     tau_pos_day_cpu=(w24[:, 0] > 0).mean(), tau_pos_6h_cpu=(w6[:, 0] > 0).mean(),
                     oversize_stream=r["oversize"]["stream"]))
display(pd.DataFrame(rows).round(4))
f_sel = te.select_f(R)
print(f"selection rule ('% time >= 95% full' closest to 20-30%, ties to 25%): f = {f_sel}")

# %% [markdown]
# ## 3. Figures

# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
te.fig_tau_vs_f(R, axs)
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(8, 4.6))
te.fig_reward_vs_f(R, ax)
plt.tight_layout(); plt.show()

# %%
rs = next(r for r in R if r["f"] == f_sel)
fig, axs = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True)
te.fig_occupancy(rs, axs, cfg.time.tick_seconds)
plt.tight_layout(); plt.show()

fig, axs = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True)
te.fig_tau_windows(rs, axs, 24)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 4. Unseen types, H vs stream, bucketed MILP on one week
#
# Read from the cache of `scripts/week32/theory_models_2.py` (`reports/week32/theory_models_2_cache.pkl`): the
# bucketed MILP alone takes about 15 minutes per f. Run that script first.

# %%
import pickle
with open(ROOT / "reports" / "week32" / "theory_models_2_cache.pkl", "rb") as fh:
    C2 = pickle.load(fh)
print("static Phase 0 policy: unseen types rejected vs dual rule")
display(C2["n1"])
print("f selected by the fixed rule (closest to 20-30%, ties to 25%):", C2["sel"])

# %%
print("H vs stream (types seen in both, weighted by H's rate):",
      {k: round(C2["hv"][k], 4) for k in ("lambda_ratio", "rd_ratio", "r_tilde_ratio", "load_H", "load_stream",
                                          "load_stream_unseen")})
display(C2["top"])
display(C2["n4"])
display(C2["n5"])
if "n6" in C2:
    print("one day (end-of-window effect: the MILP can exceed the fluid LP, see theory_models_2.md)")
    display(C2["n6"])

# %%
rs = C2["selected"]
fig, axs = plt.subplots(4, 1, figsize=(11, 10.5), sharex=True)
te.fig_occupancy(rs, axs[:2], cfg.time.tick_seconds, option="reject")
te.fig_occupancy(rs, axs[2:], cfg.time.tick_seconds, option="dual_rule")
plt.tight_layout(); plt.show()

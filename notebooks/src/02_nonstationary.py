# %% [markdown]
# # 02 — Non-stationary bids (week 34)
#
# The stream's markups change once, at t* = 240 h after the stream start. Nothing else changes: not H, the costs,
# the arrivals, the requests, the durations, the prices or the carbon.
#
# | Scenario | Change at t* |
# |---|---|
# | none | no change (stationary) |
# | S1 bids fall | m → 0.5 m |
# | S6 bids spread | σ 0.3 → 0.5, E[m] = 1 |
# | S2 bids rise | m → 1.75 m |
#
# Models, in every environment: fluid OPT (hindsight, before and after t*), Practical (C3), Greedy-Pos, No learning
# (the practical model frozen at Phase 0), Theoretical, and, when cached, Practical + calibrated tests.
#
# Everything here is read from the caches of `scripts/run_scenarios.py`; nothing is run. The tables come from
# `scripts/scenarios_analysis.py`. The write-up is `reports/week34/scenarios_results.md`.
#
# Rebuild: `dlent_venv/bin/python notebooks/src/build.py 02_nonstationary --execute`.

# %%
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
warnings.simplefilter("ignore")
import scenarios_analysis as sa  # noqa: E402
from dlent_rt import scenarios as sc  # noqa: E402

pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
pd.set_option("display.width", 200)
MODELS = sa.models_cached()
# fixed colour per model (Okabe-Ito, colourblind-safe), never by rank
COLOR = dict(fluid="#000000", practical="#0072B2", practical_calibrated="#56B4E9", no_learning="#E69F00",
             greedy_pos="#009E73", theoretical="#CC79A7")
T_STAR = 240.0
print("models cached in every environment:", [sa.LABEL[m] for m in MODELS])

# %% [markdown]
# ## 1. Reward before and after t*
#
# "% OPT after" is the share of fluid OPT's reward after t*. Fluid OPT is the sum of two hindsight LPs, one before
# and one after t*: an upper bound, not a policy.

# %%
rw = sa.rewards()
rw.replace({"env": sa.ENV_LABEL, "model": sa.LABEL})[
    ["env", "model", "reward", "before t*", "after t*", "% OPT", "% OPT after", "admitted after"]]

# %% [markdown]
# **The value of learning** is Practical against No learning, after t*. **The value of the practical changes** is
# Practical against Theoretical.

# %%
sa.learning_gain(rw).replace({"env": sa.ENV_LABEL})

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
for ax, e in zip(axes.flat, sa.ENVS):
    for m in [m for m in MODELS if m != "fluid"]:
        d = sa.reward_rate(e, m)
        ax.plot(d.hour, d["reward per h"], color=COLOR[m], lw=1.6, label=sa.LABEL[m])
    ax.axvline(T_STAR, color="#777777", lw=1, ls="--")
    ax.set_title(sa.ENV_LABEL[e], fontsize=11)
    ax.grid(alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
for ax in axes[1]:
    ax.set_xlabel("hours since the stream start (t* dashed)")
for ax in axes[:, 0]:
    ax.set_ylabel("reward per hour (24 h windows)")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=9, frameon=False)
fig.suptitle("Realised reward over time", fontsize=12)
fig.tight_layout(rect=(0, 0.05, 1, 1))

# %% [markdown]
# ## 2. Detection: how the phases ended
#
# Tests 2 and 3 never fire, with the paper's thresholds (`reports/week34/test_margins.md`): they are out of reach by
# 2–3 orders of magnitude. Phases end only by Test 1 (capacity) or by timeout. Adaptation therefore comes from that
# cadence: the phase that contains t* ends, ReInit refits (μ̂, σ̂) on data that mix the two regimes, and the next
# phase refits on post-change data only.
#
# "Adapted phase starts" is the first phase whose μ̂ is within 0.1 of the true post-change μ (S1, S2), or whose σ̂ is
# within 0.05 of the true σ (S6).

# %%
sa.detection().replace({"env": sa.ENV_LABEL, "model": sa.LABEL})

# %% [markdown]
# ## 3. The markup model per phase
#
# The estimate each phase used (step lines) against the truth (dashed). Practical and Theoretical refit (μ̂, σ̂) at
# every ReInit; No learning keeps Phase 0's values all run long.

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
for ax, (e, par) in zip(axes, (("s1_bids_fall", "mu"), ("s2_bids_rise", "mu"), ("s6_bids_spread", "sigma"))):
    for m in ("practical", "theoretical"):
        t = sa.markups(e, m)
        x = np.r_[t["start h"].to_numpy(), t["end h"].iloc[-1]]
        y = t[f"{par}_hat"].to_numpy()
        ax.step(x, np.r_[y, y[-1]], where="post", color=COLOR[m], lw=1.8, label=sa.LABEL[m])
    tr = sc.truth(sa.cfg(e))
    ax.step([0, T_STAR, x[-1]], [tr[f"{par}0"], tr[f"{par}1"], tr[f"{par}1"]], where="post", color="#555555",
            lw=1.2, ls="--", label="truth")
    ax.axhline(sa.markups(e, "practical")[f"{par}_hat"].iloc[0], color=COLOR["no_learning"], lw=1.4,
               label="No learning (Phase 0)")
    ax.set_title(f"{sa.ENV_LABEL[e]}: {'μ̂' if par == 'mu' else 'σ̂'}", fontsize=11)
    ax.set_xlabel("hours since the stream start")
    ax.grid(alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].legend(fontsize=8, frameon=False, loc="lower left")
fig.tight_layout()

# %%
for e in ("s1_bids_fall", "s6_bids_spread", "s2_bids_rise"):
    print(sa.ENV_LABEL[e])
    display(sa.markups(e, "practical")[["phase", "start h", "end h", "end", "mu_hat", "sigma_hat", "tau CPU",  # noqa: F821
                                        "tau RAM", "true mu", "true sigma"]])

# %% [markdown]
# ## 4. Myerson check: expected against realised profit
#
# For the admitted jobs, the profit φ·d − C_elec expected under the run's own markup model ("belief"), against the
# realised profit, payment − C_elec. A stale model shows up as a ratio far from 1 after t*.

# %%
my = sa.myerson()
my.replace({"env": sa.ENV_LABEL, "model": sa.LABEL})[
    ["env", "model", "expected (belief) after", "realised after", "belief / realised after"]]

# %% [markdown]
# ## 5. Bids in bin 1 (never admitted)
#
# The grid is fixed at Phase 0. Halved bids below H's smallest bid fall into bin 1. This could be a limit of the
# fixed grid. In practice it is small.

# %%
sa.bin1().replace({"env": sa.ENV_LABEL})

# %% [markdown]
# ## 6. Practical + calibrated tests
#
# `tests.threshold: calibrated`: Test 2's threshold × κ2 = 0.00725 and Test 3's confidence terms × κ3 = 0.00427,
# chosen on H with a safety factor of 2. There were 0 false alarms on the stationary stream
# (`reports/week34/test_margins.md`, section 4). This section fills in once the four runs are cached.

# %%
if "practical_calibrated" in MODELS:
    d = sa.detection()
    display(d[d.model.isin(["practical", "practical_calibrated"])].replace(  # noqa: F821
        {"env": sa.ENV_LABEL, "model": sa.LABEL}))
    display(rw[rw.model.isin(["practical", "practical_calibrated", "no_learning"])].replace(  # noqa: F821
        {"env": sa.ENV_LABEL, "model": sa.LABEL})[["env", "model", "after t*", "% OPT after"]])
else:
    print("practical_calibrated is not cached in every environment yet")

# %% [markdown]
# ## 7. The stationary runs against week 33
#
# Week 34 estimates μ̂ (week 33 assumed μ = −σ²/2). On the stationary stream the totals move by about ±0.3%.

# %%
sa.week33_stationary()

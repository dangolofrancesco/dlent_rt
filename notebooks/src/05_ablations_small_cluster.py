# %% [markdown]
# # DLENT-RT — Ablations, round 1 (single factors), small-cluster regime
#
# Kash: apply the ablations one at a time, find which ones matter most, then decide the cumulative order. This notebook
# covers the **single-factor round only** (no cumulative runs).
#
# **Regime**: the committed `configs/default.yaml` + ONLY the capacity block of `configs/regime_small_cluster.yaml`
# (both from git HEAD, `scripts/regime_config.py`); 30k trace; theory model with ξ = 0.001, γ = 1.0, δ = 1/T³.
# **Seeds**: only the thinning seed varies (`online.thinning_seed_offset` = 2, 3, 4, 5, 6); Phase 0, H, the stream and the
# bids are fixed — the thinning draws are the model's only online randomness.
#
# **Ablations** (each a `cfg.ablation` flag or an existing `constants` key, default = the theory model):
# * **A1 thinning** (only the thinning cap p_U; the tests are untouched): A1a p_U = 1; A1b δ = 1/T inside p_U's ε_D;
#   A1c ε_D = 0 inside p_U (cap 1 − ξ).
# * **A2a ξ** ∈ {0.01, 0.05} (γ = 1); **A2b γ** ∈ {0.3, 0.1} (ξ = 0.001): forced constants, global by nature.
# * **A3 fixed grid**: ReInit keeps Phase 0's valuation grid instead of refitting it.
#
# This notebook only READS the cache of `scripts/ablation_runner.py` (`reports/week30/ablations/`). Build:
# `python notebooks/src/build.py 05_ablations_small_cluster --execute`.

# %%
import subprocess, sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "dlent_rt").is_dir())
ADIR = ROOT / "reports" / "week30" / "ablations"
FIGS = ROOT / "reports" / "week30" / "figs_ablations"
FIGS.mkdir(parents=True, exist_ok=True)
for f in ("runs.csv", "summary.csv", "references.json"):
    if not (ADIR / f).is_file():
        raise FileNotFoundError(f"{ADIR / f} missing: run `dlent_venv/bin/python scripts/ablation_runner.py run` first")
RUNS = pd.read_csv(ADIR / "runs.csv")
SUM = pd.read_csv(ADIR / "summary.csv", index_col=0)
REF = json.loads((ADIR / "references.json").read_text())
_git = lambda *a: subprocess.check_output(["git", "-C", str(ROOT), *a], text=True).strip()
NB_COMMIT = _git("rev-parse", "HEAD")
REGIME = "configs/regime_small_cluster.yaml"
print(f"NB05 RUN MARKER  notebook built at commit {NB_COMMIT}; runs at commit(s) {sorted(RUNS['commit'].unique())}")
print(f"configs/default.yaml blob: runs {sorted(RUNS['default blob'].unique())}, HEAD {_git('rev-parse', 'HEAD:configs/default.yaml')}")
print(f"{REGIME} blob: runs {sorted(RUNS['regime blob'].unique())}, HEAD {_git('rev-parse', f'HEAD:{REGIME}')}")
print(f"capacity {sorted(RUNS['capacity'].unique())}  s_C {sorted(RUNS['s_C'].round(6).unique())}  "
      f"tau_0 {sorted(RUNS['tau_0'].unique())}")
print(f"{len(RUNS)} runs: {RUNS.ablation.nunique()} ablations x seed offsets {sorted(RUNS['seed offset'].unique())}")
ORDER = list(SUM.index)
JM, CEIL, SP = REF["job MILP $"], REF["feasible ceiling $"], REF["static Phase 0 $"]


def savefig(fig, name):
    fig.savefig(FIGS / f"{name}.png", dpi=200, bbox_inches="tight")

# %% [markdown]
# ## 1. Baseline check and references
#
# The baseline (all flags off, thinning seed offset 2) must reproduce the theory run of `small_cluster_nofilter.md`:
# DLENT-RT = 40,892.14 $. References (baseline config): job MILP, feasible ceiling, static Phase 0, static + thinning per
# seed (the same thinning draws as DLENT-RT with that seed).

# %%
b2 = RUNS[(RUNS.ablation == "baseline") & (RUNS["seed offset"] == 2)].iloc[0]
print(f"baseline, offset 2: R = {b2['R $']:,.2f} $  (expected 40,892.14)")
assert round(b2["R $"], 2) == 40_892.14
assert RUNS["s_C"].round(6).nunique() == 1 and RUNS["tau_0"].nunique() == 1 and RUNS["capacity"].nunique() == 1
refs = pd.Series({k: v for k, v in REF.items() if k.endswith("$")}, name="$")
display(refs.to_frame().style.format("{:,.2f}"))
print("job MILP:", REF["job MILP status"], "gap", REF["job MILP gap"])

# %% [markdown]
# ## 2–5. One section per ablation family
#
# Per level: the constants it changes (seed-independent), the reward over the 5 seeds (mean ± std), the paired
# difference vs the baseline with the same thinning seed (mean and 95% t-interval, 4 degrees of freedom), and the
# reward as % of the job MILP, of the feasible ceiling and of static Phase 0. Figure: reward per seed and level, with
# the baseline per seed as a line.

# %%
CONST = ["xi", "gamma", "eps_D (tests)", "eps_D (thinning)", "p_U cap", "p_U at phase-1 start", "p_U at t - tau = d_max",
         "eta(d_max)", "Test 2 factor 1-eps_D-xi", "eps_Test2(d_max)", "eps_Test3(d_max)", "Test 1 margin xi*c CPU",
         "Test 1 margin xi*c RAM"]
RES = ["R mean $", "R std $", "paired delta mean $", "paired delta 95% lo $", "paired delta 95% hi $",
       "beyond seed noise (95% CI excludes 0)", "% of job MILP (mean)", "% of feasible ceiling (mean)",
       "% of static Phase 0 (mean)"]
base_seed = RUNS[RUNS.ablation == "baseline"].set_index("seed offset")["R $"]


def family(fam, title, fig_name):
    names = ["baseline"] + [n for n in ORDER if SUM.loc[n, "family"] == fam]
    print(f"=== {title}")
    display(SUM.loc[names, ["level"] + CONST].T)
    display(SUM.loc[names, ["level"] + RES].style.format(precision=2))
    fig, ax = plt.subplots(figsize=(9, 3.6))
    seeds = sorted(RUNS["seed offset"].unique())
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        g = RUNS[RUNS.ablation == n].set_index("seed offset").reindex(seeds)
        ax.bar(np.arange(len(seeds)) + (i - (len(names) - 1) / 2) * w, g["R $"], w,
               label=f"{n} ({SUM.loc[n, 'level']})", color="dimgray" if n == "baseline" else None)
    ax.axhline(JM, color="k", ls="--", lw=0.8, label=f"job MILP {JM:,.0f}")
    ax.axhline(SP, color="darkorange", ls=":", lw=1.0, label=f"static Phase 0 {SP:,.0f}")
    ax.set_xticks(np.arange(len(seeds)), [f"offset {s}" for s in seeds])
    ax.set_ylabel("R $ (DLENT-RT)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout(); savefig(fig, fig_name); plt.show()


family("A1 thinning", "A1 — thinning relaxed (p_U only)", "A1_thinning")

# %%
family("A2a xi", "A2a — xi_override (gamma = 1)", "A2a_xi")

# %%
family("A2b gamma", "A2b — gamma_override (xi = 0.001)", "A2b_gamma")

# %%
family("A3 grid", "A3 — fixed grid across phases (Phase 0 grid kept)", "A3_fixed_grid")

# %% [markdown]
# ## 6. Ranking of the single-factor effects
#
# Paired difference vs the baseline (same thinning seed), mean over the 5 seeds with its 95% t-interval, in $ and as % of
# the job MILP. Bars whose interval excludes 0 are larger than the seed noise (dark); the others are not (light).

# %%
RANK = SUM.drop(index="baseline").sort_values("paired delta mean $", ascending=False)
display(RANK[["family", "level", "paired delta mean $", "paired delta 95% lo $", "paired delta 95% hi $",
              "paired delta % of job MILP", "beyond seed noise (95% CI excludes 0)", "% of job MILP (mean)",
              "% of static Phase 0 (mean)"]].style.format(precision=2))
fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
y = np.arange(len(RANK))[::-1]
err = np.vstack([RANK["paired delta mean $"] - RANK["paired delta 95% lo $"],
                 RANK["paired delta 95% hi $"] - RANK["paired delta mean $"]])
col = ["teal" if b else "lightsteelblue" for b in RANK["beyond seed noise (95% CI excludes 0)"]]
axes[0].barh(y, RANK["paired delta mean $"], xerr=err, color=col, capsize=3)
axes[0].set_yticks(y, [f"{n} ({RANK.loc[n, 'level']})" for n in RANK.index], fontsize=8)
axes[0].axvline(0, color="k", lw=0.7); axes[0].set_xlabel("paired Δ R vs baseline [$] (95% CI)")
axes[1].barh(y, RANK["paired delta % of job MILP"], xerr=100 * err / JM, color=col, capsize=3)
axes[1].set_yticks(y, [""] * len(y)); axes[1].axvline(0, color="k", lw=0.7)
axes[1].set_xlabel("paired Δ R vs baseline [% of job MILP]")
fig.suptitle("Single-factor ablation effects (dark = beyond the seed noise)", fontsize=10)
plt.tight_layout(); savefig(fig, "ranking"); plt.show()

# %% [markdown]
# ## 7. Micro-benchmarks per ablation (means over the 5 seeds)
#
# Phase end reasons, phases with τ > 0, whether Test 1 fires (number of runs out of 5), admitted jobs, thinning and
# physical rejects, and the reward split (customer value / profit / carbon).

# %%
MICRO = ["level", "phases (mean)", "end timeout (mean)", "end test1 (mean)", "end test2 (mean)", "end test3 (mean)",
         "end horizon (mean)", "phases with tau > 0 (mean)", "Test 1 fires (runs)", "admitted (mean)",
         "thinning rejects (mean)", "physical rejects (mean)", "customer value $ (mean)", "profit $ (mean)",
         "carbon $ (mean)"]
display(SUM[MICRO].style.format(precision=1))
fig, ax = plt.subplots(figsize=(12, 3.8))
x = np.arange(len(SUM))
for j, (k, c) in enumerate((("admitted (mean)", "seagreen"), ("thinning rejects (mean)", "darkorange"),
                            ("physical rejects (mean)", "indianred"))):
    ax.bar(x + (j - 1) * 0.27, SUM[k], 0.27, color=c, label=k)
ax.set_xticks(x, SUM.index, rotation=25, ha="right", fontsize=8); ax.legend(fontsize=8)
ax.set_title("Admissions and rejections per ablation (means over 5 seeds)", fontsize=10)
plt.tight_layout(); savefig(fig, "micro_rejects"); plt.show()

# %% [markdown]
# ## 8. Which guarantee each ablation gives up (stated from the code; the paper's proofs are not in the repo)

# %%
display(SUM[["level", "voids"]])

# %% [markdown]
# ## 9. All tables as markdown (`reports/week30/ablations/ablations_tables.md`)

# %%
def _fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, (float, np.floating)):
        return "nan" if not np.isfinite(v) else (f"{v:,.2f}" if abs(v) >= 1 or v == 0 else f"{v:.3g}")
    return str(v)


def md_table(df, index_name):
    head = "| " + index_name + " | " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * (len(df.columns) + 1)
    body = ["| " + " / ".join(map(str, k if isinstance(k, tuple) else (k,))) + " | "
            + " | ".join(_fmt(v) for v in row) + " |" for k, row in zip(df.index, df.itertuples(index=False))]
    return "\n".join([head, sep, *body])


md = ["# Ablations round 1: generated tables (notebooks/05_ablations_small_cluster.ipynb; do not edit by hand)\n",
      f"Notebook commit `{NB_COMMIT}`; runs at commit(s) {sorted(RUNS['commit'].unique())}; regime `{REGIME}`.\n",
      "## References\n", md_table(refs.to_frame(), "reference"), "",
      "## Constants per ablation\n", md_table(SUM[["level"] + CONST], "ablation"), "",
      "## Reward per ablation (5 thinning seeds)\n", md_table(SUM[["level"] + RES], "ablation"), "",
      "## Ranking (paired delta vs baseline)\n",
      md_table(RANK[["level", "paired delta mean $", "paired delta 95% lo $", "paired delta 95% hi $",
                     "paired delta % of job MILP", "beyond seed noise (95% CI excludes 0)"]], "ablation"), "",
      "## Micro-benchmarks (means over 5 seeds)\n", md_table(SUM[MICRO], "ablation"), "",
      "## Guarantee given up\n", md_table(SUM[["level", "voids"]], "ablation"), "",
      "## Every run\n", md_table(RUNS.set_index(["ablation", "seed offset"])[
          ["R $", "customer value $", "profit $", "carbon $", "admitted", "thinning rejects", "physical rejects",
           "phases", "end timeout", "end test1", "end test2", "end test3", "end horizon", "phases with tau > 0",
           "% of job MILP", "% of feasible ceiling", "% of static Phase 0"]], "ablation / seed"), ""]
(ADIR / "ablations_tables.md").write_text("\n".join(md))
print("markdown ->", ADIR / "ablations_tables.md")

# %% [markdown]
# ## 10. Cumulative round: the changes stacked in order
#
# Configurations (same regime, ξ = 0.001 everywhere, thinning seed offsets 2–6):
# * **C0** = theory (all flags off); **C1** = A1c (ε_D = 0 in p_U); **C2** = C1 + A3 (Phase 0 grid kept);
#   **C3** = C2 + γ = 0.3.
# * References: **R1** = A1a (thinning off); **R2** = A1a + A3.
#
# **R1 and R2 have no thinning randomness**: with p_U = 1 the run uses no random numbers, so all seeds give the same run.
# They were run ONCE; their seed std is 0, and their confidence intervals come only from the seed noise of the
# configuration they are compared with. They are NOT 5 independent replications.
#
# Paired differences: vs C0 and vs the previous step (C1 vs C0, C2 vs C1, C3 vs C2; R1 vs C0, R2 vs R1), same thinning
# seed, mean and 95% t-interval (4 degrees of freedom).

# %%
CDIR = ADIR
for f in ("cumulative_summary.csv", "cumulative_runs.csv", "cumulative_phase_tables.csv", "cumulative_checks.csv"):
    if not (CDIR / f).is_file():
        raise FileNotFoundError(f"{CDIR / f} missing: run `dlent_venv/bin/python scripts/ablation_runner.py cumulative`")
CSUM = pd.read_csv(CDIR / "cumulative_summary.csv", index_col=0)
CRUNS = pd.read_csv(CDIR / "cumulative_runs.csv")
CPH = pd.read_csv(CDIR / "cumulative_phase_tables.csv")
CCHK = pd.read_csv(CDIR / "cumulative_checks.csv")
print("runs' git HEAD:", sorted(CRUNS["run git HEAD"].unique()), " dirty:", sorted(CRUNS["run dirty"].unique()))
print("dirty files at run time:", sorted(CRUNS["run dirty files"].unique()))
print("C0 / C1 / R1 identical to round 1 (baseline / A1c / A1a), every seed:", bool(CCHK["identical"].all()))
display(CCHK)
c0 = CSUM.loc["C0", "R mean $"]
print(f"C0 mean R = {c0:,.2f} $ (round-1 baseline mean 39,739.06)")
assert round(c0, 2) == 39_739.06 and CCHK["identical"].all()

# %%
CCOLS = ["description", "runs", "R mean $", "R std $", "% of job MILP", "% of feasible ceiling", "% of static Phase 0",
         "vs C0: delta mean $", "vs C0: 95% lo $", "vs C0: 95% hi $", "vs C0: beyond seed noise",
         "vs previous step: previous", "vs previous step: delta mean $", "vs previous step: 95% lo $",
         "vs previous step: 95% hi $", "vs previous step: beyond seed noise"]
display(CSUM[CCOLS].style.format(precision=2))
display(CSUM[["seed note"]])
CMICRO = ["customer value $ (mean)", "profit $ (mean)", "carbon $ (mean)", "admitted (mean)", "thinning rejects (mean)",
          "physical rejects (mean)", "phases (mean)", "end timeout (mean)", "end test1 (mean)", "end test2 (mean)",
          "end test3 (mean)", "end horizon (mean)", "phases with tau > 0 (mean)", "Test 1 fires (runs)"]
display(CSUM[CMICRO].style.format(precision=1))

# %% [markdown]
# ### Figure: R by configuration
#
# Bars: mean R (error bars: ± one seed std; none for R1 / R2, which are deterministic). Lines: static Phase 0,
# static + thinning (mean over the 5 seeds), the bucketed MILP on reference buckets (the type-level offline benchmark,
# from the small-cluster nb03 run), the job-level MILP and the feasible ceiling.

# %%
nb03_small = (ROOT / "reports" / "week30" / "small_cluster_nb" / "nb03_numbers.md").read_text()
BUCKET_REF = 115_705.84                                     # bucketed MILP A (reference buckets), realised
assert "| bucketed MILP A (reference buckets) | 115,705.84 |" in nb03_small
ST_THIN = np.mean([REF[f"static + thinning $ (offset {o})"] for o in sorted(RUNS["seed offset"].unique())])
fig, ax = plt.subplots(figsize=(11, 4.6))
order_c = ["C0", "C1", "C2", "C3", "R1", "R2"]
x = np.arange(len(order_c))
cols = ["dimgray", "teal", "teal", "teal", "steelblue", "steelblue"]
ax.bar(x, CSUM.loc[order_c, "R mean $"], color=cols,
       yerr=[0 if c in ("R1", "R2") else CSUM.loc[c, "R std $"] for c in order_c], capsize=4)
for xi_, c in zip(x, order_c):
    ax.text(xi_, CSUM.loc[c, "R mean $"], f"{CSUM.loc[c, 'R mean $']:,.0f}\n{CSUM.loc[c, '% of job MILP']:.1f}%",
            ha="center", va="bottom", fontsize=7)
for val, lab, ls, col in ((SP, "static Phase 0", ":", "darkorange"), (ST_THIN, "static + thinning (mean)", ":", "sienna"),
                          (BUCKET_REF, "bucketed MILP (reference buckets)", "--", "purple"),
                          (JM, "job MILP", "--", "k"), (CEIL, "feasible ceiling", "-", "gray")):
    ax.axhline(val, color=col, ls=ls, lw=1.0, label=f"{lab} {val:,.0f}")
ax.set_xticks(x, [f"{c}\n{CSUM.loc[c, 'description']}" for c in order_c], fontsize=7)
ax.set_ylabel("R $ over the stream")
ax.set_title("Cumulative ablations (C0 → C3) and the thinning-off references (R1, R2; deterministic)", fontsize=10)
ax.legend(fontsize=7, loc="upper left", ncol=2)
ax.set_ylim(0, CEIL * 1.08)
plt.tight_layout(); savefig(fig, "cumulative"); plt.show()

# %% [markdown]
# ### Per-phase tables (seed offset 2): where the remaining loss comes from
#
# For C0, C1 and R1: phase index, how it ended, its duration (ticks), admitted jobs, reward, whether τ > 0, and p_U at
# the start of the phase (`ErrorParams.p_U(W_prev, 1)`; 1 for R1, where thinning is off).

# %%
for c in ("C0", "C1", "R1"):
    g = CPH[CPH.config == c][["phase", "end reason", "duration (ticks)", "arrivals", "admitted", "thinning rejects",
                              "physical rejects", "reward $", "tau > 0", "tau (cpu, ram)", "p_U at start"]]
    print(f"=== {c}: {CSUM.loc[c, 'description']} — {len(g)} phases, "
          f"{int((g['end reason'] == 'test1').sum())} ended by Test 1")
    display(g.set_index("phase").style.format(precision=4, subset=["p_U at start"]).format(precision=2, subset=["reward $"]))
summ_ph = (CPH.assign(test1=CPH["end reason"] == "test1")
           .groupby("config").agg(phases=("phase", "count"), ended_by_test1=("test1", "sum"),
                                  mean_duration=("duration (ticks)", "mean"),
                                  reward_in_test1_phases=("reward $", lambda r: r[CPH.loc[r.index, "end reason"] == "test1"].sum()),
                                  reward_total=("reward $", "sum")).loc[["C0", "C1", "R1"]])
display(summ_ph.style.format(precision=1))

# %%
md = ["# Ablations — cumulative round: generated tables (notebook 05, Section 10; do not edit by hand)\n",
      f"Runs' git HEAD {sorted(CRUNS['run git HEAD'].unique())}, dirty {sorted(CRUNS['run dirty'].unique())} "
      f"(dirty files: {sorted(CRUNS['run dirty files'].unique())}).\n",
      "## Identity checks vs round 1\n", md_table(CCHK.set_index(["config", "seed offset"]), "config / seed"), "",
      "## Configurations\n", md_table(CSUM[CCOLS], "config"), "",
      "## Seed note\n", md_table(CSUM[["seed note"]], "config"), "",
      "## Micro-benchmarks (means)\n", md_table(CSUM[CMICRO], "config"), "",
      "## Per-phase tables (seed offset 2)\n"]
for c in ("C0", "C1", "R1"):
    g = CPH[CPH.config == c].drop(columns="config").set_index("phase")
    md += [f"### {c}\n", md_table(g, "phase"), ""]
md += ["### Phases ended by Test 1\n", md_table(summ_ph, "config"), "",
       "## Every run\n", md_table(CRUNS.set_index(["ablation", "seed offset"])[
           ["R $", "customer value $", "profit $", "carbon $", "admitted", "thinning rejects", "physical rejects",
            "phases", "end timeout", "end test1", "end test2", "end test3", "end horizon", "phases with tau > 0"]],
           "config / seed"), ""]
(CDIR / "cumulative_tables.md").write_text("\n".join(md))
print("markdown ->", CDIR / "cumulative_tables.md")

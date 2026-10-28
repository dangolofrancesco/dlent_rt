# Week 30: notebooks 03 and 04 on the named regime `configs/regime_small_cluster.yaml`

**What was run:** the 30k trace, seed 42.
* **Config:** the committed `configs/default.yaml` (`git show HEAD`), with ONLY the `capacity` block of
  `configs/regime_small_cluster.yaml` applied (also read from git HEAD, through `scripts/regime_config.py`).
* **Run markers:** commit `3549fc9`, configs/default.yaml blob `68e7bd7`, regime blob `a0d29b5`.
* **Checked:** the loaded cluster is c = (0.4014237441083392, 0.2616208305922613) exactly, with τ₀ = (0, 0.020386)
  and s_C = 228.1843.
* **Phase 0, rewards, defaults and committed configs are unchanged. No Phase 0 change.**

**Outputs:**
* **Notebooks** (one source each; the regime is chosen with `DLENT_REGIME`, and every output goes to its own folders):
  * `notebooks/03_main_algorithm_analysis_small_cluster.ipynb`;
  * `notebooks/04_pareto_lambda_small_cluster.ipynb`;
  * the default-regime `03_main_algorithm_analysis.ipynb` and `04_pareto_lambda.ipynb`, rebuilt from the same sources.
* **PNGs:** `reports/week30/figs_small_cluster/` (26 files, nb03 and nb04).
* **Tables:** `reports/week30/small_cluster_nb/` (nb03) and `reports/week30/pareto_small_cluster/` (nb04 cache and
  tables).

## Comparison with `small_cluster_nofilter.md` (f = 0.15)

**Every number matches to the cent:**

| | reference $ | this run $ |
|---|---|---|
| job MILP | 120,742.02 | 120,742.02 |
| DLENT-RT | 40,892.14 | 40,892.14 |
| G2 | 107,551.45 | 107,551.45 |
| G1 | 64,503.43 | 64,503.43 |
| static Phase 0 | 44,716.94 | 44,716.94 |
| fluid OPT | 146,395.19 | 146,395.19 |
| feasible ceiling | 141,126.62 | 141,126.62 |

* nb04 at λ = (1, 1, 1) gives the same DLENT-RT run (40,892.14; 6 timeout / 1 Test 1 / 1 Test 2 / 1 horizon).
* The default-regime nb03 still matches its references to the cent (job OPT 155,397.61, DLENT-RT 48,391.61, …).
* No difference found.

## nb03 on the small cluster (scale 1.0)

| row | realised $ | % of job MILP | % of feasible ceiling |
|---|---|---|---|
| job-level, no capacity | 155,579.35 | 128.85 | 110.24 |
| **feasible ceiling** (Σ R_job⁺ over A ≤ c) | 141,126.62 | 116.88 | 100.00 |
| job-level MILP | 120,742.02 | 100.00 | 85.56 |
| bucketed MILP A (reference buckets) | 115,705.84 | 95.83 | 81.99 |
| bucketed MILP A (Phase 0 buckets) | 49,134.15 | 40.69 | 34.82 |
| fluid OPT | 146,395.19 | 121.25 | 103.73 |
| **fluid OPT on feasible types only** | 135,400.78 | 112.14 | 95.94 |
| G2 job greedy | 107,551.45 | 89.08 | 76.21 |
| G1 type greedy | 64,503.43 | 53.42 | 45.71 |
| static Phase 0 | 44,716.94 | 37.04 | 31.69 |
| static Phase 0 + thinning | 31,285.95 | 25.91 | 22.17 |
| DLENT-RT (theory) | 40,892.14 | 33.87 | 28.98 |

* **The fluid OPT on feasible types only has τ = (0, 0).** Removing the 504 populated types whose hardware profile
  exceeds c makes the fluid LP slack. The τ > 0 of this cluster comes entirely from load that can never run; with every
  type, τ* = (0, 0.0425).
* **static → static + thinning → DLENT-RT:**
  * thinning costs −13,431 $ (37.0% → 25.9% of the job MILP);
  * learning / estimation (ReInit, tests, phase-wise thinning) recovers +9,606 $ (→ 33.9%).
* **Decomposition with the reference buckets (believed):**
  * discretization cost 5,283 $;
  * fluid cost −30,936 $: the fluid OPT is above the bucketed MILP, by more than on the default cluster (−160 $),
    because it counts the oversize load.
* **Oversize jobs:**
  * 1,680 stream jobs (10.97%) and 2,158 H jobs (21.58%);
  * 43.9% (CPU) and 37.4% (RAM) of Phase 0's profitable load;
  * ρ = 1.23 / 1.29 with them, and 0.69 / 0.81 without them;
  * 14,452.73 $ of the stream's Σ R_job⁺.
* **Physical rejects** (% of arrivals; the part that is oversize jobs in brackets): DLENT-RT 3.5% (18%), static 4.4%
  (9%), static + thinning 2.1% (13%), G1 8.4% (13%), G2 14.5% (17%). Nothing larger than c is ever admitted (asserted).
* **Reward split:** customer value dominates in every model, and carbon is −3.4k to −7.5k $. All rows are in
  Appendix A.

## nb04 (Pareto) on the small cluster

* **Unchanged by the regime:** the job components (CV, PR, CO) and s_C = 228.18. There is no drop filter, so the
  jobs, the bids and Phase 0's quantiles are the same. This was checked for all 13 Phase 0 rebuilds (s_C, grid, hardware
  profiles and job components identical).
* **Rerun with the new c:** the fluid LP, the job MILP (16 + 6 λ, all optimal; 392 s in total, 84 s at most), G2, the
  standard and legacy Chebyshev, StaticPolicy, and DLENT-RT on the 7-point subset plus the 6 perturbations (13 runs,
  4 at a time).
* **Which λ matters least on THIS cluster:** **λ3 (carbon)** for the fluid LP, G2, the job MILP and StaticPolicy, with
  jobs changed over ×2 and ×0.5 of 445 / 837 / 563 / 342. **λ2** for DLENT-RT (1,449).
  * On the default cluster λ2 mattered least for every model. With the capacity binding, carbon moves fewer jobs.
  * DLENT-RT's counts (1,449–1,869) come from one thinning seed.
* **Corners:**
  * (0, 0, 3): every weighted-sum model admits nothing. The standard Chebyshev admits 5 near-zero-carbon types (the
    augmentation tie-break).
  * (3, 0, 0): the fluid LP admits a FRACTION of the types (capacity binds on average), unlike the default cluster.
* **Policies vs the frontier:** at λ = (1, 1, 1), DLENT-RT is at 27.9% of the fluid OPT and 33.9% of the job MILP; its
  normalised distance to the offline Pareto set is d₂ = 0.203.
* **Weighted sum vs Chebyshev (fluid LP):**
  * 0 of the 16 Chebyshev points coincide with a weighted-sum grid point within 10⁻⁶; 6 are within 5·10⁻⁴.
  * All are non-dominated: the dominance slack is at most 5·10⁻⁴ of the range.
  * The fluid front is convex (an LP), so no non-convexity is claimed.
* **Hypervolume** (7-point subset, ×10¹³ $³): job MILP 4.10, Chebyshev 3.81, G2 3.49, fluid 3.20, StaticPolicy 0.58,
  DLENT-RT 0.47.

## Anything odd

* **The first nb03 attempt failed at Section 9** because an import was missing (my patch had not been saved). It was
  fixed and the notebook rerun; pyflakes now checks the sources for undefined names.
* **The first Pareto attempt crashed at λ = (0.75, 2.25, 0).** The dominance-slack LP of a Chebyshev point on the
  capacity boundary was numerically infeasible.
  * Fix: `pareto.fluid_dominance_slack` re-solves once with the lower bounds relaxed by 10⁻⁷ of each objective's range
    (NaN if that also fails).
  * Result: no NaN in either regime; the unit tests are unchanged.
* **The Mac slept overnight** during the first chain (on battery). The chain is now run under `caffeinate` on AC.
* **nb03 Section 8 pool:** capped at 4 workers (it used 6); its numbers are unchanged.
* **Regime-independent sections:** on the small regime, nb03 Section 8 (capacity × grid sweep) and Section 11 (f = 0.2
  with the filter) override the capacity themselves, so they reproduce the default-regime numbers.

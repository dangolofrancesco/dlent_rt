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


---

# Appendix A — notebook 03 tables (generated)

## nb03 numbers — regime configs/regime_small_cluster.yaml (generated by the notebook; do not edit by hand)

Commit `3549fc97caa8d93b2d86262bc166391a07aad322`, configs/default.yaml blob `68e7bd76b57c376cae6d74139f11cc456559520b` (the COMMITTED config + the capacity block of `configs/regime_small_cluster.yaml`, blob `a0d29b5313ea41184961aa085366bc988c5dd7d0`, via a temporary file); working-copy default.yaml edited: True (not used). Seed 42, σ = 0.3, n0 = 10,000. Generated 2026-10-23 13:58.

### Check against the reference numbers

| row | expected $ | this run $ | this run (cents) | matches to the cent |
|---|---|---|---|---|
| job-level MILP | 120,742.02 | 120,742.02 | 120,742.02 | 1 |
| DLENT-RT (theory) | 40,892.14 | 40,892.14 | 40,892.14 | 1 |
| G2 job greedy | 107,551.45 | 107,551.45 | 107,551.45 | 1 |
| G1 type greedy | 64,503.43 | 64,503.43 | 64,503.43 | 1 |
| static Phase 0 | 44,716.94 | 44,716.94 | 44,716.94 | 1 |
| fluid OPT | 146,395.19 | 146,395.19 | 146,395.19 | 1 |
| feasible ceiling (Σ R_job⁺ over A ≤ c) | 141,126.62 | 141,126.62 | 141,126.62 | 1 |

### Section 9: benchmark ladder (scale 1.0)

| row | realised $ | % of job-level MILP | % of feasible ceiling | gap to row above $ |
|---|---|---|---|---|
| job-level, no capacity (Σ R_job⁺) | 155,579.35 | 128.85 | 110.24 | nan |
| feasible ceiling (Σ R_job⁺ over A ≤ c) | 141,126.62 | 116.88 | 100.00 | 14,452.73 |
| job-level MILP | 120,742.02 | 100.00 | 85.56 | 20,384.60 |
| bucketed MILP A (reference buckets) | 115,705.84 | 95.83 | 81.99 | 5,036.17 |
| bucketed MILP A (Phase 0 buckets) | 49,134.15 | 40.69 | 34.82 | 66,571.69 |
| fluid OPT | 146,395.19 | 121.25 | 103.73 | -97,261.04 |
| fluid OPT on feasible types only | 135,400.78 | 112.14 | 95.94 | 10,994.41 |
| G2 job greedy | 107,551.45 | 89.08 | 76.21 | 27,849.33 |
| G1 type greedy | 64,503.43 | 53.42 | 45.71 | 43,048.02 |
| static Phase 0 | 44,716.94 | 37.04 | 31.69 | 19,786.50 |
| static Phase 0 + thinning | 31,285.95 | 25.91 | 22.17 | 13,430.99 |
| DLENT-RT (theory) | 40,892.14 | 33.87 | 28.98 | -9,606.19 |

fluid OPT on feasible types only: {'fluid feasible-types tau': [0.0, 0.0], 'fluid feasible-types tau is 0': True, 'fluid (all types) tau': [0.0, 0.042513], 'types removed (A > c), populated': 504, 'types populated': 3033, 'stream jobs that fit': 13628, 'stream jobs': 15308}

#### static → static + thinning → DLENT-RT

| step | $ | % of job MILP | change $ | attributed to |
|---|---|---|---|---|
| static Phase 0 | 44,716.94 | 37.04 | nan | nan |
| static Phase 0 + thinning | 31,285.95 | 25.91 | -13,430.99 | cost of thinning (same frozen oracle) |
| DLENT-RT (theory) | 40,892.14 | 33.87 | 9,606.19 | learning / estimation (ReInit, tests, phase-wise thinning) |

#### Decomposition

| buckets / bucketed value | bucketed $ | discretization cost (job OPT − bucketed) $ | fluid cost (bucketed − fluid) $ | sum = job OPT − fluid $ |
|---|---|---|---|---|
| reference / believed Σ r̃·x | 115,459.27 | 5,282.75 | -30,935.93 | -25,653.18 |
| reference / realised Σ R·x | 115,705.84 | 5,036.17 | -30,689.35 | -25,653.18 |
| phase0 / believed Σ r̃·x | 57,034.21 | 63,707.81 | -89,360.98 | -25,653.18 |
| phase0 / realised Σ R·x | 49,134.15 | 71,607.87 | -97,261.04 | -25,653.18 |

#### MILPs

| MILP | status | gap | runtime s | candidates | unseen jobs | unseen types | chosen with R_job<0 | peak true occ / c |
|---|---|---|---|---|---|---|---|---|
| job-level | Optimization terminated successfully. (HiGHS Status 7: Optimal) | 9.91e-05 | 14.11 | nan | nan | nan | nan | nan |
| bucketed A (reference) | Optimization terminated successfully. (HiGHS Status 7: Optimal) | 9.92e-05 | 28.87 | 12,621.00 | 0.00 | 0.00 | 439.00 | [0.9999, 0.9999] |
| bucketed A (phase0) | Optimization terminated successfully. (HiGHS Status 7: Optimal) | 1.86e-07 | 0.447 | 10,227.00 | 2,762.00 | 1,155.00 | 564.00 | [0.9991, 0.9994] |

### Section 10: reward split per model (scale 1.0)

| row | λ1·customer value $ | λ2·profit $ | −λ3·carbon $ | R $ | admitted |
|---|---|---|---|---|---|
| job-level MILP | 94,609.09 | 33,324.04 | -7,191.12 | 120,742.02 | 10,775 |
| bucketed MILP A (reference buckets) | 94,800.51 | 28,095.63 | -7,190.30 | 115,705.84 | 10,705 |
| bucketed MILP A (Phase 0 buckets) | 44,421.43 | 9,665.20 | -4,952.47 | 49,134.15 | 9,339 |
| G2 job greedy | 84,155.28 | 30,884.85 | -7,488.67 | 107,551.45 | 10,616 |
| G1 type greedy | 59,095.82 | 11,149.13 | -5,741.51 | 64,503.43 | 10,377 |
| static Phase 0 | 40,572.86 | 8,814.15 | -4,670.07 | 44,716.94 | 9,237 |
| static Phase 0 + thinning | 27,808.94 | 6,830.80 | -3,353.79 | 31,285.95 | 6,253 |
| DLENT-RT (theory) | 37,633.19 | 7,317.65 | -4,058.71 | 40,892.14 | 6,657 |

### Section 10b: oversize jobs

| quantity | value |
|---|---|
| c (norm) CPU, RAM | [0.401424, 0.261621] |
| oversize stream jobs | 1,680 (10.97%) |
| oversize H jobs | 2,158 (21.58%) |
| share of P0 profitable load from oversize, CPU % | 43.86 |
| share of P0 profitable load from oversize, RAM % | 37.38 |
| rho = P0 profitable load / c, CPU | 1.23 |
| rho, RAM | 1.29 |
| rho without oversize, CPU | 0.692 |
| rho without oversize, RAM | 0.807 |
| stream Σ R_job⁺ of oversize jobs $ | 14,452.73 |

| model | admitted | physical rejects | physical rejects % of arrivals | of which oversize jobs | oversize share of its physical rejects % |
|---|---|---|---|---|---|
| DLENT-RT (online) | 6,657 | 537 | 3.51 | 99 | 18.44 |
| static tau* (full stream, ORACLE) | 10,478 | 1,427 | 9.32 | 150 | 10.51 |
| static Phase 0 (H) | 9,237 | 670 | 4.38 | 60 | 8.96 |
| static Phase 0 (H) + thinning at cap | 6,253 | 327 | 2.14 | 42 | 12.84 |
| G1 (type greedy, cumulative r~) | 10,377 | 1,290 | 8.43 | 167 | 12.95 |
| G2 (job greedy, R_job > 0) | 10,616 | 2,224 | 14.53 | 383 | 17.22 |

### Section 11: PART 2, f = 0.2, q = 0.7, drop_jobs_exceeding_capacity

c = [0.53523, 0.34883], dropped {'n_H': 1423, 'n_stream': 977}, |H| = 8,577, |stream| = 14,331, s_C = 533.4979, τ₀ = [0.0, 0.0], DLENT-RT phases 6 {'test1': 0, 'test2': 0, 'test3': 0, 'timeout': 5, 'horizon': 1}.

| row | realised $ | % of job-level MILP | % of feasible ceiling | gap to row above $ |
|---|---|---|---|---|
| job-level, no capacity (Σ R_job⁺) | 123,757.69 | 110.20 | 100.00 | nan |
| feasible ceiling (Σ R_job⁺ over A ≤ c) | 123,757.69 | 110.20 | 100.00 | -1.46e-11 |
| job-level MILP | 112,304.68 | 100.00 | 90.75 | 11,453.01 |
| bucketed MILP A (reference buckets) | 108,479.99 | 96.59 | 87.66 | 3,824.69 |
| bucketed MILP A (Phase 0 buckets) | 46,528.00 | 41.43 | 37.60 | 61,951.99 |
| fluid OPT | 119,169.15 | 106.11 | 96.29 | -72,641.15 |
| fluid OPT on feasible types only | 119,169.15 | 106.11 | 96.29 | -0.00 |
| G2 job greedy | 102,752.91 | 91.49 | 83.03 | 16,416.24 |
| G1 type greedy | 65,408.71 | 58.24 | 52.85 | 37,344.21 |
| static Phase 0 | 44,733.34 | 39.83 | 36.15 | 20,675.37 |
| static Phase 0 + thinning | 31,032.86 | 27.63 | 25.08 | 13,700.48 |
| DLENT-RT (theory) | 38,222.30 | 34.03 | 30.88 | -7,189.44 |

fluid OPT on feasible types only: {'fluid feasible-types tau': [0.0, 0.0], 'fluid feasible-types tau is 0': True, 'fluid (all types) tau': [0.0, 0.0], 'types removed (A > c), populated': 0, 'types populated': 2758, 'stream jobs that fit': 14331, 'stream jobs': 14331}

#### static → static + thinning → DLENT-RT

| step | $ | % of job MILP | change $ | attributed to |
|---|---|---|---|---|
| static Phase 0 | 44,733.34 | 39.83 | nan | nan |
| static Phase 0 + thinning | 31,032.86 | 27.63 | -13,700.48 | cost of thinning (same frozen oracle) |
| DLENT-RT (theory) | 38,222.30 | 34.03 | 7,189.44 | learning / estimation (ReInit, tests, phase-wise thinning) |

#### Decomposition

| buckets / bucketed value | bucketed $ | discretization cost (job OPT − bucketed) $ | fluid cost (bucketed − fluid) $ | sum = job OPT − fluid $ |
|---|---|---|---|---|
| reference / believed Σ r̃·x | 108,122.58 | 4,182.09 | -11,046.57 | -6,864.47 |
| reference / realised Σ R·x | 108,479.99 | 3,824.69 | -10,689.16 | -6,864.47 |
| phase0 / believed Σ r̃·x | 52,019.21 | 60,285.47 | -67,149.94 | -6,864.47 |
| phase0 / realised Σ R·x | 46,528.00 | 65,776.68 | -72,641.15 | -6,864.47 |

#### MILPs

| MILP | status | gap | runtime s | candidates | unseen jobs | unseen types | chosen with R_job<0 | peak true occ / c |
|---|---|---|---|---|---|---|---|---|
| job-level | Optimization terminated successfully. (HiGHS Status 7: Optimal) | 4.86e-05 | 1.83 | nan | nan | nan | nan | nan |
| bucketed A (reference) | Optimization terminated successfully. (HiGHS Status 7: Optimal) | 9.4e-05 | 2.14 | 11,551.00 | 0.00 | 0.00 | 509.00 | [0.9997, 0.9992] |
| bucketed A (phase0) | Optimization terminated successfully. (HiGHS Status 7: Optimal) | 1.48e-05 | 0.13 | 9,561.00 | 2,621.00 | 1,042.00 | 697.00 | [0.9989, 0.9995] |

#### Reward split

| row | λ1·customer value $ | λ2·profit $ | −λ3·carbon $ | R $ | admitted |
|---|---|---|---|---|---|
| job-level MILP | 97,536.32 | 32,531.69 | -17,763.33 | 112,304.68 | 10,962 |
| bucketed MILP A (reference buckets) | 97,508.19 | 28,815.85 | -17,844.04 | 108,479.99 | 10,798 |
| bucketed MILP A (Phase 0 buckets) | 44,529.94 | 11,069.55 | -9,071.50 | 46,528.00 | 9,295 |
| G2 job greedy | 90,321.61 | 30,463.57 | -18,032.27 | 102,752.91 | 10,765 |
| G1 type greedy | 64,145.60 | 13,423.52 | -12,160.41 | 65,408.71 | 10,468 |
| static Phase 0 | 43,147.84 | 10,662.82 | -9,077.32 | 44,733.34 | 9,304 |
| static Phase 0 + thinning | 30,311.74 | 7,395.50 | -6,674.39 | 31,032.86 | 6,250 |
| DLENT-RT (theory) | 37,432.84 | 8,346.63 | -7,557.17 | 38,222.30 | 6,507 |

### Section 8: capacity × type-grid sweep (summary)

| quantity | (0.3, 'default (k*=128, hw=32)') | (0.3, 'reduced (k*=64, hw=16)') | (0.5, 'default (k*=128, hw=32)') | (0.5, 'reduced (k*=64, hw=16)') | (1.0, 'default (k*=128, hw=32)') | (1.0, 'reduced (k*=64, hw=16)') |
|---|---|---|---|---|---|---|
| c CPU (cores) | 51.38 | 51.05 | 85.64 | 85.09 | 171.27 | 170.17 |
| c RAM (GB) | 133.95 | 130.69 | 223.25 | 217.82 | 446.50 | 435.63 |
| would drop: stream jobs | 977.00 | 1,008.00 | 523.00 | 447.00 | 0.00 | 0.00 |
| tau_0 CPU | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| tau_0 RAM | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| phases | 6.00 | 8.00 | 6.00 | 9.00 | 11.00 | 12.00 |
| end timeout | 5.00 | 6.00 | 5.00 | 7.00 | 9.00 | 9.00 |
| end test1 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 1.00 |
| end test2 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| end test3 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| end horizon | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| phases with tau>0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| R DLENT-RT $ | 40,740.42 | 52,546.16 | 41,909.92 | 53,342.75 | 48,242.27 | 68,419.24 |
| R StaticPolicy (P0) $ | 46,525.88 | 66,757.56 | 44,053.62 | 71,390.21 | 54,121.64 | 85,655.07 |
| R G1 $ | 68,840.45 | 85,794.41 | 71,740.30 | 97,515.47 | 82,310.77 | 111,975.23 |
| R G2 $ | 113,505.78 | 115,963.36 | 126,304.02 | 130,643.17 | 153,582.18 | 155,002.13 |
| R fluid OPT $ | 119,169.15 | 118,524.21 | 127,443.36 | 130,025.90 | 149,478.42 | 148,855.58 |
| R job OPT (MILP) $ | 118,745.22 | 121,527.05 | 129,670.85 | 134,602.10 | 154,637.33 | 156,438.37 |
| DLENT-RT % of fluid | 34.19 | 44.33 | 32.89 | 41.02 | 32.27 | 45.96 |
| StaticPolicy (P0) % of fluid | 39.04 | 56.32 | 34.57 | 54.90 | 36.21 | 57.54 |
| G1 % of fluid | 57.77 | 72.39 | 56.29 | 75.00 | 55.07 | 75.22 |
| G2 % of fluid | 95.25 | 97.84 | 99.11 | 100.47 | 102.75 | 104.13 |
| DLENT-RT % of job OPT | 34.31 | 43.24 | 32.32 | 39.63 | 31.20 | 43.74 |
| StaticPolicy (P0) % of job OPT | 39.18 | 54.93 | 33.97 | 53.04 | 35.00 | 54.75 |
| G1 % of job OPT | 57.97 | 70.60 | 55.32 | 72.45 | 53.23 | 71.58 |
| G2 % of job OPT | 95.59 | 95.42 | 97.40 | 97.06 | 99.32 | 99.08 |
| discretization loss (job OPT - fluid) $ | -423.93 | 3,002.84 | 2,227.49 | 4,576.20 | 5,158.91 | 7,582.79 |
| decision errors P0 types % | 11.55 | 13.31 | 12.21 | 11.17 | 12.48 | 11.99 |
| decision errors G1 % | 11.60 | 12.41 | 11.57 | 11.09 | 12.08 | 11.08 |
| H jobs per populated type | 3.90 | 8.98 | 3.88 | 8.87 | 3.74 | 8.68 |
| MILP gap | 4.42e-05 | 2.88e-05 | 2.66e-05 | 5.4e-08 | 1.93e-05 | 1.66e-05 |

### table00_setup

|  | value |
|---|---|
| trace | batch_may2019_30k.csv |
| grid data | grid_data_may_2019.csv |
| data.n0 (notebook override) | 10000 |
| seed run (Phase-0 bids) | 42 |
| seed stream bids | 43 |
| seed thinning | 44 |
| |H| | 10000 |
| |stream| | 15308 |
| T (ticks) | 1432910 |
| tick seconds | 1.0 |
| d_max (ticks) | 14304 |
| c_cpu | 0.4014237441083392 |
| c_ram | 0.2616208305922613 |
| capacity mode | absolute |
| capacity fraction | 0.4 |
| capacity concurrency_quantile | 0.8 |
| a_max_cpu | 1.972725906898494 |
| a_max_ram | 1.27754913741676 |
| xi_data | 4.914322921481396 |
| xi_used | 0.001 |
| gamma_used | 1.0 |
| gamma_data | 14304.0 |
| delta | 3.398938130502317e-19 |
| eps_D | 0.5022202415073336 |
| thinning cap (1-xi)/(1+eps_D) | 0.6650156697380136 |
| p_U min | 0.18650219675758256 |
| p_U max | 0.6601055506306498 |
| lambda1 | 1.0 |
| lambda2 | 1.0 |
| lambda3 | 1.0 |
| lambda sum | 3.0 |
| s_C ($/kgCO2, frozen) | 228.18428294842903 |
| normalization | quantile_linear |
| quantile q | 0.9 |
| bid model | lognormal |
| bid sigma | 0.3 |
| u0 | 23.926 |
| k_star | 128 |
| hw profiles | 32 |
| model | theory |
| n_type_min | 5 |
| n_min | 80 |

### table00b_constants_per_scale

|  | 0.5 | 1.0 |
|---|---|---|
| c_cpu | 0.2007118720541696 | 0.4014237441083392 |
| c_ram | 0.13081041529613066 | 0.2616208305922613 |
| xi_data | 9.828645842962793 | 4.914322921481396 |
| xi_used | 0.001 | 0.001 |
| eps_D | 0.5022202415073336 | 0.5022202415073336 |
| gamma_used | 1.0 | 1.0 |
| p_U cap (1-xi)/(1+eps_D) | 0.6650156697380136 | 0.6650156697380136 |
| eta(1) = gamma/c_min | 7.644651213255339 | 3.8223256066276696 |
| p_U at phase-1 start | 0.10845974855482277 | 0.18650219675758256 |
| p_U at t-tau=d_max | 0.6128616533187865 | 0.6378743804039215 |
| Test1 thr c-a_eff (cpu) | 0.20051116018211543 | 0.40102232036423086 |
| Test1 thr c-a_eff (ram) | 0.13067960488083452 | 0.26135920976166904 |
| stream jobs with A>c (never fit) | 2650 | 1680 |
| share A>c % | 17.31120982492814 | 10.974653775803501 |
| s_C $/kg | 228.18428294842903 | 228.18428294842903 |
| tau*_P0 | [0.       0.102355] | [0.       0.020386] |

### table01_objective_quantiles

|  | objective | min | p1 | p10 | p50 | p90 | p99 | max | share<0 % |
|---|---|---|---|---|---|---|---|---|---|
| job | CV | 0.141 | 0.242 | 0.378 | 1.93 | 20.94 | 119.24 | 1,150.04 | 0.00 |
| job | Pi | -1,907.55 | -24.71 | -0.561 | 0.672 | 7.38 | 51.41 | 343.73 | 17.09 |
| job | C$ | 1.88e-09 | 0.00321 | 0.00972 | 0.242 | 8.19 | 151.71 | 876.26 | 0.00 |
| type | CV | 0.141 | 0.24 | 0.371 | 1.94 | 20.82 | 119.24 | 1,150.04 | 0.00 |
| type | Pi | -712.48 | -18.96 | -0.823 | 0.383 | 6.31 | 50.54 | 343.73 | 24.16 |
| type | C$ | 1.88e-09 | 0.00344 | 0.0101 | 0.252 | 8.76 | 148.69 | 876.26 | 0.00 |

### table02_negative_reward

|  | scope | n | R<0 % | R<0 no carbon % |
|---|---|---|---|---|
| job | all phases | 15,308 | 16.12 | 6.66 |
| job | phase 1 | 152 | 23.03 | 9.21 |
| job | phase 2 | 400 | 21.00 | 4.00 |
| job | phase 3 | 684 | 21.05 | 7.60 |
| job | phase 4 | 1,463 | 20.78 | 7.31 |
| job | phase 5 | 5,185 | 15.45 | 6.87 |
| job | phase 6 | 6,232 | 15.29 | 6.45 |
| job | phase 7 | 218 | 14.68 | 8.72 |
| job | phase 8 | 470 | 11.28 | 6.17 |
| job | phase 9 | 504 | 12.30 | 4.76 |
| type | phase 1 state | 2,673 | 26.00 | 7.53 |
| type | phase 2 state | 139 | 30.26 | 14.47 |
| type | phase 3 state | 350 | 30.25 | 11.50 |
| type | phase 4 state | 543 | 20.76 | 7.31 |
| type | phase 5 state | 934 | 28.16 | 10.46 |
| type | phase 6 state | 1,773 | 16.70 | 7.00 |
| type | phase 7 state | 2,194 | 16.43 | 7.28 |
| type | phase 8 state | 204 | 17.89 | 12.39 |
| type | phase 9 state | 400 | 14.47 | 8.51 |
| type | reference (full stream) | 3,033 | 17.55 | 6.56 |

### table03_phase_tau_points

|  | tau_cpu | tau_ram | arrivals | points plotted | subsampled |
|---|---|---|---|---|---|
| 1 | 0.00 | 0.0204 | 152 | 152 | 0 |
| 2 | 0.00 | 0.00 | 400 | 400 | 0 |
| 3 | 0.0201 | 0.00 | 684 | 684 | 0 |
| 4 | 0.00 | 0.0196 | 1,463 | 800 | 1 |
| 5 | 0.00 | 0.00 | 5,185 | 800 | 1 |
| 6 | 0.00 | 0.00 | 6,232 | 800 | 1 |
| 7 | 0.00444 | 0.00666 | 218 | 218 | 0 |
| 8 | 0.0941 | 0.00 | 470 | 470 | 0 |
| 9 | 0.05 | 0.00 | 504 | 504 | 0 |

### table04_phase_results

|  | start | end | steps | jobs | end reason | admitted | rej IR | rej oracle | rej thinning | rej physical | adm rate % | R $ | l1*CV $ | l2*Pi $ | -l3*s_C*K $ | adm R<0 | adm R<0 % | occ cpu mean % | occ ram mean % | occ cpu peak % | occ ram peak % | tau_cpu start | tau_ram start |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.00 | 28,608.00 | 28,608.00 | 152.00 | timeout | 64.00 | 0.00 | 48.00 | 40.00 | 0.00 | 42.11 | 254.16 | 239.66 | 49.71 | -35.21 | 3.00 | 4.69 | 9.72 | 8.50 | 71.02 | 75.88 | 0.00 | 0.0204 |
| 2 | 28,609.00 | 85,824.00 | 57,216.00 | 400.00 | timeout | 136.00 | 0.00 | 172.00 | 81.00 | 11.00 | 34.00 | 809.64 | 649.04 | 270.42 | -109.82 | 4.00 | 2.94 | 16.77 | 16.72 | 96.13 | 93.82 | 0.00 | 0.00 |
| 3 | 85,825.00 | 200,256.00 | 114,432.00 | 684.00 | timeout | 242.00 | 0.00 | 286.00 | 134.00 | 22.00 | 35.38 | 1,930.99 | 1,518.08 | 582.48 | -169.58 | 11.00 | 4.55 | 12.75 | 14.39 | 88.29 | 99.40 | 0.0201 | 0.00 |
| 4 | 200,257.00 | 429,120.00 | 228,864.00 | 1,463.00 | timeout | 550.00 | 0.00 | 535.00 | 338.00 | 40.00 | 37.59 | 2,833.82 | 2,693.45 | 666.24 | -525.88 | 44.00 | 8.00 | 19.90 | 22.78 | 98.48 | 98.80 | 0.00 | 0.0196 |
| 5 | 429,121.00 | 886,848.00 | 457,728.00 | 5,185.00 | timeout | 2,341.00 | 0.00 | 1,416.00 | 1,287.00 | 141.00 | 45.15 | 11,868.45 | 10,257.43 | 2,790.69 | -1,179.67 | 140.00 | 5.98 | 23.06 | 21.52 | 99.73 | 99.66 | 0.00 | 0.00 |
| 6 | 886,849.00 | 1,365,570.00 | 478,722.00 | 6,232.00 | test1 | 2,820.00 | 0.00 | 1,534.00 | 1,590.00 | 288.00 | 45.25 | 19,567.58 | 17,874.78 | 3,444.08 | -1,751.28 | 172.00 | 6.10 | 30.07 | 26.85 | 99.74 | 99.90 | 0.00 | 0.00 |
| 7 | 1,365,571.00 | 1,380,020.00 | 14,450.00 | 218.00 | test2 | 103.00 | 0.00 | 60.00 | 46.00 | 9.00 | 47.25 | 764.16 | 1,013.44 | -181.73 | -67.55 | 7.00 | 6.80 | 39.69 | 28.21 | 99.68 | 99.90 | 0.00444 | 0.00666 |
| 8 | 1,380,021.00 | 1,408,920.00 | 28,900.00 | 470.00 | timeout | 210.00 | 0.00 | 141.00 | 110.00 | 9.00 | 44.68 | 500.15 | 1,414.97 | -792.44 | -122.38 | 15.00 | 7.14 | 34.62 | 23.20 | 99.69 | 93.72 | 0.0941 | 0.00 |
| 9 | 1,408,921.00 | 1,432,910.00 | 23,990.00 | 504.00 | horizon | 191.00 | 0.00 | 178.00 | 118.00 | 17.00 | 37.90 | 2,363.20 | 1,972.34 | 488.20 | -97.34 | 7.00 | 3.66 | 32.70 | 24.79 | 99.96 | 83.84 | 0.05 | 0.00 |
| total | 1.00 | 1,432,910.00 | 1,432,910.00 | 15,308.00 | nan | 6,657.00 | 0.00 | 4,370.00 | 3,744.00 | 537.00 | 43.49 | 40,892.14 | 37,633.19 | 7,317.65 | -4,058.71 | 403.00 | 6.05 | 24.12 | 22.64 | 99.96 | 99.90 | 0.169 | 0.0467 |

### table04b_admitted_negative

|  | value |
|---|---|
| arrivals | 15,308.00 |
| R_job<0 | 2,468.00 |
| R_job<0 % | 16.12 |
| R_nocarb<0 % | 6.66 |
| admitted | 6,657.00 |
| admitted with R_job<0 | 403.00 |
| admitted R_job<0 % of admitted | 6.05 |
| admitted R_job<0 % of negative jobs | 16.33 |
| of admitted R_job<0: score>0 % | 100.00 |
| of admitted R_job<0: Pi_job<0 % | 95.78 |
| of admitted R_job<0: phi_mech<0 % | 95.78 |
| of admitted R_job<0: R_nocarb>=0 % (carbon flips sign) | 16.38 |
| of admitted R_job<0: mean r~_type $ | 4.78 |
| of admitted R_job<0: mean R_job $ | -12.76 |
| admitted with payment<0 | 1,091.00 |

### table05_accuracy

|  | n used (arrivals) | built from | TV(p_hat,p_ref) | thin mass (N_j<n_type_min) | d rel err mean | d rel err median | d CI rel half-width | d CI mass covered | KS(F_hat,F_emp) | phi mean abs err $ | phi rel err | r mean abs err $ | r rel err | decision errors (discretization) | ||tau_k - tau*|| | agree hindsight oracle | agree offline x* | eps_D | eta(W_prev) | p_U start | p_U end | CI1(W) | eps_Test2(W) | eps_Test3(W) | max(Test2 lhs - thr) | min Test3 screen margin |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 10,000 | H (Phase 0) | 0.416 | 0.592 | 0.311 | 0.152 | 0.182 | 0.679 | 0.00671 | 0.00 | 0.00 | 16.11 | 1.21 | 0.276 | 0.0221 | 0.849 | 0.822 | 0.502 | 0.032 | 0.187 | 0.642 | 0.347 | 0.413 | 0.619 | -0.415 | 1.11 |
| 2 | 152 | phase 1 | 0.874 | 1.00 | 0.348 | 0.218 | 0.199 | 0.0135 | 0.00671 | 0.00 | 0.00 | 12.80 | 0.961 | 0.303 | 0.0425 | 0.767 | 0.757 | 0.502 | 0.0226 | 0.187 | 0.648 | 0.244 | 0.291 | 0.434 | -0.297 | 0.785 |
| 3 | 400 | phase 2 | 0.755 | 1.00 | 0.361 | 0.187 | 0.163 | 0.00954 | 0.00671 | 0.00 | 0.00 | 13.86 | 1.04 | 0.279 | 0.047 | 0.8 | 0.775 | 0.502 | 0.016 | 0.187 | 0.653 | 0.172 | 0.205 | 0.306 | -0.207 | 0.55 |
| 4 | 684 | phase 3 | 0.669 | 1.00 | 0.457 | 0.274 | 0.12 | 0.055 | 0.00671 | 0.00 | 0.00 | 20.96 | 1.57 | 0.292 | 0.0229 | 0.789 | 0.758 | 0.502 | 0.0113 | 0.187 | 0.657 | 0.121 | 0.145 | 0.215 | -0.136 | 0.385 |
| 5 | 1,463 | phase 4 | 0.517 | 0.934 | 0.35 | 0.211 | 0.146 | 0.148 | 0.00671 | 0.00 | 0.00 | 18.61 | 1.40 | 0.241 | 0.0425 | 0.8 | 0.753 | 0.502 | 0.00799 | 0.187 | 0.659 | 0.0857 | 0.102 | 0.152 | -0.108 | 0.273 |
| 6 | 5,185 | phase 5 | 0.249 | 0.586 | 0.271 | 0.119 | 0.196 | 0.456 | 0.00671 | 0.00 | 0.00 | 10.35 | 0.779 | 0.155 | 0.0425 | 0.813 | 0.764 | 0.502 | 0.00565 | 0.187 | 0.66 | 0.0838 | 0.0954 | 0.149 | -0.0887 | 0.266 |
| 7 | 6,232 | phase 6 | 0.233 | 0.574 | 0.292 | 0.112 | 0.205 | 0.5 | 0.00671 | 0.00 | 0.00 | 14.13 | 1.06 | 0.13 | 0.0361 | 0.775 | 0.688 | 0.502 | 0.00552 | 0.187 | 0.649 | 0.492 | 0.504 | 0.879 | -0.431 | 1.58 |
| 8 | 218 | phase 7 | 0.887 | 1.00 | 0.42 | 0.175 | 0.103 | 0.00418 | 0.00671 | 0.00 | 0.00 | 22.49 | 1.68 | 0.203 | 0.103 | 0.76 | 0.736 | 0.502 | 0.0318 | 0.187 | 0.642 | 0.346 | 0.411 | 0.615 | -0.353 | 1.11 |
| 9 | 470 | phase 8 | 0.776 | 1.00 | 0.393 | 0.178 | 0.269 | 0.0255 | 0.00671 | 0.00 | 0.00 | 11.28 | 0.845 | 0.232 | 0.0656 | 0.764 | 0.704 | 0.502 | 0.0225 | 0.187 | 0.645 | 0.38 | 0.426 | 0.677 | -0.345 | 1.22 |

### table05b_monotonicity

|  | direction checked | phase 1 | last phase | monotone (all phases) | monotone (phases >= 2) | last < first |
|---|---|---|---|---|---|---|
| TV(p_hat,p_ref) | decrease | 0.416 | 0.776 | no | no | no |
| thin mass (N_j<n_type_min) | decrease | 0.592 | 1.00 | no | no | no |
| d rel err mean | decrease | 0.311 | 0.393 | no | no | no |
| d rel err median | decrease | 0.152 | 0.178 | no | no | no |
| KS(F_hat,F_emp) | decrease | 0.00671 | 0.00671 | yes | yes | no |
| phi mean abs err $ | decrease | 0.00 | 0.00 | yes | yes | no |
| r mean abs err $ | decrease | 16.11 | 11.28 | no | no | yes |
| r rel err | decrease | 1.21 | 0.845 | no | no | yes |
| decision errors (discretization) | decrease | 0.276 | 0.232 | no | no | yes |
| ||tau_k - tau*|| | decrease | 0.0221 | 0.0656 | no | no | no |
| agree hindsight oracle | increase | 0.849 | 0.764 | no | no | no |
| agree offline x* | increase | 0.822 | 0.704 | no | no | no |

### table06_baselines

|  | admitted | adm rate % | phys rejects | R $ | adm with R_job<0 % | R / R_OPT fluid | R / R_OPT job | R_ALG / R | regret vs fluid $ | regret vs job OPT $ |
|---|---|---|---|---|---|---|---|---|---|---|
| DLENT-RT (online) | 6,657 | 43.49 | 537 | 40,892.14 | 6.05 | 0.279 | 0.339 | 1.00 | 105,503.05 | 79,849.88 |
| static tau* (full stream, ORACLE) | 10,478 | 68.45 | 1,427 | 105,587.09 | 4.05 | 0.721 | 0.874 | 0.387 | 40,808.10 | 15,154.93 |
| static Phase 0 (H) | 9,237 | 60.34 | 670 | 44,716.94 | 6.11 | 0.305 | 0.37 | 0.914 | 101,678.25 | 76,025.08 |
| static Phase 0 (H) + thinning at cap | 6,253 | 40.85 | 327 | 31,285.95 | 6.22 | 0.214 | 0.259 | 1.31 | 115,109.25 | 89,456.07 |
| G1 (type greedy, cumulative r~) | 10,377 | 67.79 | 1,290 | 64,503.43 | 6.00 | 0.441 | 0.534 | 0.634 | 81,891.76 | 56,238.58 |
| G2 (job greedy, R_job > 0) | 10,616 | 69.35 | 2,224 | 107,551.45 | 0.00 | 0.735 | 0.891 | 0.38 | 38,843.74 | 13,190.56 |
| offline job-level OPT (ORACLE) | 10,775 | 70.39 | 0 | 120,742.02 | 0.00 | 0.825 | 1.00 | 0.339 | 25,653.18 | 0.00 |

### table06b_benchmarks

|  | $ |
|---|---|
| R_OPT fluid (bench_global) | 146,395.19 |
| R_OPT job MILP | 120,742.02 |
| R_OPT job LP relaxation | 126,227.41 |
| sum of R_job>0 (no capacity) | 155,579.35 |
| R_ALG logged (summary R_total) | 40,892.14 |
| fluid regret (summary regret_global) | 105,503.05 |
| per-phase fluid bench sum (summary) | 147,031.94 |

### table07_regret_decomposition

|  | attributed to | $ | % of total |
|---|---|---|---|
| job OPT -> fluid OPT | discretization loss | -25,653.18 | -32.13 |
| fluid OPT -> DLENT-RT online | estimation + thinning + learning (not yet separated; preliminary) | 105,503.05 | 132.13 |
| total regret (job OPT - online) | nan | 79,849.88 | 100.00 |

### table07b_regret_per_phase

|  | arrivals | R_OPT job (phase) $ | DLENT fluid regret (code: bench_hindsight - R) $ | regret DLENT-RT (online) $ | regret static tau* (full stream, ORACLE) $ | regret static Phase 0 (H) $ | regret static Phase 0 (H) + thinning at cap $ | regret G1 (type greedy, cumulative r~) $ | regret G2 (job greedy, R_job > 0) $ | DLENT regret per arrival $ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 152.00 | 1,467.11 | 1,238.29 | 1,212.95 | 52.50 | 1,489.23 | 1,212.33 | 1,488.77 | 0.644 | 7.98 |
| 2 | 400.00 | 3,363.68 | 3,518.17 | 2,554.04 | 188.81 | 2,245.62 | 2,612.51 | 2,226.84 | 150.60 | 6.39 |
| 3 | 684.00 | 5,669.75 | 4,914.75 | 3,738.76 | 640.74 | 2,688.18 | 3,441.44 | 2,458.03 | 583.98 | 5.47 |
| 4 | 1,463.00 | 10,117.22 | 10,191.25 | 7,283.40 | 917.62 | 6,266.04 | 7,421.11 | 5,644.12 | 721.85 | 4.98 |
| 5 | 5,185.00 | 38,641.56 | 36,423.54 | 26,773.11 | 5,169.79 | 23,247.27 | 28,472.90 | 19,572.79 | 4,334.78 | 5.16 |
| 6 | 6,232.00 | 49,685.82 | 39,239.62 | 30,118.25 | 5,932.45 | 30,345.21 | 35,564.24 | 19,647.76 | 4,774.21 | 4.83 |
| 7 | 218.00 | 2,949.43 | 2,733.17 | 2,185.27 | 680.64 | 3,249.74 | 3,514.93 | 2,074.86 | 907.67 | 10.02 |
| 8 | 470.00 | 4,766.45 | 4,955.69 | 4,266.30 | 956.14 | 3,461.71 | 3,815.52 | 1,886.61 | 1,153.81 | 9.08 |
| 9 | 504.00 | 4,080.99 | 2,925.32 | 1,717.79 | 616.23 | 3,032.08 | 3,401.09 | 1,238.82 | 563.02 | 3.41 |
| total | 15,308.00 | 120,742.02 | 106,139.80 | 79,849.88 | 15,154.93 | 76,025.08 | 89,456.07 | 56,238.58 | 13,190.56 | 5.22 |

### table08_capacity_sweep

|  | c_cpu | c_ram | xi_data | phys gate: jobs A>c | p_U min | phases | end reasons | admitted | adm rate % | rej physical | rej thinning | occ cpu mean % | occ ram mean % | occ cpu peak % | occ ram peak % | R_ALG $ | R_OPT job $ | R_ALG/R_OPT job | R_OPT fluid $ | R_ALG/R_OPT fluid | tau* (hindsight) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.201 | 0.131 | 9.83 | 2,650 | 0.108 | 12 | test1:5, timeout:6, horizon:1 | 5,617 | 36.69 | 585 | 3,520 | 21.43 | 17.22 | 99.95 | 99.98 | 33,103.76 | 95,131.86 | 0.348 | 131,430.68 | 0.252 | [0.042665 0.08822 ] |
| 1.0 | 0.401 | 0.262 | 4.91 | 1,680 | 0.187 | 9 | test1:1, test2:1, timeout:6, horizon:1 | 6,657 | 43.49 | 537 | 3,744 | 24.12 | 22.64 | 99.96 | 99.90 | 40,892.14 | 120,742.02 | 0.339 | 146,395.19 | 0.279 | [0.       0.042513] |

### table10_anomalies

|  | value |
|---|---|
| s_C with data.n0=10,000 | 228.184 $/kgCO2 (|H|=10,000, |stream|=15,308, K_q=0.06984, D_q=15.94) |
| admitted jobs with payment < 0 | 1,091 / 6,657 (16.4%), sum = -8,926.09 $; per phase {1: 8, 2: 13, 3: 27, 4: 97, 5: 374, 6: 483, 7: 25, 8: 38, 9: 26}; bins k [6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]... |
| admitted jobs with payment > bid v | 0 |
| phase 1 state: populated types with phi_tilde < 0 | 0 / 2673 (p-mass 0.0%) |
| phase 2 state: populated types with phi_tilde < 0 | 0 / 139 (p-mass 0.0%) |
| phase 3 state: populated types with phi_tilde < 0 | 0 / 350 (p-mass 0.0%) |
| phase 4 state: populated types with phi_tilde < 0 | 0 / 543 (p-mass 0.0%) |
| phase 5 state: populated types with phi_tilde < 0 | 0 / 934 (p-mass 0.0%) |
| phase 6 state: populated types with phi_tilde < 0 | 0 / 1773 (p-mass 0.0%) |
| phase 7 state: populated types with phi_tilde < 0 | 0 / 2194 (p-mass 0.0%) |
| phase 8 state: populated types with phi_tilde < 0 | 0 / 204 (p-mass 0.0%) |
| phase 9 state: populated types with phi_tilde < 0 | 0 / 400 (p-mass 0.0%) |
| xi_data = max a_max/c (scale 1) | 4.9143  (forced xi_used = 0.001) |
| stream jobs with A_i > c_i (never admissible) | 1,680 (10.97%) |
| phase 1 state: tau, fluid load sum_(r~>0) p_j v_j / c | tau=[0.       0.020386], load=[1.2324 1.2892] |
| phase 2 state: tau, fluid load sum_(r~>0) p_j v_j / c | tau=[0. 0.], load=[0.4691 0.4379] |
| phase 3 state: tau, fluid load sum_(r~>0) p_j v_j / c | tau=[0.020123 0.      ], load=[1.2481 1.1175] |
| phase 4 state: tau, fluid load sum_(r~>0) p_j v_j / c | tau=[0.      0.01965], load=[1.089  1.2307] |
| phase 5 state: tau, fluid load sum_(r~>0) p_j v_j / c | tau=[0. 0.], load=[0.6848 0.807 ] |
| phase 6 state: tau, fluid load sum_(r~>0) p_j v_j / c | tau=[0. 0.], load=[0.7868 0.8181] |
| phase 7 state: tau, fluid load sum_(r~>0) p_j v_j / c | tau=[0.004437 0.006662], load=[1.2127 1.2003] |
| phase 8 state: tau, fluid load sum_(r~>0) p_j v_j / c | tau=[0.094067 0.      ], load=[1.8694 1.3244] |
| phase 9 state: tau, fluid load sum_(r~>0) p_j v_j / c | tau=[0.049993 0.      ], load=[1.605  1.0579] |
| reference (full stream): tau, fluid load sum_(r~>0) p_j v_j / c | tau=[0.       0.042513], load=[1.4429 1.4804] |
| reference types: populated / stream jobs | 3,033 / 15,308 (5.05 jobs per populated type) |
| max |r~| over populated types (all phase states) | 24293.219 $ vs gamma_used = 1.0 |
| phase end reasons | {'test1': 1, 'test2': 1, 'test3': 0, 'timeout': 6, 'horizon': 1} |
| max over phases of (Test2 lhs - thr) | -0.08868 |
| min over phases of Test3 screen margin | 0.2662 |
| R_ALG logged (summary R_total) vs re-scored with R_job | 40,892.14 vs 40,892.14 $ (asserted equal) |
| R_OPT fluid vs offline job-level MILP | 146,395.19 vs 120,742.02 $ |

---

# Appendix B — notebook 04 tables (generated)

## nb04 tables — regime configs/regime_small_cluster.yaml (generated by the notebook; do not edit by hand)

Results cached at commit `3549fc97caa8d93b2d86262bc166391a07aad322`, configs/default.yaml blob `68e7bd76b57c376cae6d74139f11cc456559520b`, regime blob `a0d29b5313ea41184961aa085366bc988c5dd7d0`; s_C = 228.1843.

### job-level MILP

| lambda | sum_CV | sum_PR | sum_CO | scalarised | n_admitted | nondominated |
|---|---|---|---|---|---|---|
| (0, 0, 3) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | True |
| (0, 0.75, 2.25) | 67,975.97 | 29,207.74 | 3,051.30 | 15,040.38 | 7,970.00 | True |
| (0, 1.5, 1.5) | 84,617.02 | 34,465.80 | 5,873.76 | 42,888.05 | 9,329.00 | True |
| (0, 2.25, 0.75) | 86,445.11 | 35,101.44 | 6,852.02 | 73,839.22 | 9,626.00 | True |
| (0, 3, 0) | 87,068.76 | 35,183.73 | 7,283.28 | 105,551.19 | 9,702.00 | True |
| (0.75, 0, 2.25) | 94,299.66 | 20,445.02 | 5,814.82 | 57,641.41 | 10,797.00 | True |
| (0.75, 0.75, 1.5) | 93,947.71 | 33,168.26 | 6,659.13 | 85,348.29 | 10,612.00 | True |
| (0.75, 1.5, 0.75) | 93,437.33 | 34,157.96 | 7,172.14 | 115,935.83 | 10,464.00 | True |
| (0.75, 2.25, 0) | 92,833.59 | 34,487.61 | 7,442.89 | 147,222.32 | 10,348.00 | True |
| (1.5, 0, 1.5) | 96,785.48 | 20,062.66 | 7,111.27 | 134,511.32 | 11,471.00 | True |
| (1.5, 0.75, 0.75) | 95,836.82 | 31,837.08 | 7,392.91 | 162,088.36 | 11,052.00 | True |
| (1.5, 1.5, 0) | 94,919.17 | 33,200.93 | 7,553.03 | 192,180.15 | 10,777.00 | True |
| (2.25, 0, 0.75) | 97,137.67 | 18,458.03 | 7,615.89 | 212,847.83 | 11,498.00 | True |
| (2.25, 0.75, 0) | 96,366.33 | 30,738.48 | 7,661.86 | 239,878.10 | 11,171.00 | True |
| (3, 0, 0) | 97,156.75 | 18,515.45 | 7,721.22 | 291,470.24 | 11,477.00 | True |
| (1, 1, 1) | 94,609.09 | 33,324.04 | 7,191.12 | 120,742.02 | 10,775.00 | True |

### fluid LP OPT

| lambda | sum_CV | sum_PR | sum_CO | scalarised | n_admitted | nondominated |
|---|---|---|---|---|---|---|
| (0, 0, 3) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | True |
| (0, 0.75, 2.25) | 62,204.45 | 24,855.31 | 2,795.64 | 12,351.30 | 6,588.00 | True |
| (0, 1.5, 1.5) | 95,783.48 | 35,227.67 | 8,933.26 | 39,441.61 | 8,739.00 | True |
| (0, 2.25, 0.75) | 109,761.76 | 39,876.43 | 16,318.22 | 77,483.30 | 9,830.49 | True |
| (0, 3, 0) | 110,476.86 | 40,079.63 | 17,359.18 | 120,238.90 | 9,913.83 | True |
| (0.75, 0, 2.25) | 114,679.99 | 22,993.37 | 9,693.23 | 64,200.21 | 11,572.00 | True |
| (0.75, 0.75, 1.5) | 124,918.97 | 35,462.26 | 15,425.61 | 97,147.51 | 11,765.00 | True |
| (0.75, 1.5, 0.75) | 125,280.75 | 37,909.16 | 17,416.03 | 137,762.28 | 11,451.22 | True |
| (0.75, 2.25, 0) | 123,962.68 | 38,527.86 | 17,633.40 | 179,659.69 | 11,158.26 | True |
| (1.5, 0, 1.5) | 130,223.53 | 25,414.20 | 17,432.16 | 169,187.05 | 12,834.06 | True |
| (1.5, 0.75, 0.75) | 129,339.62 | 34,220.23 | 17,683.65 | 206,411.86 | 12,403.06 | True |
| (1.5, 1.5, 0) | 127,976.02 | 36,038.57 | 17,668.96 | 246,021.89 | 11,951.47 | True |
| (2.25, 0, 0.75) | 130,449.74 | 25,395.19 | 17,731.83 | 280,213.03 | 12,847.91 | True |
| (2.25, 0.75, 0) | 129,682.37 | 33,364.82 | 17,711.16 | 316,808.95 | 12,569.42 | True |
| (3, 0, 0) | 130,453.80 | 25,400.20 | 17,745.31 | 391,361.40 | 12,847.17 | True |
| (1, 1, 1) | 127,742.61 | 36,076.51 | 17,423.93 | 146,395.19 | 11,905.31 | True |

### Chebyshev fluid LP

| lambda | sum_CV | sum_PR | sum_CO | scalarised | n_admitted | nondominated |
|---|---|---|---|---|---|---|
| (0, 0, 3) | 5.88 | 1.41 | 0.000464 | -0.00139 | 5.00 | True |
| (0, 0.75, 2.25) | 24,080.72 | 12,985.05 | 497.86 | 8,618.60 | 3,616.47 | True |
| (0, 1.5, 1.5) | 76,877.00 | 29,467.93 | 4,698.35 | 37,154.37 | 7,505.64 | True |
| (0, 2.25, 0.75) | 103,192.03 | 37,733.51 | 12,176.46 | 75,768.06 | 9,332.65 | True |
| (0, 3, 0) | 110,482.07 | 40,079.63 | 17,359.34 | 120,238.89 | 9,914.26 | True |
| (0.75, 0, 2.25) | 41,351.01 | 10,957.73 | 870.68 | 29,054.23 | 5,184.07 | True |
| (0.75, 0.75, 1.5) | 58,954.60 | 18,112.76 | 1,706.41 | 55,240.90 | 6,656.09 | True |
| (0.75, 1.5, 0.75) | 98,114.47 | 35,986.23 | 9,778.83 | 120,231.07 | 8,942.08 | True |
| (0.75, 2.25, 0) | 110,482.07 | 40,079.63 | 17,359.34 | 173,040.72 | 9,914.26 | True |
| (1.5, 0, 1.5) | 93,914.80 | 20,449.39 | 4,970.31 | 133,416.73 | 9,511.02 | True |
| (1.5, 0.75, 0.75) | 115,294.70 | 23,097.02 | 9,903.68 | 182,837.05 | 11,647.93 | True |
| (1.5, 1.5, 0) | 124,614.18 | 38,285.51 | 17,623.80 | 244,349.53 | 11,248.59 | True |
| (2.25, 0, 0.75) | 121,083.81 | 24,041.38 | 12,255.06 | 263,247.28 | 12,408.63 | True |
| (2.25, 0.75, 0) | 130,453.80 | 25,402.76 | 17,745.34 | 312,573.12 | 12,845.46 | True |
| (3, 0, 0) | 130,453.75 | 25,419.45 | 17,745.36 | 391,361.25 | 12,844.64 | True |
| (1, 1, 1) | 91,894.77 | 28,233.05 | 5,245.09 | 114,882.73 | 8,876.81 | True |

### G2 job greedy

| lambda | sum_CV | sum_PR | sum_CO | scalarised | n_admitted | nondominated |
|---|---|---|---|---|---|---|
| (0, 0, 3) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | True |
| (0, 0.75, 2.25) | 67,723.56 | 29,077.37 | 3,047.29 | 14,951.63 | 7,959.00 | True |
| (0, 1.5, 1.5) | 80,534.35 | 33,050.14 | 5,766.53 | 40,925.40 | 9,317.00 | True |
| (0, 2.25, 0.75) | 80,334.32 | 33,056.36 | 7,060.04 | 69,081.78 | 9,529.00 | True |
| (0, 3, 0) | 79,713.60 | 32,665.47 | 7,320.29 | 97,996.42 | 9,600.00 | False |
| (0.75, 0, 2.25) | 89,009.11 | 18,733.87 | 5,646.66 | 54,051.85 | 10,847.00 | True |
| (0.75, 0.75, 1.5) | 85,394.69 | 31,213.00 | 6,790.67 | 77,269.77 | 10,497.00 | True |
| (0.75, 1.5, 0.75) | 83,017.46 | 31,752.80 | 7,469.33 | 104,290.29 | 10,293.00 | True |
| (0.75, 2.25, 0) | 82,597.80 | 31,799.41 | 7,564.90 | 133,497.02 | 10,187.00 | True |
| (1.5, 0, 1.5) | 86,105.45 | 17,888.65 | 7,331.35 | 118,161.15 | 11,348.00 | False |
| (1.5, 0.75, 0.75) | 84,644.48 | 29,221.83 | 7,654.80 | 143,141.99 | 10,901.00 | False |
| (1.5, 1.5, 0) | 83,539.49 | 30,260.86 | 7,648.65 | 170,700.51 | 10,650.00 | False |
| (2.25, 0, 0.75) | 84,653.17 | 15,289.89 | 7,743.64 | 184,661.91 | 11,327.00 | False |
| (2.25, 0.75, 0) | 84,059.30 | 27,752.58 | 7,708.36 | 209,947.87 | 11,046.00 | False |
| (3, 0, 0) | 84,653.17 | 15,289.89 | 7,743.64 | 253,959.52 | 11,327.00 | False |
| (1, 1, 1) | 84,155.28 | 30,884.85 | 7,488.67 | 107,551.45 | 10,616.00 | False |

### StaticPolicy (Phase 0)

| lambda | sum_CV | sum_PR | sum_CO | scalarised | n_admitted | nondominated |
|---|---|---|---|---|---|---|
| (3, 0, 0) | 42,331.40 | 8,113.51 | 4,594.07 | 126,994.19 | 9,887.00 | False |
| (0, 3, 0) | 38,090.47 | 8,951.00 | 4,700.36 | 26,853.01 | 8,419.00 | True |
| (0, 0, 3) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | True |
| (1, 1, 1) | 40,572.86 | 8,814.15 | 4,670.07 | 44,716.94 | 9,237.00 | True |
| (1.5, 1.5, 0) | 40,574.29 | 8,820.86 | 4,676.22 | 74,092.73 | 9,231.00 | True |
| (1.5, 0, 1.5) | 42,382.34 | 8,192.68 | 4,578.23 | 56,706.16 | 9,847.00 | True |
| (0, 1.5, 1.5) | 37,207.47 | 8,276.12 | 3,516.14 | 7,139.97 | 8,026.00 | True |

### DLENT-RT theory

| lambda | sum_CV | sum_PR | sum_CO | scalarised | n_admitted | nondominated |
|---|---|---|---|---|---|---|
| (3, 0, 0) | 38,903.12 | 7,116.97 | 4,407.33 | 116,709.36 | 7,423.00 | True |
| (0, 3, 0) | 32,541.34 | 5,790.92 | 3,911.79 | 17,372.76 | 5,944.00 | True |
| (0, 0, 3) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | True |
| (1, 1, 1) | 37,633.19 | 7,317.65 | 4,058.71 | 40,892.14 | 6,657.00 | True |
| (1.5, 1.5, 0) | 37,784.23 | 7,291.63 | 4,235.09 | 67,613.79 | 6,738.00 | True |
| (1.5, 0, 1.5) | 40,327.00 | 5,938.84 | 3,977.71 | 54,523.94 | 7,265.00 | True |
| (0, 1.5, 1.5) | 34,337.94 | 5,734.72 | 3,032.21 | 4,053.76 | 5,576.00 | True |

### Corners

| lambda / model | n_admitted | sum_CV | sum_PR | sum_CO | corner_solution |
|---|---|---|---|---|---|
| (0, 0, 3) / Chebyshev fluid LP | 5.00 | 5.88 | 1.41 | 0.000464 | admit only (near-)zero-carbon types: the augmentation term breaks the tie at CO = 0 |
| (0, 0, 3) / DLENT-RT theory | 0.00 | 0.00 | 0.00 | 0.00 | admit nothing (degenerate: R <= 0 for every job) |
| (0, 0, 3) / G2 job greedy | 0.00 | 0.00 | 0.00 | 0.00 | admit nothing (degenerate: R <= 0 for every job) |
| (0, 0, 3) / StaticPolicy (Phase 0) | 0.00 | 0.00 | 0.00 | 0.00 | admit nothing (degenerate: R <= 0 for every job) |
| (0, 0, 3) / fluid LP OPT | 0.00 | 0.00 | 0.00 | 0.00 | admit nothing (degenerate: R <= 0 for every job) |
| (0, 0, 3) / job-level MILP | 0.00 | 0.00 | 0.00 | 0.00 | admit nothing (degenerate: R <= 0 for every job) |
| (0, 3, 0) / Chebyshev fluid LP | 9,914.26 | 110,482.07 | 40,079.63 | 17,359.34 | admit the profitable jobs / types (PR > 0), carbon ignored |
| (0, 3, 0) / DLENT-RT theory | 5,944.00 | 32,541.34 | 5,790.92 | 3,911.79 | admit the profitable jobs / types (PR > 0), carbon ignored |
| (0, 3, 0) / G2 job greedy | 9,600.00 | 79,713.60 | 32,665.47 | 7,320.29 | admit the profitable jobs / types (PR > 0), carbon ignored |
| (0, 3, 0) / StaticPolicy (Phase 0) | 8,419.00 | 38,090.47 | 8,951.00 | 4,700.36 | admit the profitable jobs / types (PR > 0), carbon ignored |
| (0, 3, 0) / fluid LP OPT | 9,913.83 | 110,476.86 | 40,079.63 | 17,359.18 | admit the profitable jobs / types (PR > 0), carbon ignored |
| (0, 3, 0) / job-level MILP | 9,702.00 | 87,068.76 | 35,183.73 | 7,283.28 | admit the profitable jobs / types (PR > 0), carbon ignored |
| (3, 0, 0) / Chebyshev fluid LP | 12,844.64 | 130,453.75 | 25,419.45 | 17,745.36 | capacity binds on average: the fluid LP admits a fraction of the types |
| (3, 0, 0) / DLENT-RT theory | 7,423.00 | 38,903.12 | 7,116.97 | 4,407.33 | every type SEEN in H has r~ > 0 (unseen types have r~ = 0 and stay rejected); then the physical gate and thinning |
| (3, 0, 0) / G2 job greedy | 11,327.00 | 84,653.17 | 15,289.89 | 7,743.64 | every job has R > 0: admit-all up to the per-tick capacity |
| (3, 0, 0) / StaticPolicy (Phase 0) | 9,887.00 | 42,331.40 | 8,113.51 | 4,594.07 | every type SEEN in H has r~ > 0 (unseen types have r~ = 0 and stay rejected); then the physical gate |
| (3, 0, 0) / fluid LP OPT | 12,847.17 | 130,453.80 | 25,400.20 | 17,745.31 | capacity binds on average: the fluid LP admits a fraction of the types |
| (3, 0, 0) / job-level MILP | 11,477.00 | 97,156.75 | 18,515.45 | 7,721.22 | every job has R > 0: admit-all up to the per-tick capacity |

### DLENT-RT and StaticPolicy vs the offline frontier

| model / lambda | scalarised $ | % of fluid OPT | % of job MILP | d2 to offline Pareto set | eps to offline Pareto set |
|---|---|---|---|---|---|
| StaticPolicy (Phase 0) / (3, 0, 0) | 126,994.19 | 32.45 | 43.57 | 0.222 | 0.21 |
| StaticPolicy (Phase 0) / (0, 3, 0) | 26,853.01 | 22.33 | 25.44 | 0.223 | 0.216 |
| StaticPolicy (Phase 0) / (0, 0, 3) | 0.00 | nan | nan | 0.00 | 0.00 |
| StaticPolicy (Phase 0) / (1, 1, 1) | 44,716.94 | 30.55 | 37.04 | 0.221 | 0.214 |
| StaticPolicy (Phase 0) / (1.5, 1.5, 0) | 74,092.73 | 30.12 | 38.55 | 0.221 | 0.214 |
| StaticPolicy (Phase 0) / (1.5, 0, 1.5) | 56,706.16 | 33.52 | 42.16 | 0.22 | 0.209 |
| StaticPolicy (Phase 0) / (0, 1.5, 1.5) | 7,139.97 | 18.10 | 16.65 | 0.166 | 0.149 |
| DLENT-RT theory / (3, 0, 0) | 116,709.36 | 29.82 | 40.04 | 0.222 | 0.199 |
| DLENT-RT theory / (0, 3, 0) | 17,372.76 | 14.45 | 16.46 | 0.225 | 0.171 |
| DLENT-RT theory / (0, 0, 3) | 0.00 | nan | nan | 0.00 | 0.00 |
| DLENT-RT theory / (1, 1, 1) | 40,892.14 | 27.93 | 33.87 | 0.203 | 0.18 |
| DLENT-RT theory / (1.5, 1.5, 0) | 67,613.79 | 27.48 | 35.18 | 0.212 | 0.19 |
| DLENT-RT theory / (1.5, 0, 1.5) | 54,523.94 | 32.23 | 40.53 | 0.215 | 0.175 |
| DLENT-RT theory / (0, 1.5, 1.5) | 4,053.76 | 10.28 | 9.45 | 0.186 | 0.13 |

### Does every lambda matter? (jobs changed vs the centre, x2 + x0.5)

| model | lambda1 | lambda2 | lambda3 | least-changing lambda |
|---|---|---|---|---|
| fluid LP OPT | 1,270.40 | 1,167.50 | 444.80 | lambda3 |
| G2 job greedy | 1,349.00 | 946.00 | 837.00 | lambda3 |
| job-level MILP | 1,179.00 | 998.00 | 563.00 | lambda3 |
| StaticPolicy (Phase 0) | 712.00 | 671.00 | 342.00 | lambda3 |
| DLENT-RT theory | 1,869.00 | 1,449.00 | 1,568.00 | lambda2 |

### Weighted sum vs Chebyshev (fluid LP)

| lambda | Chebyshev CV | Chebyshev PR | Chebyshev CO | dominance slack | distance to nearest weighted-sum point | same as a weighted-sum point (< 1e-6) |
|---|---|---|---|---|---|---|
| (0, 0, 3) | 5.88 | 1.41 | 0.000464 | 0.000194 | 5.71e-05 | False |
| (0, 0.75, 2.25) | 24,080.72 | 12,985.05 | 497.86 | 0.000449 | 0.374 | False |
| (0, 1.5, 1.5) | 76,877.00 | 29,467.93 | 4,698.35 | 9.41e-05 | 0.193 | False |
| (0, 2.25, 0.75) | 103,192.03 | 37,733.51 | 12,176.46 | 5.01e-05 | 0.201 | False |
| (0, 3, 0) | 110,482.07 | 40,079.63 | 17,359.34 | -5.12e-13 | 4.1e-05 | False |
| (0.75, 0, 2.25) | 41,351.01 | 10,957.73 | 870.68 | 0.000454 | 0.397 | False |
| (0.75, 0.75, 1.5) | 58,954.60 | 18,112.76 | 1,706.41 | 5.76e-06 | 0.181 | False |
| (0.75, 1.5, 0.75) | 98,114.47 | 35,986.23 | 9,778.83 | 4.39e-05 | 0.0543 | False |
| (0.75, 2.25, 0) | 110,482.07 | 40,079.63 | 17,359.34 | -5.12e-13 | 4.1e-05 | False |
| (1.5, 0, 1.5) | 93,914.80 | 20,449.39 | 4,970.31 | 0.000499 | 0.294 | False |
| (1.5, 0.75, 0.75) | 115,294.70 | 23,097.02 | 9,903.68 | 0.00022 | 0.013 | False |
| (1.5, 1.5, 0) | 124,614.18 | 38,285.51 | 17,623.80 | -1.43e-14 | 0.00786 | False |
| (2.25, 0, 0.75) | 121,083.81 | 24,041.38 | 12,255.06 | 0.000245 | 0.155 | False |
| (2.25, 0.75, 0) | 130,453.80 | 25,402.76 | 17,745.34 | -6.3e-13 | 6.38e-05 | False |
| (3, 0, 0) | 130,453.75 | 25,419.45 | 17,745.36 | -3.64e-13 | 0.00048 | False |
| (1, 1, 1) | 91,894.77 | 28,233.05 | 5,245.09 | 7.35e-07 | 0.273 | False |

### Hypervolume

| model | 7-point subset | all its grid points |
|---|---|---|
| Chebyshev fluid LP | 38,125,427,348,724.03 | 53,322,189,447,721.11 |
| Chebyshev fluid LP (legacy, inverted) | 38,092,686,033,959.02 | 51,396,243,034,506.61 |
| DLENT-RT theory | 4,653,936,190,420.15 | 4,653,936,190,420.15 |
| G2 job greedy | 34,852,093,978,833.86 | 41,529,484,675,769.93 |
| StaticPolicy (Phase 0) | 5,783,337,219,158.75 | 5,783,337,219,158.75 |
| fluid LP OPT | 32,048,468,834,005.96 | 47,052,200,072,704.34 |
| job-level MILP | 41,032,450,224,275.23 | 47,441,454,374,112.63 |

# Week 30: notebook 03 baselines before and after the reward fix

**Setup:** real trace, `configs/default.yaml` (from_amax capacity, xi_target 0.5, n0 = 10,000), lognormal σ = 0.3,
theory model, seed 42, capacity scale 1.0.

**Populations:** |H| = 10,000, |stream| = 15,308, T = 1,432,910.

**Shadow prices:** τ = 0 in all 11 phases, and τ* = 0 in the full-stream hindsight LP.

**Notebooks:**
* "Before" is the executed notebook as of week 29 (copy kept locally at `reports/week30/old_03_main_algorithm_analysis.ipynb`).
* "After" is `notebooks/03_main_algorithm_analysis.ipynb`, regenerated from `notebooks/src/03_main_algorithm_analysis.py`
  at commit `9700a6d`.

**What changed between the two:** only the notebook's scoring.
* The online run, the admitted sets and the fluid OPT are identical (the regression snapshot is bit-identical).
* Every row is now scored with one job reward, R_job = λ1·q·v + λ2·(phi_job(v, c_j) − C_elec) − λ3·s_C·K. That is
  the online loop's own formula.
* Before, rows were scored with the "yardstick" φ_ref = `myerson.phi(v)`: the markup φ applied to dollar bids.

| benchmark | admitted | R before ($) | R after ($) | R / fluid OPT after | reason for the change |
|---|---|---|---|---|---|
| DLENT-RT (online) | 7,405 | logged 48,391.61; yardstick −157,117.07 | **48,391.61** | 0.324 | Yardstick removed. The logged R was already correct and equals R_job (asserted). |
| static τ* (full stream, ORACLE) | 12,612 | −227,136.52 | **148,916.82** | 0.996 | Scored with R_job instead of the markup φ on $ bids. Equal to the fluid OPT minus 561.60 $, which is the 9 jobs blocked by the physical gate. |
| static Phase 0 (H) | 10,226 | −610,450.66 | **54,216.40** | 0.363 | Same scoring fix |
| static Phase 0 (H) + thinning at cap | 6,793 | −398,384.97 | **35,473.28** | 0.237 | Same scoring fix |
| R_OPT fluid (λ̂_ref·T, `bench_global`) | – | 149,478.42 | **149,478.42** | 1 | Unchanged: built in `dlent_rt/` with `phi_job` |
| job-level OPT without capacity (Σ R_job⁺) | 12,840 | 183,677.08 | **155,579.35** | 1.041 | Σ of positive R_job instead of positive R_ref; the wrong φ inflated it |
| job-level MILP (ORACLE) | 10,033 before / 12,829 after | 183,390.81 | **155,397.61** (LP relaxation 155,433.19; optimal, gap 0) | 1.040 | Objective is now R_job, so it solves a different set. Capacity costs 181.74 $ against the no-capacity sum. |

**Regret of the online run (after):**
* 101,086.81 $ against the fluid OPT (= `summary["regret_global"]`, unchanged).
* 107,006.00 $ against the job-level MILP.

Split (`table07_regret_decomposition`): the **discretization loss** is job OPT − fluid OPT = 155,397.61 − 149,478.42 =
**5,919.19 $** (6,100.93 $ with the no-capacity bound Σ R_job⁺ instead of the MILP). The remaining 101,086.81 $
(fluid OPT − online) mixes estimation, thinning and learning. It is not separated yet; this attribution is preliminary.

The online run admits 7,405 jobs, while static τ* admits 12,612 with the same physical gate. The gap is real and is
not explained by the accounting. Learning, thinning (p_U ≈ 0.53 at a phase start) and the 9 timeout restarts are
candidates; they are not diagnosed here.

## Accuracy metrics (phase 1 to 11)

| metric | before | after |
|---|---|---|
| KS(F̂, F_emp) | 0.4609 in every phase (markup CDF vs dollar bids) | **0.0067** in every phase (vs markups v/c_j; F̂ is the same known distribution in every phase) |
| r mean abs err $ | 9,720–9,731 | **10.2–21.6** |
| r rel err | ≈ 1.000 | **0.77–1.62** |
| r sign mismatch | 0.32–0.47 | **0.15–0.30** |
| φ mean abs err | 0 | 0 (one known φ, 0 by construction) |

The remaining r error (rel ≈ 1, sign mismatch 15–30%) is genuine. It is the difference between the type means of the
data that built each phase's state (H, or the previous phase's arrivals) and those of the full stream on the same
types, i.e. thin types and nonstationarity. It is not an accounting bug.

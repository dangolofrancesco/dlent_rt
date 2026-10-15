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

## Greedy policies G1 and G2

Both run through `dlent_rt.models.policy.simulate` with the same physical gate, stream, seed (42) and R_job scoring
as DLENT-RT and StaticPolicy. Neither uses τ, an LP, thinning or phases. Source: `notebooks/src/greedy_check.py`
(both configs) and nb03 `table06_baselines` (default config).

* **G1** (`TypeGreedyPolicy`) admits iff r̃_j > 0 for the arrival's Phase-0 type (grid never refit). r̃_j starts at
  Phase 0's value (H) and is the cumulative running mean of R_job over H and every stream arrival seen so far, admitted
  or not, updated after each decision. It therefore uses the realised R_job, and so the realised energy, of rejected
  jobs. A type with no observation is rejected ("unseen type").
* **G2** (`JobGreedyPolicy`) admits iff the job's own R_job > 0 (phi_job under the known F, frozen s_C, realised energy).
* Decision errors (G1 only): arrivals with an observed type where sign(R_job) ≠ sign(r̃_j used in the decision).

**configs/default.yaml** (from_amax). Fluid OPT 149,478.42 $, job OPT (MILP) 155,397.61 $.

| policy | R ($) | % of fluid OPT | admitted | rej r̃ ≤ 0 or R ≤ 0 | rej unseen type | rej physical | decision errors | admitted with R_job < 0 |
|---|---|---|---|---|---|---|---|---|
| DLENT-RT (online) | 48,391.61 | 32.4% | 7,405 | – | – | 2 | – | 6.9% |
| static Phase 0 (H) | 54,216.40 | 36.3% | 10,226 | – | – | 1 | – | 6.9% |
| **G1** | **82,626.18** | **55.3%** | 11,662 | 2,486 | 1,155 | 5 | 12.1% | 6.5% |
| **G2** | **155,011.00** | **103.7%** | 12,826 | 2,468 | 0 | 14 | – | 0.0% |

**configs/binding_capacity.yaml** (fraction_of_concurrency_quantile 0.30 / 0.60). Fluid OPT 149,478.42 $ (capacity
slack in the fluid LP, τ* = 0). Job OPT not computed.

| policy | R ($) | % of fluid OPT | admitted | rej r̃ ≤ 0 or R ≤ 0 | rej unseen type | rej physical | decision errors | admitted with R_job < 0 |
|---|---|---|---|---|---|---|---|---|
| DLENT-RT (online) | 43,692.76 | 29.2% | 6,965 | – | – | 338 | – | – |
| static Phase 0 (H) | 47,682.20 | 31.9% | 9,691 | – | – | 536 | – | 6.6% |
| **G1** | **71,071.96** | **47.5%** | 10,871 | 2,486 | 1,155 | 796 | 12.1% | 6.2% |
| **G2** | **123,081.60** | **82.3%** | 11,546 | 2,468 | 0 | 1,294 | – | 0.0% |

**Check, G1 on frozen H types vs StaticPolicy (Phase 0):** the admit masks are identical on every arrival in both configs
(τ in Phase 0 = 0 in both, so StaticPolicy's rule 1[r̃_j − τᵀv_j > 0] reduces to G1's). With the cumulative update G1
differs from StaticPolicy on 1,980 (default) and 2,098 (binding) arrivals. Frozen G1 rejects 2,762 arrivals of types
unseen in H; the cumulative update cuts that to 1,155, one per type absent from H (only its first arrival, since
the update after the decision makes the type observed).

**Reading (preliminary):**
* G2 exceeds the fluid OPT because the fluid OPT decides on type means, while G2 sees each job's R_job. On the default
  config it reaches 99.75% of the job OPT: with capacity slack, R_job > 0 is close to optimal.
* G1 beats DLENT-RT by 34k $ (default) with the same type information and no τ, no thinning and no restarts.

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

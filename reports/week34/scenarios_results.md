# Non-stationary bids: results (week 34, Parts C and D)

**Setting.** f = 0.5, seed 42, one run per model and environment. The stream's markups change once, at t* = 240 h
(of 576 h). Nothing else changes, including H and Phase 0. The expectations are in `scenarios_expectations.md`.

| Scenario | Change at t* | True markup law after t* |
|---|---|---|
| none | none | ln m ~ N(−0.045, 0.30²) |
| S1 bids fall | m → 0.5 m | μ = −0.738, σ = 0.30 |
| S6 bids spread | σ → 0.5, E[m] = 1 | μ = −0.125, σ = 0.50 |
| S2 bids rise | m → 1.75 m | μ = +0.515, σ = 0.30 |

**Sources.**
* Caches: `reports/week34/cache/scenarios/<env>/<model>.pkl`.
* Tables: `scripts/scenarios_analysis.py`.
* Figures: `notebooks/02_nonstationary.ipynb`.

**Scoring.** A post-change job's realised reward uses the true post-change markup law. Every model is scored on the
same scale.

## 1. Main result: reward after t*

Reward after t*, and its share of fluid OPT's reward after t*. Fluid OPT is the sum of two hindsight LPs, one before
and one after t*. It is an upper bound, not a policy.

| | none | S1 bids fall | S6 bids spread | S2 bids rise |
|---|---|---|---|---|
| fluid OPT | 2,091.6 (100%) | 514.5 (100%) | 2,068.8 (100%) | 4,548.4 (100%) |
| **Practical (C3)** | **1,833.0 (87.6%)** | **397.2 (77.2%)** | 1,825.5 (88.2%) | **3,457.5 (76.0%)** |
| No learning | 1,823.2 (87.2%) | 147.5 (28.7%) | **1,854.4 (89.6%)** | 2,915.8 (64.1%) |
| Greedy-Pos | 1,688.4 (80.7%) | 190.9 (37.1%) | 1,694.6 (81.9%) | 2,824.9 (62.1%) |
| Theoretical | 1,352.7 (64.7%) | 309.4 (60.1%) | 1,375.0 (66.5%) | 2,915.0 (64.1%) |

| After t* | none | S1 | S6 | S2 |
|---|---|---|---|---|
| **Value of learning:** Practical / No learning | 1.005 | **2.69** | 0.98 | **1.19** |
| **Value of the practical changes:** Practical / Theoretical | 1.36 | 1.28 | 1.33 | 1.19 |

Before t* every environment is the same stream. There, Practical earns 1,271.1 (87.4% of OPT), No learning 1,255.7,
Greedy-Pos 1,146.4 and Theoretical 895.3.

Total rewards over 576 h:

| | none | S1 | S6 | S2 |
|---|---|---|---|---|
| fluid OPT | 3,546.1 | 1,969.0 | 3,523.3 | 6,003.0 |
| Practical (C3) | 3,104.0 | 1,668.2 | 3,096.6 | 4,728.6 |
| No learning | 3,078.9 | 1,403.3 | 3,110.1 | 4,171.5 |
| Greedy-Pos | 2,834.8 | 1,337.3 | 2,841.0 | 3,971.3 |
| Theoretical | 2,248.0 | 1,204.7 | 2,270.3 | 3,810.3 |

## 2. Expected against observed

| Scenario | Expected | Observed |
|---|---|---|
| S1 | Test 2 fires. After ReInit, μ̂ moves to the new level and admissions recover; no learning never recovers. Learning wins by far. | **No Test 2.** Learning still wins by far: × 2.69 after t*, 77% of OPT against 29%. Adaptation is slow: the first phase at the new μ̂ starts 259 h after t* (section 3). |
| S2 | Test 3, or Test 1 first. Merging after Test 1 slows adaptation. Learning wins slightly. | **No Test 3.** Test 1 ends the phase 3 h after t*. The adapted phase starts 48 h after t*. Learning wins by × 1.19: less than in S1, but more than "slightly". |
| S6 | Maybe no test. The stale model overestimates φ, which shows in the Myerson check. Moderate learning gain. | No test, as expected. The Myerson check shows the stale model's error (belief / realised = 1.25 against 1.04). **But there is no learning gain:** No learning earns 1.6% more after t*. |

**Finding 1: the theoretical thresholds are out of reach by 2–3 orders of magnitude** (`test_margins.md`).
* With the paper's thresholds, Test 2 reaches at most 0.5% of its threshold, and Test 3 at most 0.4%.
* The thresholds are confidence terms of order √(log(1/δ)/n), with n counted in seconds and γ = 1. The signal is the
  LP value per tick, about 0.0015 r_max, so it is two to three orders of magnitude smaller.
* So Tests 2 and 3 fired **in none of the 8 learning runs** (Practical and Theoretical, four environments), and the "detection delay by Test 2 / 3" column is
  empty everywhere: **none fired**.
* Phases end only by Test 1 (capacity) or by timeout, and learning adapts through that cadence.
* Like the thinning finding of week 33, this is a property of the theoretical constants at our scale, not of the
  data. The calibrated thresholds (section 6) are the response.

**Finding 2: learning matters exactly where the market moves the level of the bids** (S1, S2). In S1 the stale model
almost stops admitting: 10,042 admissions after t* against 94,527 in the stationary run. It reaches only 29% of OPT.
In S2 it keeps prices tuned to the old level and reaches 64% of OPT, against 76% for Practical.

**Finding 3: a change in spread alone (S6) costs no reward here.** σ̂ adapts within 56 h, and the Myerson check sees
the stale model's error. Yet No learning earns slightly more. A possible reading, not tested: S6 keeps E[m] = 1, and
the cluster stays capacity-bound (Test 1 every ~48 h). The stale φ shifts the admitted bins only a little, so the
ranking of jobs barely changes. The difference, 29 reward units (1.6%), is of the order of the gap between
Practical and No learning in the stationary run.

**Finding 4: the practical changes help in every environment**: × 1.19 to 1.36 after t*. In S2 the two learning
models adapt equally fast, and the gap is the smallest. There Theoretical (2,915.0) equals No learning (2,915.8)
after t*: its learning offsets its thinning.

## 3. How the learning models adapted

| | Phases | Phase ends | First phase end after t* | Test 2/3 delay | Adapted phase starts (h after t*) |
|---|---|---|---|---|---|
| none, Practical | 13 | Test 1 ×12, horizon | 3.4 h, Test 1 | none fired | (no change) |
| none, Theoretical | 11 | Test 1 ×8, timeout ×2, horizon | 45.2 h, timeout | none fired | (no change) |
| S1, Practical | 8 | Test 1 ×5, timeout ×2, horizon | 55.6 h, timeout | none fired | 258.9 |
| S1, Theoretical | 7 | Test 1 ×3, timeout ×3, horizon | 45.2 h, timeout | none fired | 234.5 |
| S6, Practical | 13 | Test 1 ×12, horizon | 3.9 h, Test 1 | none fired | 56.1 |
| S6, Theoretical | 10 | Test 1 ×7, timeout ×2, horizon | 45.2 h, timeout | none fired | 94.1 |
| S2, Practical | 13 | Test 1 ×12, horizon | 3.0 h, Test 1 | none fired | 48.4 |
| S2, Theoretical | 11 | Test 1 ×9, timeout ×1, horizon | 9.5 h, Test 1 | none fired | 54.3 |

"Adapted" means the first phase whose μ̂ is within 0.1 of the true μ (S1, S2), or whose σ̂ is within 0.05 of the
true σ (S6).

**Two phases to adapt, as expected.** The phase that contains t* ends, and ReInit refits on data that mix the two
regimes. The next phase uses the mixed estimate; the one after it uses post-change data only. In S2, Practical:
* μ̂ = −0.047 in the phase with t*;
* then −0.014 (mixed);
* then +0.519 (truth +0.515).
In S6, σ̂ goes 0.298 → 0.322 → 0.499 (truth 0.50).

**Why S1 adapts slowly: the timeout doubles when the cluster empties.** After the fall, the stale model admits few
jobs, so Test 1 (capacity) cannot fire. Practical's S1 phases:

| Phase | Hours | End | μ̂ used | σ̂ used |
|---|---|---|---|---|
| 5 (contains t*) | 193.9–295.6 | timeout | −0.047 | 0.298 |
| 6 | 295.6–498.9 (203 h) | timeout (2 × the previous phase) | −0.421 (mixed) | 0.456 (mixed) |
| 7 | 498.9–546.4 | Test 1 | −0.740 | 0.299 |
| 8 | 546.4–575.9 | horizon | −0.739 | 0.300 |

* The mixed phase lasts 203 h because the timeout is twice the previous phase's length.
* The mixture also inflates σ̂ to 0.46.
* Practical adapts fully only 259 h after t*, almost half the remaining stream. It still earns 77% of OPT after t*,
  because the mixed estimate already admits far more than the stale one.
* A Test 2 that could fire would have cut phase 5 short. This is where the calibrated tests should help most.

**By-end-reason merging after Test 1** (S2, S6). The phase with t* ends by Test 1, so its statistics are merged with
the old ones, not replaced. Adaptation still took only two phases (48 h in S2, 56 h in S6), because μ̂ and σ̂ are
refit on each phase's own markups. The merge affects only the other statistics (arrival rates, durations, costs).

## 4. Myerson check: expected against realised profit after t*

Profit of the admitted jobs after t*: expected φ·d − C_elec under the run's own markup model ("belief"), against
realised payment − C_elec.

| | S1 belief / realised | S6 belief / realised | S2 belief / realised | none |
|---|---|---|---|---|
| Practical (C3) | 198.3 / 280.5 = 0.71 | 1,049.3 / 1,008.0 = 1.04 | 1,856.3 / 1,676.0 = 1.11 | 1.00 |
| Theoretical | 0.66 | 1.05 | 1.09 | 1.00 |
| No learning | 43.2 / 98.7 = **0.44** | 1,296.7 / 1,036.2 = **1.25** | 2,308.1 / 1,216.0 = **1.90** | 1.00 |
| Greedy-Pos | 0.33 | 1.24 | 2.04 | 1.01 |

* The stale model is wrong in the direction the expectations predicted.
  * In S1 it reads the halved markups as nearly worthless; φ is underestimated.
  * In S2 and S6 it overestimates φ.
* The learning models stay within about 10% of 1 in S2 and S6.
* In S1 their ratio is 0.71, not 1: the post-change part includes the long mixed phase 6.

## 5. Bids in bin 1 (never admitted)

The grid is fixed at Phase 0. After t*, 0.72% of S1's bids fall into bin 1 (0.23% in S6, 0% in S2). Before t* the
share is 0.01%. This is the fixed-grid limit the expectations mentioned. At this size it is minor.

## 6. Practical + calibrated tests

`tests.threshold: calibrated` scales the thresholds by κ, chosen on H alone (`test_margins.md`, section 4):
* Test 2's threshold × **κ2 = 0.00725**;
* **κ3 scales every confidence term of Test 3** (ε̃_Test3 and ε_A′ + ε_B′ inside λ̃): × **κ3 = 0.00427**.

The safety factor is 2. There were 0 false alarms in a replay of the stationary run. By the decision rule, the four
runs (`practical_calibrated` in none, s1, s6, s2) started right after Part C.

**Pending:** their results will be added here when the runs finish.

## 7. The stationary runs against week 33 (μ̂ estimated)

Week 34 estimates μ̂ in Phase 0 and at every ReInit; week 33 assumed μ = −σ²/2 (`mu_estimation.md`). On the
stationary stream:

| Model | Week 33 | Week 34 | Change |
|---|---|---|---|
| Practical (C3) | 3,094.7 | 3,104.0 | +0.30% |
| Theoretical | 2,256.2 | 2,248.0 | −0.36% |

Both move by less than 0.5%, within the criterion set for μ̂ on the real data.

## 8. Limits

* **One seed per run.** The S6 difference (1.6%) and the stationary Practical / No learning gap (0.5%) are within
  what another seed could change.
* **Test 3's false-alarm count is an upper bound.** It uses the monitor's per-gap screen.
* **Fluid OPT knows t*.** It is the sum of two hindsight LPs, one per regime.

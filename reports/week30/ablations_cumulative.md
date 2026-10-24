# Week 30: ablations, cumulative round (changes stacked in order)

**Setup:**
* The small-cluster regime: the committed default plus only the capacity block of
  `configs/regime_small_cluster.yaml`. 30k trace; ξ = 0.001 everywhere.
* Five thinning seeds (offsets 2–6). Phase 0, H, the stream and the bids are the same in every run.
* Runner: `scripts/ablation_runner.py cumulative`. Notebook: Section 10 of `notebooks/05_ablations_small_cluster.ipynb`.
* Figure: `reports/week30/figs_ablations/cumulative.png`.

**The configurations:**
* **C0:** the theory model, all flags off.
* **C1:** C0 + A1c (ε_D = 0 inside the thinning cap p_U).
* **C2:** C1 + A3 (Phase 0 grid kept in every phase).
* **C3:** C2 + γ = 0.3.
* **R1:** the A1a reference (thinning off).
* **R2:** A1a + A3.

## Checks

* **C0 reproduces round 1 exactly:** every seed gives the same reward, admitted jobs and phases, with mean
  **39,739.06 $**.
* **C1 equals round 1's A1c and R1 equals round 1's A1a**, bit for bit (`cumulative_checks.csv`).
* With all flags off the model is unchanged, and the fast suite passes. **No Phase 0 change. No change to defaults or
  to the closed decisions** (additive reward, s_C frozen, known bid distribution, quantile 0.90).
* **Provenance** (requirement 1):
  * Every run now records the real git HEAD, a dirty flag and the dirty files.
  * The cumulative runs ran at HEAD `e6e9bee`. The tree was dirty, but the dirty files were `.gitignore`,
    `configs/default.yaml` (your edits, never read), the notebook source and one report table. None of them is code that
    ran.
  * Round 1 ran at HEAD `eba0c63` with the ablation code still uncommitted; it was committed later as `be4c6d3`. So the
    hash round 1 recorded was real, but there was no dirty flag. `meta.csv` now says so.

## Results

| config | what it is | R mean ± std $ | % job MILP | % feasible ceiling | % static Phase 0 | Δ vs C0 $ (95% CI) | Δ vs previous step $ (95% CI) | beyond seed noise (vs previous) |
|---|---|---|---|---|---|---|---|---|
| C0 | theory | 39,739 ± 987 | 32.9 | 28.2 | 88.9 | — | — | — |
| C1 | + ε_D = 0 in p_U | 57,655 ± 1,159 | 47.8 | 40.9 | 128.9 | +17,916 (16,489 … 19,343) | +17,916 vs C0 | **yes** |
| C2 | + fixed grid | 59,193 ± 1,858 | 49.0 | 41.9 | 132.4 | +19,454 (16,594 … 22,313) | +1,538 (−988 … 4,063) vs C1 | no |
| C3 | + γ = 0.3 | **62,119 ± 956** | **51.4** | 44.0 | 138.9 | +22,380 (20,887 … 23,873) | +2,926 (129 … 5,724) vs C2 | **yes** (just) |
| R1 | thinning off | 58,621 (1 run) | 48.6 | 41.5 | 131.1 | +18,882 (17,657 … 20,108)* | — | — |
| R2 | thinning off + fixed grid | 63,285 (1 run) | 52.4 | 44.8 | 141.5 | +23,546 (22,320 … 24,771)* | +4,663 vs R1 (exact, one run each) | n/a |

**About R1 and R2:**
* With the thinning off, the run uses no random numbers, so every seed gives the same run. Each was run **once**, and
  its seed std is 0.
* The intervals marked * come only from C0's seed noise. **They are not 5 independent replications.**
* R2 vs R1 is one exact difference between two single runs, with no interval.

**Micro-benchmarks** (means over the runs):

| config | customer value $ | profit $ | carbon $ | admitted | thinning rejects | physical rejects | phases | timeout / Test 1 / Test 2 / Test 3 / horizon | phases with τ > 0 | Test 1 fires (runs) |
|---|---|---|---|---|---|---|---|---|---|---|
| C0 | 37,471 | 6,198 | −3,930 | 6,636 | 3,811 | 501 | 9.4 | 6.4 / 1.6 / 0.4 / 0 / 1 | 5.8 | 4/5 |
| C1 | 53,512 | 9,330 | −5,186 | 9,596 | 326 | 1,051 | 11.2 | 5.2 / 5.0 / 0 / 0 / 1 | 7.2 | 5/5 |
| C2 | 54,110 | 10,541 | −5,458 | 9,850 | 370 | 1,128 | 13.4 | 7.0 / 5.4 / 0 / 0 / 1 | 5.8 | 5/5 |
| C3 | 56,815 | 10,769 | −5,465 | 10,119 | 132 | 1,110 | 15.8 | 8.4 / 6.4 / 0 / 0 / 1 | 8.0 | 5/5 |
| R1 | 54,641 | 9,267 | −5,287 | 9,880 | 0 | 1,144 | 10 | 5 / 4 / 0 / 0 / 1 | 7 | 1/1 |
| R2 | 57,765 | 10,991 | −5,472 | 10,259 | 0 | 1,105 | 17 | 9 / 7 / 0 / 0 / 1 | 9 | 1/1 |

## What the numbers say

* **Thinning is almost the whole first step:** C1 adds 17.9k $. With ε_D = 0 the thinning still rejects about 326 jobs
  per run, mainly at phase starts (p_U ≈ 0.21 there, because of η). R1, with no thinning at all, is only 1.0k $ above
  C1.
* **The fixed grid adds +1.5k $ on top of C1, within the seed noise.** On top of R1 (thinning off) it adds 4.7k $, one
  exact difference.
* **γ = 0.3 adds +2.9k $ on top of C2, just beyond the noise** (the interval starts at +129 $). It gives more and
  shorter phases (15.8 against 13.4) and fewer thinning rejects (132 against 370).
* **Best stacked result: C3 = 62,119 $** (51.4% of the job MILP, 139% of static Phase 0). R2, with no thinning, is
  63,285 $.
* **Everything is still far below the offline benchmarks:** the bucketed MILP on reference buckets is 115,706 $, the
  job MILP 120,742 $ and the feasible ceiling 141,127 $.

## Where the remaining loss comes from (per-phase tables, seed offset 2)

| config | phases | ended by Test 1 | Test 1 phase lengths, as a share of the allowed 2·W_prev | reward earned in Test 1 phases | total reward |
|---|---|---|---|---|---|
| C0 | 9 | 1 | 0.52 | 19,568 $ | 40,892 $ |
| C1 | 9 | 3 | 0.55, 0.45, 0.27 | 27,681 $ | 59,109 $ |
| R1 | 10 | 4 | 0.55, 0.45, 0.27, 0.13 | 31,290 $ | 58,621 $ |

**Observations:**
* Once the thinning is relaxed, **Test 1 ends more phases early**: 3 for C1 and 4 for R1, against 1 for C0. These phases
  use only 13–55% of their allowed length.
* After each early end, the next phase starts with a shorter window (2·W) and a new warm-up of d_max ticks. In C0 and
  C1, p_U at a phase start is only about 0.19–0.21.
* **The phases that Test 1 ends are not low-reward phases:** they earn about half of the total in C1 and R1.
* **With no thinning at all (R1), DLENT-RT is at 48.6% of the job MILP.** That is close to the type-level policies of
  the same regime: G1 is at 53.4%, and the bucketed MILP on Phase 0 buckets at 40.7% (`nb03_nb04_small_cluster.md`).
* So most of what remains after the thinning is the **type-level decision with Phase 0's estimates**, not the thinning
  itself. The bucketed MILP on reference buckets, which uses the same type idea with the stream's own estimates,
  reaches 95.8%.
* These are numbers from one seed (per-phase) and one regime. No claim beyond them.

## Anything odd

* **The Mac slept for most of the night** while the C3, R1 and R2 runs were active, even under `caffeinate`, because the
  machine was on battery. The results are deterministic and unaffected, but the recorded run times (up to 29,685 s) are
  wall-clock and include the sleep. A normal run takes 4–10 min.
* **A first version of the summary gave R2 vs R1 a zero-width interval** marked "beyond seed noise", which is wrong for
  two single deterministic runs. It was fixed: no interval, "n/a".

---

# Appendix — generated tables (notebook 05, Section 10)

## Ablations — cumulative round: generated tables (notebook 05, Section 10; do not edit by hand)

Runs' git HEAD ['e6e9bee7ec63195e45f28fd54c7d6afb9993dc6e'], dirty [np.True_] (dirty files: ['.gitignore; configs/default.yaml; notebooks/src/05_ablations_small_cluster.py; reports/week30/tables/capacity_scan_table.md']).

### Identity checks vs round 1

| config / seed | round-1 name | R new | R round 1 | identical |
|---|---|---|---|---|
| C0 / 2 | baseline | 40,892.14 | 40,892.14 | True |
| C0 / 3 | baseline | 38,685.04 | 38,685.04 | True |
| C0 / 4 | baseline | 39,173.90 | 39,173.90 | True |
| C0 / 5 | baseline | 39,254.62 | 39,254.62 | True |
| C0 / 6 | baseline | 40,689.58 | 40,689.58 | True |
| C1 / 2 | A1c_epsD_0 | 59,109.22 | 59,109.22 | True |
| C1 / 3 | A1c_epsD_0 | 58,105.99 | 58,105.99 | True |
| C1 / 4 | A1c_epsD_0 | 57,588.40 | 57,588.40 | True |
| C1 / 5 | A1c_epsD_0 | 55,910.04 | 55,910.04 | True |
| C1 / 6 | A1c_epsD_0 | 57,561.74 | 57,561.74 | True |
| R1 / 2 | A1a_thinning_off | 58,621.35 | 58,621.35 | True |

### Configurations

| config | description | runs | R mean $ | R std $ | % of job MILP | % of feasible ceiling | % of static Phase 0 | vs C0: delta mean $ | vs C0: 95% lo $ | vs C0: 95% hi $ | vs C0: beyond seed noise | vs previous step: previous | vs previous step: delta mean $ | vs previous step: 95% lo $ | vs previous step: 95% hi $ | vs previous step: beyond seed noise |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 | theory (all flags off) | 5 | 39,739.06 | 987.18 | 32.91 | 28.16 | 88.87 | nan | nan | nan | False | nan | nan | nan | nan | False |
| C1 | A1c eps_D = 0 in p_U | 5 | 57,655.08 | 1,159.33 | 47.75 | 40.85 | 128.93 | 17,916.02 | 16,489.10 | 19,342.95 | True | C0 | 17,916.02 | 16,489.10 | 19,342.95 | True |
| C2 | C1 + A3 fixed grid | 5 | 59,192.58 | 1,857.71 | 49.02 | 41.94 | 132.37 | 19,453.53 | 16,594.22 | 22,312.83 | True | C1 | 1,537.50 | -988.46 | 4,063.47 | False |
| C3 | C2 + gamma = 0.3 | 5 | 62,118.70 | 956.26 | 51.45 | 44.02 | 138.92 | 22,379.65 | 20,886.76 | 23,872.53 | True | C2 | 2,926.12 | 128.60 | 5,723.64 | True |
| R1 | A1a thinning off | 1 | 58,621.35 | 0.00 | 48.55 | 41.54 | 131.09 | 18,882.30 | 17,656.55 | 20,108.05 | True | C0 | 18,882.30 | 17,656.55 | 20,108.05 | True |
| R2 | A1a + A3 | 1 | 63,284.75 | 0.00 | 52.41 | 44.84 | 141.52 | 23,545.70 | 22,319.95 | 24,771.45 | True | R1 | 4,663.40 | nan | nan | n/a: both deterministic, one exact difference |

### Seed note

| config | seed note |
|---|---|
| C0 | 5 thinning seeds |
| C1 | 5 thinning seeds |
| C2 | 5 thinning seeds |
| C3 | 5 thinning seeds |
| R1 | deterministic: thinning off, no online randomness; 1 run, seed std = 0; its CI comes only from the other configuration's seed noise, NOT a replication |
| R2 | deterministic: thinning off, no online randomness; 1 run, seed std = 0; its CI comes only from the other configuration's seed noise, NOT a replication |

### Micro-benchmarks (means)

| config | customer value $ (mean) | profit $ (mean) | carbon $ (mean) | admitted (mean) | thinning rejects (mean) | physical rejects (mean) | phases (mean) | end timeout (mean) | end test1 (mean) | end test2 (mean) | end test3 (mean) | end horizon (mean) | phases with tau > 0 (mean) | Test 1 fires (runs) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 | 37,471.34 | 6,197.71 | -3,930.00 | 6,636.00 | 3,811.20 | 500.60 | 9.40 | 6.40 | 1.60 | 0.4 | 0.00 | 1.00 | 5.80 | 4/5 |
| C1 | 53,511.45 | 9,329.82 | -5,186.20 | 9,596.00 | 326.20 | 1,050.60 | 11.20 | 5.20 | 5.00 | 0.00 | 0.00 | 1.00 | 7.20 | 5/5 |
| C2 | 54,110.10 | 10,540.91 | -5,458.42 | 9,849.80 | 370.20 | 1,127.80 | 13.40 | 7.00 | 5.40 | 0.00 | 0.00 | 1.00 | 5.80 | 5/5 |
| C3 | 56,815.19 | 10,768.91 | -5,465.40 | 10,119.20 | 132.00 | 1,109.60 | 15.80 | 8.40 | 6.40 | 0.00 | 0.00 | 1.00 | 8.00 | 5/5 |
| R1 | 54,640.97 | 9,267.22 | -5,286.83 | 9,880.00 | 0.00 | 1,144.00 | 10.00 | 5.00 | 4.00 | 0.00 | 0.00 | 1.00 | 7.00 | 1/1 |
| R2 | 57,765.17 | 10,991.13 | -5,471.54 | 10,259.00 | 0.00 | 1,105.00 | 17.00 | 9.00 | 7.00 | 0.00 | 0.00 | 1.00 | 9.00 | 1/1 |

### Per-phase tables (seed offset 2)

#### C0

| phase | end reason | start tick | end tick | duration (ticks) | W_prev | arrivals | admitted | thinning rejects | physical rejects | reward $ | tau > 0 | tau (cpu, ram) | p_U at start |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | timeout | 1 | 28,608 | 28,608 | 14,304 | 152 | 64 | 40 | 0 | 254.16 | True | (0, 0.02039) | 0.187 |
| 2 | timeout | 28,609 | 85,824 | 57,216 | 28,608 | 400 | 136 | 81 | 11 | 809.64 | False | (0, 0) | 0.187 |
| 3 | timeout | 85,825 | 200,256 | 114,432 | 57,216 | 684 | 242 | 134 | 22 | 1,930.99 | True | (0.02012, 0) | 0.187 |
| 4 | timeout | 200,257 | 429,120 | 228,864 | 114,432 | 1,463 | 550 | 338 | 40 | 2,833.82 | True | (0, 0.01965) | 0.187 |
| 5 | timeout | 429,121 | 886,848 | 457,728 | 228,864 | 5,185 | 2,341 | 1,287 | 141 | 11,868.45 | False | (0, 0) | 0.187 |
| 6 | test1 | 886,849 | 1,365,570 | 478,722 | 457,728 | 6,232 | 2,820 | 1,590 | 288 | 19,567.58 | False | (0, 0) | 0.187 |
| 7 | test2 | 1,365,571 | 1,380,020 | 14,450 | 478,722 | 218 | 103 | 46 | 9 | 764.16 | True | (0.004437, 0.006662) | 0.187 |
| 8 | timeout | 1,380,021 | 1,408,920 | 28,900 | 14,450 | 470 | 210 | 110 | 9 | 500.15 | True | (0.09407, 0) | 0.187 |
| 9 | horizon | 1,408,921 | 1,432,910 | 23,990 | 28,900 | 504 | 191 | 118 | 17 | 2,363.20 | True | (0.04999, 0) | 0.187 |

#### C1

| phase | end reason | start tick | end tick | duration (ticks) | W_prev | arrivals | admitted | thinning rejects | physical rejects | reward $ | tau > 0 | tau (cpu, ram) | p_U at start |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | timeout | 1 | 28,608 | 28,608 | 14,304 | 152 | 94 | 8 | 2 | -35.78 | True | (0, 0.02039) | 0.206 |
| 2 | timeout | 28,609 | 85,824 | 57,216 | 28,608 | 400 | 197 | 8 | 23 | 1,124.36 | False | (0, 0) | 0.206 |
| 3 | timeout | 85,825 | 200,256 | 114,432 | 57,216 | 684 | 355 | 7 | 38 | 2,696.05 | True | (0.01777, 0) | 0.206 |
| 4 | timeout | 200,257 | 429,120 | 228,864 | 114,432 | 1,463 | 804 | 29 | 94 | 4,105.41 | True | (0, 0.02187) | 0.207 |
| 5 | test1 | 429,121 | 679,919 | 250,799 | 228,864 | 2,624 | 1,729 | 46 | 156 | 7,017.44 | False | (0, 0) | 0.207 |
| 6 | test1 | 679,920 | 903,521 | 223,602 | 250,799 | 2,835 | 1,899 | 46 | 192 | 12,494.73 | False | (0, 0) | 0.207 |
| 7 | test1 | 903,522 | 1,022,374 | 118,853 | 223,602 | 1,890 | 1,136 | 34 | 278 | 8,168.89 | True | (0.002061, 0) | 0.207 |
| 8 | timeout | 1,022,375 | 1,260,080 | 237,706 | 118,853 | 2,568 | 1,653 | 58 | 109 | 12,191.95 | True | (0.007653, 0.06167) | 0.207 |
| 9 | horizon | 1,260,081 | 1,432,910 | 172,830 | 237,706 | 2,692 | 1,753 | 78 | 261 | 11,346.17 | True | (0.002104, 0) | 0.207 |

#### R1

| phase | end reason | start tick | end tick | duration (ticks) | W_prev | arrivals | admitted | thinning rejects | physical rejects | reward $ | tau > 0 | tau (cpu, ram) | p_U at start |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | timeout | 1 | 28,608 | 28,608 | 14,304 | 152 | 102 | 0 | 2 | -22.12 | True | (0, 0.02039) | 1.00 |
| 2 | timeout | 28,609 | 85,824 | 57,216 | 28,608 | 400 | 201 | 0 | 27 | 1,195.34 | False | (0, 0) | 1.00 |
| 3 | timeout | 85,825 | 200,256 | 114,432 | 57,216 | 684 | 359 | 0 | 39 | 2,729.37 | True | (0.0197, 0) | 1.00 |
| 4 | timeout | 200,257 | 429,120 | 228,864 | 114,432 | 1,463 | 834 | 0 | 93 | 4,300.11 | True | (0, 0.02282) | 1.00 |
| 5 | test1 | 429,121 | 679,919 | 250,799 | 228,864 | 2,624 | 1,774 | 0 | 157 | 7,144.63 | False | (0, 0) | 1.00 |
| 6 | test1 | 679,920 | 903,521 | 223,602 | 250,799 | 2,835 | 1,941 | 0 | 196 | 12,717.57 | False | (0, 0) | 1.00 |
| 7 | test1 | 903,522 | 1,022,374 | 118,853 | 223,602 | 1,890 | 1,162 | 0 | 286 | 8,328.79 | True | (0.002061, 0) | 1.00 |
| 8 | timeout | 1,022,375 | 1,260,080 | 237,706 | 118,853 | 2,568 | 1,699 | 0 | 122 | 12,769.82 | True | (0.00794, 0.06114) | 1.00 |
| 9 | test1 | 1,260,081 | 1,320,344 | 60,264 | 237,706 | 774 | 581 | 0 | 38 | 3,099.44 | True | (0.002104, 0) | 1.00 |
| 10 | horizon | 1,320,345 | 1,432,910 | 112,566 | 60,264 | 1,918 | 1,227 | 0 | 184 | 6,358.40 | True | (0.006956, 0) | 1.00 |

#### Phases ended by Test 1

| config | phases | ended_by_test1 | mean_duration | reward_in_test1_phases | reward_total |
|---|---|---|---|---|---|
| C0 | 9 | 1 | 159,212.22 | 19,567.58 | 40,892.14 |
| C1 | 9 | 3 | 159,212.22 | 27,681.07 | 59,109.22 |
| R1 | 10 | 4 | 143,291.00 | 31,290.42 | 58,621.35 |

### Every run

| config / seed | R $ | customer value $ | profit $ | carbon $ | admitted | thinning rejects | physical rejects | phases | end timeout | end test1 | end test2 | end test3 | end horizon | phases with tau > 0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 / 2 | 40,892.14 | 37,633.19 | 7,317.65 | -4,058.71 | 6,657 | 3,744 | 537 | 9 | 6 | 1 | 1 | 0 | 1 | 6 |
| C0 / 3 | 38,685.04 | 36,623.54 | 6,061.74 | -4,000.24 | 6,606 | 3,822 | 523 | 9 | 6 | 2 | 0 | 0 | 1 | 6 |
| C0 / 4 | 39,173.90 | 38,518.08 | 4,508.08 | -3,852.26 | 6,589 | 3,742 | 476 | 11 | 9 | 0 | 1 | 0 | 1 | 7 |
| C0 / 5 | 39,254.62 | 37,110.45 | 6,119.57 | -3,975.40 | 6,653 | 3,905 | 451 | 11 | 6 | 4 | 0 | 0 | 1 | 6 |
| C0 / 6 | 40,689.58 | 37,471.46 | 6,981.49 | -3,763.38 | 6,675 | 3,843 | 516 | 7 | 5 | 1 | 0 | 0 | 1 | 4 |
| C1 / 2 | 59,109.22 | 53,747.38 | 10,694.34 | -5,332.50 | 9,620 | 314 | 1,153 | 9 | 5 | 3 | 0 | 0 | 1 | 6 |
| C1 / 3 | 58,105.99 | 54,268.06 | 9,026.07 | -5,188.14 | 9,565 | 339 | 1,024 | 13 | 5 | 7 | 0 | 0 | 1 | 7 |
| C1 / 4 | 57,588.40 | 53,702.92 | 9,072.19 | -5,186.71 | 9,638 | 306 | 1,079 | 10 | 5 | 4 | 0 | 0 | 1 | 7 |
| C1 / 5 | 55,910.04 | 52,102.23 | 8,893.56 | -5,085.75 | 9,511 | 350 | 938 | 14 | 6 | 7 | 0 | 0 | 1 | 9 |
| C1 / 6 | 57,561.74 | 53,736.68 | 8,962.96 | -5,137.90 | 9,646 | 322 | 1,059 | 10 | 5 | 4 | 0 | 0 | 1 | 7 |
| C2 / 2 | 60,684.20 | 55,547.74 | 10,558.93 | -5,422.47 | 9,917 | 357 | 1,091 | 12 | 5 | 6 | 0 | 0 | 1 | 5 |
| C2 / 3 | 60,641.49 | 55,775.87 | 10,245.00 | -5,379.39 | 9,930 | 397 | 1,037 | 16 | 9 | 6 | 0 | 0 | 1 | 8 |
| C2 / 4 | 57,634.76 | 53,132.36 | 10,012.34 | -5,509.95 | 9,831 | 375 | 1,176 | 14 | 7 | 6 | 0 | 0 | 1 | 6 |
| C2 / 5 | 60,247.29 | 54,189.96 | 11,586.73 | -5,529.40 | 9,809 | 364 | 1,152 | 12 | 7 | 4 | 0 | 0 | 1 | 5 |
| C2 / 6 | 56,755.18 | 51,904.56 | 10,301.52 | -5,450.90 | 9,762 | 358 | 1,183 | 13 | 7 | 5 | 0 | 0 | 1 | 5 |
| C3 / 2 | 62,566.03 | 57,233.22 | 10,767.64 | -5,434.83 | 10,130 | 146 | 1,088 | 17 | 9 | 7 | 0 | 0 | 1 | 9 |
| C3 / 3 | 62,768.89 | 57,353.43 | 10,845.55 | -5,430.09 | 10,178 | 115 | 1,070 | 16 | 9 | 6 | 0 | 0 | 1 | 8 |
| C3 / 4 | 62,331.06 | 57,052.04 | 10,714.55 | -5,435.54 | 10,124 | 150 | 1,090 | 17 | 9 | 7 | 0 | 0 | 1 | 9 |
| C3 / 5 | 60,431.32 | 55,271.16 | 10,769.81 | -5,609.65 | 10,011 | 123 | 1,214 | 12 | 6 | 5 | 0 | 0 | 1 | 5 |
| C3 / 6 | 62,496.21 | 57,166.07 | 10,747.01 | -5,416.87 | 10,153 | 126 | 1,086 | 17 | 9 | 7 | 0 | 0 | 1 | 9 |
| R1 / 2 | 58,621.35 | 54,640.97 | 9,267.22 | -5,286.83 | 9,880 | 0 | 1,144 | 10 | 5 | 4 | 0 | 0 | 1 | 7 |
| R2 / 2 | 63,284.75 | 57,765.17 | 10,991.13 | -5,471.54 | 10,259 | 0 | 1,105 | 17 | 9 | 7 | 0 | 0 | 1 | 9 |

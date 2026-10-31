# Week 33: final results at f = 0.5

All numbers come from notebook 01 (`notebooks/01_model_check.ipynb`), rebuilt and run top to bottom on
`configs/default.yaml`. They also come from the caches in `reports/week33/cache/`. The logs of every run are in
`reports/week33/logs/`.

**What was run.**

| Step | Command | Time | Log |
|---|---|---|---|
| f rule on H | `scripts/select_f.py --seed 42` | 30 s | `select_f.log` |
| Stream models at f = 0.5 | `scripts/model_check_cache.py bench` | under 1 min | `bench.log` |
| Full theoretical run, seed 42 | `scripts/model_check_cache.py theory` | 223 min | `theory.log` |
| Discretization MILPs, 7 days | `scripts/discretization_milp.py --days 7 --time-limit 3600 --seed 42` | 2 × 3,600 s | `disc.log` |
| Notebook 01 | `notebooks/src/build.py 01_model_check --execute` | – | `nb01_build.log` |

The three long runs ran in parallel. The Mac slept from 01:24 to 10:07, so they paused during that time.
`caffeinate` then kept the Mac awake. The MILP check "no sleep during a solve" passes: our timer and HiGHS's
timer agree for both MILPs.

## 1. The choice of f

**The rule (D3), on H only.**
* (a) Required: τ > 0 (CPU or RAM) in the Phase 0 LP.
* (b) Among the f that pass (a), take the f whose "% of H's time with the cluster ≥ 95% full" is closest to the
  20–30% band.
* **Ties go to the largest f.** This gives the least artificial scarcity.

**What changed.** Ties used to go to the f closest to 25%, and that picked f = 0.35. Now they follow the original
wording of rule (b): among the f inside the band, take the largest. The reason: for f = 0.30–0.50 the percentages
differ only by noise. They are not monotone in f (22.3, 25.9, 21.6, 23.2, 22.3%). So "closest to 25%" was picking
noise.

| f | τ CPU | τ RAM | (a) | % of H's time ≥ 95% full | distance to band | C (cores / GB) | Q (cores / GB) | H jobs kept |
|---|---|---|---|---|---|---|---|---|
| 0.30 | 6.6e-4 | 2.0e-3 | yes | 22.3 | 0 | 18.8 / 51.3 | 3.8 / 10.3 | 43,056 |
| 0.35 | 3.9e-4 | 1.7e-3 | yes | 25.9 | 0 | 23.2 / 64.1 | 4.7 / 12.8 | 43,675 |
| 0.40 | 3.5e-4 | 1.4e-3 | yes | 21.6 | 0 | 27.8 / 75.9 | 5.6 / 15.2 | 44,133 |
| 0.45 | 3.1e-5 | 1.1e-3 | yes | 23.2 | 0 | 33.5 / 91.9 | 6.7 / 18.4 | 44,496 |
| **0.50** | 0 | 9.5e-4 | yes | 22.3 | 0 | **40.1 / 108.8** | **8.0 / 21.8** | 44,719 |
| 0.55 | 0 | 2.6e-4 | yes | 16.8 | 3.2 | 60.1 / 149.7 | 12.0 / 30.0 | 45,164 |
| 0.60 | 0 | 0 | no | 12.8 | 7.2 | 73.6 / 188.9 | 14.7 / 37.8 | 45,402 |

At f = 0.5 the cluster is C = 0.626 / 0.425 machines and the quota is Q = 0.2 C. One machine has 64 cores and
256 GB. At f = 0.5 only RAM is scarce in the Phase 0 LP: τ CPU = 0 and the CPU LP fill is 97.4%.

Code: `theory_eval.select_f` (key: distance to the band, then −f). The test is
`tests/test_f_rule_H.py::test_select_f_rule`. The config comment is in `configs/default.yaml`.

## 2. The slow run at f = 0.35, and a fix

**What happened.** The run at f = 0.35 printed its last progress line after 17.6 min, at 10% of the arrivals in
phase 2. After about 6 hours it had printed nothing new. It was stopped. Its log is `theory_f035_stopped.log`.

**Diagnosis.** py-spy needs root on macOS, and sudo was not available. So the stack was sampled with macOS
`sample`, three times, 10 s each (`theory_f035_sample1-3.txt`). The process was in the normal arrival loop. The
main thread's time split as follows:
* **About 45%:** `np.asarray` on a Python list of ints, in `PhaseMonitor._counts`. On every call, it converted
  the list of all arrival ticks of the phase into an array. It is called once per arrival, plus once per Test 3
  checkpoint (about 18). So the cost of one arrival grows with the number of arrivals in the phase. A long phase
  costs O(n²).
* **About 35%:** small HiGHS LP solves in the window LP. Most of this is set-up cost per solve.
  For comparison, the f = 0.5 run uses about 10 LP solves per arrival.

**Why f = 0.35 was so much slower: not fully explained.** The quadratic list conversion is real, and it hurts most
in long phases. Phase 2 can last up to 2 × the previous phase (96 h). But the samples alone do not explain a
slowdown of 10× or more. The LP part probably also grew at f = 0.35, where both resources are scarce (τ CPU > 0
and τ RAM > 0). The per-phase LP counts of the stopped run were never written, so this was not measured.

**Fix (a real performance bug, the results do not change).**
* `PhaseMonitor` now keeps the arrivals of the phase in growing numpy arrays (doubling).
* `_counts` takes slices of them and no longer copies a list.
* Same counts, same decisions. These tests pass:
  * the new `tests/test_monitor_counts.py`, which compares with the old list version across the buffer growth;
  * `tests/test_monitor_exact.py`, the exact monitor against the tick-by-tick brute force (5 cases);
  * `tests/test_regression_snapshot.py`;
  * `tests/test_online_smoke.py`.
* The new f = 0.5 run took 223 min. The last f = 0.5 run, before the H-only hardware profiles, took 323 min. The
  data differ slightly between the two runs, so the speed-up cannot all be credited to the fix.

## 3. Results at f = 0.5

**Notebook checks.**
* Every PASS / FAIL check passes but one: "Theorem 1 not empty at the data ξ". It is expected (D5). The data ξ is
  0.200 against ξ_max = 0.003, so ξ = 0.001 and γ = 1 are forced. At the forced ξ, 1 − ε_D − ξ = 0.489 > 0.
* The D1–D6 checklist is all PASS where a check exists.

**Data.**
* 300,000 jobs. 13,806 are removed by the quota (4.6% of the jobs, 99% of the CPU-hours).
* H = 44,719 jobs (first 7 days). The stream = 241,475 jobs over 575.9 h.

**Bids (section 2).**
* U0 = 0.0214 $/h. The refit σ = 0.301. The mean markup is 1.001.
* Hourly bid on H: p10 / p50 / p90 / p99 / max = 0.043 / 0.072 / 0.141 / 0.280 / 0.753 $/h.
* Total value b·d: p50 = 0.0016 $, p90 = 0.037 $, max = 11.7 $. Most jobs are short: the median duration is 1 minute.

**Objectives (section 4).**
* s_C = 9.88 $/kg.
* Shares of Σ|term| (customer value / profit / carbon):

  | Weights | Customer value | Profit | Carbon |
  |---|---|---|---|
  | LP-admitted mass p·x | 43% | 31% | 26% |
  | types with r̃ > 0 | 42% | 29% | 29% |
  | all types (not used) | 0% | 100% | 0% |

  The share over all types is not used: one never-admitted type has r̃ = −9.6·10⁹, and it takes all of the profit
  term.

| Set | Jobs | Total V $ | Total Π $ | Total carbon $ |
|---|---|---|---|---|
| H (all jobs) | 44,719 | 1,367.8 | 406.2 | 1,170.4 |
| H (at or above k*) | 31,580 | 983.9 | 694.5 | 628.6 |
| run (admitted) | 100,028 | 2,028.0 | 1,432.6 | 1,204.4 |

**Phase 0 on H (section 8).**
* 4,195 populated types out of 20,480. 67.5% of them have r̃ > 0, carrying 79.1% of the arrival mass.
* LP optimum 6.09 $/h. r_max = 1.253 $.
* If every job were admitted, the cluster would be 1.99× full on CPU and 1.98× on RAM.

**Benchmarks (section 9).**

| Model | Customer value $ | Profit $ | Carbon $ | Reward $ | Social welfare $ | Jobs admitted |
|---|---|---|---|---|---|---|
| fluid OPT (upper bound) | 3,166 | 2,251 | −1,894 | 3,523 | – | – |
| Phase 0 LP, expected | 3,208 | 2,246 | −1,945 | 3,509 | – | – |
| static Phase 0 policy | 2,692 | 1,890 | −1,537 | 3,045 | 1,147 | 142,410 |
| no-learning model | 2,113 | 1,498 | −1,297 | 2,314 | 809 | 103,154 |
| **DLENT-RT (theoretical)** | 2,028 | 1,433 | −1,204 | **2,256** | 817 | 100,028 |

* **τ by window:** τ RAM > 0 in 100% of the 24 h windows and in 76% of the 6 h windows. τ CPU > 0 in 78% and 62%.
* **Discretization cost (first 7 days, 58,201 jobs, 48,945 candidates).** Both MILPs hit their 3,600 s limit with a
  gap of 0.70%.
  * (a) Value loss OPT_true − OPT_bin: 26.4 .. 40.4 $ (2.6 .. 4.0%).
  * (b) Decision loss: 0.61 .. 7.69 $ (0.06 .. 0.76%).
  * (c) Worst case of the grid, 1 − 1/ρ: 3.07%.
  * So most of the grid's cost is in valuing a job at L_k, not in the jobs it chooses.

**The theoretical run (section 10).**
* 10 phases: 7 ended by Test 1, 2 by timeout, 1 by the horizon. Test 2 and Test 3 never fired.
* Mechanism:
  * IR violations: 0 of 100,028.
  * Upper-set violations: 0.
  * Truthfulness spot check: 0 violations in 1,600 re-runs.
  * The reserve never sets the payment.
* Payments: 1,428 $. Realised profit (payments − C_elec): 1,422 $. Expected profit: 1,433 $.
* Rejections:
  * bin below k*: 83,556;
  * thinning: 54,549;
  * no space: 3,342.
* **DLENT-RT earns 64% of fluid OPT. The static Phase 0 policy earns 86%.** The main gap is the theoretical
  thinning: it removes 54,549 jobs that the threshold would admit. The no-learning model, which has the same
  thinning, is close to DLENT-RT (2,314 $ vs 2,256 $).

## 4. Files changed

* `dlent_rt/theory_eval.py`: `select_f` tie-break goes to the largest f.
* `tests/test_f_rule_H.py`: the tie-break test.
* `configs/default.yaml`: f = 0.5, with the comment.
* `scripts/select_f.py`, `README.md`: the rule's wording.
* `dlent_rt/models/core/monitor.py`: arrivals kept in numpy arrays.
* `tests/test_monitor_counts.py`: new test for the fix.
* `dlent_rt/diagnostics.py`, `notebooks/src/01_model_check.py`, `notebooks/01_model_check.ipynb`:
  * bid definition and distribution;
  * the three objectives, job by job and one at a time;
  * term shares over p·x and over r̃ > 0;
  * τ by window over time;
  * C and Q in cores and GB;
  * section 8 title.

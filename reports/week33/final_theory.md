# Week 33: same mechanism in the baselines, f with hourly bids, one-week MILP, full theoretical run

All raw numbers are in `reports/week33/final_theory_numbers.md`. It is written by `scripts/final_theory_numbers.py`
from the caches in `reports/week33/cache/` (`f_scan.pkl`, `week_milp.pkl`, `theory_run.pkl`).

**What was run.**
* The fast tests (`pytest -m "not slow"`).
* The f scan over f ∈ {0.3, 0.35, …, 0.6}.
* Phase 0 at the chosen f.
* The one-week bucketed MILP.
* The full theoretical run at f = 0.5 with seed 42. It took 323 min.
* Notebook 01, rebuilt and run top to bottom.

The model was not changed.

## Part 1. The same mechanism in the baselines

### What changed

The new class `ThresholdPolicy` is in `dlent_rt/models/policy.py`. It applies DLENT-RT's mechanism on a frozen state:
* k* per column = the lowest bin with a positive margin r̃ − v·τ, using `threshold_kstar`.
* A job is admitted if three things hold: its bin is at or above k*, its hourly bid passes the reserve P·price, and
  it fits in the cluster.
* The payment is max(L_k*, r_j) × the hours the job runs.

| Baseline | State | τ | Thinning |
|---|---|---|---|
| static Phase 0 | Phase 0, never updated | Phase 0 LP | none |
| no-learning model | Phase 0, never updated | Phase 0 LP | the theoretical p_U(W0, t), same random numbers as DLENT-RT |
| Greedy-Pos | Phase 0 | 0 (admit if r̃ > 0) | none |
| fluid OPT, fluid on feasible types, bucketed MILP | unchanged (upper bounds) | – | – |

Other changes:
* `mechanism_summary` gives the same checks for every policy: payments, IR violations, upper-set violations and
  rejection reasons.
* `theory_eval.static_run` and `benchmarks.ladder` use `ThresholdPolicy`.

### Tests

* `tests/test_mechanism.py::test_baselines_use_the_same_mechanism` checks these points:
  * admit = (bin ≥ k*) and (passes the reserve) and (not thinned);
  * payment = max(L_k*, r) × hours;
  * IR = 0 and upper set = 0;
  * the static k* equals Phase 0's k*;
  * only the no-learning model thins;
  * Greedy-Pos's k* is at or below the static k* in every column.
* `tests/test_ladder.py::test_baselines_use_the_mechanism`: in every baseline, payments are > 0, IR = 0 and upper set = 0.

## Part 2. Recompute with hourly bids

### f rule

The rule:
* (a) τ > 0 in both the Phase 0 LP and fluid OPT;
* (b) among the f that pass (a), the one whose static policy is ≥ 95% full closest to 20–30% of the time (ties go to
  25%).

| f | τ Phase 0 (CPU, RAM) | τ fluid (CPU, RAM) | (a) | static ≥ 95% full | fluid OPT $ | static $ | no-learning $ |
|---|---|---|---|---|---|---|---|
| 0.3 | 0.000597, 0.001964 | 0.000696, 0.00193 | yes | 26.9% | 2,575 | 2,182 | 1,660 |
| 0.35 | 0.000253, 0.001648 | 0.000417, 0.001585 | yes | 26.7% | 2,732 | 2,224 | 1,696 |
| 0.4 | 0.000325, 0.001443 | 0.000487, 0.001246 | yes | 27.8% | 3,090 | 2,702 | 2,027 |
| 0.45 | 0.000176, 0.001153 | 0.000373, 0.000969 | yes | 26.0% | 3,311 | 2,854 | 2,136 |
| **0.5** | 0, 0.000831 | 0.000048, 0.000821 | yes | **25.6%** | 3,363 | 2,933 | 2,215 |
| 0.55 | 0, 0 | 0, 0.000196 | **no** | 20.3% | 3,517 | 3,002 | 2,140 |
| 0.6 | 0, 0 | 0, 0 | **no** | 20.0% | 3,973 | 3,502 | 2,420 |

* f = 0.3 to 0.5 pass (a), and all five are inside the band. f = 0.5 is the closest to 25%.
* The rule picks **f = 0.5**. This is the value already in `configs/default.yaml`, and its comment now explains the
  choice.

### Phase 0 at f = 0.5

| Quantity | Value |
|---|---|
| U0 | 0.02138 $/h |
| σ | 0.301 |
| grid v_min, v_max | 0.01385, 0.7526 $/h |
| ρ, 1 − 1/ρ | 1.0317, 3.07% |
| bins used by H | 125 of 128 |
| s_C | 10.62 $/kg |
| τ (CPU, RAM) | 0, 0.000831 |
| LP fill (CPU, RAM) | 91.7%, 100% |
| r_max (Tests 2–3 unit) | $1.19 |
| columns with k* = 0 | 0 |

Objective shares (% of the sum of |term|):

| | customer value | profit | carbon |
|---|---|---|---|
| over LP-admitted mass | 42.7 | 30.1 | 27.2 |
| over types with r̃ > 0 | 41.9 | 28.4 | 29.7 |

### One-week bucketed MILP

Setup: the first 7 days of the stream, 58,201 jobs (46,054 candidates). The run had a 3,600 s limit and its own HiGHS
instance.

| | $ |
|---|---|
| fluid, volume inside the window = fluid on feasible types | 1,065.55 |
| MILP best / bound | 1,067.89 / 1,068.26 (gap 0.035%, time limit) |
| discretization cost (fluid − MILP) | **−2.34 to −2.71 (−0.22% to −0.25%)** |
| grid worst case 1 − 1/ρ | 3.07% |
| MILP schedule, valued at the jobs' true rewards | 643.48 |

The MILP schedule is feasible on the true durations.

* The cost is **negative**: the MILP is above the fluid. So this is not a valid discretization cost.
* The reason: the bucket reward uses the type's mean volume, not the job's true duration. So inside each type, the
  MILP picks the short jobs. They use little capacity but are credited the full mean reward.
* Valued at the true job rewards, the same schedule is worth only $643.
* What we can say: the believed MILP and the fluid agree within 0.25%, far below the grid's worst case of 3.1%. The
  grid's coarseness costs nothing visible at this scale. A proper lower-bound comparison would need a MILP that
  rewards each job with its own duration.

## Part 3. Full theoretical run (f = 0.5, seed 42)

### Phases

| Phase | Start h | Length h | End | Arrivals | Admitted | τ CPU | τ RAM | σ | DLENT-RT $ | no-learning $ | Δ $ | static $ | fluid on the phase $ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 48.0 | timeout | 15,804 | 7,076 | 0 | 0.000831 | 0.301 | 200.0 | 200.0 | 0.0 | 276.8 | 307.0 |
| 2 | 48 | 52.0 | Test 1 | 18,260 | 8,096 | 0 | 0.001173 | 0.301 | 173.7 | 183.0 | −9.3 | 238.4 | 292.9 |
| 3 | 100 | 47.1 | Test 1 | 17,266 | 8,446 | 0 | 0.000214 | 0.299 | 177.6 | 168.2 | +9.4 | 254.7 | 282.1 |
| 4 | 147 | 50.9 | Test 1 | 15,191 | 6,704 | 0.000299 | 0.000577 | 0.298 | 199.4 | 200.5 | −1.1 | 265.2 | 282.9 |
| 5 | 198 | 101.9 | timeout | 23,412 | 10,093 | 0 | 0.000921 | 0.298 | 332.6 | 336.6 | −4.0 | 481.8 | 553.8 |
| 6 | 300 | 57.5 | Test 1 | 29,370 | 13,363 | 0 | 0.000718 | 0.299 | 250.7 | 258.6 | −7.9 | 303.9 | 373.7 |
| 7 | 357 | 48.3 | Test 1 | 25,221 | 10,029 | 0.000184 | 0.000842 | 0.299 | 184.7 | 187.8 | −3.1 | 237.4 | 294.2 |
| 8 | 406 | 65.6 | Test 1 | 40,205 | 15,773 | 0.000342 | 0 | 0.299 | 222.2 | 228.9 | −6.7 | 314.3 | 361.7 |
| 9 | 471 | 47.7 | Test 1 | 17,920 | 7,025 | 0.000129 | 0.000315 | 0.299 | 209.7 | 210.5 | −0.8 | 264.2 | 295.8 |
| 10 | 519 | 41.7 | Test 1 | 26,867 | 9,874 | 0.000046 | 0.001308 | 0.300 | 180.9 | 180.7 | +0.2 | 220.4 | 290.9 |
| 11 | 561 | 15.2 | horizon | 11,959 | 4,537 | 0.000336 | 0.000591 | 0.302 | 59.3 | 60.1 | −0.8 | 75.7 | 94.6 |
| **Total** | | 575.9 | | 241,475 | 101,016 | | | | **2,190.8** | **2,214.9** | **−24.0** | 2,932.7 | 3,429.6 |

How the run went:
* 11 phases: 8 ended by Test 1, 2 by timeout and 1 at the horizon.
* Tests 2 and 3 never fired, and there were no short phases.
* The τ and σ shown for a phase are the values used in that phase. Phase 1 uses the Phase 0 state.
* The no-learning and static columns are the baselines' rewards over the same time windows.

### Reward, welfare, payments

| | customer value | profit | carbon | reward | % of fluid OPT | admitted | payments | social welfare |
|---|---|---|---|---|---|---|---|---|
| fluid OPT | 3,164.85 | 2,245.86 | −2,047.51 | 3,363.21 | 100.0 | – | – | – |
| DLENT-RT | 2,040.89 | 1,439.51 | −1,289.57 | 2,190.83 | 65.1 | 101,016 | 1,438.95 | 744.83 |
| no-learning | 2,076.53 | 1,474.10 | −1,335.75 | 2,214.87 | 65.9 | 101,292 | 1,473.22 | 734.01 |
| static Phase 0 | 2,680.85 | 1,889.12 | −1,637.28 | 2,932.69 | 87.2 | 139,732 | 1,882.11 | 1,035.33 |

### Mechanism checks

| | DLENT-RT | no-learning | static |
|---|---|---|---|
| expected profit Σ(φ·d − C_elec) $ | 1,439.51 | 1,474.10 | 1,889.12 |
| realised profit (payments − C_elec) $ | 1,432.46 | 1,466.46 | 1,873.86 |
| mean payment / bid | 0.721 | 0.724 | 0.721 |
| share of payments set by the reserve | 0 | 0 | 0 |
| IR violations | 0 | 0 | 0 |
| upper-set violations | 0 | 0 | 0 |
| rejected: bin below k* | 82,233 | 81,021 | 81,021 |
| rejected: below reserve | 0 | 0 | 0 |
| rejected: thinning | 55,084 | 55,075 | 0 |
| rejected: no space | 3,142 | 4,087 | 20,722 |

* **Myerson check (DLENT-RT).** Expected profit $1,439.51 against realised $1,432.46, a 0.5% difference.
  C_elec = $6.50.
* **Truthfulness spot check.**
  * Sample: 200 admitted and 200 rejected jobs.
  * Re-run: 1,600 decisions, with bids × 0.5, 0.8, 1.25 and 2.
  * Result: 0 violations. The logged k* equals the recomputed k* in every case.
* **The reserve never binds.** L_k* is always above P·price, so no job is rejected by the reserve and no payment is
  set by it.

### Does learning help (DLENT-RT vs no-learning)?

**No.** DLENT-RT earns $2,190.83 and no-learning earns $2,214.87: 1.1% less. Per phase it wins 2 of 10 phases
(phases 3 and 10), ties phase 1 and loses the rest by small amounts.

Why:
1. **Phase 1 is identical** (200.0 vs 200.0). This is expected: both use the Phase 0 state and the same thinning
   draws. It confirms that the two models share the mechanism.
2. **Learning has little to learn.**
   * The oracle is frozen, so the type values r̃ only change through σ and τ.
   * σ stays between 0.298 and 0.302 in every phase (Phase 0: 0.301).
   * τ moves between 0 and 0.0013. Each τ is fitted on the previous phase, which is only 42–102 h long, while Phase 0
     uses H's 7 days. So the phase τ is noisier, not better.
   * The τ swings show up directly in the per-phase Δ:
     * Phase 3 uses a lower RAM τ (0.00021 vs 0.00083). It admits more and gains $9.4.
     * Phase 2 uses a higher one (0.00117). It admits less and loses $9.3.
   * Over the run, the gains and losses roughly cancel, with a small net loss.
3. **The thinning sets the level, and learning cannot change it.**
   * p_U ≈ 0.655 in every phase, in both models.
   * It depends on η(W_prev) and η(t − τ_{m−1}), which barely move for phases of 40–100 h.
   * It rejects about 55,000 jobs in each model (55,084 and 55,075), about one third of the jobs both models want.
   * That is the 22-point gap between the static policy (87% of fluid OPT) and both DLENT-RT and no-learning (65–66%).
4. **Phases are long and similar.** All phases but the last last 41–102 h. Test 1 (capacity) ends 8 of them. Tests 2
   and 3, the ones that detect a change in the reward environment, never fire. So no phase starts because the
   environment changed. ReInit refits τ on a window of about the same length, with no new information to exploit.

So, with a frozen oracle and stable σ, the learning loop only re-estimates the dual prices τ on short windows. Phase
0's estimate from 7 days of H is already as good. What separates DLENT-RT from the static policy is the theoretical
thinning, not learning.

## Part 4. Notebook 01

* A copy of notebook 01 (with your uncommitted edits) was saved as `archive/notebooks/01_model_check_local.ipynb` at
  12:15 on 30 Oct. It is not committed. The notebook changed again later that day, before the rebuild, and that later
  version was not kept.
* The notebook was rebuilt from `notebooks/src/01_model_check.py` with the new caches and run top to bottom.
* Sections updated:
  * 6: the f scan over 0.3–0.6, with the static policy using the mechanism;
  * 9: the static and no-learning mechanism results, and 1 − 1/ρ in the MILP table;
  * 10: the full run's Myerson, IR, upper-set, spot-check and rejection numbers.

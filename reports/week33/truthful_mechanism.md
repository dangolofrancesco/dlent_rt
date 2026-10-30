# Week 33: hourly bids, column costs, truthful threshold admission, electricity reserve

Reference: report Decision 7. Goal: a truthful mechanism in the Myerson sense, with monotone admission and a
critical-value payment.

**What was run.**
* The fast tests (`pytest -m "not slow"`): 93 passed.
* The synthetic snapshot run.
* A small real batch: Phase 0 on H, then the theoretical model on the first 48 h of the stream. The same batch was
  run on the last commit (72f9e3f) for comparison.
* Not run: the full simulation, the f scan, notebook 01.

Code and numbers: `scripts/truthful_batch.py` writes `reports/week33/truthful_batch.json`.

This commit also contains Part 3 (`reports/week33/monotone_reserve.md`): the grid and types fixed from Phase 0,
carry-forward, and the reserve. Its option A search is replaced here by the threshold below.

## 1. What changed

### Change 1: the bid is an hourly price

`dlent_rt/bid.py`
* `cost_base` (line 102): c_j = U0 (1 + ln q_j) + γ1 cores_j + γ2 GB_j, in $ per hour. There is no duration.
* `synthesize_bids` (line 107): b_j ~ lognormal with mean c_j and σ = 0.3. It never reads the duration.
* φ_j = c_j φ_m(b_j / c_j), in $ per hour.
* Unchanged: the U0 calibration (the hourly cost of H's median job) and the σ refit on the markups b / c (Phase 0
  and ReInit).

`dlent_rt/reward.py`, `job_terms` (line 55): the results use the true duration d_j.
* V_j = b_j d_j.
* Π_j = φ_j d_j − C_elec,j.
* R_j = V + Π − s_C K.
* No decision reads these.

The grid is geometric on the hourly bids of H (K* = 128), built once in Phase 0 and never changed.

### The φ_m fix

* **Before.** φ_m(m) = m − (1 − F)/f used `0` for the inverse hazard where f < 1e-12. Deep in the left tail this
  gave φ_m = m > 0. So φ_m was **not monotone**: for example φ_m(0.05) = +0.05 but φ_m(0.3) = −392.
* **Why it matters.** Truthfulness needs φ increasing. Then the margin rises with the bid, the admitted bids form an
  upper set, and the critical-value payment is well defined. Hourly bins far below a column's cost base hit exactly
  that tail.
* **Now** (`MyersonModel.phi_markup`, `bid.py` line 43). φ_m = m (1 − σ · Mills(z)), with z = (ln m − μ)/σ and the
  Mills ratio computed in log space. It is floored at `PHI_M_FLOOR` = −10¹² (line 21), and m ≤ 0 gives the floor.
* **Checked.** φ_m is non-decreasing on 200,000 markups from 10⁻⁸ to 10⁶, for σ = 0.3, 0.5 and 1. The test
  `E[φ_m] = 0` and the closed form on z ∈ [−6, 6] still pass. The floor binds only below z ≈ −7.5.

### Change 2: the bin sets only the value, the column sets the costs

`dlent_rt/phase0.py`. For type (k, q, h), with λ = 1:

```
r̃_(k,q,h) = d̄_(q,h) · (L_k + φ_(q,h)(L_k)) − C̄elec_(q,h) − s_C · K̄_(q,h)
v_(k,q,h)  = v̄_(q,h) = mean request × mean duration of the column
φ_(q,h)(L) = c_(q,h) · φ_m(L / c_(q,h))
```

* L_k is the lower edge of bin k. c_(q,h) is the column's hourly cost base.
* `type_values` (line 172) builds V = d̄ L_k and d̄ φ(L_k).
* The column helpers: `column_ids` (132), `cols_to_types` (138), `column_means` (144), `column_fallback` (156),
  `column_cost_base` (186), and `PhaseState.col` (250).
* **Phase 0** (steps 6–8, lines 316–340).
  * p per type is the empirical arrival rate.
  * d̄, energy, power, C̄elec, K̄ and the request are column means over H (every H job ran).
  * A column with no H job takes the mean over H's jobs with the same hardware profile. 10 of the 160 columns
    on real data.
* **ReInit** (`models/core/reinit.py`, `build_window_state` line 58).
  * c_(q,h) comes from all arrivals (line 79): it is known at arrival.
  * d̄, energy, costs and the request come from the admitted jobs only (line 81).
  * A column with no job keeps its previous values: the previous phase, then Phase 0, then the fallback.
  * Source codes are now per column: `src_cost`, `src_dur`.
* s_C = D_q / K_q is computed on these type terms in Phase 0, then frozen.
* **Durations.** Every decision uses only the column's estimated d̄_(q,h). It never uses the true duration or a
  per-bin duration. The true duration is read only after the decision: occupancy, billing and results.

### Admission and payment

`dlent_rt/mechanism.py`, used in `models/theory.py`.
* **Threshold.** In a column, margin_k = r̃ − τᵀv grows with k: only L_k + φ(L_k) changes with the bin.
  * `threshold_kstar` (line 24): k* = the lowest bin with margin > 0, or 0 if none.
  * It **raises** if the admitted bins are not an upper set.
  * The option A search and `group_duration_hours` are removed.
* **Reserve per hour.** `reserve_price` (line 38): r_j = P_j · price(t_j). P_j is the job's power from its request;
  the price is the one at arrival.
* **Admission.** admit ⇔ bin(b_j) ≥ k* and b_j ≥ r_j and thinning (theory only) and the job fits.
* **Payment.** The hourly payment is p_j = max(L_k*, r_j), so 0 < p_j ≤ b_j. The job pays p_j × the hours it runs
  (`theory.py` line 216). It is logged as `payment_rate` and `payment`. IR is checked per hour.
* **Live oracle** (`_live_scores`, line 161). The same margin, but C_elec and K use the job's power × d̄_(q,h) ×
  the price or carbon intensity at arrival. These do not depend on the bid.

### Other files

* `diagnostics.py`: the checks and the D1–D6 checklist anchors.
* `theory_eval.py`: docstring.
* `config.py` and `configs/default.yaml`: comments only.
* `README.md`: D1, D2, D4, D5, D6.
* `notebooks/src/01_model_check.py`: text and check names. The `.ipynb` is not touched: it has your uncommitted
  edits.

## 2. Tests

All pass: `pytest -m "not slow"`, 93 passed.

* **New** in `tests/test_mechanism.py`:
  * The bid does not depend on d: the same request, priority and seed with other durations give identical bids.
  * r̃ follows the formula, and d̄, costs and volume are the same in every bin of a column. The margin is
    non-decreasing in k in every column. Checked on the Phase 0 state and on a ReInit state.
  * `threshold_kstar` returns the lowest bin with margin > 0 and raises on a non-upper set.
  * Hand-made jobs:
    * once a bid is admitted, every higher bid is admitted;
    * the hourly payment is the same for every admitted bid and equals the critical bid;
    * a bid below k* is rejected;
    * 0 ≤ p ≤ b.
  * Synthetic run: IR per hour, total = hourly × hours, and the rejection reasons.
  * Truthfulness on every arrival of the synthetic run, with bids × 0.5, 0.8, 1.25 and 2:
    * any admitted bid pays the critical value, never less;
    * a higher bid of an admitted job is admitted;
    * a lower bid of a rejected job is rejected.
  * **True duration:** other true durations (× 0.1 to 10; same priority, request and bid) never change the decision,
    k*, reserve or hourly payment. Checked with the frozen and the live oracle, on the Phase 0 state and on a ReInit
    state.
  * The grid is fixed. Columns carry forward, and never-seen columns stay fallback.
* **Updated:**
  * `test_bid` (hourly cost base, V = b d);
  * `test_reward` (job terms with d; the type reward from the type terms);
  * `test_reward_identity` (now: the column terms are the column means of the jobs; it is no longer "r̃ = the mean
    of the job reward");
  * `test_data_pipeline` (column volume);
  * `test_payments` (hourly payment);
  * `test_online_quantile_linear`;
  * `test_quantile_normalization` (a round-off tolerance on the scale of r̃);
  * `test_bucketed_opt`.

### Re-frozen snapshot (`tests/data/snapshot_synthetic.npz`)

The old versions are kept in `archive/week33_part3/`.

| array | Part 3 → now |
|---|---|
| `p0_p` | 217 entries changed (hourly grid, so other bins) |
| `p0_r_tilde`, `p0_v` | all changed (column terms) |
| `p0_s_C` | 9.905 → 11.032 $/kg |
| `p0_lambda_star` | 0.00051 → 0.00043 |
| `p0_util_opt` | (0.362, 0.410) → (0.301, 0.366) |
| `p0_u0_sigma_rmax` | U0 and σ unchanged (0.03375, 0.28337); r_max 0.575 → 0.199 |
| `on_decisions` | admitted 422 → 971; below k* 1,847 → 1,088; thinned 257 → 502; no space 53 → 18 |
| `on_phases`, `on_R`, `on_admit`, `on_payment`, `on_R_total` | changed (R_total 23.02 → 53.08) |
| `on_reasons` | unchanged (timeout 2, horizon 1) |
| `p0_tau` | unchanged (0, 0) |

## 3. Validation on a small real batch

Real data, default config (f = 0.5), seed 42. Phase 0 on H, then the theoretical model on the first 48 h of the
stream (15,804 arrivals). The "last commit" column is the same batch on 72f9e3f, which has total-dollar bids and the
per-type rule.

### Phase 0

| | now | last commit |
|---|---|---|
| U0 ($/h) | 0.02138 | 0.02138 |
| σ (refit) | 0.3010 | 0.3010 |
| grid v_min / v_max | $0.0138 / $0.753 per hour | $0.00023 / $11.68 (total $) |
| ρ; bins used by H | 1.0317; 125 of 128 | 1.0883; 124 of 128 |
| s_C ($/kg) | 10.62 | 11.89 |
| D_q = (Q_0.9(V) + Q_0.9(\|Π\|))/2; K_q | (0.0513 + 0.0464)/2 = 0.0489 $; 0.00460 kg | – |
| τ (CPU, RAM) | (0, 0.000831) | (0, 0.000735) |
| LP fill (CPU, RAM) | 91.7%, 100% | 90.7%, 100% |
| LP optimum | $5.79 per hour | – |
| populated types; with r̃ > 0 | 4,234; 64.5% (77.5% of the arrival rate) | 68.0% |
| H jobs at or above their column's k* (the old run: k̂ = 1) | 69.5% | 68.5% |
| columns with k* = 0 | 0 of 160 | – |

**Objective shares** (Σ\|term\|, weighted):

| | customer value | profit | carbon |
|---|---|---|---|
| over the LP-admitted mass p·x, now | 42.7% | 30.1% | 27.2% |
| over the LP-admitted mass p·x, last commit | 42.2% | 28.0% | 29.8% |
| over types with r̃ > 0 (p-weighted), now | 41.9% | 28.4% | 29.7% |

The share over all types is dropped. A few types whose bin floor lies far below their column's cost base have a
huge negative φ(L_k). They dominate a mean of \|term\|, although they carry 0.002% of the arrival rate and are never
admitted.

**The φ floor is not a numerical problem.**
* The LP solves to "Optimal".
* Every r̃ is finite (the lowest is −6.6e12).
* No type with r̃ < 0 gets x > 0.
* s_C uses 0.9 quantiles, which the one populated type at the floor (2e-5 of the arrival rate) cannot reach.

### The 48 h online run

| | now | last commit |
|---|---|---|
| arrivals | 15,804 | 15,804 |
| passed the price rule (bin ≥ k*) / the oracle | 11,015 | 10,365 |
| **admitted** | **7,257 (45.9%)** | 7,032 (44.5%) |
| rejected: below k* / below reserve / thinning / no space | 4,789 / 0 / 3,561 / 197 | not logged |
| phases; end reasons | 1; horizon | 1; horizon |

* No test can end a phase in 48 h: the tests start 24 h after the phase start, and the first timeout is at 48 h.
* **IR violations: 0.** **Upper-set violations: 0** in the phase states and **0** in the run log (no rejected bin
  above an admitted bin in the same phase and column).

**Median k* vs median job bin per column.** Over the 133 columns with arrivals, (median job bin − median k*) has
arrival-weighted quantiles −4 / 6.5 / 11 bins (10th, 50th, 90th). The busiest columns:

| column (q, profile) | arrivals | median k* | median job bin | jobs at or above k* |
|---|---|---|---|---|
| (4, 28) | 1,971 | 47 | 53 | 74% |
| (4, 29) | 1,411 | 41 | 50 | 84% |
| (4, 16) | 1,256 | 36 | 46 | 85% |
| (4, 19) | 859 | 35 | 46 | 90% |
| (4, 15) | 754 | 40 | 50 | 84% |
| (4, 13) | 679 | 66 | 65 | 47% |
| (4, 2) | 618 | 41 | 50 | 83% |
| (4, 24) | 599 | 73 | 69 | 38% |

**Payments** (admitted jobs):

| | value |
|---|---|
| mean / median hourly payment ÷ hourly bid | 0.720 / 0.724 |
| share set by the reserve (r_j > L_k*) | 0% |
| share paying the floor of bin 1 (k* = 1) | 0% (bin 1's lower edge is 0, so its φ is at the floor and it is never admitted) |
| total payments | $133.40 |
| total value of admitted jobs Σ b d | $185.27 |
| electricity cost of admitted jobs | $0.54 |

The reserve never binds: one hour of electricity (power × price) is far below the bin floors that are admitted.

**Myerson check.**
* Expected profit Σ(φ d − C_elec) over the admitted jobs is **$129.66**.
* Realised profit (payments − C_elec) is $133.40 − $0.54 = **$132.86**.
* They differ by 2.5%. Myerson's lemma says that, in expectation over the bids, the payments equal the virtual value
  of the winners. So this is the expected agreement on one sample.

**Realised reward** (true durations): customer value $185.27 + profit $129.66 − carbon $112.91 = **$202.03**. This
is not comparable with the last commit's $77.86: that one valued jobs by total-dollar bids.

### Truthfulness spot check

* 200 random admitted and 200 random rejected arrivals of the batch.
* For each, the decision is re-run on the same phase state with the bid × 0.5, 0.8, 1.25 and 2. Thinning and the
  gate are kept as logged, since they do not depend on the bid.
* 1,600 decisions. The logged k* equals the recomputed k* in every case.
* **Violations: 0.**
  * No different bid is admitted at a lower hourly payment.
  * No higher bid of an admitted job is rejected.
  * No rejected job is admitted below its true bid.
  * No lower bid of a rejected job is admitted.

## 4. Open issues

* **Not yet re-run with hourly bids:** the f rule (f = 0.5 was chosen with total bids), the one-week MILP, and the
  full simulation. That is the next step. The cached full run (`reports/week33/cache/theory_run.pkl`) is stale.
* **The reserve is inactive on this data** (0% of payments). With hourly bids it acts only as a guard. A reserve
  with real effect would need a higher floor, for example including carbon.
* **The low bins of a column are hopeless.** For jobs whose bin floor is far below their column's cost base, φ(L_k)
  is very negative. This is correct: such bids are never admitted. But it makes an all-types mean of the reward terms
  meaningless, so reports use the admitted mass or the types with r̃ > 0.
* **The type value uses the bin's lower edge.** With ρ = 1.032, a bid loses at most 3.1% to rounding down. Admission
  is slightly conservative: the payment is the lower edge of k*, about 72% of the bid on average.
* **Tests 2–3 unit.** r_max (the largest positive r̃ of a populated type) is now $1.19 on real data, against $7.74
  before. Tests 2–3 did not get a chance to fire in 48 h; check them in the full run.
* **Benchmarks.** The static, no-learning and greedy policies use the new type terms but not the threshold, reserve
  or payment (as in Part 3).

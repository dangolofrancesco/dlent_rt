# Week 33, Part 3: monotone admission (option A), reserve price, grid fixed from Phase 0

> **Superseded in part** by `truthful_mechanism.md`: option A, the per-job reserve and the bid-bin fallback are
> replaced by hourly bids, column costs, a threshold k* and an hourly reserve. The fixed grid and carry-forward stay.

**What was run.**
* Run: the unit tests (`pytest -m "not slow"`: 89 passed), one synthetic online run (the snapshot run), and Phase 0
  on the real config.
* Not run: the full-stream simulation and notebook 01.

## 1. What changed

### Monotone admission (option A)

`dlent_rt/mechanism.py`, used in `dlent_rt/models/theory.py`.
* For the arrival's (priority, hardware profile), the oracle decides every bid bin: admitted iff r̃ − τ·v > 0.
* k* is the lowest bin such that EVERY bin from k* up is admitted (`upper_set_kstar`). k* = 0 if the top bin is
  rejected.
* Bins admitted below a rejected bin are not admitted.
* Frozen oracle: k* is computed once per phase. Live oracle: k* is computed per job, scoring every bin with the
  job's own electricity cost and carbon.

### Reserve price

r_j = P_j · d̄_(q,a) · price(t_j), the job's estimated electricity cost at arrival.
* P_j is the job's power (kW, PUE included) and price(t_j) the price at the arrival hour.
* **One deviation from the request.** d̄ is the mean duration of the job's (priority, profile) in the current phase
  state, p-weighted over the bid bins. It is not the mean of the job's own type.
  * The type includes the bid bin, so the type's d̄ would make r_j depend on the bid.
  * A lower bid could then fall in a type with a shorter d̄ and pay less. That breaks test (4).

### Admission and payment

* admit ⇔ bin(v) ≥ k* and v ≥ r_j and thinning (theory only) and the job fits.
* p = max(L_k*, r_j), with L_k* the lower edge of bin k* (0 for bin 1).
* bin(v) ≥ k* ⇔ v ≥ L_k*, so p is the lowest bid that would still be admitted. p does not depend on the bid.

### Logs per arrival

`k_star`, `L_kstar`, `reserve`, `reserve_ok`, `payment`, `payment_by_reserve`, and `reason`: "bin below k*",
"below reserve", "thinning" or "no space". Per phase:
* the columns with k* = 0;
* the types whose statistics are carried forward or are fallback estimates;
* the jobs below the reserve.

### Design change: grid and types fixed from Phase 0, in every model

* **ReInit** (`dlent_rt/models/core/reinit.py`) no longer rebuilds the grid. It updates only the per-type
  statistics.
  * p_j = N_j / W.
  * Bids, φ, σ and requests come from all arrivals.
  * Duration, energy and costs come from admitted jobs.
  * A type with no job in the phase keeps its previous values (carry-forward).
* **Phase 0** (`dlent_rt/phase0.py`, `fallback_estimates`) gives every type with no job in H the fallback
  estimates: hardware-profile means for duration, energy and costs, bid-bin means for bid and φ, the profile as the
  request. So every bin has an r̃, and k* is computed with no skipping.
* **Removed:** `phase_transition.n_type_min` (thin types under 5 jobs no longer carry; a type carries only when it
  has no job). The grid refit (`refit_grid`) and the hindsight `fallback` switch are gone.
* **Kept:** σ is still refit at ReInit when the phase has ≥ `n_min` arrivals.
* **Updated:** the README (sections D4–D6) and the notebook source text of sections 5 and 10.

### Tests

* **New:** `tests/test_mechanism.py`.
  * (1) The admitted bins of each (priority, profile) form an upper set: on hand-made arrays, and on every arrival of
    the synthetic run.
  * (2) Every admitted job pays 0 < p ≤ v.
  * (3) Raising a bid never turns yes into no: on hand-made jobs, and on 200 admitted arrivals of the run at
    1.01×, 2× and 50× their bid.
  * (4) Truthfulness on hand-made jobs: the payment is the same for every admitted bid, it equals the lowest admitted
    bid, and no lower bid that is still admitted pays less.
  * The reserve formula on a hand-made job.
  * The fixed grid, with carry-forward and fallback after a ReInit.
* **Updated:** `test_payments.py` (p = max(L_k*, r)), `test_bucketed_opt.py` and `test_greedy.py` (Phase 0 now
  gives unseen types fallback values, not r̃ = 0), `test_reward_identity.py`.

## 2. Numbers

### The reserve on H (real config, Phase 0 only)

* 10 of 44,719 H jobs (0.02%) bid below their reserve.
* The reserve is a small part of the bid: median reserve / bid 0.034, p90 0.12.

### The synthetic run (the snapshot run, 2,579 arrivals)

| | value |
|---|---|
| admitted | 422 (16.4%; 1,073 before Part 3) |
| rejected: bin below k* / below reserve / thinning / no space | 1,847 / 0 / 257 / 53 |
| admitted jobs whose payment is set by the reserve (r_j > L_k*) | 0.0% |
| columns with k* = 0 | 0 of 30, in every phase (29 of 30 before the fixed grid, when empty bins counted as rejected) |
| IR violations, payments ≤ 0 | 0, 0 |

Types by source of their statistics (480 types):

| phase | arrivals | admitted | bid / φ: carried | bid / φ: fallback | duration / costs: carried | duration / costs: fallback |
|---|---|---|---|---|---|---|
| 1 (Phase 0 state) | 506 | 31 | 0 | 253 | 0 | 253 |
| 2 | 1,047 | 145 | 61 | 174 | 217 | 244 |
| 3 | 1,026 | 246 | 43 | 70 | 196 | 208 |

## 3. Re-frozen arrays (`tests/data/snapshot_synthetic.npz`, against 72f9e3f)

| array | change |
|---|---|
| `p0_r_tilde`, `p0_v` | changed for the 253 types with no job in H (fallback estimates; before r̃ = 0, v = 0) |
| `on_phases` | 4 → 3 phases |
| `on_reasons` | (test1 1, timeout 2, horizon 1) → (timeout 2, horizon 1) |
| `on_decisions` | admitted 1,073 → 422; no k̂ 837 → 1,847; thinned 593 → 257; no space 76 → 53 |
| `on_R`, `on_admit`, `on_payment` | changed (new admissions and payments) |
| `on_R_total` | 16.77 → 23.02 |
| `p0_p`, `p0_s_C`, `p0_tau`, `p0_lambda_star`, `p0_util_opt`, `p0_u0_sigma_rmax` | unchanged |

## 4. Open issue: option A admits almost nothing on the real data

**On H with Phase 0's state, 0.4% of the jobs (189 of 44,719) are in a bin ≥ k*.** The old per-type rule admitted
68.5%. No column has k* = 0, but k* is high: its median over the 160 columns is bin 98.5, while the median H job is
in bin 24.

The upper set is broken high up in two ways, both from the fallback estimates and φ's noise:

1. **Empty bins.** Bins 124–126 have no H job. Their bid-bin fallback becomes the mean over ALL H jobs: a bid of
   $0.03 in bins whose lower edges are $7.65–9.06. After the carbon cost, their r̃ is negative, so the upper set
   starts at bin 127 in 69 columns.
2. **φ's heavy tail.** Bin 96 has 27 H jobs and a mean φ of −2.3 (bins 95 and 97: +0.35, −0.63). It is rejected and
   breaks the upper set in 76 columns.

Possible fixes, not implemented:
* (a) For a bin with no job, take the bid at the bin's geometric midpoint and φ from the markup model, instead of
  the overall mean.
* (b) Estimate a bin's φ from neighbouring bins, or as a monotone fit over bins.
* (c) Make k* robust: allow a small number of rejected bins with few jobs above k*.

Each changes D4/D5, so it is your decision.

## 5. Other notes

* `notebooks/01_model_check.ipynb` has uncommitted changes (yours, I assume). I did not touch it. I updated only its
  source `notebooks/src/01_model_check.py` and the check functions in `dlent_rt/diagnostics.py` that section 10 calls.
  If you rebuild the notebook from the source, your unsaved notebook edits would be lost. Copy or commit them first.
* The cached full run (`reports/week33/cache/theory_run.pkl`) predates Parts 2 and 3 and lacks the new log fields.
  Section 10 needs a new run.
* The static and no-learning policies (benchmarks) still use the per-type LP fractions. They do not use option A or
  the reserve, as the request was only for the frozen and live oracles.

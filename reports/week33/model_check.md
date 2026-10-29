# Week 33: the final model (D1–D6), one config, the model-check notebook

* Notebook: `notebooks/01_model_check.ipynb` (source `notebooks/src/01_model_check.py`). It replaces notebooks 01
  and 05. It ran top to bottom with no error.
* Config: `configs/default.yaml`, the only one. The model variants are switches inside it.
* Long computations: `scripts/model_check_cache.py`. The cache is in `reports/week33/cache/` (not in git).
* Tag before any deletion: `pre-cleanup-week33` (pushed).
* All numbers are at f = 0.5, seed 42, on `data/batch_v2_300k.csv`, unless a row says otherwise.

**Phase 0 and the default numbers change on purpose.** The bid, the customer value, the grid, the quota and f all
change. The snapshot is re-frozen (section "Re-frozen arrays").

## The notebook, section by section

**1. Data and limits.** All 6 checks PASS.
* Of the 300,000 jobs, 13,806 (4.6%) are over the quota. 286,194 are kept: 44,719 in H, 241,475 in the stream
  (T = 575.9 h).
* There are no other filters.
* Durations stay inside [1 minute, 24 h]. 755 kept jobs are cut at 24 h. 113,681 kept jobs (39.7%) are raised to
  1 minute.
* The quota and the cap remove most of the work. Over all 300k jobs, the quota removes jobs that hold 99.0% of the
  effective CPU-hours, and the cap cuts 65.9% (the two overlap). Only 0.55% of the CPU-hours are left; in H, 1.3%.

**2. Bid and U0.** All 5 checks PASS.
* U0 = 0.02138 $ per hour per job: the cost rate of the median H job (0.43 cores, 1.57 GB).
* The cost base matches the formula by hand on H and on the stream.
* The mean markup is 1.0012, and the std of ln(markup) is 0.3010.
* The mean bid per hour grows with priority (1 + ln q): from $0.058 at q = 1 to about $0.09 at q = 3–5.

**3. Virtual value.** All 4 checks PASS.
* By quadrature, E[φ_m] is 2.5e-3 at σ = 0.1 (quadrature tolerance) and 1.5e-8 at σ = 0.3. φ_m is increasing.
* σ refit on H = 0.3010. At every ReInit it is in [0.2961, 0.3032] (16 phases).
* The reserve markup (φ_m = 0) is 0.715.

**4. Objectives.** All 6 checks PASS.
* All λ = 1, and r̃ = V + Π − s_C K exactly.
* The customer value is the bid (no q). SW per job = v − C_elec − s_C K.
* Shares of Σ|term| over H's types: customer value 35.2%, profit 28.6%, carbon 36.3%.
* Σ p V / Σ p |Π| = 1.23.
* s_C = D_q / K_q = (0.03764 + 0.01843) / 2 / 0.002358 = **11.89 $/kg**, recomputed by hand.

**5. Grid.** All 5 checks PASS.
* v_min = $2.31e-4, v_max = $11.68, ρ = 1.0883. H uses 124 of the 128 bins.
* Worst-case loss from rounding a bid down to its bin edge: 1 − 1/ρ = 8.1%.
* The bins cover every bid: bin 1 = [0, L_1), bin 128 = [L_127, +∞).

**6. Cluster and quota.** All checks PASS.
* D̄ = (1.2518, 0.8501) machines, so C = 0.5 · D̄ = (0.626, 0.425).
* Q = (0.1252, 0.0850) = 0.2 C. The fixed point took 12 steps; the last move was 0.07%.
* **The f rule was made explicit after the preview.** The first version ranked f only by "% time ≥ 95% full", over
  {0.3, …, 0.6}. It picked f = 0.6, where τ = 0 in both LPs. The rule now has two conditions, in order:
  (a) required: τ > 0 (CPU or RAM) in both the Phase 0 LP and fluid OPT;
  (b) among the f that pass (a): the closest to the 20–30% band.

| f | (a) τ > 0 in both LPs | % time ≥ 95% full | distance to band (points) | C CPU / RAM | τ Phase 0 (CPU, RAM) | τ fluid OPT (CPU, RAM) | static $ | fluid OPT $ |
|---|---|---|---|---|---|---|---|---|
| 0.30 | yes | 50.3 | 20.3 | 0.294 / 0.200 | 0.00019, 0.00189 | 0.00023, 0.00182 | 1,162 | 2,105 |
| 0.35 | yes | 49.6 | 19.6 | 0.363 / 0.251 | 0, 0.00167 | 0, 0.00148 | 929 | 2,168 |
| 0.40 | yes | 47.9 | 17.9 | 0.434 / 0.296 | 0, 0.00116 | 0, 0.00124 | 1,371 | 2,494 |
| 0.45 | yes | 47.3 | 17.3 | 0.523 / 0.359 | 0, 0.00086 | 0, 0.00097 | 1,458 | 2,609 |
| **0.50** | **yes** | **44.6** | **14.6** | 0.626 / 0.425 | 0, 0.00073 | 0, 0.00077 | 1,317 | 2,844 |

For reference, the dropped f = 0.6 gave C = 1.15 / 0.74, 31.2% of the time full, and τ = 0 in both LPs.
* **Job-level check** (Greedy-Pos as a capacity probe): it rejects 35,441 of 182,270 positive-reward jobs for lack
  of space (19.4%). That is below 50%, so the cluster would not be enlarged even with the option on.

**7. ξ and γ.** 3 of 4 checks PASS. The FAIL is expected.
* L = ln(|I|/δ) = 44.33 with δ = 1/T³.
* ξ_max = 0.0030 (the largest ξ with 1 − ε_D − ξ > 0).
* The data ξ = 0.200, and the data γ = 86,400 (d_max in ticks dominates).
* At the forced ξ = 0.001: ε_D = 0.510 and 1 − ε_D − ξ = 0.489.
* **FAIL: Theorem 1 is empty at the data ξ** (ε_D = 23.7).
* The quota caps jobs at 0.2 C, so the data ξ stays near 0.2 at every f. For ξ ≤ ξ_max with today's jobs, C would
  need to be 66 times larger.

**8. Phase 0.** All 9 checks PASS.
* 5,408 of 20,480 types have jobs. 68.0% of them have r̃ > 0 (75.7% of the arrival mass).
* τ = (0, 0.000735). LP fill: CPU 90.7%, RAM 100%.
* Types fully / partly / rejected: 3,147 / 1 / 2,260.
* Tests 2–3 reward unit r_max = $0.00552.

**9. Theoretical benchmarks.** All 5 checks of the one-week MILP PASS. Totals over the stream (576 h):

| model | customer value $ | profit $ | carbon $ | reward $ | SW $ |
|---|---|---|---|---|---|
| fluid OPT | 3,046.80 | 1,892.07 | −2,095.20 | 2,843.68 | – |
| fluid on feasible types | same (no type is larger than C) | | | 2,843.68 | – |
| Phase 0 LP, expected | 3,060.59 | 2,017.87 | −2,163.23 | 2,915.23 | – |
| static Phase 0 policy | 2,485.79 | 833.25 | −2,002.12 | 1,316.92 | 474.54 |
| no-learning model | 2,058.78 | 618.58 | −1,775.42 | 901.94 | 275.32 |

* **τ by window.** τ_RAM > 0 in 96% of the 1-day windows and 81% of the 6-h windows. τ_CPU > 0 in 30% and 36%.
* **From H to the stream.** Phase 0 LP expected $2,915 → LP on the stream's statistics with Phase 0's grid $2,833
  → fluid OPT $2,844. Phase 0's x valued on the stream's statistics gives only $1,166. The stream's arrival rate is
  1.53 times H's (weighted over shared types).
* **One-week MILP** (58,201 jobs).
  * Fluid counting only the volume inside the window = fluid on feasible types = $912.19. Fluid counting the whole
    volume = $904.90.
  * MILP: best $862.59, bound $867.81, gap 0.61%. It used all 3,601 s, and its set is feasible on the true
    durations.
  * **Discretization cost: $44.4–49.6 (4.9–5.4%).**

**10. One theoretical DLENT-RT run** (seed 42, 120 min). All 12 checks PASS.
* **Oracle.** It is frozen: every score uses the phase's r̃ (no live price), and k̂ = 1[r̃ − τv > 0].
* **Order.** admit = k̂ ∧ U ∧ fits on all 241,475 arrivals: thinning comes after the LP decision, then the gate.
  The share of U = 1 is 0.653 among arrivals with k̂ and 0.651 without it, so the draw does not depend on the
  decision.
* **ReInit.** It used all arrivals for rates, bids, σ and requests, and only admitted jobs for durations, energy
  and costs. The grid was rebuilt at every ReInit.
* **Payments.** Every payment equals the threshold price. 0 IR violations among 101,449 admitted jobs. 9 admitted
  bids were above the phase grid's v_max; none were below its v_min.
* **Phases.** 16 phases: 6 ended by Test 1, 7 by Test 2, 2 by timeout (at 2 × the previous phase length), 1 by the
  horizon.
* **Totals.**
  * Reward $720.55, SW $286.46.
  * Customer value $1,603.96, profit (φ − C_elec) $428.20, carbon −$1,311.61.
  * Payments $119.95, customer surplus $1,484.01, realised profit (p − C_elec) $114.06.
  * Thinning removed 55,631 jobs that the oracle wanted. The gate blocked 3,263.
  * For comparison: static $1,316.92, no-learning $901.94, fluid OPT $2,843.68.

**11. Checklist.** In the notebook, each item has its file and line. Items marked "code" are checked by reading the
code and by the tests, not by a notebook number.

## D1–D6 checklist

"In place" means the code already did this before week 33; "changed" means I changed it this week.

| | item | status | where |
|---|---|---|---|
| D1 | V = v, Π = φ − C_elec, carbon = s_C K | changed (V was q·v) | `dlent_rt/reward.py:62` |
| D1 | R linear, all λ = 1 | changed (scalarization switch removed) | `dlent_rt/reward.py:138` |
| D1 | s_C = D_q / K_q, once in Phase 0 | in place (now with V = v) | `dlent_rt/reward.py:112` |
| D1 | SW metric only | changed (new) | `dlent_rt/reward.py:66` |
| D1 | Chebyshev only in the offline Pareto code | changed (removed from Phase 0 and config) | `dlent_rt/pareto.py:15` |
| D2 | cost_base = (U0 (1 + ln q) + γ1 cores + γ2 GB) d | changed (one formula) | `dlent_rt/bid.py:97` |
| D2 | U0 = rate of the median H job | changed (new) | `dlent_rt/bid.py:84` |
| D2 | φ = cost_base · φ_m(v / cost_base) | in place | `dlent_rt/bid.py:37` |
| D2 | σ refit in Phase 0 and at ReInit; no-learning keeps 0.3 | changed (new, `bid.refit_sigma`) | `dlent_rt/phase0.py:261`, `dlent_rt/models/core/reinit.py:71` |
| D3 | H = first 7 days | in place (`data.h_days`, now required) | `dlent_rt/data.py:259` |
| D3 | C = f D̄ on H, one cluster | changed (the only capacity rule) | `dlent_rt/data.py:382` |
| D3 | Q = 0.2 C at the actual f | changed (`quota_from_f` removed) | `dlent_rt/data.py:292` |
| D3 | 24 h cap, 1-minute floor, no outlier filters | changed (cap now in `time.d_cap_hours`; R_max removed) | `dlent_rt/data.py:268` |
| D3 | f rule (a) τ > 0, then (b) band | changed (new, explicit after preview) | `dlent_rt/theory_eval.py` `select_f` |
| D3 | job-level enlargement option + check | changed (new, off) | `dlent_rt/benchmarks.py:303` |
| D4 | geometric grid, no floor, covering every bid | changed | `dlent_rt/grid.py` |
| D4 | grid rebuilt at every ReInit | in place | `dlent_rt/models/core/reinit.py:72` |
| D4 | unseen types: dual rule | in place (now the only option) | `dlent_rt/theory_eval.py:45` |
| D5 | frozen oracle | in place | `dlent_rt/models/theory.py` `want` |
| D5 | live oracle switch | changed (new, `oracle.kind: live`) | `dlent_rt/models/theory.py` `want` |
| D5 | thinning after the LP decision, physical gate | in place | `dlent_rt/models/theory.py` `want`, `dlent_rt/models/policy.py:65` |
| D5 | thinning-off switch | in place (A1a; other ablations removed) | `dlent_rt/models/core/error_params.py` |
| D5 | no-learning model | changed (new `NoLearningPolicy`) | `dlent_rt/models/policy.py:226` |
| D5 | ξ = 0.001, γ = 1; Test 1 = (1 − ξ) c | in place | `dlent_rt/models/theory.py` |
| D5 | Tests 2–3 in the same units | changed (rewards / r_max; see below) | `dlent_rt/models/core/monitor.py:87` |
| D5 | tests start d_max after phase start; timeout 2 W_prev | in place | `dlent_rt/models/core/monitor.py:68,134` |
| D5 | ReInit: all arrivals vs admitted; carry-forward | in place | `dlent_rt/models/core/reinit.py` |
| D6 | threshold payment (lowest admitted bin; 0 for bin 1) | changed (was φ at the job's own bin floor) | `dlent_rt/models/theory.py` `record` |
| D6 | surplus, realised profit, IR | changed (new) | `dlent_rt/models/theory.py` `record` |
| D6 | benchmarks; each MILP its own HiGHS instance | in place | `dlent_rt/benchmarks.py` |
| D6 | fluid on a window: volume inside the window | changed (now the main window number) | `dlent_rt/theory_eval.py` `window_models` |

## Code changes

* **`config.py` / `default.yaml`.** One config.
  * Removed: `data.n0`, `limits.r_max_machines`, fixed quotas, `job_quota_rule`, `quota_from_f`,
    `capacity.mode`, `exceeded_share`, `absolute`, `time.d_max_hours`, `d_max_horizon_fraction`, `bid.mode`,
    `base_utility_*`, `u0_per_machine_hour`, `known_distribution`, `grid.space`, `phi_floor`, `model.kind`,
    `scalarization.*`, `policy.unseen_types`, and the thinning ablations A1b, A1c and A3.
  * Added: `bid.refit_sigma`, `time.d_cap_hours`, `limits.quota_tol`, `capacity.job_level_*`, `oracle.kind: live`,
    and `logging.progress_every` / `progress_phases` (a progress line every 10% of arrivals and at every phase end).
* **`bid.py`.**
  * One cost-base formula, `calibrate_u0` and `fit_sigma`.
  * `MyersonModel` is always the known markup. The pooled MLE is gone.
* **`grid.py`.** Geometric edges from the smallest to the largest bid, with no floor. The bins cover [0, +∞), and
  `lower_edges` gives the payment edges.
* **`reward.py`.** V = v, plus `sw` and `social_welfare`. The linear sum replaces the scalarization dispatch.
* **`data.py` / `preprocess.py`.**
  * The quota fixed point at the actual f. `apply_quota` replaces `apply_r_max`.
  * Capacity is mean demand only, and d_max = the 24 h cap.
* **`phase0.py`.**
  * U0, the σ refit, the new grid and `r_max` are stored in the state. `phi_tilde` and the IR-reject sentinel are
    removed.
* **`models/theory.py`.**
  * The live oracle and the threshold payment.
  * Logs for surplus, realised profit, IR, SW and bids off the grid; progress lines.
  * Realised rewards are scored with Phase 0's bid model, one scale for every model. The decisions still use each
    phase's σ.
* **`models/core/monitor.py`.** Tests 2–3 take rewards in units of r_max.
  * Before, rewards were in $ and the thresholds were built for rewards bounded by γ = 1. Now r̃ and R are divided
    by Phase 0's r_max, the p-weighted 0.8-quantile of |r̃|.
  * This happens only when γ is forced. With the data γ, which contains r_max in $, rewards stay in $.
* **`models/core/reinit.py`.** σ refit and grid rebuild on the phase's arrivals; legacy paths removed.
* **`models/policy.py`.** New `NoLearningPolicy`.
* **`benchmarks.py`.** `job_level_space_check` and `enlarge_for_job_models`.
* **`theory_eval.py`.**
  * The static policy always uses the dual rule. The no-learning run is added, and so is `select_f` with (a), (b).
  * `window_models` counts the volume inside the window.
* **`diagnostics.py`.** Rewritten for the model-check notebook.
* **Tests.**
  * New: `test_bid.py` (from `test_phi_and_size_u0.py`), `test_data_pipeline.py` (from `test_v2_pipeline.py`),
    `test_payments.py` (bins cover every bid; IR for every admitted job, also below v_min and above v_max).
  * Updated: the reward, carbon map, identity, unseen types, policy and snapshot tests.

## Deletions

* **Configs:** `configs/v3_mean_demand.yaml`, `v4_300k.yaml`, `v5_size_bid.yaml`, `v5_theory.yaml`,
  `v6_size_bid.yaml`, `v6_theory.yaml`.
* **Scripts:** `scripts/week32/*` (9 files), `scripts/calibrate_u0.py`, `scripts/ablation_runner.py`.
* **Code:** `dlent_rt/legacy/` (pooled MLE, φ-space grids).
* **Notebooks:** `01_phase0_diagnostics`, `05_theory_models`, `05_ablations_small_cluster` (and their sources).
  * **My mistake:** `01_phase0_diagnostics.ipynb` had uncommitted local changes (+1642 / −1 lines vs HEAD). I took
    them for editor metadata and deleted them. They are lost: there is no checkpoint, and the VS Code history copies
    are older. The committed version is in the tag `pre-cleanup-week33`.
* **Tests:** `test_size_bid.py`, `test_v3_limits.py`, `test_ablations.py`, `test_known_distribution.py` and its
  golden `tests/data/golden_known_false.npz`.
* **Moved:** `reports/week32/` → `archive/reports/week32/`.
* **Kept:** `scripts/data/*`, `scripts/pareto_run.py`, `scripts/regime_config.py` (the Pareto run imports it),
  `dlent_rt/pareto.py`, and the benchmark ladder.

## Re-frozen arrays (`tests/data/snapshot_synthetic.npz`)

| array | change |
|---|---|
| `p0_p`, `p0_r_tilde`, `p0_v` | changed (new bids, V = v, new grid and quota) |
| `p0_s_C` | 11,813.81 → 9.90 $/kg |
| `p0_lambda_star` | 0.68583 → 0.00051 |
| `p0_util_opt` | (0.115, 0.303) → (0.362, 0.410) |
| `p0_tau` | unchanged (0, 0) |
| `on_phases` | 5 × 11 → 4 × 13 (σ and ρ columns added) |
| `on_reasons` | (0, 1, 0, 3, 1) → (1, 0, 0, 2, 1) |
| `on_decisions` | (867, 0, 1228, 474, 0) → (1073, 0, 837, 593, 76); column 2 is now IR violations, was IR rejects |
| `on_R`, `on_admit` | 2,569 → 2,579 arrivals (different quota) |
| `on_R_total` | 36,778.40 → 16.77 |
| `on_payment` | added |
| `p0_u0_sigma_rmax` | added: (0.03375, 0.28337, 0.03325) |

## Notebook 03

`git diff --stat notebooks/03_main_algorithm_analysis.ipynb`: 1 insertion, 1 deletion. The only change is the
kernel display name (`"Python 3"` → `"dlent_venv (3.11.3)"`). Sources and outputs are unchanged. Not touched and
not committed.

## Open issues

1. **DLENT-RT earns less than the simple policies.** It earns $720.55, below the static policy ($1,316.92) and the
   no-learning model ($901.94); fluid OPT is $2,843.68.
   * Thinning removes 55,631 of the 160,343 jobs the oracle wants (35%).
   * Phases 4 and 14 have a negative reward.
   * This needs its own study; I did not change the model.
2. **No f is inside the 20–30% band.** f = 0.5 keeps the cluster ≥ 95% full 44.6% of the time. Condition (a)
   holds only through RAM from f = 0.35 up; τ_CPU = 0 there.
3. **The quota and the cap leave 0.55% of the trace's CPU-hours.** The quota removes 4.6% of the jobs, but they
   hold 99.0% of the CPU-hours. The model sees only small jobs. 39.7% of kept jobs are raised to the 1-minute floor.
4. **The Tests 2–3 unit is a choice.** r_max = $0.0055 is the 0.8-quantile of |r̃|; another quantile changes how
   often Tests 2–3 fire. In the paper, γ also bounds v and d, which stay in machine-ticks (d_max = 86,400). Only
   the rewards were made consistent.
5. **The data ξ is about 0.2 by construction** (Q = 0.2 C). Theorem 1 holds only with the forced constants.
6. **The practical model can break IR.** On the synthetic run, the live oracle with no thinning gives 15 jobs with
   p > v. The threshold comes from the frozen τ ranking, but the live oracle can admit a job in a bin below it. A
   payment rule for the live oracle is still to decide.
7. **φ has a heavy left tail.** The realised profit is noisy: the no-learning model's profit ($619) is far below the
   static policy's ($833), with the same LP fractions.
8. **One reward scale.** Realised rewards are scored with Phase 0's σ for every model. The decisions use each
   phase's refit σ (0.296–0.303), so the gap is small.
9. **Notebooks 03 and 04 were not rebuilt.** Notebook 03's source already used config keys removed before week 33
   (`capacity.fraction`), so it does not run on the current code.
10. **There is no FCFS class.** "FCFS" in D6 has no code. The practical-only benchmarks in the code are Greedy-Pos
    (G2), the type greedy (G1) and the job MILP.
11. **Run time.** The full theoretical run takes 2 hours. The new progress lines show its status during the run.
12. The notebook prints commit `53c19cf`, the parent of this week's commit, because it ran before the commit.

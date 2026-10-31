# Practical model as ablations of the theoretical model (code only)

Branch `practical-ablations`, from cleanup-week30 at f0234d6. Nothing was simulated. Only the new and changed tests
were run: 23 passed in about 35 s (`test_practical_ablations.py`, `test_bid.py`, `test_config_defaults.py`), on hand-made jobs and a 1,500-job synthetic trace.

**The default config is still the theoretical model, with the same behavior.** The synthetic regression snapshot
(Phase 0 and a full online run) is bit-identical between this branch and f0234d6: 15 of 15 arrays.

## 1. The elements (one config switch each)

| Element | Switch | Default (theoretical) |
|---|---|---|
| thinning off | `ablation.thinning_off: true` | `false` |
| statistics by end reason | `stats.update: by_end_reason` (new) | `replace` |
| σ update | `bid.sigma_update: cumulative` or `fixed` (new; replaces `bid.refit_sigma`) | `phase` |
| live oracle | `oracle.kind: live` | `frozen` |

1. **Thinning off.** This switch already existed. It works with the current mechanism:
   * p_U = 1 and no "thinning" rejections;
   * admit = bin ≥ k*, and bid ≥ reserve, and the job fits;
   * payment = max(L_k*, r), with IR = 0 and no upper-set violations.
2. **Live oracle.** This switch already existed. Each bin of the job's column is scored with the job's own power ×
   d̄_(q,h) × the price and carbon intensity at arrival. The tests check this against a hand computation. Tripling
   the bid or multiplying the true duration by 5 leaves the scores unchanged.
   * On hand-made jobs (25 bids from far below to far above the grid, true durations × 0.1 and × 10), k* and the
     reserve do not move with the bid.
   * A job is admitted exactly when bin(b) ≥ k* and b ≥ reserve, and it pays one critical bid with 0 < p ≤ b.
   * The true duration changes nothing.
3. **σ update** (`bid.sigma_update`). Phase 0 always fits σ on H's markups.
   * `phase`: refit on the previous phase's markups if it has ≥ n_min arrivals. This is the same as before.
   * `cumulative`: refit on the pooled markups of H and of every phase so far.
   * `fixed`: Phase 0's σ, never updated.

   `bid.refit_sigma` is removed. Its old `false` meant σ = 0.3 from the start; `fixed` means Phase 0's estimate.
4. **Statistics by end reason** (`stats.update`). With `by_end_reason`:
   * after a timeout or Test 1, ReInit merges the old and new statistics, weighted by their number of jobs:
     * p = (old arrivals + new arrivals) / (old ticks + new ticks), per type;
     * per column, d̄, energy, power, C_elec, K and the request are weighted by the admitted jobs, and the cost base by
       the arrivals;
   * after Test 2 or Test 3, the new statistics replace the old ones, as in `replace`.

   A column or type with no new data keeps its old values and counts (carry-forward). Phase 0's counts are H's.
   `PhaseState` now carries the counts behind its statistics.
5. **Benchmarks** (fluid OPT, the per-phase hindsight LP, the window LPs, the H-vs-stream decomposition) always use
   the theoretical rules, through `reinit.hindsight_cfg`. So they are the same for every ablation.

## 2. Files and lines

| File | Lines | What |
|---|---|---|
| `dlent_rt/config.py` | 5–9, 81, 201, 253, 261 | docstring; `BidCfg.sigma_update`; new `StatsCfg` (`stats.update`); validation |
| `configs/default.yaml` | header, `bid.sigma_update`, `stats.update` | the same switches, set to the theoretical values |
| `dlent_rt/bid.py` | 69, 79 | `log_markups`; `bid_model` fits σ whenever bids are given (no `refit_sigma`) |
| `dlent_rt/phase0.py` | 250, 406 | `PhaseState.n_type`, `T_obs`, `n_col_arr`, `n_col_obs` (the counts behind the statistics), set from H |
| `dlent_rt/models/core/reinit.py` | 38, 41, 48, 82, 94, 122, 136 | `MERGE_REASONS`, `hindsight_cfg`, `_merge`; `build_window_state(..., reason, pooled_log_markups)`; the σ rule; the merge of column statistics and of p; `ReinitDiag.stats_merged` |
| `dlent_rt/models/theory.py` | 95, 143–147 | the markup pool (H, then each phase); ReInit gets the end reason and the pool; phase log `reinit_stats_merged` |
| `dlent_rt/benchmarks.py`, `dlent_rt/theory_eval.py` | 238; 105, 438 | the hindsight states use `hindsight_cfg` |
| `scripts/run_ablations.py` | new | the ablation runner (section 4) |
| `README.md` | D2 (σ), D5 (practical model) | the practical model = ablations; the switches table. Also fixed: the no-learning model keeps Phase 0's σ, not 0.3 |
| `tests/test_bid.py` | 60 | `refit_sigma` replaced by the three modes |
| `tests/test_practical_ablations.py` | new | section 3 |

## 3. Tests (`tests/test_practical_ablations.py`, 15 tests)

| Test | Checks |
|---|---|
| `test_thinning_off_keeps_the_mechanism` | p_U = 1, no "thinning" rejections, admit = k_hat and reserve and fits, payment = max(L_k*, r) ≤ b, IR = 0, no upper-set violations |
| `test_live_oracle_uses_power_dbar_price_never_the_bid` | scores = r~ − (P·d̄·price − C̄elec) − s_C(P·d̄·CI − K̄) − v·τ; unchanged with bid × 3 and duration × 5 |
| `test_live_oracle_truthful_and_ir_on_hand_made_jobs` | 5 jobs × 25 bids × 3 durations: k* and the reserve fixed, admitted iff bin ≥ k* and b ≥ r, one critical payment, 0 < p ≤ b; a live run has IR = 0 and k_hat = (bin ≥ own k*) |
| `test_sigma_update_on_a_markup_stream` | three windows: `fixed` never changes; `phase` = fit on the last window; `cumulative` = fit on H + every window so far; `cumulative` without the pool raises |
| `test_sigma_update_in_the_online_run` (× 3 modes) | every phase's next σ, in a real run |
| `test_stats_merge_after_timeout_equals_the_pooled_phases` | after a timeout and after Test 1: p = the pooled phases' p; column statistics = the pooled ones (columns seen in phase 1); counts add up; k* is an upper set |
| `test_stats_after_test2_or_3_are_the_last_phase_only` (× 2) | identical to `replace` |
| `test_stats_merge_keeps_empty_columns_and_the_upper_set` | empty columns keep their old values and counts exactly; merged values lie between old and new; the margin increases in the bin |
| `test_stats_by_end_reason_in_the_online_run` | merged exactly after timeout / Test 1; IR = 0, no upper-set violations |
| `test_hindsight_benchmarks_ignore_the_switches` | the benchmarks' states are the same under any switch |
| `test_ablation_runner_plan_and_configs`, `test_ablation_report_tables_on_fake_runs` | the runs, their configs, the tables and the deltas |

## 4. The ablation runner (not run)

* **Runs.**
  * `--single`: each element alone, 5 runs (thinning off, stats by end reason, σ cumulative, σ fixed, live oracle).
  * `--cumulative`: the elements added one by one, 4 runs: thinning off; + stats by end reason; + σ cumulative;
    + live oracle. The table shows the delta of each step.
* **Execution.** Same seed, f and data as the theoretical run. Each run is its own process. Each result is cached in
  `reports/week33/cache/ablations/<run>.pkl`, and skipped if cached with the same config.
* **Table per run:**
  * reward and its split, social welfare, payments;
  * the Myerson check (Σ(φd − C_elec) vs payments − C_elec);
  * admitted jobs and the rejection reasons;
  * phases and end reasons;
  * IR, upper set, and the truthfulness spot check (200 + 200 jobs × 4 bids; with the live oracle, k* is recomputed
    per job);
  * % of fluid OPT, delta vs the theoretical run, delta vs no-learning.

  It goes to `reports/week33/ablations_results.md`.

**Run it only after this branch is merged** into cleanup-week30 and the running work there is finished. The runner
needs f = 0.35, `fit_on: "H"`, `data/` (not in this worktree) and the caches `theory_run.pkl` and `benchmarks_f.pkl`
at the same f.

```
dlent_venv/bin/python scripts/run_ablations.py --single --cumulative --dry-run      # the plan, nothing run
dlent_venv/bin/python scripts/run_ablations.py --single --workers 2                 # 5 runs
dlent_venv/bin/python scripts/run_ablations.py --cumulative --workers 2             # 4 runs
dlent_venv/bin/python scripts/run_ablations.py --single --cumulative --report-only  # the tables again
```

**Expected runtime per run:** about the theoretical run's runtime at the same f.
* At f = 0.5 that was 5.4 h.
* At f = 0.35 it is not known yet: the theory run now in progress has spent more than 6 h in its phase 2.
* Use its final runtime as the estimate.
* The live oracle scores every bin per arrival, so it may be a little slower.
* Thinning off admits about 1.5 times as many jobs.

With `--workers 2`, 9 runs take about 4.5 × that runtime. Do not use more than 2 workers on this 16 GB machine: the
last theory and MILP runs hit swap.

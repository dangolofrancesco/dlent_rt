# Week 32, step 3 B: repository cleanup plan (waiting for your OK)

Tag before any change: `pre-cleanup-week32`. Every old result can be rebuilt from it. The data files are not in
git, so the old data files stay on disk in `data/archive/` (see below).

## 1. Summary

* The repository has 116 files in the code folders and 349 under `reports/` (59 of them tracked).
* What the current path needs: the v2 fetch and build, `dlent_rt/`, the ablation runner, and the tests.
* Proposed: keep the current path, move the week-29 / week-30 one-off scripts, notebooks, configs and reports to
  `archive/`, delete only the code and tests of the removed paths and the 300k file (tables below).
* **You must decide 6 things** (section 5). Two of them change numbers: the default config and the regression
  snapshots.

## 2. The rules for the proposed action

1. **keep**: used by the current path (v2 pipeline, models, ablation runner, current tests), or needed next
   (calibrate_u0).
2. **archive**: produced a past report, not used now. It moves to `archive/` with its folder kept
   (e.g. `archive/scripts/capacity_regime.py`).
3. **delete**: only code that tests or serves a removed code path, and the 300k file.
4. "Used" comes from an import scan that starts at `scripts/data/fetch_v2.py`, `scripts/data/build_v2.py`,
   `scripts/ablation_runner.py` and `tests/*.py`.
5. "Last report" is the last report that **produced or used** the file. A file that is only listed in the week-32
   audit does not count.

## 3. Inventory

### scripts/

| file | what it is | used now | last report | action |
|---|---|---|---|---|
| scripts/ablation_runner.py | ablation runner (C0-C3, R1/R2) | yes | week30/ablations_cumulative.md | keep |
| scripts/regime_config.py | committed default + a named regime (imported by the runner) | yes | week30/ablations_cumulative.md | keep (point it at regime_v2 later) |
| scripts/calibrate_u0.py | bisection for bid.base_utility_u0 | no | (u0 = 23.926, before week 29) | keep (needed for recalibration) |
| scripts/data/fetch_v2.py | v2 fetch, BigQuery | yes | week32/data_v2.md | keep |
| scripts/data/build_v2.py | v2 job file builder | yes | week32/data_v2.md | keep |
| scripts/data/fetch_data.py | old fetch (biased sample) | no | week30 (30k file) | archive |
| scripts/data/data_generator.py | old job file builder | no | week30 (30k file) | archive |
| scripts/capacity_regime.py | week-30 cluster regimes | no | week30/cluster_regime.md | archive |
| scripts/capacity_from_rho.py | week-30 load-factor rule | no | week30/small_cluster.md | archive |
| scripts/small_cluster_nofilter.py | week-30 small cluster without the filter | no | week30/small_cluster_nofilter.md | archive |
| scripts/combine_regime_md.py | merges nb03 / nb04 markdown | no | week30/nb03_nb04_small_cluster.md | archive |
| scripts/pareto_run.py | Pareto front over the lambdas | no | week30/pareto_lambda.md | archive (keep if the thesis needs new fronts) |
| scripts/smoke_phase0.py | Phase 0 smoke test, all capacity modes | no | week30/code_audit.md | delete (the tests cover it) |
| scripts/week32/bq_checks.py | week-32 audit BigQuery checks | no | week32/data_audit.md | archive |
| scripts/week32/data_audit.py | week-32 audit numbers | no | week32/data_audit.md | archive |
| scripts/week32/data_v2.py | step-2 checks | no | week32/data_v2.md | archive |
| scripts/week32/fingerprint_dataset.py | old-path bit-identity check | no | week32/data_v2.md | archive (meaningless once the old path is gone) |
| scripts/week32/limits.py | step-3 limits | no | week32/limits.md | keep until the limits are decided, then archive |

### notebooks/ and notebooks/src/

| file | what it is | used now | last report | action |
|---|---|---|---|---|
| notebooks/01_phase0_diagnostics.ipynb + src/01_phase0_diagnostics.py | Phase 0 diagnostics | no | week30 (default nb01) | keep, rebuild on regime_v2 |
| notebooks/03_main_algorithm_analysis.ipynb + src/03_main_algorithm_analysis.py | main algorithm analysis | no | week30/nb03_nb04_small_cluster.md | keep, rebuild on regime_v2 |
| notebooks/03_main_algorithm_analysis_small_cluster.ipynb | nb03 on the small cluster | no | week30/nb03_nb04_small_cluster.md | archive |
| notebooks/04_pareto_lambda.ipynb + src/04_pareto_lambda.py | Pareto over lambda | no | week30/nb03_nb04_small_cluster.md | keep, rebuild on regime_v2 |
| notebooks/04_pareto_lambda_small_cluster.ipynb | nb04 on the small cluster | no | week30/nb03_nb04_small_cluster.md | archive |
| notebooks/05_ablations_small_cluster.ipynb + src/05_ablations_small_cluster.py | ablations | no | week30/ablations_cumulative.md | keep, rebuild on regime_v2 |
| notebooks/05_known_distribution_check.ipynb + src/05_known_distribution_check.py | known-distribution check | no | (week 29) | archive |
| notebooks/src/build.py | builds the notebooks from src | no | week30 | keep |
| notebooks/legacy/04_phi_diagnostic.py, phi_diag.py | old φ diagnostic | no | week30/code_audit.md | delete |
| notebooks/src/arrival_compression_check.py (untracked) | arrival compression | no | week30/arrival_compression.md | delete with the code path |
| notebooks/src/intensity_experiment.py (untracked) | arrival compression k = 1..4 | no | week30/intensity_experiment.md | delete with the code path |
| notebooks/src/binding_capacity_check.py | binding-capacity run | no | week30/small_cluster_nofilter.md | archive |
| notebooks/src/bucketed_ladder.py (untracked) | bucketed MILP ladder | no | week30/bucketed_opt.md | archive |
| notebooks/src/capacity_grid_sweep.py | capacity × grid sweep | no | week30/capacity_grid_sweep.md | archive |
| notebooks/src/capacity_scan.py, capacity_scan_online.py | capacity scan | no | week30/capacity_scan.md, cluster_regime.md | archive |
| notebooks/src/greedy_check.py | greedy G1 / G2 | no | week30/baselines_after_fix.md | archive |
| notebooks/src/grid_sweep_clean.py (untracked) | type-grid sweep | no | week30/grid_sweep_clean.md | archive |
| notebooks/src/lambda_sensitivity_check.py | lambda sensitivity | no | week30/lambda_sensitivity.md | archive |
| notebooks/src/phase0_regime_check.py | Phase 0 regime check | no | week30/cluster_regime.md | archive |
| notebooks/src/reward_identity_check.py | reward identity | no | week30/reward_identity_report.md | archive |
| notebooks/src/snapping_check.py | snapping size | no | week30/snapping_check.md | delete with the snapping path |
| notebooks/src/table08_jobopt_pct.py | adds a column to table 08 | no | week30/capacity_grid_sweep.md | archive |
| notebooks/src/test_margins_check.py | test margins | no | week30/cluster_regime.md | archive |

### dlent_rt/

| file | what it is | used now | last report | action |
|---|---|---|---|---|
| \_\_init\_\_.py, config.py, data.py, preprocess.py, bid.py, energy.py, grid.py, reward.py, lp_oracle.py, phase0.py, benchmarks.py, pareto.py | the package | yes | week32 | keep (dead branches removed, section 4) |
| models/runner.py, theory.py, policy.py, core/*.py (error_params, monitor, occupancy, reinit, stream, window_lp) | the online model | yes | week30 / week32 | keep |
| models/\_\_init\_\_.py, core/\_\_init\_\_.py | package files | yes | – | keep |
| legacy/\_\_init\_\_.py, legacy/pooled_mle.py | known_distribution = false ablation | yes (tests) | week30/code_audit.md | keep (decision 4) |

### configs/

| file | what it is | used now | last report | action |
|---|---|---|---|---|
| configs/default.yaml | default config (from_amax, old data) | yes (tests) | week30 | keep, **replaced by the v2 content** (decision 1) |
| configs/regime_v2.yaml | v2 path | yes | week32/limits.md | merge into default.yaml, then delete |
| configs/regime_small_cluster.yaml | week-30 small cluster (absolute) | no | week30/ablations_cumulative.md | archive |
| configs/binding_capacity.yaml | week-30 binding capacity | no | week30/small_cluster_nofilter.md | archive |

### tests/

| file | what it is | used now | last report | action |
|---|---|---|---|---|
| test_capacity_from_amax.py, data/golden_capacity_modes.npz | old capacity modes | yes | week30 | delete with the modes |
| test_capacity_drop.py | drop_jobs_exceeding_capacity | yes | week30 | delete with the flag |
| test_arrival_compression.py | arrival_time_scale | yes | week30/arrival_compression.md | delete with the option |
| data/snapshot_real.npz | real-trace snapshot (old path) | no (manual) | week30/cluster_regime.md | delete; save a v2 snapshot when a run is approved |
| data/snapshot_synthetic.npz | synthetic snapshot | yes | week30 | **re-save** (default config changes) |
| test_config_defaults.py, test_v2_pipeline.py, regression_snapshot.py, test_regression_snapshot.py, synthetic.py, brute.py | config, v2, snapshot | yes | week32 | keep; update to the new default |
| test_ablations, test_benchmarks, test_bucketed_opt, test_event_queue, test_greedy, test_known_distribution, test_monitor_exact, test_online_quantile_linear, test_online_smoke, test_pareto, test_policy, test_quantile_normalization, test_reward, test_reward_identity, test_window_lp (+ data/golden_known_false.npz) | model tests | yes | week30 | keep |

### data/ (not in git)

| file | what it is | used now | action |
|---|---|---|---|
| batch_v2_30k.csv, raw/v2/* | v2 data | yes | keep |
| grid_data_may_2019.csv | prices / carbon (UTC) | yes | keep (extend, open decision 5 of limits.md) |
| batch_may2019_30k.csv, raw/raw_events_30k.csv, raw/raw_usage_30k.csv | old data (biased sample) | no | move to data/archive/ (needed to rebuild old results from the tag) |
| batch_may2019_300k.csv | old 300k file (wrong query, 2–23× requests) | no | **delete** |

### reports/

| folder | content | action |
|---|---|---|
| reports/week29/ | 2 folders of figures and tables (untracked) | archive/reports/week29/ |
| reports/week30/ | 29 .md files (reports and tables) + figures / tables / CSV (most untracked) | archive/reports/week30/ |
| reports/week32/ | data_audit, data_v2, limits, cleanup_plan (+ figs, bq) | keep |
| reports/week30/tables/capacity_scan_table.md (untracked, not mine) | – | ask you |

## 4. Code paths no longer needed after v2

| code path | where it lives | depends on it | proposed |
|---|---|---|---|
| capacity modes fraction_of_volume, fraction_of_peak, fraction_of_concurrency_quantile, from_amax | `data.py: _size_capacity`; `config.py` (CapacityCfg fraction, concurrency_quantile, xi_target, validate) | default.yaml (from_amax), test_capacity_from_amax, smoke_phase0, week-30 scripts | remove |
| capacity mode absolute | `_size_capacity`; CapacityCfg.absolute | regime_small_cluster.yaml | **keep** (one line; needed if you choose "cluster ≥ R_max" or a fixed real cluster) |
| drop_jobs_exceeding_capacity | `data.py: load_dataset`; `Dataset.capacity_drop` | test_capacity_drop, benchmarks docstring, week-30 scripts | remove |
| arrival_time_scale compression | `data.py: _compress_arrival_steps`, load_dataset; `Dataset.arrival_compression`; TimeCfg | test_arrival_compression, 2 untracked notebook scripts | remove |
| snapping of requests (snap_requests = true) | `data.py: _build_job_arrays` | default.yaml (default true), snapshots | remove the option (A = real request always; profiles only for the type id) |
| outlier filters drop / clip, the d_max quantile rule | `preprocess.py: handle_outliers`; OutliersCfg quantiles; `data.py: _compute_d_max_steps` (d_max_quantile) | default.yaml, snapping_check | remove; keep policy "none" / "r_max" and d_max_hours |
| span duration, "trace" time base, profile type volume | `data.py: _read_csv`, `phase0.py: type_request` | old data file | remove once the defaults change (they exist only for the old data) |
| data_generator.py, fetch_data.py | scripts/data | old data file | archive |
| the 300k file | data/ | nothing | delete |

## 5. What I need from you

1. **Default config.** Removing these paths breaks the committed `default.yaml` (from_amax, outlier filter, old
   file). My recommendation: make `default.yaml` = the v2 settings and delete `regime_v2.yaml`. This changes the
   default numbers. The old numbers stay rebuildable from the tag.
2. **Regression snapshot.** `tests/data/snapshot_synthetic.npz` must be re-saved after decision 1. This is an
   intended numeric change.
3. **Your uncommitted edits** to `configs/default.yaml` (capacity fraction_of_concurrency_quantile, 0.30 / 0.70) and
   `.gitignore` (ignore notebooks/legacy/ and notebooks/src/). My recommendation: drop the default.yaml edit, since
   decision 1 replaces the file. Commit the `.gitignore` edit only if you really want `notebooks/src/` out of git,
   but `build.py` and the notebook sources live there, so I suggest keeping them tracked.
4. **legacy/pooled_mle.py** (the known_distribution = false ablation). My recommendation: keep it until the thesis
   figures are final.
5. **capacity mode absolute.** My recommendation: keep it (see section 4).
6. **reports/week30/tables/capacity_scan_table.md** is untracked and not mine. Keep it, archive it, or delete it?

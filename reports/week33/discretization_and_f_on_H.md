# Week 33: discretization cost = MILP_true vs MILP_bin, f rule on H only

Implementation only. No simulation, MILP, f scan or notebook was run. Only the new or changed tests were run: 11
passed in 7 s. They use hand-made jobs and a 1,500-job synthetic trace.

## 1. What changed

### Part 1: discretization cost with two job-level MILPs

The grid now changes only the value of a job (L_k instead of its bid), so the cost compares two job-level MILPs that
differ only in that.

| File | Lines | What |
|---|---|---|
| `dlent_rt/benchmarks.py` | 147 | `bin_values(v, grid)`: L_k of each bid, the lower edge of its bin (0 for bin 1). |
| `dlent_rt/benchmarks.py` | 152 | `discretization_milps(S, state, lam, c, time_limit)`: MILP_true and MILP_bin (see below). |
| `dlent_rt/benchmarks.py` | 318, 338 | `ladder`: the two bucketed-MILP rows are replaced by "job-level MILP on bin values, at true rewards". "job-level MILP" is now MILP_true. The result has a `discretization` entry. |
| `dlent_rt/theory_eval.py` | 464 | `window_discretization(cfg, days, time_limit)`: the two MILPs on the first `days` of the stream. |
| `scripts/discretization_milp.py` | new | CLI `--days --time-limit --seed`; writes `reports/week33/cache/discretization_milp.pkl`. |
| `dlent_rt/diagnostics.py` | 546, 561 | `checks_discretization`, `discretization_table` (they replace `checks_week`). |

How the two MILPs work:

* **Shared setup.** Both use the same jobs, the same capacity checks (at every arrival, true durations), the same
  carbon map and s_C, and λ = 1.
  * MILP_true: R_j = b_j·d_j + (φ_j(b_j)·d_j − C_elec,j) − s_C·K_j.
  * MILP_bin: R_j = L_k·d_j + (φ_j(L_k)·d_j − C_elec,j) − s_C·K_j, with φ_j(L) = c_j·φ_m(L / c_j).
* **Candidates.** The candidates are the jobs with R_true > 0, the same set for both MILPs.
  * A candidate with R_bin ≤ 0 is left out of MILP_bin. A job with a negative reward is never in an optimal
    schedule, so OPT_bin does not change.
  * This also keeps the −1e12 φ floor (bids in bin 1, where L = 0) out of HiGHS.
* **Solver.** Each MILP and its LP relaxation runs on its own HiGHS instance with its own time limit.
* **Reported.**
  * OPT_true and OPT_bin, each with its best value, bound, gap and status.
  * (a) Value loss = OPT_true − OPT_bin, in $ and %, with a range from the bounds.
  * (b) Decision loss = OPT_true − (MILP_bin's schedule valued at the true rewards).
  * (c) The grid's worst case, 1 − 1/ρ.
  * Feasibility of both schedules on the true durations.

**Removed** (dead code):
* the bucketed MILP from `evaluate_f`, and its arguments `max_milp_nonzeros` and `milp_time_limit`;
* `milp_nonzeros`;
* `window_models` (the window fluid and the bucketed MILP);
* `checks_week`;
* the unused scan plots `fig_tau_vs_f`, `fig_reward_vs_f`, `model_reward` and `MODELS`.

`Benchmarks.job_opt_bucketed` stays, because the source of notebook 03 still uses it. Its docstring now says so.

**Kept as upper bounds:** fluid OPT and fluid on feasible types (`evaluate_f`, `ladder`).

### Part 2: the f rule on H only

| File | Lines | What |
|---|---|---|
| `dlent_rt/data.py` | 316 | `NoStream`: a stream that raises on any access. |
| `dlent_rt/data.py` | 323, 348 | `load_history(cfg)`: the Dataset of H alone (details below). `load_dataset` shares its first steps. |
| `dlent_rt/data.py` | 84 | **The sort by arrival time is now stable** (details below). |
| `dlent_rt/models/core/stream.py` | 63 | `build_history_stream`: H's jobs as a stream, with Phase 0's own bids. |
| `dlent_rt/theory_eval.py` | 205, 230, 236, 246 | `f_rule_point(ds, cfg)`, `evaluate_f_H(cfg, f)`, `tau_positive` (Phase 0 LP only), `select_f` (on `full_time` of the H replay). |
| `scripts/select_f.py` | new | CLI `--f ... --seed`; writes `reports/week33/cache/f_scan_H.pkl` and prints the chosen f. It does not change the config. |
| `dlent_rt/diagnostics.py` | 367, 385 | `scan_table` (the H rule) and `checks_cluster` (no fluid τ any more). Checklist: new D3 row "f rule on H only". The D6 row now points to `discretization_milps`. |
| `scripts/model_check_cache.py` | – | Steps `select_f` (→ `f_scan_H.pkl`), `bench` (stream models at the configured f → `benchmarks_f.pkl`), `disc` (→ `discretization_milp.pkl`) and `theory`. `scan` and `week` are gone. |

What `load_history` does:
* It uses the same quota fixed point, H, C = f·D̄ and Q as `load_dataset`.
* The hardware profiles are fit on H only.
* The stream is never built: `stream = NoStream()`.
* T is set to H's span and a_max to H's largest request. Only Phase 0's data-ξ and γ diagnostics read them.

The rule, at each f, is computed only on H:
* (a) τ > 0 (CPU or RAM) in the Phase 0 LP;
* (b) H's jobs are replayed through the static `ThresholdPolicy` of that f's Phase 0 state, with capacity C. The rule
  picks the f whose "% of H's time with the cluster ≥ 95% full" is closest to 20–30% (ties go to 25%).

Fluid OPT and every stream statistic are out of the rule. `evaluate_f` (stream models) stays only for sections 9 and
10 of the notebook, at the configured f.

### Two things you should know

1. **The stable sort changes results slightly.**
   * Before, `_read_csv` sorted with pandas' default sort, which is not stable.
   * On `data/batch_v2_300k.csv`, 68,697 rows share an arrival time with another row. The old sort put 17,984 rows in
     a different order from the file.
   * So H's order, and therefore which bid each H job gets, depended on rows after H. The empty-stream guard test
     caught this.
   * Now ties keep the file's order. Every number moves slightly: Phase 0, the f rule, the theory run.
   * So `theory_run.pkl` and `final_theory.md` are from the old order.
   * The synthetic traces of the tests also have ties (6 rows reorder), so `tests/test_regression_snapshot.py` will
     probably fail until its snapshot is saved again. I did not run it.
2. **`hardware.fit_on: "full"` in `configs/default.yaml`.**
   * The model's 32 hardware profiles are fit on H plus the stream. So the model's own Phase 0 is not strictly H-only.
   * The f rule does not use this: `load_history` always fits on H.
   * At the chosen f, the rule's Phase 0 can therefore differ slightly from the model's Phase 0.
   * Setting `fit_on: "H"` would make them the same. I did not change it: it is a model decision.

### Docs

* `README.md`:
  * D3 "f": chosen on H only, with the rule;
  * D4: a new paragraph "Discretization cost of the grid";
  * D6: the benchmarks.
* `dlent_rt/config.py`: comments on `CapacityCfg.f`, `GridCfg` and `HardwareCfg.fit_on`.
* `configs/default.yaml`: the f comment describes the rule on H, and says 0.5 is the old rule's pick and the H rule
  has not been run. f is still 0.5. The D4 comment mentions MILP_true vs MILP_bin.
* `notebooks/src/01_model_check.py`: section 6 reads `f_scan_H.pkl`; section 9 reads `discretization_milp.pkl`;
  sections 9 and 10 read the stream models from `benchmarks_f.pkl`. The `.ipynb` was not rebuilt.
* `scripts/final_theory_numbers.py`: kept so that `final_theory.md` can be reproduced from the old caches. It no
  longer uses `te.tau_positive`.

## 2. New tests

| Test | Checks |
|---|---|
| `tests/test_discretization.py::test_bin_reward_below_true_reward_and_opt_bin_below_opt_true` | 40 hand-made jobs: L_k ≤ b, R_bin ≤ R_true for every job, OPT_bin ≤ OPT_true, value and decision loss ≥ 0, worst case = 1 − 1/ρ, candidates = R_true > 0 |
| `…::test_one_job_per_bin_edge_gives_identical_milps` | b = L_k for bins 2–8: R_bin = R_true, the same schedule, losses = 0 |
| `…::test_both_schedules_feasible_on_true_durations` | both schedules fit at every moment; the capacity binds |
| `…::test_window_discretization_on_the_synthetic_trace` | the script's path, end to end, on 6 h of the synthetic stream (10 s limit) |
| `tests/test_f_rule_H.py::test_nan_or_empty_stream_gives_identical_outputs` | U0, σ, grid, D̄, C, Q, s_C, r_max, τ, r̃ and the f-rule point are identical with the stream's rows set to NaN, and with the stream removed |
| `…::test_f_rule_and_phase0_never_read_the_stream` | `load_history`'s stream raises on access; Phase 0 and the f rule on a Dataset whose stream raises give the same outputs as on the real one; with `fit_on: "H"` the H-only H equals the full Dataset's H |
| `…::test_select_f_rule` | (a) before (b), the distance to the band, the tie to 25%, the error when no f passes (a) |
| `tests/test_ladder.py` (changed) | new rows; `test_discretization_in_the_ladder` |

## 3. Commands to run (in this order)

Runtimes are estimates from earlier runs.

| Step | Command | Time |
|---|---|---|
| 1 | `dlent_venv/bin/python -m tests.regression_snapshot check synthetic`; if it fails only because of the new order, `... save synthetic` | a few minutes |
| 2 | `dlent_venv/bin/python -m pytest -m "not slow"` | about 4 min |
| 3 | `dlent_venv/bin/python scripts/select_f.py --seed 42` → `f_scan_H.pkl` | about 3–5 min (7 f) |
| 4 | If step 3 picks another f: set it in `configs/default.yaml`, with the comment | – |
| 5 | `dlent_venv/bin/python scripts/model_check_cache.py bench` → `benchmarks_f.pkl` | about 1–2 min |
| 6 | `dlent_venv/bin/python scripts/discretization_milp.py --days 7 --time-limit 3600 --seed 42` → `discretization_milp.pkl` | up to about 2–2.5 h (two MILPs of up to 1 h each, plus their LP relaxations) |
| 7 | `dlent_venv/bin/python scripts/model_check_cache.py theory` → `theory_run.pkl` (needed again: stable sort, maybe a new f) | about 5.5 h |
| 8 | `dlent_venv/bin/python notebooks/src/build.py 01_model_check --execute` | about 5–10 min with the caches |

Steps 6 and 7 can run at the same time in two terminals.

## 4. What the notebook will show

* **Section 6**: the scan table on H, one row per f:
  * (a) τ > 0 in the Phase 0 LP, and (b) the distance to the band;
  * % of H's time ≥ 95% full, τ, LP fill;
  * C, Q, H jobs and the jobs admitted in the replay;
  * U0, σ, s_C, r_max.

  Then the band plot on H, and a PASS / FAIL row "f = the rule's pick on H". That row fails if the config f is not the
  pick.
* **Section 9**: a PASS / FAIL table:
  * reward_bin ≤ reward_true;
  * OPT_bin ≤ OPT_true's bound;
  * MILP_bin's schedule at the true rewards ≤ OPT_true's bound;
  * both schedules feasible;
  * each MILP solved or used its own limit.

  Then the table with OPT_true and OPT_bin (best, bound, gap), (a) the value loss in $ and %, the bin schedule at the
  true rewards, (b) the decision loss in $ and %, and (c) 1 − 1/ρ.
* **Sections 9 and 10** keep the stream baselines (fluid OPT, fluid on feasible types, static, no-learning, window τ),
  now from `benchmarks_f.pkl`.
* **Checklist (section 11)**: the new D3 row "f rule on H only" and the D6 row "discretization cost = MILP_true vs
  MILP_bin".

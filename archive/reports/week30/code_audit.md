# Week 30, Stage B: code audit (read-only)

**Branch:** `cleanup-week30`, as of commit `9700a6d` (after the Stage-A fix).

**Method:** `coverage` is not installed in `dlent_venv`, and I did not install it during a read-only stage. Instead:
* **Static reference count:** every function and class in `dlent_rt/`, counted across `dlent_rt/`, `tests/`,
  `notebooks/src/` and the code cells of the executed notebooks.
* **Config keys:** each one grepped, including access through local aliases such as `o = cfg.outliers`.
* **Tests:** per-test durations from `pytest --durations`.

**How to answer:** every item has an ID. Reply with **KEEP / REMOVE / MOVE** per ID (MOVE = to `legacy/` or
`scripts/`). The proposal is in the last column. Nothing has been deleted.

**Closed decisions, not reopened:**
* additive dollar reward
* `quantile_linear` normalization with q = 0.90
* known bid distribution (σ from the config, never refit)
* s_C frozen in Phase 0

"Phase-0 risk" marks removals that could change a Phase-0 number. The bit-identical regression snapshot
(`tests/regression_snapshot.py`, synthetic and real trace) is the gate for every such commit.

## 1. Dead or unused code

| ID | What | Where | Evidence | Proposal |
|---|---|---|---|---|
| D1 | `_single_objective_max` | `phase0.py:460` | 0 references | REMOVE |
| D2 | `TypeSpace.type_id`, `type_id_from_hw_idx`, `hw_index`, `_hw_lookup` | `phase0.py:129-141` | Superseded by the vectorised `assign_type_ids`; referenced only in its docstring. (`decode` is used by the notebooks and stays.) | REMOVE |
| D3 | `dlent_rt/data_generator.py` (`DataGenerator`, 130 lines) | | Imported nowhere. It builds the batch CSV from the raw Google trace (data provenance). | MOVE to `scripts/data/` |
| D4 | `dlent_rt/fetch_data.py` (92 lines, BigQuery) | | Imported nowhere. It runs on import (side effects) and needs GCP credentials. | MOVE to `scripts/data/` |
| D5 | `calibrate_base_utility_u0` | `phase0.py:602` (~40 lines) | Not called by any code. It is the documented origin of `base_utility_u0 = 23.926`. | MOVE to `scripts/calibrate_u0.py` |
| D6 | `PhaseState.r_rev` (revenue-only reward) | `phase0.py:199,232`, `reinit.py:132,145` | Documented as "for the time-aware oracle", which is not implemented. Read only by `test_quantile_normalization` and nb03 (share of R<0 without carbon). | KEEP as a diagnostic, rename to `r_nocarbon` |
| D7 | `PhaseState.energy_bar`, `power_bar` | `phase0.py`, `reinit.py` | Computed and carried, never read by a decision (only by the skipped refactor test). | KEEP (diagnostic, cheap) |
| D8 | `PhaseState.norm_strategy`, `norm_degenerate`, `n_types_hw_degenerate` | `phase0.py` | Written, read only by `smoke_phase0.py` or nobody. `norm_degenerate` is always `(False, False, False)`. | REMOVE `norm_degenerate`; KEEP the others |
| D9 | `BidResult.bid_low`, and lognormal `bid_high = 3 v` "rough upper bound for normalization" | `bid.py:194-195` | `bid_low` is never read. Lognormal `bid_high` has no meaning; it is only used by the uniform payment branch. | REMOVE `bid_low`; `bid_high` goes with U1 |
| D10 | `bid.base_multiplier` | `bid.py:139`, `phase0.py:71`, YAML | Passed into `LognormalBidParams`, never used in a computation | REMOVE |
| D11 | Phase-0 ε_D formula | `phase0.py:280-281` | Duplicates `ErrorParams.eps_D` (`error_params.py:73`). The loop uses the latter, while `PhaseState.eps_D` and `gamma` / `xi` are logged. | REMOVE the duplicate; Phase 0 calls `ErrorParams` |
| D12 | `tests/test_phase0_refactor.py` | | Always skipped: it imports `dlent_rt._phase0_pre_week28`, which does not exist | REMOVE |
| D13 | `tests/smoke_phase0.py` (203-line CLI script, not collected by pytest) | | Default `--bid-strategy uniform`, chebyshev and normalization flags (U1, L3, L1) | MOVE to `scripts/` with lognormal-only flags, or REMOVE |
| D14 | `dlent_rt/__init__.py` docstring ("First slice: config + data + Phase 0") and exports (`compute_phi_tilde`, `fit_myerson`) | | Stale | Update the docstring; export `bid_model` and `phi_job` instead |
| D15 | `notebooks/src/phi_diag.py` + `04_phi_diagnostic.py` | | The pooled-fit diagnosis of week 28. The question is answered (known distribution) and the `.ipynb` no longer exists. | MOVE to `notebooks/legacy/` |
| D16 | `notebooks/src/reward_identity_check.py`, `phase0_regime_check.py` | | One-off week-30 diagnostics | KEEP (referenced by the reports) |

## 2. Legacy ablations kept alive only for old results

| ID | Path | Who uses it | Size | Risk of removing | Proposal |
|---|---|---|---|---|---|
| L1 | Normalization `per_job`, `common_currency`, plus the "utopian" and "moving_avg" values | `_normalize_objectives` `phase0.py:410-434`; the utopian branch in `scalarized_reward` `phase0.py:195-198`; `common_scale` / `s_norm` plumbing in `reinit.py` and `theory.py`; `test_quantile_normalization.py:108-109`; `smoke_phase0`; `normalization.carbon_tax_per_kg` | ~45 lines + plumbing | Low. **Note:** `validate()` accepts `"moving_avg"` and `"utopian"`, but `_normalize_objectives` raises for both, so these values are already broken. `s_norm` is 1 under quantile_linear. | REMOVE `per_job`, `common_currency`, `utopian` and `moving_avg`; drop `common_scale` / `s_norm` and `carbon_tax_per_kg`. Phase-0 risk: none for the default. |
| L2 | Pooled-MLE φ (`known_distribution: false`) | `fit_myerson`, `bid_model`, the `reinit.py:78` refit, `phase0.py:481` (`phi_rate` shortcut), `compute_phi_tilde` pooled branch, `theory.py:163` payment branch; tests `test_known_false_reproduces_previous_numbers` and `test_ir_reject_never_indexes_last_type`; nb04, nb05 (old run), `phi_diag` | ~40 lines in `src/` + 2 tests + 2 notebooks | Low for the default. nb05's before/after and the golden test (e) die with it. | MOVE: keep the code until the thesis figures that compare old and new are final, then REMOVE. Decide together with L4. |
| L3 | Chebyshev scalarization | `lp_oracle.scalarize_chebyshev` (`:20`), the `scalarize` dispatch, the `ScalarizationCfg` method and rho, the warning at `phase0.py:525`, the refusal at `theory.py:52`, `smoke_phase0` | ~30 lines | None (the online loop already refuses it) | REMOVE (`scalarize` becomes the linear sum) |
| L4 | φ-space grids (`grid.space` "phi" / "phi_shifted") and the IR-reject path | `grid.py:106-120,143-155` (`rejects_below_floor`); `assign_type_ids -> -1`; `theory.py:140-143,176` `ir_reject`; `test_ir_reject_never_indexes_last_type` | ~40 lines + 1 test | None for the default. Already impossible with the known distribution (validation). | REMOVE together with L2 |
| L5 | Uniform bid strategy | `synthesize_uniform` (`bid.py:158`), `UniformBidParams`, the uniform `compute_phi_tilde` branch, `phase0._synthesize_bids` branch, `calibrate_base_utility_u0(strategy=...)`, `BidCfg.strategy` default `"uniform"` in the **dataclass**, `smoke_phase0` default | ~60 lines | None for the default. It cannot run with `known_distribution: true` (validation). | REMOVE |
| L6 | Grid spacings `quantile`, `log_linear`, `linear` | `grid.py:17-67` | ~35 lines | None (default geometric) | KEEP `geometric`; REMOVE the rest (or KEEP as cheap ablations, your call) |
| L7 | Hardware methods `quantile_grid`, `geometric_grid`, `quantile_1d`; `log_space` | `preprocess.py:71-136` | ~65 lines | None (default kmeans) | KEEP (Phase-0 type space ablation) or REMOVE |
| L8 | Capacity modes `fraction_of_volume`, `fraction_of_peak`, `fraction_of_concurrency_quantile`, `absolute` | `data._size_capacity`, `test_capacity_from_amax` (b) | ~25 lines | None (default `from_amax`) | KEEP (the capacity regime is still open) |
| L9 | `outliers.policy: clip` | `preprocess.handle_outliers` | ~10 lines | None (default drop) | KEEP or REMOVE |
| L10 | Practical-model stubs: `model.kind: practical`, `oracle.kind: time_aware`, `admission.*`, `revision.*`, `phantom.*` | `config.py`, `runner.py:62` (`NotImplementedError`), `theory.py:55` | Config only | None. **None of these keys is read** (see C1). | REMOVE the keys until the practical model exists; KEEP `model.kind` / `oracle.kind` with validation to `theory` / `frozen` |
| L11 | Payment `phi_tilde` (φ at the bin floor) | `compute_phi_tilde`, `PhaseState.phi_tilde` (NaN per type under the known distribution), `theory.py` payment | | Open decision (payment definition) | KEEP until payments are decided |
| L12 | `time.step_semantics` (only `"real_tick"` validates) | `config.py`, `data.py`, `phase0.py:492` | Key + 1 branch | None | REMOVE the key; hard-code `real_tick` |

## 3. Duplication in the reward pipeline

Every place where a reward term is computed (file:line on this commit):

| Term | Type level | Job level |
|---|---|---|
| E (kWh), power | `phase0._compute_job_energy` / `_compute_job_power` (H) | `stream.build_stream` (same helpers) |
| C_elec = E·price | `phase0.py:323` (`_mean(energy_kwh * elec_price)`) | `stream.py:59` |
| K = E·CI | `phase0.py:327` | `stream.py:59` |
| SW = q·v | `phase0.py:188` (`priority_of_type * v_bar`) | `theory.py:158`; nb03 `job_terms`; nb04 `:181,491`; nb05 `:209`; `reward_identity_check.py:105`; `test_reward_identity.job_reward`; `test_online_quantile_linear.py:56` |
| Π = φ − C_elec | `phase0.py:189`; φ̄ from `phase0.py:319` / `reinit.py:90` | same list as SW |
| carbon $ = s_C·K | `phase0.py:450` (`_normalize_objectives`) | `theory.py:161` (`self.tax`) and every notebook copy |
| s_C | `carbon_scale_from_quantile` (`phase0.py:351`); frozen via `prev.carbon_price` (`reinit.py:132`) and `TheoryModel.tax` | |
| R | `scalarized_reward` -> `lp_oracle.scalarize` (`phase0.py:200`) | `theory.py:161` (scalar, `/ s_norm`) and 6 notebook/test copies |
| phi | `MyersonModel.phi_job` (one source since commit `e8944c8`) | |

**Proposal: `dlent_rt/reward.py`**, the one source of truth. It is vectorised throughout: no loops, only `bincount`
for type means.

```python
@dataclass(frozen=True)
class RewardTerms:          # arrays, one entry per job (or per type)
    customer_value: ...     # q * v                     ($)
    profit: ...             # phi - C_elec              ($)
    carbon_kg: ...          # E * CI                    (kgCO2)
    carbon_usd: ...         # s_C * carbon_kg           ($)
    R: ...                  # l1*cv + l2*profit - l3*carbon_usd

def job_terms(q, v, c, C_elec, K, model, s_C, lam) -> RewardTerms       # phi = model.phi_job(v, c)
def type_terms(type_ids, n_types, q_type, v, phi, C_elec, K, s_C, lam) -> RewardTerms
    # means of v, phi, C_elec, K per type, THEN combined (current operation order -> bit-identical r_tilde)
def carbon_price(type_terms_no_carbon, p, q) -> dict                     # moved carbon_scale_from_quantile
def job_reward(stream, state, cfg) -> np.ndarray                         # convenience: R for every stream job
```

**Users after the move:**
* `phase0._per_type_stats` and `scalarized_reward` call `type_terms`.
* `reinit.build_window_state` calls `type_terms`.
* `TheoryModel` precomputes `job_reward` once for the stream and reads `R[i]` on admission.
* The benchmarks (Stage D) and notebooks 01, 03, 05 call `job_terms` / `job_reward`.
* `test_reward_identity.job_reward` is deleted in favour of the module.

**Bit-identity caveats (the snapshot will check):**
* (i) r̃ must keep "means first, then combine". Computing it as the mean of R_job is equal only to about 1e-13.
* (ii) The online loop today computes R with Python-float scalar arithmetic. A vectorised precomputation uses the
  same IEEE operations per element, but scipy/numpy SIMD paths in `lognorm.cdf/pdf` could differ in the last ulp. If
  the snapshot shows a diff, `theory.py` keeps calling `job_terms` on the single element.

## 4. Config

**C1. Keys no code reads.** I propose removing them from both `config.py` and `default.yaml`:
* `run.log_level`
* `admission.a_buffer_fraction`
* `revision.mode`
* `phantom.revise_in_flight`
* `phase_transition.short_phase_tracking`
* `bid.base_multiplier` (D10)
* `time.step_semantics` (L12)
* `scalarization.*` (L3)
* `normalization.carbon_tax_per_kg` (L1)

**C2. Misleading defaults.** The `config.py` dataclass defaults differ from the YAML:

| key | dataclass default | YAML |
|---|---|---|
| `bid.strategy` | uniform | lognormal |
| `bid.sigma` | 1.5 | 0.3 |
| `bid.gamma1/gamma2` | 0.01 / 0.002 | 0.0332 / 0.00445 |
| `capacity.mode` | concurrency_quantile, fraction 0.60, q 0.90 | from_amax |
| `outliers.a_max_quantile` | 0.95 | 0.90 |

`default_config()` loads the YAML, so the dataclass defaults only apply when the YAML is missing. Proposal: make the
YAML the only place where values live, with the dataclass defaults equal to it, and a test that
`Config().validate() == default_config()` field by field.

**C3. Stale or wrong comments in `default.yaml`:**
* **`bid.gamma1/gamma2`:** the comment quotes "~$0.00996/core-hour, ~$0.00135/GB-hour" for values 0.0332 / 0.00445,
  i.e. 3.3× the quoted prices. One of the two is wrong; you decide which.
* **`bid.base_utility_u0`:** "NOT against this file's own strategy default ('uniform')". The file's strategy is
  lognormal; the "uniform" refers to the dataclass.
* **`constants.xi_override`:** "gives eps_D ~ 0.45 ... for T ~ 1e5". Now T = 1.43e6 and eps_D = 0.502.
* **`constants.gamma_override`:** "= 1.0: >= r_max of the common_currency-normalised reward (max |r~| <= 0.9)". Wrong
  since quantile_linear: the reward is in dollars and max |r̃| = 24,293 $ (phase0_regime_check). The ξ/γ values are
  your open decision; the comment should say "forced, not data-justified".
* **`outliers.xi_warn_threshold`:** "~ 1/log(T) for T ~ 25k". T is now 1.43e6, and the warning always fires with
  `from_amax`, ξ = 0.5.
* **`capacity.fraction`:** "load factor rho (static sim used 0.60)", but it is unused under `from_amax` (it should
  say so, like `concurrency_quantile`).
* **`hardware.method`:** lists `"kmeans" | "quantile_grid"`, while 4 methods validate.
* **`model` / `oracle`:** these describe the practical model and the time-aware oracle, which are not implemented.
* **`data.batch_csv/grid_csv`:** absolute paths to `/Users/francescodangolo/...`, which are machine-specific. The
  dataclass uses relative paths, and notebooks re-root them.

**C4. Proposed minimal config** (values unchanged, keys only):
* `run` {name, seed, output_dir}
* `data` {batch_csv, grid_csv (relative), n0, columns}
* `time` {tick_seconds, d_max_quantile, d_max_horizon_fraction, d_max_ratio_abort, duration_floor_hours}
* `hardware` {n_profiles, method, log_space, fit_on, hw_floor}
* `outliers` {policy, a_max_quantile, d_max_quantile, xi_warn_threshold}
* `capacity` {mode, xi_target, fraction, concurrency_quantile, absolute}
* `energy_model` {...}
* `bid` {gamma1, gamma2, sigma, base_utility_u0, base_utility_mode} (strategy and known_distribution only if L2/L5
  are kept)
* `normalization` {quantile}
* `objective` {lambda1..3}
* `grid` {k_star, phi_floor, priority_classes} (space and spacing only if L4/L6 are kept)
* `confidence` {delta, delta_power}
* `constants` {xi_override, gamma_override, r_max_quantile}
* `tests` {drop_dual_learning_terms, piecewise_tol}
* `test3` {s_grid, geometric_base}
* `phase_transition` {n_type_min, n_min}
* `logging` {per_step, per_phase, short_phase_log}
* `online` {bid_seed_offset, thinning_seed_offset}
* `lp` {presolve}

## 5. Terminology (labels, docstrings, notebook text; identifiers only where cheap)

| ID | Change | Where |
|---|---|---|
| T1 | "social welfare" / "SW" for q·v -> **"customer value"** (`cv` in short labels) | `phase0.py:188-199,352-457` (`raw_sat` / `V_sat` -> `raw_cv` / `V_cv`, cheap and internal; `carbon_scale_from_quantile` keys `Q_sw` -> `Q_cv`, which **renames a `norm_info` key used by tests and nb01**); `config.py:54`; nb01 (`SW0`, labels); nb03 `:217-231,369-400,487,512,569,948`; nb04 `:479-492`; `reward_identity_check.py` |
| T2 | "sign mismatch" -> **"decision errors due to discretization"** | nb03 `:590,661,679,724` (accuracy table, METRICS figure, verdict list); nb04 `:475,497,501`; `reward_identity_report.md` |
| T3 | Add the type-level expected reward **r̃ (r_tilde) column** to the job tables | nb03: the arrivals frame already has `r_tilde`. Add it to the displayed job tables (negative-reward table `:406-432`, `table04b_admitted_negative`, per-phase margin plots) and to nb01's job-level table (cell 21 has `r_t`; make it a visible column) |

## 6. Tests

| Test | Time | Data | Status / proposal |
|---|---|---|---|
| `test_known_distribution.py`: `test_same_phi_everywhere`, `test_sigma_constant_across_phases`, `test_threshold_price_logged` | minutes (full online run on the real trace) | **real trace** (the file now uses `default_config()`) | Marked `@pytest.mark.slow` (commit `9700a6d`). Proposal: keep these as `slow` real-trace tests, and **add synthetic twins** so the fast suite covers them. |
| `test_known_distribution.py::test_known_false_reproduces_previous_numbers` | minutes | real trace | **Fails**: the golden `tests/data/golden_known_false.npz` was frozen on `synthetic_config` (480 types) and the file now runs `default_config()` (20,480 types). Recommendation: go back to `synthetic_config` for this test, which then passes and is fast. It goes with L2 if the pooled path is removed. |
| `test_known_distribution.py::test_scaling`, `test_negative_share_matches_theory` | < 1 s | none | ok |
| `test_reward_identity.py` (new) | 12 s | synthetic + real slice (skips if the trace is missing) | ok |
| `test_monitor_exact.py` | 31 s (17 s for one case) | synthetic | ok; the slowest fast test |
| `test_online_smoke`, `test_online_quantile_linear` | 8 s each | synthetic | `test_ir_reject_never_indexes_last_type` goes with L2/L4 |
| `test_capacity_from_amax`, `test_quantile_normalization`, `test_window_lp`, `test_event_queue` | < 3 s | synthetic | the per_job / common_currency cases in `test_quantile_normalization:108-109` go with L1 |
| `test_phase0_refactor.py` | skipped | | REMOVE (D12) |
| `tests/regression_snapshot.py` (new) | 10 s synthetic / 6 min real | both | the Stage-C gate. Proposal: add `test_regression_snapshot_synthetic` to the fast suite once Stage C starts. |

Fast suite: `pytest -m "not slow"` gives 36 passed, 1 skipped, 4 deselected, in 68 s.

## 7. Notes found while auditing (code wins)

* **`test_online_quantile_linear.py:56`:** builds the expected R from the logged `a["phi_mech"]`, so it cannot detect a
  wrong φ. `test_reward_identity.test_logged_equals_rescored` now checks it against `phi_job` independently.
* **nb03's `.ipynb` vs `.py`:** they are now regenerated from one source (commit `33f188f`). Notebook 01 has **no
  `.py` source** (`.ipynb` only), so it can still diverge. Proposal: add `notebooks/src/01_phase0_diagnostics.py`.
* **`notebooks/04_phi_diagnostic.ipynb`:** no longer in the tree. Only its source remains (D15).

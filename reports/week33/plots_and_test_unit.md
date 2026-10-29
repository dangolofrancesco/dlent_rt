# Week 33: limit plots in notebook 01, and the Tests 2–3 reward unit

Nothing long was run. Not run: the online model, notebook 01. Run: Phase 0 on the real config, the two plot
functions once (to check that they draw), and `pytest -m "not slow"` (79 passed).

## 1. New section 1b in `notebooks/01_model_check.ipynb`: "What the job limits leave out"

* **Where it is.** Section 1b sits right after section 1, in the notebook and in its source
  `notebooks/src/01_model_check.py`. The cells only call two new functions in `dlent_rt/diagnostics.py`:
  `plot_quota` and `plot_cap`.
* **What it uses.** All 300,000 jobs of `data/batch_v2_300k.csv`. The duration is the effective duration
  (CPU-hours / CPU request) with the 1-minute floor and no cap. Q is read from the dataset: the fixed point at the
  configured f.
* **Cell A: "Quota: requests and work".**
  * (a) CPU vs RAM request on log-log axes, with the lines x = Q_CPU and y = Q_RAM.
  * (b) CPU-hours vs RAM-hours per job.
  * Kept jobs are solid. Jobs over the quota are faded. The points are rasterized.
  * A text box gives the jobs over the quota and their share of CPU-hours and RAM-hours. A table follows the
    figure.
* **Cell B: "24 h cap: durations".**
  * (a) A histogram of the duration, counted by jobs. The bins are log-spaced, with 24 h on a bin edge.
  * (b) The same bins, weighted by CPU-hours.
  * The 24 h line is drawn, and the bars above it are faded.
  * A text box gives the jobs over 24 h, the share of CPU-hours the cap cuts, and the jobs raised to 1 minute. A
    table follows the figure.
* **Notes.** Each figure has a short markdown cell before it and an empty "Reading" cell after it.
* **Not executed.** The new cells have no output yet.

Numbers from the one check run (f = 0.5):

| | jobs | CPU-hours | RAM-hours |
|---|---|---|---|
| over the quota (Q = 0.1252 / 0.0850 machines) | 13,806 (4.6%) | 99.0% | 99.0% |
| longer than 24 h | 901 (0.30%) | 80.6% of the CPU-hours are in these jobs; the cap cuts 65.9% | – |
| raised to 1 minute | 115,009 (38.3%) | 0.04% | – |

163 jobs with a zero request cannot be drawn on log axes; the text box says so.

## 2. Reward unit of Tests 2 and 3

* **Before:** r_max = the p-weighted 0.8-quantile of |r̃| over Phase 0's types.
* **Now (paper definition):** the rewards of admissible actions are bounded by γ, and γ is forced to 1. So
  r_max = the largest r̃ over Phase 0's populated types with r̃ > 0. Negative types are never admitted, so they do
  not count.
* **Code.** The new function `phase0.reward_unit` computes it. Phase 0 stores it in `PhaseState.r_max`. ReInit
  keeps it: the online model reads it from the Phase 0 state for the whole run.
* **Unchanged.** `constants.r_max_quantile` (0.8) now enters only the data γ, which is not used, because γ is
  forced.
* **Text updated** (not re-run): the code comments (`config.py`, `default.yaml`, `monitor.py`, `theory.py`), the
  README, and the notebook text in sections 8 and 10.
* **New test:** `tests/test_reward_unit.py`.
  * On a hand-made state, r_max is the largest positive r̃. A negative type and a type with no jobs are both
    ignored.
  * Phase 0 stores the unit, and the online model uses it.

**The new r_max on the real config** (Phase 0 only, f = 0.5):

| | value |
|---|---|
| new r_max = max positive r̃ | **$7.744** |
| old r_max = 0.8-quantile of \|r̃\| | $0.00552 |
| ratio | 1,403 |
| p-weighted median of the positive r̃ | $0.0016 |

**Check this before the next run.**
* The maximum comes from one type with a single job in H (0.006 arrivals per hour).
* Rewards in the tests are now about 1,400 times smaller in test units than before.
* So Tests 2 and 3 will fire much less often. In the week-33 run, 7 of the 16 phases ended by Test 2.

## 3. Re-frozen arrays (`tests/data/snapshot_synthetic.npz`, against commit 712bbe9)

| array | change |
|---|---|
| `p0_u0_sigma_rmax` | r_max 0.03325 → 0.57497 (U0 and σ unchanged) |
| everything else | unchanged |

**Phase ends do not change on the synthetic run.** It has 4 phases before and after: Test 1 ×1, timeout ×2,
horizon ×1. That run never ended a phase by Test 2 or 3, even with the old, smaller unit. So a larger unit cannot
change it.

## Before you run the new cells

* **Delete or move the old cached run first:** `reports/week33/cache/theory_run.pkl`. It holds the run made with the
  old unit, and section 10 loads it if it exists. I did not delete it.
* Sections 8 and 10 still show the old outputs (r_max = $0.0055) until you re-run them.

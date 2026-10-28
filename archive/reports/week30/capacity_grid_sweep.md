# Week 30: capacity × type-grid sweep

**Source:** notebook 03, Section 8 (`notebooks/src/03_main_algorithm_analysis.py`; the per-config runner is
`notebooks/src/capacity_grid_sweep.py`). Full generated tables: `capacity_grid_sweep_tables.md`; CSVs:
`tables/table08_capacity_grid_sweep*.csv`.

## Setup

* **Base:** `configs/default.yaml` with notebook 03's overrides (σ = 0.3, n0 = 10,000), seed 42, theory model.
  The six configurations are generated in the notebook by overriding this config; no yaml file was added.
* **Capacity:** `fraction_of_concurrency_quantile` with `concurrency_quantile` = 0.7 and fraction ∈ {0.3, 0.5, 1.0}.
  c_i = fraction × (time-weighted 0.7 quantile of the H + stream concurrent occupancy).
  * A mode literally named `fraction_of_peak` exists. It sizes c on the MAX concurrency and takes no quantile, so it
    does not match "quantile 0.7" and was not used.
* **`capacity.drop_jobs_exceeding_capacity` = true** (new flag, default false). It runs after the unchanged outlier
  policy (0.90 / 0.95) and the capacity sizing, and drops from H and the stream every job with A_ij > c_i on some
  resource. c is not re-sized, and T, the H span, d_max and the hardware catalogue are unchanged. The "would drop"
  numbers are computed with the flag off on the same c.
* **Grids:** default (k* = 128, 32 hardware profiles, 20,480 types) and reduced (k* = 64, 16 profiles, 5,120 types).
  Both keep the 5 priority classes.
* **Policies and benchmarks:** each run uses one stream, one R_job and one physical gate for DLENT-RT online,
  StaticPolicy (Phase 0), G1, G2, the fluid OPT and the job-level OPT.
  * The MILP has a 600 s time limit. All six solved to optimality (HiGHS status 7, gap ≤ 4.4e-5).
  * **G1 is the only G1 that exists** (`TypeGreedyPolicy`). It is the full-information variant: its running mean of
    R_job uses every arrival, including the realised R_job of rejected jobs. G1_strict is not implemented.
* **Decision errors:** the share of arrivals where sign(R_job) ≠ sign(r̃ of their type). It is computed three ways:
  * on Phase 0 types, over stream arrivals of types populated in H;
  * on G1, using the r̃ it held at the decision;
  * on DLENT-RT, using the phase state's r̃.

### Caveat: the two grids do not run on an identical cluster or reward

* **The grid changes A and therefore c.** Each job's request A is snapped to its hardware profile, so
  `hardware.n_profiles` changes A. That moves the occupancy series that sizes c, the drop filter, the physical gate,
  the job energy and hence R_job. At a given fraction, c differs by 0.2 to 0.7% on CPU and 2.4% on RAM between the
  grids.
* **The grid changes s_C.** R_job also depends on the grid through s_C, the Phase 0 quantiles: for example
  533.50 vs 495.63 $/kgCO2 at fraction 0.3.
* **How large the non-type effects are:** G2 uses no types, so its change between grids is a rough measure of these
  effects. G2 moves by +1.4k to +4.3k $ (1 to 3%).

## Summary table (all six runs)

| fraction / grid | c CPU (cores) | c RAM (GB) | stream jobs dropped | τ₀ | phases (timeout / T1 / T2 / T3 / horizon) | phases τ>0 | R DLENT-RT | R Static | R G1 | R G2 | R fluid | R job OPT | DLENT % fluid | Static % fluid | G1 % fluid | G2 % fluid | discr. loss | dec. err. P0 types | dec. err. G1 | H jobs / populated type |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.3 / default | 51.38 | 133.95 | 977 | (0, 0) | 6 (5/0/0/0/1) | 0 | 40,740 | 46,526 | 68,840 | 113,506 | 119,169 | 118,745 | 34.2% | 39.0% | 57.8% | 95.3% | −424 | 11.6% | 11.6% | 3.90 |
| 0.3 / reduced | 51.05 | 130.69 | 1,008 | (0, 0) | 8 (6/1/0/0/1) | 0 | 52,546 | 66,758 | 85,794 | 115,963 | 118,524 | 121,527 | 44.3% | 56.3% | 72.4% | 97.8% | 3,003 | 13.3% | 12.4% | 8.98 |
| 0.5 / default | 85.64 | 223.25 | 523 | (0, 0) | 6 (5/0/0/0/1) | 0 | 41,910 | 44,054 | 71,740 | 126,304 | 127,443 | 129,671 | 32.9% | 34.6% | 56.3% | 99.1% | 2,227 | 12.2% | 11.6% | 3.88 |
| 0.5 / reduced | 85.09 | 217.82 | 447 | (0, 0) | 9 (7/1/0/0/1) | 0 | 53,343 | 71,390 | 97,515 | 130,643 | 130,026 | 134,602 | 41.0% | 54.9% | 75.0% | 100.5% | 4,576 | 11.2% | 11.1% | 8.87 |
| 1.0 / default | 171.27 | 446.50 | 0 | (0, 0) | 11 (9/0/1/0/1) | 0 | 48,242 | 54,122 | 82,311 | 153,582 | 149,478 | 154,637 | 32.3% | 36.2% | 55.1% | 102.7% | 5,159 | 12.5% | 12.1% | 3.74 |
| 1.0 / reduced | 170.17 | 435.63 | 0 | (0, 0) | 12 (9/1/1/0/1) | 0 | 68,419 | 85,655 | 111,975 | 155,002 | 148,856 | 156,438 | 46.0% | 57.5% | 75.2% | 104.1% | 7,583 | 12.0% | 11.1% | 8.68 |

**R as % of the job OPT (MILP) of the same configuration** (added by `notebooks/src/table08_jobopt_pct.py` from the
columns above; also in `tables/table08_capacity_grid_sweep*.csv`):

| fraction / grid | DLENT-RT | StaticPolicy (P0) | G1 | G2 |
|---|---|---|---|---|
| 0.3 / default | 34.31% | 39.18% | 57.97% | 95.59% |
| 0.3 / reduced | 43.24% | 54.93% | 70.60% | 95.42% |
| 0.5 / default | 32.32% | 33.97% | 55.32% | 97.40% |
| 0.5 / reduced | 39.63% | 53.04% | 72.45% | 97.06% |
| 1.0 / default | 31.20% | 35.00% | 53.23% | 99.32% |
| 1.0 / reduced | 43.74% | 54.75% | 71.58% | 99.08% |

All rewards are in $ over the stream. Discretization loss = job OPT − fluid OPT. Regrets against the fluid OPT and
the job OPT for every policy, and the τ per phase, physical rejects, thinning rejects and G1's rejection counts, are in
`capacity_grid_sweep_tables.md`.

**Capacity filter, would drop (flag off, same c):**

| fraction / grid | H jobs | stream jobs | % of stream Σ R_job⁺ | % of P0 profitable load (CPU / RAM) |
|---|---|---|---|---|
| 0.3 / default | 1,423 | 977 | 3.85% | 29.6% / 22.7% |
| 0.3 / reduced | 1,491 | 1,008 | 4.94% | 33.4% / 29.0% |
| 0.5 / default | 731 | 523 | 0.85% | 15.7% / 9.0% |
| 0.5 / reduced | 622 | 447 | 0.15% | 8.4% / 8.3% |
| 1.0 / either | 0 | 0 | 0 | 0 |

**Phase 0 with the flag on:**
* LP usage equals the profitable load in every run, so the LP admits every profitable type and its capacity
  constraint is slack.
* Profitable load is 26 to 33% of c at fraction 0.3, 22 to 30% at 0.5, and 18 to 23% at 1.0.

## Findings

* **τ > 0:** no fraction gives τ₀ > 0 in Phase 0, and no phase of any run has τ > 0. With the large jobs dropped, the
  Phase 0 profitable load is at most 33% of c, even at fraction 0.3.
* **Tests:**
  * Test 1 fires once in each reduced-grid run and never in a default-grid run.
  * Test 2 fires once at fraction 1.0, in both grids.
  * Test 3 never fires.
  * Every other phase ends by timeout, plus one horizon phase per run.
* **Discretization loss can be negative.** At fraction 0.3 on the default grid, job OPT is below fluid OPT (−424 $).
  The fluid LP constrains average occupancy on type means, while the MILP constrains occupancy at every arrival tick.
  The fluid OPT is therefore not an upper bound on the job OPT, and the difference is not purely a discretization loss
  when the per-tick constraint binds.
* **Reduced vs default grid** (reduced minus default, at fractions 0.3 / 0.5 / 1.0):

  | quantity | 0.3 | 0.5 | 1.0 | direction |
  |---|---|---|---|---|
  | R DLENT-RT | +11,806 | +11,433 | +20,177 | consistent (up) |
  | R StaticPolicy | +20,232 | +27,337 | +31,533 | consistent (up) |
  | R G1 | +16,954 | +25,775 | +29,664 | consistent (up) |
  | R G2 (no types) | +2,458 | +4,339 | +1,420 | up, small |
  | R fluid OPT | −645 | +2,583 | −623 | mixed |
  | R job OPT | +2,782 | +4,931 | +1,801 | up, small |
  | discretization loss | +3,427 | +2,349 | +2,424 | consistent (up) |
  | decision errors, P0 types (pts) | +1.8 | −1.0 | −0.5 | mixed |
  | decision errors, G1 (pts) | +0.8 | −0.5 | −1.0 | mixed |
  | decision errors, DLENT-RT (pts) | −6.5 | −9.1 | −6.9 | consistent (down) |
  | H jobs per populated type | 3.90 → 8.98 | 3.88 → 8.87 | 3.74 → 8.68 | consistent (×2.3) |
  | G1 rejections of unseen types | 1,042 → 259 | 1,106 → 297 | 1,155 → 321 | consistent (down) |

* **How much one seed can support:**
  * All six runs use one seed and one stream. The three fractions share the stream and bids, so they are not
    independent replications.
  * The type-based policies (DLENT-RT, StaticPolicy, G1) gain 11k to 32k $ with the reduced grid. That is much larger
    than G2's 1.4k to 4.3k change, which measures the effects that do not come from the types (A snapping, c, s_C).
  * The rise in jobs per type and the drop in unseen types follow from the grid sizes.
  * The changes in the fluid OPT and in the Phase 0 / G1 decision-error rates are small, and their sign varies across
    fractions. They are within what one seed and the grid-dependent A and s_C can produce.
  * The discretization loss rises with the reduced grid at every fraction (+2.3k to +3.4k $). Part of this is the job
    OPT rising with the changed A and R_job, which G2 also shows. It is not separated here.
  * No conclusion beyond these directions is drawn; repeating over seeds is needed to bound the variation.

# Week 30: clean type-grid sweep (k* = 128, 64, 32, 16; 32 hardware profiles fixed)

**Source:** `notebooks/src/grid_sweep_clean.py` (not committed: `notebooks/src/` is in the working-copy .gitignore).
CSV and JSON: `tables/grid_sweep_clean.csv`, `tables/grid_sweep_clean.json`.

**Config:**
* The committed `configs/default.yaml`, bid seed 42. Capacity `fraction_of_concurrency_quantile` with q = 0.7 and
  f = 1.0, `drop_jobs_exceeding_capacity` ON (nothing dropped). `hardware.n_profiles` = 32 for all runs; only
  `grid.k_star` changes.
* Job OPT: MILP with a 600 s limit. All four solved to optimality (gap ≤ 3.5e-5).
* **Seeds:** DLENT-RT ran with thinning seeds 42, 43 and 44 (`online.thinning_seed_offset` 0, 1, 2; 44 is the default
  run). The thinning draws are DLENT-RT's only random numbers. The world, i.e. the bids from `run.seed` = 42, is the
  same for every run.
* **Deterministic policies:** StaticPolicy, G1, G2 and the MILP use no random numbers, so one run each.
* **Std:** sample standard deviation over 3 seeds (ddof = 1).

## Same world? Comparability across k*

| k* | c (CPU, RAM; normalised) | dropped | s_C $/kgCO2 | fluid OPT $ | job OPT $ | G2 $ |
|---|---|---|---|---|---|---|
| 128 | 2.6762, 1.7441 | 0 | 228.18 | 149,478.42 | 154,637.33 | 153,582.18 |
| 64 | same | 0 | 224.70 | 148,289.54 | 155,010.26 | 153,750.11 |
| 32 | same | 0 | 202.36 | 149,649.83 | 157,469.63 | 155,740.57 |
| 16 | same | 0 | 225.62 | 145,642.52 | 154,911.50 | 153,674.95 |

* **Identical across k*:** each job's snapped A, c and the job set (32 profiles fixed).
* **Not identical: s_C**, which comes from Phase 0 quantiles over the types and so depends on the grid. It spans
  202.4 to 228.2 (−11% at k* = 32). Through s_C, R_job changes too.
* **The resulting spread is small:** job OPT spans 1.8% (154.6k to 157.5k), G2 1.4% and fluid OPT 2.7%. These
  benchmarks are comparable across k* to within about 3%.

## Results

| k* | populated types | H jobs per populated type | R DLENT-RT mean ± std (seeds 42 / 43 / 44) | R StaticPolicy | R G1 | R G2 | % job OPT: DLENT (mean ± std) / Static / G1 / G2 | discretization loss (job OPT − fluid) | decision errors: P0 types / G1 / DLENT online (mean ± std) |
|---|---|---|---|---|---|---|---|---|---|
| 128 | 2,673 | 3.74 | 47,988 ± 2,145 (45,728 / 49,995 / 48,242) | 54,121.64 | 82,310.77 | 153,582.18 | 31.0 ± 1.4 / 35.0 / 53.2 / 99.3 | 5,158.91 | 12.48% / 12.08% / 22.49 ± 0.004% |
| 64 | 1,721 | 5.81 | 54,750 ± 2,916 (51,705 / 57,516 / 55,030) | 67,186.67 | 95,096.99 | 153,750.11 | 35.3 ± 1.9 / 43.3 / 61.3 / 99.2 | 6,720.71 | 12.81% / 11.77% / 19.00 ± 0.03% |
| 32 | 1,075 | 9.30 | 66,995 ± 3,804 (62,757 / 70,112 / 68,117) | 86,159.52 | 110,584.04 | 155,740.57 | 42.5 ± 2.4 / 54.7 / 70.2 / 98.9 | 7,819.80 | 12.46% / 11.65% / 15.84 ± 0.55% |
| 16 | 656 | 15.24 | 74,516 ± 5,056 (68,689 / 77,123 / 77,738) | 99,742.83 | 122,087.95 | 153,674.95 | 48.1 ± 3.3 / 64.4 / 78.8 / 99.2 | 9,268.98 | 12.82% / 11.52% / 14.66 ± 0.08% |

**DLENT-RT phases (seeds 42 / 43 / 44):**
* k* = 128: 11 / 11 / 11, each with 1 Test 2, 9 timeouts and the horizon.
* k* = 64: 7 / 6 / 6. Test 1 fired once with seed 42; the others are timeouts and the horizon.
* k* = 32: 11 / 11 / 8. Seeds 42 and 43 have 1 Test 2; seed 44 has 2 Test 1.
* k* = 16: 11 / 11 / 11. Seed 42 has 1 Test 1 and 1 Test 2; seeds 43 and 44 have 1 Test 2 each.
* No phase of any run has τ > 0.
* Which test fires depends on the thinning seed.

## Summary (numbers only)

* **The benchmarks are comparable across k*** (job OPT within 1.8%, G2 within 1.4%, fluid OPT within 2.7%). The one
  input that is not identical is s_C (−11% at k* = 32).
* **Coarser grids give more reward to the type-based policies**, monotonically from k* 128 to 16:
  * DLENT-RT: 48.0k to 74.5k (31.0% to 48.1% of job OPT);
  * StaticPolicy: 54.1k to 99.7k;
  * G1: 82.3k to 122.1k.

  G2 stays at about 99% of job OPT.
* **Within the seed spread or not:** the DLENT-RT std across thinning seeds is 2.1k to 5.1k, or 4% to 7% of its mean.
  * Adjacent steps (+6.8k, +12.2k, +7.5k) are 1.5 to 4 times the larger std of the two grids involved.
  * The 128 → 16 difference (+26.5k) is more than 5 times either std.
  * With only 3 seeds the std itself is a rough estimate, so the 32 → 16 step (+7.5k against a std of 5.1k) is not
    clearly outside the seed variation.
* **Discretization loss** rises with coarser grids: 5.2k, 6.7k, 7.8k, 9.3k.
* **Decision errors:**
  * on the Phase 0 types: flat, 12.5% to 12.8%;
  * for G1: 12.1% to 11.5%;
  * for DLENT-RT online: down from 22.5% to 14.7%, with a seed std of 0.55 points at most.
* **Jobs per populated type** rise from 3.7 to 15.2.
* **StaticPolicy and G1 are deterministic.** Their changes across k* are not seed noise, but they share the
  grid-dependent s_C.

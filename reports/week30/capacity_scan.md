# Week 30: capacity scan (Phase 0 and full-stream reference, no online run)

**Source:** `notebooks/src/capacity_scan.py`; CSV `tables/capacity_scan.csv`.

**Config:**
* The committed `configs/default.yaml` (git HEAD), seed 42, σ = 0.3, n0 = 10,000, default grid (k* 128, 32 profiles).
* Capacity `fraction_of_concurrency_quantile` with q = 0.7, `drop_jobs_exceeding_capacity` ON, fraction f as in the
  columns.
* The uncommitted working-copy edit of default.yaml was not used.

**Definitions:**
* **Removed:** jobs with A_ij > c_i on some resource. Their share of Σ R_job⁺ is computed with the filter OFF on the
  same c, under that Phase 0 state (its bid model and s_C).
* **Profitable load:** Σ_{r̃_j > 0} p_j v_ij / c_i after the filter.
  * H: the Phase 0 state.
  * Stream: the full-stream reference state, which is the fluid OPT's state.
* **τ:** τ₀ of Phase 0 (H), and τ* of the reference (stream).
* **Fluid LP objective:** λ*₀·T (H) and fluid OPT = λ_ref·T (stream), in $.
* **Units:** c in cores and GB (1.0 normalised = 64 cores, 256 GB).

| quantity | f = 0.02 | f = 0.03 | f = 0.05 | f = 0.075 | f = 0.1 | f = 0.15 | f = 0.2 | f = 0.3 |
|---|---|---|---|---|---|---|---|---|
| c CPU (cores) | 3.43 | 5.14 | 8.56 | 12.85 | 17.13 | 25.69 | 34.25 | 51.38 |
| c RAM (GB) | 8.93 | 13.39 | 22.32 | 33.49 | 44.65 | 66.97 | 89.30 | 133.95 |
| H removed | 4,544 | 3,974 | 3,891 | 3,192 | 3,192 | 2,158 | 1,423 | 1,423 |
| H removed % | 45.44 | 39.74 | 38.91 | 31.92 | 31.92 | 21.58 | 14.23 | 14.23 |
| stream removed | 6,323 | 3,907 | 3,538 | 2,650 | 2,650 | 1,680 | 977 | 977 |
| stream removed % | 41.31 | 25.52 | 23.11 | 17.31 | 17.31 | 10.97 | 6.38 | 6.38 |
| H R_job+ removed % | 54.53 | 47.61 | 42.46 | 27.90 | 27.90 | 9.82 | 5.66 | 5.66 |
| stream R_job+ removed % | 57.14 | 37.71 | 32.50 | 21.34 | 21.34 | 9.29 | 3.85 | 3.85 |
| H prof load CPU % | 29.63 | 27.64 | 20.45 | 27.91 | 20.93 | 34.30 | 39.29 | 26.20 |
| H prof load RAM % | 37.93 | 33.37 | 22.03 | 27.96 | 20.97 | 45.63 | 49.39 | 32.93 |
| stream prof load CPU % | 41.87 | 61.08 | 42.75 | 48.58 | 36.44 | 45.06 | 56.16 | 37.44 |
| stream prof load RAM % | 48.48 | 51.09 | 30.85 | 40.58 | 30.44 | 50.62 | 61.45 | 40.97 |
| tau_0 (H) CPU | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| tau_0 (H) RAM | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| tau* (stream) CPU | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| tau* (stream) RAM | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| fluid LP H: lambda*_0 * T $ | 21,763.81 | 29,476.91 | 31,659.01 | 41,991.20 | 41,991.20 | 59,048.46 | 76,539.03 | 76,539.03 |
| fluid LP stream: fluid OPT $ | 36,206.20 | 53,457.65 | 55,591.79 | 74,704.21 | 74,704.21 | 102,542.43 | 119,169.15 | 119,169.15 |
| s_C $/kgCO2 | 4,303.27 | 3,032.53 | 3,052.50 | 1,762.89 | 1,762.89 | 955.02 | 533.50 | 533.50 |

**Notes (numbers only):**
* τ₀ = 0 and τ* = 0 at every f.
* The filter removes the same jobs at f = 0.075 and 0.1, and again at f = 0.2 and 0.3. Every A is a snapped hardware
  profile (32 values), so the set of jobs with A > c changes only when c crosses a profile value.
* Within such a pair, the profitable load % scales as 1/c and the fluid objective is unchanged.

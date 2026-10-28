# Week 30: the simplest small cluster with τ > 0, WITHOUT the drop filter (30k trace)

**Script:** `scripts/small_cluster_nofilter.py`; `fast` took 11 s, `full` 11 min with 4 processes. Results:
`reports/week30/small_cluster_nofilter/full.csv`, plus the Test 1/2/3 margins per run in `margins_*.csv`.
**Named config:** `configs/regime_small_cluster.yaml`.

**Setup:**
* The committed `configs/default.yaml` (`git show HEAD:configs/default.yaml` into a temporary file); seed 42.
* `xi_override` = 0.001 and `gamma_override` = 1.0, unchanged.
* Outlier quantiles 0.90 / 0.95, unchanged.
* **`drop_jobs_exceeding_capacity` = false:** jobs with A > c stay in H and the stream, and the physical gate rejects
  them.
* Capacity `fraction_of_concurrency_quantile` with q = 0.7, f ∈ {0.3, 0.2, 0.15, 0.1, 0.07, 0.05}.
* No change to Phase 0, rewards, defaults or committed configs. **No Phase 0 change.**

**Conflict, resolved by the code: the sanity check is q = 0.6, not 0.7.** The earlier ρ = 0.6 binding check
(`binding_capacity_check.py --rho 0.6`) was `fraction_of_concurrency_quantile` with **q = 0.6**, fraction 0.142948, and
no filter. That config is reproduced exactly; f = 0.143 at q = 0.7 would be a different cluster.
* **It matches the earlier result exactly:** c = (0.31295, 0.20237); τ₀ = (0, 0.0490); τ* = (0, 0.0700); 8 phases
  (5 timeout, **2 Test 1**, 1 horizon); **τ > 0 in 5 of 8 phases**; 808 physical rejects; R = 38,067.05.

**Units:** c is in the trace's normalised units of A, 1.0 CPU = 64 cores and 1.0 RAM = 256 GB. The values in the table
are exactly `capacity.absolute.cpu` / `.ram` for `mode: absolute`.

## Table (seed 42)

**Cluster and spurious load.** "Oversize" means A > c on CPU or RAM. The spurious share is the share of Phase 0's
profitable load Σ_{r̃ > 0} p_j·v_ij that comes from oversize H jobs; a job's A is its type's snapped profile, so this is
exact.

| f | capacity.absolute cpu / ram (norm) | c (cores / GB) | a_max / c (CPU / RAM) | oversize stream | oversize H | spurious share of P0 profitable load (CPU / RAM) | ρ = P0 profitable load / c (CPU / RAM) | ρ without oversize |
|---|---|---|---|---|---|---|---|---|
| 0.3 | 0.802847 / 0.523242 | 51.4 / 133.9 | 2.46 / 2.44 | 977 (6.4%) | 1,423 (14.2%) | 29.6% / 22.7% | 0.62 / 0.64 | 0.43 / 0.50 |
| 0.2 | 0.535232 / 0.348828 | 34.3 / 89.3 | 3.69 / 3.66 | 977 (6.4%) | 1,423 (14.2%) | 29.6% / 22.7% | 0.92 / 0.97 | 0.65 / 0.75 |
| **0.15** | **0.401424 / 0.261621** | **25.7 / 67.0** | 4.91 / 4.88 | 1,680 (11.0%) | 2,158 (21.6%) | **43.9% / 37.4%** | **1.23 / 1.29** | **0.69 / 0.81** |
| 0.1 | 0.267616 / 0.174414 | 17.1 / 44.6 | 7.37 / 7.32 | 2,650 (17.3%) | 3,192 (31.9%) | 68.9% / 72.8% | 1.85 / 1.93 | 0.58 / 0.53 |
| 0.07 | 0.187331 / 0.122090 | 12.0 / 31.3 | 10.53 / 10.46 | 3,127 (20.4%) | 3,593 (35.9%) | 83.8% / 83.5% | 2.64 / 2.76 | 0.43 / 0.46 |
| 0.05 | 0.133808 / 0.087207 | 8.6 / 22.3 | 14.74 / 14.65 | 3,538 (23.1%) | 3,891 (38.9%) | 87.5% / 89.4% | 3.70 / 3.87 | 0.46 / 0.41 |
| sanity (q = 0.6) | 0.312952 / 0.202372 | 20.0 / 51.8 | 6.30 / 6.31 | 1,680 (11.0%) | 2,158 (21.6%) | 43.9% / 37.4% | 1.58 / 1.67 | 0.89 / 1.04 |

**Prices and DLENT-RT** (forced ξ and γ). The last Test 1 column is the maximum Test 1 ratio in the phases where Test 1
did NOT fire (1 = firing).

| f | τ₀ (Phase 0) | τ* (hindsight) | phases | ended by (timeout / T1 / T2 / T3 / horizon) | phases with τ > 0 | Test 1 fires | Test 1 max where not fired | thinning rejects | phys. rejects (% of arrivals) | R $ | % of job MILP | % of fluid OPT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.3 | (0, 0) | (0, 0) | 6 | 5 / 0 / 0 / 0 / 1 | 0 / 6 | no | 0.9995 | 3,730 | 242 (1.6%) | 43,080 | 31.0 | 28.8 |
| 0.2 | (0, 0) | (0, 0.0083) | 9 | 5 / 3 / 0 / 0 / 1 | 2 / 9 | yes (3) | 0.9996 | 3,751 | 417 (2.7%) | 39,054 | 30.3 | 26.2 |
| **0.15** | **(0, 0.0204)** | (0, 0.0425) | 9 | 6 / **1** / 1 / 0 / 1 | **6 / 9** | **yes (1)** | 0.9983 | 3,744 | 537 (3.5%) | 40,892 | 33.9 | 27.9 |
| 0.1 | (0, 0.0654) | (0.0115, 0.0713) | 8 | 5 / 2 / 0 / 0 / 1 | 7 / 8 | yes (2) | 0.9994 | 3,645 | 644 (4.2%) | 37,795 | 35.8 | 27.3 |
| 0.07 | (0, 0.1103) | (0.0455, 0.0943) | 7 | 5 / 1 / 0 / 0 / 1 | 7 / 7 | yes (1) | 0.9998 | 3,530 | 684 (4.5%) | 33,400 | 36.1 | 25.8 |
| 0.05 | (0, 0.1416) | (0.1258, 0.0751) | 8 | 5 / 2 / 0 / 0 / 1 | 8 / 8 | yes (2) | 0.9980 | 3,257 | 669 (4.4%) | 27,241 | 34.2 | 22.9 |
| sanity | (0, 0.0490) | (0, 0.0700) | 8 | 5 / 2 / 0 / 0 / 1 | 5 / 8 | yes (2) | 0.9989 | 3,753 | 808 (5.3%) | 38,067 | 33.9 | 26.8 |

τ per phase at f = 0.15 (CPU, RAM): (0, 0.0204); (0, 0); (0.0201, 0); (0, 0.0196); (0, 0); (0, 0); (0.0044, 0.0067);
(0.0941, 0); (0.05, 0). Every run's τ is in `full.csv`.

**Benchmarks.**
* The no-capacity reward Σ R_job⁺ = 155,579.35 $ in every row, because nothing is dropped.
* The "feasible ceiling" is Σ R_job⁺ over the stream jobs with A ≤ c: the most any policy could earn.
* Capacity cost = (no-capacity − job MILP) / no-capacity. The "of which over the ceiling" column is
  (ceiling − job MILP) / ceiling: the part caused by capacity on jobs that can fit.

| f | feasible ceiling $ (% of no-cap) | job MILP $ (gap, runtime) | capacity cost | of which over the ceiling | G2 / G1 / static / fluid, % of job MILP | G2 / G1 / static phys. rejects (% of arrivals) | job MILP − fluid OPT $ |
|---|---|---|---|---|---|---|---|
| 0.3 | 149,582 (96.1%) | 139,177 (9.8e-5, 3.4 s) | 10.5% | 7.0% | 93.4 / 53.1 / 36.1 / 107.4 | 6.4 / 3.4 / 2.4 | −10,302 |
| 0.2 | 149,582 (96.1%) | 128,992 (9.9e-5, 7.7 s) | 17.1% | 13.8% | 90.5 / 52.9 / 34.8 / 115.7 | 10.4 / 6.4 / 4.6 | −20,225 |
| **0.15** | **141,127 (90.7%)** | **120,742 (9.9e-5, 16.1 s)** | **22.4%** | **14.4%** | 89.1 / 53.4 / 37.0 / **121.2** | 14.5 / 8.4 / 4.4 | **−25,653** |
| 0.1 | 122,374 (78.7%) | 105,610 (9.4e-5, 6.7 s) | 32.1% | 13.7% | 86.3 / 54.2 / 38.8 / 131.3 | 20.4 / 12.8 / 5.1 | −33,025 |
| 0.07 | 114,123 (73.4%) | 92,647 (9.4e-5, 11.1 s) | 40.5% | 18.8% | 84.1 / 55.3 / 40.3 / 139.7 | 25.0 / 17.3 / 5.6 | −36,800 |
| 0.05 | 105,020 (67.5%) | 79,614 (9.7e-5, 11.2 s) | 48.8% | 24.2% | 83.0 / 54.9 / 42.8 / 149.7 | 31.0 / 23.2 / 7.6 | −39,546 |
| sanity | 141,127 (90.7%) | 112,147 (9.6e-5, 14.9 s) | 27.9% | 20.5% | 85.7 / 53.1 / 37.5 / 126.4 | 19.0 / 11.9 / 5.2 | −29,647 |

All job MILPs report "optimal" with a gap below 10⁻⁴ (HiGHS' default relative MIP gap).

## Selection (the rule applied, not chosen)

The rule: the **largest c** such that (1) τ₀ > 0 or τ* > 0, (2) τ > 0 in at least half of the online phases, and
(3) Test 1 fires at least once.

* **f = 0.3:** fails (1), with τ₀ = τ* = 0. It also fails (2) and (3).
* **f = 0.2:** (1) holds (τ* = (0, 0.0083)) and (3) holds (3 × Test 1), but **(2) fails: only 2 of 9 phases have
  τ > 0.**
* **f = 0.15: all three hold.** τ₀ = (0, 0.0204), τ > 0 in 6 of 9 phases, and Test 1 fires once. **Selected.**

**Selected cluster:** c = (0.4014237441083392, 0.2616208305922613) normalised, which is **25.69 cores and 66.97 GB**.

**Override block** (saved as `configs/regime_small_cluster.yaml`; checked to reproduce the f = 0.15 cluster exactly):

    capacity.mode=absolute  capacity.absolute.cpu=0.4014237441083392  capacity.absolute.ram=0.2616208305922613  capacity.drop_jobs_exceeding_capacity=false

**Side effects at f = 0.15:**
* 1,680 stream jobs (11.0%) and 2,158 H jobs (21.6%) are larger than c and can never run.
* They carry **43.9% (CPU) and 37.4% (RAM) of Phase 0's profitable load**, and 9.3% of the stream's Σ R_job⁺ (the
  feasible ceiling is 90.7%).
* **Without them, ρ is 0.69 / 0.81 < 1: the binding fluid LP (τ₀ > 0) exists only because of load that can never
  run.**
* **Physical rejects:**
  * DLENT-RT 537 (3.5% of arrivals);
  * G2 14.5% of arrivals, which includes every oversize job it wants;
  * G1 8.4%; static 4.4%.

## How to read the fluid OPT in this regime

* The fluid LP (and Phase 0's LP-RS) sees every type's average volume p_j·v_ij, including the oversize jobs, against a
  time-average capacity.
* It therefore "admits" load that can never pass the physical gate. At f = 0.15 the fluid OPT is 146,395 $:
  * **121% of the job MILP**;
  * and even **above the feasible ceiling** (141,127 $), the most any policy could earn.
* So in this regime the fluid OPT is not an achievable benchmark, and "job MILP − fluid OPT < 0" (−25.7k $) mixes two
  effects: the average-vs-per-tick relaxation (`cluster_regime.md` §3c) and the oversize jobs it counts.
* The **job MILP** and the **feasible ceiling** are the meaningful references here. The same holds for τ: its positive
  values price capacity partly against jobs that never fit.

## Anything odd

* **The binding comes from the spurious load.** ρ without oversize stays below 1 for every q = 0.7 cluster
  (0.43–0.81). τ₀ > 0 appears exactly when the oversize share of the profitable load reaches 37–44% (f ≤ 0.15).
* **The sanity cluster (q = 0.6) is the one exception: ρ_RAM without oversize = 1.04.** The filtered search of
  `small_cluster.md` never reached ρ ≥ 1 because dropping the large jobs also changes s_C: it rose from 228 to 533 $/kg
  at f = 0.2, which makes fewer types profitable. Without the filter, s_C stays at 228.18 in every row. Part of the
  filtered result is therefore an s_C effect, not only a volume effect.
* **Test 1 reaches 0.998–0.9998 of its threshold in the phases where it does not fire**, at every f. Firing is
  decided by fractions of a percent.
* **The number of oversize jobs moves in steps** (977 → 1,680 → 2,650 …), because A takes only the 32 snapped profile
  values.

## Checks

* The sanity run matches the earlier ρ = 0.6 run exactly.
* The no-capacity reward (155,579.35) equals the default's.
* Neither the default job OPT (155,397.61) nor the default DLENT-RT (48,391.61) was re-run: no default-cluster row is
  part of this scan.
* No disagreement with earlier reports.
* **Runtimes:** fast 11 s; full 11 min. The DLENT-RT runs took 293–356 s each, and the job MILPs 3–16 s, all optimal.

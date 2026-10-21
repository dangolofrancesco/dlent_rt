# Week 30: bucketed job-level MILP ("job_opt_bucketed")

**Purpose:** the benchmark between the job-level OPT and the fluid OPT asked for at the Oct 19 meeting. It is the
job-level MILP solved on BUCKETED jobs, and it separates:
* the **discretization cost**: job level with bucketed rewards vs without them;
* the **fluid-relaxation cost**: the bucketed MILP vs the fluid LP.

**Code:** `dlent_rt/benchmarks.py`:
* `occupancy_milp` — the shared per-tick-capacity MILP;
* `peak_occupancy`;
* `Benchmarks.bucket_types`;
* `Benchmarks.job_opt_bucketed(source, duration)`.

**Tests:** `tests/test_bucketed_opt.py`.

**Runner:** `notebooks/src/bucketed_ladder.py` (not committed: `notebooks/src/` is in the working-copy .gitignore);
output `tables/bucketed_ladder.json`.

**Phase 0 is not touched.** The new code only READS the Phase 0 state (grid, types, p, r̃, d̄) and the reference
state. Rewards, configs and defaults are unchanged.

## The benchmark

* **The MILP:** max Σ_j r̃_k(j)·x_j with x_j ∈ {0, 1}, where r̃_k(j) is the type reward of job j's bucket.
* **Constraints:** identical to the job-level MILP (`job_level_opt`): at the arrival tick of every candidate job, the
  occupancy of the chosen jobs is ≤ c. It uses the same stream, snapped A, drop filter and arrival times.
* **Two values are reported:**
  * **realised** = Σ_j R_job,j·x_j of the chosen set. This is comparable with every other row.
  * **believed** = Σ_j r̃_k(j)·x_j, the MILP's own objective.
* **Variant A** (`duration="true"`): the occupancy uses each job's true D.
* **Variant B** (`duration="type"`): the occupancy uses the type's d̄, rounded to whole ticks (≥ 1). Its set can
  exceed c under the true durations; the table reports its peak true occupancy / c.
* **Two bucket sources** (your choice, both run):
  * **phase0** — Phase 0's grid and r̃ from H, the estimate the static and online policies use.
  * **reference** — the state behind `fluid_opt()`: the grid refit on the stream's bids, with p, r̃ and d̄ from all
    stream jobs. Only this source isolates the two costs exactly. With phase0, "discretization" also contains the
    H-vs-stream estimation error and the change of grid.

## Decisions made

1. **Unseen types.** A stream job whose type has no H job in Phase 0 (p = 0) gets r̃ = 0. Phase 0 itself gives
   exactly 0 for such types, because all their means are 0 (checked in the tests). So these jobs are never chosen.
   * phase0: **2,762 stream jobs in 1,155 types** (default config) and 2,621 jobs in 1,042 types (f = 0.2).
   * reference: 0, since the reference is fit on the stream itself.
2. **Candidates** are the jobs with r̃ > 0; the others cannot raise the objective, as in `job_level_opt`, which uses
   R > 0. The capacity rows are the candidates' arrival ticks, which is enough because occupancy only rises at
   arrivals.
3. **Bucket rewards can be positive for jobs with R_job < 0**, and the MILP then picks them. The number of such chosen
   jobs is reported (`n_chosen_R_neg`): for example 594 in variant A (reference) on the default config.
4. **Decomposition.** It is given with both the realised and the believed bucketed value; each version telescopes
   exactly to job OPT − fluid OPT. The clean split is **reference + believed**: the same objective units and the same
   types as the fluid LP, with integrality and per-tick capacity as the only difference.
5. **Refactor of `job_level_opt`.** Its MILP was moved into `occupancy_milp` unchanged, and a `runtime_s` key was
   added. On the synthetic trace, the chosen set, R_ip, R_lp, R_nocap, gap and status are identical to before.
   `test_benchmarks.py` passes and the regression snapshot is bit-identical.
6. **Seeds.** DLENT-RT uses its default thinning seed (one run). The other rows use no random numbers.

## Ladder (seed 42)

Each row gives the realised reward [$], its % of the job-level MILP, and the gap to the row above. Every MILP has a
600 s limit; all of them solved to optimality, with status and gap shown.

**Committed configs/default.yaml** (from_amax; 15,308 stream jobs; no jobs dropped; s_C = 228.18):

| row | realised $ | % job MILP | gap to row above $ | believed $ | MILP status / gap / runtime | peak true occ / c (CPU, RAM) |
|---|---|---|---|---|---|---|
| job-level, no capacity (Σ R_job⁺) | 155,579.35 | 100.12% | – | – | – | – |
| job-level MILP | 155,397.61 | 100.00% | 181.74 | – | optimal / 0 / 0.1 s | – |
| bucketed A (phase0) | 54,216.40 | 34.89% | 101,181.21 | 64,862.73 | optimal / 0 / 0.1 s | 0.956, 0.850 |
| bucketed A (reference) | 149,299.86 | 96.08% | −95,083.46 | 149,318.03 | optimal / 0 / 0.1 s | 0.982, 0.990 |
| bucketed B (phase0) | 54,216.40 | 34.89% | 95,083.46 | 64,862.73 | optimal / 0 / 0.1 s | 0.956, 0.850 |
| bucketed B (reference) | 149,278.87 | 96.06% | −95,062.47 | 149,309.84 | optimal / 0 / 0.1 s | **1.039**, 0.977 |
| fluid OPT | 149,478.42 | 96.19% | −199.55 | – | – | – |
| G2 job greedy | 155,011.00 | 99.75% | −5,532.58 | – | – | – |
| G1 type greedy | 82,626.18 | 53.17% | 72,384.82 | – | – | – |
| StaticPolicy (Phase 0) | 54,216.40 | 34.89% | 28,409.78 | – | – | – |
| DLENT-RT | 48,391.61 | 31.14% | 5,824.79 | – | – | – |

**f = 0.2** (fraction_of_concurrency_quantile q = 0.7, drop_jobs_exceeding_capacity ON; 14,331 stream jobs, 977
dropped; s_C = 533.50):

| row | realised $ | % job MILP | gap to row above $ | believed $ | MILP status / gap / runtime | peak true occ / c |
|---|---|---|---|---|---|---|
| job-level, no capacity (Σ R_job⁺) | 123,757.69 | 110.20% | – | – | – | – |
| job-level MILP | 112,304.68 | 100.00% | 11,453.01 | – | optimal / 4.9e-5 / 2.4 s | – |
| bucketed A (phase0) | 46,528.00 | 41.43% | 65,776.68 | 52,019.21 | optimal / 1.5e-5 / 0.2 s | 0.999, 1.000 |
| bucketed A (reference) | 108,479.99 | 96.59% | −61,951.99 | 108,122.58 | optimal / 9.4e-5 / 2.4 s | 1.000, 0.999 |
| bucketed B (phase0) | 46,227.99 | 41.16% | 62,252.00 | 52,041.45 | optimal / 8.8e-5 / 0.2 s | **1.556, 2.018** |
| bucketed B (reference) | 108,294.57 | 96.43% | −62,066.58 | 108,148.97 | optimal / 8.3e-5 / 7.8 s | **1.250, 1.329** |
| fluid OPT | 119,169.15 | 106.11% | −10,874.58 | – | – | – |
| G2 job greedy | 102,752.91 | 91.49% | 16,416.24 | – | – | – |
| G1 type greedy | 65,408.71 | 58.24% | 37,344.21 | – | – | – |
| StaticPolicy (Phase 0) | 44,733.34 | 39.83% | 20,675.37 | – | – | – |
| DLENT-RT | 38,222.30 | 34.03% | 6,511.04 | – | – | – |

## Decomposition

Discretization cost = job OPT − bucketed OPT, and fluid cost = bucketed OPT − fluid OPT. Their sum is
job OPT − fluid OPT, which is +5,919.19 on default and −6,864.47 at f = 0.2. A negative fluid cost means the fluid OPT
is ABOVE the bucketed MILP.

| config | bucketed row | value used | discretization cost $ | fluid cost $ |
|---|---|---|---|---|
| default | A (reference) | **believed (clean)** | **6,079.58** | **−160.39** |
| default | A (reference) | realised | 6,097.76 | −178.56 |
| default | B (reference) | believed | 6,087.77 | −168.58 |
| default | A (phase0) | realised | 101,181.21 | −95,262.02 |
| default | A (phase0) | believed | 90,534.88 | −84,615.69 |
| f = 0.2 | A (reference) | **believed (clean)** | **4,182.09** | **−11,046.57** |
| f = 0.2 | A (reference) | realised | 3,824.69 | −10,689.16 |
| f = 0.2 | B (reference) | believed | 4,155.71 | −11,020.18 |
| f = 0.2 | A (phase0) | realised | 65,776.68 | −72,641.15 |
| f = 0.2 | A (phase0) | believed | 60,285.47 | −67,149.94 |

## Checks

* **Realised bucketed ≤ job-level MILP** holds for all 8 bucketed runs. The runner asserts it for variant A,
  allowing for the job MILP's own relative gap.
* **Variant A is feasible under the true durations** in every run (peak true occupancy / c ≤ 1).
* **Variant B is infeasible under the true durations** for:
  * default, reference: CPU 1.039;
  * f = 0.2, phase0: 1.556 / 2.018;
  * f = 0.2, reference: 1.250 / 1.329.

  Its realised reward is reported anyway, but it is not an achievable value.
* **On the default config, bucketed A (phase0) equals StaticPolicy (Phase 0) to every digit** (54,216.397902…). The
  capacity does not bind there, so the MILP takes every job with r̃ > 0 that fits. StaticPolicy admits the same jobs
  (10,226 admitted plus 1 physical reject = the 10,227 candidates).
* **Unit test** on a two-job instance where the bucketed and job-level solutions differ: the job level takes R = 3,
  the bucketed MILP takes the job its type values more (realised 1). Plus invariants on the synthetic trace:
  * A is feasible and ≤ job OPT for both sources;
  * unseen types have r̃ = 0;
  * the reference source has no unseen jobs;
  * B runs.

## Reading (numbers only)

* **With the reference buckets:**
  * the discretization cost is 6.1k $ (default) and 4.2k $ (f = 0.2);
  * the fluid cost is −0.16k $ (default: fluid OPT slightly above the bucketed MILP) and −11.0k $ (f = 0.2: the fluid
    LP's average-occupancy constraint allows 11.0k $ more than per-tick capacity does).

  So the negative job OPT − fluid OPT at f = 0.2 comes from the fluid relaxation, not from the buckets.
* **With the Phase 0 buckets** the bucketed MILP realises only 34.9% (default) and 41.4% (f = 0.2) of the job MILP. The
  difference from the reference buckets (95.1k and 62.0k $ realised) is the H-vs-stream estimation, including the
  unseen types and the grid change, not the bucketing itself.
* **Fast suite:** 61 passed; the synthetic snapshot is bit-identical. The new synthetic bucketed tests add about
  140 s to the fast suite.

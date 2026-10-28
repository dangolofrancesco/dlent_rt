# Week 30: intensity experiment (arrival-time compression k = 1…4)

**Source:** `notebooks/src/intensity_experiment.py` (not committed: `notebooks/src/` is in the working-copy
.gitignore). CSV and JSON: `tables/intensity_phase0.csv`, `tables/intensity_phase0_resized_c.csv`,
`tables/intensity_online.csv`, `tables/intensity_online.json`.

**Config:**
* The committed `configs/default.yaml`, seed 42 (bids). Capacity `fraction_of_concurrency_quantile` with f = 0.2 and
  q = 0.7, `drop_jobs_exceeding_capacity` ON, default grid (k* 128, 32 profiles).
* `time.arrival_time_scale` = k (see `arrival_compression.md`).
* **c is sized on the original ticks**: one cluster for every k (34.25 cores, 89.30 GB), and the same 1,423 H and 977
  stream jobs dropped. |H| = 8,577 and |stream| = 14,331 at every k.
* Same stream, R_job and physical gate for every policy. Job OPT: MILP with a 600 s limit.
* k = 1 reproduces the earlier f = 0.2 run exactly (R DLENT-RT 38,222.30, job OPT 112,304.68).

## Phase 0 and the full-stream reference (c on the original ticks; the setting run online)

| k | H profitable load CPU / RAM (% of c) | stream profitable load CPU / RAM | P0 LP usage CPU / RAM | τ₀ (CPU, RAM) | τ* stream (CPU, RAM) |
|---|---|---|---|---|---|
| 1 | 39.3 / 49.4 | 56.2 / 61.5 | 39.3 / 49.4 | (0, 0) | (0, 0) |
| 2 | 78.6 / 98.8 | 112.3 / 122.9 | 78.6 / 98.8 | (0, 0) | (0, 0.0357) |
| 3 | 117.9 / 148.2 | 168.5 / 184.4 | 80.4 / 100.0 | (0, **0.0741**) | (0.0041, 0.0786) |
| 4 | 157.2 / 197.6 | 224.6 / 245.8 | 87.7 / 100.0 | (0, **0.1000**) | (0.0656, 0.0760) |

The profitable load scales with k: H's span shrinks by k, so every p_j grows by k. **Alternative, not run online:** if
c is RE-SIZED on the compressed ticks, c grows with k (×2.02, ×2.94 and ×3.79 on CPU at k = 2, 3, 4). The profitable
load then stays between 49% and 101% of c, and τ₀ = (0, 0.00064) only at k = 4 (`tables/intensity_phase0_resized_c.csv`).

## Online (c on the original ticks)

| k | phases | ended by (timeout / T1 / T2 / T3 / horizon) | phases τ>0 | τ per phase (CPU, RAM) |
|---|---|---|---|---|
| 1 | 6 | 5 / 0 / 0 / 0 / 1 | 0 | all (0, 0) |
| 2 | 7 | 3 / **3** / 0 / 0 / 1 | 0 | all (0, 0) |
| 3 | 7 | 2 / **4** / 0 / 0 / 1 | **5** | (0, .0741); (0, .0242); (0, 0); (0, .0423); (0, 0); (.0800, 0); (.0285, .00291) |
| 4 | 8 | 4 / **3** / 0 / 0 / 1 | **8** | (0, .1000); (0, .0358); (0, .0437); (0, .0908); (0, .0599); (.143, 0); (.0663, .0395); (.294, 0) |

| k | job OPT $ (MILP status, gap) | fluid OPT $ | R DLENT-RT | R Static | R G1 | R G2 | % job OPT: DLENT / Static / G1 / G2 | physical rejects: DLENT / Static / G1 / G2 | DLENT thinning rejects |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 112,304.68 (optimal, 4.9e-5) | 119,169.15 | 38,222.30 | 44,733.34 | 65,408.71 | 102,752.91 | 34.0 / 39.8 / 58.2 / 91.5 | 189 / 257 / 445 / 996 | 3,413 |
| 2 | 104,482.16 (time limit, 1.5e-3) | 118,075.51 | 38,111.30 | 43,310.03 | 60,051.38 | 91,453.93 | 36.5 / 41.5 / 57.5 / 87.5 | 311 / 439 / 894 / 1,636 | 3,417 |
| 3 | 97,581.86 (time limit, 2.5e-3) | 113,174.11 | 38,797.37 | 42,678.39 | 55,377.60 | 80,869.20 | 39.8 / 43.7 / 56.7 / 82.9 | 433 / 340 / 1,467 / 2,358 | 3,321 |
| 4 | 92,341.37 (time limit, 4.5e-3) | 107,890.95 | 36,487.10 | 41,041.94 | 50,582.35 | 73,034.14 | 39.5 / 44.4 / 54.8 / 79.1 | 411 / 343 / 1,851 / 2,813 | 3,169 |

* For k ≥ 2 the MILP stopped at the 600 s limit. The "job OPT" is then the best feasible solution found, and the
  optimum is above it by at most the reported relative gap (0.15% to 0.45%). The "% of job OPT" is relative to that
  incumbent, so it is overstated by at most the same gap.
* Only DLENT-RT uses random numbers (the thinning draws). StaticPolicy, G1, G2 and the MILP are deterministic, and one
  seed was run.

## Summary (numbers only)

* **First τ > 0:** τ* of the stream at k = 2 (RAM 0.036). τ₀ in Phase 0, and τ > 0 in online phases, at **k = 3**
  (τ₀ RAM 0.074; 5 of 7 phases). At k = 4, all 8 phases have τ > 0, with CPU prices appearing in phases 6–8.
* **Tests:** Test 1 fires from k = 2 on (3, 4 and 3 times at k = 2, 3, 4). Test 2 and Test 3 never fire, and Test 1
  never fires at k = 1.
* **Rewards:**
  * As k rises, the job OPT falls from 112.3k to 92.3k $, because the same jobs compete for the same cluster.
  * G2 falls from 91.5% to 79.1% of it, and G1 from 58.2% to 54.8%.
  * DLENT-RT rises from 34.0% to 39.8% (k = 3) and 39.5% (k = 4); StaticPolicy rises from 39.8% to 44.4%.
* **Ranking:** G2 > G1 > StaticPolicy > DLENT-RT at every k. The DLENT-RT changes in $ across k (36.5k to 38.8k) come
  from one thinning seed. The size of the seed variation at this capacity has not been measured: the 3-seed variation
  of item 3 (`grid_sweep_clean.md`) is at f = 1.0, so it does not apply here, and no claim is made about DLENT-RT's
  trend with k.

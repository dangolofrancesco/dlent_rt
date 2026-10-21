# Week 30: Pareto frontier over the objective weights λ (offline)

**Code:**
* `dlent_rt/pareto.py` (new functions; the existing per-type `scalarize_chebyshev` is untouched);
* `scripts/pareto_run.py` (runs the grid, writes the cache `reports/week30/pareto/*.csv`);
* `tests/test_pareto.py` (fast).

**Notebook:** `notebooks/04_pareto_lambda.ipynb`, which only reads the cache. Its source
`notebooks/src/04_pareto_lambda.py` is on disk but git-ignored by the working-copy `.gitignore`. Figures are in
`reports/week30/pareto/figs/`.

**Instance:** the same as notebook 03:
* the committed `configs/default.yaml` (`git show HEAD:configs/default.yaml` into a temporary file);
* seed 42, no drop filter, 15,308 stream jobs;
* s_C = 228.1843 $/kgCO2;
* commit `c2ad514`.

Phase 0, the reward, s_C, the defaults and the configs are unchanged. Nothing enters the online loop.

## Definitions (checked against the code)

* `reward.job_terms` (`reward.py:69–72`) computes cv = q·v, profit = φ_job − C_elec, and R = λ1·cv + λ2·profit −
  λ3·K·s_C. That is CV, PR and CO = s_C·K as specified.
* The three components are taken once per job (`pareto.job_components`) and once per type
  (`pareto.type_components`, from the state's type means), then combined linearly for each λ.
* Checked to the cent at λ = (1, 1, 1), both per job and per type.
* **λ-invariance of Phase 0.** s_C (`reward.py:140–143`, quantiles of λ-free CV and |PR|), the valuation grid, the
  hardware profiles and the job components are identical for all 7 subset λ. This was verified on each rebuilt
  Phase 0 (`lambda_invariance_checks.csv`).

## Grid and models

* **Grid:** λ = 3·w with w on the 1/4 simplex: 15 points, corners included. The centre (1, 1, 1), today's reward, is
  added, because it is not on the 1/4 grid: **16 points**.
* **7-point subset:** the corners, the centre, and the edge midpoints (1.5, 1.5, 0), (1.5, 0, 1.5), (0, 1.5, 1.5).
* **All 16 points:**
  * fluid LP OPT (weighted sum, on the reference state of `Benchmarks.fluid_opt`);
  * the Chebyshev epigraph LP on the same fluid problem;
  * G2 job greedy;
  * the job-level MILP. Its total runtime was 15.8 s and its maximum 3.6 s; all 16 solved to optimality, so no
    subset was needed.
* **Subset only:** StaticPolicy (Phase 0) and DLENT-RT theory, with Phase 0 rebuilt for each λ. Each DLENT-RT run took
  209–541 s, the 7 in parallel.
* **Total runtime:** 550 s.
* **Hypervolume:** exact in 3-D, with (CV, PR, −CO) maximised. The reference point is common to all models: the
  componentwise worst value over every row, pushed out by 1% of the range,
  (−1,562.6, −525.8, −93,043.2) $.

## Check

At λ = (1, 1, 1) the grid reproduces notebook 03 to the cent:

| model | value $ |
|---|---|
| job MILP | 155,397.61 |
| fluid OPT | 149,478.42 |
| G2 | 155,011.00 |
| StaticPolicy | 54,216.40 |
| DLENT-RT | 48,391.61 |

DLENT-RT was run through the λ-rebuild path.

## 7-point subset (Σ $ over the stream)

| λ | job MILP CV / PR / CO | fluid CV / PR / CO | G2 CV / PR / CO | Static CV / PR / CO | DLENT-RT CV / PR / CO |
|---|---|---|---|---|---|
| (3, 0, 0) | 150,157 / 31,657 / 64,042 | 156,265 / 31,256 / 92,122 | 142,779 / 27,256 / 62,255 | 63,469 / 13,088 / 45,741 | 57,944 / 10,096 / 34,038 |
| (0, 3, 0) | 135,748 / 52,580 / 58,351 | 136,622 / 48,859 / 78,492 | 132,429 / 50,960 / 56,512 | 56,968 / 13,285 / 41,967 | 50,104 / 11,360 / 30,708 |
| (0, 0, 3) | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| (1, 1, 1) | 135,657 / 44,998 / 25,257 | 135,904 / 38,784 / 25,209 | 135,118 / 44,874 / 24,981 | 54,376 / 11,243 / 11,403 | 50,288 / 9,116 / 11,013 |
| (1.5, 1.5, 0) | 146,847 / 49,670 / 61,371 | 153,260 / 44,691 / 88,972 | 140,567 / 47,469 / 59,539 | 61,154 / 13,348 / 44,213 | 55,995 / 10,153 / 33,217 |
| (1.5, 0, 1.5) | 134,905 / 27,685 / 21,053 | 134,960 / 26,804 / 21,176 | 134,883 / 27,600 / 21,067 | 54,214 / 10,326 / 8,662 | 48,220 / 8,231 / 8,600 |
| (0, 1.5, 1.5) | 101,837 / 39,990 / 9,313 | 95,783 / 35,228 / 8,933 | 101,837 / 39,990 / 9,313 | 43,200 / 9,954 / 5,022 | 40,708 / 8,776 / 6,193 |

**Hypervolume on the 7-point subset (×10¹⁴ $³):**

| model | hypervolume |
|---|---|
| job MILP | 5.51 |
| G2 | 5.32 |
| fluid | 4.51 |
| Chebyshev fluid | 4.27 |
| StaticPolicy | 0.69 |
| DLENT-RT | 0.55 |

All points are in `results.csv` and in the notebook.

## Corners and degenerate cases

* **(0, 0, 3):** R = −3·CO ≤ 0 for every job, so **every model admits nothing** and the frontier point is (0, 0, 0).
  This corner is degenerate.
* **(3, 0, 0):** R = 3·q·v > 0 for every job.
  * The job MILP and G2 want every job and are limited only by the per-tick capacity: 14,755 and 14,732 of 15,308
    admitted.
  * The fluid LP admits every type (x = 1, 15,308 expected), because its average capacity is slack.
  * StaticPolicy and DLENT-RT admit every type seen in H (unseen types have r̃ = 0 and are rejected) up to the gate,
    plus thinning for DLENT-RT: 12,200 and 9,181.
* **(0, 3, 0):** the jobs or types with PR > 0 are admitted, with carbon ignored.
* **Not monotone:** G2's (3, 0, 0) point is **dominated** by its own (2.25, 0, 0.75) point (CV 143.8k vs 142.8k,
  PR 28.8k vs 27.3k, CO 39.1k vs 62.3k). Admitting every arrival first-come blocks capacity.

## Things to flag

* **Chebyshev orientation.** In the requested form max t s.t. t ≤ w_k·(f_k − nadir_k)/(ideal_k − nadir_k), a larger
  w_k makes objective k's constraint LOOSER, so objective k matters LESS.
  * Example: λ = (0, 0.75, 2.25) gives CO = 61.8k $ under Chebyshev and 2.8k $ under the weighted sum.
  * The LP is also degenerate: t is set by the smallest positive weight, so many x tie, and the reported point is one
    LP vertex among them.
  * It was run as specified. The usual form, t ≤ (f_k − nadir_k)/(w_k·range_k), restores "larger weight = more
    important"; it was not run.
* **The fluid problem is an LP, so its front is convex and fully supported.** Chebyshev cannot find unsupported points
  there. The unsupported-point test is therefore on a discrete toy front (`chebyshev_discrete`).
* **The existing `pareto.scalarize_chebyshev` was not reused.** It is a per-type augmented Chebyshev used inside
  `type_reward` when `scalarization.method = chebyshev`, which is a different object from the epigraph LP over the
  fluid solution.
* **DLENT-RT's tests fire depending on λ:**
  * (3, 0, 0): 2 Test 1 and 2 **Test 3**;
  * (1.5, 1.5, 0): 2 Test 1 and 1 Test 3;
  * (0, 3, 0): 1 Test 1.

  This is the first time Test 3 has fired in any run. τ₀ = 0 at every subset λ.
* **Tests:** the fast suite has 66 passed (clean worktree with the new code), and the synthetic snapshot is
  bit-identical.

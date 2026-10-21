# Week 30: Pareto frontier over the objective weights λ (offline)

**Code:**
* `dlent_rt/pareto.py` (new functions; the existing per-type `scalarize_chebyshev` is untouched);
* `scripts/pareto_run.py` (runs the grid, writes the cache `reports/week30/pareto/*.csv`);
* `tests/test_pareto.py` (fast).

**Notebook:** `notebooks/04_pareto_lambda.ipynb`, which only reads the cache. Its source
`notebooks/src/04_pareto_lambda.py` and the cached results `reports/week30/pareto/*.csv` are force-added, because both
are git-ignored. Figures are in `reports/week30/pareto/figs/`.

**Instance:** the same as notebook 03:
* the committed `configs/default.yaml` (`git show HEAD:configs/default.yaml` into a temporary file);
* seed 42, no drop filter, 15,308 stream jobs;
* s_C = 228.1843 $/kgCO2;
* results cached at commit `08e9627`, configs/default.yaml blob `68e7bd7`.

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
  * the **standard augmented Chebyshev LP** on the same fluid problem (see below), and the legacy form for reference;
  * G2 job greedy;
  * the job-level MILP. Its total runtime was 15.8 s and its maximum 3.6 s; all 16 solved to optimality, so no
    subset was needed.
* **Subset only:** StaticPolicy (Phase 0) and DLENT-RT theory, with Phase 0 rebuilt for each λ. Each DLENT-RT run took
  209–541 s, the 7 in parallel.
* **Perturbations** (Kash's "does every λ matter?"): the centre and λ_k × 2 / × 0.5 (6 points). Fluid, G2, the MILP,
  StaticPolicy and DLENT-RT ran on all 6.
* **Total runtime:** 24 min, with 4 parallel workers for the 13 StaticPolicy / DLENT-RT runs.
  * The 6 DLENT-RT perturbation runs took 2,499 s (41.6 min) summed and ran in parallel.
  * A first attempt with 13 workers in parallel thrashed the 16 GB machine (about 1 GB per DLENT-RT run, 13 GB of swap)
    and was stopped after 3.5 h. The script now caps the pool at 4.
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

**Hypervolume on the 7-point subset (×10¹⁴ $³, rerun):**

| model | hypervolume |
|---|---|
| job MILP | 5.51 |
| G2 | 5.32 |
| fluid | 4.51 |
| Chebyshev fluid (standard) | 4.30 |
| Chebyshev fluid (legacy, inverted) | 4.27 |
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

## Chebyshev: standard form, and why the old one was wrong

* **Standard (`pareto.chebyshev_fluid`):**

      max  t + ρ·Σ_k g_k    s.t.  t ≤ g_k / w_k  for every k with w_k > 0,
      g_k = (f_k − nadir_k)/(ideal_k − nadir_k),  f = (CV, PR, −CO),  ρ = 10⁻³

  This is the point where the ray from the nadir in direction w meets the front, so a **larger weight means a more
  important objective**.
* **Augmentation:** ρ·Σg makes the solution Pareto-optimal rather than only weakly Pareto-optimal. The cost is that
  objectives can be traded at a rate of at most ρ/(1 + ρ).
* **Zero-weight objectives** have no t-constraint but stay in the augmentation term. A corner w = e_k therefore
  maximises objective k and breaks ties towards the other two.
  * At the CO corner this admits only zero-carbon types: CV 48 $, PR 38 $, CO 0.06 $. The weighted sum admits nothing
    there.
* **Not the same as your "equivalent" form.** min max_k w_k·(ideal_k − f_k)/range_k follows the ray from the ideal in
  direction 1/w, so it generally selects different points. Both orient the weights correctly.
* **The legacy form was wrong.** `chebyshev_fluid_legacy`, max t s.t. t ≤ w_k·g_k, has an **inverted weight
  orientation**: a larger w_k LOOSENS objective k's constraint, so that objective matters less.
  * At λ = (0, 0.75, 2.25), the legacy form gives CO = 61.8k $, against 858 $ for the standard form.
  * The legacy results look like the standard ones with the weights swapped.
  * Without augmentation the legacy LP is degenerate, with many tied x.
* **Tests:**
  * a larger weight never lowers its objective (k = 1, 2, 3; weights ×2 and ×4);
  * the augmented solutions are not dominated (LP dominance test);
  * the legacy form fails monotonicity.
* **On the real instance,** the dominance slack is at most 1.3·10⁻⁴ of the range for the standard Chebyshev and
  2.2·10⁻⁴ for the weighted sum, i.e. HiGHS tolerance level.

## Notebook sections added

* **Section 5:** simplex heat maps of Σ R(λ), Σ CV, Σ PR and Σ CO for the job MILP and the fluid LP.
* **Section 6:** DLENT-RT and StaticPolicy against the offline frontier.
  * The offline Pareto set is the non-dominated points of the union of the MILP, fluid, Chebyshev and G2 grid points:
    52 of 64.
  * Objectives are normalised by their range over that set.
  * d₂ is the normalised Euclidean distance to the nearest point of the set.
  * ε is the additive ε-indicator: the smallest uniform improvement, as a share of each objective's range, after which
    the point would weakly dominate some point of the set.

  | λ | Static % fluid / % MILP | Static d₂ / ε | DLENT-RT % fluid / % MILP | DLENT-RT d₂ / ε |
  |---|---|---|---|---|
  | (3, 0, 0) | 40.6 / 42.3 | 0.489 / 0.391 | 37.1 / 38.6 | 0.362 / 0.336 |
  | (0, 3, 0) | 27.2 / 25.3 | 0.443 / 0.369 | 23.3 / 21.6 | 0.321 / 0.300 |
  | (0, 0, 3) | – (0 / 0) | 0 / 0 (admit nothing = offline point) | – | 0 / 0 |
  | (1, 1, 1) | 36.3 / 34.9 | 0.116 / 0.110 | 32.4 / 31.1 | 0.128 / 0.106 |
  | (1.5, 1.5, 0) | 37.6 / 37.9 | 0.471 / 0.374 | 33.4 / 33.7 | 0.352 / 0.327 |
  | (1.5, 0, 1.5) | 40.0 / 40.0 | 0.096 / 0.080 | 34.8 / 34.8 | 0.121 / 0.088 |
  | (0, 1.5, 1.5) | 18.8 / 16.1 | 0.086 / 0.055 | 9.8 / 8.4 | 0.117 / 0.067 |

* **Section 7, "Does every λ matter?":** jobs changed (added + dropped) against the centre solution of the same model,
  summed over × 2 and × 0.5. Fluid counts are expected jobs.

  | model | λ1 | λ2 | λ3 |
  |---|---|---|---|
  | fluid LP | 2,157 | **1,344** | 1,592 |
  | G2 | 1,780 | **1,173** | 1,520 |
  | job MILP | 1,724 | **1,134** | 1,460 |
  | StaticPolicy | 1,485 | **880** | 1,205 |
  | DLENT-RT | 1,920 | **1,395** | 1,774 |

  **λ2 (profit) changes the solution least in every model.** Even so, λ2 × 2 or × 0.5 changes 505–658 MILP jobs out of
  12,829 (3.9–5.1%), so it is not negligible. The per-component changes are in `perturbations.csv` and the notebook.
* **Section 8, weighted sum vs Chebyshev:**
  * **On the fluid LP:** 3 of the 16 standard-Chebyshev points coincide with a weighted-sum grid point. The others are
    different Pareto points, all non-dominated. Because the fluid problem is an LP, its front is convex and every one
    of them is supported by some weight, just not by the 16 grid weights.
  * **On the discrete toy front:** over 101 weights, the weighted sum picks the unsupported point C 0 times and
    Chebyshev picks it 99 times.
  * **No claim of non-convexity is made for the real front.** The fluid front is convex by construction, and the
    MILP and G2 points are weighted-sum optima, so they are supported by construction.

## Other flags

* G2's (3, 0, 0) point is dominated by its own (2.25, 0, 0.75) point: admitting every arrival first-come blocks
  capacity.
* DLENT-RT's tests depend on λ. In this rerun:
  * Test 1 fires at (3, 0, 0), (0, 3, 0), (1.5, 1.5, 0), (2, 1, 1) and (1, 1, 0.5);
  * Test 3 fires at (3, 0, 0) and (1.5, 1.5, 0);
  * τ₀ = 0 at every λ run.
* **Tests:** the fast suite passes; see the commit message for the counts.

# Week 32: the theoretical models at f = 0.5, and the MILP fixes

* Raw numbers: `theory_f050_numbers.md`, from `scripts/week32/theory_f050.py`.
* Notebook: `notebooks/05_theory_models.ipynb`, whose section 0 shows the summary table.
* Config: `configs/v5_theory.yaml`.

Only type / average models were run: no online model and no job-level model. `configs/default.yaml`, the bid, the
carbon map and the type grid are unchanged. With the default config the numbers are the same as before, and the
snapshots do not change.

## 1. Decisions

| decision | why |
|---|---|
| `policy.unseen_types = "dual_rule"`, with the fallback estimates: hardware-profile means from H for duration, energy and costs; bid-bin means from H for bid and φ | It is the same rule ReInit uses for types it has never seen. |
| In `configs/v5_theory.yaml`: `capacity.f = 0.50` and `policy.unseen_types = dual_rule` | f = 0.50 is what the fixed rule picks with the dual rule (24.1% of the time ≥ 95% full). |
| H stays 7 days; the type grid is unchanged | |

## 2. MILP fixes and checks

**Time limit.**
* **The fix.** `benchmarks.occupancy_milp` now solves the LP relaxation and the MILP with `highspy` directly: a **new
  HiGHS instance for each solve, each with its own time limit**. The LP relaxation no longer uses the MILP's budget.
* **The test.** `tests/test_milp_solves.py` runs an LP and then two MILPs in one process, each with a 2 s limit; each
  MILP gets its full 2 s.
* **In the runs,** the one-week MILPs now use their full budget: 3,603 s at a 3,600 s limit, 901 s at 900 s. Their LP
  relaxations take 7–35 s.
* **I could not reproduce the old sharing** on small problems with scipy's `milp`, so I do not know its exact cause.
  The new code avoids it by construction.

**End of window.**
* **The new option.** `occupancy_milp(..., tail_step)` / `job_opt_bucketed(..., tail_step)` / `window_models(...,
  tail_hours=1.0)` adds capacity checks every hour after the last arrival, until the last chosen job ends. The old
  behaviour is `tail_hours=None`.
* **These checks can never bind.** No job arrives after the last arrival, so the occupancy can only fall. Between
  arrivals the occupancy also only falls. So the arrival-time checks already cover every peak. A test
  (`test_tail_checks_never_bind`) confirms the same optimum with and without them.
* **The one-day run confirms this.** With the tail checks the MILP still gives $1,456 (best solution), against
  $1,452 without, and fluid OPT is $1,298. So **fluid OPT ≥ bucketed MILP still fails on one day.**
* **My previous explanation was wrong.** The end-of-window effect is in the **fluid LP**, not the MILP. The fluid LP
  charges each admitted type its whole volume A × d̄ against C × T, including the part that runs after the window
  ends. So on a short window the fluid LP is too strict.
* **The check.** A fluid LP that counts only each job's time inside the window gives **$1,690, above the MILP's
  bound of $1,462.** So "fluid ≥ MILP" holds once both count the same time.

## 3. Discretization cost

The first 7 days of the stream: 59,985 jobs, 53,214 MILP candidates, 9.27 million nonzeros. Reference buckets, true
durations, each MILP in its own process. The cost is fluid on feasible types − bucketed MILP: the high end of the
range uses the MILP's best solution, the low end its bound.

| f | MILP limit | fluid OPT $ | fluid feasible $ | MILP best $ | MILP bound $ | gap | discretization cost $ | % |
|---|---|---|---|---|---|---|---|---|
| **0.5 (tail checks)** | 3,600 s | 6,175.8 | 6,175.8 | 5,578.4 | 5,637.9 | 1.07% | **538–597** | **8.7–9.7%** |
| 0.5 (old behaviour) | 3,600 s | 6,175.8 | 6,175.8 | 5,563.4 | 5,638.6 | 1.35% | 537–612 | 8.7–9.9% |
| 0.3 (tail checks) | 900 s | 4,491.0 | 4,491.0 | 3,619.9 | 4,189.0 | 15.7% | 302–871 | 6.7–19.4% |
| 0.7 (tail checks) | 900 s | 6,867.9 | 6,867.9 | 6,336.2 | 6,393.6 | 0.91% | 474–532 | 6.9–7.7% |

* **Fluid OPT ≥ fluid on feasible types ≥ bucketed MILP holds at every f.** Fluid OPT equals fluid on feasible types
  at these f, because no job is larger than C.
* **The two f = 0.5 runs differ only in the MILP's best solution** found in the time (the tail checks change
  nothing). Their bounds agree.
* **On one week the end-of-window effect is small.** The fluid LP counting only the volume inside the window is 2.3%
  higher at f = 0.5 ($6,316 against $6,176). So these costs are slightly too low, by up to that amount.
* **At f = 0.3 the MILP is far from optimal** (15.7% gap), so its range is wide.

## 4. The theoretical models at f = 0.5 (summary)

| quantity | value |
|---|---|
| C, CPU / RAM | 1.737 / 1.156 machines (111 cores / 296 GB, declared 64 / 256 per machine) |
| τ, Phase 0 LP (CPU, RAM) | 0.00206, 0.00162 |
| τ, fluid OPT (CPU, RAM) | 0.00210, 0.00169 |
| windows with τ > 0, 1 day (CPU, RAM) | 73.9%, 82.6% |
| windows with τ > 0, 6 h (CPU, RAM) | 50.5%, 50.5% |
| static policy (dual rule): time ≥ 95% full | 24.1% |
| static policy: mean occupancy / C (CPU, RAM) | 79.3%, 78.9% |
| static policy: jobs rejected for lack of space | 4.97% |
| static policy: realised reward (customer value / profit / carbon) | $11,017 (18,642 / 1,828 / −9,453) |
| fluid OPT (customer value / profit / carbon) | $20,346 (27,823 / 4,183 / −11,660) |
| Phase 0 LP expected | $22,204 |
| Phase 0 LP → stream statistics (Phase 0 grid) → fluid OPT | $22,204 → $20,380 → $20,346 |
| static policy / fluid OPT | 54.1% |
| discretization cost, one week | $538–597 (8.7–9.7%; MILP gap 1.07%) |

## 5. Open issues

1. **The fluid LP charges volume past the window's end.** On short windows that makes it pessimistic: on one day it
   is below the MILP. A window version (volume counted only inside the window) is in `window_models` as
   `fluid_window`, for diagnosis only. Decide whether the benchmarks should use it on windows.
2. **The discretization cost is still a range** (MILP gap 1.07% at f = 0.5 after an hour). An exact value needs a
   longer run or a decomposition.
3. **The static policy realises 54% of fluid OPT.** H's statistics are optimistic (−8% from H to the stream,
   section 4). The bigger part is H's admission fractions applied to the stream's different type mix: in
   `theory_models_2.md`, Phase 0's x valued on the stream statistics gives about half of fluid OPT. The capacity gate
   also blocks 5% of the arrivals.
4. **The old time-limit sharing was not reproduced on small problems,** so its exact cause is unknown. The new code
   gives each solve its own HiGHS instance.

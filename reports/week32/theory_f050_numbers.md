# Week 32: theoretical models at f = 0.5, all numbers

Config configs/v5_theory.yaml: f = 0.5, policy.unseen_types = dual_rule, quota at f = 1.25.

### N1. Bucketed MILP on a window (reference buckets, true durations; each MILP in its own process)

| run | jobs | candidates | nonzeros | tail checks | fluid OPT $ | fluid feasible $ | fluid, volume inside the window $ | MILP best $ | MILP bound $ | gap | discretization cost $ (best .. bound) | cost % | fluid >= MILP best | fluid (inside window) >= MILP bound | MILP status | MILP s | LP relaxation s | MILP realised (job rewards) $ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| day, f 0.5, tail checks | 8,003 | 6,974 | 1,021,856 | 23 | 1,297.59 | 1,297.59 | 1,689.94 | 1,456.42 | 1,461.49 | 0.35% | -158.83 .. -163.90 | -12.24 .. -12.63 | False | True | Time limit reached | 900 | 1 | 1,457.05 |
| day, f 0.5, old | 8,003 | 6,974 | 1,020,170 | 0 | 1,297.59 | 1,297.59 | 1,689.94 | 1,452.33 | 1,461.86 | 0.66% | -154.74 .. -164.27 | -11.93 .. -12.66 | False | True | Time limit reached | 900 | 1 | 1,453.39 |
| week, f 0.5, tail checks | 59,985 | 53,214 | 9,267,322 | 22 | 6,175.83 | 6,175.83 | 6,315.76 | 5,578.35 | 5,637.86 | 1.07% | 597.47 .. 537.96 | 9.67 .. 8.71 | True | True | Time limit reached | 3603 | 14 | 5,678.23 |
| week, f 0.5, old | 59,985 | 53,214 | 9,266,342 | 0 | 6,175.83 | 6,175.83 | 6,315.76 | 5,563.36 | 5,638.57 | 1.35% | 612.46 .. 537.26 | 9.92 .. 8.70 | True | True | Time limit reached | 3601 | 14 | 5,659.30 |
| week, f 0.3, tail checks | 59,985 | 53,214 | 9,267,322 | 22 | 4,491.03 | 4,491.03 | 4,715.05 | 3,619.86 | 4,188.98 | 15.72% | 871.17 .. 302.05 | 19.40 .. 6.73 | True | True | Time limit reached | 901 | 35 | 3,754.51 |
| week, f 0.7, tail checks | 59,985 | 53,214 | 9,267,322 | 22 | 6,867.87 | 6,867.87 | 6,927.19 | 6,336.22 | 6,393.57 | 0.91% | 531.65 .. 474.30 | 7.74 .. 6.91 | True | True | Time limit reached | 901 | 7 | 6,367.64 |

### N2. The theoretical models at f = 0.5 (v5_theory, dual rule)

| quantity | value |
|---|---|
| f | 0.5 |
| C, CPU / RAM (machines) | 1.737 / 1.156 (111 cores / 296 GB, declared 64 / 256) |
| tau, Phase 0 LP (CPU, RAM) | 0.00206, 0.00162 |
| tau, fluid OPT (CPU, RAM) | 0.00210, 0.00169 |
| windows with tau > 0, 1 day (CPU, RAM) | 73.9%, 82.6% |
| windows with tau > 0, 6 h (CPU, RAM) | 50.5%, 50.5% |
| static policy (dual rule): time >= 95% full | 24.1% |
| static policy: mean occupancy / C (CPU, RAM) | 79.3%, 78.9% |
| static policy: jobs rejected for lack of space | 4.97% |
| static policy: realised reward $ (CV / profit / carbon) | 11,017 (18,642 / 1,828 / -9,453) |
| fluid OPT $ (CV / profit / carbon) | 20,346 (27,823 / 4,183 / -11,660) |
| Phase 0 LP expected $ | 22,204 |
| decomposition: Phase 0 LP -> stream statistics (Phase 0 grid) -> fluid OPT $ | 22,204 -> 20,380 -> 20,346 |
| static policy / fluid OPT | 54.1% |
| discretization cost, one week (fluid feasible - bucketed MILP) | 597 .. 538 $ (9.7% .. 8.7%; MILP gap 1.07%) |

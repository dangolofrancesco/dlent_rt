# Week 32: theory models part 2 (configs/v5_theory.yaml), all numbers


### N1. Static Phase 0 policy on the real stream: unseen types rejected vs dual rule (seed 42)

| f | unseen types | % time >= 95% full (CPU or RAM) | mean occupancy / C (CPU, RAM) | % jobs rejected for lack of space | jobs admitted | unseen-type jobs admitted | realised reward $ | split CV / profit / carbon $ | static / fluid OPT |
|---|---|---|---|---|---|---|---|---|---|
| 0.1 | reject | 1.4% | 28.6%, 20.2% | 0.09% | 10,561 | 0 | 1,107.8 | 1,632 / 160 / -684 | 17.4% |
| 0.1 | dual_rule | 26.0% | 79.9%, 72.0% | 1.47% | 9,246 | 1,410 | 1,994.7 | 3,486 / 431 / -1,922 | 31.3% |
| 0.2 | reject | 1.4% | 40.1%, 35.7% | 0.48% | 66,397 | 0 | 3,249.7 | 4,658 / 495 / -1,903 | 30.3% |
| 0.2 | dual_rule | 28.0% | 79.2%, 76.2% | 4.03% | 59,796 | 1,686 | 4,203.7 | 7,078 / 884 / -3,758 | 39.2% |
| 0.3 | reject | 4.0% | 48.7%, 45.9% | 0.89% | 102,801 | 0 | 5,635.9 | 8,313 / 831 / -3,507 | 38.9% |
| 0.3 | dual_rule | 25.9% | 80.8%, 78.0% | 4.38% | 96,408 | 1,871 | 7,222.9 | 11,576 / 1,436 / -5,789 | 49.8% |
| 0.5 | reject | 4.8% | 55.0%, 54.7% | 1.18% | 171,133 | 0 | 10,105.1 | 15,323 / 1,403 / -6,621 | 49.7% |
| 0.5 | dual_rule | 24.1% | 79.3%, 78.9% | 4.97% | 164,104 | 2,092 | 11,016.9 | 18,642 / 1,828 / -9,453 | 54.1% |
| 0.7 | reject | 4.9% | 52.8%, 53.1% | 1.06% | 201,087 | 0 | 11,970.5 | 19,220 / 1,623 / -8,872 | 52.3% |
| 0.7 | dual_rule | 16.3% | 75.8%, 75.5% | 3.21% | 198,373 | 2,395 | 14,127.6 | 24,298 / 2,428 / -12,599 | 61.7% |
| 1.25 | reject | 0.5% | 37.2%, 37.9% | 0.09% | 220,136 | 0 | 13,592.4 | 22,618 / 2,000 / -11,025 | 59.0% |
| 1.25 | dual_rule | 5.0% | 59.8%, 60.2% | 0.69% | 221,352 | 2,630 | 18,074.9 | 32,490 / 3,175 / -17,590 | 78.5% |

Stream jobs in types unseen in H: 2.01%, holding 41.8% of the stream's CPU-hours.
Fixed rule (closest to 20-30%, ties to 25%), reject: f = 0.7 (4.9%).
Fixed rule (closest to 20-30%, ties to 25%), dual_rule: f = 0.5 (24.1%).

N2. Types (Phase 0 grid): with jobs in H 4,005, in the stream 5,558, in both 3,724. Seen in both, weighted by H's rate: mean ratio stream / H of the arrival rate 1.533, of the mean r*d 1.034, of r~ 0.624. CPU load (machines = sum of rate x mean r*d per hour): H 3.474, stream 3.219, of which in types unseen in H 1.347.

### N3. The 10 types (seen in both) whose CPU load differs most between H and the stream (f-independent)

| type | lambda H /h | lambda stream /h | mean r*d H | mean r*d stream | load H | load stream | load gap | bid bin | priority | hw profile |
|---|---|---|---|---|---|---|---|---|---|---|
| 20418 | 0.01191 | 0.001736 | 15.61 | 16.74 | 0.1859 | 0.02907 | -0.1568 | 128 | 4.0 | 2 |
| 19074 | 0.01191 | 0.005209 | 12.5 | 9.687 | 0.1488 | 0.05046 | -0.09839 | 120 | 2.0 | 2 |
| 18511 | 0.02381 | 0.001736 | 3.771 | 2.247 | 0.0898 | 0.003901 | -0.0859 | 116 | 4.0 | 15 |
| 18832 | 0.01786 | 0.006946 | 6.034 | 4.141 | 0.1077 | 0.02876 | -0.07898 | 118 | 4.0 | 16 |
| 19618 | 0.005953 | 0.001736 | 15.43 | 11.28 | 0.09186 | 0.01958 | -0.07227 | 123 | 4.0 | 2 |
| 19714 | 0.005953 | 0.001736 | 15.36 | 13.02 | 0.09144 | 0.02261 | -0.06883 | 124 | 2.0 | 2 |
| 19778 | 0.005953 | 0.005209 | 15.9 | 10.28 | 0.09466 | 0.05354 | -0.04112 | 124 | 4.0 | 2 |
| 19298 | 0.005953 | 0.001736 | 9.002 | 8.246 | 0.05359 | 0.01432 | -0.03927 | 121 | 4.0 | 2 |
| 18818 | 0.005953 | 0.001736 | 6.876 | 5.058 | 0.04093 | 0.008783 | -0.03215 | 118 | 4.0 | 2 |
| 18191 | 0.01191 | 0.003473 | 3.389 | 2.644 | 0.04034 | 0.009184 | -0.03116 | 114 | 4.0 | 15 |

### N4. Phase 0 LP expected vs fluid OPT: decomposition (stream T)

| f | Phase 0 LP expected $ | LP on stream statistics, Phase 0 grid $ | Phase 0 x on stream statistics $ | fluid OPT (refit grid) $ |
|---|---|---|---|---|
| 0.3 | 16,315.9 | 14,625.1 | 6,672.6 | 14,501.5 |
| 0.5 | 22,204.0 | 20,380.1 | 10,446.4 | 20,346.2 |
| 0.7 | 25,481.9 | 22,836.9 | 13,227.5 | 22,901.2 |

### N5. One window of the stream (days 0-7 after H), reference buckets, true durations, MILP time limit 900 s

| f | jobs in the window | MILP candidates / nonzeros | fluid OPT $ | fluid feasible $ | bucketed MILP (type rewards) $ | bucketed MILP realised (job rewards) $ | discretization cost $ | discretization cost % | fluid >= feasible >= MILP | MILP status, gap | MILP runtime s | LPs runtime s | true-duration feasible | MILP LP-relaxation bound $ | cost range (LP bound .. incumbent) $ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.3 | 59,985 | 53,214 / 9,266,342 | 4,491.03 | 4,491.03 | 3,619.86 | 3,754.51 | 871.17 | 19.40% | True | Time limit reached. (HiGHS Status 13: Time limit reached), 1.57e-01 | 562 | 0.1 | True | 4,226.07 | 264.96 .. 871.17 |
| 0.5 | 59,985 | 53,214 / 9,266,342 | 6,175.83 | 6,175.83 | 5,271.86 | 5,360.98 | 903.97 | 14.64% | True | Time limit reached. (HiGHS Status 13: Time limit reached), 6.98e-02 | 561 | 0.1 | True | 5,659.51 | 516.32 .. 903.97 |
| 0.7 | 59,985 | 53,214 / 9,266,342 | 6,867.87 | 6,867.87 | 6,353.44 | 6,395.09 | 514.43 | 7.49% | True | Time limit reached. (HiGHS Status 13: Time limit reached), 6.32e-03 | 561 | 0.1 | True | 6,412.64 | 455.22 .. 514.43 |

### N6. One DAY of the stream (days 0-1 after H), same models

| f | jobs in the window | MILP candidates / nonzeros | fluid OPT $ | fluid feasible $ | bucketed MILP (type rewards) $ | bucketed MILP realised (job rewards) $ | discretization cost $ | discretization cost % | fluid >= feasible >= MILP | MILP status, gap | MILP runtime s | MILP LP-relaxation bound $ | cost range (LP bound .. incumbent) $ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.3 | 8,003 | 6,974 / 1,020,170 | 916.20 | 916.20 | 1,096.44 | 1,098.60 | -180.24 | -19.67% | False | Time limit reached. (HiGHS Status 13: Time limit reached), 2.26e-03 | 900 | 1,109.95 | -193.75 .. -180.24 |
| 0.5 | 8,003 | 6,974 / 1,020,170 | 1,297.59 | 1,297.59 | 1,454.58 | 1,456.09 | -156.98 | -12.10% | False | Time limit reached. (HiGHS Status 13: Time limit reached), 4.56e-03 | 900 | 1,467.90 | -170.31 .. -156.98 |
| 0.7 | 8,003 | 6,974 / 1,020,170 | 1,555.17 | 1,555.17 | 1,658.02 | 1,658.99 | -102.85 | -6.61% | False | Time limit reached. (HiGHS Status 13: Time limit reached), 1.78e-04 | 900 | 1,663.18 | -108.01 .. -102.85 |

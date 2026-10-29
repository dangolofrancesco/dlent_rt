# Week 32: v6 bid (size_u0), all numbers


### N1. u0 scan (Phase 0, v6; tau and LP fill at f = 0.5 and f = 1.25)

| u0 $/machine-h | s_C $/kg | positive types | positive mass | |r| share CV / profit / carbon | bid per machine-h mean / p10 / p90 $ | bid / electricity (median, jobs with electricity > 0) | tau f 0.5 | LP fill f 0.5 | tau f 1.25 | LP fill f 1.25 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 9.896 | 77.0% | 90.3% | 0.557 / 0.129 / 0.315 | 3.38 / 1.81 / 5.12 | 97.57 | [0.0009, 0.00071] | 100.0% / 100.0% | [0.0, 0.0] | 70.1% / 69.5% |
| 3.26 | 20.54 | 77.8% | 91.0% | 0.560 / 0.131 / 0.309 | 6.64 / 4.04 / 9.63 | 202.8 | [0.002639, 0.000509] | 100.0% / 100.0% | [0.0, 0.0] | 70.9% / 68.8% |
| 10.0 | 42.91 | 77.2% | 90.0% | 0.560 / 0.132 / 0.307 | 13.4 / 8.51 / 19 | 415.7 | [0.006232, 0.0] | 100.0% / 98.2% | [0.0, 0.0] | 70.5% / 67.7% |
| 23.93 | 88.75 | 76.5% | 89.7% | 0.561 / 0.133 / 0.306 | 27.3 / 17.6 / 38.6 | 852.6 | [0.013316, 0.0] | 100.0% / 95.3% | [0.0, 0.0] | 70.6% / 67.3% |
| 50.0 | 176.5 | 75.7% | 90.3% | 0.560 / 0.133 / 0.308 | 53.4 / 34.6 / 75.3 | 1668 | [0.026002, 0.0] | 100.0% / 93.8% | [0.0, 0.0] | 70.6% / 66.8% |

### N2. f selection on v6 (static Phase 0 policy, dual rule, seed 42)

| f | % time >= 95% full | mean occupancy / C (CPU) | realised reward $ | fluid OPT $ | tau Phase 0 LP |
|---|---|---|---|---|---|
| 0.3 | 24.7% | 78.0% | 28,214.69 | 56,574.14 | [0.019993, 0.0] |
| 0.4 | 25.2% | 80.8% | 38,223.46 | 68,994.73 | [0.016415, 0.0] |
| 0.5 | 23.7% | 80.0% | 47,579.16 | 79,784.01 | [0.013316, 0.0] |
| 0.6 | 20.5% | 78.4% | 54,045.19 | 88,077.55 | [0.011568, 0.0] |
| 0.7 | 15.5% | 75.1% | 61,721.01 | 93,857.78 | [0.008187, 0.0] |

Fixed rule (closest to 20-30%, ties to 25%): f = 0.4. configs/v6_theory.yaml has f = 0.4.

### N3. Theory summary, v5 vs v6 (no MILP)

| quantity | v5_theory (size bid, f 0.5) | v6 (size_u0 bid) at f 0.5 | v6_theory (size_u0 bid, f 0.4) |
|---|---|---|---|
| f | 0.5 | 0.5 | 0.4 |
| C, CPU / RAM (machines) | 1.737 / 1.156 (111 cores / 296 GB, declared 64 / 256) | 1.737 / 1.156 (111 cores / 296 GB, declared 64 / 256) | 1.390 / 0.925 (89 cores / 237 GB, declared 64 / 256) |
| tau, Phase 0 LP (CPU, RAM) | 0.00206, 0.00162 | 0.01332, 0.00000 | 0.01642, 0.00000 |
| tau, fluid OPT (CPU, RAM) | 0.00210, 0.00169 | 0.01349, 0.00000 | 0.01621, 0.00000 |
| windows with tau > 0, 1 day (CPU, RAM) | 73.9%, 82.6% | 87.0%, 43.5% | 91.3%, 52.2% |
| windows with tau > 0, 6 h (CPU, RAM) | 50.5%, 50.5% | 68.4%, 45.3% | 81.1%, 50.5% |
| static policy (dual rule): time >= 95% full | 24.1% | 23.7% | 25.2% |
| static policy: mean occupancy / C (CPU, RAM) | 79.3%, 78.9% | 80.0%, 78.2% | 80.8%, 79.4% |
| static policy: jobs rejected for lack of space | 4.97% | 5.55% | 4.81% |
| static policy: realised reward $ (CV / profit / carbon) | 11,017 (18,642 / 1,828 / -9,453) | 47,579 (76,778 / 8,420 / -37,619) | 38,223 (61,658 / 7,036 / -30,470) |
| fluid OPT $ (CV / profit / carbon) | 20,346 (27,823 / 4,183 / -11,660) | 79,784 (108,314 / 17,366 / -45,896) | 68,995 (90,441 / 15,506 / -36,953) |
| Phase 0 LP expected $ | 22,204 | 85,967 | 74,977 |
| decomposition: Phase 0 LP -> stream statistics (Phase 0 grid) -> fluid OPT $ | 22,204 -> 20,380 -> 20,346 | 85,967 -> 80,265 -> 79,784 | 74,977 -> 69,830 -> 68,995 |
| static policy / fluid OPT | 54.1% | 59.6% | 55.4% |
| discretization cost, one week | skipped (MILP not run) | skipped (MILP not run) | skipped (MILP not run) |

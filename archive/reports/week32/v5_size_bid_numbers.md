# Week 32: size-based bid (v5) vs v4, all numbers


### N1. Phase 0, v4 vs v5

| config | s_C $/kg | types with jobs in H | tau CPU, RAM | types admitted fully / partly | LP fill CPU / RAM | positive types / arrival mass | split all CV / profit / carbon $/h | split admitted $/h | C CPU / RAM |
|---|---|---|---|---|---|---|---|---|---|
| v4 (base-utility bid, f 1.00) | 1.434e+04 | 5,845 | [0.0, 0.0] | 3,595 / 0 | 26.0% / 34.7% | 61.5% / 80.7% | 20,134 / 2,292 / -19,985 | 16,514 / 2,443 / -5,438 | 2.542 / 1.658 |
| v5 (size bid, f 1.25) | 22.47 | 4,005 | [0.0, 0.0] | 2,670 / 0 | 65.9% / 65.0% | 66.7% / 87.8% | 76 / 6 / -43 | 70 / 11 / -35 | 4.343 / 2.891 |

v5 quota (fixed point): Q = [0.8686, 0.5782]; jobs H 46,596, stream 248,185; flags: ['tau = 0 on both resources: the capacity does not bind in the LP'].

### N2. Scale check on v5: gamma1 and gamma2 both x k (same data, same random numbers)

| gamma factor | positive types | positive arrival mass | s_C $/kg | s_C / s_C(1) | types with jobs in H | tau |
|---|---|---|---|---|---|---|
| 0.5 | 67.33% | 87.95% | 10.8617 | 0.483422 | 4126 | [0.0, 0.0] |
| 1.0 | 66.67% | 87.78% | 22.4683 | 1.000000 | 4005 | [0.0, 0.0] |
| 2.0 | 66.35% | 87.63% | 45.2064 | 2.012012 | 3929 | [0.0, 0.0] |

### N3. v5: per-job reward terms by decile of CPU request x capped duration (H + stream)

| decile | mean CPU | mean d (h) | mean bid $ | mean customer value $ | mean profit $ | mean carbon $ | mean reward $ | reward per machine-h $ | % positive reward | share of CPU-hours |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.0020 | 0.020 | 0.0003715 | 0.001456 | 4.268e-05 | 0.000446 | 0.001053 | 28.85 | 92.9% | 0.04% |
| 2 | 0.0037 | 0.019 | 0.0005951 | 0.002337 | 0.0001579 | 0.00078 | 0.001715 | 25.96 | 92.6% | 0.08% |
| 3 | 0.0051 | 0.022 | 0.000845 | 0.003295 | -3.014e-05 | 0.001171 | 0.002094 | 20.9 | 91.7% | 0.12% |
| 4 | 0.0073 | 0.023 | 0.001048 | 0.00409 | 0.0002578 | 0.001594 | 0.002754 | 19.62 | 91.1% | 0.17% |
| 5 | 0.0088 | 0.035 | 0.001613 | 0.006209 | 0.0003949 | 0.002472 | 0.004132 | 19.27 | 89.8% | 0.26% |
| 6 | 0.0147 | 0.051 | 0.002747 | 0.01043 | 0.0005451 | 0.004477 | 0.006493 | 16.62 | 88.0% | 0.47% |
| 7 | 0.0258 | 0.062 | 0.004269 | 0.01627 | 0.0008866 | 0.007569 | 0.009586 | 14.29 | 88.3% | 0.81% |
| 8 | 0.0346 | 0.111 | 0.007889 | 0.02967 | 0.001752 | 0.01403 | 0.01739 | 14.13 | 87.6% | 1.49% |
| 9 | 0.0435 | 0.247 | 0.0175 | 0.06524 | 0.005241 | 0.03118 | 0.0393 | 14.65 | 86.8% | 3.24% |
| 10 | 0.0884 | 2.015 | 0.4648 | 1.599 | 0.1553 | 0.9136 | 0.8408 | 10.9 | 83.2% | 93.31% |

v5 all jobs: positive reward 89.2%; CPU-hours of jobs with positive reward 77.1%; markup bid / cost base 0.9991.

### N4. v5: two worked examples

| term | median job (by r*d) | job nearest the CPU quota Q = 0.869 |
|---|---|---|
| CPU (machines) | 0.0057 | 0.8685 |
| RAM (machines) | 0.0026 | 0.3851 |
| priority tier | 4 | 4 |
| duration (h) | 0.0492 | 0.0167 |
| r*d (machine-h) | 0.0002787 | 0.01448 |
| (1 + ln q) | 2.3863 | 2.3863 |
| size rate $/h g1 cores + g2 GB | 0.0150 | 2.2841 |
| cost base $ | 0.0017639 | 0.090846 |
| bid $ | 0.0020142 | 0.11644 |
| phi $ | 0.0015145 | 0.093265 |
| customer value $ | 0.0080569 | 0.46577 |
| electricity $ | 3.0258e-06 | 0.00035988 |
| energy kWh | 0.0003412 | 0.01769 |
| carbon kg | 0.0001271 | 0.007108 |
| carbon $ (s_C x kg) | 0.0028568 | 0.15971 |
| profit $ | 0.0015115 | 0.092905 |
| reward $ | 0.0067116 | 0.39897 |

N5. Per CPU machine-hour (v5 jobs with r*d > 0): bid mean 7.602 $, p10 3.954, p90 11.6; electricity mean 0.02782 $, p10 0.01582, p90 0.03948; bid / electricity: median 248.1.
N6. LP load p*v / C: positive types [0.6586, 0.6501] (CPU, RAM), negative types [0.1562, 0.1558]; arrival mass positive 87.8%; mean type r*d (machine-h, p-weighted) positive 0.01175, negative 0.02001; tau [0.0, 0.0].
N6 (v4). LP load p*v / C: positive types [0.2603, 0.3474] (CPU, RAM), negative types [0.7403, 0.6542]; arrival mass positive 80.7%; mean type r*d (machine-h, p-weighted) positive 0.002983, negative 0.03538; tau [0.0, 0.0].

### N7. v5 at f = 1.25: quota share 0.2 vs 0.5 (each with its fixed point)

| quota share | C CPU / RAM | Q CPU / RAM | fixed-point steps | % time demand >= 95% C | jobs kept | CPU-hours kept | tau | LP fill CPU / RAM | positive types / mass | s_C $/kg |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.2 | 4.343 / 2.891 | 0.8686 / 0.5782 | 8 | 30.2% | 98.3% | 3.8% | [0.0, 0.0] | 65.9% / 65.0% | 66.7% / 87.8% | 22.47 |
| 0.5 | 21.817 / 13.390 | 10.9084 / 6.6952 | 5 | 24.9% | 99.6% | 16.9% | [0.0, 0.0] | 60.7% / 64.3% | 68.0% / 87.5% | 21.57 |

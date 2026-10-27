# Week 32 step 3: all numbers (limits.md)

Jobs: 30000 (data/batch_v2_30k.csv). Jobs with a zero CPU request (keep the span): 12.

### N1. Span versus effective duration (no cap, no limit)

| duration | CPU-hours | RAM-hours | mean duration (h) |
|---|---|---|---|
| span (first start to last end) | 15,298 | 9,616 | 0.570 |
| effective (real CPU-hours / request) | 7,341 | 4,719 | 0.494 |
| real, each instance on its own span (reference) | 7,341 | 4,743 | - |

Jobs whose duration changes by more than 10%: 4422 (14.7%); they hold 71.0% of the span CPU-hours. Effective / span: median 1.000, q0.10 0.677.

### N2. R_max x d_max (machines). CPU-hours kept: of all 30,000 jobs, effective duration, no cap

| R_max | d_max | T (h) | jobs kept | CPU-hours kept | cluster CPU | cluster RAM | jobs larger than cluster | their work | FCFS jobs accepted | FCFS CPU-hours served | time >= 95% full |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 4.00 h | 425.7 | 98.4% | 2.0% | 0.24 | 0.15 | 516 | 43.4% | 93.1% | 43.2% | 2.0% |
| 1 | 12.00 h | 425.7 | 98.4% | 2.9% | 0.35 | 0.22 | 308 | 40.3% | 94.6% | 46.2% | 3.5% |
| 1 | 21.28 h (0.05 T) | 425.7 | 98.4% | 3.8% | 0.47 | 0.29 | 229 | 41.3% | 96.6% | 49.1% | 2.7% |
| 2 | 4.00 h | 427.5 | 98.8% | 3.1% | 0.27 | 0.17 | 560 | 61.8% | 94.2% | 30.1% | 1.9% |
| 2 | 12.00 h | 427.5 | 98.8% | 4.6% | 0.44 | 0.28 | 364 | 60.3% | 96.5% | 34.2% | 1.6% |
| 2 | 21.37 h (0.05 T) | 427.5 | 98.8% | 6.0% | 0.79 | 0.39 | 234 | 54.4% | 98.2% | 35.2% | 1.0% |
| 5 | 4.00 h | 429.6 | 99.4% | 5.7% | 0.38 | 0.24 | 606 | 77.2% | 95.9% | 19.6% | 1.0% |
| 5 | 12.00 h | 429.6 | 99.4% | 7.7% | 0.75 | 0.41 | 428 | 70.7% | 97.6% | 24.0% | 1.0% |
| 5 | 21.48 h (0.05 T) | 429.6 | 99.4% | 9.5% | 1.08 | 0.75 | 315 | 59.0% | 96.4% | 35.0% | 8.6% |
| 10 | 4.00 h | 430.3 | 99.7% | 8.1% | 0.43 | 0.28 | 640 | 83.7% | 96.3% | 14.5% | 1.0% |
| 10 | 12.00 h | 430.3 | 99.7% | 11.6% | 0.99 | 0.61 | 424 | 75.6% | 97.8% | 20.1% | 2.0% |
| 10 | 21.51 h (0.05 T) | 430.3 | 99.7% | 13.4% | 1.40 | 0.92 | 357 | 70.0% | 98.2% | 27.7% | 1.5% |
| 20 | 4.00 h | 430.9 | 99.9% | 13.6% | 0.55 | 0.34 | 624 | 89.4% | 96.4% | 9.4% | 1.2% |
| 20 | 12.00 h | 430.9 | 99.9% | 18.6% | 1.16 | 0.75 | 444 | 83.8% | 97.3% | 14.5% | 2.5% |
| 20 | 21.55 h (0.05 T) | 430.9 | 99.9% | 21.9% | 1.92 | 0.99 | 378 | 77.0% | 98.4% | 17.6% | 0.5% |
| no limit | 4.00 h | 431.5 | 100.0% | 36.9% | 0.84 | 0.48 | 568 | 95.5% | 97.7% | 4.1% | 0.3% |
| no limit | 12.00 h | 431.5 | 100.0% | 58.8% | 2.40 | 1.42 | 361 | 92.2% | 98.6% | 5.9% | 0.2% |
| no limit | 21.57 h (0.05 T) | 431.5 | 100.0% | 63.9% | 3.07 | 1.89 | 308 | 89.9% | 98.7% | 8.9% | 0.6% |

### N2b. Extra scan below 1 machine (same columns)

| R_max | d_max | T (h) | jobs kept | CPU-hours kept | cluster CPU | cluster RAM | jobs larger than cluster | their work | FCFS jobs accepted | FCFS CPU-hours served | time >= 95% full |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.02 | 4.00 h | 292.8 | 67.2% | 0.3% | 0.04 | 0.03 | 0 | 0.0% | 78.5% | 70.4% | 5.7% |
| 0.02 | 12.00 h | 292.8 | 67.2% | 0.4% | 0.05 | 0.04 | 0 | 0.0% | 80.2% | 73.2% | 7.2% |
| 0.02 | 14.64 h (0.05 T) | 292.8 | 67.2% | 0.4% | 0.05 | 0.04 | 0 | 0.0% | 81.8% | 73.2% | 6.3% |
| 0.05 | 4.00 h | 403.2 | 87.2% | 0.7% | 0.09 | 0.06 | 0 | 0.0% | 78.7% | 68.3% | 7.4% |
| 0.05 | 12.00 h | 403.2 | 87.2% | 0.8% | 0.11 | 0.07 | 0 | 0.0% | 82.3% | 71.3% | 7.5% |
| 0.05 | 20.16 h (0.05 T) | 403.2 | 87.2% | 0.9% | 0.12 | 0.08 | 0 | 0.0% | 84.3% | 74.3% | 7.5% |
| 0.1 | 4.00 h | 414.2 | 94.7% | 0.9% | 0.12 | 0.07 | 90 | 2.5% | 82.5% | 65.8% | 4.7% |
| 0.1 | 12.00 h | 414.2 | 94.7% | 1.1% | 0.14 | 0.09 | 19 | 0.3% | 84.7% | 68.2% | 6.0% |
| 0.1 | 20.71 h (0.05 T) | 414.2 | 94.7% | 1.3% | 0.17 | 0.10 | 0 | 0.0% | 85.7% | 70.9% | 6.7% |
| 0.2 | 4.00 h | 419.5 | 96.8% | 1.1% | 0.15 | 0.10 | 307 | 18.4% | 86.3% | 60.2% | 4.7% |
| 0.2 | 12.00 h | 419.5 | 96.8% | 1.5% | 0.21 | 0.14 | 111 | 7.5% | 91.0% | 64.3% | 3.8% |
| 0.2 | 20.98 h (0.05 T) | 419.5 | 96.8% | 1.8% | 0.25 | 0.17 | 59 | 2.3% | 91.2% | 69.7% | 4.7% |
| 0.3 | 4.00 h | 421.8 | 97.4% | 1.3% | 0.17 | 0.11 | 405 | 24.7% | 88.8% | 57.8% | 3.5% |
| 0.3 | 12.00 h | 421.8 | 97.4% | 1.8% | 0.25 | 0.16 | 170 | 7.4% | 90.8% | 64.6% | 6.1% |
| 0.3 | 21.09 h (0.05 T) | 421.8 | 97.4% | 2.2% | 0.30 | 0.20 | 69 | 1.2% | 92.5% | 68.2% | 5.6% |
| 0.5 | 4.00 h | 423.4 | 97.8% | 1.5% | 0.20 | 0.13 | 445 | 28.8% | 91.0% | 53.8% | 2.9% |
| 0.5 | 12.00 h | 423.4 | 97.8% | 2.0% | 0.28 | 0.18 | 253 | 17.0% | 93.0% | 59.8% | 3.8% |
| 0.5 | 21.17 h (0.05 T) | 423.4 | 97.8% | 2.4% | 0.32 | 0.22 | 161 | 10.1% | 93.1% | 63.9% | 5.9% |

Rule: largest R_max with the jobs larger than the cluster holding < 1% of the kept work; among those, the d_max that keeps the most work. -> R_max = 0.1, d_max = 20.71 h (0.05 T). Keeps 94.7% of the jobs and 1.3% of the CPU-hours. Cluster 0.17 CPU / 0.10 RAM machines (11 cores / 27 GB under 64 / 256). Pairs that pass: (0.02, 4.0), (0.02, 12.0), (0.02, 0.05T), (0.05, 4.0), (0.05, 12.0), (0.05, 0.05T), (0.1, 12.0), (0.1, 0.05T).
T at the recommended pair: 414.2 h; 0.05 T = 20.71 h; xi on the data = 0.951 (forced xi_override = 0.001).

### N4. Option (not a current rule): cluster = max(level exceeded 25% of the time, R_max)

| R_max | d_max | CPU-hours kept | cluster CPU | cluster RAM | jobs larger than cluster | FCFS jobs accepted | FCFS CPU-hours served | time >= 95% full |
|---|---|---|---|---|---|---|---|---|
| 1 | 21.28 h (0.05 T) | 3.8% | 1.00 | 1.00 | 0 | 99.5% | 71.6% | 0.4% |
| 2 | 21.37 h (0.05 T) | 6.0% | 2.00 | 2.00 | 0 | 99.6% | 75.7% | 0.7% |
| 5 | 21.48 h (0.05 T) | 9.5% | 5.00 | 5.00 | 0 | 99.7% | 90.7% | 0.8% |
| 10 | 21.51 h (0.05 T) | 13.4% | 10.00 | 10.00 | 0 | 99.9% | 83.4% | 0.2% |
| 20 | 21.55 h (0.05 T) | 21.9% | 20.00 | 20.00 | 0 | 99.9% | 88.5% | 0.6% |

Zero-CPU jobs left out of the log-scale figures: 12.
CPU request <= 1 machines: 98.41% of jobs, 13.0% of the CPU-hours.
CPU request <= 2 machines: 98.80% of jobs, 26.6% of the CPU-hours.
CPU request <= 5 machines: 99.46% of jobs, 30.8% of the CPU-hours.
CPU request <= 10 machines: 99.71% of jobs, 35.1% of the CPU-hours.
CPU request <= 20 machines: 99.88% of jobs, 58.3% of the CPU-hours.
Effective duration <= 4.00 h: 99.20% of jobs, 15.6% of the CPU-hours.
Effective duration <= 12.00 h: 99.49% of jobs, 40.5% of the CPU-hours.
Effective duration <= 21.57 h: 99.61% of jobs, 56.5% of the CPU-hours.

### N3. Phase 0 type volume: mean real request of the type / hardware profile (H jobs at the recommended pair; same d_bar, so the volume ratio = the request ratio)

| resource | types with jobs | median real / profile | q0.10 | q0.90 | job-weighted median | total volume real / profile |
|---|---|---|---|---|---|---|
| CPU | 3080 | 1.005 | 0.808 | 1.246 | 1.008 | 1.008 |
| RAM | 3080 | 0.999 | 0.829 | 1.229 | 1.000 | 1.036 |

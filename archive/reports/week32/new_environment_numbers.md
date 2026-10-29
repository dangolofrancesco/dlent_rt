# Week 32: the v3 environment, all numbers

Config: configs/v3_mean_demand.yaml. Jobs in the file: 30,000.
Quota (p99 of H, machines): CPU 3.9312, RAM 2.8142. Rejected: 269 jobs (0.90%). Kept: 29,731. d_max = d_cap = 24 h; T = 428.8 h; |H| = 315.1 h, n_H = 10000.

P1. Grid filled from 2019-06-01T04:00 to 2019-06-03T00:00 UTC (45 hours). Jobs running at least partly in filled hours: 242 (0.81%); of them arriving in filled hours (so priced with a filled hour): 232. Their share of the total energy cost: running 0.25%, arriving 0.15%. (The model prices a job at its arrival hour.)

P4. H: lambda_hat = 31.738 jobs/h; mean r*d = 0.02234 CPU, 0.01343 RAM machine-hours/job; D_bar = 0.7089 CPU, 0.4262 RAM machines. Little's law: time-average occupancy of H with no capacity over H's span = 0.7015 CPU, 0.4201 RAM; gap -1.0% CPU, -1.4% RAM.

### N1. f scan: full month, FCFS, no prices (C = f x D_bar from H)

| f | C CPU | C RAM | % time demand >= 95% C (CPU) | (RAM) | (CPU or RAM) | FCFS jobs accepted | FCFS CPU-hours served | largest job / C (CPU) | largest job / C (RAM) |
|---|---|---|---|---|---|---|---|---|---|
| 0.10 | 0.0709 | 0.0426 | 90.5% | 91.1% | 93.8% | 53.9% | 5.6% | 55.44 | 64.78 |
| 0.25 | 0.1772 | 0.1065 | 66.0% | 67.6% | 72.7% | 81.6% | 12.9% | 22.18 | 25.91 |
| 0.50 | 0.3545 | 0.2131 | 44.8% | 47.4% | 50.9% | 92.5% | 20.9% | 11.09 | 12.96 |
| 0.75 | 0.5317 | 0.3196 | 35.4% | 36.5% | 39.1% | 95.9% | 26.9% | 7.39 | 8.64 |
| 1.00 | 0.7089 | 0.4262 | 31.3% | 30.2% | 33.2% | 97.8% | 29.1% | 5.54 | 6.48 |

Selection rule (fixed in advance): '% time demand >= 95% of C' (CPU or RAM) closest to the 25-50% band, ties to 35%. -> f = 1.00. Configured f = 1.0.
At the configured f: C = 0.7089 CPU, 0.4262 RAM machines; largest admitted request / C = 5.54 CPU, 6.48 RAM (flag if > 0.2: True).

### N2. Percentiles: by jobs (p..) and weighted by CPU-hours (w p..)

| quantity | p50 | p75 | p90 | p95 | p99 | p99.9 | max | w p50 | w p75 | w p90 | w p95 | w p99 | w p99.9 | w max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CPU request (machines), all | 0.008011 | 0.02747 | 0.06284 | 0.09964 | 2.581 | 21.77 | 175.4 | 11.48 | 29.46 | 175.4 | 175.4 | 175.4 | 175.4 | 175.4 |
| RAM request (machines), all | 0.006508 | 0.01028 | 0.02419 | 0.05638 | 1.74 | 15.53 | 99.12 | 7.961 | 20.19 | 99.12 | 99.12 | 99.12 | 99.12 | 99.12 |
| effective duration (h), all | 0.02194 | 0.06667 | 0.3161 | 0.7459 | 3.975 | 123.2 | 371.3 | 12.96 | 117.4 | 287.9 | 287.9 | 371.3 | 371.3 | 371.3 |
| CPU request x eff. duration, all | 0.0002321 | 0.001281 | 0.00587 | 0.02015 | 0.4977 | 20.64 | 1348 | 416.4 | 1162 | 1348 | 1348 | 1348 | 1348 | 1348 |
| CPU request, after cap + quota | 0.007843 | 0.02589 | 0.05706 | 0.09156 | 0.6726 | 2.67 | 3.931 | 0.9172 | 1.638 | 2.627 | 2.627 | 3.656 | 3.889 | 3.931 |
| RAM request, after cap + quota | 0.006508 | 0.009911 | 0.02252 | 0.04105 | 0.4053 | 1.908 | 2.761 | 0.4894 | 1.018 | 1.957 | 1.957 | 1.96 | 2.551 | 2.761 |
| duration after 24 h cap, after quota | 0.02187 | 0.06583 | 0.3108 | 0.7374 | 3.974 | 24 | 24 | 24 | 24 | 24 | 24 | 24 | 24 | 24 |
| CPU request x capped duration, after cap + quota | 0.0002848 | 0.001251 | 0.004972 | 0.01549 | 0.1513 | 2.582 | 63.05 | 4.721 | 34.71 | 63.05 | 63.05 | 63.05 | 63.05 | 63.05 |

Cap 24 h: 92 jobs truncated (0.31%); CPU-hours lost to truncation: 35.3% of all CPU-hours. Quota: 269 jobs rejected (0.90%), 71.5% of all CPU-hours. Work left after cap + quota: 7.8% of all CPU-hours.
1-minute duration floor (time.duration_floor_hours): 11496 jobs lifted (38.3%); CPU-hours added: 0.022%.

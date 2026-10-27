# Week 32 data audit: all numbers

Jobs: 29670. Window W: 2019-05-01 00:05:00 to 2019-06-01 00:09:19 = 744.1 h. Units: machines.

Thresholds: CPU q0.9 = 3.2198, RAM q0.9 = 1.9271 machines, duration q0.95 = 3.9733 h. policy = drop.

### T1a. What the outlier filters remove

| filter | jobs removed | share of jobs | share of CPU-hours | share of RAM-hours |
|---|---|---|---|---|
| CPU request > q0.90 | 2967 | 10.0% | 92.57% | 90.26% |
| RAM request > q0.90 | 2967 | 10.0% | 88.83% | 92.13% |
| either request > q0.90 (resource filter) | 3164 | 10.7% | 92.76% | 93.12% |
| duration > q0.95 = 3.97 h (duration filter) | 1481 | 5.0% | 98.04% | 97.84% |
| removed by both filters together | 4362 | 14.7% | 99.79% | 99.76% |

Check: handle_outliers keeps 25308 jobs (85.3%), same as above.

### T1b. Hardware snapping (32 profiles): change of the requests

| jobs | moved > 5% (CPU or RAM) | total CPU request | total RAM request | CPU-hours | RAM-hours |
|---|---|---|---|---|---|
| after filters (current model), n = 25308 | 95.1% | -3.2% | -3.6% | -1.2% | -6.5% |
| all jobs, n = 29670 | 95.2% | -8.1% | -9.2% | -49.7% | -44.2% |

### T2. Demand of running jobs (machines). '>X%' = level exceeded X% of the time

| case | CPU mean | CPU peak | CPU >10% | CPU >25% | CPU >50% | RAM mean | RAM peak | RAM >10% | RAM >25% | RAM >50% |
|---|---|---|---|---|---|---|---|---|---|---|
| all jobs, real durations | 1061.9 | 2018.1 | 1511.2 | 1242.7 | 967.0 | 629.0 | 1170.0 | 809.2 | 690.3 | 606.6 |

### T2b. The same after the outlier filters (machines)

| case | CPU mean | CPU peak | CPU >10% | CPU >25% | CPU >50% | RAM mean | RAM peak | RAM >10% | RAM >25% | RAM >50% |
|---|---|---|---|---|---|---|---|---|---|---|
| after filters, raw requests | 2.2 | 40.8 | 5.0 | 3.2 | 1.6 | 1.5 | 12.0 | 3.4 | 2.1 | 1.1 |
| after filters, snapped (what the model sees) | 2.2 | 30.6 | 4.9 | 3.0 | 1.6 | 1.4 | 10.5 | 3.2 | 2.0 | 1.0 |

d_max of the model after filters: 3.97 h; T = 398.0 h.

T (stream horizon of the code, all jobs, n0 = 10000) = 459.5 h. Code ceiling d_max <= 0.05 T = 23.0 h; abort if d_max/T > 0.1.

### T3. Duration cap: CPU-hours kept and demand (machines)

| d_max | d_max / T | CPU-hours kept | jobs cut | CPU mean | CPU peak | CPU >10% | CPU >25% | CPU >50% | RAM mean | RAM peak | RAM >10% | RAM >25% | RAM >50% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 h | 0.009 | 5.5% | 1417 | 58.9 | 827.5 | 120.1 | 66.8 | 33.3 | 42.0 | 600.3 | 76.7 | 45.9 | 23.4 |
| 24 h | 0.052 | 15.2% | 809 | 161.8 | 838.5 | 376.9 | 171.7 | 103.1 | 110.1 | 600.3 | 258.2 | 112.8 | 68.4 |
| 72 h | 0.157 | 29.1% | 647 | 308.8 | 994.4 | 788.3 | 397.2 | 205.7 | 194.1 | 737.7 | 566.2 | 260.2 | 124.8 |
| no cap | 1.619 | 100.0% | 0 | 1061.9 | 2018.1 | 1511.2 | 1242.7 | 967.0 | 629.0 | 1170.0 | 809.2 | 690.3 | 606.6 |

### T4. New rule: cluster = demand level exceeded X% of the time; FCFS result

| d_max | X | cluster CPU | cluster RAM | f CPU | f RAM | jobs larger than cluster | their CPU-hours | FCFS jobs accepted | FCFS CPU-hours served | time >= 95% full |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 h | 10% | 120.1 | 76.7 | 0.145 | 0.128 | 12 | 20.6% | 98.6% | 60.3% | 2.0% |
| 4 h | 25% | 66.8 | 45.9 | 0.081 | 0.076 | 30 | 28.4% | 97.2% | 44.2% | 3.2% |
| 4 h | 50% | 33.3 | 23.4 | 0.040 | 0.039 | 84 | 41.1% | 93.2% | 25.0% | 6.9% |
| 4 h | peak (f = 1) | 827.5 | 600.3 | 1.000 | 1.000 | 0 | 0.0% | 100.0% | 100.0% | 0.3% |
| 24 h | 10% | 376.9 | 258.2 | 0.450 | 0.430 | 3 | 5.4% | 99.5% | 74.6% | 2.8% |
| 24 h | 25% | 171.7 | 112.8 | 0.205 | 0.188 | 9 | 23.4% | 98.3% | 48.8% | 4.5% |
| 24 h | 50% | 103.1 | 68.4 | 0.123 | 0.114 | 14 | 25.5% | 97.6% | 35.2% | 6.7% |
| 24 h | peak (f = 1) | 838.5 | 600.3 | 1.000 | 1.000 | 0 | 0.0% | 100.0% | 100.0% | 1.0% |
| 72 h | 10% | 788.3 | 566.2 | 0.793 | 0.767 | 0 | 0.0% | 99.8% | 88.4% | 7.4% |
| 72 h | 25% | 397.2 | 260.2 | 0.399 | 0.353 | 3 | 2.9% | 99.2% | 60.7% | 8.7% |
| 72 h | 50% | 205.7 | 124.8 | 0.207 | 0.169 | 8 | 21.4% | 97.1% | 35.9% | 12.7% |
| 72 h | peak (f = 1) | 994.4 | 737.7 | 1.000 | 1.000 | 0 | 0.0% | 100.0% | 100.0% | 1.3% |
| no cap | 10% | 1511.2 | 809.2 | 0.749 | 0.692 | 0 | 0.0% | 99.8% | 97.1% | 13.0% |
| no cap | 25% | 1242.7 | 690.3 | 0.616 | 0.590 | 0 | 0.0% | 99.6% | 90.1% | 18.5% |
| no cap | 50% | 967.0 | 606.6 | 0.479 | 0.518 | 0 | 0.0% | 99.0% | 70.4% | 25.3% |
| no cap | peak (f = 1) | 2018.1 | 1170.0 | 1.000 | 1.000 | 0 | 0.0% | 100.0% | 100.0% | 2.4% |

### T5. Current regime, same FCFS (filtered jobs, snapped requests; n = 25308)

| case | cluster CPU | cluster RAM | f CPU (c / peak) | f RAM (c / peak) | jobs larger than cluster | their CPU-hours | FCFS jobs accepted | FCFS CPU-hours served | time >= 95% full |
|---|---|---|---|---|---|---|---|---|---|
| current rule at f = 1 (c = q0.70 level after filters) | 2.676 | 1.744 | 0.088 | 0.166 | 0 | 0.0% | 90.6% | 51.1% | 5.1% |
| current regime, f = 0.15 (regime_small_cluster.yaml) | 0.401 | 0.262 | 0.013 | 0.025 | 3838 | 84.0% | 70.5% | 8.1% | 5.0% |

### T6. collection_type of the 29,670 collections

| kind | collections | share | share of requested CPU | share of requested RAM |
|---|---|---|---|---|
| job (collection_type 0) | 29353 | 98.9% | 99.6% | 99.6% |
| alloc set (collection_type 1) | 317 | 1.1% | 0.4% | 0.4% |
| job that runs inside an alloc set | 703 | 2.4% | 0.6% | 0.4% |
| not found in collection_events | 0 | 0.0% | 0.0% | 0.0% |

### T7. Number of instances per job

| instances per job | jobs | share of jobs | share of requested CPU |
|---|---|---|---|
| 1 | 11710 | 39.5% | 0.3% |
| 2-10 | 6675 | 22.5% | 1.0% |
| 11-100 | 6203 | 20.9% | 6.0% |
| 101-1000 | 4129 | 13.9% | 38.3% |
| > 1000 | 953 | 3.2% | 54.4% |

Instances per job: median 3, q0.75 47, q0.90 301, q0.99 2489, max 48077. Event rows per instance: median 6.0.
Step-1 sampling: inclusion chance = 1 - 0.95^(event rows); in our sample it is 53.7% for the median job, 0.0% of jobs have a chance < 20%. Mean request per job: 1.363 (sample) vs 0.619 (re-weighted by 1 / inclusion chance) machines CPU.

T8 (task 6). Jobs in both files: 7136. Ratio 300k / 30k request (CPU): q0.10 2.01, median 5.05, q0.90 22.74. Event rows per instance on the same jobs: q0.10 5.00, median 6.00, q0.90 14.16. Spearman(ratio, rows per instance) = 0.558; share with ratio within 10% of rows per instance: 27.5%; ratio <= rows per instance: 67.2%.

T9 (8b). raw_events_30k: 30000 collections; raw_usage_30k: 29670; lost in the inner merge: 330 (all have no usage row).
Lost: 330; alloc sets 5; never scheduled 293; last collection event: {'KILL': 305, 'SUBMIT': 6, 'LOST': 5, 'ENABLE': 5, 'FINISH': 4, 'QUEUE': 3, 'SCHEDULE': 2}.

### T10 (8a). Instances per job: our sample vs a fair sample

| set | collections | median instances | mean instances | share with 1 instance | share with > 100 instances |
|---|---|---|---|---|---|
| all of cell a | 5213668 | 1 | 15.4 | 78.0% | 1.7% |
| fair random sample (29670 ids) | 29670 | 1 | 13.5 | 78.0% | 1.6% |
| our sample | 29670 | 3 | 135.5 | 39.5% | 17.1% |

Cell a: 5212740 collections in collection_events (5176566 jobs, 36174 alloc sets); 5213668 collections in instance_events. Our sample = 0.57% of the collections and 5.0% of the instances.

T11 (8f). Raw priorities in 351-359 (code: tier 5, doc: production): 0; >= 360: 39.

### T12 (8e). Jobs cut at the borders of the trace

| jobs | count | share of jobs | share of CPU-hours | share of RAM-hours |
|---|---|---|---|---|
| started before the trace (start < 600 s) | 430 | 1.45% | 36.3% | 45.8% |
| still running at the end (last window ends < 5 min before the end) | 460 | 1.55% | 30.0% | 33.8% |
| either | 600 | 2.02% | 47.5% | 53.7% |

T13 (8g). actual_cpu_hours / (request x duration): median 1.60; share > 1: 59.9%.

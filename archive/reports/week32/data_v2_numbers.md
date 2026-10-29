# Week 32 step 2: numbers of the new data

Sample: 30000 jobs, 404384 instances. Population (cell a, type 0, with usage): 5,146,808 jobs.

### T1. Instances per job (instances that ran, unless said otherwise)

| set | jobs | with 1 instance | mean instances | median | with > 100 |
|---|---|---|---|---|---|
| our new sample | 30,000 | 79.9% | 13.48 | 1 | 1.61% |
| its population (no border, with a request) | 5,125,385 | 79.5% | 14.05 | 1 | 1.64% |
| all jobs of cell a with usage (border included) | 5,146,808 | 79.4% | 14.41 | 1 | 1.66% |
| all collections of cell a, instance_events (step 1) | 5,213,668 | 78.0% | 15.43 | 1 | 1.72% |
| old sample (batch_may2019_30k), instance_events | 29,670 | 39.5% | 135.46 | 3 | 17.13% |

Border jobs in the population: 21,423 (0.42%), holding 3.0% of the summed CPU request. Non-border jobs with no request: 0.
Share we hold: 0.583% of the jobs with usage, 0.585% of the eligible jobs; 0.554% of their summed CPU request (0.572% of the eligible jobs'), 0.553% of RAM.
Check: the sample = the 30000 eligible jobs with the smallest fingerprint: True.
Request changed (an instance shows > 1 request value): 100.0% of jobs; jobs with UPDATE events: 95.2%.

T2a. Per job, SUM / real peak (CPU): equal (ratio 1) for 99.5% of jobs; among multi-instance jobs (6022): median 1.00, q0.90 1.00, q0.99 1.03, max 2.3.

### T2b. SUM versus the real peak and the real instance spans (CPU, machines)

| what | SUM | real peak | real (each instance its own span) | SUM / real |
|---|---|---|---|---|
| total request (sum over jobs) | 4232.4 | 4201.7 | - | 1.007 |
| CPU-hours (request x job span) | 15,298 | 15,285 | 7,341 | 2.084 |
| concurrent CPU demand, mean | 20.57 | 20.55 | 9.87 | 2.084 |
| concurrent CPU demand, peak | 209.79 | 209.60 | 178.47 | 1.176 |
| concurrent CPU demand, level exceeded 25% | 27.32 | 27.30 | 15.38 | 1.777 |

### T3. d_max: horizon, jobs cut, CPU-hours kept

| d_max asked | d_max used | T | d_max / T | jobs cut | CPU-hours kept |
|---|---|---|---|---|---|
| 4 h | 4.00 h | 431.5 h | 0.0093 | 255 | 23.1% |
| 12 h | 12.00 h | 431.5 h | 0.0278 | 164 | 42.1% |
| 23 h | 21.57 h | 431.5 h | 0.0500 | 120 | 49.2% |

### T4. Cluster by the 25% rule (machines; cores and GB under the declared 64 / 256)

| d_max | cluster CPU (machines) | cluster RAM (machines) | CPU cores (assumed 64 / machine) | RAM GB (assumed 256 / machine) | f CPU = c / peak | f RAM | jobs larger than cluster | their CPU-hours |
|---|---|---|---|---|---|---|---|---|
| 4 h | 2.42 | 1.68 | 155 | 430 | 0.013 | 0.016 | 341 | 91.2% |
| 12 h | 7.69 | 5.35 | 492 | 1,371 | 0.037 | 0.041 | 139 | 85.0% |
| 23 h | 9.09 | 8.23 | 582 | 2,106 | 0.043 | 0.063 | 128 | 83.9% |

### T5. First come, first served (admit a job if it fits now)

| d_max | cluster | jobs accepted | CPU-hours served | time >= 95% full |
|---|---|---|---|---|
| 4 h | 25% rule | 98.4% | 6.9% | 1.1% |
| 4 h | f = 1 (cluster = peak) | 100.0% | 100.0% | 0.5% |
| 12 h | 25% rule | 99.0% | 12.5% | 1.3% |
| 12 h | f = 1 (cluster = peak) | 100.0% | 100.0% | 0.1% |
| 23 h | 25% rule | 99.4% | 14.8% | 0.8% |
| 23 h | f = 1 (cluster = peak) | 100.0% | 100.0% | 0.1% |

Grid file ends at 2019-06-01T03:00:00.000000 UTC; jobs arriving after its last hour: 233 (they get the last hour's price, energy.GridData.lookup clips the index).
xi on the data (max a_max / c) at d_max 23 h: 19.30 (forced xi_override = 0.001).

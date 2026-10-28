# Week 32: v4 (uniform 300k sample), all numbers

300k file: 300,000 jobs; 30k file: 30,000; eligible population (cell a, jobs that ran, no border, with a request): 5,125,385; sample rate 5.8532%.

### N1. 300k data checks

| check | value | result |
|---|---|---|
| single-instance share vs the eligible cell (within 1 point) | 79.5% vs 79.5% | PASS |
| priority-tier mix vs the 30k sample (max gap < 1 point; the cell's priorities are not fetched) | tier 1 1.6% / 1.8%, tier 2 8.9% / 8.9%, tier 3 3.8% / 3.7%, tier 4 85.6% / 85.5%, tier 5 0.2% / 0.2% | PASS |
| arrivals per hour vs the cell x the sample rate (correlation > 0.95) | corr 0.9931; mean 403.2 vs 403.2 per hour | PASS |
| the 30k jobs inside the 300k, identical rows (integers / text exact; floats to 1e-12 relative: the instance sums run in a different order) | 30,000 of 30,000 found; exact columns equal: True; largest float gap 2.4e-14 | PASS |
| SUM of first requests = real peak for >= 99% of jobs | 99.53% | PASS |
| effective duration = real CPU-hours / request: finite, >= 0, <= span; no NaN | min 2.78e-04 h | PASS |
| time base UTC: first arrival at or after 2019-05-01 07:00 UTC | 2019-05-01 07:00:10+00:00 | PASS |
| no border jobs | first start >= 600 s and last end <= end - 300 s for all: True | PASS |
| distributions close to the 30k: each 30k percentile p50-p99 inside the 95% band of 30k-sized subsamples of the 300k | 20 of 20 inside | PASS |

### N2. Percentiles, 300k / 30k (p99.9 and max are noisy on 30k)

| quantity | p50 | p75 | p90 | p95 | p99 | p99.9 | max | 30k inside the 95% band (p50-p99) |
|---|---|---|---|---|---|---|---|---|
| CPU request | 0.008011 / 0.008011 | 0.02756 / 0.02747 | 0.06226 / 0.06284 | 0.09985 / 0.09964 | 2.511 / 2.581 | 23.03 / 21.77 | 313.1 / 175.4 | 5 of 5 |
| RAM request | 0.006508 / 0.006508 | 0.01025 / 0.01028 | 0.02365 / 0.02419 | 0.05582 / 0.05638 | 1.654 / 1.74 | 15.74 / 15.53 | 1470 / 99.12 | 5 of 5 |
| effective duration | 0.02194 / 0.02194 | 0.06694 / 0.06667 | 0.315 / 0.3161 | 0.7547 / 0.7459 | 3.975 / 3.975 | 73.87 / 123.2 | 540.4 / 371.3 | 5 of 5 |
| CPU request x eff. duration | 0.0002342 / 0.0002321 | 0.001292 / 0.001281 | 0.006034 / 0.00587 | 0.02047 / 0.02015 | 0.4218 / 0.4977 | 18.14 / 20.64 | 3.113e+04 / 1348 | 5 of 5 |

### N3. Choice of H (f = 1.0; weeks = 7-day blocks from the first arrival, week 5 = last 2-3 days; D_bar of a week / the month = its work / its length, same quota)

| H | jobs in H | H span (h) | Q CPU / RAM | D_bar of H, CPU / RAM | week 1 CPU | week 2 CPU | week 3 CPU | week 4 CPU | week 5 CPU | month CPU / RAM | H vs month (CPU) | H vs month (RAM) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| first 10% of jobs | 30000 | 109.8 | 0.4988 / 0.2983 | 2.481 / 1.487 | 2.464 | 2.528 | 2.415 | 2.322 | 2.819 | 2.470 / 1.653 | +0.5% | -10.1% |
| first 7 days | 46205 | 168.0 | 0.5088 / 0.3318 | 2.542 / 1.658 | 2.542 | 2.748 | 2.436 | 2.378 | 2.896 | 2.562 / 1.724 | -0.8% | -3.8% |

H closest to the month (sum of the CPU and RAM gaps): first 7 days ({'first 10% of jobs': 10.6, 'first 7 days': 4.6}).

### N4. Quota fixed point Q = 0.2 x C (H = 7 days, f = 1.0)

| iteration | Q CPU | Q RAM | jobs kept | jobs in H | D_bar CPU | D_bar RAM | next Q CPU | next Q RAM | change CPU / RAM |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 4.1370 | 2.8384 | 297,445 | 47,380 | 11.1995 | 7.4074 | 2.2399 | 1.4815 | 45.9% / 47.8% |
| 1 | 2.2399 | 1.4815 | 296,568 | 47,147 | 6.9759 | 4.6711 | 1.3952 | 0.9342 | 37.7% / 36.9% |
| 2 | 1.3952 | 0.9342 | 295,714 | 46,861 | 4.4646 | 3.0173 | 0.8929 | 0.6035 | 36.0% / 35.4% |
| 3 | 0.8929 | 0.6035 | 294,858 | 46,628 | 3.4820 | 2.3178 | 0.6964 | 0.4636 | 22.0% / 23.2% |
| 4 | 0.6964 | 0.4636 | 294,327 | 46,435 | 3.1402 | 2.0533 | 0.6280 | 0.4107 | 9.8% / 11.4% |
| 5 | 0.6280 | 0.4107 | 294,069 | 46,352 | 2.6654 | 1.7506 | 0.5331 | 0.3501 | 15.1% / 14.7% |
| 6 | 0.5331 | 0.3501 | 293,630 | 46,245 | 2.5575 | 1.6675 | 0.5115 | 0.3335 | 4.0% / 4.7% |
| 7 | 0.5115 | 0.3335 | 293,491 | 46,210 | 2.5441 | 1.6592 | 0.5088 | 0.3318 | 0.5% / 0.5% |

### N5. f scan (H = 7 days; full month, FCFS, no prices; capped durations)

| f | C CPU | C RAM | Q CPU | Q RAM | iterations | % time demand >= 95% C (CPU or RAM) | FCFS jobs accepted | FCFS CPU-hours served | largest admitted / C CPU | largest admitted / C RAM | quota keeps: jobs | quota keeps: CPU-hours |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.10 | collapsed: Q -> 0 (no positive fixed point) | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| 0.25 | 0.203 | 0.127 | 0.0406 | 0.0253 | 13.0 | 100.0% | 39.8% | 21.6% | 0.200 | 0.200 | 81.3% | 1.0% |
| 0.50 | 0.626 | 0.425 | 0.1252 | 0.0850 | 12.0 | 99.7% | 65.0% | 41.0% | 0.200 | 0.200 | 95.4% | 1.6% |
| 0.75 | 1.661 | 1.083 | 0.3322 | 0.2166 | 7.0 | 87.1% | 88.8% | 63.4% | 0.200 | 0.200 | 97.3% | 2.5% |
| 1.00 | 2.542 | 1.658 | 0.5088 | 0.3318 | 8.0 | 60.4% | 95.6% | 75.2% | 0.200 | 0.200 | 97.8% | 3.0% |

Selection rule (fixed in advance): '% time demand >= 95% of C' (CPU or RAM) closest to 25-50%, ties to 35%. -> f = 1.00. Configured f = 1.0.
At f = 1.00: T = 575.9 h, H = 46,205 jobs, stream = 247,264, d_max = 24 h; quota rejected 6,531 jobs.

### N6. s_C trace on v3 (30k, before the priority fix; scripts in this report's session, values printed by Phase 0)

norm_info: q = 0.90, Q_cv = 44.272, Q_|Pi| = 9.774, D_q = 27.023 $, K_q = 0.003538 kg, s_C = 7,638.47 $/kg.
p-weighted per-type quantiles q50 / q90 / q99: CV 3.908 / 44.27 / 869.3 $; |Pi| 0.680 / 9.774 / 204.1 $;
K 8.1e-05 / 0.003538 / 0.1170 kg.
H jobs: carbon intensity 0.325-0.475 kg/kWh; energy q50 / q90 0.000209 / 0.00832 kWh; K q50 / q90 8.53e-05 / 0.00339 kg;
56.5% of H jobs at the 1-minute floor; K q90 without them 0.0140 kg.
Worked example (median-carbon H job): CPU 0.008896, RAM 0.004578 machines; priority tier 4; scheduling class 1;
d = 0.01833 h; power 0.010935 kW; energy 0.00020047 kWh; carbon intensity 0.4258 kg/kWh; carbon 8.536e-05 kg;
carbon $ at s_C 0.652; carbon $ at 0.19 $/kg 1.62e-05; price 0.022735 $/kWh; C_elec 4.56e-06 $; cost base 1.0472 $;
bid v 0.8748 $; customer value (q x v) 3.499 $.

### N7. Priority fix (collection_events instead of ANY_VALUE over instance_events)

300k: jobs with > 1 collection priority: 39. 30k: priority changed for 5,573 jobs, tier for 5,527, scheduling class for
3,937. Tiers 1-5, new / old: 529 / 6,055; 2,656 / 2,009; 1,116 / 885; 25,635 / 20,992; 64 / 59.
Default config (real trace) arrays that change: H/E, S/E, st/p, st/r_tilde, st/v, st/d_bar, st/A_type,
st/sc (lam_star, s_C, gamma, xi), S/v, S/cost, S/C_elec. Unchanged: capacity, d_max, T, H span, xi(data), requests,
durations, steps, hw_idx, prices.

### N8. Phase 0 (current u0, gamma1, gamma2)

| config | types with jobs in H | tau | types admitted | LP fill CPU / RAM | positive types / mass | s_C $/kg | split all CV / profit / carbon $/h | split admitted |
|---|---|---|---|---|---|---|---|---|
| v4 (300k, H 7 days) | 5,845 of 20,480 | 0, 0 | 3,595 | 26.0% / 34.7% | 61.5% / 80.7% | 14,342 | 20,134 / 2,292 / -19,985 | 16,514 / 2,443 / -5,438 |
| v3 (30k), fixed priorities | 2,724 | 0, 0 | 1,859 | 9.9% / 13.1% | 68.2% / 84.3% | 10,743 | 1,918 / 199 / -3,765 | 1,621 / 239 / -426 |
| v3 (30k), before the fix (new_environment.md) | 3,020 | 0, 0 | 2,040 | 9.0% / 12.0% | 67.5% / 81.3% | 7,638 | 1,589 / 178 / -2,624 | 1,340 / 209 / -270 |
| default (30k), fixed priorities | 2,703 | 0, 0 | 1,648 | 22.0% / 34.5% | 61.0% / 74.9% | 22,879 | 1,685 / 98 / -1,290 | 1,308 / 199 / -476 |

v4: lambda 275.0 jobs/h (median type 0.0119/h); C = 2.542 / 1.658 machines; data xi 0.200; gamma (data) 86,400; Phase 0
runtime 0.1 s; whole notebook ~20 s; peak kernel memory 0.62 GB. BigQuery this step: 300k selection 9.6 GB, priority
probe (instance_events, discarded) 96.2 GB, priority fix (collection_events) 0.8 GB + 0.8 GB collection_events probe.

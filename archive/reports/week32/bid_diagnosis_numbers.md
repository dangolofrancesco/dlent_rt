# Week 32: bid-size diagnosis on v4, all numbers

Config configs/v4_300k.yaml (f = 1.0); s_C = 14342.46 $/kg; jobs H 46,205, stream 247,264; u0 = 23.926, mode log, gamma1 = 0.0332, gamma2 = 0.00445, sigma = 0.3, machine 64 cores / 256 GB.

### N1. Per-job reward terms by decile of CPU request x capped duration (H + stream, v4)

| decile | r*d range (machine-h) | jobs | mean CPU | mean d (h) | mean bid $ | mean customer value $ | mean profit $ | mean carbon $ | mean reward $ | reward per machine-h $ | % positive reward | share of CPU-hours |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 - 5.7e-05 | 28773 | 0.0020 | 0.020 | 1.122 | 4.374 | 0.2803 | 0.285 | 4.370 | 1.197e+05 | 96.8% | 0.06% |
| 2 | 5.7e-05 - 7.8e-05 | 29896 | 0.0037 | 0.019 | 1.049 | 4.102 | 0.2439 | 0.497 | 3.849 | 5.837e+04 | 96.2% | 0.10% |
| 3 | 7.8e-05 - 0.00012 | 29356 | 0.0051 | 0.022 | 1.245 | 4.827 | 0.2775 | 0.745 | 4.360 | 4.374e+04 | 95.6% | 0.15% |
| 4 | 0.00012 - 0.00016 | 28936 | 0.0073 | 0.023 | 1.257 | 4.865 | 0.1515 | 1.012 | 4.005 | 2.872e+04 | 94.7% | 0.21% |
| 5 | 0.00016 - 0.00027 | 29741 | 0.0088 | 0.035 | 1.897 | 7.231 | 0.5657 | 1.554 | 6.242 | 2.952e+04 | 93.1% | 0.33% |
| 6 | 0.00027 - 0.00051 | 29345 | 0.0144 | 0.051 | 2.740 | 10.199 | 0.6696 | 2.816 | 8.053 | 2.094e+04 | 86.0% | 0.59% |
| 7 | 0.00051 - 0.00085 | 29381 | 0.0256 | 0.060 | 3.253 | 12.023 | 0.8333 | 4.757 | 8.100 | 1.226e+04 | 64.9% | 1.02% |
| 8 | 0.00085 - 0.0016 | 29347 | 0.0338 | 0.110 | 5.851 | 21.407 | -2.7511 | 8.762 | 9.894 | 8224 | 50.9% | 1.85% |
| 9 | 0.0016 - 0.0043 | 29347 | 0.0439 | 0.234 | 12.581 | 46.886 | 3.0276 | 19.093 | 30.821 | 1.196e+04 | 46.4% | 3.97% |
| 10 | 0.0043 - 11 | 29347 | 0.0650 | 2.010 | 107.938 | 393.196 | 38.6256 | 456.122 | -24.300 | -408 | 48.0% | 91.71% |

All jobs: positive reward 77.3%; CPU-hours of jobs with positive reward 25.8%. Mean bid / cost base (markup) 0.9992; share of the cost base from the base utility (median over jobs) 0.9996.

### N2. Two worked examples

| term | median job (by r*d) | job nearest the CPU quota Q = 0.509 |
|---|---|---|
| CPU (machines) | 0.0081 | 0.5086 |
| RAM (machines) | 0.0234 | 0.2905 |
| priority tier | 4 | 2 |
| duration (h) | 0.0339 | 0.1927 |
| r*d (machine-h) | 0.0002746 | 0.09802 |
| base utility $/h u0(1+ln q) | 57.094 | 40.510 |
| size term $/h g1 cores + g2 GB | 0.0439 | 1.4115 |
| cost base $ (rate x d) | 1.9364 | 8.0797 |
| bid $ | 1.9876 | 7.4245 |
| phi $ | 1.3630 | 4.3131 |
| customer value $ (q x bid) | 7.9504 | 14.8491 |
| energy kWh | 0.0004019 | 0.109 |
| carbon kg | 0.0001586 | 0.04761 |
| carbon $ (s_C x kg) | 2.2741 | 682.7978 |
| profit $ (phi - elec) | 1.3630 | 4.3106 |
| reward $ | 7.0393 | -663.6381 |

N3. LP load p*v / C by reward sign (types with jobs in H): positive types [0.2603, 0.3474] (CPU, RAM), negative types [0.7403, 0.6542]; all [1.0006, 1.0016]. Mass p: positive 80.7%. Mean type r*d (machine-h, p-weighted): positive 0.002983, negative 0.03538. tau = [0.0, 0.0].

### N4. Extended f scan (v4: H = 7 days, quota = 0.2 x C; full month, FCFS, no prices)

| f | C CPU | C RAM | Q CPU | Q RAM | iterations | % time demand >= 95% C (CPU or RAM) | FCFS jobs accepted | FCFS CPU-hours served | largest admitted / C CPU | largest admitted / C RAM | jobs kept by the quota | CPU-hours kept by the quota |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.75 | 1.661 | 1.083 | 0.3322 | 0.2166 | 7 | 87.1% | 88.8% | 63.4% | 0.200 | 0.200 | 97.3% | 2.5% |
| 1.00 | 2.542 | 1.658 | 0.5088 | 0.3318 | 8 | 60.4% | 95.6% | 75.2% | 0.200 | 0.200 | 97.8% | 3.0% |
| 1.25 | 4.343 | 2.891 | 0.8686 | 0.5782 | 8 | 30.2% | 98.6% | 85.7% | 0.200 | 0.199 | 98.3% | 3.8% |
| 1.50 | 7.220 | 5.004 | 1.4456 | 1.0018 | 12 | 10.7% | 99.6% | 92.5% | 0.200 | 0.200 | 98.6% | 4.9% |
| 2.00 | 22.506 | 14.887 | 4.4995 | 2.9762 | 2 | 6.0% | 99.9% | 93.5% | 0.200 | 0.200 | 99.2% | 9.7% |
| 2.50 | 41.874 | 25.536 | 8.3744 | 5.1069 | 4 | 3.1% | 100.0% | 97.0% | 0.198 | 0.200 | 99.5% | 14.6% |
| 3.00 | 51.404 | 31.431 | 10.2577 | 6.2747 | 4 | 0.7% | 100.0% | 99.1% | 0.199 | 0.200 | 99.6% | 15.7% |
| 4.00 | 79.267 | 49.513 | 15.8533 | 9.9027 | 5 | 0.0% | 100.0% | 100.0% | 0.200 | 0.200 | 99.8% | 20.1% |
| 5.00 | 102.101 | 64.292 | 20.4090 | 12.8512 | 4 | 0.0% | 100.0% | 100.0% | 0.199 | 0.200 | 99.8% | 24.1% |

Fixed rule ('% time demand >= 95% of C' closest to 25-50%, ties to 35%) -> f = 1.25 with 30.2%. Not set in the config (configured f = 1.0).

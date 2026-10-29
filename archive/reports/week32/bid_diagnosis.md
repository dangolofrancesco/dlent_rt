# Week 32: how the bid depends on job size

Diagnosis only. u0, γ1, γ2 and s_C were not recalibrated, and no config default changed. No online model was run.
Raw numbers are in `bid_diagnosis_numbers.md`, from `scripts/week32/bid_diagnosis.py`. Config: `configs/v4_300k.yaml`.

## 1. What we checked

* The exact bid formula in the code: does the bid grow with the request, with the duration, or both?
* The reward of the v4 jobs, by decile of CPU request × duration (r·d).
* Two worked examples: the median job, and a job at the quota.
* Why the LP fills only 26–35% of the cluster, with τ = 0.
* A larger f scan, with the same fixed selection rule.

## 2. The bid formula in the code

`dlent_rt/bid.py`, `synthesize_lognormal`, lines 134–150:

```python
cpu_cores = cpu_norm * params.machine_cores                     # line 139  (64 cores per machine, declared)
ram_gb = ram_norm * params.machine_ram_gb                       # line 140  (256 GB per machine, declared)
base_utility = compute_base_utility(priority, u0, mode)         # lines 141-143
cost_base = (base_utility + gamma1 * cpu_cores + gamma2 * ram_gb) * duration_hours   # lines 144-146
mu = np.log(cost_base) - sigma**2 / 2                           # line 148
v_rate = rng.lognormal(mean=mu, sigma=sigma)                    # line 150  (the bid, $)
```

`compute_base_utility` (lines 91–110), mode "log": base utility = u0 × (1 + ln q). Here u0 = 23.926 $/h and q is the
priority tier (1 to 5).

The reward terms come from `dlent_rt/reward.py`, `job_terms`, lines 59–74:

* customer value = q × bid;
* φ = cost base × φ_markup(bid / cost base), the virtual value of the known lognormal markup;
* profit = φ − electricity cost;
* carbon $ = s_C × kg;
* reward = customer value + profit − carbon $ (all λ = 1).

**What the bid scales with.** It scales with the **duration**, always. It scales with the **request** only through
γ1 = 0.0332 $/core-hour and γ2 = 0.00445 $/GB-hour. The size term is 3.26 $/h per machine (CPU and RAM). The base
utility is 23.9 to 62.4 $/h per job, whatever its size. The size term reaches the base utility only at 7 to 19
machines. In v4 every job is at most 0.5 machine, so the base utility makes up 99.96% of the median job's cost base.
The noise is lognormal with σ = 0.3; the mean bid equals the cost base (measured markup 0.999).

**Mismatch with the thesis (flagged, not fixed).** The thesis says the bid is lognormal "over the requested
resources". In the code, the bid is lognormal around (a size-independent hourly utility + a small size term) ×
duration. For our jobs it is, in practice, **a duration-based bid that does not depend on size**. (The thesis text
is not in the repository; this uses the wording you quoted.)

## 3. Reward by size decile (v4, H + stream, 293,469 jobs)

Deciles of CPU request × capped duration. Every value is a mean per job, at the current s_C = 14,342 $/kg.

| decile | mean CPU (machines) | mean duration (h) | bid $ | customer value $ | profit $ | carbon $ | reward $ | reward per machine-hour $ | jobs with reward > 0 | share of CPU-hours |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.002 | 0.020 | 1.12 | 4.37 | 0.28 | 0.29 | 4.37 | 119,700 | 96.8% | 0.06% |
| 2 | 0.004 | 0.019 | 1.05 | 4.10 | 0.24 | 0.50 | 3.85 | 58,370 | 96.2% | 0.10% |
| 3 | 0.005 | 0.022 | 1.25 | 4.83 | 0.28 | 0.75 | 4.36 | 43,740 | 95.6% | 0.15% |
| 4 | 0.007 | 0.023 | 1.26 | 4.87 | 0.15 | 1.01 | 4.01 | 28,720 | 94.7% | 0.21% |
| 5 | 0.009 | 0.035 | 1.90 | 7.23 | 0.57 | 1.55 | 6.24 | 29,520 | 93.1% | 0.33% |
| 6 | 0.014 | 0.051 | 2.74 | 10.20 | 0.67 | 2.82 | 8.05 | 20,940 | 86.0% | 0.59% |
| 7 | 0.026 | 0.060 | 3.25 | 12.02 | 0.83 | 4.76 | 8.10 | 12,260 | 64.9% | 1.02% |
| 8 | 0.034 | 0.110 | 5.85 | 21.41 | −2.75 | 8.76 | 9.89 | 8,224 | 50.9% | 1.85% |
| 9 | 0.044 | 0.234 | 12.58 | 46.89 | 3.03 | 19.09 | 30.82 | 11,960 | 46.4% | 3.97% |
| 10 | 0.065 | 2.010 | 107.94 | 393.20 | 38.63 | 456.12 | **−24.30** | **−408** | 48.0% | **91.71%** |

* **The bid grows with the duration, not the size.** From decile 1 to 10 the duration grows 100×, and so does the
  bid. The size grows only 32×.
* **Carbon grows with both size and duration**, so it grows faster than the bid. In decile 10 carbon is larger than
  customer value.
* **The reward per machine-hour falls from about 120,000 $ to −408 $.** The smallest jobs are by far the best per
  unit of capacity.
* **Over all jobs,** 77.3% have a positive reward, but they hold only 25.8% of the CPU-hours.

**Two worked examples**

| | median job | job at the quota |
|---|---|---|
| CPU / RAM (machines) | 0.0081 / 0.0234 | 0.509 / 0.291 |
| priority tier, duration | 4, 2.0 min | 2, 11.6 min |
| base utility $/h | 57.09 | 40.51 |
| size term $/h | 0.044 | 1.41 |
| cost base $ | 1.94 | 8.08 |
| bid $ | 1.99 | 7.42 |
| customer value $ | 7.95 | 14.85 |
| profit $ | 1.36 | 4.31 |
| carbon kg / $ | 0.00016 / 2.27 | 0.048 / 682.80 |
| **reward $** | **+7.04** | **−663.64** |

The job at the quota asks for 63 times the CPU of the median job. Its bid is only 3.7 times larger, but its carbon
cost is 300 times larger. Its reward is very negative.

## 4. Why τ = 0

The LP admits every type with a positive reward, fully. Those types are small: their mean r·d is 0.003 machine-hours.
Together they use only 26% of C on CPU and 35% on RAM. The types with a negative reward are 10 times larger (mean
r·d 0.035) and carry the rest of the load: 74% on CPU and 65% on RAM. The LP refuses them because carbon at
s_C = 14,342 $/kg outweighs their bid. Since C was set to the mean demand of **all** types, a third of it stays used
and two thirds stay empty. Capacity never binds, so τ = 0.

## 5. Extended f scan

v4 (H = 7 days, quota = 0.2 × C, recomputed for each f). The whole month, first come first served, no prices.

| f | C CPU / RAM | Q CPU | time with demand ≥ 95% of C | jobs accepted | CPU-hours served | largest job / C | jobs kept by the quota | CPU-hours kept by the quota |
|---|---|---|---|---|---|---|---|---|
| 0.75 | 1.66 / 1.08 | 0.332 | 87.1% | 88.8% | 63.4% | 0.20 | 97.3% | 2.5% |
| 1.00 | 2.54 / 1.66 | 0.509 | 60.4% | 95.6% | 75.2% | 0.20 | 97.8% | 3.0% |
| **1.25** | **4.34 / 2.89** | **0.869** | **30.2%** | **98.6%** | **85.7%** | **0.20** | **98.3%** | **3.8%** |
| 1.50 | 7.22 / 5.00 | 1.446 | 10.7% | 99.6% | 92.5% | 0.20 | 98.6% | 4.9% |
| 2.00 | 22.51 / 14.89 | 4.500 | 6.0% | 99.9% | 93.5% | 0.20 | 99.2% | 9.7% |
| 2.50 | 41.87 / 25.54 | 8.374 | 3.1% | 100.0% | 97.0% | 0.20 | 99.5% | 14.6% |
| 3.00 | 51.40 / 31.43 | 10.258 | 0.7% | 100.0% | 99.1% | 0.20 | 99.6% | 15.7% |
| 4.00 | 79.27 / 49.51 | 15.853 | 0.0% | 100.0% | 100.0% | 0.20 | 99.8% | 20.1% |
| 5.00 | 102.10 / 64.29 | 20.409 | 0.0% | 100.0% | 100.0% | 0.20 | 99.8% | 24.1% |

**The fixed rule picks f = 1.25, with 30.2%**, inside the 25–50% band. It is not set in the config (v4 keeps
f = 1.00).

* Because the quota follows C, a larger f lets in larger jobs, which raises D̄ and C again. So C grows much faster
  than f: from 2.5 machines at f = 1 to 102 machines at f = 5.
* Even at f = 1.25, the quota keeps 98.3% of the jobs but only 3.8% of the CPU-hours.

## 6. Open decisions

1. **The bid model.** Should the bid scale with the requested resources, as the thesis says? Today it is a
   per-job hourly utility × duration, with almost no size term. My recommendation: make the size term the main term
   (or scale u0 by the request), then recalibrate. This is a design change; I did not make it.
2. **The carbon map.** With s_C = 14,342 $/kg, carbon decides the sign of the reward for the large jobs, and so
   decides what the LP admits. My recommendation: decide the bid model first, then recalibrate u0 and s_C together.
3. **f.** The rule now picks 1.25. My recommendation: set it only after the bid model is decided, because τ = 0
   whatever f is, as long as the large types have a negative reward.
4. **The quota.** 0.2 × C keeps under 4% of the work at f = 1.25. My recommendation: if the large jobs are to be
   studied, loosen the quota share, or keep the 99th percentile and accept largest job / C > 0.2.

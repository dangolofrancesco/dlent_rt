# Week 32: a size-based bid (v5)

This step changes Phase 0, but only behind a config switch (`bid.mode`). `configs/default.yaml` is unchanged. With
the default config, the data path and Phase 0 give identical numbers (26 arrays checked), and the snapshots do not
change. No online model was run. u0, γ1, γ2 and s_C were not recalibrated.

* Raw numbers: `v5_size_bid_numbers.md`.
* Script: `scripts/week32/v5_size_bid.py`.
* Config: `configs/v5_size_bid.yaml`.
* Phase 0: `notebooks/01_phase0_diagnostics.ipynb`, run on v5.

## 1. What changed and why

| what | why |
|---|---|
| New `bid.mode: size`. The old mode, `base_utility`, stays the default. | In the old bid, a size-independent hourly utility (u0) made up 99.96% of the median cost base, so the bid ignored the requested resources (`bid_diagnosis.md`). |
| New config `configs/v5_size_bid.yaml` = v4 + `bid.mode: size` + f = 1.25. | f = 1.25 is what the fixed rule picked in the extended scan. The quota stays 0.2 × C, with its fixed point. |
| Tests: `tests/test_size_bid.py`. | They check the cost base on a hand-made job, linear scaling, and that the old mode is unchanged. |

## 2. The new bid formula, and the power model

`dlent_rt/bid.py`, `synthesize_lognormal`, mode "size":

```python
cost_base = (1 + ln q) * (gamma1 * cores + gamma2 * GB) * duration_hours     # u0 not used
mu = ln(cost_base) - sigma**2 / 2;   bid = lognormal(mu, sigma = 0.3)        # same noise as before
```

Everything else is unchanged:
* the markup's virtual value φ;
* the reward terms: customer value = q × bid, profit = φ − electricity, carbon = s_C × kg;
* the carbon map: s_C = D_q / K_q at q = 0.90, computed in Phase 0.

The bid is now **linear in cores, GB and duration** (tested). The priority scales it by (1 + ln q), from 1.0 for
tier 1 to 2.6 for tier 5.

**The power model is unchanged.** `dlent_rt/energy.py`, `compute_power_kw`, lines 57–61:

```python
power_w = p_core * cores * kappa + p_gb * GB        # kappa = 1 + 0.116 x scheduling class
return pue * power_w / 1000                         # kW
```

**It has no fixed per-job (idle) term:** power is proportional to the request. So energy, electricity and carbon are
all proportional to request × duration, like the new bid.

## 3. Scale check: does the reward sign depend on the bid scale?

On v5, γ1 and γ2 were both multiplied by k, with the same data and the same random numbers.

| k | types with a positive reward | positive arrival mass | s_C ($/kg) | s_C / s_C(k = 1) | types with jobs in H |
|---|---|---|---|---|---|
| 0.5 | 67.33% | 87.95% | 10.86 | 0.483 | 4,126 |
| 1 | 66.67% | 87.78% | 22.47 | 1.000 | 4,005 |
| 2 | 66.35% | 87.63% | 45.21 | 2.012 | 3,929 |

* **As expected,** the share of positive types moves by less than 1 point, and s_C scales almost exactly with k.
* **Gap 1: electricity does not scale.** Bid, φ and customer value scale with k, but profit = φ − electricity, and
  the electricity cost stays the same. So |profit| and D_q do not scale exactly.
* **Gap 2: the valuation grid has an absolute floor.** Its edges run geometrically from max(smallest bid, 10⁻⁶ $) to
  the largest bid. 0.02% of H's bids are below 10⁻⁶ $, so scaling the bids moves the bottom edge. A few jobs then
  change bin, which explains the small change in the number of types with jobs.

## 4. Phase 0 on v5 vs v4

| | v4 (old bid, f = 1.00) | v5 (size bid, f = 1.25) |
|---|---|---|
| s_C ($/kg) | 14,342 | **22.47** |
| types with jobs in H (of 20,480) | 5,845 | 4,005 |
| τ (CPU, RAM) | 0, 0 | **0, 0** |
| types admitted: fully / partly | 3,595 / 0 | 2,670 / 0 |
| how full the LP keeps the cluster (CPU / RAM) | 26.0% / 34.7% | **65.9% / 65.0%** |
| types with a positive reward / share of the arrivals | 61.5% / 80.7% | 66.7% / 87.8% |
| reward split, all types ($/h): customer value / profit / carbon | 20,134 / 2,292 / −19,985 | 76 / 6 / −43 |
| reward split, admitted types ($/h) | 16,514 / 2,443 / −5,438 | 70 / 11 / −35 |
| C, CPU / RAM (machines) | 2.54 / 1.66 | 4.34 / 2.89 |
| Q, CPU / RAM (machines) | 0.509 / 0.332 | 0.869 / 0.578 |

* **s_C falls from 14,342 to 22.5 $/kg.** It is still about 120 times a social cost of carbon of 0.19 $/kg, because
  the map scales carbon to the dollar values, not to a carbon price.
* **The dollar values are now small** ($/h in the tens, not in the thousands), because u0 is gone. γ1 and γ2 are not
  recalibrated.

**Profit sanity.** Per CPU machine-hour, over the v5 jobs:

| | mean | p10 | p90 |
|---|---|---|---|
| bid ($) | 7.60 | 3.95 | 11.60 |
| electricity cost ($) | 0.028 | 0.016 | 0.039 |

The median bid is 248 times the electricity cost, so electricity hardly changes the profit.

**Why τ is still 0.**
* The positive types use 66% of C (CPU) and 65% (RAM); the negative types would use 16%.
* With the mean-demand rule, C = f × D̄, so the whole arrival load is 1/f = 0.80 of C. **Even if the LP admitted every
  type, it could not fill C.**
* So as long as f ≥ 1, capacity never binds in the fluid LP. For it to bind, C must be smaller than the load of the
  positive types, which here means f below about 0.82.
* f = 1.25 was chosen on the time-varying FCFS demand (spikes). The LP only sees the average.

## 5. Reward by size decile (v5)

Deciles of CPU request × capped duration, H + stream. Every value is a mean per job.

| decile | mean CPU | mean duration (h) | bid $ | customer value $ | profit $ | carbon $ | reward $ | reward per machine-hour $ | jobs with reward > 0 | share of CPU-hours |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.002 | 0.020 | 0.00037 | 0.0015 | 0.00004 | 0.00045 | 0.0011 | 28.9 | 92.9% | 0.04% |
| 2 | 0.004 | 0.019 | 0.00060 | 0.0023 | 0.00016 | 0.00078 | 0.0017 | 26.0 | 92.6% | 0.08% |
| 3 | 0.005 | 0.022 | 0.00085 | 0.0033 | −0.00003 | 0.0012 | 0.0021 | 20.9 | 91.7% | 0.12% |
| 4 | 0.007 | 0.023 | 0.0010 | 0.0041 | 0.00026 | 0.0016 | 0.0028 | 19.6 | 91.1% | 0.17% |
| 5 | 0.009 | 0.035 | 0.0016 | 0.0062 | 0.00039 | 0.0025 | 0.0041 | 19.3 | 89.8% | 0.26% |
| 6 | 0.015 | 0.051 | 0.0027 | 0.0104 | 0.00055 | 0.0045 | 0.0065 | 16.6 | 88.0% | 0.47% |
| 7 | 0.026 | 0.062 | 0.0043 | 0.0163 | 0.00089 | 0.0076 | 0.0096 | 14.3 | 88.3% | 0.81% |
| 8 | 0.035 | 0.111 | 0.0079 | 0.0297 | 0.0018 | 0.0140 | 0.0174 | 14.1 | 87.6% | 1.49% |
| 9 | 0.044 | 0.247 | 0.0175 | 0.0652 | 0.0052 | 0.0312 | 0.0393 | 14.7 | 86.8% | 3.24% |
| 10 | 0.088 | 2.015 | 0.4648 | 1.5990 | 0.1553 | 0.9136 | **0.8408** | **10.9** | 83.2% | 93.31% |

* **The bid now grows with size × duration.** Decile 10's mean bid is 1,250 times decile 1's (with the old bid it was
  96 times).
* **The reward per machine-hour is positive in every decile.** It falls only from 29 $ to 11 $; with the old bid it
  fell from 120,000 $ to −408 $.
* **Over all jobs,** 89.2% have a positive reward (77.3% before), and they hold 77.1% of the CPU-hours (25.8% before).

**Two worked examples**

| | median job | job at the quota |
|---|---|---|
| CPU / RAM (machines) | 0.0057 / 0.0026 | 0.869 / 0.385 |
| priority tier, duration | 4, 3.0 min | 4, 1.0 min |
| size rate $/h (γ1 cores + γ2 GB) | 0.015 | 2.28 |
| cost base $ | 0.0018 | 0.0908 |
| bid $ | 0.0020 | 0.1164 |
| customer value $ | 0.0081 | 0.4658 |
| profit $ | 0.0015 | 0.0929 |
| carbon kg / $ | 0.00013 / 0.0029 | 0.0071 / 0.1597 |
| **reward $** | **+0.0067** | **+0.3990** |

The job at the quota was −664 $ with the old bid. Now it is positive: its bid grows with its size, like its carbon.

## 6. Quota 0.2 vs 0.5 (v5, f = 1.25)

| quota share | C CPU / RAM | Q CPU / RAM | time with demand ≥ 95% of C | jobs kept | CPU-hours kept | τ | LP fill CPU / RAM | s_C |
|---|---|---|---|---|---|---|---|---|
| 0.2 | 4.34 / 2.89 | 0.869 / 0.578 | 30.2% | 98.3% | 3.8% | 0, 0 | 65.9% / 65.0% | 22.47 |
| 0.5 | 21.82 / 13.39 | 10.91 / 6.70 | 24.9% | 99.6% | 16.9% | 0, 0 | 60.7% / 64.3% | 21.57 |

* **A share of 0.5 keeps 4.4 times more work** (16.9% of the CPU-hours), and the cluster grows 5 times.
* **The time with demand ≥ 95% of C falls just below the band** (24.9%).
* **τ stays 0 in both cases,** for the reason in section 4.
* The config is unchanged (0.2).

## 7. Open issues

1. **τ = 0 is built into the rule.** With C = f × D̄ and f ≥ 1, the LP can never bind. The choice of f looks at
   time-varying demand, but the LP only sees the average. If the LP should bind, f must be below the load of the
   positive types (about 0.82 here), or C must be sized from the positive-reward load. This is a design decision.
2. **s_C = 22.5 $/kg is still about 120 times the social cost of carbon.** That comes from the quantile map, which
   gives carbon the same weight as dollars. Decide whether to keep the map or use a fixed carbon price.
3. **The dollar scale is now set only by γ1, γ2 and q** (cents per job). They were not recalibrated, and the old u0
   target (90% of types positive) is no longer used. The scale does not change the reward sign (section 3), so the
   scale is free.
4. **The quota of 0.2 × C keeps under 4% of the work; 0.5 keeps 17%.** Choose the share, and remember it also moves C.
5. **Two small effects make the scale check inexact:** the 10⁻⁶ $ floor of the valuation grid, and electricity not
   scaling with γ. Neither changes the conclusions.

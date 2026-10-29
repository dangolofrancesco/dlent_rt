# Week 32: Phase 0 diagnostics on the new environment (notebook 01)

* Notebook: `notebooks/01_phase0_diagnostics.ipynb`. It runs top to bottom with no errors and is committed with its
  outputs.
* Logic: `dlent_rt/diagnostics.py`.
* Config: `configs/v5_theory.yaml` (f = 0.5). **[both f]** also shows `configs/v5_size_bid.yaml` (f = 1.25).
* Saved summaries: `phase0_summary_f050.json`, `phase0_summary_f125.json`.

Phase 0, the bid, the carbon map and the configs are unchanged. Every plot of bids, the grid and the objectives uses
H.

## 0. Setup and environment

**The environment:** the uniform 300k sample, H = the first 7 days, a 24 h cap, the quota = 0.2 × C (fixed point at
f = 1.25), the cluster = f × D̄, the size-based bid (σ = 0.3, γ1 = 0.0332 $/core-h, γ2 = 0.00445 $/GB-h), the
quantile carbon map (q = 0.90), PUE 1.10, a declared machine of 64 cores / 256 GB, and xi = 0.001 and gamma = 1.0
forced. Unseen types follow the dual rule.

**The dataset:**
* H has 46,596 jobs, from 1 May 07:00 to 8 May 06:59 UTC.
* The stream has 248,185 jobs, from 8 May 07:00 to 1 June 06:54 UTC (T = 575.9 h).
* The quota removes 5,219 jobs. The 24 h cap shortens 837 kept jobs.

**The cluster, step by step:**
* The total work in H is 583.6 CPU and 388.5 RAM machine-hours over |H| = 168.0 h, so D̄ = 3.474 / 2.313 machines.
* f = 0.5 gives C = 1.737 / 1.156 machines (111 cores / 296 GB under the declared machine).
* f = 1.25 gives C = 4.343 / 2.891 machines.
* Q = 0.869 / 0.578 machines. That is 0.50 × C at f = 0.5 and 0.20 × C at f = 1.25, because the quota is fixed at
  f = 1.25.
* **Little's law:** H's average occupancy with no capacity is 3.270 / 2.161 machines, **5.9% / 6.5% below D̄**. The
  jobs that arrive late in H run past its last arrival, so part of their work falls outside the window.

**Arrivals:** the H and stream curves by hour of day and by day of week are in the notebook's figure.

## 1. Jobs in H

| quantity | mean | p10 | p50 | p90 | p99 | max |
|---|---|---|---|---|---|---|
| CPU request (machines) | 0.024 | 0.0024 | 0.0074 | 0.040 | 0.42 | 0.87 |
| RAM request (machines) | 0.015 | 0.0015 | 0.0065 | 0.023 | 0.26 | 0.57 |
| capped duration (h) | 0.37 | 0.017 | 0.017 | 0.50 | 5.9 | 24 |
| work r·d (machine-h) | 0.0125 | 0.00005 | 0.00019 | 0.0071 | 0.14 | 16.8 |

**50.1% of H's jobs sit at the 1-minute duration floor** (p10 = p50 = 1 minute).

## 2. Bids (size mode)

The cost base is (1 + ln q) × (γ1 × cores + γ2 × GB) × d. The bid is lognormal with mean equal to the cost base.

* **The markup** bid / cost base matches the lognormal(σ = 0.3) curve, with mean ≈ 1 (figure).
* **φ < 0 for 16.7% of the jobs.**
* **The bid per machine-hour rises with priority:** $3.33 (tier 1) to $8.72 (tier 5).
* **The mean φ / bid is strongly negative in tier 4 (−0.70)** because of a few jobs with very low markups. φ has a
  long negative tail.

**σ sensitivity:**

| σ | mean bid $ | φ < 0 | mean φ $ |
|---|---|---|---|
| 0.1 | 0.0776 | 4.6% | 0.0504 |
| 0.3 | 0.0769 | 16.7% | 0.0203 |
| 0.5 | 0.0761 | 30.1% | 0.0073 |

## 3. Energy and grid data

* **Power check on the 5 largest jobs:** P = PUE × (15 W × cores × κ + 0.35 W × GB) / 1000 equals the code to
  machine precision (about 1.06 kW for 55 cores and 100 GB). Power is proportional to the request; there is no idle
  term.
* **The grid** runs from 1 May 00:00 to 3 June 01:00 UTC, including the 45 filled hours (1–3 June, copies of 30–31
  May).
* **Every job time is covered:** H jobs end by 9 May 05:09, stream jobs by 1 June 06:55.
* The hour-of-day profiles, the price–carbon scatter (with its correlation) and the at-arrival histograms are in the
  notebook.

## 4. Phase 0 output [both f]

| | f = 0.5 | f = 1.25 |
|---|---|---|
| types / with jobs in H | 20,480 / 4,005 | same |
| jobs assigned / rejected for IR | 46,596 / 0 | same |
| sum p = true arrival rate of H | 277.37 per hour | same |
| grid | k* = 128, geometric, ratio 1.159; edges $10⁻⁶ to $151.6 | same |
| s_C | **22.47 $/kg** = D_q / K_q, with D_q = (Q90 customer value 0.160 + Q90 |profit| 0.022) / 2 = 0.0914 $ and K_q = 0.00407 kg | same |
| r̃: min / median / max | −285.5 / 0.0045 / 491.0 | same |
| positive types / positive arrival mass | 66.7% / 87.8% | same |
| gamma (data), dominant term | 86,400, set by d_max in ticks (the others: r_max 0.019, v_rate_max 151.6, c_max 1.7, r_max / v_min 1.95) | 86,400, same cause |
| xi (data) / forced | 0.50 / 0.001 | 0.20 / 0.001 |
| eps_D (data xi) | 53.7 | 23.7 |
| d_max | 24 h | 24 h |
| C / Q (machines) | 1.737 / 1.156; 0.869 / 0.578 | 4.343 / 2.891; same Q |
| **τ (CPU, RAM)** | **0.00206, 0.00162** | **0, 0** |
| LP optimum | $38.56 per hour | $45.58 per hour |
| load if all admitted (p·v / C) | 2.04 / 2.01 | 0.81 / 0.81 |
| LP fill (at the optimum) | 100% / 100% | 66% / 65% |
| types fully / partly / rejected | 1,473 / 2 / 2,530 | 2,670 / 0 / 1,335 |

**Type coverage:**
* **31.6% of the populated types have only one H job**; the median is 3, the maximum 689.
* 112 of the 128 bid bins are used.
* The populated types by priority and hardware profile are in the notebook's heatmap.

## 5. Objective analysis

* **Share of Σ|term| (p-weighted):**
  * all populated types: customer value 55.8%, profit 12.6%, carbon 31.6%;
  * admitted types: 63.0%, 10.6% and 26.4%.
* **By size decile** (per job): the top decile holds 95.5% of the CPU-hours. Its reward per machine-hour is $10.8,
  against $14–26 in the other deciles. 83% of its jobs have a positive reward.
* **By priority tier,** the mean reward per job is negative for tiers 1 and 2 (−0.14 $ and −0.02 $): their carbon is
  larger than their customer value. Tiers 3–5 are positive.
* **Trace of 5 types:** r̃ recomputed from the raw H jobs (CI, energy, kg, price, bid, φ) equals the stored value for
  all five. The most negative type (−285.5 $) has 2 jobs with a very low markup: φ = −286.8 $ against a bid of $1.97.
* **Carbon weight × 0.5 / × 1 / × 2** (s_C frozen):
  * positive types: 81.9% / 66.7% / 36.9%;
  * at × 2, τ = 0 and the LP fill falls to 89% (CPU) / 96% (RAM).

## 6. Stability over 20 seeds (bid noise only)

| quantity | mean ± std |
|---|---|
| s_C | 22.78 ± 0.61 $/kg |
| positive types | 66.6 ± 0.9% |
| τ, CPU | 0.00218 ± 0.00033 |
| τ, RAM | 0.00174 ± 0.00046 |
| LP fill | 100% (std about 0) |
| customer value / profit / carbon shares | 0.558 / 0.120 / 0.322 (std under 0.01) |

Phase 0 is stable to the bid noise.

## 7. Checks

| check | f = 0.5 | f = 1.25 |
|---|---|---|
| sum p = true arrival rate of H | PASS | PASS |
| manual power = code power (incl. PUE) | PASS | PASS |
| stored C_elec_bar = recomputed | PASS | PASS |
| stored C_carbon_bar = recomputed | PASS | PASS |
| r̃ = customer value + profit − carbon $ (λ = 1) | PASS | PASS |
| LP fill ≤ 1 | PASS | PASS |
| τ ≥ 0 | PASS | PASS |
| the grid covers every job time (H and stream) | PASS | PASS |
| no NaN or inf in the PhaseState arrays | PASS | PASS |

## What looks odd

1. **Gamma on the data is 86,400, set by d_max counted in ticks (seconds).** The other terms are in dollars and are
   far smaller. The unit mix makes the data gamma meaningless. It is forced to 1.0, so the online model is not
   affected.
2. **The bottom of the valuation grid is the 10⁻⁶ $ floor.** A few bids are tiny, so the geometric edges start at
   the floor, and some of the 128 bins are spent on almost-empty bid ranges (112 bins used).
3. **Half of H's jobs sit at the 1-minute floor,** so their duration is not their real one.
4. **The virtual value φ has a long negative tail.** One type with 2 jobs has r̃ = −285 $. The rare low-markup
   draws dominate the most negative types.
5. **Tiers 1 and 2 earn a negative reward per job:** their carbon outweighs (1 + ln q) × bid.
6. **The data xi is 0.50 at f = 0.5,** because Q was set by the fixed point at f = 1.25. xi is forced to 0.001, so the
   online model does not see this.
7. **The LP binds at f = 0.5 but not at f = 1.25.** Doubling the carbon weight makes it stop binding at f = 0.5
   too: the sign split of r̃ drives the load.

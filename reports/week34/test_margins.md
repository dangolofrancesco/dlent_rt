# How far Tests 2 and 3 are from firing (week 34, Part B)

**Data.** The stationary theoretical run of week 33 (`reports/week33/cache/theory_run.pkl`): f = 0.5, seed 42,
`fit_on: "H"`, 10 phases. Nothing was run for this note. The thresholds are recomputed from the run's own error
parameters:
* γ = 1 and ξ = 0.001, both forced;
* c_min = 0.425 machines;
* T = 2,073,236 ticks of 1 s, δ = 1/T³.

Rewards are in units of r_max = 1.253 $ (D5).

**How the tests work** (`models/core/monitor.py`, `error_params.py`). Both are evaluated at every tick u.
* **Test 2 (market worse)** fires when the average per-tick deficit since the phase start is above its threshold:
  LHS = mean over the active ticks of (fac · λ(τ) − R_τ) > ε_Test2(n).
  * λ(τ) is the LP value per tick on the phase's arrivals so far.
  * fac = 1 − ε_D − ξ = 0.489.
  * R_τ is the realised reward at tick τ.
* **Test 3 (market better)** fires when the LP value on a recent window rises above the bound:
  λ^(s+1 : u−1) > λ̃(u−1) + ε_Test3(u−1−s).
  * λ̃ = min over the phase of (λ + ε_A′ + ε_B′).

## 1. The margins in the stationary run

| Phase | Length h | End | max Test 2 LHS | min Test 2 threshold | max LHS / threshold | LP value λ per tick | λ̃ | min Test 3 margin (RHS − LHS) |
|---|---|---|---|---|---|---|---|---|
| 1 | 48.0 | timeout | 0.00074 | 0.156 | 0.0035 | 0.0015 | 0.190 | 0.318 |
| 2 | 95.3 | Test 1 | 0.00071 | 0.110 | 0.0034 | 0.0015 | 0.134 | 0.225 |
| 3 | 47.3 | Test 1 | 0.00054 | 0.146 | 0.0027 | 0.0011 | 0.200 | 0.320 |
| 4 | 94.6 | timeout | 0.00053 | 0.111 | 0.0026 | 0.0011 | 0.136 | 0.226 |
| 5 | 45.3 | Test 1 | 0.00070 | 0.149 | 0.0035 | 0.0015 | 0.178 | 0.327 |
| 6 | 48.2 | Test 1 | 0.00086 | 0.149 | 0.0042 | 0.0018 | 0.186 | 0.317 |
| 7 | 46.3 | Test 1 | 0.00022 | 0.150 | 0.0011 | 0.0012 | 0.178 | 0.317 |
| 8 | 50.0 | Test 1 | 0.00073 | 0.147 | 0.0036 | 0.0015 | 0.196 | 0.311 |
| 9 | 50.9 | Test 1 | 0.00049 | 0.145 | 0.0024 | 0.0010 | 0.194 | 0.308 |
| 10 | 50.0 | horizon | 0.00099 | 0.146 | 0.0048 | 0.0021 | 0.180 | 0.311 |

λ, λ̃ and the margins are medians or extremes over each phase, in r_max units per tick.

* **Test 2 reaches at most 0.5% of its threshold** (max LHS / threshold = 0.0048, phase 10).
* **Test 3's right side is always at least 0.22 above its left side.** The left side, the LP value per tick, is about
  0.0015.

## 2. What it would take to fire within about 24 h

The thresholds after n hours of phase, for a previous phase of 48 h:

| n (hours since the phase start) | ε_Test2 | ε_Test3 | ε_A′ + ε_B′ (in λ̃) |
|---|---|---|---|
| 6 | 0.400 | 0.529 | 0.385 |
| 12 | 0.286 | 0.371 | 0.270 |
| 24 | 0.206 | 0.261 | 0.190 |
| 48 | 0.150 | 0.184 | 0.134 |

* **Test 2 within 24 h** needs a mean deficit above 0.206 per tick.
  * Even with **zero** reward, the deficit is at most fac · λ ≈ 0.0007, about 280 times too small.
  * To fire, the realised reward would have to be strongly **negative**: a loss of about **925 $ per hour** for a
    whole day. The run normally earns about 3.9 $ per hour, so that is a loss 236 times larger than the usual gain.
  * **No fall in bids can do this.** A fall makes the model admit less; it cannot create losses of that size.
* **Test 3 within 24 h** needs the LP value on a recent window to exceed λ + (ε_A′ + ε_B′) + ε_Test3 ≈
  λ + 0.19 + 0.26 ≈ 0.45 per tick.
  * That is about **300 times** the stationary value (0.0015 per tick).
  * In dollars: the LP value would have to rise from about 6.1 $ per hour (the LP optimum) to about 2,000 $ per hour.

**Why.** The thresholds are confidence terms of order √(log(1/δ) / n), and n counts ticks (seconds), with
log(1/δ) = 3 ln T ≈ 44. With γ forced to 1, they bound a reward of up to one r_max per tick. But the run sees about
0.12 arrivals per tick, and its LP value is about 0.0015 r_max per tick. So the signal is two to three orders of
magnitude below the thresholds.

## 3. The scenarios

| Scenario | Direction | Effect on the test's left side | Fires? |
|---|---|---|---|
| S1, bids fall (m → 0.5 m) | market worse → Test 2 | the deficit is at most fac · λ ≈ 0.0007 even if no job is admitted; the threshold is ≥ 0.11 | **no** |
| S2, bids rise (m → 1.75 m) | market better → Test 3 | λ grows by a small factor (bids × 1.75; with φ, perhaps 2–3×), to about 0.004; it would need about 0.45 | **no** |
| S6, spread (σ → 0.5) | small change in either direction | smaller than S1 or S2 | **no** |

**Conclusion: with these magnitudes, neither Test 2 nor Test 3 can fire, in any scenario.** Phases will end only by
Test 1 (capacity) or by timeout, as in the stationary runs. The detection delay "to the first Test 2 / 3" will be
empty. The models still re-estimate at every phase end, every 45–95 h, so learning can still adapt to the change. But
the adaptation will come from the Test 1 / timeout cadence, not from detection.

To make the tests able to fire, the options would change the model or the experiment:
* rescale the reward unit or γ;
* count n in arrivals instead of ticks;
* use far larger changes.

I did not change anything. This is your decision.

## 4. Calibrated thresholds (decision: `tests.threshold: calibrated`)

**What changes.** Each test's threshold is multiplied by one constant (`dlent_rt/models/core/calibration.py`):
* **Test 2:** ε̃_Test2 × κ2.
* **Test 3:** **κ3 scales every confidence term of Test 3**, i.e. ε̃_Test3 and also ε_A′ + ε_B′ inside λ̃.
  ε_A′ + ε_B′ alone is about 0.19 per tick, about 100 times λ. So scaling ε̃_Test3 alone could never make Test 3
  fire.

**How κ is chosen.** Once, in Phase 0, from H alone (7 days, 44,719 jobs):
1. H is replayed through the no-learning model (Phase 0's state, practical switches). Phases end by Test 1 or
   timeout and restart from Phase 0's state. On H: 4 phases, ending at 47, 96 and 144 h by Test 1, then at the
   horizon (168 h).
2. Each test's LHS / threshold ratio is traced at every tick.
3. κ is set so that the largest ratio on H is **0.5**, a **safety factor of 2**.

No stream data is read; a test checks this. κ depends only on H, which is the same in every environment. So it is
computed once (`scripts/calibrate_tests.py --set-kappa`) and reused by every calibrated run.

| | Largest ratio on H, paper's threshold | κ | Threshold lowered by |
|---|---|---|---|
| Test 2 | 0.0036 | **κ2 = 0.00725** | about 138 times |
| Test 3 | 0.0042 | **κ3 = 0.00427** | about 234 times |

**False alarms on the stationary stream.** I replayed the cached stationary practical run
(`scenarios/none/practical`, 576 h, 13 phases). No new run was needed:
* the run's own arrivals, rewards and admissions were fed to the monitor;
* each phase's state was rebuilt by ReInit;
* the replayed phase ends match the cached ones exactly.

A firing is one episode: a run of consecutive gaps of one phase with ratio > 1. Test 3's count uses the monitor's
screen bound, so it is an upper bound.

| Phase | Start h | Length h | End | Largest Test 2 ratio | Largest Test 3 ratio |
|---|---|---|---|---|---|
| 1 | 0.0 | 46.7 | Test 1 | 0.48 | 0.53 |
| 2 | 46.7 | 48.9 | Test 1 | 0.50 | 0.54 |
| 3 | 95.6 | 47.4 | Test 1 | 0.53 | 0.54 |
| 4 | 143.1 | 50.8 | Test 1 | 0.47 | 0.54 |
| 5 | 193.9 | 49.5 | Test 1 | 0.39 | 0.51 |
| 6 | 243.4 | 48.1 | Test 1 | 0.37 | 0.47 |
| 7 | 291.6 | 49.0 | Test 1 | 0.48 | 0.59 |
| 8 | 340.6 | 46.5 | Test 1 | 0.66 | 0.59 |
| 9 | 387.1 | 46.9 | Test 1 | 0.54 | 0.58 |
| 10 | 434.0 | 46.3 | Test 1 | 0.60 | 0.57 |
| 11 | 480.3 | 46.7 | Test 1 | 0.43 | 0.56 |
| 12 | 527.0 | 48.4 | Test 1 | 0.40 | 0.61 |
| 13 | 575.4 | 0.5 | horizon | < 0 | 0.61 |

* **False alarms: 0 over 576 h** (Test 2: 0, Test 3: 0).
* The largest ratios on the stream (0.66 for Test 2, 0.61 for Test 3) are above H's 0.5, but below 1. So the
  stream's own noise uses about a third of the safety factor.
* **By the decision rule** (at most one firing at target 0.5), the safety factor stays 2. The 4 extra runs
  (`practical_calibrated` in none, s1, s6, s2) start right after Part C.

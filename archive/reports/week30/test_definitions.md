# Week 30: the three tests and the timeout, as implemented

**Code:**
* `dlent_rt/models/core/monitor.py` (`PhaseMonitor`) — the tests and the timeout.
* `dlent_rt/models/core/error_params.py` (`ErrorParams`) — the thresholds.
* `dlent_rt/models/theory.py` — phase end and restart.
* `dlent_rt/models/core/reinit.py` (`build_window_state`) — ReInit.

**Measurements:** `notebooks/src/test_margins_check.py`; CSVs `tables/test_margins_*.csv` and
`tables/test_thresholds_*.csv`.

The paper PDF (Zhang & Cheung 2025) is not in the repo. Every statement below is the CODE definition. Points where the
code says it departs from the paper, or where it may, are marked **[to compare with paper]**. Equation numbers are
those quoted in the code's comments.

## Notation (code ↔ paper symbols)

* **Phases.** Phase m starts at tick τ_prev + 1, where τ_prev = τ^(m−1). Its previous length is W_prev = W^(m−1),
  and phase 1 uses W^(0) = d_max (`theory.py:206`).
* **Warm-up.** It ends at τ̄ = τ_prev + d_max (`monitor.py:62`). d_max = 14,304 ticks in both configs.
* **Check tick.** At tick u the tests use the information up to u − 1. When a check fails at u*, the phase ends at
  τ^(m) = u* − 1, and u* is the first tick of phase m + 1 (`theory.py:140`).
* **Window LP.** λ̂^(s+1:t) is the LP-RS optimum per tick on the window (s, t], with n = t − s:
  λ̂ = (1/n)·max{ r̃ᵀz : Σ_j v_ij z_j ≤ n·c_i, 0 ≤ z_j ≤ N_j }.
  * N_j counts the phase's arrivals of type j in the window.
  * r̃_j and v_ij = A_ij·d̄_j are the phase state's frozen values.
  * Code: `window_lp.py`, exact parametric evaluation per tick.
* **Error terms** (n = window length; δ = 1/T³; c_min = min_i c_i; |I| = 2 resources):
  * η(n) = γ / (c_min √n)
  * ε_A′(n) = (2γ²/c_min)·(√(2 log(|I|/δ)/n) + log(|I|/δ)/n)
  * ε_B′(n) = γ √(2 log(T/δ)/n)
  * ε_C′(n) = 2γ √(2 log(1/δ)/n) + 2γ log(1/δ)/n
  * ε_G(n) = γ √(2 log(1/δ)/n)
  * ε_D = 2√(ξ log(|I|/δ)) + 2ξ log(|I|/δ)  (A.7)
  * ε_E = ε_F = 0  (A.8, A.9, Exact-LP variant)

## Test 1: capacity of the in-phase admissions

1. **Condition** (`PhaseMonitor.scan`, `monitor.py:142–145`; window counter `WindowOccupancy`, `occupancy.py:53–75`).
   The phase ends at u* = u0 − 1 if, for some resource i,

   Σ_{τ = τ̄+1}^{u−1} A_{i,j(τ)} · 1(D(τ) ≥ u − τ)  >  c_i − a_max,eff,i

   with a_max,eff = ξ·c when ξ is forced and a_max (data) otherwise (`theory.py:66–67`). The comment cites eq. (8)
   [to compare with paper].
2. **LHS / RHS.**
   * **Which jobs enter.** Only jobs ADMITTED in the current phase at ticks τ ≥ τ̄ + 1, that is after the warm-up
     (`register_arrival`, `monitor.py:82–83`). A job counts at tick u while τ < u ≤ τ + D.
   * **What is excluded.** Jobs carried over from earlier phases are not included, and neither are jobs admitted
     during the warm-up. Both still occupy the physical capacity.
   * **RHS.** Capacity minus the effective largest request, per resource.
   * **When it is checked.** Only for u > τ̄ + 1, and only at the first active tick u0 of each gap between two
     arrivals. This is exact, because the LHS is non-increasing between admissions.
3. **Constants.** c_i and a_max,eff,i.

   | | default (from_amax) | binding (0.3 / 0.7, filter on) |
   |---|---|---|
   | c (normalised CPU, RAM) | 3.945, 2.555 | 0.8028, 0.5232 |
   | threshold with ξ = 0.001 (c − 0.001c) | 3.9415, 2.5525 (99.9% of c) | 0.8020, 0.5227 (99.9% of c) |
   | **without the override** (c − a_max, data) | 1.9727, 1.2775 (50% of c) | 0.3242, 0.1875 (40.4% / 35.8% of c) |

4. **On firing:** the common restart below.
5. **To compare with paper.** With ξ forced, a_max,eff = ξ·c is used instead of the data a_max. The jobs admitted
   during the warm-up are excluded from the LHS.

## Test 2: realised reward against the scaled LP benchmark

1. **Condition** (`scan`, `monitor.py:119, 134–140`; threshold `ErrorParams.eps_test2` and `CI1`,
   `error_params.py:97–105`). The phase ends at u* − 1 for the first active u with

   LHS2(u) = (1 / (u − τ̄ − 1)) · Σ_{τ = τ̄+1}^{u−1} [ (1 − ε_D − ξ)·λ̂^(τ_prev+1 : τ) − R(τ) ]  >  ε̃_Test2(u − 1)

   with:
   * ε̃_Test2 = γ·ε_E + ε_F + γ·η(W_prev) + 4γ/√W_prev + CI¹(n), where n = u − 1 − τ_prev (A.12);
   * CI¹(n) = 2ε_C′(n) + ε_G(n) + 2γ·η(n) + 4γ/√n.
2. **LHS / RHS.**
   * **Reward term.** R(τ) is the realised R_job of the job admitted at tick τ, and 0 if there is no arrival or the
     arrival is rejected.
   * **Benchmark term.** λ̂^(τ_prev+1:τ) is the window LP on all of the phase's arrivals (admitted or not) up to τ.
   * **Window.** Only ticks after the warm-up enter the average. Jobs and arrivals of earlier phases are not included.
   * **When it is checked.** Every active tick u > τ̄ + 1, exactly (vectorised over each gap).
   * **RHS.** A confidence radius that decreases in n and in W_prev.
3. **Constants.** γ, ξ, ε_D (through 1 − ε_D − ξ), c_min, δ, T, W_prev.
   * **With the overrides:** γ = 1.0, ξ = 0.001, so ε_D = 0.5022 and 1 − ε_D − ξ = 0.4968.

     | n (with W_prev = n) | ε̃_Test2, default | ε̃_Test2, binding |
     |---|---|---|
     | 14,304 (d_max) | 0.474 $/tick | 0.512 $/tick |
     | 100,000 | 0.177 $/tick | 0.191 $/tick |
     | 1,432,910 (T) | 0.046 $/tick | 0.050 $/tick |

     For scale, Phase 0's λ* = 0.0675 $/tick on the default config.
   * **Without the overrides:**
     * γ_data = 14,304. This is d_max: the max term in γ = max(r_max, v_rate_max, d_max, c_max, r_max/v_min),
       `phase0.py:209`.
     * ξ_data = 0.5 (default) or 0.642 (binding), so ε_D = 52.5 or 66.0 and 1 − ε_D − ξ = −52.0 or −65.6. The
       benchmark term of the LHS becomes NEGATIVE, so LHS2 < 0 and Test 2 cannot fire.
     * The threshold becomes 2.0e6 $/tick (default) or 9.8e6 (binding) at n = d_max, and 2.0e5 or 9.8e5 at n = T.
4. **On firing:** the common restart below.
5. **To compare with paper.**
   * ε_E = ε_F = 0 (Exact-LP: LP-RS replaces the dual-learning Algorithm 2; `tests.drop_dual_learning_terms`).
   * γ and ξ are forced.
   * λ̂ is the exact LP optimum on window counts.

## Test 3: change of the LP benchmark between windows

1. **Condition** (`PhaseMonitor._test3`, `monitor.py:171–202`, called at `monitor.py:147`; threshold
   `eps_test3` and `lam_tilde_term`, `error_params.py:108–115`). The phase ends at u* − 1 for the first active u and
   some s in the grid with

   λ̂^(s+1 : u−1)  >  λ̃(u − 1) + ε_Test3(u − 1 − s)

   with:
   * λ̃(u − 1) = min_{τ_prev < τ ≤ u−1} [ λ̂^(τ_prev+1 : τ) + ε_A′(τ − τ_prev) + ε_B′(τ − τ_prev) ] (eq. 7);
   * ε_Test3(n) = ε_A′(n) + ε_B′(n) + ε_C′(n) + 2γ²/√n (A.13).
2. **LHS / RHS.**
   * **LHS.** The window LP on the phase's arrivals in (s, u − 1].
   * **s-grid.** s ∈ τ_prev + {0, 1, 2, 4, 8, …} (`test3.s_grid = geometric`, base 2). An s is skipped if its window
     holds no arrival (s ≥ the last arrival tick) or if s > u0 − 2.
   * **λ̃.** The running minimum since the phase start, including the warm-up ticks.
   * **What is excluded.** Arrivals of earlier phases are not included.
   * **When it is checked.** Only active ticks u ≥ u0 (after the warm-up).
   * **Screen.** Within a gap the code certifies "no failure" with LHS(first u) ≤ RHS(last u). It reconstructs tick
     by tick only when this screen fails, and for s = τ_prev, where the per-tick λ̂ is already known.
3. **Constants.** γ (and γ² in ε_A′ and the last term), c_min, δ, T, |I|.
   * **With the overrides:**

     | n | ε_Test3, default | ε_Test3, binding |
     |---|---|---|
     | 14,304 | 0.329 | 0.575 |
     | 100,000 | 0.123 | 0.213 |
     | T | 0.032 | 0.056 |

     λ̃ adds ε_A′ + ε_B′, which is 0.152 (default) or 0.398 (binding) at n = d_max.
   * **Without the overrides:** ε_Test3 = 1.6e7 (default) or 6.7e7 (binding) at n = d_max, and 1.6e6 or 6.4e6 at
     n = T. Test 3 cannot fire.
4. **On firing:** the common restart below.
5. **To compare with paper.**
   * The geometric s-grid instead of every s.
   * The warm-up applies to Test 3 as well.
   * γ is forced.
   * ε_A′ and η use c_min = min_i c_i in the trace's normalised units.

## Timeout

1. **Condition** (`scan`, `monitor.py:128–130`). The phase ends at τ_prev + 2·W_prev, that is at u* = τ_prev + 2W_prev
   + 1, so W ≤ 2·W_prev. Phase 1 times out at 2·d_max = 28,608.
2. **Checked** at every tick. There is no warm-up rule.
3. **Constants.** None (W_prev only).
4. **On firing:** the common restart below.
5. **To compare with paper.** The doubling rule and W^(0) = d_max.

## Common restart (any of Test 1 / 2 / 3 / timeout)

The same handling applies to all four reasons (`theory.py:99–132`, `_end_phase`).
* **Tie-break.** If several checks fail at the same u*, the reason recorded is the alphabetically first label
  (`min(fails)`, `monitor.py:159`: test1 < test2 < test3 < timeout).
* **Phase end.** The phase ends at τ^(m) = u* − 1, with W = τ^(m) − τ_prev.
* **Hindsight λ.** It is computed and logged only; the decisions do not use it.
* **ReInit** (`build_window_state`, `reinit.py:61–156`), on the phase's own arrivals only (not cumulative):
  * **Grid.** It is refit on the phase's bids if the phase had ≥ n_min = 80 arrivals (`reinit.py:72–80`); otherwise
    the previous grid is kept. The known markup distribution F is never refit.
  * **p.** p_j = N_j / W (`reinit.py:128`).
  * **Values.** v̄ and φ̄ come from all of the phase's arrivals.
  * **Costs.** d̄, energy, C_elec and K come from the ADMITTED jobs only.
  * **Fallbacks.** A type with fewer than n_type_min = 5 observations keeps the previous phase's statistics. If a type
    has none at all, its hardware-profile mean is used.
  * **Reward.** r̃ is recomputed with s_C frozen at Phase 0 (`reinit.py:134`). v = A·d̄.
  * **LP.** LP-RS is re-solved, giving new τ and λ* (`reinit.py:136`).
* **New monitor.**
  * τ_prev ← τ^(m), W_prev ← W, τ̄ ← τ^(m) + d_max.
  * The Test 1 window counter is emptied, the Test 2 sum is reset to 0, λ̃ ← ∞, the window counts are emptied, and a
    new window LP is built on the new (r̃, v).
  * The next timeout is at τ^(m) + 2W.
  * If W < d_max / 2, the next phase times out before its warm-up ends, so no test can fire in it.
* **Not reset.** Physical occupancy: jobs admitted earlier keep their capacity until they finish. Thinning uses
  p_U(t) = (1 − ξ)/(1 + ε_D + η(W_prev) + η(t − τ_prev)), whose cap is 0.665 with the overrides and 0.0093 (default)
  or 0.0054 (binding) without them.

## Measured margins (seed 42, thresholds unchanged)

Each run's phases and R_total reproduce the cached results exactly:
* **default:** `tests/data/snapshot_real.npz`, R = 48,391.61.
* **binding:** the week-30 sweep row, R = 40,740.42.

Each ratio is (statistic / threshold); a ratio above 1 means the test fails at that tick.
* **T1:** the max over i of LHS1 / (c − a_max,eff).
* **T2:** LHS2 / ε̃_Test2.
* **T3:** the max over the s-grid of LHS / RHS. In every phase below, the T3 maximum is the monitor's screen
  value, which is an upper bound on the true maximum.
* **timeout:** W / (2 W_prev).

The tick columns give where each maximum is reached.

**configs/default.yaml** (committed, from_amax). Tests that fired: Test 2 once (phase 6), timeout 9 times, plus the
horizon.

| phase | ticks | W | W_prev | ended by | T1 max | T1 tick | T2 max | T2 tick | T3 max (bound) | T3 tick | W / 2W_prev |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1–28,608 | 28,608 | 14,304 | timeout | 0.149 | 22,060 | 0.0134 | 15,108 | 0.182 | 16,504 | 1 |
| 2 | 28,609–85,824 | 57,216 | 28,608 | timeout | 0.515 | 70,180 | 0.0166 | 42,914 | 0.139 | 73,654 | 1 |
| 3 | 85,825–200,256 | 114,432 | 57,216 | timeout | 0.763 | 143,489 | 0.0237 | 100,130 | 0.822 | 118,594 | 1 |
| 4 | 200,257–429,120 | 228,864 | 114,432 | timeout | 0.732 | 403,218 | 0.0607 | 425,761 | 0.280 | 426,053 | 1 |
| 5 | 429,121–886,848 | 457,728 | 228,864 | timeout | 0.633 | 842,336 | 0.0271 | 443,624 | 0.369 | 886,732 | 1 |
| 6 | 886,849–901,412 | 14,564 | 457,728 | **test2** | 0.068 | 901,413 | **1.11** | 901,413 | 0.242 | 901,256 | 0.016 |
| 7 | 901,413–930,540 | 29,128 | 14,564 | timeout | 0.749 | 929,312 | 0.155 | 915,718 | 0.407 | 930,064 | 1 |
| 8 | 930,541–988,796 | 58,256 | 29,128 | timeout | 0.464 | 986,409 | 0.253 | 944,846 | 0.442 | 944,909 | 1 |
| 9 | 988,797–1,105,308 | 116,512 | 58,256 | timeout | 0.655 | 1,088,047 | 0.100 | 1,003,102 | 0.400 | 1,104,484 | 1 |
| 10 | 1,105,309–1,338,332 | 233,024 | 116,512 | timeout | 0.973 | 1,161,785 | 0.0796 | 1,119,614 | 0.378 | 1,337,222 | 1 |
| 11 | 1,338,333–1,432,910 | 94,578 | 233,024 | horizon | 0.903 | 1,424,936 | 0.236 | 1,387,421 | 0.513 | 1,425,469 | 0.203 |

**Binding config** (fraction_of_concurrency_quantile 0.3 / 0.7, drop_jobs_exceeding_capacity on, default grid). Tests
that fired: none; timeout 5 times, plus the horizon.

| phase | ticks | W | W_prev | ended by | T1 max | T1 tick | T2 max | T2 tick | T3 max (bound) | T3 tick | W / 2W_prev |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1–28,608 | 28,608 | 14,304 | timeout | 0.380 | 22,060 | 0.0157 | 18,443 | 0.0756 | 16,504 | 1 |
| 2 | 28,609–85,824 | 57,216 | 28,608 | timeout | 0.935 | 71,093 | 0.0135 | 44,070 | 0.0596 | 71,221 | 1 |
| 3 | 85,825–200,256 | 114,432 | 57,216 | timeout | 0.981 | 150,083 | 0.128 | 105,326 | 0.0939 | 163,426 | 1 |
| 4 | 200,257–429,120 | 228,864 | 114,432 | timeout | 0.950 | 234,217 | 0.0070 | 214,562 | 0.0785 | 426,053 | 1 |
| 5 | 429,121–886,848 | 457,728 | 228,864 | timeout | 0.998 | 582,241 | 0.0310 | 447,183 | 0.227 | 886,689 | 1 |
| 6 | 886,849–1,432,910 | 546,062 | 457,728 | horizon | 0.999 | 924,121 | 0.119 | 901,553 | 0.335 | 1,431,862 | 0.597 |

**Reading (numbers only):**
* **Test 1.** On the binding config it comes within 0.2% of its threshold in phases 5 and 6 (0.998, 0.999) and never
  exceeds it. On default its maximum is 0.973.
* **Test 2.** It fired once (default, phase 6, ratio 1.11 at the phase's last checked tick). Elsewhere its maximum
  is 0.25 (default) and 0.13 (binding).
* **Test 3.** Its upper bound stays at or below 0.82 (default) and 0.34 (binding).
* **Timeout.** Every phase ending before the horizon, other than the Test 2 phase, ended by timeout at W = 2·W_prev.

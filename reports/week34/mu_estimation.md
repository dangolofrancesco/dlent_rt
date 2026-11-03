# Markup model: μ̂ and σ̂ both estimated (week 34)

**What changed.** The mechanism's markup model was LogNormal(μ = −σ̂²/2, σ̂): only σ̂ = std(ln m) was estimated. Now
both are estimated, μ̂ = mean(ln m) and σ̂ = std(ln m), with m = b / c. This happens in Phase 0 on H's bids and at
every ReInit (`bid.MyersonModel.fitted`, `bid.bid_model`, `reinit.build_window_state`). φ is computed with (μ̂, σ̂).
The generator is unchanged: ln m ~ N(−σ²/2, σ²), σ = 0.3. This is the new default for every learning model. It lets
the model follow a shift in the markups' level (scenarios S1 and S2), not only in their spread.

**Criterion.** The 1% limit applies to the **real** data: Phase 0 on H must move by at most 0.5% in every quantity.

## Real data: Phase 0 on H (44,719 jobs, f = 0.5, seed 42), old vs new code

Old code: commit 8b4382b.

| | Old (μ = −σ̂²/2) | New (μ̂ estimated) | Change |
|---|---|---|---|
| U0 ($/h) | 0.021382 | 0.021382 | none |
| σ̂ | 0.300994 | 0.300994 | none |
| μ used | −0.045299 | **−0.044152** | +0.0011 in ln m |
| s_C ($/kg) | 9.8809 | 9.8940 | +0.13% |
| τ (CPU, RAM) | 0, 0.0009469 | 0, 0.0009419 | RAM −0.5% |
| % of populated types with r̃ > 0 | 67.46 | 67.27 | −0.19 points |
| % of H's arrival rate with r̃ > 0 | 79.07 | 78.99 | −0.08 points |
| r_max ($) | 1.2531 | 1.2521 | −0.08% |
| LP optimum λ* (per tick) | 0.001692 | 0.001690 | −0.1% |

Every quantity moves by at most 0.5%. **The criterion passes.**

## Synthetic regression snapshot (tests only)

The synthetic trace's total reward moves by **+3.05%** (41.377 → 42.640). Per phase the change is +1.3%, +2.5% and
+4.0%. Only 8 of 2,579 decisions change.

This comes from its tiny H: 407 jobs, σ̂ = 0.283 against the true 0.30. There, μ̂ − (−σ̂²/2) = −0.006, five times the
real-data gap. The change splits roughly into three parts:

| Part | Effect on the total |
|---|---|
| s_C changes (12.93 → 12.81 $/kg) | about +1.4% |
| re-pricing φ in the realised rewards (same admitted jobs, new μ̂) | about +0.7% |
| different admissions (8 decisions) | about +1.0% |

The snapshot was re-frozen (`tests/data/snapshot_synthetic.npz`). These arrays changed:
* Phase 0: `p0_r_tilde`, `p0_s_C`, `p0_lambda_star`, `p0_u0_sigma_rmax` (r_max).
* Online run: `on_phases`, `on_decisions`, `on_admit`, `on_R`, `on_R_total`, `on_payment`.

**Effect on the stationary results of week 33.** The new stationary run ("none" in week 34) uses μ̂. So its numbers
differ slightly from `reports/week33/final_results.md`. The differences are reported in
`reports/week34/scenarios_results.md`.

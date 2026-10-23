# Week 30: ablations, round 1 (single factors), small-cluster regime

**What:** each ablation on its own, against the theory model on `configs/regime_small_cluster.yaml`.
* 30k trace; ξ = 0.001, γ = 1.0, δ = 1/T³.
* 9 configurations × 5 thinning seeds (`online.thinning_seed_offset` = 2–6) = 45 DLENT-RT runs.
* **The cumulative round was not run.**
* Code: `cfg.ablation` (default off), `ErrorParams.thin_off` / `eps_D_thin`, and `build_window_state(refit_grid=…)`.
* Runner: `scripts/ablation_runner.py`, 4 workers under `caffeinate` on AC, 55 min. It caches each run in
  `reports/week30/ablations/runs/`.
* Notebook: `notebooks/05_ablations_small_cluster.ipynb`. Figures: `reports/week30/figs_ablations/`.

**Checks:**
* **With every flag off, the model is bit-identical.** The synthetic regression snapshot is bit-identical, the fast
  suite passes, and the baseline at offset 2 gives **40,892.14 $**.
* The references match `small_cluster_nofilter.md` to the cent: job MILP 120,742.02 (optimal, gap 9.9e-5); feasible
  ceiling 141,126.62; static Phase 0 44,716.94; static + thinning 31,285.95 (offset 2).
* The only online randomness is the thinning draws (`theory.py:69`). Phase 0, H, the stream, the bids, s_C (228.18),
  τ₀ (0, 0.0204) and c are identical in all 45 runs.
* **No Phase 0 change.**
* **Commit note:** the runs used the ablation code in the working tree on top of commit `eba0c63` (the run marker shows
  `eba0c63`). That code, all flags default off, is committed with this report.

**Definitions, from the code** (`error_params.py`, `monitor.py`, `theory.py`, `reinit.py`). A1b and A1c are applied to
the **thinning only**, as you chose. Globally, δ would also change every test radius, and ε_D would also change Test 2's
factor 1 − ε_D − ξ.

| ablation | what changes | what stays | guarantee given up (from the code) |
|---|---|---|---|
| A1a | p_U = 1 | the oracle, the physical gate, tests, ε_D | the Bernoulli thinning before the oracle (no ε_D / η safety margin on admissions) |
| A1b | ε_D inside p_U computed with δ = 1/T: 0.274 instead of 0.502 | tests (δ = 1/T³), Test 2 factor | the thinning's confidence level (1/T instead of 1/T³) |
| A1c | ε_D = 0 inside p_U (cap 1 − ξ = 0.999) | as A1b | the thinning's ε_D margin |
| A2a | ξ_override 0.01 / 0.05: ε_D 2.18 / 7.26, thinning cap 0.31 / 0.115, Test 1 margin ξ·c, Test 2 factor 1 − ε_D − ξ (−1.19 / −6.31) | γ, δ | ξ is forced, not data-derived; a larger ξ widens every ξ-dependent margin |
| A2b | γ_override 0.3 / 0.1: η and every test radius shrink (ε_Test2(d_max) 0.56 → 0.15 / 0.047; p_U at a phase start 0.19 → 0.38 / 0.53) | ξ, δ, ε_D | the tests' radii no longer use the reward bound γ |
| A3 | ReInit keeps the previous phase's grid, so every phase uses Phase 0's (test: the same object) | per-type statistics are re-estimated; hindsight and benchmarks keep their own refit | ReInit's re-estimation of the valuation grid (write-up Alg. 3, lines 2–6) |

## Ranking of the single-factor effects

Paired Δ = ablation − baseline with the same thinning seed: mean over 5 seeds, with a 95% t-interval (4 degrees of
freedom).

| rank | ablation | R mean ± std $ | paired Δ $ (95% CI) | Δ % of job MILP | beyond seed noise? | % of job MILP | % of static P0 |
|---|---|---|---|---|---|---|---|
| — | baseline (theory) | 39,739 ± 987 | — | — | — | 32.9 | 88.9 |
| 1 | **A1a thinning off** | 58,621 ± 0 | **+18,882** (+17,657 … +20,108) | +15.6 | **yes** | 48.6 | **131.1** |
| 2 | **A1c ε_D = 0 in p_U** | 57,655 ± 1,159 | **+17,916** (+16,489 … +19,343) | +14.8 | **yes** | 47.8 | **128.9** |
| 3 | **A1b δ = 1/T in p_U** | 46,675 ± 1,633 | **+6,936** (+4,743 … +9,129) | +5.7 | **yes** | 38.7 | **104.4** |
| 4 | A3 fixed grid | 41,345 ± 1,647 | +1,606 (−58 … +3,269) | +1.3 | no (borderline) | 34.2 | 92.5 |
| 5 | A2b γ = 0.1 | 40,700 ± 3,308 | +961 (−1,990 … +3,913) | +0.8 | no | 33.7 | 91.0 |
| 6 | A2b γ = 0.3 | 40,690 ± 1,100 | +951 (−956 … +2,857) | +0.8 | no | 33.7 | 91.0 |
| 7 | A2a ξ = 0.01 | 19,400 ± 1,746 | **−20,339** (−22,647 … −18,031) | −16.8 | **yes** | 16.1 | 43.4 |
| 8 | A2a ξ = 0.05 | 7,295 ± 552 | **−32,445** (−33,941 … −30,948) | −26.9 | **yes** | 6.0 | 16.3 |

**Reading (numbers only):**
* **The thinning dominates.** Removing it (A1a) or dropping only its ε_D margin (A1c) adds 18–19k $, about 15–16% of
  the job MILP, and is the only change that makes DLENT-RT beat static Phase 0 (131% / 129%). A1b, with half the ε_D
  margin, gives +6.9k.
* **Raising ξ is not a relaxation.** ε_D grows with ξ, so the thinning cap falls (0.665 → 0.31 → 0.115) and Test 2's
  factor turns negative. The forced ξ = 0.001 is already the most permissive value tested.
* **γ (0.3, 0.1) and the fixed grid do not change the reward beyond the seed noise.**
* **γ does change the dynamics.** At γ = 0.1 the run has 53 phases on average (26.8 Test 2, 15.4 Test 3), against 9.4,
  and τ > 0 in 32.6 phases.
* A1a's reward has zero spread across seeds, because without thinning the run uses no random numbers. Its interval
  reflects only the baseline's seed noise.

**Micro-benchmarks** (means over 5 seeds):
* Test 1 fires in 4 of 5 baseline runs and in 5 of 5 runs for A1a, A1b, A1c and A3.
* The thinning rejects 3,811 jobs in the baseline, 0 with A1a and 326 with A1c.
* Physical rejects rise from 501 to 1,144 with A1a, as more jobs reach the gate.
* The full tables are below.

## Proposal for the cumulative order (a proposal only; not run)

1. **Thinning first** (A1c or A1a). Use A1c (ε_D = 0 in p_U) if a thinning step should remain (cap 1 − ξ; η still
   throttles each phase start), or A1a for the upper reference.
2. **Then A3 (fixed grid):** the largest of the remaining effects, though within the noise.
3. **Then A2b (γ),** to see whether its effect on the tests turns into reward once the thinning is relaxed: shorter
   phases could matter more when admissions are not thinned.
4. **Keep ξ at 0.001.** In this direction ξ only tightens the model, so it is not an ablation to stack.

The interactions are unknown until the cumulative round is run.

---

# Appendix — generated tables (notebook 05)

## Ablations round 1: generated tables (notebooks/05_ablations_small_cluster.ipynb; do not edit by hand)

Notebook commit `eba0c633ca91d6734edee66910deffb108cfa85c`; runs at commit(s) ['eba0c633ca91d6734edee66910deffb108cfa85c']; regime `configs/regime_small_cluster.yaml`.

### References

| reference | $ |
|---|---|
| job MILP $ | 120,742.02 |
| feasible ceiling $ | 141,126.62 |
| static Phase 0 $ | 44,716.94 |

### Constants per ablation

| ablation | level | xi | gamma | eps_D (tests) | eps_D (thinning) | p_U cap | p_U at phase-1 start | p_U at t - tau = d_max | eta(d_max) | Test 2 factor 1-eps_D-xi | eps_Test2(d_max) | eps_Test3(d_max) | Test 1 margin xi*c CPU | Test 1 margin xi*c RAM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | theory | 0.001 | 1.00 | 0.502 | 0.502 | 0.665 | 0.187 | 0.638 | 0.032 | 0.497 | 0.56 | 0.883 | 0.000401 | 0.000262 |
| A1a_thinning_off | p_U = 1 | 0.001 | 1.00 | 0.502 | 0.502 | 1.00 | 1.00 | 1.00 | 0.032 | 0.497 | 0.56 | 0.883 | 0.000401 | 0.000262 |
| A1b_delta_1_T | delta = 1/T in p_U | 0.001 | 1.00 | 0.502 | 0.274 | 0.784 | 0.195 | 0.747 | 0.032 | 0.497 | 0.56 | 0.883 | 0.000401 | 0.000262 |
| A1c_epsD_0 | eps_D = 0 in p_U | 0.001 | 1.00 | 0.502 | 0.00 | 0.999 | 0.206 | 0.939 | 0.032 | 0.497 | 0.56 | 0.883 | 0.000401 | 0.000262 |
| A2a_xi_0.01 | xi = 0.01 | 0.01 | 1.00 | 2.18 | 2.18 | 0.311 | 0.141 | 0.305 | 0.032 | -1.19 | 0.56 | 0.883 | 0.00401 | 0.00262 |
| A2a_xi_0.05 | xi = 0.05 | 0.05 | 1.00 | 7.26 | 7.26 | 0.115 | 0.0784 | 0.114 | 0.032 | -6.31 | 0.56 | 0.883 | 0.0201 | 0.0131 |
| A2b_gamma_0.3 | gamma = 0.3 | 0.001 | 0.3 | 0.502 | 0.502 | 0.665 | 0.376 | 0.657 | 0.00959 | 0.497 | 0.148 | 0.132 | 0.000401 | 0.000262 |
| A2b_gamma_0.1 | gamma = 0.1 | 0.001 | 0.1 | 0.502 | 0.502 | 0.665 | 0.529 | 0.662 | 0.0032 | 0.497 | 0.0474 | 0.0313 | 0.000401 | 0.000262 |
| A3_fixed_grid | Phase 0 grid kept | 0.001 | 1.00 | 0.502 | 0.502 | 0.665 | 0.187 | 0.638 | 0.032 | 0.497 | 0.56 | 0.883 | 0.000401 | 0.000262 |

### Reward per ablation (5 thinning seeds)

| ablation | level | R mean $ | R std $ | paired delta mean $ | paired delta 95% lo $ | paired delta 95% hi $ | beyond seed noise (95% CI excludes 0) | % of job MILP (mean) | % of feasible ceiling (mean) | % of static Phase 0 (mean) |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | theory | 39,739.06 | 987.18 | 0.00 | nan | nan | False | 32.91 | 28.16 | 88.87 |
| A1a_thinning_off | p_U = 1 | 58,621.35 | 8.13e-12 | 18,882.30 | 17,656.55 | 20,108.05 | True | 48.55 | 41.54 | 131.09 |
| A1b_delta_1_T | delta = 1/T in p_U | 46,675.28 | 1,632.94 | 6,936.23 | 4,743.38 | 9,129.08 | True | 38.66 | 33.07 | 104.38 |
| A1c_epsD_0 | eps_D = 0 in p_U | 57,655.08 | 1,159.33 | 17,916.02 | 16,489.10 | 19,342.95 | True | 47.75 | 40.85 | 128.93 |
| A2a_xi_0.01 | xi = 0.01 | 19,399.89 | 1,745.56 | -20,339.17 | -22,647.42 | -18,030.91 | True | 16.07 | 13.75 | 43.38 |
| A2a_xi_0.05 | xi = 0.05 | 7,294.52 | 551.60 | -32,444.53 | -33,941.32 | -30,947.74 | True | 6.04 | 5.17 | 16.31 |
| A2b_gamma_0.3 | gamma = 0.3 | 40,689.63 | 1,100.13 | 950.58 | -955.58 | 2,856.73 | False | 33.70 | 28.83 | 90.99 |
| A2b_gamma_0.1 | gamma = 0.1 | 40,700.30 | 3,307.49 | 961.25 | -1,990.23 | 3,912.73 | False | 33.71 | 28.84 | 91.02 |
| A3_fixed_grid | Phase 0 grid kept | 41,344.72 | 1,646.85 | 1,605.66 | -57.50 | 3,268.83 | False | 34.24 | 29.30 | 92.46 |

### Ranking (paired delta vs baseline)

| ablation | level | paired delta mean $ | paired delta 95% lo $ | paired delta 95% hi $ | paired delta % of job MILP | beyond seed noise (95% CI excludes 0) |
|---|---|---|---|---|---|---|
| A1a_thinning_off | p_U = 1 | 18,882.30 | 17,656.55 | 20,108.05 | 15.64 | True |
| A1c_epsD_0 | eps_D = 0 in p_U | 17,916.02 | 16,489.10 | 19,342.95 | 14.84 | True |
| A1b_delta_1_T | delta = 1/T in p_U | 6,936.23 | 4,743.38 | 9,129.08 | 5.74 | True |
| A3_fixed_grid | Phase 0 grid kept | 1,605.66 | -57.50 | 3,268.83 | 1.33 | False |
| A2b_gamma_0.1 | gamma = 0.1 | 961.25 | -1,990.23 | 3,912.73 | 0.796 | False |
| A2b_gamma_0.3 | gamma = 0.3 | 950.58 | -955.58 | 2,856.73 | 0.787 | False |
| A2a_xi_0.01 | xi = 0.01 | -20,339.17 | -22,647.42 | -18,030.91 | -16.85 | True |
| A2a_xi_0.05 | xi = 0.05 | -32,444.53 | -33,941.32 | -30,947.74 | -26.87 | True |

### Micro-benchmarks (means over 5 seeds)

| ablation | level | phases (mean) | end timeout (mean) | end test1 (mean) | end test2 (mean) | end test3 (mean) | end horizon (mean) | phases with tau > 0 (mean) | Test 1 fires (runs) | admitted (mean) | thinning rejects (mean) | physical rejects (mean) | customer value $ (mean) | profit $ (mean) | carbon $ (mean) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | theory | 9.40 | 6.40 | 1.60 | 0.4 | 0.00 | 1.00 | 5.80 | 4 | 6,636.00 | 3,811.20 | 500.60 | 37,471.34 | 6,197.71 | -3,930.00 |
| A1a_thinning_off | p_U = 1 | 10.00 | 5.00 | 4.00 | 0.00 | 0.00 | 1.00 | 7.00 | 5 | 9,880.00 | 0.00 | 1,144.00 | 54,640.97 | 9,267.22 | -5,286.83 |
| A1b_delta_1_T | delta = 1/T in p_U | 10.60 | 6.40 | 2.60 | 0.6 | 0.00 | 1.00 | 7.00 | 5 | 7,736.00 | 2,575.40 | 698.00 | 43,722.56 | 7,387.05 | -4,434.32 |
| A1c_epsD_0 | eps_D = 0 in p_U | 11.20 | 5.20 | 5.00 | 0.00 | 0.00 | 1.00 | 7.20 | 5 | 9,596.00 | 326.20 | 1,050.60 | 53,511.45 | 9,329.82 | -5,186.20 |
| A2a_xi_0.01 | xi = 0.01 | 8.20 | 5.20 | 2.00 | 0.00 | 0.00 | 1.00 | 5.00 | 4 | 3,240.00 | 7,630.60 | 134.00 | 18,604.31 | 2,907.85 | -2,112.27 |
| A2a_xi_0.05 | xi = 0.05 | 8.00 | 5.60 | 1.40 | 0.00 | 0.00 | 1.00 | 4.80 | 3 | 1,216.60 | 9,705.80 | 34.40 | 7,207.86 | 1,007.76 | -921.10 |
| A2b_gamma_0.3 | gamma = 0.3 | 11.80 | 7.60 | 1.40 | 1.00 | 0.8 | 1.00 | 7.40 | 4 | 6,668.00 | 3,708.40 | 503.60 | 38,108.78 | 6,493.09 | -3,912.24 |
| A2b_gamma_0.1 | gamma = 0.1 | 53.20 | 9.60 | 0.4 | 26.80 | 15.40 | 1.00 | 32.60 | 2 | 6,491.40 | 3,611.00 | 474.40 | 39,908.16 | 4,654.82 | -3,862.67 |
| A3_fixed_grid | Phase 0 grid kept | 11.60 | 7.20 | 2.80 | 0.6 | 0.00 | 1.00 | 5.00 | 5 | 6,757.60 | 3,939.60 | 547.20 | 38,235.64 | 7,235.99 | -4,126.91 |

### Guarantee given up

| ablation | level | voids |
|---|---|---|
| baseline | theory | - |
| A1a_thinning_off | p_U = 1 | the Bernoulli thinning p_U = (1 - xi)/(1 + eps_U) that the theory model applies before the oracle; without it admissions are no longer thinned to leave the eps_D / eta safety margin |
| A1b_delta_1_T | delta = 1/T in p_U | the confidence level of eps_D used by the thinning: delta = 1/T instead of 1/T^3 (the tests keep 1/T^3) |
| A1c_epsD_0 | eps_D = 0 in p_U | the eps_D margin of the thinning (cap (1 - xi) instead of (1 - xi)/(1 + eps_D)); Test 2 keeps eps_D |
| A2a_xi_0.01 | xi = 0.01 | xi is forced (not data): a larger xi changes eps_D, the thinning cap, Test 1's margin xi*c and Test 2's factor 1 - eps_D - xi |
| A2a_xi_0.05 | xi = 0.05 | as xi = 0.01 |
| A2b_gamma_0.3 | gamma = 0.3 | gamma is forced (not data): a smaller gamma shrinks eta (thinning at phase start) and every test radius, so the tests' confidence statements no longer use the reward bound gamma |
| A2b_gamma_0.1 | gamma = 0.1 | as gamma = 0.3 |
| A3_fixed_grid | Phase 0 grid kept | ReInit's re-estimation of the valuation grid (write-up Alg. 3 lines 2-6): types are binned with Phase 0's edges in every phase (the per-type statistics are still re-estimated) |

### Every run

| ablation / seed | R $ | customer value $ | profit $ | carbon $ | admitted | thinning rejects | physical rejects | phases | end timeout | end test1 | end test2 | end test3 | end horizon | phases with tau > 0 | % of job MILP | % of feasible ceiling | % of static Phase 0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline / 2 | 40,892.14 | 37,633.19 | 7,317.65 | -4,058.71 | 6,657 | 3,744 | 537 | 9 | 6 | 1 | 1 | 0 | 1 | 6 | 33.87 | 28.98 | 91.45 |
| baseline / 3 | 38,685.04 | 36,623.54 | 6,061.74 | -4,000.24 | 6,606 | 3,822 | 523 | 9 | 6 | 2 | 0 | 0 | 1 | 6 | 32.04 | 27.41 | 86.51 |
| baseline / 4 | 39,173.90 | 38,518.08 | 4,508.08 | -3,852.26 | 6,589 | 3,742 | 476 | 11 | 9 | 0 | 1 | 0 | 1 | 7 | 32.44 | 27.76 | 87.60 |
| baseline / 5 | 39,254.62 | 37,110.45 | 6,119.57 | -3,975.40 | 6,653 | 3,905 | 451 | 11 | 6 | 4 | 0 | 0 | 1 | 6 | 32.51 | 27.82 | 87.78 |
| baseline / 6 | 40,689.58 | 37,471.46 | 6,981.49 | -3,763.38 | 6,675 | 3,843 | 516 | 7 | 5 | 1 | 0 | 0 | 1 | 4 | 33.70 | 28.83 | 90.99 |
| A1a_thinning_off / 2 | 58,621.35 | 54,640.97 | 9,267.22 | -5,286.83 | 9,880 | 0 | 1,144 | 10 | 5 | 4 | 0 | 0 | 1 | 7 | 48.55 | 41.54 | 131.09 |
| A1a_thinning_off / 3 | 58,621.35 | 54,640.97 | 9,267.22 | -5,286.83 | 9,880 | 0 | 1,144 | 10 | 5 | 4 | 0 | 0 | 1 | 7 | 48.55 | 41.54 | 131.09 |
| A1a_thinning_off / 4 | 58,621.35 | 54,640.97 | 9,267.22 | -5,286.83 | 9,880 | 0 | 1,144 | 10 | 5 | 4 | 0 | 0 | 1 | 7 | 48.55 | 41.54 | 131.09 |
| A1a_thinning_off / 5 | 58,621.35 | 54,640.97 | 9,267.22 | -5,286.83 | 9,880 | 0 | 1,144 | 10 | 5 | 4 | 0 | 0 | 1 | 7 | 48.55 | 41.54 | 131.09 |
| A1a_thinning_off / 6 | 58,621.35 | 54,640.97 | 9,267.22 | -5,286.83 | 9,880 | 0 | 1,144 | 10 | 5 | 4 | 0 | 0 | 1 | 7 | 48.55 | 41.54 | 131.09 |
| A1b_delta_1_T / 2 | 46,443.78 | 42,960.23 | 8,046.05 | -4,562.50 | 7,681 | 2,523 | 751 | 10 | 6 | 2 | 1 | 0 | 1 | 7 | 38.47 | 32.91 | 103.86 |
| A1b_delta_1_T / 3 | 48,086.91 | 43,852.83 | 8,703.51 | -4,469.44 | 7,807 | 2,598 | 727 | 10 | 6 | 3 | 0 | 0 | 1 | 7 | 39.83 | 34.07 | 107.54 |
| A1b_delta_1_T / 4 | 44,098.59 | 43,587.90 | 4,692.37 | -4,181.67 | 7,591 | 2,506 | 585 | 14 | 10 | 1 | 2 | 0 | 1 | 10 | 36.52 | 31.25 | 98.62 |
| A1b_delta_1_T / 5 | 46,664.67 | 43,357.46 | 7,834.17 | -4,526.96 | 7,781 | 2,624 | 719 | 10 | 6 | 3 | 0 | 0 | 1 | 6 | 38.65 | 33.07 | 104.36 |
| A1b_delta_1_T / 6 | 48,082.49 | 44,854.37 | 7,659.16 | -4,431.05 | 7,820 | 2,626 | 708 | 9 | 4 | 4 | 0 | 0 | 1 | 5 | 39.82 | 34.07 | 107.53 |
| A1c_epsD_0 / 2 | 59,109.22 | 53,747.38 | 10,694.34 | -5,332.50 | 9,620 | 314 | 1,153 | 9 | 5 | 3 | 0 | 0 | 1 | 6 | 48.95 | 41.88 | 132.19 |
| A1c_epsD_0 / 3 | 58,105.99 | 54,268.06 | 9,026.07 | -5,188.14 | 9,565 | 339 | 1,024 | 13 | 5 | 7 | 0 | 0 | 1 | 7 | 48.12 | 41.17 | 129.94 |
| A1c_epsD_0 / 4 | 57,588.40 | 53,702.92 | 9,072.19 | -5,186.71 | 9,638 | 306 | 1,079 | 10 | 5 | 4 | 0 | 0 | 1 | 7 | 47.70 | 40.81 | 128.78 |
| A1c_epsD_0 / 5 | 55,910.04 | 52,102.23 | 8,893.56 | -5,085.75 | 9,511 | 350 | 938 | 14 | 6 | 7 | 0 | 0 | 1 | 9 | 46.31 | 39.62 | 125.03 |
| A1c_epsD_0 / 6 | 57,561.74 | 53,736.68 | 8,962.96 | -5,137.90 | 9,646 | 322 | 1,059 | 10 | 5 | 4 | 0 | 0 | 1 | 7 | 47.67 | 40.79 | 128.72 |
| A2a_xi_0.01 / 2 | 21,234.96 | 19,781.15 | 3,845.48 | -2,391.68 | 3,318 | 7,545 | 153 | 10 | 6 | 3 | 0 | 0 | 1 | 6 | 17.59 | 15.05 | 47.49 |
| A2a_xi_0.01 / 3 | 21,252.00 | 19,882.38 | 3,476.50 | -2,106.88 | 3,320 | 7,614 | 130 | 11 | 6 | 4 | 0 | 0 | 1 | 8 | 17.60 | 15.06 | 47.53 |
| A2a_xi_0.01 / 4 | 18,277.36 | 18,999.75 | 1,362.93 | -2,085.32 | 3,254 | 7,595 | 120 | 8 | 5 | 2 | 0 | 0 | 1 | 5 | 15.14 | 12.95 | 40.87 |
| A2a_xi_0.01 / 5 | 17,468.99 | 17,021.86 | 2,450.11 | -2,002.98 | 3,130 | 7,706 | 128 | 6 | 4 | 1 | 0 | 0 | 1 | 3 | 14.47 | 12.38 | 39.07 |
| A2a_xi_0.01 / 6 | 18,766.14 | 17,336.40 | 3,404.21 | -1,974.47 | 3,178 | 7,693 | 139 | 6 | 5 | 0 | 0 | 0 | 1 | 3 | 15.54 | 13.30 | 41.97 |
| A2a_xi_0.05 / 2 | 6,979.20 | 7,493.61 | 603.36 | -1,117.77 | 1,298 | 9,695 | 44 | 10 | 6 | 3 | 0 | 0 | 1 | 6 | 5.78 | 4.95 | 15.61 |
| A2a_xi_0.05 / 3 | 8,063.91 | 8,353.13 | 755.33 | -1,044.55 | 1,227 | 9,518 | 37 | 10 | 8 | 1 | 0 | 0 | 1 | 7 | 6.68 | 5.71 | 18.03 |
| A2a_xi_0.05 / 4 | 6,894.60 | 6,908.29 | 857.87 | -871.56 | 1,168 | 9,780 | 31 | 6 | 5 | 0 | 0 | 0 | 1 | 3 | 5.71 | 4.89 | 15.42 |
| A2a_xi_0.05 / 5 | 6,840.70 | 6,584.82 | 1,004.95 | -749.08 | 1,189 | 9,763 | 26 | 8 | 4 | 3 | 0 | 0 | 1 | 5 | 5.67 | 4.85 | 15.30 |
| A2a_xi_0.05 / 6 | 7,694.22 | 6,699.46 | 1,817.30 | -822.54 | 1,201 | 9,773 | 34 | 6 | 5 | 0 | 0 | 0 | 1 | 3 | 6.37 | 5.45 | 17.21 |
| A2b_gamma_0.3 / 2 | 40,687.37 | 37,537.97 | 7,206.11 | -4,056.72 | 6,693 | 3,672 | 548 | 10 | 5 | 1 | 1 | 2 | 1 | 7 | 33.70 | 28.83 | 90.99 |
| A2b_gamma_0.3 / 3 | 42,331.26 | 38,832.89 | 7,342.83 | -3,844.46 | 6,623 | 3,644 | 465 | 17 | 11 | 2 | 2 | 1 | 1 | 11 | 35.06 | 30.00 | 94.66 |
| A2b_gamma_0.3 / 4 | 39,546.15 | 38,906.12 | 4,521.70 | -3,881.67 | 6,659 | 3,649 | 498 | 11 | 9 | 0 | 1 | 0 | 1 | 7 | 32.75 | 28.02 | 88.44 |
| A2b_gamma_0.3 / 5 | 39,843.48 | 37,614.93 | 6,223.91 | -3,995.35 | 6,623 | 3,797 | 494 | 13 | 8 | 3 | 1 | 0 | 1 | 7 | 33.00 | 28.23 | 89.10 |
| A2b_gamma_0.3 / 6 | 41,039.91 | 37,651.98 | 7,170.92 | -3,782.99 | 6,742 | 3,780 | 513 | 8 | 5 | 1 | 0 | 1 | 1 | 5 | 33.99 | 29.08 | 91.78 |
| A2b_gamma_0.1 / 2 | 43,251.65 | 39,695.37 | 7,448.56 | -3,892.28 | 6,508 | 3,614 | 455 | 50 | 7 | 1 | 25 | 16 | 1 | 30 | 35.82 | 30.65 | 96.72 |
| A2b_gamma_0.1 / 3 | 37,435.95 | 38,473.67 | 2,826.74 | -3,864.45 | 6,435 | 3,545 | 480 | 48 | 10 | 0 | 23 | 14 | 1 | 31 | 31.00 | 26.53 | 83.72 |
| A2b_gamma_0.1 / 4 | 39,031.19 | 40,111.95 | 2,826.47 | -3,907.23 | 6,542 | 3,562 | 488 | 57 | 10 | 1 | 27 | 18 | 1 | 36 | 32.33 | 27.66 | 87.29 |
| A2b_gamma_0.1 / 5 | 38,653.35 | 39,761.83 | 2,639.86 | -3,748.34 | 6,455 | 3,681 | 414 | 57 | 11 | 0 | 30 | 15 | 1 | 34 | 32.01 | 27.39 | 86.44 |
| A2b_gamma_0.1 / 6 | 45,129.37 | 41,497.97 | 7,532.46 | -3,901.06 | 6,517 | 3,653 | 535 | 54 | 10 | 0 | 29 | 14 | 1 | 32 | 37.38 | 31.98 | 100.92 |
| A3_fixed_grid / 2 | 41,488.61 | 38,083.30 | 7,598.99 | -4,193.67 | 6,800 | 3,885 | 584 | 10 | 6 | 2 | 1 | 0 | 1 | 5 | 34.36 | 29.40 | 92.78 |
| A3_fixed_grid / 3 | 38,713.68 | 36,564.83 | 6,515.44 | -4,366.59 | 6,679 | 3,906 | 594 | 10 | 6 | 3 | 0 | 0 | 1 | 4 | 32.06 | 27.43 | 86.57 |
| A3_fixed_grid / 4 | 41,065.73 | 39,196.68 | 5,959.55 | -4,090.50 | 6,797 | 3,864 | 517 | 12 | 9 | 1 | 1 | 0 | 1 | 6 | 34.01 | 29.10 | 91.83 |
| A3_fixed_grid / 5 | 42,697.40 | 38,989.93 | 7,764.90 | -4,057.43 | 6,737 | 4,047 | 515 | 12 | 7 | 4 | 0 | 0 | 1 | 4 | 35.36 | 30.25 | 95.48 |
| A3_fixed_grid / 6 | 42,758.17 | 38,343.44 | 8,341.08 | -3,926.36 | 6,775 | 3,996 | 526 | 14 | 8 | 4 | 1 | 0 | 1 | 6 | 35.41 | 30.30 | 95.62 |

# Week 30: does each λ still matter?

**Source:** `notebooks/src/lambda_sensitivity_check.py`; CSVs in `tables/lambda_*.csv`.

**Setup:**
* Frozen Phase 0 state on H of the committed `configs/default.yaml` (git HEAD: from_amax, σ = 0.3, n0 = 10,000,
  seed 42). The uncommitted edit in the working copy was not used. No online run.
* Populations: |H| = 10,000, |stream| = 15,308, with 2,673 of the 20,480 types populated.
* R_job = λ1·q·v + λ2·(φ_job − C_elec) − λ3·s_C·K. The type reward r̃ is the same expression on the per-type means.
* Baseline λ = (1, 1, 1) and s_C = 228.184 $/kgCO2.

## 1. s_C does not depend on λ

**No.** `dlent_rt/reward.py`, `type_reward`:
* lines 140–141 compute `cv = q_type * v_bar` and `profit = phi_bar - C_elec_bar` without λ;
* line 143 calls `carbon_price(cv, profit, K_bar, p, cfg.normalization.quantile)` on them;
* λ enters only afterwards, at line 150 (`f = [λ1·cv, λ2·profit, −λ3·carbon_usd]`).

In `carbon_price` (lines 104–126), D_q = (Q_q(CV) + Q_q(|Π|))/2, K_q = Q_q(K) and s_C = D_q / K_q, so neither
depends on λ either. Checked numerically: s_C was recomputed (s_C = None) for every λ setting of the sweep and equals
228.184 in all 15.

## 2. Magnitudes of the terms [$ per job]

Share = mean |term| / Σ over the four terms of mean |term|.

**H (10,000 jobs)**

| term | mean | mean \|·\| | p10 | p50 | p90 | share |
|---|---|---|---|---|---|---|
| q·v | 9.464 | 9.464 | 0.3255 | 1.408 | 21.51 | 29.01% |
| φ_job | −7.765 | 15.31 | −0.5074 | 0.4825 | 8.806 | 46.92% |
| C_elec | 0.001927 | 0.001927 | 1.70e-6 | 6.41e-5 | 0.003913 | **0.0059%** |
| s_C·K | 7.854 | 7.854 | 0.00711 | 0.2668 | 16.17 | 24.07% |

**Stream (15,308 jobs)**

| term | mean | mean \|·\| | p10 | p50 | p90 | share |
|---|---|---|---|---|---|---|
| q·v | 10.21 | 10.21 | 0.3781 | 1.929 | 20.94 | 47.81% |
| φ_job | 2.043 | 5.124 | −0.5612 | 0.673 | 7.377 | 24.00% |
| C_elec | 0.001421 | 0.001421 | 2.02e-6 | 4.95e-5 | 0.001796 | **0.0067%** |
| s_C·K | 6.018 | 6.018 | 0.009726 | 0.2419 | 8.185 | 28.19% |

The carbon term is about 4,000 times the electricity term on both populations. That is expected from the scaling:
carbon is scaled by s_C = 228 $/kgCO2, while electricity stays at its raw $ value.

## 3. One-at-a-time sweep

Definitions:
* (a), (b): computed over the 2,673 populated types.
* (c): computed over all H jobs.
* (d): the admitted set is the H jobs of types with r̃ > 0; it reports the mean R_job and the mean of each weighted
  component on that set.
* (e): the LP-RS optimum λ* on the Phase 0 state (p, r̃(λ), v, c), per tick and × T = 1,432,910. Capacity is slack
  under from_amax, so λ* = Σ_j p_j r̃_j⁺.

| λ varied | value | (a) types r̃>0 | (b) types flipped | (b) % | (c) H jobs flipped | (d) admitted | (d) mean R_job | (d) λ1·q·v | (d) λ2·(φ−C_elec) | (d) −λ3·s_C·K | (e) λ* $/tick | (e) λ*·T $ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| λ1 | 0 | 52.38% | 598 | 22.37% | 19.70% | 5,343 | 3.269 | 0 | 4.394 | −1.125 | 0.01402 | 20,095 |
| λ1 | 0.5 | 67.38% | 197 | 7.37% | 6.77% | 6,705 | 7.154 | 5.06 | 3.962 | −1.868 | 0.03851 | 55,186 |
| baseline | 1 | 74.75% | 0 | 0 | 0 | 7,400 | 11.35 | 10.24 | 3.653 | −2.539 | 0.06747 | 96,672 |
| λ1 | 2 | 83.69% | 239 | 8.94% | 8.24% | 8,371 | 19.74 | 20.25 | 3.384 | −3.893 | 0.1327 | 190,096 |
| λ1 | 5 | 92.56% | 476 | 17.81% | 16.81% | 9,259 | 46.31 | 48.52 | 3.004 | −5.215 | 0.3443 | 493,340 |
| λ2 | 0 | 80.10% | 313 | 11.71% | 10.27% | 7,719 | 7.706 | 9.709 | 0 | −2.002 | 0.04776 | 68,440 |
| λ2 | 0.5 | 75.91% | 103 | 3.85% | 3.84% | 7,525 | 9.427 | 10.01 | 1.655 | −2.24 | 0.05696 | 81,618 |
| λ2 | 2 | 74.67% | 124 | 4.64% | 4.82% | 7,452 | 15.15 | 10.44 | 8.058 | −3.347 | 0.09067 | 129,919 |
| λ2 | 5 | 75.53% | 265 | 9.91% | 11.53% | 7,673 | 26.99 | 10.44 | 21.06 | −4.511 | 0.1663 | 238,238 |
| λ3 | 0 | 90.80% | 429 | 16.05% | 16.47% | 9,247 | 13.46 | 9.983 | 3.476 | 0 | 0.09993 | 143,190 |
| λ3 | 0.5 | 81.78% | 188 | 7.03% | 7.23% | 8,195 | 11.86 | 10.32 | 3.648 | −2.106 | 0.07805 | 111,843 |
| λ3 | 2 | 67.04% | 206 | 7.71% | 7.80% | 6,715 | 10.45 | 9.993 | 3.689 | −3.235 | 0.05633 | 80,711 |
| λ3 | 5 | 52.15% | 604 | 22.60% | 18.37% | 5,611 | 8.943 | 9.358 | 3.712 | −4.127 | 0.04029 | 57,733 |

**Term removal at baseline λ (s_C frozen):**

| term removed | types flipped | H jobs flipped |
|---|---|---|
| electricity (C_elec = 0) | **0 (0%)** | **0 (0%)** |
| carbon (K = 0) | 429 (16.05%) | 1,647 (16.47%) |

λ2 multiplies φ_job − C_elec. Since removing C_elec flips nothing, every λ2 flip above comes from φ_job.

## 4. Normalization quantile (baseline λ)

| q | s_C $/kgCO2 | D_q $ | K_q kg | carbon share of mean \|reward\| (H) | Σ s_C·K / Σ (q·v + φ − C_elec) (H) | types r̃>0 | types flipped vs q = 0.90 |
|---|---|---|---|---|---|---|---|
| 0.75 | 315.9 | 5.501 | 0.01741 | 30.5% | 640.6% | 70.78% | 106 |
| 0.90 | 228.2 | 15.94 | 0.06984 | 24.07% | 462.7% | 74.75% | 0 |
| 0.95 | 188.7 | 28.42 | 0.1506 | 20.77% | 382.6% | 77.25% | 67 |

On H, Σ s_C·K exceeds the pre-carbon reward Σ (q·v + φ − C_elec) for every q, because Σ φ_job is negative on H
(mean −7.77).

## 5. Conclusion

* **λ1:** the decision set changes materially. Between 0 and 5, 7.4% to 22.4% of types and 6.8% to 19.7% of H jobs flip.
* **λ2:** the decision set changes (3.9% to 11.7% of types). All of it comes through φ_job, since removing C_elec
  flips 0 types and 0 jobs.
* **λ3:** the decision set changes materially (7.0% to 22.6% of types).
* **No λ is below the 1% flip threshold**, but **electricity is effectively invisible**: its share is 0.006% of the
  mean |reward| on H and on the stream, and removing it flips 0 decisions.
* The carbon term, scaled by s_C, carries 24% of the mean |reward| (H) and is not invisible at any q in {0.75, 0.90, 0.95}.

# Week 30, Stage A: reward-identity diagnosis

Read-only. Script: `notebooks/src/reward_identity_check.py`. Full output: `reports/week30/reward_identity_check_output.txt`.
Config: `configs/default.yaml` (from_amax capacity, xi_target 0.5, n0 = 10,000) with bid.strategy = lognormal and
bid.sigma = 0.3, theory model, seed 42, capacity scale 1.0. |H| = 10,000, |stream| = 15,308, T = 1,432,910.
This is the configuration of the executed notebook 03. The script reproduces its numbers to the cent.

## Root cause

**`notebooks/src/03_main_algorithm_analysis.py:302`** builds the yardstick as `phi_ref = st_ref.myerson.phi(S.v)`.
Under `bid.known_distribution: true`, `MyersonModel.phi(x)` is the virtual value of the **markup** distribution
LogNormal(−σ²/2, σ). The bid's virtual value is `phi_job(v, c_j) = c_j · phi(v / c_j)`. The notebook therefore applies
the markup φ to raw dollar bids:

* For bids with v ≪ 1 $, φ is hugely negative (min −7.0e7 $).
* For 15.2% of bids, the markup pdf at v is ≤ 1e-12. There `MyersonModel.phi` zeroes the inverse hazard, so φ = v.
* The mean φ is −9,705 $ against the correct +2.04 $.

The same call without c_j appears at three more places in notebook 03:

* `:278` (`run_static`, "R own phi")
* `:631` (KS: `lognorm.cdf` with the markup's μ, σ applied to raw bids)
* `:633` (`phi_k`)

It also appears in notebook 04 at `:186`, which dates from the pooled-fit era.

**Nothing in `dlent_rt/` is wrong.** Phase 0 (`phase0.py:480`), ReInit and hindsight (`reinit.py:90`) and the online
loop (`theory.py:151`) all call `phi_job(v, c_j)`. The online logged R equals the one job-level function on every
admitted job (max |diff| = 0).

**Why it slipped in:** the API made it easy. When I introduced the known distribution last week, I kept
`MyersonModel.phi(v)` public. It returns a valid-looking number for any argument, but in known mode that argument must
be a markup. Old call sites such as `st.myerson.phi(S.v)` therefore stopped meaning "φ of the bid" without raising.

**Proposed fix** (Stage C, needs your confirmation):

1. In notebook 03, compute every per-job reward through the single job-level reward function (Stage C module), which
   calls `phi_job(v, c_j)`. Remove `phi_ref` and `job_reward` from the notebook. Compute KS on markups v / c_j.
2. In `bid.py`, make the trap impossible. Calling `MyersonModel.phi(v)` on a known model should raise, or be renamed
   `phi_markup`, with `phi_job` the only public bid-level entry point. This leaves all Phase 0 numbers unchanged, since
   no `src/` call site uses `phi` on bids in known mode. The legacy pooled path (`known=False`) keeps `phi(v)`.
3. The "yardstick" then coincides with the logged reward, because there is one known φ. The separate column can go:
   test (b) of Stage C.

## Term-by-term identity (full-stream reference types, populated types, p-weighted)

| term | type level | job-level mean, `phi_job` (loop) | max abs diff | job-level mean, `phi(v)` (nb03) | max abs diff |
|---|---|---|---|---|---|
| SW = q·v | 10.208 | 10.208 | 5.7e-14 | 10.208 | 5.7e-14 |
| Π = φ − C_elec | 2.0418 | 2.0418 | 1.1e-13 | −9,704.7 | **7.0e7** |
| K = E·CI [kg] | 0.026373 | 0.026373 | 0 | 0.026373 | 0 |
| R = r̃ | 6.2319 | 6.2319 | 1.1e-13 | −9,700.5 | **7.0e7** |

First term that disagrees: **Π**, and only with the notebook's φ. The identity on the 2,401 reference types with
r_ref > 0 (12,621 jobs; τ* = [0, 0]) holds in both forms:

| quantity | $ |
|---|---|
| Σ_j N_j r_ref,j⁺ | 149,478.42 |
| Σ R_job over those jobs, `phi_job` | 149,478.42 (rel. diff 2e-16) |
| λ̂_ref · T (`summary["bench_global"]`, fluid OPT) | 149,478.42 |
| Σ R_job over those jobs, notebook `phi(v)` | −226,489.56 |

## Units and normalisation (all consistent in `dlent_rt/`)

* **p_j:** for the reference, N_j / T per tick, so λ̂·T = Σ N_j r_j x_j in $ per run. For Phase 0, N_j / H_span_steps
  (1,245,441, not T).
* **r̃:** $ per job, the type mean of R_job. The LP objective Σ p r is in $ per tick.
* **Duration:** enters only v_ij = A·d̄ (the LP constraint), never the reward.
* **φ:**
  * In the type reward, φ̄ is the mean of `phi_job` over the type's jobs, in Phase 0, ReInit and hindsight alike.
  * Online, φ is `phi_job` of the arrival.
  * `phi_gen` equals `phi_job` (max diff 7e-12 $).
  * The bin floor φ(L) is NaN per type under the known distribution and is used only for the payment.
* **s_C:** 228.184283 $/kgCO2 in Phase 0, in the reference state and in the loop.
* **Scale:** s_norm = 1 (quantile_linear has no divisor).
* **Type space:**
  * The reference grid is refit on the full stream: edges 0.141–383.3 $.
  * Phase 0's grid is fit on H: edges 0.150–304.1 $.
  * Phase k's grid is refit on phase k−1's arrivals.
  * The notebook's accuracy table puts the stream on phase k's own types. That is correct.
* **Jobs:** the reference uses all 15,308 outlier-filtered stream jobs, with H excluded. Phase 0 uses H only.

## Notebook totals re-scored with one job-level function

| admitted set | admitted | R, one function (`phi_job`) | nb03 yardstick (`phi(v)`) | nb03 logged |
|---|---|---|---|---|
| online DLENT-RT | 7,405 | **48,391.61** | −157,117.07 | 48,391.61 |
| static τ* (full stream, ORACLE) | 12,612 | **148,917.05** | −227,137 | – |
| static Phase 0 (H) | 10,226 | **54,216.40** | −610,451 | – |
| all jobs of r_ref > 0 types | 12,621 | 149,478.42 (= fluid OPT) | −226,490 | – |
| all jobs with R_job > 0 (no capacity) | 12,840 | **155,578.80** | −433,051 | 183,677 (nb "sum of R_ref>0") |

With the correct φ:

* **static τ\*** = 148,917 $, which is the fluid OPT minus 561 $. The gap is 9 jobs blocked by the physical gate.
* **Job-level upper bound** without capacity is 155,579 $, not 183,677 $. The notebook's job-level MILP (183,391 $) used
  R_ref, so it must be re-run. It will be ≤ 155,579 $.
* **The online run** collects 48,392 $, i.e. 32% of the fluid OPT. That gap is real (it admits 7,405 jobs, against
  12,612 for static τ*) and is not an accounting artefact. It is not diagnosed here.

## KS(F̂, F_emp) = 0.46 and "r rel err ≈ 1.0"

* **KS:** F̂ = LogNormal(μ = −0.045, σ = 0.3) is the known **markup** distribution, identical in every phase because it
  is never refit. The notebook compares it with the empirical CDF of **raw bids in $**, which gives 0.4609 in every
  phase. The correct comparison, against markups v / c_j, gives 0.0067 (5% critical value 1.36/√n = 0.0110).
* **r rel err:** compares r̃_k (the phase-k type reward, from the data that built the state) with r_ref (the stream
  jobs mapped to the same phase-k types). Both are **$ per job** type means, weighted by p_ref; neither is per tick.
  With the notebook's φ_ref, the error is the 7e7 $ artefact: mean abs err 9,727 $, rel 1.0002, sign mismatch 47%.
* **With the correct φ (phase 1, Phase-0 types):** mean abs err 16.1 $, rel 1.21, sign mismatch 27.6%. This is a
  **genuine** H-vs-stream difference in type means (thin types, nonstationarity), not a bug. It stays open for the
  regret analysis.
* **"φ error" metric:** 0 by construction under the known distribution, because there is one φ.

## Other flags (code wins)

* **Notebook 03, `.ipynb` vs `.py`:** the executed `.ipynb` was edited by hand (`N0_OVERRIDE = 10000`, scales
  `0.5,1.0`). `notebooks/src/03_main_algorithm_analysis.py` still says `N0_OVERRIDE = 20_000` and scales
  `0.5,…,3.0`, and its comment talks about n0 = 24,000 / 20,000. Rebuilding with `build.py` would silently revert
  the edits.
* **Notebook 03 "fluid OPT" label:** the λ̂_ref·T label and the `assert` at `:316` are correct, and they agree with the
  identity.
* **`.gitignore`:** it ignores `notebooks/src/`, `reports/` and `tests/`. The Stage C/D deliverables "on branch
  cleanup-week30" would not be committed unless `.gitignore` changes. That is your decision.

**Stopping here, as requested: no change to `src/` or to the notebooks.**

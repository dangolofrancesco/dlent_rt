# Week 30: the cluster-capacity regime (consolidated)

**Goal** (Kash, Oct 19): capacity must bind for *reasonable* policies (G2 job greedy, the job-level MILP), not only
for the theory. The theory model may then be very conservative on that cluster; assumptions are relaxed later in
ablations.

**Instance:**
* The committed `configs/default.yaml` (`git show HEAD:configs/default.yaml` into a temporary file; blob `68e7bd7`).
  The uncommitted working-copy edit is never read.
* Seed 42, σ = 0.3, n0 = 10,000, default grid (k* 128, 32 hardware profiles).
* Phase 0, rewards, defaults and configs are unchanged; no Phase 0 change to flag.

**New code:**
* `scripts/capacity_regime.py`, the regime table. Cache: `reports/week30/cluster_regime/regimes.csv`, 36 s.
* An `f0.2` config added to `notebooks/src/test_margins_check.py` (one DLENT-RT run, 279 s). It reproduces
  `capacity_scan_online_f0.2.json` exactly.

**Wording conflict.** The brief says "fraction of peak". Every earlier sweep, and this report, uses capacity mode
`fraction_of_concurrency_quantile` with q = 0.7, so c = f × (0.7 time-weighted quantile of H + stream concurrency).
The separate mode `fraction_of_peak` (f × the max concurrency) was never used.

## 1. Inventory of capacity-related work

| item | file(s), commit | config | one-line finding |
|---|---|---|---|
| binding capacity check | `notebooks/src/binding_capacity_check.py` (1a8bdf9); `configs/binding_capacity.yaml` (86779e6) | fraction_of_concurrency_quantile 0.30 / 0.60, no filter; plus an ρ = 0.6 run (f = 0.143) | At 0.30 / 0.60: peaks of 99.9% but mean occupancy of 21%, 338 physical rejects, τ > 0 in 2 of 11 phases, Test 1 never fires. At ρ = 0.6: Test 1 fires twice and τ > 0 in 5 of 8 phases. |
| capacity × grid sweep | nb03 Section 8; `notebooks/src/capacity_grid_sweep.py` (1976ef2); `capacity_grid_sweep.md` (4dc5641), `capacity_grid_sweep_tables.md` (c2ad514) | q = 0.7, f ∈ {0.3, 0.5, 1.0}, filter on, grids 128/32 and 64/16 | τ = 0 in every phase of every run. Test 1 fires only on the reduced grid. The reduced grid raises the type-based policies' reward by 11k to 32k $. |
| capacity scan (Phase 0) | `capacity_scan.py` (dd516c7), `capacity_scan.md` (e084a40) | q = 0.7, f from 0.02 to 0.3, filter on | The post-filter profitable load is at most 49% of c (H). No f reaches c/0.6. |
| online at f = 0.2 | `capacity_scan_online.py` (e084a40) | f = 0.2 | job OPT 112,304.68; DLENT-RT at 34% of job OPT; no test fires; τ = 0. |
| arrival-time compression | `time.arrival_time_scale` (4c57e00), `arrival_compression.md`; `arrival_compression_check.py` (untracked) | f = 0.2, k = 1–4 | T ÷ k, d_max/T up to 0.04, at most 10-tick collision moves, c unchanged. |
| intensity experiment | `intensity_experiment.md` (06e5550); `intensity_experiment.py` (untracked) | f = 0.2, k = 1–4 | τ₀ > 0 from k = 3; Test 1 fires from k = 2; job MILP at the 600 s limit for k ≥ 2. |
| clean grid sweep | `grid_sweep_clean.md` (e35f1e3); `grid_sweep_clean.py` (untracked) | f = 1.0, 32 profiles, k* 128 → 16, 3 thinning seeds | A coarser k* raises DLENT-RT, static and G1 well beyond the seed spread; G2 stays at about 99%. |
| bucketed MILP ladder | `bucketed_opt.md` (bc92f82), nb03 Sections 9–11 (c2ad514) | default and f = 0.2 | job OPT − fluid = +5.9k (default) and −6.9k (f = 0.2). The negative sign comes from the fluid relaxation (−11.0k $). |
| test definitions and margins | `test_definitions.md` (92e2d50); `test_margins_check.py` | default, binding 0.3/0.7, and f = 0.2 (**new here**) | Test 1 comes within 0.1–0.2% of firing on binding clusters; Test 2 fired once on default; Test 3 never fires on these clusters. |
| λ sensitivity | `lambda_sensitivity.md` (4640367) | default | Electricity is invisible (0.006% of the reward); every λ flips at least 1% of types. |
| snapping | `snapping_check.md` (9cafc00) | default | A is the snapped profile everywhere; 95% of jobs move by more than 5%. |
| **never run before** | — | — | **Windowed load / burstiness** was never computed: it is first done here (§3c). **ξ_data** was reported only incidentally (binding check, test_definitions) and is tabulated per regime here (§2C). **Tests vs the paper** cannot be compared because the PDF is not in the repo: `test_definitions.md` states the code definitions and marks every point "to compare with paper". |

## 2. Regime comparison (seed 42)

**Rows:**
* **default:** from_amax, no filter.
* **f = …:** fraction_of_concurrency_quantile q = 0.7 with `drop_jobs_exceeding_capacity` ON.
* **k = …:** f = 0.2 plus arrival-time compression (c sized on the original ticks).

The capacity_scale knob of nb03 Section 6 is a c-multiplier, not intensity, so it is not a row. All values are $ over
the stream.

**Comparability.** The filtered rows drop jobs, so their stream (and s_C) differs from default: dollar values are
comparable within f ≤ 0.5 only up to the jobs removed, and the % columns are the comparable ones. The k rows share
the f = 0.2 jobs and rewards.

**Sources:**
* `computed now` = `scripts/capacity_regime.py`;
* DLENT-RT comes from earlier runs: default from `tests/data/snapshot_real.npz` / `nb03_numbers.md`; f = 1.0 / 0.5 /
  0.3 from `tables/table08_capacity_grid_sweep_full.csv`; f = 0.2 from `tables/capacity_scan_online_f0.2.json`; k ≥ 2
  from `tables/intensity_online.json`;
* the job MILP for k ≥ 2 comes from `intensity_online.json` (600 s limit; incumbents, gap 0.15–0.45%);
* everything else was computed now and matches the earlier reports wherever they overlap.

**A. Cluster, filter, does capacity bind for G2 / job MILP**

| regime | c CPU / RAM (cores / GB) | stream jobs dropped (%) | stream R_job⁺ removed | Σ R_job⁺ (no cap) $ | job MILP $ | **capacity cost** | G2 phys. rejects (% of jobs) | G2 time at U ≥ 95% c | job MILP − fluid OPT $ |
|---|---|---|---|---|---|---|---|---|---|
| default | 252.5 / 654.1 | 0 | 0 | 155,579 | 155,398 | **0.12%** | 14 (0.09%) | 0.07% | +5,919 |
| f = 1.0 | 171.3 / 446.5 | 0 | 0 | 155,579 | 154,637 | **0.61%** | 88 (0.57%) | 0.50% | +5,159 |
| f = 0.5 | 85.6 / 223.2 | 523 (3.4%) | 0.85% | 132,321 | 129,671 | **2.00%** | 241 (1.63%) | 1.02% | +2,227 |
| f = 0.3 | 51.4 / 133.9 | 977 (6.4%) | 3.86% | 123,758 | 118,745 | **4.05%** | 418 (2.92%) | 1.46% | −424 |
| f = 0.2 | 34.3 / 89.3 | 977 (6.4%) | 3.86% | 123,758 | 112,305 | **9.25%** | 996 (6.95%) | 3.67% | −6,864 |
| f = 0.2, k = 2 | 34.3 / 89.3 | 977 (6.4%) | 3.86% | 123,758 | 104,482 (cached) | **15.6%** | 1,636 (11.4%) | 7.74% | −13,593 |
| f = 0.2, k = 3 | 34.3 / 89.3 | 977 (6.4%) | 3.86% | 123,758 | 97,582 (cached) | **21.2%** | 2,358 (16.5%) | 12.9% | −15,592 |
| f = 0.2, k = 4 | 34.3 / 89.3 | 977 (6.4%) | 3.86% | 123,758 | 92,341 (cached) | **25.4%** | 2,813 (19.6%) | 16.5% | −15,550 |

* Capacity cost = (Σ R_job⁺ − job MILP) / Σ R_job⁺.
* "Removed" is computed with the filter OFF on the same c. At f = 0.3 and 0.2 the same jobs are dropped, because A
  takes only 32 snapped values (`snapping_check.md`).

**B. Policies as % of the job MILP; DLENT-RT phases; prices; types**

| regime | G2 | G1 | static P0 | DLENT-RT | fluid OPT | DLENT-RT phase ends (timeout / T1 / T2 / T3 / horizon) | τ₀ (Phase 0) | τ* (hindsight) | H jobs per populated type |
|---|---|---|---|---|---|---|---|---|---|
| default | 99.75 | 53.17 | 34.89 | 31.14 | 96.19 | 9 / 0 / 1 / 0 / 1 | (0, 0) | (0, 0) | 3.74 |
| f = 1.0 | 99.32 | 53.23 | 35.00 | 31.20 | 96.66 | 9 / 0 / 1 / 0 / 1 | (0, 0) | (0, 0) | 3.74 |
| f = 0.5 | 97.40 | 55.32 | 33.97 | 32.32 | 98.28 | 5 / 0 / 0 / 0 / 1 | (0, 0) | (0, 0) | 3.88 |
| f = 0.3 | 95.59 | 57.97 | 39.18 | 34.31 | 100.36 | 5 / 0 / 0 / 0 / 1 | (0, 0) | (0, 0) | 3.90 |
| f = 0.2 | 91.49 | 58.24 | 39.83 | 34.03 | 106.11 | 5 / 0 / 0 / 0 / 1 | (0, 0) | (0, 0) | 3.90 |
| f = 0.2, k = 2 | 87.53 | 57.48 | 41.45 | 36.48 | 113.0 | 3 / 3 / 0 / 0 / 1 | (0, 0) | (0, 0.0357) | 3.90 |
| f = 0.2, k = 3 | 82.87 | 56.75 | 43.74 | 39.76 | 116.0 | 2 / 4 / 0 / 0 / 1 | (0, 0.0741) | (0.0041, 0.0786) | 3.90 |
| f = 0.2, k = 4 | 79.09 | 54.78 | 44.45 | 39.51 | 116.8 | 4 / 3 / 0 / 0 / 1 | (0, 0.1000) | (0.0656, 0.0760) | 3.90 |

**C. Theory constants of the cluster** (`dlent_rt/models/core/error_params.py`; "forced" = the runs, with
`xi_override` = 0.001 and `gamma_override` = 1.0; "data" = the overrides off)

| regime | c_min (norm) | ξ data = max a_max/c | γ data | ε_D forced / data | 1 − ε_D − ξ forced / data | p_U cap forced / data | p_U at phase-1 start forced / data | Test 1 margin ξ·c (forced) | Test 1 threshold c − a_max (data), CPU / RAM |
|---|---|---|---|---|---|---|---|---|---|
| default | 2.555 | 0.500 | 14,304 | 0.502 / 52.5 | 0.497 / −52.0 | 0.665 / 0.0093 | 0.527 / 8.8e-5 | 0.1% of c | 1.973 / 1.278 (50% of c) |
| f = 1.0 | 1.744 | 0.737 | 14,304 | 0.502 / 75.0 | 0.497 / −74.7 | 0.665 / 0.0035 | 0.480 / 3.2e-5 | 0.1% of c | 0.703 / 0.467 |
| f = 0.5 | 0.872 | 0.742 | 14,304 | 0.502 / 75.5 | 0.497 / −75.2 | 0.665 / 0.0034 | 0.376 / 1.6e-5 | 0.1% of c | 0.376 / 0.225 |
| f = 0.3 | 0.523 | 0.642 | 14,304 | 0.502 / 66.0 | 0.497 / −65.6 | 0.665 / 0.0054 | 0.291 / 1.3e-5 | 0.1% of c | 0.324 / 0.188 |
| f = 0.2 | 0.349 | 0.962 | 14,304 | 0.502 / 96.1 | 0.497 / −96.0 | 0.665 / 0.00039 | 0.227 / 9.1e-7 | 0.1% of c | 0.057 / 0.013 (10.6% / 3.8% of c) |
| f = 0.2, k = 2 / 3 / 4 | 0.349 | 0.962 | 14,304 | 0.488 / 0.480 / 0.473 (forced); 91.8 / 89.2 / 87.4 (data) | ≈ 0.51–0.53 / ≈ −87 to −92 | 0.671–0.678 / 0.0004 | 0.228–0.229 / 9.1e-7 | 0.1% of c | as f = 0.2 |

ε_D changes with k only through δ = 1/T³ (T ÷ k).

## 3. Answers

### a) What still depends on c with ξ = 0.001 and γ = 1, and what "explodes"

From the code:
* **ξ:** forced, so it is independent of c (`error_params.make_error_params`, lines 130–131).
  * Without the override, ξ = max_i a_max,i / c_i (`phase0._system_constants`, line 210), which grows as 1/c.
  * The filter caps it at < 1: 0.96 at f = 0.2. Without the filter it reached 3.0 at 0.30 / 0.60 (binding check).
* **ε_D** = 2√(ξ·log(|I|/δ)) + 2ξ·log(|I|/δ) (`ErrorParams.eps_D`, lines 72–73). It depends on ξ, δ = 1/T³ and |I|
  only. With ξ forced it is **independent of c**: 0.502 in every k = 1 regime.
* **γ:** forced to 1. The data value is max(r_max, v_rate_max, d_max, c_max, r_max/v_min) (`_system_constants`,
  line 209) = **14,304 = d_max in every regime**, so it depends on c only through c_max, which never dominates.
* **c enters through c_min = min_i c_i** (`make_error_params`, line 133):
  * η(n) = γ/(c_min·√n) (line 68);
  * ε_A′(n) ∝ 2γ²/c_min (line 50).

  These feed:
  * p_U(t) = (1 − ξ)/(1 + ε_D + η(W_prev) + η(t − τ)) (`p_U` / `eps_U`, lines 89–94);
  * Test 2's ε̃ (γ·η(W_prev) and 2γ·η(n) in CI¹, lines 97–105);
  * Test 3's ε_A′ and λ̃ (lines 108–115).

  The **cap** (1 − ξ)/(1 + ε_D) does not depend on c.
* **Test 1 margin:** a_max,eff = ξ·c (`theory.py`, lines 66–67), checked as LHS > c − ξ·c (`monitor.py`, line 144).
  The threshold is 99.9% of c at any size.
* **Phase length and warm-up** do not depend on c: τ̄ = τ_prev + d_max (`monitor.py`, line 62) and the timeout at
  2·W_prev (line 128).
* **τ** is the LP-RS dual, which depends on c through the LP.

**Shrinking c by 10× or 100×** (default cluster; forced ξ = 0.001, γ = 1):

| quantity | c | c ÷ 10 | c ÷ 100 |
|---|---|---|---|
| η(d_max) | 0.0033 | 0.033 | 0.33 |
| ε_A′(d_max) | 0.063 | 0.63 | 6.3 |
| p_U at a phase start | 0.53 | 0.18 | **0.024** |
| ε_Test2(d_max) | 0.47 | 0.56 | 1.45 |
| ε_Test3(d_max) | 0.33 | 0.90 | **6.6** |
| ε_Test3(T) | 0.032 | 0.087 | 0.64 |

* So thinning collapses at the start of each phase, and Test 3, which compares λ̂ ≈ 0.07 $/tick, can no longer fire.
* With the data constants, ξ = a_max/c would itself grow ×10 or ×100 (≥ 5 > 1). Then 1 − ξ < 0, p_U ≤ 0 and nothing is
  ever admitted; even at the unshrunk default cluster, p_U at a phase start is 8.8·10⁻⁵ and the Test 2/3 thresholds
  are around 10⁶–10⁷.

**Yes, the "two parameters explode" statement is about ξ (= a_max/c) and γ:**
* ξ_data drives ε_D to 52–96, so 1 − ε_D − ξ is around −52 to −96 and the p_U cap is 0.0004–0.009 (table C).
* γ_data = 14,304 = d_max multiplies every radius, and its square enters ε_A′ and Test 3, giving thresholds around
  10⁶–10⁸.
* Both are why `xi_override` and `gamma_override` are forced (config comment: "FORCED, NOT DATA-JUSTIFIED").
* Shrinking c makes ξ_data worse; γ_data does not move, because d_max sets it.

### b) The tests as implemented, and their margins

Paper notation; the full derivation and code lines are in `test_definitions.md`. Phase m starts at τ_prev + 1, the
warm-up ends at τ̄ = τ_prev + d_max, and the tests use information up to u − 1.

* **Test 1** (`monitor.py:142–145`): Σ_{τ = τ̄+1}^{u−1} A_{i,j(τ)}·1(D(τ) ≥ u − τ) > c_i − a_max,eff,i.
  * It compares the resources held by jobs admitted in the phase after the warm-up with the capacity minus the
    effective largest request.
  * Margin, forced: 0.1% of c.
* **Test 2** (`monitor.py:119, 134–140`): (1/(u − τ̄ − 1))·Σ_{τ = τ̄+1}^{u−1} [(1 − ε_D − ξ)·λ̂^(τ_prev+1:τ) − R(τ)] >
  ε̃_Test2(u − 1).
  * It compares the scaled LP benchmark with the realised reward.
  * ε̃ = γ·ε_E + ε_F + γ·η(W_prev) + 4γ/√W_prev + CI¹(n).
* **Test 3** (`monitor.py:147, 171–202`): λ̂^(s+1:u−1) > λ̃(u − 1) + ε_Test3(u − 1 − s), for s ∈ τ_prev + {0, 1, 2,
  4, …}.
  * λ̃ = min_τ [λ̂^(τ_prev+1:τ) + ε_A′ + ε_B′].
  * ε_Test3 = ε_A′ + ε_B′ + ε_C′ + 2γ²/√n.

**Maximum statistic/threshold per phase.** A value > 1 means the test fires. For Test 3 the value is the monitor's
screening upper bound. The f = 0.2 margins are new here (`tables/test_margins_f0.2.csv`); the default margins come
from `tables/test_margins_default.csv`.

| phase | default: ended by | default: T1 / T2 / T3 | f = 0.2: ended by | f = 0.2: T1 / T2 / T3 |
|---|---|---|---|---|
| 1 | timeout | 0.149 / 0.013 / 0.182 | timeout | 0.570 / 0.015 / 0.059 |
| 2 | timeout | 0.514 / 0.017 / 0.139 | timeout | **0.990** / 0.013 / 0.046 |
| 3 | timeout | 0.763 / 0.024 / 0.822 | timeout | **0.997** / 0.123 / 0.073 |
| 4 | timeout | 0.732 / 0.061 / 0.280 | timeout | **0.984** / 0.007 / 0.061 |
| 5 | timeout | 0.633 / 0.027 / 0.369 | timeout | **0.995** / 0.030 / 0.182 |
| 6 | **test2** | 0.068 / **1.110** / 0.242 | horizon | **0.999** / 0.115 / 0.277 |
| 7 | timeout | 0.749 / 0.155 / 0.407 | – | – |
| 8 | timeout | 0.464 / 0.253 / 0.442 | – | – |
| 9 | timeout | 0.655 / 0.100 / 0.400 | – | – |
| 10 | timeout | 0.973 / 0.080 / 0.378 | – | – |
| 11 | horizon | 0.903 / 0.236 / 0.513 | – | – |

* At f = 0.2, Test 1 comes within 0.1–1.6% of firing in phases 2–6 but never fires.
* Test 2 stays at ≤ 0.12 and Test 3 at ≤ 0.28. Tests 2 and 3 are far from firing because their radii (0.47–0.73
  $/tick at n = d_max) exceed λ̂ (≈ 0.07 $/tick).

**Where the code may differ from the paper** (flagged, not fixed; the paper PDF is not in the repo):
* ε_E = ε_F = 0 (Exact LP-RS replaces Algorithm 2);
* γ and ξ are forced;
* a_max,eff = ξ·c instead of the data a_max;
* Test 3 uses a geometric s-grid and a monotone screen;
* the warm-up d_max applies to all three tests;
* jobs admitted during the warm-up and carried-over jobs are excluded from Test 1;
* c_min is in the trace's normalised units inside η and ε_A′;
* δ = 1/T³;
* ReInit refits the grid on each phase's bids (n_min = 80);
* ties between tests are broken alphabetically.

### c) Windowed load and burstiness (new)

**Profitable load** = the occupancy of every stream job with R_job > 0, all admitted, with no capacity: the load a
reasonable job-level policy would like to carry. Each entry is the peak of its window-w moving average, as % of c
(CPU / RAM). Windows are in ticks of 1 s; under compression k, one tick is k seconds of original time.

| regime | average (= fluid) | 1 day | 6 h | 1 h | instantaneous peak | peak/avg (inst / 1 h / 6 h / 1 day, max resource) | time with load > c |
|---|---|---|---|---|---|---|---|
| default | 14.9 / 15.4 | 31.9 / 33.1 | 51.7 / 48.1 | 110 / 72 | 245 / 124 | 16.5 / 7.4 / 3.5 / 2.1 | 0.17% |
| f = 1.0 | 22.0 / 22.6 | 47.0 / 48.5 | 76.2 / 70.4 | 163 / 106 | 361 / 182 | 16.5 / 7.4 / 3.5 / 2.1 | 1.7% |
| f = 0.5 | 27.7 / 29.9 | 65.5 / 68.7 | 87.7 / 88.1 | 146 / 148 | 213 / 271 | 9.1 / 5.3 / 3.2 / 2.4 | 6.1% |
| f = 0.3 | 37.9 / 41.3 | 78.9 / 79.3 | 104 / 107 | 189 / 221 | 252 / 366 | 8.9 / 5.4 / 2.8 / 2.1 | 11.1% |
| f = 0.2 | 56.9 / 61.9 | 118 / 119 | 157 / 160 | 284 / 332 | 379 / 549 | 8.9 / 5.4 / 2.8 / 2.1 | 25.1% |
| f = 0.2, k = 2 | 114 / 124 | 189 / 201 | 243 / 283 | 353 / 390 | 516 / 675 | 5.5 / 3.2 / 2.3 / 1.7 | 53.8% |
| f = 0.2, k = 3 | 171 / 186 | 264 / 284 | 329 / 356 | 503 / 508 | 695 / 949 | 5.1 / 2.9 / 1.9 / 1.5 | 73.7% |
| f = 0.2, k = 4 | 228 / 248 | 327 / 345 | 429 / 438 | 577 / 690 | 805 / 961 | 3.9 / 2.8 / 1.8 / 1.4 | 85.5% |

**This is why τ = 0 while per-tick capacity binds:**
* The LP-RS / fluid LP constrains the time-AVERAGE load, which stays below c up to f = 0.2: 57–62% of c at f = 0.2,
  and only 15% of c on default. So its capacity row is slack, and τ = τ* = 0.
* The load is very bursty: the 1-hour peak is 5–7 times the average, and the instantaneous peak 9–16 times. A
  per-tick policy therefore meets c often: at f = 0.2 the profitable load exceeds c 25% of the time, and G2
  physically rejects 7% of jobs.
* Only when the AVERAGE load reaches c, at k ≥ 2 (114–248%), does the fluid LP bind and τ* become > 0.

### d) Is the fluid OPT an upper bound?

The fluid OPT is a reference for type-based policies. It is not an upper bound on the job-level MILP:

| regime | job MILP − fluid OPT |
|---|---|
| default | +5,919 |
| f = 1.0 | +5,159 |
| f = 0.5 | +2,227 |
| f = 0.3 | −424 |
| f = 0.2 | −6,864 |
| k = 2 | −13,593 |
| k = 3 | −15,592 |
| k = 4 | −15,550 |

* At f ≥ 0.5 the fluid OPT is below the job MILP: per-job decisions beat type means (discretization).
* From f ≤ 0.3 it is ABOVE the job MILP: the average-occupancy relaxation ignores the per-tick capacity (§3c).
* Against the clean type-level benchmark, the bucketed MILP on the reference buckets (`bucketed_opt.md`, default and
  f = 0.2 only), the fluid OPT is an upper bound in both cases: by 160 $ and by 11,047 $.
* Every type-based policy (G1, static, DLENT-RT) is below the fluid OPT in every regime.

## 4. Recommendation (mine, from the evidence above)

**Recommended regime: f = 0.2** (fraction_of_concurrency_quantile, q = 0.7, `drop_jobs_exceeding_capacity` on, no
compression). It is the smallest change in this table with a **capacity cost ≥ 5%**: 9.25%, while f = 0.3 has 4.05%.

**Why:**
* G2 physically rejects 7% of its jobs and spends 3.7% of the time at ≥ 95% of c.
* The job MILP falls to 112.3k $, against the 123.8k $ of profitable reward on the filtered stream.
* G2 reaches 91.5% of the job MILP, so capacity matters for a reasonable policy.

**Candidates side by side:**

| candidate | what is lost | what is fixed by hand | what still gives τ = 0 |
|---|---|---|---|
| **f = 0.2** | 977 stream and 1,423 H jobs (6.4% / 14.2%), **3.9% of Σ R_job⁺**. 123.8k $ of profitable reward kept (> 100k). \|H\| = 8,577; still about 3.9 H jobs per populated type, the same as default (the k* 128 grid, not the cluster, sets this). | ξ = 0.001 (data 0.96), γ = 1 (data 14,304). | τ₀ = τ* = 0 and τ = 0 in every DLENT-RT phase: the average load is 57–62% of c. No test fires; Test 1 is at 0.99–0.999. |
| f = 0.3 | the same jobs as f = 0.2 (3.9% of the reward) | the same | capacity cost only 4.05%, G2 at 95.6%: borderline. τ = 0. |
| f = 0.5 | 523 jobs, 0.85% of the reward | the same | capacity cost 2.0%: capacity barely binds for G2 / the MILP. τ = 0. |
| default / f = 1.0 | nothing | the same | capacity cost 0.1–0.6%: does not bind. |

**Arrival-time compression (k = 2–4) is only a stress test.** It is the only lever here that makes the fluid LP bind
(τ₀ > 0 from k = 3, τ* > 0 from k = 2, Test 1 firing), with capacity costs of 15.6–25.4%. But it **changes the instance
in a way that needs justification in the thesis**:
* arrivals come k times faster than in the trace, with durations unchanged;
* H's span shrinks, so p is ×k;
* d_max/T rises to 0.04;
* arrivals on the same tick are moved apart (up to 1,373 moved, by at most 10 ticks);
* for k ≥ 2 the job MILP hits the 600 s limit (gap up to 0.45%).

It is a synthetic intensity, not the observed workload.

**What the thesis should state for f = 0.2:**
* The theory model is very conservative there: DLENT-RT reaches 34% of the job MILP, with p_U about 0.23 at each phase
  start.
* τ = 0 is a property of the time-average LP on a bursty trace (§3c), not a bug.
* The two forced constants (ξ, γ) are what make the guarantees non-vacuous; with the data values, the thinning cap is
  0.04% (f = 0.2) to 0.9% (default) of arrivals and no test can fire.

## Numbers checked against the earlier reports

All match to the cent:
* job OPT 155,397.61 (default);
* fluid OPT 149,478.42;
* job OPT 112,304.68 (f = 0.2);
* DLENT-RT 48,391.61;
* G2, G1 and static recomputed here equal `table08_capacity_grid_sweep_full.csv`, `capacity_scan_online_f0.2.json` and
  `intensity_online.json`.

No disagreement was found.

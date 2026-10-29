# DLENT-RT

Online admission of cloud jobs with a learned LP oracle (DLENT, Zhang & Cheung 2025) on the Google cluster trace
2019, cell a. The reward is in dollars: customer value, profit and carbon. This page describes the final model
(week 33, decisions D1–D6). One configuration holds it: `configs/default.yaml`.

## Data (D3)

* **Fetch and build.** `scripts/data/fetch_v2.py` (BigQuery) and `scripts/data/build_v2.py` produce
  `data/batch_v2_300k.csv`: a uniform 300k sample of May 2019, one row per job. The request is the sum of the
  instances' first requests, in machines (1.0 = the largest machine). `scripts/data/fill_grid.py` fills the hourly
  price and carbon file.
* **Duration.** One duration per job: the effective duration d_j = CPU-hours_j / CPU_j, floored at 1 minute and cut
  at 24 h (jobs are kept). It is used for the time in the cluster, the bids and the energy. d_max = 24 h. There are
  no outlier filters.
* **History.** H = the jobs of the first 7 days. The stream is the rest.
* **Cluster.** D̄_i = Σ_{j∈H} r_ij d_j / |H| (machines; |H| = H's span in hours), and C_i = f · D̄_i. One cluster
  for every model.
* **Job quota.** A job with r_ij > Q_i on some resource is rejected before the market. Q_i = 0.2 · C_i of the cluster
  actually used: a fixed point on H, from the 99th percentile of H's requests, until Q moves by less than 1%.
* **f.** Rule, in priority order, over f ∈ {0.3, 0.35, 0.4, 0.45, 0.5}: (a) required, τ > 0 (CPU or RAM) in both the
  Phase 0 LP and fluid OPT; (b) among those, the static Phase 0 policy's "% of the time ≥ 95% full" closest to
  20–30%. The config holds the pick and a comment.
* **Job-level enlargement** (`capacity.job_level_enlarge`, off): if a job-level model rejects more than 50% of the
  positive-reward jobs for lack of space, C grows until it does not. The check is always computed.

## Bids and virtual value (D2)

```
U0          = γ1 cores + γ2 GB of the median job of H (ranked by that rate)      [$ per hour per job]
cost_base_j = (U0 (1 + ln q_j) + γ1 cores_j + γ2 GB_j) · d_j                    [$]
v_j         ~ LogNormal(ln cost_base_j − σ²/2, σ),  σ = 0.3                      (mean = cost_base_j)
φ_j         = cost_base_j · φ_m(v_j / cost_base_j),  φ_m(m) = m − (1 − F(m)) / f(m)
```

γ1 = 0.0332 $/vCPU-h, γ2 = 0.00445 $/GiB-h, cores = CPU × 64, GB = RAM × 256 (a declared machine size). U0 is computed
in Phase 0 and then fixed. The theoretical model refits σ on the markups v / cost_base (on H in Phase 0, then on
the previous phase's arrivals at every ReInit). The no-learning model keeps σ = 0.3.

## Reward (D1)

```
V_j  = v_j                       customer value (no priority factor; priority enters only the bid)
Π_j  = φ_j − C_elec_j            profit, C_elec = energy × price
K_j  = energy × carbon intensity [kg CO2]
R_j  = λ1 V_j + λ2 Π_j − λ3 s_C K_j,   all λ = 1   (linear)
s_C  = D_q / K_q,  D_q = (Q_0.9(V) + Q_0.9(|Π|)) / 2,  K_q = Q_0.9(K)   (p-weighted over H's types, Phase 0 only)
SW   = Σ_admitted (v_j − C_elec_j − s_C K_j)    a metric only
```

The type reward r̃ is the mean of R over the type's jobs. Augmented Chebyshev is only in the offline Pareto code
(`dlent_rt/pareto.py`). There is no carbon shadow price and no other normalization.

## Types and grid (D4)

Types = (bid bin, priority, hardware profile). The bid grid is geometric: L_k = v_min ρ^k,
ρ = (v_max / v_min)^(1/128), with v_min / v_max the smallest / largest bid (no fixed floor). The bins cover every
bid: bin 1 = [0, L_1), bin 128 = [L_127, +∞). Allocation and payment use the same edges. The theoretical model
rebuilds it at every ReInit; the no-learning model keeps Phase 0's grid. Types with no job in H are scored by the
dual rule with fallback estimates (hardware-profile means for duration and costs, bid-bin means for bid and φ).

## Online model (D5)

* **Theoretical model** (the default): frozen oracle k̂ = 1[r̃_j − τᵀ v_j > 0] with the phase's (r̃, τ, v); then
  Bernoulli thinning U(t) ~ Bern((1 − ξ)/(1 + ε̃_U(t))); then the physical gate. Tests 1–3 start d_max after the
  phase start; the timeout is 2 × the previous phase length. ξ = 0.001 and γ = 1 are forced; Test 1's threshold is
  (1 − ξ) c. Tests 2 and 3 use rewards in units of Phase 0's r_max, the units of their thresholds.
* **ReInit.** Arrival rates, bids, σ and requested resources come from all arrivals of the phase. Duration, energy
  and costs come from the admitted jobs only. Types with no admitted job carry the previous phase's values, then
  Phase 0's. The carbon price stays Phase 0's.
* **Practical model** = switches: `oracle.kind: live` (the job's own electricity cost and carbon at arrival) and
  `ablation.thinning_off: true`.
* **No-learning model** = the Phase 0 state forever: the static Phase 0 policy and the thinning
  (`models.policy.NoLearningPolicy`).

## Payments and benchmarks (D6)

* **Payment.** The threshold price: the lower edge of the lowest admitted bid bin for the same (priority, hardware
  profile) at the current τ, 0 if that is bin 1. Logged per job: payment p, customer surplus v − p, realised profit p − C_elec and IR
  (p ≤ v).
* **Benchmarks** (`dlent_rt/benchmarks.py`, `dlent_rt/theory_eval.py`): fluid OPT, fluid on feasible types, the
  bucketed MILP (one-week window, each MILP its own HiGHS instance; the fluid LP on a window counts only the volume
  inside the window), the static Phase 0 policy and the no-learning model. Practical only (code kept, not run in the
  model check): Greedy-Pos (G2), the type greedy (G1) and the job MILP. There is no separate FCFS class.

## Where things are

* `configs/default.yaml` — the one configuration.
* `notebooks/01_model_check.ipynb` — checks D1–D6 and shows the results
  (`dlent_venv/bin/python notebooks/src/build.py 01_model_check --execute`; long parts cached by
  `scripts/model_check_cache.py`).
* `notebooks/03_main_algorithm_analysis.ipynb`, `notebooks/04_pareto_lambda.ipynb` — older analyses, not rebuilt
  on the final model.
* `reports/week33/model_check.md` — the week-33 report.
* Tests: `dlent_venv/bin/python -m pytest -m "not slow"`.
* Everything before week 33 is in `archive/` and in the git tags `pre-cleanup-week32` and `pre-cleanup-week33`.

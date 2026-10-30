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
* **f.** Chosen on H ONLY (no stream job is read). Rule, in priority order, over f ∈ {0.3, 0.35, …, 0.6}:
  (a) required, τ > 0 (CPU or RAM) in the Phase 0 LP; (b) among those, replay H's jobs through the static Phase 0
  policy with capacity C, and pick the f whose "% of H's time with the cluster ≥ 95% full" is closest to 20–30%
  (ties to 25%). Each f uses an H-only dataset (`data.load_history`: quota fixed point, C, Q and the hardware
  profiles all from H). `scripts/select_f.py` runs it and prints the pick; the config holds the pick and a comment.
* **Job-level enlargement** (`capacity.job_level_enlarge`, off): if a job-level model rejects more than 50% of the
  positive-reward jobs for lack of space, C grows until it does not. The check is always computed.

## Bids and virtual value (D2)

```
U0          = γ1 cores + γ2 GB of the median job of H (ranked by that rate)      [$ per hour per job]
c_j         = U0 (1 + ln q_j) + γ1 cores_j + γ2 GB_j                             [$ per hour]
b_j         ~ LogNormal(ln c_j − σ²/2, σ),  σ = 0.3                              [$ per hour] (mean = c_j)
φ_j         = c_j · φ_m(b_j / c_j),  φ_m(m) = m − (1 − F(m)) / f(m)               [$ per hour]
```

**The bid is an hourly price.** It does not depend on the job's duration; c_j is known at arrival (priority and
request). γ1 = 0.0332 $/vCPU-h, γ2 = 0.00445 $/GiB-h, cores = CPU × 64, GB = RAM × 256 (a declared machine size). U0
is computed in Phase 0 and then fixed. The theoretical model refits σ on the markups b / c (on H in Phase 0, then on
the previous phase's arrivals at every ReInit). The no-learning model keeps σ = 0.3. φ_m is computed in log space
(m (1 − σ · Mills(z))) and floored at −10¹² for markups near 0.

## Reward (D1)

Decisions use type terms. **The bin sets only the value; the column (priority, profile) sets the costs:**

```
r̃_(k,q,h) = d̄_(q,h) · (L_k + φ_(q,h)(L_k)) − C̄elec_(q,h) − s_C · K̄_(q,h)       (λ = 1)
v_(k,q,h)  = v̄_(q,h) = mean request × mean duration of the column
φ_(q,h)(L) = c_(q,h) · φ_m(L / c_(q,h)),   c_(q,h) the column's hourly cost base
s_C        = D_q / K_q,  D_q = (Q_0.9(V) + Q_0.9(|Π|)) / 2,  K_q = Q_0.9(K)   (on these type terms, p-weighted, Phase 0)
```

L_k is the lower edge of bin k. d̄ is the column's ESTIMATED duration: every decision uses it, never the job's true
duration and never a per-bin duration. None of d̄, C̄elec, K̄, v̄, c depends on the bid.

The realised results use the job's true duration d_j (hours):

```
V_j  = b_j d_j                   customer value (no priority factor; priority enters only the bid)
Π_j  = φ_j d_j − C_elec_j        profit, C_elec = energy × price
K_j  = energy × carbon intensity [kg CO2]
R_j  = λ1 V_j + λ2 Π_j − λ3 s_C K_j,   all λ = 1   (linear)
SW   = Σ_admitted (b_j d_j − C_elec_j − s_C K_j)    a metric only
```

Augmented Chebyshev is only in the offline Pareto code (`dlent_rt/pareto.py`). There is no carbon shadow price and
no other normalization.

## Types and grid (D4)

Types = (bid bin, priority, hardware profile); a column = (priority, hardware profile). The bid grid is geometric on
HOURLY bids: L_k = v_min ρ^k, ρ = (v_max / v_min)^(1/128), with v_min / v_max the smallest / largest hourly bid of H
(no fixed floor). The bins cover every bid: bin 1 = [0, L_1), bin 128 = [L_127, +∞). Allocation and payment use
the same edges.

**The grid and the types are fixed from Phase 0**, in every model (theoretical, practical, no-learning). Per type
only the arrival rate p is estimated; the duration, energy, costs, request and cost base are column means. A column
with no job in H starts from fallback estimates (hardware-profile means over H). So every type has an r̃ from the
start.

**Discretization cost of the grid.** The grid changes only the value of a job (L_k instead of its hourly bid), so the
cost is measured with two job-level MILPs on the same jobs, capacity checks (at every arrival, true durations),
carbon map and s_C: MILP_true values each job at its own bid b_j, MILP_bin at L_k (in both the customer value and
φ). Reported: the value loss OPT_true − OPT_bin, the decision loss OPT_true − (MILP_bin's schedule at the true
rewards), and the grid's worst case 1 − 1/ρ (`benchmarks.discretization_milps`, `scripts/discretization_milp.py`).

## Online model (D5)

* **Threshold admission (Myerson: monotone admission + critical-value payment).** In a column, the margin
  r̃ − τᵀ v grows with k (only the value L_k + φ(L_k) changes with the bin, and φ_m is increasing). So
  k* = the lowest bin with margin > 0 (0 if none), and the admitted bins form an upper set. The code checks the upper
  set and raises if it fails (`mechanism.threshold_kstar`).
* **Reserve price per hour.** r_j = P_j · price(t_j): the job's power (kW, from its request, PUE included) times
  the electricity price at arrival. It does not depend on the bid.
* **Admission.** admit ⇔ bin(b_j) ≥ k* and b_j ≥ r_j and the thinning draw (theory only) and the job fits.
* **Theoretical model** (the default): the frozen oracle with the phase's (r̃, τ, v); Bernoulli thinning
  U(t) ~ Bern((1 − ξ)/(1 + ε̃_U(t))) after the decision; then the physical gate. Tests 1–3 start d_max after the
  phase start; the timeout is 2 × the previous phase length. ξ = 0.001 and γ = 1 are forced; Test 1's threshold is
  (1 − ξ) c. Tests 2 and 3 use rewards in units of Phase 0's r_max = the largest positive r̃ (the paper's bound on
  the rewards of admissible actions), fixed for the whole run.
* **ReInit.** It updates only the statistics; the grid and the types stay Phase 0's. p_j = N_j / W per type. Per
  column: σ and the cost base c come from all arrivals of the phase; duration, energy, costs and the request of the
  volume from the admitted jobs only. A column with no job in the phase keeps its previous values (carry-forward:
  previous phase, then Phase 0); a column never seen since Phase 0 keeps the fallback estimates. The carbon price
  stays Phase 0's.
* **Practical model** = switches: `oracle.kind: live` (every bin scored with the job's own electricity cost and
  carbon, estimated at arrival as its power × the column's d̄ × the price / carbon intensity at arrival; k* per job)
  and `ablation.thinning_off: true`.
* **No-learning model** = the Phase 0 state forever: the static Phase 0 policy and the thinning
  (`models.policy.ThresholdPolicy` with `thin`).

## Payments and benchmarks (D6)

* **Payment.** The hourly payment is p_j = max(L_k*, r_j), with L_k* the lower edge of bin k* (0 for bin 1). It is
  the lowest hourly bid that would still be admitted, so 0 < p_j ≤ b_j and bidding the true hourly value is
  optimal. The job pays p_j × the hours it runs. Logged per job: k*, r_j, the hourly and total payment, the customer
  surplus V − payment, the realised profit payment − C_elec, IR (p_j ≤ b_j) and the rejection reason ("bin below
  k*", "below reserve", "thinning", "no space").
* **Benchmarks** (`dlent_rt/benchmarks.py`, `dlent_rt/theory_eval.py`): fluid OPT and fluid on feasible types
  (upper bounds), the discretization cost (MILP_true vs MILP_bin on the first 7 days of the stream, each MILP its own
  HiGHS instance; see D4), the static Phase 0 policy and the no-learning model. Practical only (code kept, not run in
  the model check): Greedy-Pos and the type greedy (G1). There is no separate FCFS class.
* **The baselines use DLENT-RT's mechanism** (`models.policy.ThresholdPolicy`) on the Phase 0 state, never updated:
  threshold k* per column, hourly reserve, payment max(L_k*, r_j) × hours, IR checked. The static policy has no
  thinning; the no-learning model has the theoretical thinning p_U(W0, t); Greedy-Pos uses τ = 0. Fluid OPT, fluid
  on feasible types stay as upper bounds.

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

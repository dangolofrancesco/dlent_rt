# Week 30: a small cluster where the fluid LP binds (τ > 0)? Load-factor rule ρ

**Script:** `scripts/capacity_from_rho.py`. Commands: `scan` (fixed-point search), `curve` (the ρ(c) curve) and
`dlent`. Outputs in `reports/week30/small_cluster/` (`rho_curve.csv`).

**Setup:**
* The committed `configs/default.yaml` (`git show HEAD:configs/default.yaml` into a temporary file); seed 42.
* `xi_override` = 0.001 and `gamma_override` = 1.0, unchanged.
* `drop_jobs_exceeding_capacity` ON.
* No change to Phase 0, rewards, defaults or configs. **No Phase 0 change.**

## 1. The rule and the units

* ρ_i = Σ_{j: r̃_j > 0} p_j·v_ij / c_i, with v_ij = A_ij·d̄_j and p_j per tick: exactly the Phase 0 fluid LP's
  profitable load over c. The same ρ is used for CPU and RAM, so c_i = load_i / ρ.
* **Units:** `capacity.mode = "absolute"` uses `capacity.absolute.cpu` / `.ram` directly as c (`data._size_capacity`,
  lines 313–315). They are in the trace's **normalised units, the units of A**: 1.0 CPU = 64 cores and 1.0 RAM =
  256 GB (`EnergyModelCfg.spec_cpu_core` / `spec_ram_gb`).
* **Iteration:** c ← load(c)/ρ, with the filter rebuilt and Phase 0 re-run at each step.
  * load(c) changes only when the filter's drop set changes. A takes only the 32 snapped profile values, so load(c) is
    piecewise constant in c.
  * The start is c = load₀/ρ, where load₀ is the default cluster's profitable load (0.4947, 0.3373). That cluster's own
    ρ is 0.125 / 0.132.

## 2. Result: on the 30k trace, with the filter ON, no capacity reaches ρ ≥ 1

**The fixed-point iteration does not converge: it collapses** (ρ = 1.1; the other ρ behave the same):

| iteration | c (CPU, RAM; normalised) | load / c (CPU, RAM) | H / stream jobs dropped | \|H\| kept |
|---|---|---|---|---|
| 0 | 0.4498, 0.3066 | 0.31, 0.39 | 2,158 / 1,680 | 7,842 |
| 1 | 0.1252, 0.1085 | 0.23, 0.21 | 3,676 / 3,496 | 6,324 |
| 2 | 0.0257, 0.0211 | 0.40, 0.37 | 5,572 / 9,208 | 4,428 |
| 3 | 0.0094, 0.0071 | 0.65, 0.69 | 6,451 / 10,225 | 3,549 |
| 4 | 0.0055, 0.0044 | 0.16, 0.17 | 8,813 / 13,645 | 1,187 |
| 5 | 0.0008, 0.0007 | 0.003, 0.004 | 9,987 / 15,294 | 13 |
| 6 | Phase 0 fails: no H job is left | | | |

**Every shrink of c drops the largest jobs, and they carry most of the profitable load, so the load falls faster than
c.** The scan with ρ ∈ {1.1, 1.25, 1.5, 2.0} therefore has no rows to report, and the step-2 table cannot be filled:
there is no cluster at which load/c = ρ.

**The whole curve ρ(c)**, along c = α·load₀ with α from 10⁻³ to 1 (48 points, filter ON, `rho_curve.csv`), selected
rows:

| α | c CPU / RAM (cores / GB) | ρ CPU / ρ RAM | \|H\| / \|stream\| kept | stream jobs dropped | stream Σ R_job⁺ kept $ | τ₀ |
|---|---|---|---|---|---|---|
| 1.0 | 31.7 / 86.3 | 0.43 / 0.51 | 8,577 / 14,331 | 6.4% | 123,758 | (0, 0) |
| 0.64 | 20.4 / 55.6 | 0.43 / **0.55** | 7,842 / 13,628 | 11.0% | 106,603 | (0, 0) |
| 0.41 | 13.1 / 35.8 | 0.27 / 0.26 | 6,808 / 12,658 | 17.3% | 78,693 | (0, 0) |
| 0.31 | 9.8 / 26.6 | 0.23 / 0.24 | 6,407 / 12,181 | 20.4% | 65,639 | (0, 0) |
| 0.082 | 2.6 / 7.1 | 0.32 / 0.40 | 5,166 / 8,420 | 45.0% | 32,622 | (0, 0) |
| 0.0219 | 0.69 / 1.89 | 0.56 / **0.66** (max) | 3,549 / 5,083 | 66.8% | 15,444 | (0, 0) |
| 0.0078 | 0.25 / 0.68 | 0.24 / 0.29 | 1,183 / 1,660 | 89.2% | 4,920 | (0, 0) |

* **The largest ρ reached anywhere is 0.66 (RAM) / 0.59 (CPU),** at a 0.7-core cluster that drops two thirds of the
  jobs.
* **Within about 20% of jobs dropped, ρ ≤ 0.55.**
* τ₀ = 0 at every point: with ρ < 1 on both resources the LP admits every profitable type.
* "Σ R_job⁺ kept" is measured under each point's own s_C. At α = 1 the drop set is the f = 0.2 set (977 / 1,423 jobs).
  Measured as in the regime table (filter OFF at the same c) it removes **3.86%** of Σ R_job⁺, and the 123,758 $
  equals the f = 0.2 no-capacity value of `cluster_regime.md`.

**Why:**
* Load is heavy-tailed in A·D: a few large jobs carry most of it.
* `capacity_grid_sweep.md` already showed that at f = 0.3 / 0.2 the 6.4% of stream jobs that are dropped carry
  **29.6% (CPU) and 22.7% (RAM) of Phase 0's profitable load**.
* The filter removes exactly the jobs that would make the average load exceed c. Within this rule, "small cluster" and
  "binding fluid LP" are incompatible on this trace.

## 3. DLENT-RT

**Not run.** No ρ has τ₀ > 0, so there is no "smallest ρ with τ₀ > 0" to run on.

## 4. Recommendation (mine)

**No ρ meets the conditions** (≤ about 20% of Σ R_job⁺ removed, τ₀ > 0, τ > 0 in most phases, Test 1 firing):
* **τ₀ > 0 fails.** With the filter ON, ρ never exceeds 0.66 at any c, and is ≤ 0.55 while keeping about 90% of the
  jobs. The condition fails for the reason above, not for lack of search.
* **The others** (τ > 0 online, Test 1 firing) were not reached, because their prerequisite failed.

**Levers that do give τ₀ > 0**, from earlier reports, none applied here:
* **Arrival-time compression** k ≥ 3 at f = 0.2 (`intensity_experiment.md`):
  * τ₀ = (0, 0.074) at k = 3 and (0, 0.100) at k = 4;
  * τ > 0 in 5 of 7 and 8 of 8 phases;
  * Test 1 fires 4 and 3 times;
  * the reward removed is the same as at f = 0.2 (3.9%).

  It changes the instance (arrivals k times faster), which needs justification in the thesis.
* **Filter OFF:** excluded by you, because jobs larger than c can never be admitted and would fake a binding LP.
* **Clipping A at c instead of dropping** (a new rule) would change job sizes and needs your decision.

**No overrides are given for a "recommended cluster", because none qualifies.** For reference only, the point with the
highest ρ at about 11% of jobs dropped (α = 0.64) is reproduced by `capacity.mode: absolute`,
`capacity.absolute.cpu: 0.31833`, `capacity.absolute.ram: 0.21702`, `capacity.drop_jobs_exceeding_capacity: true`. It
has ρ = 0.43 / 0.55 and τ₀ = 0: it does not bind.

## 5. The 300k-row dataset: not used yet, decision needed

| file | rows | date range | columns |
|---|---|---|---|
| `data/batch_may2019_30k.csv` | 29,670 | 2019-05-01 00:05 – 06-01 00:09 | collection_id, request_time, priority, scheduling_class, CPU, RAM, actual_cpu_hours, actual_mem_hours, duration |
| `data/batch_may2019_300k.csv` | 296,797 | 2019-05-01 00:05 – 06-01 00:09 | collection_id, job_datetime, q_j, scheduling_class, A_cpu, A_ram, actual_cpu_usage, actual_ram_usage, D (hours), v_rate, v_total, phi_rate, phi_total, w_j_kw |
| `data/raw/raw_events_30k.csv`, `raw_usage_30k.csv` | — | — | raw inputs of the 30k file |

**Conflicts with the 30k data (code wins, so I stopped before choosing):**
1. **Column names differ.** Loading needs `data.columns` overrides: datetime = job_datetime, priority = q_j,
   requested_cpu = A_cpu, requested_ram = A_ram, duration_hours = "D (hours)".
2. **The requests are not the same quantity.** On the 7,139 jobs present in both files, the durations are identical,
   but A_cpu / CPU is 2.0 / 5.0 / 22.7 (10th / 50th / 90th percentile), and similarly for RAM. That is not a constant
   unit factor: the requests seem aggregated differently.
3. **The priority differs** (q_j ≠ priority on 56% of the matched jobs).
4. **The file has pre-computed bid columns** (v_rate, phi_rate, w_j_kw) that the model would ignore: it synthesises
   its own bids.

**What loads with the by-name mapping** (n0 = 10,000 unchanged, report only):
* 296,797 rows, 252,099 after the outlier policy (44,698 dropped; cutoffs A ≤ 15.2 / 8.05, D ≤ 3.97 h).
* |H| = 10,000, covering **1.7 days** (2019-05-01 00:05 – 05-02 16:04); the 30k H covers 14.4 days.
* |stream| = 242,099.
* T = 2,534,705 ticks; d_max = 14,309; d_max / T = 0.0057.
* Only 244,292 distinct arrival seconds, so about 52,500 same-second arrivals are moved to the next free tick.
* from_amax c = (22.7, 13.0) in the file's units.

**Not run on it yet:** ρ, Phase 0, the policies and the MILP. The column mapping, the meaning of A_cpu / A_ram, the
priority encoding and the value of n0 (10,000 = 1.7 days of H) have to be decided first.

**Runtime estimates** (asked before running):
* **DLENT-RT:** about 16× the arrivals and 1.8× the horizon of the 30k run, which took 5–25 min. With the work roughly
  linear in arrivals and window-LP solves, I estimate **1.5–7 h for one run**.
* **Job MILP:** 242k candidates with long durations would give a constraint matrix of roughly 10⁸ non-zeros. It is
  unlikely to solve within 600 s and may not fit in memory.

## Runtimes and checks

* **Runtimes:** the scan attempt (collapsed) 21 s; the iteration trace about 20 s; the ρ(c) curve 59 s; loading the
  300k file 20 s.
* **Numbers against earlier reports:** no job MILP was re-solved here. At α = 1 the kept Σ R_job⁺ (123,758 $) and the
  drop set (977 / 1,423) equal the f = 0.2 values. No disagreement was found.

# Practical model: the single ablations (f = 0.5, seed 42)

Each run changes **one** switch of the theoretical model. Everything else is the same as the final theory run:
f = 0.5, `fit_on: "H"`, seed 42, the same data, the same bids and the same thinning draws. The reference is the
theoretical run of commit eae298b (`reports/week33/cache/theory_run.pkl`). The no-learning, static and fluid OPT
rows come from `benchmarks_f.pkl` at the same f. Greedy-Pos (DLENT-RT's mechanism with τ = 0: admit every bin with
r̃ > 0 that fits) comes from `greedy_pos.pkl` (`scripts/model_check_cache.py greedy`). Every table is rebuilt from
the caches by `scripts/ablations_tables.py` (raw output: `reports/week33/ablations_numbers.md`).

| Run | Switch (one change from the default) |
|---|---|
| thinning off | `ablation.thinning_off: true` (p_U = 1) |
| live oracle | `oracle.kind: live` (the job's own power × d̄ × price / carbon at arrival) |
| σ cumulative | `bid.sigma_update: cumulative` (σ refit on H's and every phase's markups pooled) |
| σ fixed | `bid.sigma_update: fixed` (Phase 0's σ, never updated) |
| stats by end reason | `stats.update: by_end_reason` (merge after timeout / Test 1, replace after Test 2 / 3) |

How the runs were done:
* Command: `caffeinate -i dlent_venv/bin/python scripts/run_ablations.py --single --workers 2`, at commit b0d0e8b.
* Each run used its own process.
* Each run took 171–205 min of compute. The Mac slept once during the runs (14:43–20:07 on 10-31), which only paused
  them.
* Caches: `reports/week33/cache/ablations/single_*.pkl`. Log: `reports/week33/logs/ablations_single.log`.

## 1. All runs

| | fluid OPT | static Phase 0 | Greedy-Pos | no-learning | **DLENT-RT (theoretical)** | thinning off | live oracle | σ cumulative | σ fixed | stats by end reason |
|---|---|---| --- |---|---|---|---|---|---|---|
| reward $ | 3,522.8 | 3,045.1 | 2,837.1 | 2,314.2 | **2,256.2** | **3,020.3** | **2,326.7** | 2,258.6 | 2,251.2 | 2,304.4 |
| % of fluid OPT | 100.0 | 86.4 | 80.5 | 65.7 | **64.0** | 85.7 | 66.0 | 64.1 | 63.9 | 65.4 |
| Δ vs theoretical $ | – | +788.9 | +580.9 | +58.1 | 0 | **+764.1** | **+70.5** | +2.4 | −5.0 | +48.2 |
| Δ vs theoretical % | – | +35.0 | +25.7 | +2.6 | 0 | **+33.9** | **+3.1** | +0.1 | −0.2 | +2.1 |
| Δ vs no-learning $ | – | +730.9 | +522.8 | 0 | −58.1 | +706.1 | +12.4 | −55.7 | −63.1 | −9.9 |
| customer value $ | 3,166.1 | 2,692.2 | 2,659.5 | 2,112.8 | 2,028.0 | 2,686.8 | 2,030.0 | 2,012.7 | 2,028.5 | 2,076.6 |
| profit (φd − C_elec) $ | 2,250.7 | 1,889.9 | 1,792.6 | 1,498.4 | 1,432.6 | 1,873.2 | 1,449.6 | 1,423.5 | 1,428.2 | 1,480.2 |
| carbon $ | −1,894.0 | −1,537.0 | −1,615.1 | −1,297.0 | −1,204.4 | −1,539.6 | −1,153.0 | −1,177.6 | −1,205.5 | −1,252.4 |
| social welfare $ | – | 1,146.9 | 1,035.6 | 808.7 | 817.0 | 1,138.7 | 870.8 | 828.7 | 816.5 | 817.4 |
| payments $ | – | 1,882.5 | 1,780.1 | 1,483.5 | 1,428.5 | 1,866.6 | 1,454.9 | 1,422.3 | 1,428.4 | 1,464.8 |
| admitted jobs | – | 142,410 | 141,894 | 103,154 | 100,028 | 141,613 | 100,523 | 98,617 | 99,789 | 100,844 |
| rejected: bin below k* | – | 75,832 | 55,244 | 75,832 | 83,556 | 78,619 | 84,506 | 86,393 | 83,659 | 81,010 |
| rejected: below reserve | – | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| rejected: thinning | – | 0 | 0 | 56,903 | 54,549 | 0 | 54,144 | 53,526 | 54,560 | 55,516 |
| rejected: no space | – | 23,233 | 44,337 | 5,586 | 3,342 | 21,243 | 2,302 | 2,939 | 3,467 | 4,105 |
| phases | – | – | – | – | 10 | 13 | 10 | 10 | 11 | 11 |
| end reasons (Test 1 / timeout / horizon) | – | – | – | – | 7 / 2 / 1 | 12 / 0 / 1 | 8 / 1 / 1 | 7 / 2 / 1 | 8 / 2 / 1 | 9 / 1 / 1 |
| Myerson: Σ(φd − C_elec) $ | – | 1,889.9 | 1,792.6 | 1,498.4 | 1,432.6 | 1,873.2 | 1,449.6 | 1,423.5 | 1,428.2 | 1,480.2 |
| Myerson: payments − C_elec $ | – | 1,874.1 | 1,771.3 | 1,476.4 | 1,421.9 | 1,858.2 | 1,448.7 | 1,415.9 | 1,421.9 | 1,457.9 |
| IR violations | – | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| upper-set violations | – | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| spot-check violations (1,600 re-runs) | – | – | – | – | 0 | 0 | 0 | 0 | 0 | 0 |
| runtime min | – | – | < 1 | – | 223 | 180 | 171 | 194 | 182 | 205 |

Test 2 and Test 3 never fired in any run. With the live oracle, k* is computed per job; the upper-set check is then
"admitted ⇔ bin ≥ the job's own k*" and the spot check recomputes the live k*.

## 2. What each switch does

* **Thinning off is by far the largest effect: +764 $ (+34%).**
  * It reaches 85.7% of fluid OPT, just below the static Phase 0 policy (86.4%).
  * The 54,549 thinned jobs are now admitted where they fit: 41,585 more jobs are admitted. 17,901 more are
    rejected for lack of space.
  * All 12 phase ends are now Test 1 (capacity). There are no timeouts: with more jobs admitted, the cluster fills
    faster.
* **The live oracle is the second: +70 $ (+3.1%).**
  * It is the only learning ablation that beats the no-learning model (+12 $).
  * It admits about the same number of jobs (100,523 vs 100,028), but different ones. The carbon cost falls by 51 $
    (−1,153 vs −1,204), and the profit rises by 17 $. Pricing each job at its own power and the grid at its arrival
    picks cleaner and cheaper jobs.
* **Statistics by end reason: +48 $ (+2.1%).**
  * After a timeout or Test 1, ReInit merges the old and new statistics instead of replacing them.
  * τ RAM then stays between 0.00095 and 0.0011 in every phase (theoretical run: 0.00008 to 0.0018). The prices are
    steadier and less noisy.
  * It is still 10 $ below the no-learning model.
* **σ update does not matter.**
  * σ cumulative gives +2 $ and σ fixed gives −5 $.
  * σ stays at 0.296–0.302 in every run, because the markups follow one known distribution (σ = 0.3). Refitting it
    changes almost nothing.

## 3. Per phase: the two most important ablations

Each row is a phase **of the theoretical run**. The other columns sum each run's rewards over the same time window,
so the windows are the same for every column. The ablations have their own phases, listed after the table.

| Phase | Hours | End (theoretical) | DLENT-RT $ | no-learning $ | Greedy-Pos $ | thinning off $ | Δ thinning off | live oracle $ | Δ live |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0–48.0 | timeout | 220.6 | 220.6 | 249.9 | 279.7 | +59.1 | 220.0 | −0.6 |
| 2 | 48.0–143.3 | Test 1 | 309.0 | 341.2 | 438.2 | 487.8 | +178.8 | 332.9 | +23.9 |
| 3 | 143.3–190.6 | Test 1 | 190.3 | 204.2 | 251.1 | 250.4 | +60.1 | 203.2 | +12.9 |
| 4 | 190.6–285.2 | timeout | 331.5 | 326.4 | 424.9 | 462.9 | +131.4 | 333.0 | +1.5 |
| 5 | 285.2–330.5 | Test 1 | 190.2 | 191.9 | 254.7 | 247.1 | +56.9 | 198.0 | +7.8 |
| 6 | 330.5–378.7 | Test 1 | 199.8 | 205.1 | 242.2 | 244.9 | +45.1 | 207.2 | +7.4 |
| 7 | 378.7–425.0 | Test 1 | 175.2 | 174.1 | 220.8 | 229.3 | +54.1 | 186.3 | +11.1 |
| 8 | 425.0–475.0 | Test 1 | 186.4 | 198.8 | 240.0 | 255.0 | +68.6 | 189.3 | +2.9 |
| 9 | 475.0–525.9 | Test 1 | 232.8 | 236.1 | 260.7 | 280.2 | +47.4 | 233.8 | +1.0 |
| 10 | 525.9–575.9 | horizon | 220.4 | 215.8 | 254.7 | 283.1 | +62.7 | 222.7 | +2.3 |
| **Total** | 575.9 h | | **2,256.2** | **2,314.2** | **2,837.1** | **3,020.4** | **+764.2** | **2,326.4** | **+70.2** |

(The totals match section 1 up to rounding.)

* **Thinning off wins in every window, by 20–58%.** Its largest gains are in the two long windows (phases 2 and 4,
  each about 95 h), where the thinning had the most jobs to remove.
* **The live oracle ties in phase 1** (same state, so only the per-job costs differ). It wins in all 9 later
  windows, by 1–24 $. Most of its gain is in phases 2, 3 and 7.

**Their own phases:**

| Thinning off: phase | Start h | Length h | End | Arrivals | Admitted | τ CPU | τ RAM | Reward $ |
|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 47.6 | Test 1 | 15,666 | 9,910 | 0 | 0.00095 | 278.9 |
| 2 | 47.6 | 52.4 | Test 1 | 18,373 | 11,880 | 0 | 0.00137 | 260.9 |
| 3 | 100.0 | 47.1 | Test 1 | 17,274 | 11,088 | 0.00035 | 0.00023 | 248.1 |
| 4 | 147.1 | 42.6 | Test 1 | 13,046 | 8,372 | 0 | 0.00102 | 226.9 |
| 5 | 189.7 | 47.5 | Test 1 | 11,028 | 6,811 | 0 | 0.00068 | 224.6 |
| 6 | 237.1 | 52.8 | Test 1 | 12,373 | 8,549 | 0.00003 | 0.00065 | 267.8 |
| 7 | 289.9 | 45.1 | Test 1 | 18,809 | 11,724 | 0 | 0.00027 | 246.1 |
| 8 | 335.1 | 48.2 | Test 1 | 27,861 | 16,794 | 0.00057 | 0.00052 | 248.5 |
| 9 | 383.3 | 46.3 | Test 1 | 25,807 | 13,740 | 0 | 0.00046 | 226.0 |
| 10 | 429.6 | 44.2 | Test 1 | 25,784 | 14,780 | 0.00021 | 0.00042 | 225.7 |
| 11 | 473.9 | 44.9 | Test 1 | 16,480 | 8,892 | 0.00036 | 0.00018 | 251.7 |
| 12 | 518.8 | 46.6 | Test 1 | 30,654 | 14,639 | 0.00029 | 0.00080 | 265.2 |
| 13 | 565.4 | 10.5 | horizon | 8,320 | 4,434 | 0.00028 | 0.00094 | 49.9 |

| Live oracle: phase | Start h | Length h | End | Arrivals | Admitted | τ CPU | τ RAM | Reward $ |
|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 48.0 | timeout | 15,804 | 7,311 | 0 | 0.00095 | 220.0 |
| 2 | 48.0 | 94.3 | Test 1 | 33,567 | 14,796 | 0 | 0.00167 | 322.7 |
| 3 | 142.3 | 72.0 | Test 1 | 20,747 | 9,484 | 0 | 0.00105 | 313.9 |
| 4 | 214.4 | 89.9 | Test 1 | 20,960 | 9,494 | 0.00002 | 0.00113 | 313.0 |
| 5 | 304.3 | 53.3 | Test 1 | 28,300 | 13,106 | 0 | 0.00095 | 237.1 |
| 6 | 357.5 | 47.0 | Test 1 | 24,292 | 9,923 | 0.00032 | 0.00071 | 190.1 |
| 7 | 404.5 | 66.8 | Test 1 | 41,035 | 15,840 | 0.00072 | 0.00014 | 249.1 |
| 8 | 471.3 | 47.7 | Test 1 | 17,994 | 6,818 | 0.00023 | 0.00051 | 231.7 |
| 9 | 519.1 | 51.8 | Test 1 | 34,896 | 12,354 | 0.00032 | 0.00148 | 237.3 |
| 10 | 570.9 | 5.0 | horizon | 3,880 | 1,397 | 0.00051 | 0.00082 | 11.7 |

## 4. Reading

* **The thinning is the main cost of the theoretical model:** about 34% of its reward. Without it, DLENT-RT is
  within 1% of the static Phase 0 policy (3,020 vs 3,045 $). With it, DLENT-RT is 2.5% below the no-learning model.
* **Learning itself adds little.** The best learning switch is the live oracle, at +3.1%. Merging statistics by end
  reason adds +2.1%. σ does not matter.
* **The mechanism holds in every run:** IR, upper-set and spot-check violations are all 0. Payments − C_elec stays
  within 1.6% of Σ(φd − C_elec), and the reserve never binds.
* The cumulative runs are in section 5.

## 5. Cumulative: the final practical model

The switches are added one at a time. σ update is left out: it had no effect (section 2).
* **C1** = thinning off. This is the single thinning-off run: same config, so its cache was reused, not rerun.
* **C2** = C1 + live oracle.
* **C3** = C2 + `stats.update: by_end_reason`. **This is the final practical model.**

The runs used seed 42, f = 0.5, `fit_on: "H"`, and the same data and caches. Command:
`caffeinate -i dlent_venv/bin/python scripts/run_ablations.py --cumulative --workers 2` (commit c81ae24). C2 and C3
took 183 and 184 min. Log: `reports/week33/logs/ablations_cumulative.log`.

| | DLENT-RT (theoretical) | C1 = thinning off | C2 = C1 + live oracle | **C3 = C2 + stats by end reason** | static Phase 0 | Greedy-Pos | no-learning | fluid OPT |
|---|---|---|---|---|---|---|---|---|
| reward $ | 2,256.2 | 3,020.3 | 3,069.4 | **3,094.7** | 3,045.1 | 2,837.1 | 2,314.2 | 3,522.8 |
| **step Δ of this switch $** | – | **+764.1** | **+49.1** | **+25.3** | – | – | – | – |
| % of fluid OPT | 64.0 | 85.7 | 87.1 | **87.8** | 86.4 | 80.5 | 65.7 | 100.0 |
| Δ vs theoretical $ | 0 | +764.1 | +813.3 | **+838.6** | +789.0 | +580.9 | +58.1 | – |
| customer value $ | 2,028.0 | 2,686.8 | 2,692.9 | 2,714.3 | 2,692.2 | 2,659.5 | 2,112.8 | 3,166.1 |
| profit (φd − C_elec) $ | 1,432.6 | 1,873.2 | 1,889.1 | 1,910.3 | 1,889.9 | 1,792.6 | 1,498.4 | 2,250.7 |
| carbon $ | −1,204.4 | −1,539.6 | −1,512.5 | −1,529.8 | −1,537.0 | −1,615.1 | −1,297.0 | −1,894.0 |
| social welfare $ | 817.0 | 1,138.7 | 1,172.2 | 1,176.2 | 1,146.9 | 1,035.6 | 808.7 | – |
| payments $ | 1,428.5 | 1,866.6 | 1,877.1 | 1,907.2 | 1,882.5 | 1,780.1 | 1,483.5 | – |
| admitted jobs | 100,028 | 141,613 | 144,744 | 144,683 | 142,410 | 141,894 | 103,154 | – |
| rejected: bin below k* | 83,556 | 78,619 | 74,842 | 73,337 | 75,832 | 55,244 | 75,832 | – |
| rejected: below reserve | 0 | 0 | 0 | 0 | 0 | 0 | 0 | – |
| rejected: thinning | 54,549 | 0 | 0 | 0 | 0 | 0 | 56,903 | – |
| rejected: no space | 3,342 | 21,243 | 21,889 | 23,455 | 23,233 | 44,337 | 5,586 | – |
| phases | 10 | 13 | 13 | 13 | – | – | – | – |
| end reasons (Test 1 / timeout / horizon) | 7 / 2 / 1 | 12 / 0 / 1 | 12 / 0 / 1 | 12 / 0 / 1 | – | – | – | – |
| Myerson: Σ(φd − C_elec) $ | 1,432.6 | 1,873.2 | 1,889.1 | 1,910.3 | 1,889.9 | 1,792.6 | 1,498.4 | – |
| Myerson: payments − C_elec $ | 1,421.9 | 1,858.2 | 1,868.9 | 1,898.9 | 1,874.1 | 1,771.3 | 1,476.4 | – |
| IR violations | 0 | 0 | 0 | 0 | 0 | 0 | 0 | – |
| upper-set violations | 0 | 0 | 0 | 0 | 0 | 0 | 0 | – |
| spot-check violations (1,600 re-runs) | 0 | 0 | 0 | 0 | – | – | – | – |

* **Thinning off does most of the work** (+764 $).
* **The live oracle adds +49 $ on top of it.** Alone it added +70 $. It lowers the carbon cost (−1,512 vs −1,540) and
  admits 3,131 more jobs.
* **The merged statistics add +25 $.** Alone they added +48 $. Profit and payments rise, and the carbon cost rises a
  little.
* The two switches still add up after thinning off, but each gives less than it did alone.
* The mechanism holds in all three runs: IR, upper-set and spot-check violations are all 0. Payments − C_elec stays
  within 1.1% of Σ(φd − C_elec).

**Per phase: C3 vs static vs no-learning.** Each row is one of the theoretical run's phases, used as a fixed time
window. Each column sums one run's rewards over that window.

| Phase | Hours | DLENT-RT (theoretical) $ | no-learning $ | static Phase 0 $ | Greedy-Pos $ | **C3** $ | C3 − static | C3 − Greedy-Pos |
|---|---|---|---|---|---|---|---|---|
| 1 | 0–48.0 | 220.6 | 220.6 | 279.7 | 249.9 | 282.6 | +2.9 | +32.7 |
| 2 | 48.0–143.3 | 309.0 | 341.2 | 485.7 | 438.2 | 488.0 | +2.3 | +49.8 |
| 3 | 143.3–190.6 | 190.3 | 204.2 | 260.1 | 251.1 | 261.7 | +1.6 | +10.6 |
| 4 | 190.6–285.2 | 331.5 | 326.4 | 462.6 | 424.9 | 462.5 | −0.1 | +37.6 |
| 5 | 285.2–330.5 | 190.2 | 191.9 | 251.3 | 254.7 | 260.0 | +8.7 | +5.3 |
| 6 | 330.5–378.7 | 199.8 | 205.1 | 261.9 | 242.2 | 258.5 | −3.4 | +16.3 |
| 7 | 378.7–425.0 | 175.2 | 174.1 | 234.3 | 220.8 | 234.6 | +0.3 | +13.8 |
| 8 | 425.0–475.0 | 186.4 | 198.8 | 257.2 | 240.0 | 270.3 | +13.1 | +30.3 |
| 9 | 475.0–525.9 | 232.8 | 236.1 | 275.7 | 260.7 | 292.3 | +16.6 | +31.6 |
| 10 | 525.9–575.9 | 220.4 | 215.8 | 276.7 | 254.7 | 284.2 | +7.5 | +29.5 |
| **Total** | 575.9 h | **2,256.2** | **2,314.2** | **3,045.1** | **2,837.1** | **3,094.7** | **+49.6** | **+257.6** |

* **C3 beats the static policy in 8 of the 10 windows** and loses 2 by at most 3.4 $.
* Its largest gains are in phases 8 and 9 (+13 and +17 $).
* **C3 beats Greedy-Pos in every window.**

## 6. One week against the job-level optimum

The window is the first 7 days of the stream: 58,201 jobs, the same window as the discretization MILP.
* Each run's reward is summed over the jobs that **arrive** in the window. A job's reward counts in full, even if the
  job runs past the window's end, as in the MILP.
* OPT_true is the job-level MILP with the true bids and true durations: best $1,017.34, bound $1,024.42, gap 0.70%
  at its 1 h limit. It is cached in `discretization_milp.pkl` and was not rerun.
* Fluid OPT on the window counts only the volume inside the window (`theory_eval.window_fluid`, `week_fluid.pkl`).
* Every run starts with an empty cluster, as the MILP does.

| Run | Reward over the week $ | % of OPT_true (best) | % of OPT_true (bound) |
|---|---|---|---|
| fluid OPT on the window | 1,118.3 | 109.9 | 109.2 |
| OPT_true (job-level MILP), bound | 1,024.4 | 100.7 | 100.0 |
| OPT_true (job-level MILP), best | 1,017.3 | 100.0 | 99.3 |
| OPT_bin (MILP on bin values), best | 984.1 | 96.7 | 96.1 |
| **C3 = final practical model** | **909.0** | **89.3** | **88.7** |
| static Phase 0 | 904.0 | 88.9 | 88.2 |
| C1 = thinning off | 898.0 | 88.3 | 87.7 |
| C2 = C1 + live oracle | 880.7 | 86.6 | 86.0 |
| Greedy-Pos | 815.7 | 80.2 | 79.6 |
| no-learning | 670.0 | 65.9 | 65.4 |
| stats by end reason (alone) | 665.5 | 65.4 | 65.0 |
| live oracle (alone) | 660.7 | 64.9 | 64.5 |
| DLENT-RT (theoretical) | 629.5 | 61.9 | 61.5 |
| σ cumulative (alone) | 629.5 | 61.9 | 61.5 |
| σ fixed (alone) | 629.5 | 61.9 | 61.4 |

* **The fluid LP is above the job-level optimum** (110%). It is an average over types, not a feasible schedule. So
  OPT_true is the fair yardstick here.
* **C3 reaches 89% of OPT_true** in the first week: the best online policy, just ahead of the static policy (89%).
* **The theoretical model reaches 62%.** The thinning costs about a quarter of the job-level optimum.
* In this first week, C2 is a little below C1 (−17 $). The live oracle helps over the whole stream (+49 $), but not
  in the first week alone.

**Does the final practical model (C3) beat the static Phase 0 policy and Greedy-Pos?** Yes. Over the whole stream, C3
earns 3,094.7 $ (87.8% of fluid OPT). That is 49.6 $ (+1.6%) more than the static Phase 0 policy and 257.6 $ (+9.1%)
more than Greedy-Pos. It is also ahead of both in the first week against the job-level optimum (89.3% vs 88.9% and
80.2% of OPT_true). Its margin over the static policy is small, though.

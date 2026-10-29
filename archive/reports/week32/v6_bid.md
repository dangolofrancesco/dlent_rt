# Week 32: the φ check, a bid with u0 per machine-hour (v6), and the v6 configs

* Raw numbers: `v6_bid_numbers.md`, from `scripts/week32/v6_bid.py`.
* Notebook 01 now runs on `configs/v6_theory.yaml` and `configs/v6_size_bid.yaml`. Its summaries are in
  `phase0_summary_v6_theory.json` and `phase0_summary_v6_size_bid.json`.

**This step changes Phase 0, but only in the new v6 configs.** It goes through the new switch
`bid.mode: size_u0`. `configs/default.yaml` and the carbon map are unchanged; with the default config the numbers are
the same as before, and the snapshots do not change. No online model was run, and the one-week MILP was skipped.

## 1. What changed and why

| what | why |
|---|---|
| New `bid.mode: size_u0` with `bid.u0_per_machine_hour` (default 23.93) | Francesco's decision. The bid pays a utility per machine-hour plus the resource prices, and **priority enters only the customer value** (q × bid). |
| `configs/v6_theory.yaml` = v5_theory + size_u0, u0 = 23.93; `configs/v6_size_bid.yaml` = v5_size_bid + the same | The new bid on the two cluster sizes. |
| **f in v6_theory set to 0.40** (from 0.50) | The fixed rule picks 0.40 on v6 (section 5). |
| φ unchanged | The φ check found no bug (section 2). |
| Tests: `tests/test_phi_and_size_u0.py` | They check E[φ_m] = 0, the cost base on a hand-made job, that the bid does not depend on q, that customer value = q × bid, and that the old modes are unchanged. |

## 2. φ check

**The code.** `dlent_rt/bid.py`:
* `MyersonModel.phi_job` (lines 38–45) computes φ = cost_base × φ_m(bid / cost_base).
* `_phi` (lines 56–64) computes φ_m(m) = m − (1 − F(m)) / f(m), with F the lognormal of the markup (μ = −σ²/2, so the
  mean markup is 1, σ = 0.3).
* Line 63 sets (1 − F) / f to 0 where f(m) ≤ 10⁻¹².

**The theory holds.** E[φ_m] = E[m] − ∫(1 − F) dm = 0. With quadrature, written as 1 − σ∫ m(z)(1 − Φ(z)) dz:

| σ | E[φ_m], exact | mean of 10⁶ draws | E[φ_m; z < −4.81] | probability of z < −4.81 |
|---|---|---|---|---|
| 0.1 | ≈ 0 (2.5 × 10⁻³, quadrature tolerance) | +0.61 | −0.61 | 7 × 10⁻⁷ |
| 0.3 | 1.5 × 10⁻⁸ | +0.22 | −0.23 | 7 × 10⁻⁷ |
| 0.5 | 8 × 10⁻¹⁴ | +0.08 | −0.08 | 7 × 10⁻⁷ |

**Why H gives mean φ / mean bid ≈ +0.26 and not 0.** This is not a bug.
* **The balancing mass is extremely rare.** All of the negative mass that brings E[φ_m] to 0 sits in markups so low
  that fewer than one draw in a million reaches them. There φ_m is huge and negative (φ_m has infinite variance). So
  the mean of any realistic sample is positive: +0.22 over 10⁶ draws at σ = 0.3, and +0.18 (unweighted) over H's
  46,596 jobs, whose lowest z is −4.39.
* **The cost base adds noise.** The ratio weights each job by its cost base, and the cost bases are heavy-tailed: the
  10 largest hold 22% of the total. That moves +0.18 to +0.26.
* **The clip on line 63 is not the cause.** It acts only below z ≈ −7.8 (probability 3.5 × 10⁻¹⁵), which no draw
  reaches. In theory it would shift E[φ_m] by +0.09 at σ = 0.3, but it never changes a computed number.

No fix was needed. The test documents the identity and checks the code against the stable form m (1 − σ·Mills(z)).

## 3. The new bid formula

`dlent_rt/bid.py`, `synthesize_lognormal`, mode "size_u0":

```
cost_base = (u0 × r_j + γ1 × cores_j + γ2 × GB_j) × d_j      r_j = CPU request (machines), u0 in $ per machine-hour
bid ~ lognormal, mean = cost_base, σ = 0.3                    (no priority factor)
customer value = q_j × bid;  profit = φ − electricity;  carbon = s_C × kg;  reward = the sum (all λ = 1)
```

At u0 = 23.93, the bid per CPU machine-hour is $27.3 on average ($17.6 at p10, $38.6 at p90). It is the same for
every priority tier ($27.2–28.2), as intended. The median bid is 853 times the electricity cost.

## 4. The u0 scan

v6, Phase 0 on H. τ and the LP fill are given for f = 0.5 and f = 1.25.

| u0 ($/machine-h) | s_C ($/kg) | positive types / mass | Σ\|r̃\| share: customer value / profit / carbon | bid per machine-h: mean (p10–p90) | bid / electricity (median) | τ at f = 0.5 (CPU, RAM) | LP fill at f = 0.5 | τ at f = 1.25 | LP fill at f = 1.25 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 9.9 | 77.0% / 90.3% | 0.557 / 0.129 / 0.315 | 3.4 (1.8–5.1) | 98 | 0.0009, 0.0007 | 100% / 100% | 0, 0 | 70% / 70% |
| 3.26 | 20.5 | 77.8% / 91.0% | 0.560 / 0.131 / 0.309 | 6.6 (4.0–9.6) | 203 | 0.0026, 0.0005 | 100% / 100% | 0, 0 | 71% / 69% |
| 10 | 42.9 | 77.2% / 90.0% | 0.560 / 0.132 / 0.307 | 13.4 (8.5–19.0) | 416 | 0.0062, 0 | 100% / 98% | 0, 0 | 71% / 68% |
| **23.93** | **88.7** | **76.5% / 89.7%** | 0.561 / 0.133 / 0.306 | 27.3 (17.6–38.6) | 853 | 0.0133, 0 | 100% / 95% | 0, 0 | 71% / 67% |
| 50 | 176.5 | 75.7% / 90.3% | 0.560 / 0.133 / 0.308 | 53.4 (34.6–75.3) | 1,668 | 0.0260, 0 | 100% / 94% | 0, 0 | 71% / 67% |

* **The reward signs barely depend on u0.** Positive types stay at 75.7–77.8% and the positive mass at 90–91%.
  The three terms keep the same shares (56% / 13% / 31%).
* **The small drift comes from the CPU / RAM mix.** u0 adds value per CPU, but not per RAM. So as u0 grows,
  CPU-heavy types gain, RAM-heavy types do not, and RAM stops binding (τ_RAM = 0 from u0 = 10 on).
* **s_C scales with the bid level,** because the quantile map follows the dollar values. τ at f = 0.5 grows with u0
  for the same reason.

## 5. Selected f

The static Phase 0 policy (dual rule, seed 42) on v6, with the same fixed rule: "time ≥ 95% full" closest to
20–30%, preferring 25%.

| f | time ≥ 95% full | mean occupancy / C (CPU) | realised reward $ | fluid OPT $ |
|---|---|---|---|---|
| 0.3 | 24.7% | 78.0% | 28,215 | 56,574 |
| **0.4** | **25.2%** | 80.8% | 38,223 | 68,995 |
| 0.5 | 23.7% | 80.0% | 47,579 | 79,784 |
| 0.6 | 20.5% | 78.4% | 54,045 | 88,078 |
| 0.7 | 15.5% | 75.1% | 61,721 | 93,858 |

**The rule picks f = 0.4**, which differs from 0.5, so **I set f = 0.40 in `configs/v6_theory.yaml`.**
`configs/v6_size_bid.yaml` keeps f = 1.25. Note that f = 0.3, 0.4 and 0.5 are all inside the band and differ by only
about 1 point. The pick among them is sensitive to small changes.

## 6. Phase 0 and the theory summary, v6 vs v5

| quantity | v5_theory (size bid, f = 0.5) | v6 (size_u0) at f = 0.5 | **v6_theory (size_u0, f = 0.4)** |
|---|---|---|---|
| C, CPU / RAM (machines) | 1.737 / 1.156 | 1.737 / 1.156 | 1.390 / 0.925 (89 cores / 237 GB) |
| s_C ($/kg) | 22.5 | 88.7 | 88.7 |
| types with jobs in H | 4,005 | 3,861 | 3,861 |
| positive types / arrival mass | 66.7% / 87.8% | 76.5% / 89.7% | 76.5% / 89.7% |
| τ, Phase 0 LP (CPU, RAM) | 0.00206, 0.00162 | 0.0133, 0 | 0.0164, 0 |
| τ, fluid OPT (CPU, RAM) | 0.00210, 0.00169 | 0.0135, 0 | 0.0162, 0 |
| types fully / partly / rejected (Phase 0 LP) | 1,473 / 2 / 2,530 | – | 1,141 / 1 / 2,719 |
| LP fill CPU / RAM | 100% / 100% | 100% / 95% | 100% / 95% |
| windows with τ > 0, 1 day (CPU, RAM) | 74%, 83% | 87%, 43% | 91%, 52% |
| windows with τ > 0, 6 h (CPU, RAM) | 50%, 50% | 68%, 45% | 81%, 50% |
| static policy: time ≥ 95% full | 24.1% | 23.7% | 25.2% |
| static policy: mean occupancy / C (CPU) | 79.3% | 80.0% | 80.8% |
| static policy: jobs rejected for lack of space | 4.97% | 5.55% | 4.81% |
| static policy: realised reward (customer value / profit / carbon) | $11,017 (18,642 / 1,828 / −9,453) | $47,579 (76,778 / 8,420 / −37,619) | $38,223 (61,658 / 7,036 / −30,470) |
| fluid OPT (customer value / profit / carbon) | $20,346 (27,823 / 4,183 / −11,660) | $79,784 (108,314 / 17,366 / −45,896) | $68,995 (90,441 / 15,506 / −36,953) |
| Phase 0 LP expected | $22,204 | $85,967 | $74,977 |
| Phase 0 LP → stream statistics → fluid OPT | 22,204 → 20,380 → 20,346 | 85,967 → 80,265 → 79,784 | 74,977 → 69,830 → 68,995 |
| static policy / fluid OPT | 54.1% | 59.6% | 55.4% |

On v6_size_bid (f = 1.25): τ = 0, 0; LP fill 71% / 67%; types fully / partly / rejected = 2,955 / 0 / 906.

**What changed from v5 to v6:**
* **The dollar level is about 4 times higher,** because u0 adds 23.93 $/machine-h. s_C follows: 22.5 → 88.7 $/kg.
* **More types have a positive reward:** 66.7% → 76.5%. At the same time two things changed: priority no longer scales
  the bid (so it no longer scales φ or the profit; it only multiplies the customer value), and u0 adds a CPU term. I
  did not separate their effects.
* **CPU is now the only binding resource.**
* **The static policy recovers more of fluid OPT:** 54% → 55–60%.

**Notebook 01 on v6:** all 9 checks PASS on both configs. Across 20 seeds, s_C = 89.8 ± 2.4 $/kg and positive types
= 76.7 ± 1.2%. Doubling the carbon weight cuts positive types to 37.9%.

## 7. Open issues

1. **The f choice is flat.** f = 0.3, 0.4 and 0.5 are all inside the 20–30% band (24.7%, 25.2%, 23.7%). The rule's pick
   of 0.4 depends on differences of about 1 point. Consider several seeds of the thinning before freezing f.
2. **RAM no longer binds (τ_RAM = 0)** with u0 ≥ 10, because u0 pays only for CPU. Decide whether u0 should also pay
   for RAM.
3. **s_C is now 88.7 $/kg,** about 470 times a 0.19 $/kg social cost of carbon. It follows the dollar level, because
   the map is a quantile map (not changed here).
4. **φ's heavy left tail** makes per-type means noisy for types with few jobs. A type with 2 jobs can have a very
   negative φ (see the notebook 01 trace). This is a property of the lognormal, not a bug.
5. **The one-week MILP (discretization cost) was not rerun on v6.**

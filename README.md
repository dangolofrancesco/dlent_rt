# DLENT-RT

Real-time DLENT-Exact simulator for multi-objective cloud resource allocation
with mechanism design and online learning. Companion to the static
`cloud_pricing_sim` (offline Pareto analysis); this branch runs the *online*
DLENT algorithm with nonstationarity tests.

**Status: first slice** — configuration, data ingestion, and Phase 0 bootstrap.
The online loop (theory + practical models), tests, phase transitions, harness,
and dashboards come in subsequent slices.

---

## Install

```bash
pip install -e .            # editable install
# deps: numpy, pandas, scipy, pyyaml, highspy
```

## Quick start

```bash
python -m tests.smoke_phase0 \
    --csv data/batch_may300k.csv \
    --n0 5000 \
    --capacity-mode fraction_of_peak --fraction 0.5 \
    --scalarization linear
```

Or from Python:

```python
from dlent_rt import load_config, load_dataset, run_phase0

cfg = load_config("configs/default.yaml")
ds  = load_dataset(cfg)          # ingests CSV, splits H/stream, sizes capacity
st  = run_phase0(ds, cfg)        # Phase 0 -> PhaseState (grid, priors, tau*, ...)
print(st.tau, st.lam_star)
```

## Architecture

```
dlent_rt/
  config.py       # frozen dataclass tree, strict YAML loader (every axis is a flag)
  data.py         # ingestion, hours->steps, H/stream split, capacity, d_max
  grid.py         # Myerson virtual value + geometric discretization grid
  lp_oracle.py    # scalarization + LP-RS (highspy, dual shadow prices)
  phase0.py       # Algorithm 1: Phase 0 bootstrap -> PhaseState
  models/         # (next slice) base / theory / practical online loops
configs/          # YAML configs; default.yaml is the paper-faithful baseline
tests/            # smoke tests
notebooks/        # (later) Pareto analysis, theory-vs-practical comparison
```

Every experimental axis is a config flag, so an ablation is a config sweep:

| Axis | Config key | Values |
|---|---|---|
| Duration observability | `model.kind` | `theory` (immediate D) / `practical` (delayed D + strategy-D) |
| Scalarization | `scalarization.method` | `chebyshev` / `linear` / `eps_constraint` |
| Oracle | `oracle.kind` | `frozen` / `time_aware` |
| Admission buffer | `admission.a_buffer_fraction` | `0.0` .. `1.0` |
| Revision mode | `revision.mode` | `continuous` / `at_completion` |
| Capacity | `capacity.mode` | `fraction_of_volume` / `fraction_of_peak` / `absolute` |

---

## Two modeling decisions (resolved against the static simulator)

These were verified against `04_pareto_analysis.ipynb` so the online branch
stays consistent with the offline one.

### 1. Capacity sizing

The fluid LP-RS constraint is $\sum_j p_j v_{ij} x_j \le c_i$ with
$v_{ij} = A_{ij}\,\bar d_j$. Capacity must be on the same scale as this expected
consumption, or the LP never binds. Three modes:

- **`fraction_of_volume`** (default, matches the static sim's
  $b_i = \rho \sum_j V_{ij}$): $c_i = \text{fraction} \times \sum_j A_{ij} D_j$.
- **`fraction_of_peak`**: $c_i = \text{fraction} \times$ peak concurrent
  physical occupancy of resource $i$ over the stream. Physically meaningful
  "cluster is X% of peak demand"; makes Test 1 bind realistically.
- **`absolute`**: user-supplied $c_i$ (real deployments / dashboard).

### 2. Objective normalization and the reward sign

Objectives live on very different scales (satisfaction ~$10^6$, profit ~$10^5$,
carbon ~$10^4$). Following the static simulator, each is normalized to $[0,1]$
via its utopian point $z^*$ (a single-objective LP maximum):

$$\tilde V_{sat} = V_{sat}/z^*_{sat},\quad
  \tilde V_{prof} = V_{prof}/z^*_{prof},\quad
  \tilde V_{sus} = 1 - C_{carbon}/z^*_{carb}.$$

The **online oracle/LP reward** is the normalized linear combination
$r = \lambda_1 \tilde V_{sat} + \lambda_2 \tilde V_{prof} - \lambda_3 \tilde V_{sus}$,
which is positively-valued so the knapsack LP admits (rather than degenerating
to "admit nothing"), and `r >= 0` is the admissibility test — exactly as in the
static sim's per-job reward.

**Chebyshev and $\varepsilon$-constraint** are Pareto-front *enumeration* methods
(min-$t$ / threshold LPs), run in the analysis notebooks — not per-type scalar
rewards for the online knapsack. This mirrors how the static simulator uses
them. (A per-type Chebyshev reward is negative-distance-to-ideal, so it cannot
drive a maximizing admission oracle; normalization does not change its sign.)

---

## Time model

One step = one arrival (jobs indexed in arrival order). Durations in hours are
converted to integer arrival-steps via the global arrival rate:
`duration_steps = round(duration_hours * arrivals_per_hour)`, capped at
`d_max_steps` (the configured quantile of step-durations). This keeps the
step↔hour map stationary, which the tests' $\sqrt{t-\tau}$ envelopes assume.

## Data contract

Input CSV columns (configurable in `data.columns`): `collection_id`,
`job_datetime`, `q_j`, `A_cpu`, `A_ram`, `D (hours)`, `v_rate`, `phi_rate`,
`w_j_kw`, `elec_price_per_kWh`, `carbon_intensity_gCO2_per_kWh`, `C_elec`,
`C_carbon`. First `n0` rows → historical batch $H$ (Phase 0, offline); the rest
→ online stream. $H$ is never reused online.

---

## Next slices

1. **This slice** — config, data, Phase 0. ✅
2. LP-RS + oracle + Test 3 (shared core).
3. `TheoryModel` end-to-end + phase-transition unit test.
4. `PracticalModel` + strategy-D revision + four-case gate.
5. Run-harness (both models on one trace) + structured logging.
6. Evaluation notebooks (Pareto, theory-vs-practical).
7. Real-time dashboards.

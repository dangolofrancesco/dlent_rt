# %% [markdown]
# # DLENT-RT — Why is the virtual value φ negative for most jobs? (read-only diagnostic)
#
# Config: `BASE = apply_overrides(default_config(), {"bid.sigma": 0.3, "bid.known_distribution": False})`
# (week 30: pinned to the legacy pooled fit this diagnostic was written for; `MyersonModel.phi` is now
# `phi_job(v)` for a dollar bid / `phi_markup(m)` for a markup, so the file:line references below are historical), normalization
# `quantile_linear`, q = 0.90 (from `configs/default.yaml`). Nothing in `dlent_rt/`, the defaults, the tests or the
# online loop is modified: every φ below is computed with the mechanism's own estimator (`dlent_rt.bid.fit_myerson` +
# `MyersonModel.phi`); only the **conditioning** (which bids are pooled into one F) changes. Helpers:
# `notebooks/legacy/phi_diag.py`.
#
# Sections: 0. Code reading &middot; 1. Heterogeneity &middot; 2. φ under alternative conditioning &middot;
# 3. Bias vs variance &middot; 4. Where the negatives sit &middot; 5. Impact on the reward &middot; 6. Summary tables.
#
# LEGACY (week 30, audit D15): the pooled-fit question is closed (known distribution). Kept for the record;
# not built by `notebooks/src/build.py`. To execute it, copy it to notebooks/src/ and run the build.
# Synthetic fallback: `DLENT_SYNTHETIC=1` (or a missing trace).

# %% [markdown]
# ## Section 0: Code reading (file:line, current working tree)
#
# **0.1 Bid generation — conditional lognormal.** `phase0._synthesize_bids` (`phase0.py:53`) calls
# `bid.synthesize_lognormal` (`bid.py:136-163`) with `jobs.A[:,0], jobs.A[:,1]` (CPU/RAM **snapped to the hw-profile
# centroid**, `data.py:138-139`), `jobs.duration_hours` (floored at 1 min, `data.py:145`) and `jobs.priority`:
#
# * `cpu_cores = cpu_norm * spec_cpu_core`, `ram_gb = ram_norm * spec_ram_gb` (`bid.py:141-142`, 64 cores / 256 GB)
# * `base_utility = u0 * (1 + ln q)` for `mode="log"` (`bid.py:69`, u0 = 23.926 $/h)
# * `cost_base = (base_utility + gamma1*cpu_cores + gamma2*ram_gb) * duration_hours` (`bid.py:146-148`)
# * `mu = ln(cost_base) - sigma^2/2` (`bid.py:150`), `v = rng.lognormal(mean=mu, sigma=sigma)` (`bid.py:152`)
#
# So ln v_j = ln cost_base_j − σ²/2 + σ Z_j, Z_j ~ N(0,1) i.i.d.: the lognormal is **conditional on the job's
# features** (each job has its own μ_j, common σ = 0.3); E[v_j | features] = cost_base_j. Marginally the bids are a
# continuous scale mixture of lognormals, not one lognormal.
#
# **0.2 The three φ's.**
# * `phi_gen` (`bid.py:155-158`): v − (1 − F_j(v))/f_j(v) with F_j = LogNormal(μ_j, σ = 0.3), i.e. the job's **own
#   conditional** distribution. In the stream it is kept as `StreamData.phi_gen` "diagnostic only" (`stream.py:45,52`).
# * `phi_mech` / `phi_ref`: `MyersonModel.phi` (`bid.py:18-25`) with (μ, σ) from `fit_myerson` (`bid.py:42-49`):
#   μ = mean(ln v), σ = sd(ln v, ddof=1) over **all** bids of the window, pooled. Parametric lognormal MLE: **no
#   grid, no empirical density/hazard, no smoothing, no ironing** (the lognormal is regular, φ is increasing).
#   `inv_hazard` is set to 0 where f ≤ 1e-12 (`bid.py:24`). φ_mech uses the phase's F̂ (`reinit.py:76,88`,
#   `theory.py:135`); φ_ref = the same fit on the whole stream.
# * The grid (`grid.space="v"`, geometric) only bins bids into types; it does not enter φ(v). The bin floor L_k
#   enters only the payment.
#
# **0.3 Where φ enters.**
# * Π = φ − C_elec. Type level: `raw_prof = phi_bar - C_elec_bar` (`phase0.py:189`) → r̃ = λ1·q·v̄ + λ2·Π̄ − λ3·s_C·K̄.
# * **Phase 0** sets `phi_bar = _mean(bids.phi_rate)` (`phase0.py:317`), and `bids.phi_rate` is the **generator** φ
#   (`phi_gen`). **ReInit** sets `phi_bar` from the pooled fit `myerson.phi(v_arr)` (`reinit.py:88-90`). So the phase-1
#   type reward uses φ_gen, every later phase the pooled F̂ (see Anomalies).
# * Realised job reward online: `phi = st.myerson.phi(v)`, `prof = phi - C_elec` (`theory.py:135,141`) → pooled F̂ of
#   the current phase in every phase, including phase 1.
# * Logged payment: `pay = phi_tilde[j]` (`theory.py:144`), `phi_tilde = myerson.phi(L)` at the type's bid-bin floor
#   for lognormal (`bid.py:168-178`; `phase0.py:510`, `reinit.py:127`).
#
# **0.4 Analytic fact used below.** For a lognormal F with parameter σ, φ(v) = v·[1 − σ·M(z)], z = (ln v − μ)/σ,
# M = (1 − Φ)/φ_N the Mills ratio. Hence φ(v) < 0 ⇔ σ·M(z) > 1 ⇔ z < z*(σ), independent of μ, and the share of
# bids with φ < 0 *under the fitted F itself* is Φ(z*(σ)) — a function of σ only.

# %% [markdown]
# ## Setup

# %%
import os, sys, time, warnings, dataclasses, tempfile
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
from IPython.display import display
warnings.simplefilter("ignore")

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "dlent_rt").is_dir())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks" / "legacy"))

from dlent_rt.config import default_config
from dlent_rt.data import load_dataset
from dlent_rt.phase0 import (run_phase0, assign_type_ids, type_arrays, scalarized_reward, _weighted_quantile,
                             _compute_job_energy, _compute_job_power, _lookup_grid)
from dlent_rt.bid import MyersonModel, compute_base_utility
from dlent_rt.models.theory import TheoryModel
from dlent_rt.models.core.stream import build_stream
import phi_diag as pdg

plt.rcParams["figure.figsize"] = (13, 3.6)
plt.rcParams["figure.dpi"] = 110
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

# palette (notebook 01/03 conventions)
COL = {"a pooled": "darkorange", "b bucket": "seagreen", "c1 per core-hour": "teal", "c2 per hour": "slateblue",
       "d residual": "purple", "e phi_gen": "gray"}

# %%
def _replace_path(obj, path, value):
    if len(path) == 1:
        return dataclasses.replace(obj, **{path[0]: value})
    head, rest = path[0], path[1:]
    return dataclasses.replace(obj, **{head: _replace_path(getattr(obj, head), rest, value)})

def apply_overrides(cfg, overrides: dict):
    for dotted_key, value in overrides.items():
        cfg = _replace_path(cfg, dotted_key.split("."), value)
    return cfg

cfg = default_config()
cfg = apply_overrides(cfg, {"data.batch_csv": str(ROOT / cfg.data.batch_csv),     # path plumbing only
                            "data.grid_csv": str(ROOT / cfg.data.grid_csv)})
# pooled-fit era diagnostic (week 28): runs the legacy pooled MLE, bid.known_distribution = false
BASE = apply_overrides(cfg, {"bid.sigma": 0.3, "bid.known_distribution": False})

USE_SYNTHETIC = (os.environ.get("DLENT_SYNTHETIC") == "1" or not Path(BASE.data.batch_csv).is_file())
if USE_SYNTHETIC:
    from tests.synthetic import write_synthetic
    _b, _g = write_synthetic(Path(tempfile.mkdtemp()), n_jobs=3000, drift=True)
    BASE = apply_overrides(BASE, {
        "data.batch_csv": str(_b), "data.grid_csv": str(_g), "data.n0": 1000,
        "grid.k_star": 16, "hardware.n_profiles": 6,
        "time.d_max_horizon_fraction": 0.10, "time.d_max_ratio_abort": 0.2,
    })
    print("USING SYNTHETIC FALLBACK DATA (tests/synthetic.py, drift=True): numbers are NOT meaningful")
BASE = BASE.validate()
SEED = BASE.run.seed
SIGMA = BASE.bid.sigma
L1, L2, L3 = BASE.objective.lambda1, BASE.objective.lambda2, BASE.objective.lambda3
print(f"seeds: run.seed={SEED} (H bids)  stream bids={SEED + BASE.online.bid_seed_offset}  "
      f"thinning={SEED + BASE.online.thinning_seed_offset}  subsampling/bootstrap rng={SEED}")
print(f"bid: lognormal sigma={SIGMA} u0={BASE.bid.base_utility_u0} mode={BASE.bid.base_utility_mode} "
      f"gamma1={BASE.bid.gamma1} gamma2={BASE.bid.gamma2}")
print(f"normalization: {BASE.normalization.strategy} q={BASE.normalization.quantile}   lambda=({L1}, {L2}, {L3})")

# %%
ds = load_dataset(BASE)
st0 = run_phase0(ds, BASE)
S = build_stream(ds, BASE)
H = ds.H
s_C = st0.carbon_price
print(f"data.n0 = {BASE.data.n0:,} -> |H| = {H.n:,}, |stream| = {S.n:,}, T = {ds.T:,}; s_C (Phase 0, frozen) = {s_C:.4f} $/kgCO2")

# analysis population = H U stream (every job of the filtered trace, with the bids the pipeline generates for it)
pH, ciH = _lookup_grid(H, ds)
E_H = _compute_job_energy(H, BASE)
dt = np.concatenate([H.datetime, ds.stream.datetime])
P = pd.DataFrame({
    "src": np.r_[np.repeat("H", H.n), np.repeat("stream", S.n)],
    "v": np.r_[st0.bids.v_rate, S.v], "phi_gen": np.r_[st0.bids.phi_rate, S.phi_gen],
    "q": np.r_[H.priority, S.priority], "hw": np.r_[H.hw_idx, S.hw_idx],
    "A_cpu": np.r_[H.A[:, 0], S.A[:, 0]], "A_ram": np.r_[H.A[:, 1], S.A[:, 1]],
    "d_h": np.r_[H.duration_hours, S.duration_hours], "d_steps": np.r_[H.duration_steps, S.D],
    "sc": np.r_[H.scheduling_class, ds.stream.scheduling_class],
    "power": np.r_[_compute_job_power(H, BASE), S.power_kw], "energy": np.r_[E_H, S.energy_kwh],
    "C_elec": np.r_[E_H * pH, S.C_elec], "K": np.r_[E_H * ciH, S.C_carbon],
    "hour": pd.DatetimeIndex(dt).hour,
})
N, N_H = len(P), H.n
v, phi_gen = P["v"].to_numpy(), P["phi_gen"].to_numpy()
print(f"population N = {N:,} jobs (H {N_H:,} + stream {S.n:,})")
print(f"log10(v): p1={np.log10(np.quantile(v, .01)):.2f}  p99={np.log10(np.quantile(v, .99)):.2f}  "
      f"-> {np.log10(np.quantile(v, .99) / np.quantile(v, .01)):.2f} decades;  sigma={SIGMA} (ln) = "
      f"{SIGMA / np.log(10):.3f} dex, p1-p99 of exp(sigma Z) spans {2 * 2.326 * SIGMA / np.log(10):.2f} decades")

# %% [markdown]
# ### 0.5 The observation, reproduced with the current default (`data.n0` from `configs/default.yaml`)
# `phi_mech` comes from one online run of the theory model (default capacity, scale 1.0); φ_ref = pooled fit on the
# stream; φ_P0 = Phase 0's pooled fit (on H) applied to the same bids.

# %%
class RecordingModel(TheoryModel):
    def run(self):
        self.states = []
        return super().run()
    def _new_phase(self, state, tau_prev, W_prev):
        self.states.append(state)
        super()._new_phase(state, tau_prev, W_prev)

t0 = time.time()
model = RecordingModel(ds, st0, BASE)
res = model.run()
ARR = pd.DataFrame(res.arrivals)
ARR["pop"] = N_H + ARR["i"].to_numpy()
ARR["R_job"] = (L1 * ARR["q"] * ARR["v"] + L2 * (ARR["phi_mech"] - S.C_elec[ARR["i"]])
                - L3 * s_C * S.C_carbon[ARR["i"]])
print(f"online run: {len(ARR):,} arrivals, {res.summary['n_phases']} phases, {int(ARR['admit'].sum()):,} admitted, "
      f"R_total = {res.summary['R_total']:,.2f} $ ({time.time() - t0:.0f} s)")
phi_ref_s, m_ref = pdg.phi_pooled(S.v)
obs = pd.DataFrame([
    {"phi": "phi_mech (phase F_hat, online log)", "n": len(ARR), "share<0 %": 100 * (ARR["phi_mech"] < 0).mean()},
    {"phi": "phi_ref (pooled fit on stream)", "n": S.n, "share<0 %": 100 * (phi_ref_s < 0).mean()},
    {"phi": "phi_P0 (Phase-0 pooled fit, on stream bids)", "n": S.n, "share<0 %": 100 * (st0.myerson.phi_job(S.v) < 0).mean()},
    {"phi": "phi_gen (stream)", "n": S.n, "share<0 %": 100 * (S.phi_gen < 0).mean()},
    {"phi": "phi_P0 on H bids", "n": N_H, "share<0 %": 100 * (st0.myerson.phi_job(st0.bids.v_rate) < 0).mean()},
    {"phi": "phi_gen (H)", "n": N_H, "share<0 %": 100 * (st0.bids.phi_rate < 0).mean()},
]).set_index("phi")
display(obs.style.format(precision=2))
print(f"fitted sigma: Phase 0 (H) = {st0.myerson.sigma:.4f}, stream = {m_ref.sigma:.4f}; "
      + ", ".join(f"phase {k} = {s.myerson.sigma:.4f}" for k, s in enumerate(model.states, 1)))

# %%
def z_star(sig):
    return brentq(lambda z: sig * norm.sf(z) / norm.pdf(z) - 1.0, -20, 20)

sig_rows = [{"sigma": s, "z*": z_star(s), "share phi<0 under F itself = Phi(z*) %": 100 * norm.cdf(z_star(s))}
            for s in sorted({0.1, SIGMA, 0.5, 0.8, 1.0, round(st0.myerson.sigma, 4), 1.5, 2.0})]
display(pd.DataFrame(sig_rows).set_index("sigma").style.format(precision=4))

# %% [markdown]
# ## Section 1: Is the bid explained by job features? (heterogeneity test)
#
# OLS of ln v on the population. M1: ln d, ln A_cpu, ln A_ram, ln power, priority (numeric). M2: ln d, ln power,
# priority / hw-profile / hour-of-day dummies (ln A is collinear with the hw dummies because A is snapped). M0:
# the generator's own predictor ln cost_base (recomputed from the config with `dlent_rt.bid.compute_base_utility`),
# shown as the ground-truth reference. Residual sd to compare with σ = 0.3.

# %%
lnv = np.log(v)
X1 = np.column_stack([np.ones(N), np.log(P["d_h"]), np.log(P["A_cpu"]), np.log(P["A_ram"]),
                      np.log(P["power"]), P["q"]])
X2 = np.column_stack([np.ones(N), np.log(P["d_h"]), np.log(P["power"]), pdg.dummies(P["q"].to_numpy()),
                      pdg.dummies(P["hw"].to_numpy()), pdg.dummies(P["hour"].to_numpy())])
bb = BASE.bid
cost_base = ((compute_base_utility(P["q"].to_numpy(), bb.base_utility_u0, bb.base_utility_mode)
              + bb.gamma1 * P["A_cpu"] * BASE.energy_model.spec_cpu_core
              + bb.gamma2 * P["A_ram"] * BASE.energy_model.spec_ram_gb) * P["d_h"]).to_numpy()
X0 = np.column_stack([np.ones(N), np.log(cost_base)])
M0, M1, M2 = pdg.ols(X0, lnv), pdg.ols(X1, lnv), pdg.ols(X2, lnv)

reg = pd.DataFrame([
    {"model": "M0 ln cost_base (generator)", "R2": M0["R2"], "resid sd": M0["resid_sd"], "k": X0.shape[1]},
    {"model": "M1 continuous", "R2": M1["R2"], "resid sd": M1["resid_sd"], "k": X1.shape[1]},
    {"model": "M2 dummies", "R2": M2["R2"], "resid sd": M2["resid_sd"], "k": X2.shape[1]},
]).set_index("model")
display(reg.style.format(precision=4))
print(f"pooled sd(ln v) = {lnv.std():.4f}   sigma = {SIGMA}   M0 coefficients: intercept {M0['coef'][0]:.4f} "
      f"(generator: -sigma^2/2 = {-SIGMA ** 2 / 2:.4f}), slope {M0['coef'][1]:.4f} (generator: 1)")
coef1 = pd.Series(M1["coef"], index=["const", "ln d_h", "ln A_cpu", "ln A_ram", "ln power", "priority"])
display(coef1.to_frame("M1 coefficient").T.style.format(precision=4))

# sequential variance decomposition (share of var(ln v) explained, adding blocks in order)
blocks = [("ln d_h", np.log(P["d_h"].to_numpy())[:, None]), ("priority", pdg.dummies(P["q"].to_numpy())),
          ("hw profile", pdg.dummies(P["hw"].to_numpy())), ("ln power", np.log(P["power"].to_numpy())[:, None]),
          ("hour", pdg.dummies(P["hour"].to_numpy()))]
Xs, prev, dec = np.ones((N, 1)), 0.0, []
for name, B in blocks:
    Xs = np.column_stack([Xs, B])
    r2 = pdg.ols(Xs, lnv)["R2"]
    dec.append({"block added": name, "cumulative R2": r2, "increment": r2 - prev})
    prev = r2
dec.append({"block added": "residual (unexplained)", "cumulative R2": 1.0, "increment": 1 - prev})
display(pd.DataFrame(dec).set_index("block added").style.format(precision=4))

# %%
fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
ax[0].hist(lnv - lnv.mean(), bins=80, density=True, color="steelblue", alpha=0.7, label=f"ln v pooled (sd {lnv.std():.3f})")
ax[0].hist(M2["resid"], bins=80, density=True, color="purple", alpha=0.6, label=f"ln v residual M2 (sd {M2['resid'].std():.3f})")
xx = np.linspace(-4, 4, 400)
ax[0].plot(xx, norm.pdf(xx, 0, SIGMA), "k--", lw=1, label=f"N(0, {SIGMA})")
ax[0].set_xlim(-4, 4); ax[0].legend(fontsize=8); ax[0].set_title("centred ln v: pooled vs residual", fontsize=10)
ax[1].scatter(np.log10(P["d_h"]), np.log10(v), s=2, alpha=0.25, c=P["q"], cmap="viridis")
ax[1].set_xlabel("log10 duration [h]"); ax[1].set_ylabel("log10 v [$]")
ax[1].set_title("bid vs duration (colour = priority)", fontsize=10)
plt.tight_layout(); plt.show()

# effective number of components: perplexity of the fitted log-locations at resolution sigma_resid
w_bin = M2["resid"].std()
cnt, _ = np.histogram(M2["fitted"], bins=np.arange(M2["fitted"].min(), M2["fitted"].max() + w_bin, w_bin))
pr = cnt[cnt > 0] / cnt.sum()
print(f"Conclusion (numbers): features explain R2 = {M2['R2']:.3f} of var(ln v) (M0: {M0['R2']:.3f}); residual sd = "
      f"{M2['resid'].std():.3f} vs sigma = {SIGMA}; pooled sd = {lnv.std():.3f} = {lnv.std() / SIGMA:.1f} x sigma.")
print(f"pooled F is a scale mixture of lognormals with per-job location; effective components "
      f"(exp entropy of fitted ln-locations binned at width {w_bin:.3f}) = {np.exp(-(pr * np.log(pr)).sum()):.1f} "
      f"over {len(cnt)} bins spanning {M2['fitted'].max() - M2['fitted'].min():.2f} in ln v.")

# %% [markdown]
# ## Section 2: φ under alternative conditioning
#
# Same estimator (`fit_myerson` + `MyersonModel.phi`), same jobs:
# * **(a) pooled**: one F for all N bids.
# * **(b) bucket**: one F per bucket = priority × hw profile × quantile bin of the volume proxy V = CPU core-hours
#   (A_cpu·64·d_h); every bucket ≥ 200 jobs (merging rules in `phi_diag.size_buckets`).
# * **(c) per-unit bid**: u = v / s pooled, mapped back with φ_v(v) = s · φ_u(v/s) (because F_v|s(x) = F_u(x/s));
#   (c1) s = core-hours, (c2) s = duration hours.
# * **(d) residual bid**: w = v / exp(ŷ_M2) pooled, mapped back with φ_v = exp(ŷ) · φ_w(w).
# * **(e) φ_gen**: the generator's true conditional φ (ground truth).
#
# Columns: F̂(r*) = model quantile of the reserve price r* = φ⁻¹(0) (for (b) bucket-size-weighted mean; for (e) the
# generator's Φ(z*(0.3))); non-monotonicity within the conditioning cell (the cell's φ sorted by the pooled variable:
# v for (a)/(b), φ_u vs u for (c)/(d), φ_gen vs v inside the (b) buckets for (e)) and across
# the Phase-0 geometric bid bins (mean φ per bin); E[φ⁺]; confusion of sign(φ) vs sign(φ_gen) (first sign = row).

# %%
MIN_BUCKET = 200 if N >= 4000 else 100
V_core = (P["A_cpu"] * BASE.energy_model.spec_cpu_core * P["d_h"]).to_numpy()
lab, btab = pdg.size_buckets(P["q"].to_numpy(), P["hw"].to_numpy(), V_core, min_size=MIN_BUCKET)
print(f"buckets: {len(btab)} (min size {MIN_BUCKET}): size min={btab['n'].min()}, median={int(btab['n'].median())}, "
      f"max={btab['n'].max()}; cells (priority x hw) = {btab['cell'].nunique()}")
with pd.option_context("display.max_rows", 200):
    display(btab.groupby("cell").agg(buckets=("n", "size"), jobs=("n", "sum"), min_n=("n", "min"))
            .sort_values("jobs", ascending=False).T)

k_bin = st0.grid.assign(v, st0.myerson)                  # Phase-0 geometric bid bins
phi_a, m_a = pdg.phi_pooled(v)
phi_b, m_b, nfb = pdg.phi_by_group(v, lab, fallback=phi_a)
phi_c1, m_c1 = pdg.phi_scaled(v, V_core)
phi_c2, m_c2 = pdg.phi_scaled(v, P["d_h"].to_numpy())
s_d = np.exp(M2["fitted"])
phi_d, m_d = pdg.phi_scaled(v, s_d)
PHI = {"a pooled": phi_a, "b bucket": phi_b, "c1 per core-hour": phi_c1, "c2 per hour": phi_c2,
       "d residual": phi_d, "e phi_gen": phi_gen}
rq_b = np.average([pdg.reserve_quantile(m) for m in m_b.values()],
                  weights=[int((lab == g).sum()) for g in m_b.keys()])
RQ = {"a pooled": pdg.reserve_quantile(m_a), "b bucket": rq_b, "c1 per core-hour": pdg.reserve_quantile(m_c1),
      "c2 per hour": pdg.reserve_quantile(m_c2), "d residual": pdg.reserve_quantile(m_d),
      "e phi_gen": pdg.reserve_quantile(MyersonModel(mu=0.0, sigma=SIGMA))}
KEY = {"a pooled": (v, np.zeros(N)), "b bucket": (v, lab), "c1 per core-hour": (v / V_core, np.zeros(N)),
       "c2 per hour": (v / P["d_h"].to_numpy(), np.zeros(N)), "d residual": (v / s_d, np.zeros(N)),
       "e phi_gen": (v, lab)}
SIG = {"a pooled": m_a.sigma, "b bucket": np.median([m.sigma for m in m_b.values()]), "c1 per core-hour": m_c1.sigma,
       "c2 per hour": m_c2.sigma, "d residual": m_d.sigma, "e phi_gen": SIGMA}
WITHIN = {"c1 per core-hour": phi_c1 / V_core, "c2 per hour": phi_c2 / P["d_h"].to_numpy(),
          "d residual": phi_d / s_d}                    # phi of the pooled variable u (scaled variants)
T2 = pd.DataFrame([{**pdg.summary_row(n, PHI[n], phi_gen, k_bin, *KEY[n], RQ[n], phi_within=WITHIN.get(n)),
                    "fitted sigma (b: median)": SIG[n]} for n in PHI]).set_index("conditioning")
display(T2.style.format(precision=3))
print(f"(b): {nfb} jobs in buckets with < 2 bids kept the pooled phi (fit_myerson needs >= 2).")

# %%
med_v = pd.Series(v).groupby(lab).median().sort_values()
pick = med_v.index[np.unique(np.linspace(0, len(med_v) - 1, 6).round().astype(int))]
fig, axes = plt.subplots(2, 3, figsize=(15, 7), squeeze=False)
for ax, g in zip(axes.ravel(), pick):
    ix = lab == g
    vv = np.geomspace(v[ix].min(), v[ix].max(), 200)
    ax.scatter(v[ix], phi_gen[ix], s=4, color=COL["e phi_gen"], alpha=0.5, label="e phi_gen")
    ax.scatter(v[ix], phi_b[ix], s=4, color=COL["b bucket"], alpha=0.6, label="b bucket")
    ax.scatter(v[ix], phi_d[ix], s=3, color=COL["d residual"], alpha=0.4, label="d residual")
    ax.plot(vv, m_a.phi_job(vv), color=COL["a pooled"], lw=2, label="a pooled curve")
    ax.axhline(0, color="k", lw=0.7); ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=1.0)
    ax.set_title(f"{g} (n={ix.sum()})\nphi<0: a {100 * (phi_a[ix] < 0).mean():.0f}%  b {100 * (phi_b[ix] < 0).mean():.0f}%  "
                 f"d {100 * (phi_d[ix] < 0).mean():.0f}%  gen {100 * (phi_gen[ix] < 0).mean():.0f}%", fontsize=8)
    ax.set_xlabel("v [$]"); ax.set_ylabel("phi [$]")
axes[0, 0].legend(fontsize=7)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Section 3: Bias vs variance
#
# Subsamples of size n (without replacement, rng seed = run.seed), 20 repeats each. (a) pooled fit on the subsample;
# (b) one fit per bucket present in the subsample (buckets with < 2 bids keep the subsample's pooled φ);
# (d) OLS M2 refit on the subsample, then the pooled fit on the residual bids. Band = mean ± 1 sd over repeats;
# dashed line = φ_gen level on the whole population.

# %%
REPS = 20
NS = [n for n in (500, 1_000, 2_000, 5_000, 10_000, 20_000) if n <= 0.9 * N] or [N // 2]
rng = np.random.default_rng(SEED)
rows = []
for n in NS:
    for r in range(REPS):
        ix = rng.choice(N, n, replace=False)
        vs = v[ix]
        pa, _ = pdg.phi_pooled(vs)
        pb, _, _ = pdg.phi_by_group(vs, lab[ix], fallback=pa)
        fit = pdg.ols(X2[ix], np.log(vs))["fitted"]
        pd_, _ = pdg.phi_scaled(vs, np.exp(fit))
        for name, ph in (("a pooled", pa), ("b bucket", pb), ("d residual", pd_), ("e phi_gen", phi_gen[ix])):
            rows.append({"n": n, "rep": r, "phi": name, "share<0": (ph < 0).mean(), "E[phi+]": np.maximum(ph, 0).mean()})
SUB = pd.DataFrame(rows)
agg = SUB.groupby(["phi", "n"]).agg(share_mean=("share<0", "mean"), share_sd=("share<0", "std"),
                                     Ep_mean=("E[phi+]", "mean"), Ep_sd=("E[phi+]", "std"))
display((agg * pd.Series({"share_mean": 100, "share_sd": 100, "Ep_mean": 1, "Ep_sd": 1})).unstack("phi")
        .style.format(precision=3))

fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
for name in ("a pooled", "b bucket", "d residual"):
    g = agg.loc[name]
    for a_, m_, s_, sc in ((ax[0], "share_mean", "share_sd", 100), (ax[1], "Ep_mean", "Ep_sd", 1)):
        a_.plot(g.index, sc * g[m_], "o-", color=COL[name], label=name)
        a_.fill_between(g.index, sc * (g[m_] - g[s_]), sc * (g[m_] + g[s_]), color=COL[name], alpha=0.2)
ax[0].axhline(100 * (phi_gen < 0).mean(), color="k", ls="--", lw=1, label="phi_gen (population)")
ax[1].axhline(np.maximum(phi_gen, 0).mean(), color="k", ls="--", lw=1, label="phi_gen (population)")
for a_, t in zip(ax, ("share phi<0 [%]", "E[phi+] [$]")):
    a_.set_xscale("log"); a_.set_xlabel("subsample size n"); a_.set_title(t, fontsize=10); a_.legend(fontsize=8)
plt.tight_layout(); plt.show()

n_bs = min(20_000, N)
bs = np.array([(pdg.phi_pooled(v[rng.integers(0, N, n_bs)])[0] < 0).mean() for _ in range(300)])
print(f"bootstrap (300 resamples, n={n_bs:,}) of the pooled share phi<0: mean {100 * bs.mean():.2f}%, "
      f"95% CI [{100 * np.quantile(bs, .025):.2f}%, {100 * np.quantile(bs, .975):.2f}%]; "
      f"phi_gen level {100 * (phi_gen < 0).mean():.2f}%; analytic Phi(z*(sigma_hat={m_a.sigma:.3f})) = "
      f"{100 * norm.cdf(z_star(m_a.sigma)):.2f}%")

# %% [markdown]
# ## Section 4: Where do the negatives sit?

# %%
dec_V = pd.qcut(V_core, 10, labels=False, duplicates="drop") + 1
by = {"priority": P["q"].astype(int).to_numpy(), "size decile (core-hours)": dec_V}
for name, key in by.items():
    t = pd.DataFrame({k: pd.Series(PHI[k] < 0).groupby(key).mean() * 100 for k in ("a pooled", "b bucket", "d residual", "e phi_gen")})
    t.insert(0, "n", pd.Series(key).value_counts().sort_index())
    t.insert(1, "median v $", pd.Series(v).groupby(key).median())
    print(f"share phi<0 [%] by {name}")
    display(t.style.format(precision=2))
thw = pd.DataFrame({k: pd.Series(PHI[k] < 0).groupby(P["hw"].to_numpy()).mean() * 100
                    for k in ("a pooled", "b bucket", "e phi_gen")})
thw.insert(0, "n", P["hw"].value_counts().sort_index())
print("share phi<0 [%] by hw profile (sorted by n)")
display(thw.sort_values("n", ascending=False).T.style.format(precision=1))

kk = pd.Series(k_bin)
fig, ax = plt.subplots(figsize=(13, 3.6))
for name in ("a pooled", "b bucket", "d residual", "e phi_gen"):
    sh = pd.Series(PHI[name] < 0).groupby(kk).mean() * 100
    ax.plot(sh.index, sh.values, "o-", ms=3, color=COL[name], label=name)
tw = ax.twinx(); cnt_k = kk.value_counts().sort_index()
tw.bar(cnt_k.index, cnt_k.values, color="gray", alpha=0.2); tw.set_ylabel("jobs per bin", color="gray")
ax.set_zorder(tw.get_zorder() + 1); ax.patch.set_visible(False)
ax.set_xlabel("Phase-0 geometric bid bin k (low v -> high v)"); ax.set_ylabel("share phi<0 [%]")
ax.legend(fontsize=8); ax.set_title("share of phi<0 by bid bin", fontsize=10)
plt.tight_layout(); plt.show()

low = v <= np.quantile(v, 0.2)
print("concentration in small jobs: share of the phi<0 jobs that are in the bottom quintile of v / "
      "median v of phi<0 vs phi>=0 jobs")
for name in ("a pooled", "b bucket", "d residual", "e phi_gen"):
    ng = PHI[name] < 0
    print(f"  {name:<18} {100 * (low & ng).sum() / max(ng.sum(), 1):6.1f}%   median v: {np.median(v[ng]):.3f} vs "
          f"{np.median(v[~ng]):.3f} $")

# %% [markdown]
# ### 4.b Π < 0: φ-driven vs electricity-driven
# Π = φ − C_elec. Among Π < 0 jobs: "φ-driven" = φ < 0; "electricity-driven" = φ ≥ 0 but φ < C_elec. Negative-Π
# mass Σ|Π| over Π < 0 split as Σ max(−φ, 0) (φ part) + Σ C_elec (electricity part) over the same jobs.

# %%
Ce = P["C_elec"].to_numpy()
print(f"C_elec per job: median {np.median(Ce):.3g} $, p99 {np.quantile(Ce, .99):.3g} $, max {Ce.max():.3g} $;  "
      f"median |phi_a| {np.median(np.abs(phi_a)):.3g} $")
rows = []
for name in ("a pooled", "b bucket", "d residual", "e phi_gen"):
    ph = PHI[name]; Pi = ph - Ce; ng = Pi < 0
    m_phi, m_el = np.maximum(-ph[ng], 0).sum(), Ce[ng].sum()
    rows.append({"phi": name, "share Pi<0 %": 100 * ng.mean(), "phi-driven (phi<0) %": 100 * (ng & (ph < 0)).sum() / max(ng.sum(), 1),
                 "electricity-driven %": 100 * (ng & (ph >= 0)).sum() / max(ng.sum(), 1),
                 "neg mass sum|Pi| $": np.abs(Pi[ng]).sum(), "phi part %": 100 * m_phi / (m_phi + m_el),
                 "C_elec part %": 100 * m_el / (m_phi + m_el)})
display(pd.DataFrame(rows).set_index("phi").style.format(precision=3))

# %% [markdown]
# ### 4.c Admitted jobs with R_job < 0 in the online run (theory model, capacity scale 1.0)
# R_job uses φ_mech (as logged). For those with φ_mech < 0: how many would have φ ≥ 0 under (b), (d) or φ_gen.

# %%
an = ARR[ARR["admit"] & (ARR["R_job"] < 0)]
ip = an["pop"].to_numpy()
neg_m = an["phi_mech"].to_numpy() < 0
print(f"admitted jobs: {int(ARR['admit'].sum())}; with R_job < 0: {len(an)}; of these phi_mech < 0: {int(neg_m.sum())}")
br = {"phi_mech<0 & phi_b>=0": int((neg_m & (phi_b[ip] >= 0)).sum()),
      "phi_mech<0 & phi_d>=0": int((neg_m & (phi_d[ip] >= 0)).sum()),
      "phi_mech<0 & phi_gen>=0": int((neg_m & (phi_gen[ip] >= 0)).sum()),
      "phi_mech<0 & (b or d or gen) >= 0": int((neg_m & ((phi_b[ip] >= 0) | (phi_d[ip] >= 0) | (phi_gen[ip] >= 0))).sum()),
      "phi_mech<0 & all of b, d, gen < 0": int((neg_m & (phi_b[ip] < 0) & (phi_d[ip] < 0) & (phi_gen[ip] < 0)).sum()),
      "phi_mech>=0 (negative from carbon/electricity)": int((~neg_m).sum())}
display(pd.Series(br, name="jobs").to_frame().T)
if len(an):
    print("same breakdown per phase:")
    display(an.assign(phi_mech_neg=neg_m, gen_pos=phi_gen[ip] >= 0, b_pos=phi_b[ip] >= 0, d_pos=phi_d[ip] >= 0)
            .groupby("m")[["phi_mech_neg", "b_pos", "d_pos", "gen_pos"]].sum().assign(n=an.groupby("m").size()))

# %% [markdown]
# ## Section 5: Impact on the reward (counterfactual, no change to the mechanism)
#
# Job level on the whole population: R = λ1·q·v + λ2·(φ − C_elec) − λ3·s_C·K with s_C = Phase 0's value (frozen).
# Type level: Phase-0 type space (grid + hw profiles), p_j = population counts, r̃ from `phase0.scalarized_reward`
# with `carbon_price = s_C`, φ̄_j = mean of the chosen φ over the type's jobs. Sign mismatch = share of jobs whose
# sign(R_job) differs from sign(r̃) of their type. The last row is Phase 0 as implemented (H only, φ̄ = mean φ_gen).

# %%
SW, K = (P["q"] * P["v"]).to_numpy(), P["K"].to_numpy()
tid = assign_type_ids(st0.type_space, k_bin, P["q"].to_numpy(), P["hw"].to_numpy())
nT = st0.n_types
Nt = np.bincount(tid, minlength=nT).astype(float)
mean_t = lambda x: np.divide(np.bincount(tid, weights=np.asarray(x, float), minlength=nT), Nt,
                             out=np.zeros(nT), where=Nt > 0)
_, prio_t, A_t = type_arrays(st0.type_space, st0.grid)
p_t = Nt / N
pop_t = p_t > 0
rows = []
for name in ("a pooled", "b bucket", "d residual", "e phi_gen"):
    ph = PHI[name]
    R = L1 * SW + L2 * (ph - Ce) - L3 * s_C * K
    Rnc = L1 * SW + L2 * (ph - Ce)
    r_t, *_ = scalarized_reward(prio_t, mean_t(v), mean_t(ph), mean_t(Ce), mean_t(K), p_t,
                                A_t * mean_t(P["d_steps"])[:, None], ds, BASE, common_scale=1.0, carbon_price=s_C)
    rows.append({"phi": name, "R<0 %": 100 * (R < 0).mean(), "R<0 no carbon %": 100 * (Rnc < 0).mean(),
                 "mean R $": R.mean(), "type r~<0 % (p-weighted)": 100 * np.average(r_t[pop_t] < 0, weights=p_t[pop_t]),
                 "sign mismatch job vs type %": 100 * (np.sign(R) != np.sign(r_t[tid])).mean()})
p0m = st0.p > 0
rows.append({"phi": "Phase 0 as implemented (H, phi_bar = mean phi_gen)", "R<0 %": np.nan, "R<0 no carbon %": np.nan,
             "mean R $": np.nan, "type r~<0 % (p-weighted)": 100 * np.average(st0.r_tilde[p0m] < 0, weights=st0.p[p0m]),
             "sign mismatch job vs type %": np.nan})
display(pd.DataFrame(rows).set_index("phi").style.format(precision=3))

# %% [markdown]
# ### 5.b Admissions of the online run that would flip
# For each phase state, φ̄_j is recomputed from the SAME jobs that built it (phase 1: H; phase m: arrivals of phase
# m−1) with the alternative φ; types whose stats were carried forward (N_j < n_type_min, m ≥ 2) keep the original
# value (counted). r̃′ = r̃ + λ2(φ̄′ − φ̄); oracle′ = 1[r̃′ − τᵀv > 0]. "would reject" = admitted jobs with oracle′ = 0;
# "would pass oracle" = oracle rejects with oracle′ = 1 (their admission would still need U(t) = 1 and the gate).

# %%
nmin = BASE.phase_transition.n_type_min
rows = []
for m, st in enumerate(model.states, start=1):
    g = ARR[ARR["m"] == m]
    if m == 1:
        b_rows = np.arange(N_H); thr = 1
    else:
        b_rows = ARR.loc[ARR["m"] == m - 1, "pop"].to_numpy(); thr = nmin
    b_tid = assign_type_ids(st.type_space, st.grid.assign(v[b_rows], st.myerson), P["q"].to_numpy()[b_rows],
                            P["hw"].to_numpy()[b_rows])
    ok = b_tid >= 0
    Nb = np.bincount(b_tid[ok], minlength=st.n_types).astype(float)
    use = Nb >= thr
    j = g["type"].to_numpy()
    row = {"phase": m, "arrivals": len(g), "admitted": int(g["admit"].sum()), "oracle pass": int(g["k_hat"].sum()),
           "arrivals on carried types": int((~use[j]).sum()) if len(j) else 0}
    for name in ("a pooled", "b bucket", "d residual", "e phi_gen"):
        pb_ = np.divide(np.bincount(b_tid[ok], weights=PHI[name][b_rows][ok], minlength=st.n_types), Nb,
                        out=np.zeros(st.n_types), where=Nb > 0)
        r_new = st.r_tilde + L2 * np.where(use, pb_ - st.phi_bar, 0.0)
        k_new = (r_new[j] - g["shadow_cost"].to_numpy()) > 0
        adm, kh = g["admit"].to_numpy(), g["k_hat"].to_numpy()
        row[f"{name}: would reject"] = int((adm & ~k_new).sum())
        row[f"{name}: would pass oracle"] = int((~kh & k_new).sum())
    rows.append(row)
FL = pd.DataFrame(rows).set_index("phase")
FL.loc["total"] = FL.sum()
display(FL.T)

# %% [markdown]
# ## Section 6: Anomalies (numbers only)

# %%
anom = [
    ("Phase 0 phi_bar source", "phase0.py:317 averages bids.phi_rate = phi_gen (generator, conditional); ReInit "
     "(reinit.py:88) and the realised reward (theory.py:135) use the pooled F_hat. Phase-1 r~ and every other phase "
     "use different phi definitions."),
    ("data.n0 in default.yaml", f"{BASE.data.n0:,} -> |stream| = {S.n:,} (notebook 03 ran with 20,000 -> 4,133)."),
    ("pooled sigma_hat vs generator sigma", f"{m_a.sigma:.3f} (population), {st0.myerson.sigma:.3f} (Phase 0) vs {SIGMA}"),
    ("share phi<0: pooled vs analytic Phi(z*(sigma_hat))", f"{100 * (phi_a < 0).mean():.2f}% vs {100 * norm.cdf(z_star(m_a.sigma)):.2f}%"),
    ("share phi_gen<0 vs Phi(z*(0.3))", f"{100 * (phi_gen < 0).mean():.2f}% vs {100 * norm.cdf(z_star(SIGMA)):.2f}%"),
    ("negative payments (online run)", f"{int((ARR.loc[ARR['admit'], 'payment'] < 0).sum())} / {int(ARR['admit'].sum())} admitted pay < 0"),
    ("Phase-0 types with phi_tilde < 0 (payment rule, bin floor)", f"{int((st0.phi_tilde[p0m] < 0).sum())} / {int(p0m.sum())} populated "
     f"(p-mass {100 * st0.p[p0m][st0.phi_tilde[p0m] < 0].sum() / st0.p[p0m].sum():.1f}%)"),
    ("phi_gen inv_hazard set to 0 where f <= 1e-12", f"{int(((P['phi_gen'] == P['v'])).sum())} jobs have phi_gen == v exactly"),
    ("bucket count / min size", f"{len(btab)} buckets, min {btab['n'].min()} jobs"),
    ("online phases with < n_min arrivals (grid / F_hat NOT refit)",
     f"{int((pd.DataFrame(res.phases)['n_arrivals'] < BASE.phase_transition.n_min).sum())} / {len(res.phases)} phases "
     f"(n_min = {BASE.phase_transition.n_min}); fitted sigma per phase state: "
     + ", ".join(f"{s.myerson.sigma:.4f}" for s in model.states)),
    ("lognormal phi monotone in v? (no ironing in the mechanism)",
     "; ".join(f"sigma={s}: {100 * (np.diff(MyersonModel(0.0, s).phi_job(np.exp(s * np.linspace(-4, 4, 4001)))) < 0).mean():.1f}% "
               f"of a z-grid decreasing" for s in (SIGMA, round(m_a.sigma, 3), 1.5, round(m_c1.sigma, 3), 2.5))),
    ("arrivals whose type stats were carried forward (5.b)", f"{int(FL.loc['total', 'arrivals on carried types'])} / {len(ARR)}"),
]
with pd.option_context("display.max_colwidth", None):
    display(pd.DataFrame(anom, columns=["item", "value"]).set_index("item"))

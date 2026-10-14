# %% [markdown]
# # DLENT-RT — Known bid distribution: check
#
# The bid distribution is configured and treated as **known**: markup m = v / c_j ~ LogNormal(−σ²/2, σ), σ =
# `bid.sigma`, c_j = `cost_base` (the generator's reference cost), φ_v(v | c_j) = c_j · φ_m(v / c_j)
# (`MyersonModel.phi_job`, `bid.known_distribution: true`). The old behaviour (pooled lognormal MLE, refit at every
# ReInit) is the ablation `bid.known_distribution: false`.
#
# Config: default + `bid.sigma=0.3`, `data.n0=10000`. Real trace, synthetic fallback with
# `DLENT_SYNTHETIC=1` (or a missing trace). Rebuild: `python notebooks/src/build.py 05_known_distribution_check --execute`.

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

from dlent_rt.config import default_config
from dlent_rt.data import load_dataset
from dlent_rt.phase0 import run_phase0
from dlent_rt.bid import MyersonModel
from dlent_rt.models.theory import TheoryModel

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
# reference categorical palette, fixed order (slots 1-4)
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK2 = "#52514e"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "lines.linewidth": 2})


def _replace_path(obj, path, value):
    if len(path) == 1:
        return dataclasses.replace(obj, **{path[0]: value})
    return dataclasses.replace(obj, **{path[0]: _replace_path(getattr(obj, path[0]), path[1:], value)})

def apply_overrides(cfg, overrides: dict):
    for k, v in overrides.items():
        cfg = _replace_path(cfg, k.split("."), v)
    return cfg

cfg = default_config()
cfg = apply_overrides(cfg, {"data.batch_csv": str(ROOT / cfg.data.batch_csv),
                            "data.grid_csv": str(ROOT / cfg.data.grid_csv)})
BASE = apply_overrides(cfg, {"bid.sigma": 0.3, "data.n0": 10000})
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
NEW = apply_overrides(BASE, {"bid.known_distribution": True}).validate()
OLD = apply_overrides(BASE, {"bid.known_distribution": False}).validate()
SIGMA = NEW.bid.sigma
L1, L2, L3 = NEW.objective.lambda1, NEW.objective.lambda2, NEW.objective.lambda3
print(f"seed: run.seed={NEW.run.seed} (H bids), stream bids={NEW.run.seed + NEW.online.bid_seed_offset}, "
      f"thinning={NEW.run.seed + NEW.online.thinning_seed_offset}")
print(f"bid: lognormal sigma={SIGMA}  normalization: quantile_linear "
      f"q={NEW.normalization.quantile}  capacity: {NEW.capacity.mode} fraction={NEW.capacity.fraction}  "
      f"model: {NEW.model.kind}")

# %%
ds = load_dataset(NEW)
st_new = run_phase0(ds, NEW)
st_old = run_phase0(ds, OLD)
print(f"data.n0 = {NEW.data.n0:,} -> |H| = {ds.H.n:,}, |stream| = {ds.stream.n:,}, T = {ds.T:,}")


class RecordingModel(TheoryModel):
    def run(self):
        self.states = []
        return super().run()
    def _new_phase(self, state, tau_prev, W_prev):
        self.states.append(state)
        super()._new_phase(state, tau_prev, W_prev)

runs = {}
for name, c, st in (("new", NEW, st_new), ("old", OLD, st_old)):
    t0 = time.time()
    m = RecordingModel(ds, st, c)
    r = m.run()
    runs[name] = (m, r, pd.DataFrame(r.arrivals))
    print(f"{name}: {len(r.arrivals):,} arrivals, {r.summary['n_phases']} phases, "
          f"{r.summary['n_admitted']:,} admitted ({time.time() - t0:.0f} s)")
S = runs["new"][0].stream
c_s = S.cost_base
phi_new = np.asarray(st_new.myerson.phi_job(S.v, c_s))          # the one phi (known distribution)
phi_old_p0 = np.asarray(st_old.myerson.phi_job(S.v))                # Phase-0 pooled fit, on the stream bids
phi_old_online = runs["old"][2]["phi_mech"].to_numpy()          # phase F_hat, as logged online


def z_star(sig):
    return brentq(lambda z: sig * norm.sf(z) / norm.pdf(z) - 1.0, -30, 30)
P_TH = norm.cdf(z_star(SIGMA))

# %% [markdown]
# ## A. Share φ < 0 and quantiles of φ (stream bids)

# %%
def row(name, phi, sigma):
    q10, q50, q90 = np.quantile(phi, [0.1, 0.5, 0.9])
    return {"phi": name, "sigma": sigma, "share phi<0 %": 100 * (phi < 0).mean(),
            "phi p10 $": q10, "phi p50 $": q50, "phi p90 $": q90}

sig_old_ph = [s.myerson.sigma for s in runs["old"][0].states]
tabA = pd.DataFrame([
    row("old: Phase-0 pooled fit (known_distribution=false)", phi_old_p0, st_old.myerson.sigma),
    row("old: online phi_mech (phase F_hat refits)", phi_old_online,
        f"{min(sig_old_ph):.4f}..{max(sig_old_ph):.4f}"),
    row("new: known sigma, c_j * phi_m(v / c_j)", phi_new, SIGMA),
    {"phi": "theory Phi(z*(sigma))", "sigma": SIGMA, "share phi<0 %": 100 * P_TH},
]).set_index("phi")
display(tabA.style.format(precision=4, na_rep=""))
print(f"sigma used (new) = {SIGMA};  sigma_hat old: Phase 0 = {st_old.myerson.sigma:.4f}, "
      f"online phases = {', '.join(f'{s:.4f}' for s in sig_old_ph)};  z*({SIGMA}) = {z_star(SIGMA):.4f}")
print(f"sanity: new phi vs generator phi, max |diff| = {np.max(np.abs(phi_new - S.phi_gen)):.3e} $")

# %% [markdown]
# ## B. φ(v) for four cost levels, and share φ < 0 by size decile
# Left: new φ_v(v | c) for c at the p10 / p50 / p90 / p99 of c_j (stream), each over v ∈ [0.3c, 2.5c]; the dot is
# the reserve r(c) = c · φ_m⁻¹(0), where φ crosses zero. The dashed grey line is the old pooled Phase-0 φ̂(v), which
# does not depend on c. Right: share φ < 0 by decile of c_j (size).

# %%
mdl = st_new.myerson
r_m = float(mdl.phi_inv(0.0))                       # reserve markup, same for every c
levels = dict(zip(["low (p10)", "mid (p50)", "high (p90)", "very high (p99)"],
                  np.quantile(c_s, [0.10, 0.50, 0.90, 0.99])))
fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 4.4))
for (lab, c), col in zip(levels.items(), (C1, C2, C3, C4)):
    vv = np.geomspace(0.3 * c, 2.5 * c, 300)
    ax.plot(vv, mdl.phi_job(vv, c), color=col, label=f"c = {c:,.2f} $ {lab}")
    ax.plot([r_m * c], [0.0], "o", ms=8, color=col, mec="white", mew=2)
    ax.annotate(f"r = {r_m * c:,.2f}", (r_m * c, 0), textcoords="offset points", xytext=(4, 8), fontsize=8,
                color=INK2)
vall = np.geomspace(0.3 * min(levels.values()), 2.5 * max(levels.values()), 400)
ax.plot(vall, st_old.myerson.phi_job(vall), "--", color="#9a9992", lw=1.5,
        label=f"old pooled φ̂ (σ̂ = {st_old.myerson.sigma:.2f})")
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=max(1.0, min(levels.values())))
ax.set_xlabel("bid v [$]"); ax.set_ylabel("φ(v) [$] (symlog)")
ax.set_title(f"φ_v(v | c) = c·φ_m(v/c), σ = {SIGMA}; reserve markup r_m = {r_m:.3f}", fontsize=10)
ax.legend(fontsize=8, frameon=False, loc="upper left")

dec = pd.qcut(c_s, 10, labels=False, duplicates="drop") + 1
D = pd.DataFrame({"dec": dec, "new": phi_new < 0, "old": phi_old_p0 < 0, "old online": phi_old_online < 0})
by = 100 * D.groupby("dec")[["new", "old", "old online"]].mean()
bx.plot(by.index, by["new"], "-o", color=C1, ms=8, label="new (known σ)")
bx.plot(by.index, by["old"], "-o", color=C2, ms=8, label="old (Phase-0 pooled fit)")
bx.plot(by.index, by["old online"], ":o", color=C2, ms=6, mfc="white", label="old (online phase F̂)")
bx.axhline(100 * P_TH, color=INK2, lw=1, ls="--")
bx.annotate(f"Φ(z*) = {100 * P_TH:.1f}%", (1, 100 * P_TH), textcoords="offset points", xytext=(0, -14),
            fontsize=8, color=INK2)
bx.set_xticks(range(1, 11)); bx.set_ylim(0, 100)
bx.set_xlabel("decile of reference cost c_j (1 = smallest)"); bx.set_ylabel("share φ < 0 [%]")
bx.set_title("share φ < 0 by size decile (stream)", fontsize=10)
bx.legend(fontsize=8, frameon=False)
plt.tight_layout(); plt.show()
display(by.T.style.format(precision=1))

# %% [markdown]
# ## C. Same φ everywhere, σ constant across phases (new run, default capacity, theory model)

# %%
m_new, r_new, A_new = runs["new"]
v_fix, c_fix = S.v[:1000], c_s[:1000]                 # a fixed set of bids
ref = np.asarray(st_new.myerson.phi_job(v_fix, c_fix))
b0 = st_new.bids
phi_H = np.asarray(st_new.myerson.phi_job(b0.v_rate, b0.cost_base))
diffC = {
    "Phase 0 type reward vs ReInit states (fixed bids)":
        max(np.max(np.abs(np.asarray(s.myerson.phi_job(v_fix, c_fix)) - ref)) for s in m_new.states),
    "Phase 0 vs realised online reward (phi_mech, all arrivals)":
        float(np.max(np.abs(A_new["phi_mech"].to_numpy() - phi_new))),
    "ReInit per-job phi (state.bids.phi_rate) vs Phase-0 function":
        max((float(np.max(np.abs(s.bids.phi_rate - st_new.myerson.phi_job(s.bids.v_rate, s.bids.cost_base))))
             for s in m_new.states[1:]), default=np.nan),
    "Phase 0 phi_job on H vs generator phi (H)": float(np.max(np.abs(phi_H - b0.phi_rate))),
}
display(pd.DataFrame({"max |diff| $": diffC}).style.format("{:.3e}"))
sig_states = sorted({s.myerson.sigma for s in m_new.states})
sig_arr = sorted(A_new["sigma_used"].unique())
sig_ph = sorted({p["sigma"] for p in r_new.phases})
print(f"states: {len(m_new.states)} (Phase 0 + {len(m_new.states) - 1} ReInit), sigma in states = {sig_states}, "
      f"in arrivals log = {sig_arr}, in phase log = {sig_ph}, all known = {all(s.myerson.known for s in m_new.states)}")
SIGMA_CONST = sig_states == sig_arr == sig_ph == [SIGMA]

# %% [markdown]
# ## D. Before / after (one online run each)
# R_job = λ1·q·v + λ2·(φ − C_elec) − λ3·s_C·K for every arrival with that run's φ (`phi_mech`) and s_C ("w/o carbon"
# drops the last term). "Re-scored" = Σ over the run's admitted jobs of R_job with the **new** φ and the new s_C.

# %%
def job_R(A, phi, s_C, carbon=True):
    i = A["i"].to_numpy()
    R = L1 * A["q"].to_numpy() * A["v"].to_numpy() + L2 * (phi - S.C_elec[i])
    return R - L3 * s_C * S.C_carbon[i] if carbon else R

rows = {}
for name, st in (("old", st_old), ("new", st_new)):
    m, r, A = runs[name]
    adm = A["admit"].to_numpy()
    Rj = job_R(A, A["phi_mech"].to_numpy(), st.carbon_price)
    Rj_nc = job_R(A, A["phi_mech"].to_numpy(), st.carbon_price, carbon=False)
    R_new_phi = job_R(A, phi_new[A["i"].to_numpy()], st_new.carbon_price)
    rows[name] = {
        "s_C Phase 0 [$/kgCO2]": st.carbon_price,
        "share arrivals R_job<0 (with carbon) %": 100 * (Rj < 0).mean(),
        "share arrivals R_job<0 (w/o carbon) %": 100 * (Rj_nc < 0).mean(),
        "admitted": int(adm.sum()),
        "admitted with R_job<0": int((adm & (Rj < 0)).sum()),
        "share admitted payments phi(L)<0 %": 100 * (A.loc[adm, "payment"] < 0).mean(),
        "R_total logged [$]": r.summary["R_total"],
        "R_total re-scored with new phi [$]": float(R_new_phi[adm].sum()),
        "phases": r.summary["n_phases"],
        "threshold_price logged (non-NaN) %": 100 * A["threshold_price"].notna().mean(),
    }
tabD = pd.DataFrame(rows)
display(tabD.style.format(precision=4))
print(f"s_C old = {st_old.carbon_price:.10g}, new = {st_new.carbon_price:.10g}, "
      f"rel. diff = {abs(st_new.carbon_price / st_old.carbon_price - 1):.2e}")

# %% [markdown]
# ## E. Verdict (numbers only)

# %%
neg_new = 100 * (phi_new < 0).mean()
se = 100 * np.sqrt(P_TH * (1 - P_TH) / len(phi_new))
print(f"1. sigma constant over all phases of the new run: {SIGMA_CONST} "
      f"({len(m_new.states)} states, sigma = {sig_states})")
print(f"2. share phi<0 (new, stream): {neg_new:.2f}% vs Phi(z*) = {100 * P_TH:.2f}% "
      f"(|diff| = {abs(neg_new - 100 * P_TH):.2f} pp, MC se = {se:.2f} pp); "
      f"old Phase-0 fit: {100 * (phi_old_p0 < 0).mean():.2f}%, old online: {100 * (phi_old_online < 0).mean():.2f}%")
print(f"3. by c_j decile, new: {by['new'].min():.1f}..{by['new'].max():.1f}%; "
      f"old Phase-0 fit: {by['old'].iloc[0]:.1f}% (decile 1) -> {by['old'].iloc[-1]:.1f}% (decile 10)")
print(f"4. max |phi diff| between Phase 0, ReInit, online: {max(diffC.values()):.3e} $")
print(f"5. s_C old vs new: {st_old.carbon_price:.6g} vs {st_new.carbon_price:.6g}")
anom = []
if not SIGMA_CONST: anom.append("sigma not constant")
if abs(neg_new - 100 * P_TH) > 4 * se: anom.append("share phi<0 outside 4 MC se of Phi(z*)")
if max(diffC.values()) > 1e-6: anom.append("phi differs across phases")
print("6. anomalies:", "; ".join(anom) if anom else "none flagged by these checks")

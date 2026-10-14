"""
Reward-identity check (READ-ONLY, week 30, Stage A). Real trace only.

Question: notebook 03 scores the online admitted set at +48k $ (logged) but -157k $ with its "yardstick",
the static full-stream tau* policy at -227k $, while the fluid OPT is +149k $. With a known phi and a
non-binding capacity, admitting every reference type with r_ref > 0 must give, in sample,
    sum_j N_j r_ref_j^+  ==  sum of R_job over the jobs of those types  ==  lambda_hat_ref * T.
This script locates every reward computation, checks the identity term by term, and recomputes the
notebook's totals with ONE job-level reward function.

    dlent_venv/bin/python notebooks/src/reward_identity_check.py

Config: default (configs/default.yaml: from_amax capacity, n0 = 10,000) + bid.strategy=lognormal,
bid.sigma=0.3 -- the configuration notebook 03 was executed with (scale 1.0). Nothing is modified.
"""
from __future__ import annotations

import dataclasses, re, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import lognorm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.simplefilter("ignore")

from dlent_rt.config import default_config
from dlent_rt.data import load_dataset
from dlent_rt.phase0 import run_phase0, assign_type_ids, type_arrays, scalarized_reward
from dlent_rt.models.theory import TheoryModel
from dlent_rt.models.core.occupancy import Occupancy
from dlent_rt.models.core.reinit import build_window_state

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 120)
pd.set_option("display.float_format", lambda x: f"{x:,.6g}")


def section(t):
    print("\n" + "=" * 110 + f"\n{t}\n" + "=" * 110)


# --------------------------------------------------------------------------- 1. where rewards are computed
section("1. Where job-level / type-level / logged / yardstick rewards are computed (file:line, found by regex)")
SITES = [
    ("dlent_rt/phase0.py", r"phi_job = \(np.asarray\(myerson.phi_job", "Phase 0: per-job phi (known: phi_job(v, c_j))"),
    ("dlent_rt/phase0.py", r"phi_bar      = _mean\(phi_job\)", "Phase 0: type mean phi_bar"),
    ("dlent_rt/phase0.py", r"raw_sat  = priority_of_type \* v_bar", "scalarized_reward: SW = q * v_bar (TYPE level)"),
    ("dlent_rt/phase0.py", r"raw_prof = phi_bar - C_elec_bar", "scalarized_reward: Pi = phi_bar - C_elec_bar"),
    ("dlent_rt/phase0.py", r"V_carb = raw_carb \* s_C", "quantile_linear: carbon $ = s_C * K_bar"),
    ("dlent_rt/phase0.py", r"r_tilde = np.asarray\(scalarize", "r_tilde (type reward, $/job)"),
    ("dlent_rt/phase0.py", r"p = N / max\(n_total_for_p", "Phase 0: p_j = N_j / H_span_steps (per tick)"),
    ("dlent_rt/models/core/reinit.py", r"phi_arr = \(np.asarray\(myerson.phi_job", "ReInit / hindsight: per-job phi"),
    ("dlent_rt/models/core/reinit.py", r"r_tilde, r_rev, norm_ref, norm_deg, _ = scalarized_reward", "ReInit / hindsight: r_tilde via scalarized_reward"),
    ("dlent_rt/models/core/reinit.py", r"p = N / max\(W, 1\)", "ReInit / hindsight: p_j = N_j / W"),
    ("dlent_rt/models/core/reinit.py", r"def hindsight_lambda", "hindsight benchmark (lam_hat per tick)"),
    ("dlent_rt/models/theory.py", r"phi = float\(st.myerson.phi_job", "online: per-job phi of the arrival"),
    ("dlent_rt/models/theory.py", r"R = \(l1 \* sat \+ l2 \* prof - l3 \* carb \* self.tax\) / self.s_norm", "online: LOGGED job reward R"),
    ("dlent_rt/models/theory.py", r"lam_all, _ = hindsight_lambda", "summary: bench_global = lam_hat_ref * T (fluid OPT)"),
    ("notebooks/src/03_main_algorithm_analysis.py", r"def job_reward", "nb03: job_reward(model, phi) (yardstick formula)"),
    ("notebooks/src/03_main_algorithm_analysis.py", r"phi_ref = np.asarray\(st_ref.myerson.phi\(S.v\)\)", "nb03: phi_ref = myerson.phi(v)  <-- no c_j"),
    ("notebooks/src/03_main_algorithm_analysis.py", r"R_own=job_reward\(model, st.myerson.phi\(S.v\)\)", "nb03 run_static: R_own = myerson.phi(v)  <-- no c_j"),
    ("notebooks/src/03_main_algorithm_analysis.py", r'a\["PI"\] = a\["phi_mech"\]', "nb03 arrivals_frame: R_job from logged phi_mech"),
    ("notebooks/src/03_main_algorithm_analysis.py", r"F_hat = lognorm.cdf\(v_sorted", "nb03 accuracy: KS F_hat (markup params) vs raw bids"),
    ("notebooks/src/03_main_algorithm_analysis.py", r"phi_k = np.asarray\(st.myerson.phi\(S.v\)\)", "nb03 accuracy: phi_k = myerson.phi(v)  <-- no c_j"),
    ("notebooks/src/03_main_algorithm_analysis.py", r"r_ref, \*_ = scalarized_reward", "nb03 accuracy: r_ref per phase-k type (from phi_ref)"),
    ("notebooks/src/03_main_algorithm_analysis.py", r"Ry = float\(R_ref\[adm\].sum\(\)\)", "nb03 baseline_table: R yardstick = sum R_ref"),
    ("notebooks/src/04_phi_diagnostic.py", r"st0.myerson.phi\(S.v\)", "nb04: myerson.phi(v) (pooled-fit era)"),
]
rows = []
for f, pat, what in SITES:
    lines = (ROOT / f).read_text().splitlines()
    hits = [i + 1 for i, l in enumerate(lines) if re.search(pat, l)]
    rows.append({"site": f"{f}:{','.join(map(str, hits)) or 'NOT FOUND'}", "what": what})
print(pd.DataFrame(rows).to_string(index=False))

# --------------------------------------------------------------------------- setup
cfg = default_config()
cfg = dataclasses.replace(cfg, bid=dataclasses.replace(cfg.bid, strategy="lognormal", sigma=0.3)).validate()
assert Path(cfg.data.batch_csv).is_file(), "real trace required"
L1, L2, L3 = cfg.objective.lambda1, cfg.objective.lambda2, cfg.objective.lambda3
t0 = time.time()
ds = load_dataset(cfg)
st0 = run_phase0(ds, cfg)
print(f"\nconfig: n0={cfg.data.n0:,} |H|={ds.H.n:,} capacity.mode={cfg.capacity.mode} c={ds.capacity} "
      f"known_distribution={cfg.bid.known_distribution} sigma={cfg.bid.sigma} normalization="
      f"{cfg.normalization.strategy} q={cfg.normalization.quantile} lambda=({L1},{L2},{L3})")
model = TheoryModel(ds, st0, cfg)
res = model.run()
S, T, sC = model.stream, int(ds.T), st0.carbon_price
print(f"online run: |stream|={S.n:,} T={T:,} phases={res.summary['n_phases']} admitted={res.summary['n_admitted']:,} "
      f"R_total={res.summary['R_total']:,.2f} $  ({time.time() - t0:.0f} s)")

# reference (full stream, true D, no fallback) = notebook 03 reference_state = code's hindsight benchmark
cfg_h = dataclasses.replace(cfg, phase_transition=dataclasses.replace(cfg.phase_transition, n_min=2))
idx = np.arange(S.n)
st_ref, _ = build_window_state(S, idx, idx, T, st0, ds, cfg_h, common_scale=model.s_norm, fallback=False)


# ONE job-level reward function (the online loop's formula, theory.py), vectorised
def job_terms(i, phi):
    SW = S.priority[i] * S.v[i]
    PI = phi - S.C_elec[i]
    K = S.C_carbon[i]
    return SW, PI, K, L1 * SW + L2 * PI - L3 * sC * K


phi_true = np.asarray(st_ref.myerson.phi_job(S.v, S.cost_base))       # c_j * phi_m(v / c_j)
phi_nb = np.asarray(st_ref.myerson.phi(S.v))                          # notebook 03: no c_j
R_true = job_terms(idx, phi_true)[3]
R_nb = job_terms(idx, phi_nb)[3]

# --------------------------------------------------------------------------- 2. per-term identity
section("2. Type level vs mean of job level, full-stream reference types (populated types only)")
tid = assign_type_ids(st_ref.type_space, st_ref.grid.assign(S.v, st_ref.myerson), S.priority, S.hw_idx)
assert (tid >= 0).all()
N = np.bincount(tid, minlength=st_ref.n_types).astype(float)
pop = N > 0
_, prio_t, _ = type_arrays(st_ref.type_space, st_ref.grid)


def mean_by(x):
    return np.divide(np.bincount(tid, weights=np.asarray(x, float), minlength=st_ref.n_types), N,
                     out=np.zeros(st_ref.n_types), where=N > 0)


type_lvl = {"SW = q v": prio_t * st_ref.v_bar, "Pi = phi - C_elec": st_ref.phi_bar - st_ref.C_elec_bar,
            "K = E CI [kg]": st_ref.C_carbon_bar, "R = r_tilde": st_ref.r_tilde}
rows = []
for label, phi in (("phi_job(v, c_j)  [loop]", phi_true), ("myerson.phi(v)  [notebook 03]", phi_nb)):
    SW, PI, K, R = job_terms(idx, phi)
    for (name, tl), jl in zip(type_lvl.items(), (SW, PI, K, R)):
        jm = mean_by(jl)
        d = np.abs(tl[pop] - jm[pop])
        rows.append({"job-level phi": label, "term": name, "type-level (p-weighted mean)": np.average(tl[pop], weights=N[pop]),
                     "job-level mean": np.average(jm[pop], weights=N[pop]), "max abs diff": d.max(),
                     "max rel diff": (d / np.maximum(np.abs(tl[pop]), 1e-12)).max()})
T2 = pd.DataFrame(rows)
print(T2.to_string(index=False))
for label, g in T2.groupby("job-level phi", sort=False):
    bad = g[g["max abs diff"] > 1e-6 * np.maximum(1, np.abs(g["type-level (p-weighted mean)"]))]
    print(f"first term that disagrees with {label}: {bad['term'].iloc[0] if len(bad) else 'none'}")

pos = st_ref.r_tilde > 0
in_pos = pos[tid]
lhs = float((N[pos] * st_ref.r_tilde[pos]).sum())
print(f"\nidentity, reference types with r_ref > 0 ({int((pos & pop).sum())} types, {int(in_pos.sum()):,} jobs):")
print(f"  sum_j N_j r_ref_j^+                         = {lhs:,.4f} $")
print(f"  sum R_job over those jobs, phi_job(v, c_j)  = {R_true[in_pos].sum():,.4f} $   (rel diff {abs(R_true[in_pos].sum() / lhs - 1):.2e})")
print(f"  sum R_job over those jobs, notebook phi(v)  = {R_nb[in_pos].sum():,.4f} $")
print(f"  lambda_hat_ref * T (summary bench_global)   = {res.summary['bench_global']:,.4f} $   "
      f"(tau* = {st_ref.tau}, so x_j = 1[r_ref_j > 0])")

# --------------------------------------------------------------------------- 3. units / normalisation
section("3. Units and normalisation")
checks = [
    ("p_j (reference)", "N_j / W with W = T (per tick)", f"max |p - N/T| = {np.abs(st_ref.p - N / T).max():.2e}"),
    ("lambda_hat * T", "sum_j (N_j/T) r_j x_j * T = sum_j N_j r_j x_j ($ per run)",
     f"{st_ref.lam_star * T:,.4f} vs sum N r^+ {lhs:,.4f}"),
    ("p_j (Phase 0)", "N_j / H_span_steps (per tick of H)", f"H_span_steps = {ds.H_span_steps:,} (not T = {T:,})"),
    ("duration", "enters v_ij = A d_bar (LP constraint) only; NOT the reward", "reward terms are per job, no d"),
    ("r_tilde units", "$ per job (type mean of R_job)", "LP objective sum p r = $ per tick"),
    ("phi in reward (type)", "phi_bar = mean phi_job(v, c_j) over the type's jobs", "Phase 0, ReInit, hindsight"),
    ("phi in reward (online)", "phi_job(v_i, c_i) of the arrival", "theory.py"),
    ("phi(L) bin floor", "payment only (NaN per type under known_distribution)", f"st_ref.phi_tilde all NaN: {bool(np.isnan(st_ref.phi_tilde).all())}"),
    ("phi_gen (generator)", "diagnostic; equals phi_job analytically",
     f"max |phi_job - phi_gen| = {np.abs(phi_true - S.phi_gen).max():.2e} $"),
    ("phi in notebook 03", "myerson.phi(v): MARKUP phi on raw $ bids",
     f"max |phi_nb - phi_job| = {np.abs(phi_nb - phi_true).max():,.2f} $"),
    ("s_C", "frozen at Phase 0", f"state0 {st0.carbon_price:.6f}, ref {st_ref.carbon_price:.6f}, loop {model.tax:.6f}"),
    ("reward scale", "s_norm = norm_ref[0] = 1 (quantile_linear: no divisor)", f"s_norm = {model.s_norm}"),
    ("type space", "reference grid = refit on the full stream; phase k grid refit on phase k-1 arrivals",
     f"ref grid edges [{st_ref.grid.edges[0]:.4g}, {st_ref.grid.edges[-1]:.4g}] vs Phase 0 "
     f"[{st0.grid.edges[0]:.4g}, {st0.grid.edges[-1]:.4g}]"),
    ("jobs included", "reference: all outlier-filtered STREAM jobs (H excluded); Phase 0: H only",
     f"|stream| = {S.n:,}, sum N_ref = {int(N.sum()):,}"),
]
print(pd.DataFrame(checks, columns=["quantity", "definition in code", "value"]).to_string(index=False))

# --------------------------------------------------------------------------- 4. totals with ONE job-level function
section("4. Totals of the notebook's admitted sets, re-scored with ONE job-level function")


def types_under(st):
    return assign_type_ids(st.type_space, st.grid.assign(S.v, st.myerson), S.priority, S.hw_idx)


def run_static(st):
    """notebook 03 run_static: fixed (r~, tau, v), same event-driven loop and physical gate, no thinning."""
    j = types_under(st)
    jj = np.maximum(j, 0)
    khat = np.where(j >= 0, st.r_tilde[jj] - st.v[jj] @ st.tau, -np.inf) > 0
    occ = Occupancy(model.c, record=False)
    adm = np.zeros(S.n, bool)
    for i in range(S.n):
        t = int(S.tick[i])
        occ.release_until(t)
        if khat[i] and occ.fits(S.A[i]):
            occ.add(i, S.A[i], t, int(S.D[i]))
            adm[i] = True
    return adm


A = pd.DataFrame(res.arrivals)
adm_online = np.zeros(S.n, bool)
adm_online[A.loc[A["admit"], "i"].to_numpy()] = True
sets = {"online (DLENT-RT)": adm_online, "static tau* (full stream, ORACLE)": run_static(st_ref),
        "static Phase 0 (H)": run_static(st0), "all jobs of r_ref>0 types": in_pos,
        "all jobs with R_job>0 (no capacity)": R_true > 0}
NB = {"online (DLENT-RT)": (48_391.61, -157_117.07), "static tau* (full stream, ORACLE)": (np.nan, -227_137.0),
      "static Phase 0 (H)": (np.nan, -610_451.0), "all jobs with R_job>0 (no capacity)": (np.nan, 183_677.0)}
rows = []
for name, m in sets.items():
    rows.append({"admitted set": name, "admitted": int(m.sum()),
                 "R (ONE job function, phi_job)": float(R_true[m].sum()),
                 "R (notebook yardstick, phi(v))": float(R_nb[m].sum()),
                 "nb03 logged": NB.get(name, (np.nan, np.nan))[0], "nb03 yardstick": NB.get(name, (np.nan, np.nan))[1],
                 "share R_job<0 %": 100 * (R_true[m] < 0).mean() if m.any() else np.nan})
T4 = pd.DataFrame(rows).set_index("admitted set")
print(T4.to_string())
print(f"logged R_total = {res.summary['R_total']:,.4f} $;  sum logged R over arrivals = {A['R'].sum():,.4f} $;  "
      f"ONE-function total on the online set = {R_true[adm_online].sum():,.4f} $  "
      f"(max |R_logged - R_job| = {np.abs(A.loc[A['admit'], 'R'].to_numpy() - R_true[A.loc[A['admit'], 'i'].to_numpy()]).max():.2e})")
print(f"fluid OPT lambda_hat_ref*T = {res.summary['bench_global']:,.4f} $")

# --------------------------------------------------------------------------- 5. KS and r rel err
section("5. What KS(F_hat, F_emp) and 'r rel err' compare")
m = st0.myerson
v_sorted = np.sort(S.v)
n = S.n
e_hi, e_lo = np.arange(1, n + 1) / n, np.arange(n) / n


def ks(F):
    return max(np.abs(F - e_hi).max(), np.abs(F - e_lo).max())


ks_nb = ks(lognorm.cdf(v_sorted, s=m.sigma, scale=np.exp(m.mu)))
mk = np.sort(S.v / S.cost_base)
ks_mk = ks(lognorm.cdf(mk, s=m.sigma, scale=np.exp(m.mu)))
print(f"F_hat = LogNormal(mu={m.mu:.4f}, sigma={m.sigma}) = the KNOWN MARKUP distribution, identical in every phase "
      f"(never refit) -> KS is the same number in every phase.")
print(f"  notebook: KS(F_markup, empirical CDF of RAW bids v [$])   = {ks_nb:.4f}")
print(f"  correct : KS(F_markup, empirical CDF of markups v / c_j)  = {ks_mk:.4f}   (1.36/sqrt(n) = {1.36 / np.sqrt(n):.4f})")
# r rel err, phase 1 on Phase 0's type space (notebook accuracy_table, k = 1)
tid0 = types_under(st0)
ok0 = tid0 >= 0
N0 = np.bincount(tid0[ok0], minlength=st0.n_types).astype(float)
mb0 = lambda x: np.divide(np.bincount(tid0[ok0], weights=np.asarray(x, float)[ok0], minlength=st0.n_types), N0,
                          out=np.zeros(st0.n_types), where=N0 > 0)
_, prio0, A0 = type_arrays(st0.type_space, st0.grid)
p0 = N0 / N0.sum()
d0 = mb0(S.D)
out = {}
for label, phi in (("notebook phi_ref = phi(v)", phi_nb), ("phi_ref = phi_job(v, c_j)", phi_true)):
    r_ref0, *_ = scalarized_reward(prio0, mb0(S.v), mb0(phi), mb0(S.C_elec), mb0(S.C_carbon), p0, A0 * d0[:, None],
                                   ds, cfg, common_scale=model.s_norm, carbon_price=sC)
    mm = p0 > 0
    err = np.average(np.abs(st0.r_tilde[mm] - r_ref0[mm]), weights=p0[mm])
    out[label] = dict(mean_abs_err=err, rel_err=err / np.average(np.abs(r_ref0[mm]), weights=p0[mm]),
                      sign_mismatch=np.average(np.sign(st0.r_tilde[mm]) != np.sign(r_ref0[mm]), weights=p0[mm]))
print("phase 1 (Phase 0 types) r_tilde (H) vs r_ref (stream jobs on the same types); both are $ per JOB (type means), "
      "weighted by p_ref:")
print(pd.DataFrame(out).T.to_string())
print(f"phi error: notebook mean |phi_k - phi_ref| uses phi(v) on both sides -> 0 under known_distribution; "
      f"with phi_job both sides are also identical (one known phi) -> the metric is 0 by construction.")
print(f"\nmean phi_nb = {phi_nb.mean():,.2f} $ vs mean phi_job = {phi_true.mean():,.2f} $; "
      f"share phi_nb<0 = {100 * (phi_nb < 0).mean():.1f}% vs phi_job<0 = {100 * (phi_true < 0).mean():.1f}%; "
      f"share of bids with markup-pdf f(v) <= 1e-12 (inv_hazard zeroed -> phi_nb = v): "
      f"{100 * (lognorm.pdf(S.v, s=m.sigma, scale=np.exp(m.mu)) <= 1e-12).mean():.1f}%")
print(f"\ndone in {time.time() - t0:.0f} s")

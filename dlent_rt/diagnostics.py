"""
The model check (week 33): every function behind notebooks/01_model_check.ipynb. Read-only: nothing here changes the
model. Each section of the notebook has a `checks_*` function that returns a PASS / FAIL table and the numbers it
shows.

    from dlent_rt import diagnostics as dg
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import integrate, optimize, stats

from .bid import MyersonModel, calibrate_u0, cost_base, cost_rate, fit_sigma
from .config import REPO_ROOT, resolve_path
from .lp_oracle import solve_lp_rs
from .models.core.error_params import make_error_params
from .phase0 import _compute_job_energy, _compute_job_power, _lookup_grid, assign_type_ids, type_arrays
from .reward import carbon_kg, electricity_cost, job_terms, weighted_quantile
from . import theory_eval as te

QS = (0.10, 0.50, 0.90, 0.99)


def check_table(rows) -> pd.DataFrame:
    """[(check, bool, detail)] -> a PASS / FAIL table."""
    return pd.DataFrame([(n, "PASS" if bool(ok) else "FAIL", d) for n, ok, d in rows],
                        columns=["check", "result", "detail"])


def where(path: str, pattern: str) -> str:
    """'path:line' of the first line of a repository file that contains `pattern` (for the D1-D6 checklist)."""
    lines = (REPO_ROOT / path).read_text().splitlines()
    for k, line in enumerate(lines, 1):
        if pattern in line:
            return f"{path}:{k}"
    raise ValueError(f"{pattern!r} not found in {path}")


def stats_table(cols: dict) -> pd.DataFrame:
    rows = []
    for name, x in cols.items():
        x = np.asarray(x, float)
        rows.append({"quantity": name, "count": x.size, "mean": x.mean(),
                     **{f"p{int(q * 100)}": float(np.quantile(x, q)) for q in QS}, "max": x.max()})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 1. Data and limits
# --------------------------------------------------------------------------- #
def raw_jobs(cfg) -> pd.DataFrame:
    """Every job of the file with its effective duration (before the floor, the cap and the quota)."""
    col = cfg.data.columns
    df = pd.read_csv(resolve_path(cfg.data.batch_csv), usecols=[col.collection_id, col.datetime, col.requested_cpu,
                                                                  col.requested_ram, col.duration_hours,
                                                                  col.cpu_hours_real])
    cpu, span = df[col.requested_cpu].to_numpy(float), df[col.duration_hours].to_numpy(float)
    df["eff"] = np.divide(df[col.cpu_hours_real].to_numpy(float), cpu, out=span.copy(), where=cpu > 0)
    df["t"] = pd.to_datetime(df[col.datetime]).dt.tz_convert("UTC").dt.tz_localize(None)
    return df.sort_values("t", kind="stable").reset_index(drop=True)


def checks_data(ds, cfg, raw: pd.DataFrame):
    col, tm = cfg.data.columns, cfg.time
    Q = np.asarray(ds.limit_report.quota)
    kept = raw[col.collection_id].isin(np.concatenate([ds.H.collection_id, ds.stream.collection_id])).to_numpy()
    eff = raw["eff"].to_numpy()
    d = np.concatenate([ds.H.duration_hours, ds.stream.duration_hours])
    A = np.concatenate([ds.H.A, ds.stream.A])
    h_end = raw["t"].iloc[0] + pd.Timedelta(days=cfg.data.h_days)
    over = (raw[col.requested_cpu].to_numpy() > Q[0]) | (raw[col.requested_ram].to_numpy() > Q[1])
    summary = {"jobs in the file": len(raw), "rejected by the quota": int(ds.limit_report.n_quota_rejected),
               "jobs kept": int(kept.sum()), "H (first 7 days)": ds.H.n, "stream": ds.stream.n,
               "stream horizon T (h)": ds.T * tm.tick_seconds / 3600,
               "kept jobs cut by the 24 h cap": int((eff[kept] > tm.d_cap_hours).sum()),
               "kept jobs raised to 1 minute": int((eff[kept] < tm.duration_floor_hours).sum()),
               "quota Q (machines), CPU / RAM": Q.tolist(), "d_max (h)": ds.d_max_steps * tm.tick_seconds / 3600}
    rows = [
        ("only the quota removes jobs (no outlier filter)", kept.sum() + over.sum() == len(raw) and not (kept & over).any(),
         f"{len(raw):,} = {kept.sum():,} kept + {over.sum():,} over the quota"),
        ("every kept request <= Q", np.all(A <= Q[None, :]), f"max CPU {A[:, 0].max():.4f} <= {Q[0]:.4f}, "
                                                              f"max RAM {A[:, 1].max():.4f} <= {Q[1]:.4f}"),
        ("durations inside [1 minute, 24 h]", d.min() >= tm.duration_floor_hours and d.max() <= tm.d_cap_hours + 1e-9,
         f"min {d.min() * 60:.3f} min, max {d.max():.3f} h"),
        ("duration = effective duration (CPU-hours / CPU), floored and capped",
         np.allclose(np.sort(d), np.sort(np.clip(eff[kept], tm.duration_floor_hours, tm.d_cap_hours)), rtol=1e-12),
         "same multiset of durations"),
        ("H = the jobs of the first 7 days", pd.Timestamp(ds.H.datetime[-1]) < h_end <= pd.Timestamp(ds.stream.datetime[0]),
         f"H ends {pd.Timestamp(ds.H.datetime[-1])}, stream starts {pd.Timestamp(ds.stream.datetime[0])}"),
        ("d_max = 24 h and d_max / T <= 0.10", ds.d_max_steps * tm.tick_seconds == tm.d_cap_hours * 3600
         and ds.d_max_diag["ratio_to_T"] <= tm.d_max_ratio_abort, f"d_max / T = {ds.d_max_diag['ratio_to_T']:.4f}"),
    ]
    return check_table(rows), summary


# --------------------------------------------------------------------------- #
# 2. Bid and U0
# --------------------------------------------------------------------------- #
def bid_frame(ds, st, cfg) -> pd.DataFrame:
    """Per H job: priority, cost rate, cost base, bid, markup, phi (Phase 0's bids)."""
    H, b = ds.H, st.bids
    rate = cost_rate(H.A[:, 0], H.A[:, 1], cfg)
    return pd.DataFrame({"priority": H.priority, "rate": rate, "cost_base": b.cost_base, "bid": b.v_rate,
                         "markup": b.v_rate / b.cost_base, "phi": np.asarray(st.myerson.phi_job(b.v_rate, b.cost_base)),
                         "hours": H.duration_hours})


def checks_bid(ds, st, cfg, S):
    H = ds.H
    rate = cost_rate(H.A[:, 0], H.A[:, 1], cfg)
    o = np.argsort(rate, kind="stable")
    med_job = int(o[(len(o) - 1) // 2])
    u0 = calibrate_u0(H, cfg)
    cH = cost_base(H.A[:, 0], H.A[:, 1], H.duration_hours, H.priority, u0, cfg)
    cS = cost_base(S.A[:, 0], S.A[:, 1], S.duration_hours, S.priority, u0, cfg)
    em = cfg.energy_model
    m = st.bids.v_rate / st.bids.cost_base
    summary = {"U0 ($ per hour per job)": u0, "median H job: cores / GB": [float(H.A[med_job, 0] * em.machine_cores),
                                                                           float(H.A[med_job, 1] * em.machine_ram_gb)],
               "median H job: gamma1 cores + gamma2 GB": float(rate[med_job]),
               "mean markup v / cost_base (H)": float(m.mean()), "std of ln markup (H)": float(np.log(m).std())}
    rows = [
        ("U0 = gamma1 cores + gamma2 GB of the median H job", np.isclose(u0, np.median(rate), rtol=1e-12),
         f"U0 = {u0:.5f} $/h; the job at the middle position has {rate[med_job]:.5f} $/h"),
        ("U0 is the same for H and the stream (fixed after Phase 0)", np.isclose(st.u0, u0, rtol=1e-12),
         f"Phase 0 state U0 = {st.u0:.5f}"),
        ("cost_base = (U0 (1 + ln q) + gamma1 cores + gamma2 GB) d, H", np.allclose(st.bids.cost_base, cH, rtol=1e-12),
         "recomputed by hand"),
        ("cost_base formula, stream", np.allclose(S.cost_base, cS, rtol=1e-12), "recomputed by hand"),
        ("mean bid = cost_base (mean markup 1, within 1%)", abs(m.mean() - 1) < 0.01, f"mean markup {m.mean():.4f}"),
    ]
    return check_table(rows), summary


def bid_tier_table(bf: pd.DataFrame) -> pd.DataFrame:
    g = bf.groupby("priority")
    return pd.DataFrame({"jobs": g.size(), "1 + ln q": 1 + np.log(g.size().index.to_numpy(float)),
                         "mean bid $": g["bid"].mean(), "mean cost base $": g["cost_base"].mean(),
                         "mean bid per hour $": g.apply(lambda d: (d["bid"] / d["hours"]).mean()),
                         "phi < 0 %": g["phi"].apply(lambda s: 100 * (s < 0).mean())}).reset_index()


# --------------------------------------------------------------------------- #
# 3. Virtual value
# --------------------------------------------------------------------------- #
def expected_phi_markup(sigma: float) -> float:
    """E[phi_m] = 1 - sigma int m(z) (1 - Phi(z)) dz by quadrature (exactly 0 for the lognormal markup)."""
    mu = -sigma ** 2 / 2
    I, _ = integrate.quad(lambda z: np.exp(mu + sigma * z) * stats.norm.sf(z), -60, 60, limit=500, epsabs=1e-13)
    return float(1.0 - sigma * I)


def checks_phi(st, cfg, phase_sigmas=None):
    sig0, sig = cfg.bid.sigma, st.myerson.sigma
    E = {s: expected_phi_markup(s) for s in (0.1, sig0, sig, 0.5)}
    mdl = MyersonModel.known_markup(sig)
    z = np.linspace(-6, 6, 4001)
    phi = mdl.phi_markup(np.exp(mdl.mu + sig * z))
    rows = [("E[phi_m] = 0 by quadrature (sigma = 0.1, 0.3, refit, 0.5)", all(abs(v) < 5e-3 for v in E.values()),
             ", ".join(f"{s:.4f}: {v:.1e}" for s, v in E.items())),
            ("phi_m increasing (z in [-6, 6])", np.all(np.diff(phi) > 0), f"min step {np.diff(phi).min():.2e}"),
            ("sigma refit on H within 0.01 of 0.3", abs(sig - sig0) < 0.01, f"refit sigma = {sig:.4f}")]
    if phase_sigmas is not None and len(phase_sigmas):
        ps = np.asarray(phase_sigmas, float)
        rows.append(("sigma refit at every ReInit within 0.01 of 0.3", np.all(np.abs(ps - sig0) < 0.01),
                     f"{ps.size} phases, sigma in [{ps.min():.4f}, {ps.max():.4f}]"))
    return check_table(rows), dict(E_phi=E, sigma_refit=sig, reserve_markup=float(mdl.phi_inv(0.0)))


# --------------------------------------------------------------------------- #
# 4. Objectives
# --------------------------------------------------------------------------- #
def type_terms(st, cfg) -> pd.DataFrame:
    """Per type: p, V (lambda1 v_bar), Pi (lambda2 (phi_bar - C_elec_bar)), carbon $ (lambda3 s_C K_bar), r~."""
    lam = cfg.objective
    _, prio, _ = type_arrays(st.type_space, st.grid)
    ids = np.arange(st.n_types)
    ts = st.type_space
    return pd.DataFrame({"type": ids, "p": st.p, "priority": prio, "hw": ids % ts.n_hw,
                         "bin": ids // (len(ts.priorities) * ts.n_hw) + 1,
                         "customer value": lam.lambda1 * st.v_bar, "profit": lam.lambda2 * (st.phi_bar - st.C_elec_bar),
                         "carbon $": lam.lambda3 * st.carbon_price * st.C_carbon_bar, "r_tilde": st.r_tilde})


def term_shares(tt: pd.DataFrame, x=None) -> dict:
    """Share of sum |term| (p-weighted) from each term, over populated types or over admitted mass p x."""
    w = tt.p.to_numpy() * (1.0 if x is None else np.asarray(x))
    a = {k: float((w * np.abs(tt[k])).sum()) for k in ("customer value", "profit", "carbon $")}
    s = sum(a.values())
    return {k: v / s for k, v in a.items()}


def checks_objectives(ds, st, cfg, S):
    lam = cfg.objective
    tt = type_terms(st, cfg)
    pop = st.p > 0
    w = st.p[pop]
    ni = st.norm_info
    q = cfg.normalization.quantile
    Q_v = weighted_quantile(st.v_bar[pop], w, q)
    Q_pi = weighted_quantile(np.abs(st.phi_bar - st.C_elec_bar)[pop], w, q)
    K_q = weighted_quantile(st.C_carbon_bar[pop], w, q)
    t = job_terms(S.v, S.cost_base, S.C_elec, S.C_carbon, st.myerson, st.carbon_price, lam)
    VPi = float((st.p * st.v_bar).sum() / (st.p * np.abs(st.phi_bar - st.C_elec_bar)).sum())
    summary = {"term shares (sum |term|, p-weighted)": term_shares(tt), "sum p V / sum p |Pi| over H's types": VPi,
               "Q_0.9(V) $": Q_v, "Q_0.9(|Pi|) $": Q_pi, "D_q $": ni["D_q"], "K_q kg": K_q, "s_C $/kg": st.carbon_price,
               "lambda": (lam.lambda1, lam.lambda2, lam.lambda3)}
    rows = [
        ("all lambda = 1", (lam.lambda1, lam.lambda2, lam.lambda3) == (1.0, 1.0, 1.0), str((lam.lambda1, lam.lambda2, lam.lambda3))),
        ("r~ = V + Pi - s_C K (linear, per type)", np.allclose(tt["customer value"] + tt["profit"] - tt["carbon $"], st.r_tilde,
                                                             rtol=1e-12, atol=1e-12), "max abs diff "
         f"{np.abs(tt['customer value'] + tt['profit'] - tt['carbon $'] - st.r_tilde).max():.1e}"),
        ("s_C = D_q / K_q, D_q = (Q_0.9(V) + Q_0.9(|Pi|)) / 2, recomputed",
         np.isclose(st.carbon_price, 0.5 * (Q_v + Q_pi) / K_q, rtol=1e-12), f"s_C = {st.carbon_price:.4f} $/kg"),
        ("customer value V = v (no priority factor), stream", np.array_equal(t.cv, S.v), "job_terms(...).cv is the bid"),
        ("SW = v - C_elec - s_C K per job (metric only)", np.allclose(t.sw, S.v - S.C_elec - st.carbon_price * S.C_carbon,
                                                                       rtol=1e-12, atol=1e-12),
         "reward.job_terms(...).sw; no decision reads it"),
        ("priority enters only the bid: cost_base / rate grows as 1 + ln q at a fixed size",
         _priority_only_in_bid(cfg, st.u0), "hand-made jobs, q = 1..5"),
    ]
    return check_table(rows), summary, t


def _priority_only_in_bid(cfg, u0) -> bool:
    q = np.arange(1, 6, dtype=float)
    c = cost_base(np.full(5, 0.01), np.full(5, 0.01), np.ones(5), q, u0, cfg)
    return bool(np.allclose(np.diff(c), u0 * np.diff(np.log(q)), rtol=1e-12))


# --------------------------------------------------------------------------- #
# 5. Grid
# --------------------------------------------------------------------------- #
def checks_grid(st, cfg):
    g, v = st.grid, st.bids.v_rate
    k = g.assign(v)
    used = np.unique(k).size
    rho = g.ratio
    summary = {"v_min $": float(g.bnd_v[0]), "v_max $": float(g.bnd_v[-1]), "rho": rho, "K*": g.k_star,
               "bins used by H": used, "worst-case loss 1 - 1/rho": 1 - 1 / rho}
    rows = [("bins cover every bid: bin 1 = [0, L_1), bin K* = [L_K*-1, +inf)",
             np.array_equal(g.assign([0.0, g.bnd_v[0] / 10, g.bnd_v[-1] * 10]), [1, 1, g.k_star]) and g.lower_edges[0] == 0,
             "a bid of 0, one below v_min and one above v_max"),
            ("v_min = smallest bid, v_max = largest bid (no floor)", g.bnd_v[0] == v.min() and np.isclose(g.bnd_v[-1], v.max(), rtol=1e-12),
             f"v_min = {g.bnd_v[0]:.3e}, v_max = {g.bnd_v[-1]:.3e}"),
            ("edges geometric: L_k = v_min rho^k", np.allclose(g.bnd_v, g.bnd_v[0] * rho ** np.arange(g.k_star + 1), rtol=1e-10),
             f"rho = {rho:.5f}"),
            ("K* = 128 bins", g.k_star == 128, str(g.k_star)),
            ("every bid has a bin", np.all((k >= 1) & (k <= g.k_star)), f"{used} of {g.k_star} bins used")]
    return check_table(rows), summary


# --------------------------------------------------------------------------- #
# 6. Cluster and quota
# --------------------------------------------------------------------------- #
def scan_table(scan: dict) -> pd.DataFrame:
    rows = []
    for r in scan["results"]:
        s = r["static"]
        rows.append({"f": r["f"], "(a) tau > 0 in Phase 0 LP and fluid OPT": te.tau_positive(r),
                     "(b) distance to the 20-30% band (points)": 100 * te.band_distance(r), "C CPU": r["C"][0], "C RAM": r["C"][1], "Q CPU": r["quota"][0], "Q RAM": r["quota"][1],
                     "H jobs": r["n_H"], "stream jobs": r["n_stream"], "% time >= 95% full": 100 * s["full_time"],
                     "mean occupancy / C (CPU)": 100 * s["mean_occupancy_over_C"][0],
                     "rejected for space %": 100 * s["rejected_space"],
                     "tau Phase 0 (CPU, RAM)": str(np.round(r["phase0_lp"]["tau"], 5).tolist()),
                     "tau fluid OPT (CPU, RAM)": str(np.round(r["fluid"]["tau"], 5).tolist()),
                     "static reward $": s["split"]["reward"], "fluid OPT $": r["fluid"]["split"]["reward"]})
    out = pd.DataFrame(rows)
    out["picked"] = out.f == scan["selected"]
    return out


def checks_cluster(ds, st, cfg, scan: dict, fluid_tau):
    md = ds.capacity_diag
    its = ds.limit_report.quota_iterations
    Q = np.asarray(ds.limit_report.quota)
    rows = [("C = f x D_bar (H only)", np.allclose(ds.capacity, cfg.capacity.f * md["D_bar"] * md["enlargement"], rtol=1e-12),
             f"f = {cfg.capacity.f}, D_bar = {np.round(md['D_bar'], 4).tolist()}"),
            ("Q = 0.2 C at the fixed point (within 2%)", np.allclose(Q / ds.capacity, cfg.limits.quota_share, rtol=0.02),
             f"Q / C = {np.round(Q / ds.capacity, 4).tolist()}"),
            ("fixed point stopped with a move < 1%", np.all(np.asarray(its[-1]["change"]) < cfg.limits.quota_tol),
             f"{len(its)} steps, last move {np.round(its[-1]['change'], 4).tolist()}"),
            ("f = the rule's pick: (a) tau > 0 in both LPs, then (b) closest to the 20-30% band",
             scan["selected"] == cfg.capacity.f, f"rule picks {scan['selected']}, config f = {cfg.capacity.f}; "
             f"{100 * _full(scan, cfg.capacity.f):.1f}% of the time >= 95% full"),
            ("tau > 0 in the Phase 0 LP (some resource)", np.any(np.asarray(st.tau) > 0), f"tau = {np.round(st.tau, 6).tolist()}"),
            ("tau > 0 in fluid OPT (some resource)", np.any(np.asarray(fluid_tau) > 0), f"tau = {np.round(fluid_tau, 6).tolist()}")]
    iters = pd.DataFrame([{"step": i["iteration"], "Q CPU": i["Q"][0], "Q RAM": i["Q"][1], "jobs kept": i["n_kept"],
                           "H jobs": i["n_H"], "C CPU": i["C"][0], "C RAM": i["C"][1],
                           "move CPU %": 100 * i["change"][0], "move RAM %": 100 * i["change"][1]} for i in its])
    return check_table(rows), iters


def _full(scan, f):
    return next(r["static"]["full_time"] for r in scan["results"] if r["f"] == f)


# --------------------------------------------------------------------------- #
# 7. xi and gamma
# --------------------------------------------------------------------------- #
def eps_D(xi, L):
    return 2.0 * np.sqrt(xi * L) + 2.0 * xi * L


def xi_max(L: float) -> float:
    """The largest xi with 1 - eps_D(xi) - xi > 0 (Theorem 1 not empty), eps_D = 2 sqrt(xi L) + 2 xi L."""
    return float(optimize.brentq(lambda x: 1.0 - eps_D(x, L) - x, 1e-15, 1.0))


def checks_constants(ds, st, cfg):
    ep = make_error_params(st, ds, cfg)
    L = ep.log_I
    xm = xi_max(L)
    need_c = np.asarray(ds.a_max) / xm
    rows = [("xi forced = 0.001, gamma forced = 1", (cfg.constants.xi_override, cfg.constants.gamma_override) == (0.001, 1.0),
             f"data xi = {ds.xi:.3f}, data gamma = {st.gamma:.4g}"),
            ("delta = 1 / T^3, L = ln(|I| / delta)", np.isclose(ep.delta, 1.0 / ds.T ** 3, rtol=1e-12), f"L = {L:.2f}"),
            ("Theorem 1 not empty at the forced xi: 1 - eps_D - xi > 0", 1 - ep.eps_D - ep.xi > 0,
             f"eps_D = {ep.eps_D:.4f}, 1 - eps_D - xi = {1 - ep.eps_D - ep.xi:.4f}"),
            ("Theorem 1 not empty at the data xi", ds.xi <= xm, f"data xi = {ds.xi:.3f} vs xi_max = {xm:.2e}")]
    summary = {"data xi = max_i a_max,i / C_i": ds.xi, "forced xi": ep.xi, "data gamma": st.gamma, "forced gamma": ep.gamma,
               "L = ln(|I| / delta)": L, "xi_max": xm, "eps_D at the forced xi": ep.eps_D,
               "eps_D at the data xi": float(eps_D(ds.xi, L)),
               "C needed for xi <= xi_max (machines, same jobs)": need_c.tolist(),
               "that C / the current C": (need_c / ds.capacity).tolist()}
    return check_table(rows), summary


# --------------------------------------------------------------------------- #
# 8. Phase 0 output and the 9 checks
# --------------------------------------------------------------------------- #
def power_check(ds, cfg, n=5) -> pd.DataFrame:
    H, em = ds.H, cfg.energy_model
    i = np.argsort(H.A[:, 0])[-n:][::-1]
    cores, gb = H.A[i, 0] * em.machine_cores, H.A[i, 1] * em.machine_ram_gb
    kappa = 1.0 + H.scheduling_class[i] * 0.116
    manual = em.pue * (em.p_core * cores * kappa + em.p_gb * gb) / 1000.0
    code = _compute_job_power(H, cfg)[i]
    return pd.DataFrame({"CPU (machines)": H.A[i, 0], "cores": cores, "GB": gb, "sched class": H.scheduling_class[i],
                         "kappa": kappa, "manual kW": manual, "code kW": code, "match": np.isclose(manual, code, rtol=1e-12)})


def energy_frame(ds, cfg, J=None) -> pd.DataFrame:
    J = ds.H if J is None else J
    P, E = _compute_job_power(J, cfg), _compute_job_energy(J, cfg)
    pr, ci = _lookup_grid(J, ds)
    return pd.DataFrame({"cpu": J.A[:, 0], "power_kW": P, "energy_kWh": E, "price": pr, "ci": ci,
                         "C_elec": electricity_cost(E, pr), "kg": carbon_kg(E, ci), "hour": pd.to_datetime(J.datetime).hour})


def grid_coverage(ds, cfg) -> dict:
    """Is every job time (arrival to end, H and stream) inside the hours of the grid file?"""
    g = pd.read_csv(resolve_path(cfg.data.grid_csv), parse_dates=["timestamp_utc"])
    lo, hi = g.timestamp_utc.min(), g.timestamp_utc.max() + pd.Timedelta(hours=1)
    out = {}
    for name, J in (("H", ds.H), ("stream", ds.stream)):
        a = pd.to_datetime(J.datetime)
        e = a + pd.to_timedelta(J.duration_hours, unit="h")
        out[name] = dict(first=str(a.min()), last_end=str(e.max()), covered=bool(a.min() >= lo and e.max() <= hi))
    return out


def phase0_table(ds, st, cfg) -> dict:
    per_h = 3600.0 / cfg.time.tick_seconds
    p = np.asarray(st.p)
    pop = p > 0
    lp = solve_lp_rs(p=p, r=st.r_tilde, v=st.v, c=ds.capacity, presolve=cfg.lp.presolve)
    x = np.zeros(st.n_types)
    x[lp.support_idx] = lp.x
    return {
        "n_types": int(st.n_types), "populated types": int(pop.sum()), "jobs assigned": int(st.n_assigned),
        "sum p (per h)": float(p.sum() * per_h), "true arrival rate of H (per h)": float(ds.H.n / ds.H_span_steps * per_h),
        "U0 ($/h per job)": float(st.u0), "sigma (refit on H)": float(st.myerson.sigma), "rho": float(st.grid.ratio),
        "s_C ($/kg)": float(st.carbon_price),
        "r_tilde min / median / max (populated)": [float(st.r_tilde[pop].min()), float(np.median(st.r_tilde[pop])),
                                                   float(st.r_tilde[pop].max())],
        "positive types %": float(100 * (st.r_tilde[pop] > 0).mean()),
        "positive arrival mass %": float(100 * p[pop & (st.r_tilde > 0)].sum() / p.sum()),
        "r_max (Tests 2-3 reward unit, $)": float(st.r_max),
        "C (machines)": np.asarray(ds.capacity).tolist(), "Q (machines)": list(ds.limit_report.quota),
        "tau": np.asarray(st.tau).tolist(), "LP optimum ($ per h)": float(st.lam_star * per_h),
        "consumption if all admitted (p.v / C)": (p @ st.v / ds.capacity).tolist(),
        "consumption at the optimum = LP fill": ((p * x) @ st.v / ds.capacity).tolist(),
        "types fully / partly / rejected": [int((x[pop] > 1 - 1e-9).sum()), int(((x[pop] > 1e-9) & (x[pop] <= 1 - 1e-9)).sum()),
                                            int((x[pop] <= 1e-9).sum())],
        "_x": x,
    }


def checks_phase0(ds, st, cfg, tab) -> pd.DataFrame:
    """The 9 checks of Phase 0."""
    per_h = 3600.0 / cfg.time.tick_seconds
    ef = energy_frame(ds, cfg)
    tid = assign_type_ids(st.type_space, st.grid.assign(st.bids.v_rate), ds.H.priority, ds.H.hw_idx)
    N = np.bincount(tid, minlength=st.n_types)
    mean = lambda x: np.divide(np.bincount(tid, weights=np.asarray(x), minlength=st.n_types), N,
                               out=np.zeros(st.n_types), where=N > 0)
    tt = type_terms(st, cfg)
    pc = power_check(ds, cfg)
    cov = grid_coverage(ds, cfg)
    fields = [getattr(st, f) for f in ("p", "r_tilde", "v", "d_bar", "v_bar", "phi_bar", "C_elec_bar", "C_carbon_bar",
                                       "energy_bar", "tau", "A_type")]
    rows = [
        ("sum p = true arrival rate of H", np.isclose(tab["sum p (per h)"], tab["true arrival rate of H (per h)"], rtol=1e-9), ""),
        ("manual power = code power (incl. PUE)", bool(pc["match"].all()), ""),
        ("stored C_elec_bar = recomputed", bool(np.allclose(mean(ef.C_elec), st.C_elec_bar, rtol=1e-9, atol=1e-15)), ""),
        ("stored C_carbon_bar = recomputed", bool(np.allclose(mean(ef.kg), st.C_carbon_bar, rtol=1e-9, atol=1e-15)), ""),
        ("r_tilde = customer value + profit - carbon $ (lambda = 1)",
         bool(np.allclose(tt["customer value"] + tt["profit"] - tt["carbon $"], st.r_tilde, rtol=1e-9, atol=1e-12)), ""),
        ("LP fill <= 1", bool(np.all(np.asarray(tab["consumption at the optimum = LP fill"]) <= 1 + 1e-9)), ""),
        ("tau >= 0", bool(np.all(np.asarray(st.tau) >= 0)), ""),
        ("grid covers every job time (H and stream)", bool(cov["H"]["covered"] and cov["stream"]["covered"]), ""),
        ("no NaN or inf in PhaseState arrays", bool(all(np.all(np.isfinite(np.asarray(a, float))) for a in fields)), ""),
    ]
    return check_table(rows)


# --------------------------------------------------------------------------- #
# 9. Theoretical benchmarks
# --------------------------------------------------------------------------- #
def benchmark_table(r: dict) -> pd.DataFrame:
    """Reward split of the type-level models at one f (theory_eval.evaluate_f result)."""
    rows = [("fluid OPT (hindsight)", r["fluid"]["split"]), ("fluid on feasible types", r["fluid_feasible"]["split"]),
            ("Phase 0 LP, expected", r["phase0_lp"]["expected"]), ("static Phase 0 policy, realised", r["static"]["split"]),
            ("no-learning model, realised", r["no_learning"]["split"])]
    return pd.DataFrame([{"model": n, **{k: s.get(k, np.nan) for k in ("customer value", "profit", "carbon", "reward",
                                                                        "social welfare")}} for n, s in rows])


def checks_week(w: dict):
    best, bound = w["bucketed_believed"], min(w["bucketed_dual_bound"], w["bucketed_lp_believed"])
    rows = [("fluid (inside the window) >= fluid on feasible types", w["fluid"] >= w["fluid_feasible"] - 1e-6,
             f"{w['fluid']:,.2f} >= {w['fluid_feasible']:,.2f}"),
            ("fluid on feasible types (inside the window) >= MILP bound", w["fluid_feasible"] >= bound - 1e-6,
             f"{w['fluid_feasible']:,.2f} >= {bound:,.2f}"),
            ("MILP bound >= MILP best", bound >= best - 1e-6, f"{bound:,.2f} >= {best:,.2f}"),
            ("the MILP's chosen set is feasible on the true durations", w["feasible_true"], ""),
            ("the MILP used its own 3,600 s budget", w["milp_runtime_s"] >= 0.9 * w["milp_time_limit"] or w["gap"] <= 1e-4,
             f"{w['milp_runtime_s']:.0f} s, gap {100 * w['gap']:.2f}%")]
    return check_table(rows)


# --------------------------------------------------------------------------- #
# 10. One theoretical DLENT-RT run
# --------------------------------------------------------------------------- #
def phase_table(ph: pd.DataFrame, tick_seconds: float) -> pd.DataFrame:
    h = tick_seconds / 3600.0
    out = pd.DataFrame({"phase": ph.m, "start (h)": (ph.tau_start - 1) * h, "end (h)": ph.tau_end * h,
                        "length (h)": ph.W * h, "reason": ph.reason, "arrivals": ph.n_arrivals, "admitted": ph.n_admitted,
                        "tau CPU": ph.tau_cpu, "tau RAM": ph.tau_ram, "rho": ph.rho, "sigma": ph.sigma,
                        "reward $": ph.R, "payments $": ph.payments, "IR violations": ph.n_ir_violations})
    if "reinit_grid_refit" in ph:
        out["next grid rebuilt"] = ph.reinit_grid_refit
        out["next rho"] = ph.next_rho
        out["next sigma"] = ph.next_sigma
        out["ReInit: arrivals used (p, v, sigma, A)"] = ph.reinit_n_arrivals
        out["ReInit: admitted used (d, energy, costs)"] = ph.reinit_n_observed
    return out


def checks_run(run: dict, cfg) -> pd.DataFrame:
    A, ph = run["arrivals"], run["phases"]
    adm = A.admit.to_numpy()
    kh, U, fits = A.k_hat.to_numpy(), A.U.to_numpy(), A.fits.to_numpy()
    nre = ph.iloc[:-1]
    # thinning after the decision: U drawn for every arrival from the common numbers; the thinned count is k_hat & ~U
    pU_k = U[kh].mean() if kh.any() else np.nan
    pU_n = U[~kh].mean() if (~kh).any() else np.nan
    adm_A = A[adm]
    viol = adm_A[~adm_A.ir.to_numpy()]
    off = adm_A[adm_A.below_grid | adm_A.above_grid]
    rows = [
        ("frozen oracle: every score uses the phase's r~ (no live price)",
         (A.oracle == "frozen").all() and np.array_equal(A.r_tilde.to_numpy(), A.r_tilde_phase.to_numpy()), ""),
        ("frozen oracle: k_hat = 1[r~ - tau v > 0]", np.array_equal(kh, (A.r_tilde - A.shadow_cost > 0).to_numpy()), ""),
        ("admit = k_hat and U and fits (thinning after the LP decision, then the gate)",
         np.array_equal(adm, kh & U & fits), f"{adm.sum():,} admitted"),
        ("thinning does not depend on the decision (U rate with and without k_hat)",
         np.isnan(pU_k) or np.isnan(pU_n) or abs(pU_k - pU_n) < 0.02, f"{pU_k:.3f} vs {pU_n:.3f}"),
        ("thinned count = k_hat and not U", int(ph.n_thinned.sum()) == int((kh & ~U).sum()), f"{int(ph.n_thinned.sum()):,}"),
        ("ReInit statistics: rates, bids, sigma, requests from ALL arrivals of the phase",
         np.array_equal(nre.reinit_n_arrivals.to_numpy(), nre.n_arrivals.to_numpy()), ""),
        ("ReInit statistics: duration, energy, costs from ADMITTED jobs only",
         np.array_equal(nre.reinit_n_observed.to_numpy(), nre.n_admitted.to_numpy()), ""),
        ("grid rebuilt at every ReInit with >= n_min arrivals",
         np.array_equal(nre.reinit_grid_refit.to_numpy(), (nre.n_arrivals >= cfg.phase_transition.n_min).to_numpy()), ""),
        ("payment = lower edge of the lowest admitted bin of (q, hw) at the phase's tau",
         np.allclose(adm_A.payment, adm_A.threshold_price, rtol=0, atol=0), ""),
        ("IR: payment <= bid for every admitted job", len(viol) == 0 and np.all(adm_A.payment <= adm_A.v),
         f"{len(viol):,} of {len(adm_A):,} violate"),
        ("IR also for admitted bids below the phase grid's v_min or above its v_max", np.all(off.payment <= off.v),
         f"{int(adm_A.below_grid.sum()):,} below, {int(adm_A.above_grid.sum()):,} above"),
        ("tests start d_max after the phase start; timeout at 2 x the previous phase length",
         np.all((ph.reason != "timeout") | (ph.W == 2 * ph.W_prev)), ""),
    ]
    return check_table(rows)


# --------------------------------------------------------------------------- #
# 11. D1-D6 checklist
# --------------------------------------------------------------------------- #
CHECKLIST = [
    ("D1", "V = v (no q factor); Pi = phi - C_elec; carbon = s_C K", "dlent_rt/reward.py", "profit = phi - C_elec"),
    ("D1", "R = l1 V + l2 Pi - l3 s_C K, linear, all lambda = 1", "dlent_rt/reward.py", "r_tilde = r_nocarbon - lam.lambda3"),
    ("D1", "s_C = D_q / K_q over H's types, once in Phase 0", "dlent_rt/reward.py", "D_q = 0.5 * (Q_cv + Q_abs_pi)"),
    ("D1", "SW metric only", "dlent_rt/reward.py", "sw=v - C_elec - s_C * K"),
    ("D1", "Chebyshev only in the offline Pareto code", "dlent_rt/pareto.py", "def scalarize_chebyshev("),
    ("D2", "cost_base = (U0 (1 + ln q) + g1 cores + g2 GB) d", "dlent_rt/bid.py", "return (u0 * (1.0 + np.log("),
    ("D2", "U0 = rate of the median H job", "dlent_rt/bid.py", "return float(np.median(cost_rate("),
    ("D2", "phi = cost_base phi_m(v / cost_base)", "dlent_rt/bid.py", "return c * np.asarray(self.phi_markup("),
    ("D2", "sigma refit in Phase 0", "dlent_rt/phase0.py", "myerson = bid_model(cfg, bids.v_rate, bids.cost_base)"),
    ("D2", "sigma refit at ReInit", "dlent_rt/models/core/reinit.py", "myerson = bid_model(cfg, v_arr, c_arr)"),
    ("D3", "H = first 7 days", "dlent_rt/data.py", "def _n_history("),
    ("D3", "C = f D_bar on H", "dlent_rt/data.py", "capacity = cfg.capacity.f * capacity_diag"),
    ("D3", "Q = 0.2 C fixed point at the actual f", "dlent_rt/data.py", "Q_new = lim.quota_share * C"),
    ("D3", "24 h cap and 1-minute floor", "dlent_rt/data.py", "np.minimum(np.maximum(d, cfg.time.duration_floor_hours)"),
    ("D3", "job-level enlargement option and check", "dlent_rt/benchmarks.py", "def job_level_space_check("),
    ("D4", "geometric grid between the smallest and largest bid, no floor", "dlent_rt/grid.py", "lo, hi = float(v.min()), float(v.max())"),
    ("D4", "grid rebuilt at every ReInit", "dlent_rt/models/core/reinit.py", "grid = build_valuation_grid(v_arr"),
    ("D4", "unseen types: dual rule with fallback estimates", "dlent_rt/theory_eval.py", "def unseen_type_estimates("),
    ("D5", "frozen oracle; live switch", "dlent_rt/models/theory.py", "if self.live:"),
    ("D5", "thinning after the LP decision", "dlent_rt/models/theory.py", "# 2. thinning, after the decision"),
    ("D5", "physical gate", "dlent_rt/models/policy.py", "fits = occ.fits(stream.A[i])"),
    ("D5", "no-learning model", "dlent_rt/models/policy.py", "class NoLearningPolicy("),
    ("D5", "xi = 0.001, gamma = 1 forced; Test 1 = (1 - xi) c", "dlent_rt/models/theory.py", "self.a_max_eff = self.ep.xi * self.c"),
    ("D5", "Tests 2-3: rewards in the thresholds' units (/ r_max)", "dlent_rt/models/core/monitor.py", "self.R[int(t)] = float(R) / self.r_unit"),
    ("D5", "tests start d_max after the phase start", "dlent_rt/models/core/monitor.py", "self.tau_bar = self.tau_prev + int(d_max)"),
    ("D5", "timeout at 2 x the previous phase length", "dlent_rt/models/core/monitor.py", "timeout = self.tau_prev + 2 * self.W_prev + 1"),
    ("D5", "ReInit: all arrivals vs admitted", "dlent_rt/models/core/reinit.py", "M, md = _means(tid_obs, n_types"),
    ("D6", "threshold payment = lowest admitted bin's lower edge", "dlent_rt/models/theory.py", "pay = thr if np.isfinite(thr)"),
    ("D6", "surplus, realised profit, IR", "dlent_rt/models/theory.py", "surplus, prof_pay = v_i - pay"),
    ("D6", "benchmarks: fluid OPT, fluid feasible, bucketed MILP", "dlent_rt/benchmarks.py", "def fluid_opt_feasible_types("),
    ("D6", "fluid on windows: volume inside the window only", "dlent_rt/theory_eval.py", "v_in = ref.A_type * d_in"),
]


def checklist(results: dict) -> pd.DataFrame:
    """results: decision -> 'PASS' / 'FAIL' / text from the notebook sections."""
    rows = [{"decision": d, "what": w, "where": where(f, pat), "notebook": results.get((d, w), "")}
            for d, w, f, pat in CHECKLIST]
    return pd.DataFrame(rows)


def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else float(x))

"""
Phase 0 summary (week 32): what Phase 0 produced on a dataset, for planning the recalibration. Read-only: it re-solves
the same LP-RS to get the admitted share x of every type, nothing else.

    from dlent_rt.diagnostics import phase0_summary
    s = phase0_summary(ds, state0, cfg)      # dict; s["flags"] lists what looks off

Rates are per hour (p_j is per tick). Reward split = p-weighted sums of the three terms of r~
(customer value l1 q v_bar, profit l2 (phi_bar - C_elec_bar), carbon -l3 s_C K_bar), over all populated types and
over the LP's admitted mass (p_j x_j).
"""
from __future__ import annotations

import numpy as np

from .lp_oracle import solve_lp_rs
from .phase0 import type_arrays


def phase0_summary(ds, st, cfg) -> dict:
    per_h = 3600.0 / cfg.time.tick_seconds
    lam = cfg.objective
    p = np.asarray(st.p, float)
    pop = p > 0
    _, prio, _ = type_arrays(st.type_space, st.grid)
    lp = solve_lp_rs(p=p, r=st.r_tilde, v=st.v, c=ds.capacity, presolve=cfg.lp.presolve)
    x = np.zeros(st.n_types)
    x[lp.support_idx] = lp.x
    terms = {"customer value": lam.lambda1 * prio * st.v_bar,
             "profit": lam.lambda2 * (st.phi_bar - st.C_elec_bar),
             "carbon": -lam.lambda3 * st.carbon_price * st.C_carbon_bar}
    split_all = {k: float((p * v).sum() * per_h) for k, v in terms.items()}
    split_adm = {k: float((p * x * v).sum() * per_h) for k, v in terms.items()}
    lam_h = p[pop] * per_h
    fill = (p * x) @ st.v / ds.capacity
    s = dict(
        n_types=int(st.n_types), n_types_populated=int(pop.sum()), k_star=int(cfg.grid.k_star),
        n_hw_profiles=int(st.type_space.n_hw), n_priorities=int(len(st.type_space.priorities)),
        grid_space=cfg.grid.space, grid_bid_range=[float(np.min(st.grid.bnd_v)), float(np.max(st.grid.bnd_v))],
        lambda_total_per_h=float(lam_h.sum()),
        lambda_per_type_per_h=dict(median=float(np.median(lam_h)), q90=float(np.quantile(lam_h, 0.9)),
                                   max=float(lam_h.max())),
        tau=np.asarray(st.tau, float).tolist(), lam_star_per_h=float(st.lam_star * per_h),
        n_types_admitted=int((x > 1e-9).sum()), n_types_fully_admitted=int((x > 1 - 1e-9).sum()),
        share_positive_reward_types=float((st.r_tilde[pop] > 0).mean()),
        share_positive_reward_mass=float(p[pop][st.r_tilde[pop] > 0].sum() / p[pop].sum()),
        reward_split_per_h_all=split_all, reward_split_per_h_admitted=split_adm,
        capacity=np.asarray(ds.capacity, float).tolist(), lp_fill=np.asarray(fill, float).tolist(),
        demand_load=np.asarray(p @ st.v / ds.capacity, float).tolist(),
        s_C=float(st.carbon_price), gamma_data=float(st.gamma), xi_data=float(st.xi), eps_D_data=float(st.eps_D),
        n_H=int(ds.H.n), d_max_h=float(ds.d_max_steps / per_h),
    )
    flags = []
    if np.all(np.asarray(st.tau) <= 1e-12):
        flags.append("tau = 0 on both resources: the capacity does not bind in the LP")
    if s["n_types_populated"] < 50:
        flags.append(f"only {s['n_types_populated']} populated types")
    if s["share_positive_reward_types"] < 0.5:
        flags.append(f"only {100 * s['share_positive_reward_types']:.0f}% of the populated types have r~ > 0")
    if np.any(np.asarray(s["lp_fill"]) < 0.5):
        flags.append("the LP solution fills less than half of the cluster on some resource")
    if ds.xi > 1.0:
        flags.append(f"data xi = {ds.xi:.2f} > 1: some admitted job is larger than the cluster")
    s["flags"] = flags
    return s


# --------------------------------------------------------------------------- #
# Phase 0 diagnostics for the week-32 environment (notebook 01). Read-only: nothing here changes Phase 0.
# --------------------------------------------------------------------------- #
import json
from dataclasses import replace

import pandas as pd

from .bid import MyersonModel
from .config import resolve_path
from .data import _occupancy_stats, _OccupancyBasis

from .phase0 import _compute_job_energy, _compute_job_power, _lookup_grid, assign_type_ids, run_phase0
from .reward import carbon_kg, electricity_cost, type_reward, weighted_quantile

QS = (0.10, 0.50, 0.90, 0.99)


def settings(cfg) -> dict:
    lim, b, em, co = cfg.limits, cfg.bid, cfg.energy_model, cfg.constants
    return {"data file": cfg.data.batch_csv, "grid file": cfg.data.grid_csv,
            "H": f"{cfg.data.h_days} days" if cfg.data.h_days else f"first {cfg.data.n0} jobs",
            "duration cap d_cap (h)": lim.d_cap_hours, "quota rule": lim.job_quota_rule,
            "quota share / quota at f": f"{lim.quota_share} / {lim.quota_from_f}", "R_max": lim.r_max_machines,
            "capacity mode": cfg.capacity.mode, "f": cfg.capacity.f, "bid mode": b.mode, "sigma": b.sigma,
            "gamma1 ($/core-h)": b.gamma1, "gamma2 ($/GB-h)": b.gamma2,
            "carbon map": f"quantile_linear, q = {cfg.normalization.quantile}", "PUE": em.pue,
            "machine (declared)": f"{em.machine_cores:g} cores / {em.machine_ram_gb:g} GB",
            "p_core (W) / p_gb (W)": f"{em.p_core} / {em.p_gb}", "xi forced": co.xi_override,
            "gamma forced": co.gamma_override, "thinning off (ablation A1a)": cfg.ablation.thinning_off,
            "unseen types (static policies)": cfg.policy.unseen_types, "k* (bid bins)": cfg.grid.k_star,
            "hardware profiles": cfg.hardware.n_profiles}


def dataset_summary(ds, cfg) -> dict:
    col = cfg.data.columns
    df = pd.read_csv(resolve_path(cfg.data.batch_csv), usecols=[col.collection_id, col.requested_cpu,
                                                                  col.duration_hours, col.cpu_hours_real])
    cpu, span = df[col.requested_cpu].to_numpy(float), df[col.duration_hours].to_numpy(float)
    eff = np.divide(df[col.cpu_hours_real].to_numpy(float), cpu, out=span.copy(), where=cpu > 0)
    kept = set(np.concatenate([ds.H.collection_id, ds.stream.collection_id]).tolist())
    m = df[col.collection_id].isin(kept).to_numpy()
    cap = cfg.limits.d_cap_hours
    return {"jobs in the file": len(df), "H jobs": ds.H.n, "stream jobs": ds.stream.n,
            "T (stream horizon, h)": round(ds.T * cfg.time.tick_seconds / 3600, 2),
            "H from (UTC)": str(pd.Timestamp(ds.H.datetime[0])), "H to (UTC)": str(pd.Timestamp(ds.H.datetime[-1])),
            "stream starts (UTC)": str(pd.Timestamp(ds.stream.datetime[0])),
            "stream ends (UTC)": str(pd.Timestamp(ds.stream.datetime[-1])),
            "jobs removed by the quota": ds.limit_report.n_quota_rejected,
            "kept jobs cut by the duration cap": int((eff[m] > cap).sum()) if cap else 0,
            "d_max (h)": ds.d_max_steps * cfg.time.tick_seconds / 3600}


def cluster_derivation(ds, cfg) -> dict:
    """C = f x D_bar step by step (mean_demand mode), the quota fixed point and Little's law on H."""
    H, tick = ds.H, cfg.time.tick_seconds
    span_h = ds.H_span_steps * tick / 3600
    work = (H.A * H.duration_hours[:, None]).sum(0)
    D_bar = work / span_h
    basis = _OccupancyBasis(arrival_step=H.arrival_step, duration_steps=H.duration_steps, A=H.A)
    st = _occupancy_stats(basis)
    keep = st["times"][:-1] <= H.arrival_step[-1]
    occ = (st["levels"][keep] * st["weights"][keep, None]).sum(0) / st["weights"][keep].sum()
    its = ds.limit_report.quota_iterations or []
    return {"total work in H (machine-h), CPU / RAM": work.tolist(), "|H| (h)": span_h,
            "D_bar = work / |H| (machines)": D_bar.tolist(), "f": cfg.capacity.f,
            "C = f x D_bar (machines)": np.asarray(ds.capacity).tolist(),
            "C in cores / GB (declared)": [float(ds.capacity[0] * cfg.energy_model.machine_cores),
                                           float(ds.capacity[1] * cfg.energy_model.machine_ram_gb)],
            "Q (machines)": list(ds.limit_report.quota), "Q / C": (np.array(ds.limit_report.quota) / ds.capacity).tolist(),
            "quota fixed point at f": cfg.limits.quota_from_f or cfg.capacity.f, "fixed-point steps": len(its),
            "Little: time-average occupancy of H, no capacity": occ.tolist(),
            "Little gap vs D_bar": (occ / D_bar - 1).tolist()}


def arrival_profile(ds) -> pd.DataFrame:
    """Arrivals per hour by hour of day (UTC) and by day of week, H vs stream (rate per hour of exposure)."""
    rows = []
    for name, J in (("H", ds.H), ("stream", ds.stream)):
        t = pd.to_datetime(J.datetime)
        hours = pd.date_range(t.min().floor("h"), t.max().ceil("h"), freq="h", inclusive="left")
        for key, ser, expo in (("hour of day", t.hour, hours.hour), ("day of week", t.dayofweek, hours.dayofweek)):
            cnt = pd.Series(ser).value_counts()
            ex = pd.Series(expo).value_counts()
            for k in sorted(ex.index):
                rows.append({"set": name, "by": key, "value": int(k), "jobs per hour": cnt.get(k, 0) / ex[k]})
    return pd.DataFrame(rows)


def stats_table(cols: dict) -> pd.DataFrame:
    rows = []
    for name, x in cols.items():
        x = np.asarray(x, float)
        rows.append({"quantity": name, "count": x.size, "mean": x.mean(),
                     **{f"p{int(q * 100)}": float(np.quantile(x, q)) for q in QS}, "max": x.max()})
    return pd.DataFrame(rows)


def h_jobs(ds) -> dict:
    H = ds.H
    return {"CPU request (machines)": H.A[:, 0], "RAM request (machines)": H.A[:, 1],
            "capped duration (h)": H.duration_hours, "work r*d (CPU machine-h)": H.A[:, 0] * H.duration_hours}


def bid_frame(ds, st, cfg) -> pd.DataFrame:
    """Per H job: cost base, bid, markup, phi (Phase 0's bids, the same random numbers), work, priority."""
    H, b = ds.H, st.bids
    phi = np.asarray(st.myerson.phi_job(b.v_rate, b.cost_base))
    rd = H.A[:, 0] * H.duration_hours
    return pd.DataFrame({"priority": H.priority, "cost_base": b.cost_base, "bid": b.v_rate,
                         "markup": b.v_rate / b.cost_base, "phi": phi, "rd": rd,
                         "bid_per_machine_h": np.divide(b.v_rate, rd, out=np.full(H.n, np.nan), where=rd > 0)})


def bid_tier_table(bf: pd.DataFrame) -> pd.DataFrame:
    g = bf.groupby("priority")
    return pd.DataFrame({"jobs": g.size(), "mean bid $": g["bid"].mean(), "mean bid per machine-h $": g["bid_per_machine_h"].mean(),
                         "phi < 0 %": g["phi"].apply(lambda s: 100 * (s < 0).mean()),
                         "mean phi / bid": g.apply(lambda d: (d["phi"] / d["bid"]).mean())}).reset_index()


def sigma_sensitivity(ds, cfg, sigmas=(0.1, 0.3, 0.5)) -> pd.DataFrame:
    from .phase0 import _synthesize_bids
    rows = []
    for s in sigmas:
        c = replace(cfg, bid=replace(cfg.bid, sigma=s)).validate()
        b = _synthesize_bids(ds.H, c, seed=cfg.run.seed)
        phi = np.asarray(MyersonModel.known_markup(s).phi_job(b.v_rate, b.cost_base))
        rows.append({"sigma": s, "bid mean $": b.v_rate.mean(), "phi < 0 %": 100 * (phi < 0).mean(),
                     "phi mean $": phi.mean()})
    return pd.DataFrame(rows)


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
                         "C_elec": electricity_cost(E, pr), "kg": carbon_kg(E, ci),
                         "hour": pd.to_datetime(J.datetime).hour})


def grid_frame(cfg) -> pd.DataFrame:
    g = pd.read_csv(resolve_path(cfg.data.grid_csv), parse_dates=["timestamp_utc"])
    if "is_filled" not in g:
        g["is_filled"] = False
    return g


def grid_coverage(ds, cfg) -> dict:
    """Is every job time (arrival to end, H and stream) inside the hours of the grid file?"""
    g = grid_frame(cfg)
    lo, hi = g.timestamp_utc.min(), g.timestamp_utc.max() + pd.Timedelta(hours=1)
    out = {}
    for name, J in (("H", ds.H), ("stream", ds.stream)):
        a = pd.to_datetime(J.datetime)
        e = a + pd.to_timedelta(J.duration_hours, unit="h")
        out[name] = dict(first=str(a.min()), last_end=str(e.max()), covered=bool(a.min() >= lo and e.max() <= hi))
    out["grid"] = dict(first=str(lo), last_end=str(hi), filled_hours=int(g.is_filled.sum()))
    return out


def _gamma_parts(ds, st, cfg) -> dict:
    """The terms of the data gamma = max(r_max, v_rate_max, d_max, c_max, r_max / v_min) (phase0._system_constants)."""
    from .phase0 import type_arrays
    p = np.asarray(st.p)
    q_r = cfg.constants.r_max_quantile
    abs_r = np.abs(st.r_tilde[p > 0])
    r_max = float(abs_r.max()) if q_r >= 1.0 else weighted_quantile(abs_r, p[p > 0], q_r)
    _, _, A_prof = type_arrays(st.type_space, st.grid)
    rel = (p > 0) & np.all(A_prof >= cfg.hardware.hw_floor, axis=1) & (st.v.max(1) > 0)
    v_min = float(np.min(st.v[rel][st.v[rel] > 0]))
    parts = {"r_max": r_max, "v_rate_max": float(st.bids.v_rate.max()), "d_max (ticks)": float(ds.d_max_steps),
             "c_max": float(np.max(ds.capacity)), "r_max / v_min": r_max / v_min}
    parts["dominant"] = max((k for k in parts), key=lambda k: parts[k])
    return parts


def phase0_table(ds, st, cfg) -> dict:
    """Every PhaseState quantity the online model starts from (one dict; also saved as JSON)."""
    from .phase0 import type_arrays
    per_h = 3600.0 / cfg.time.tick_seconds
    p = np.asarray(st.p)
    pop = p > 0
    lp = solve_lp_rs(p=p, r=st.r_tilde, v=st.v, c=ds.capacity, presolve=cfg.lp.presolve)
    x = np.zeros(st.n_types)
    x[lp.support_idx] = lp.x
    e = np.asarray(st.grid.bnd_v, float)
    ni = st.norm_info or {}
    return {
        "n_types": int(st.n_types), "populated types": int(pop.sum()), "jobs assigned": int(st.n_assigned),
        "IR rejected": int(st.n_ir_rejected), "sum p (per h)": float(p.sum() * per_h),
        "true arrival rate of H (per h)": float(ds.H.n / ds.H_span_steps * per_h),
        "grid": {"k*": int(cfg.grid.k_star), "space": cfg.grid.space, "spacing": "geometric",
                 "ratio between edges": float(e[1] / e[0]), "first edge $": float(e[0]), "last edge $": float(e[-1])},
        "s_C ($/kg)": float(st.carbon_price),
        "s_C derivation": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in ni.items()},
        "r_tilde min / median / max (populated)": [float(st.r_tilde[pop].min()), float(np.median(st.r_tilde[pop])),
                                                   float(st.r_tilde[pop].max())],
        "positive types %": float(100 * (st.r_tilde[pop] > 0).mean()),
        "positive arrival mass %": float(100 * p[pop & (st.r_tilde > 0)].sum() / p.sum()),
        "gamma (data)": float(st.gamma), "gamma parts": _gamma_parts(ds, st, cfg),
        "gamma forced": cfg.constants.gamma_override, "xi (data)": float(st.xi), "xi forced": cfg.constants.xi_override,
        "eps_D (data xi)": float(st.eps_D), "d_max (h)": float(ds.d_max_steps / per_h),
        "C (machines)": np.asarray(ds.capacity).tolist(), "Q (machines)": list(ds.limit_report.quota),
        "tau": np.asarray(st.tau).tolist(), "LP optimum ($ per h)": float(st.lam_star * per_h),
        "consumption if all admitted (p.v / C)": (p @ st.v / ds.capacity).tolist(),
        "consumption at the optimum = LP fill": ((p * x) @ st.v / ds.capacity).tolist(),
        "types fully / partly / rejected": [int((x[pop] > 1 - 1e-9).sum()), int(((x[pop] > 1e-9) & (x[pop] <= 1 - 1e-9)).sum()),
                                            int((x[pop] <= 1e-9).sum())],
        "_x": x,
    }


def save_table(tab: dict, path) -> None:
    with open(path, "w") as fh:
        json.dump({k: v for k, v in tab.items() if not k.startswith("_")}, fh, indent=2, default=float)


def type_terms(st, cfg) -> pd.DataFrame:
    """Per type: the three terms of r~ (customer value, profit, -carbon $), p, r~ (all lambda as in the config)."""
    from .phase0 import type_arrays
    lam = cfg.objective
    _, prio, A_prof = type_arrays(st.type_space, st.grid)
    ts = st.type_space
    ids = np.arange(st.n_types)
    return pd.DataFrame({"type": ids, "p": st.p, "priority": prio, "hw": ids % ts.n_hw,
                         "bin": ids // (len(ts.priorities) * ts.n_hw) + 1,
                         "customer value": lam.lambda1 * prio * st.v_bar,
                         "profit": lam.lambda2 * (st.phi_bar - st.C_elec_bar),
                         "carbon $": lam.lambda3 * st.carbon_price * st.C_carbon_bar, "r_tilde": st.r_tilde,
                         "rd": st.A_type[:, 0] * st.d_bar / 3600.0})


def job_terms_H(ds, st, cfg) -> pd.DataFrame:
    """Per H job: the three terms of R_job with Phase 0's bids, model and s_C."""
    from .reward import job_terms
    ef = energy_frame(ds, cfg)
    b = st.bids
    t = job_terms(ds.H.priority, b.v_rate, b.cost_base, ef.C_elec.to_numpy(), ef.kg.to_numpy(), st.myerson,
                  st.carbon_price, cfg.objective)
    return pd.DataFrame({"priority": ds.H.priority, "hour": ef.hour, "rd": ds.H.A[:, 0] * ds.H.duration_hours,
                         "customer value": t.cv, "profit": t.profit, "carbon $": t.carbon_usd, "reward": t.R})


def term_shares(tt: pd.DataFrame, x=None) -> dict:
    """Share of sum |term| (p-weighted) from each term, over populated types or over admitted mass p x."""
    w = tt.p.to_numpy() * (1.0 if x is None else np.asarray(x))
    a = {k: float((w * np.abs(tt[k])).sum()) for k in ("customer value", "profit", "carbon $")}
    s = sum(a.values())
    return {k: v / s for k, v in a.items()}


def decile_table(jt: pd.DataFrame) -> pd.DataFrame:
    rd = jt.rd.to_numpy()
    dec = np.clip(np.searchsorted(np.quantile(rd, np.linspace(0, 1, 11)), rd, side="right") - 1, 0, 9)
    rows = []
    for k in range(10):
        m = dec == k
        rows.append({"decile": k + 1, **{f"mean {c}": jt[c][m].mean() for c in ("customer value", "profit", "carbon $", "reward")},
                     "reward per machine-h": jt.reward[m].sum() / rd[m].sum(), "% positive": 100 * (jt.reward[m] > 0).mean(),
                     "share of CPU-hours %": 100 * rd[m].sum() / rd.sum()})
    return pd.DataFrame(rows)


def trace_types(ds, st, cfg) -> pd.DataFrame:
    """Raw -> reward for 5 types, recomputed from the H jobs of each type, checked against the stored means."""
    from .phase0 import type_arrays
    tt = type_terms(st, cfg)
    ef = energy_frame(ds, cfg)
    b = st.bids
    phi = np.asarray(st.myerson.phi_job(b.v_rate, b.cost_base))
    k = st.grid.assign(b.v_rate, st.myerson)
    tid = assign_type_ids(st.type_space, np.asarray(k), ds.H.priority, ds.H.hw_idx)
    pop = tt[tt.p > 0]
    picks = {"highest p": pop.p.idxmax(), "highest carbon $": pop["carbon $"].idxmax(),
             "median carbon $": (pop["carbon $"] - pop["carbon $"].median()).abs().idxmin(),
             "highest r_tilde": pop.r_tilde.idxmax(), "most negative r_tilde": pop.r_tilde.idxmin()}
    rows = []
    for name, j in picks.items():
        m = tid == j
        q = float(tt.priority[j])
        cv, prof = q * b.v_rate[m].mean(), phi[m].mean() - ef.C_elec[m].mean()
        carb = st.carbon_price * ef.kg[m].mean()
        r = cfg.objective.lambda1 * cv + cfg.objective.lambda2 * prof - cfg.objective.lambda3 * carb
        rows.append({"type": name, "id": int(j), "jobs": int(m.sum()), "CI kg/kWh": ef.ci[m].mean(),
                     "energy kWh": ef.energy_kWh[m].mean(), "kg": ef.kg[m].mean(), "carbon $": carb,
                     "price $/kWh": ef.price[m].mean(), "electricity $": ef.C_elec[m].mean(), "bid $": b.v_rate[m].mean(),
                     "phi $": phi[m].mean(), "customer value $": cv, "profit $": prof, "r_tilde (recomputed)": r,
                     "r_tilde (stored)": float(st.r_tilde[j]), "match": bool(np.isclose(r, st.r_tilde[j], rtol=1e-9, atol=1e-12))})
    return pd.DataFrame(rows)


def carbon_weight_sensitivity(ds, st, cfg, factors=(0.5, 1.0, 2.0)) -> pd.DataFrame:
    """% positive types and LP fill when the carbon weight lambda3 is scaled (s_C frozen; no config change)."""
    from .phase0 import type_arrays
    _, prio, _ = type_arrays(st.type_space, st.grid)
    pop = st.p > 0
    rows = []
    for k in factors:
        c = replace(cfg, objective=replace(cfg.objective, lambda3=cfg.objective.lambda3 * k))
        r, _, _ = type_reward(prio, st.v_bar, st.phi_bar, st.C_elec_bar, st.C_carbon_bar, st.p, c, s_C=st.carbon_price)
        lp = solve_lp_rs(p=st.p, r=r, v=st.v, c=ds.capacity, presolve=cfg.lp.presolve)
        x = np.zeros(st.n_types)
        x[lp.support_idx] = lp.x
        rows.append({"carbon weight x": k, "positive types %": 100 * (r[pop] > 0).mean(),
                     "LP fill CPU %": 100 * float((st.p * x) @ st.v[:, 0] / ds.capacity[0]),
                     "LP fill RAM %": 100 * float((st.p * x) @ st.v[:, 1] / ds.capacity[1]),
                     "tau": np.round(lp.tau, 6).tolist()})
    return pd.DataFrame(rows)


def seed_stability(ds, cfg, seeds=range(20)) -> pd.DataFrame:
    rows = []
    for s in seeds:
        c = replace(cfg, run=replace(cfg.run, seed=int(s)))
        st = run_phase0(ds, c)
        lp = solve_lp_rs(p=st.p, r=st.r_tilde, v=st.v, c=ds.capacity, presolve=cfg.lp.presolve)
        x = np.zeros(st.n_types)
        x[lp.support_idx] = lp.x
        sh = term_shares(type_terms(st, c))
        pop = st.p > 0
        rows.append({"seed": int(s), "s_C": st.carbon_price, "positive types %": 100 * (st.r_tilde[pop] > 0).mean(),
                     "tau CPU": lp.tau[0], "tau RAM": lp.tau[1],
                     "LP fill CPU %": 100 * float((st.p * x) @ st.v[:, 0] / ds.capacity[0]),
                     "LP fill RAM %": 100 * float((st.p * x) @ st.v[:, 1] / ds.capacity[1]),
                     "share customer value": sh["customer value"], "share profit": sh["profit"],
                     "share carbon": sh["carbon $"]})
    return pd.DataFrame(rows)


def checks(ds, st, cfg, tab) -> pd.DataFrame:
    """PASS / FAIL table (section 7 of notebook 01)."""
    from .phase0 import _per_type_stats  # noqa: F401  (the stored means come from it)
    per_h = 3600.0 / cfg.time.tick_seconds
    ef = energy_frame(ds, cfg)
    b = st.bids
    k = st.grid.assign(b.v_rate, st.myerson)
    tid = assign_type_ids(st.type_space, np.asarray(k), ds.H.priority, ds.H.hw_idx)
    ok = tid >= 0
    N = np.bincount(tid[ok], minlength=st.n_types)
    mean = lambda x: np.divide(np.bincount(tid[ok], weights=np.asarray(x)[ok], minlength=st.n_types), N,
                               out=np.zeros(st.n_types), where=N > 0)
    tt = type_terms(st, cfg)
    pc = power_check(ds, cfg)
    cov = grid_coverage(ds, cfg)
    fields = [getattr(st, f) for f in ("p", "r_tilde", "v", "d_bar", "v_bar", "phi_bar", "C_elec_bar", "C_carbon_bar",
                                       "energy_bar", "tau", "A_type")]
    rows = [
        ("sum p = true arrival rate of H", np.isclose(tab["sum p (per h)"], ds.H.n / ds.H_span_steps * per_h, rtol=1e-9)
         if st.n_ir_rejected == 0 else np.isclose(tab["sum p (per h)"], st.n_assigned / ds.H_span_steps * per_h, rtol=1e-9)),
        ("manual power = code power (incl. PUE)", bool(pc["match"].all())),
        ("stored C_elec_bar = recomputed", bool(np.allclose(mean(ef.C_elec), st.C_elec_bar, rtol=1e-9, atol=1e-15))),
        ("stored C_carbon_bar = recomputed", bool(np.allclose(mean(ef.kg), st.C_carbon_bar, rtol=1e-9, atol=1e-15))),
        ("r_tilde = customer value + profit - carbon $ (lambda = 1)",
         bool(np.allclose(tt["customer value"] + tt["profit"] - tt["carbon $"], st.r_tilde, rtol=1e-9, atol=1e-12))),
        ("LP fill <= 1", bool(np.all(np.asarray(tab["consumption at the optimum = LP fill"]) <= 1 + 1e-9))),
        ("tau >= 0", bool(np.all(np.asarray(st.tau) >= 0))),
        ("grid covers every job time (H and stream)", bool(cov["H"]["covered"] and cov["stream"]["covered"])),
        ("no NaN or inf in PhaseState arrays", bool(all(np.all(np.isfinite(np.asarray(a, float))) for a in fields))),
    ]
    return pd.DataFrame([(n, "PASS" if v else "FAIL") for n, v in rows], columns=["check", "result"])

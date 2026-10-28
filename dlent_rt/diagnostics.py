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

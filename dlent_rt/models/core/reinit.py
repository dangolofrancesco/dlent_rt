"""
EndPhaseAndReinitialize (write-up Alg. 3) and the hindsight benchmark LP.

Both build a PhaseState from a window of stream jobs with the SAME reward
construction as Phase 0 (phase0.scalarized_reward), with three week-28
decisions:
  1. phi from the mechanism's bid model, MyersonModel.phi_job(v, cost_base):
     bid.known_distribution=true (default) -> the known markup distribution of
     the previous state, never refit (the grid edges are still refit);
     false (ablation) -> F_hat^(m+1), lognormal MLE on the phase's bids.
  2. Thin types (count < n_type_min, INCLUDING types unseen in the phase):
     conditional means (v_bar, phi_bar, d_bar, costs -> r~_j, v_ij) fall back
     to the previous phase's value, which itself falls back to Phase 0
     (carry-forward chain). Without this, an arrival whose type did not occur
     in the previous phase would have r~ = 0 and never be admitted. p_j stays
     the empirical N_j / W (p_j = 0 if the type did not arrive), so the LP
     support is unchanged; only the oracle's per-type score uses the carried
     stats.
  3. The common_currency scale s is frozen at Phase 0's value so r~ lives on
     one scale across phases (lambda_hat comparable across phases). Likewise
     the quantile_linear carbon price s_C (prev.carbon_price) is never
     recomputed on the stream.

Theory model: durations are observed at admission only (Version P), so
d_bar / energy / cost statistics use ADMITTED jobs of the phase ("obs"),
while p, v_bar, phi_bar use ALL arrivals of the phase.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ...bid import BidResult, fit_myerson, compute_phi_tilde
from ...grid import build_valuation_grid
from ...lp_oracle import solve_lp_rs
from ...phase0 import PhaseState, assign_type_ids, type_arrays, scalarized_reward
from .stream import StreamData


@dataclass
class ReinitDiag:
    n_arrivals: int
    n_observed: int
    W: int
    grid_refit: bool
    n_types_seen: int
    n_value_fallback: int      # types with N_j < n_type_min (incl. 0) using carried v/phi stats
    n_dur_fallback: int        # types with M_j < n_type_min (incl. 0) using carried duration/cost stats
    n_last_resort: int         # types with N_j > 0 and no duration info anywhere -> hw-profile mean
    lam_hat: float
    tau: np.ndarray
    lp_status: str


def _means(tid: np.ndarray, n_types: int, cols: dict) -> tuple[np.ndarray, dict]:
    N = np.bincount(tid, minlength=n_types)
    out = {}
    for k, x in cols.items():
        s = np.bincount(tid, weights=np.asarray(x, float), minlength=n_types)
        out[k] = np.divide(s, N, out=np.zeros(n_types), where=N > 0)
    return N, out


def build_window_state(
    stream: StreamData, idx_arr: np.ndarray, idx_obs: np.ndarray, W: int,
    prev: PhaseState, ds, cfg, common_scale: float, fallback: bool = True,
) -> tuple[PhaseState, ReinitDiag]:
    ts = prev.type_space
    n_types = ts.n_types
    idx_arr = np.asarray(idx_arr, np.int64)
    idx_obs = np.asarray(idx_obs, np.int64)
    v_arr = stream.v[idx_arr]

    # grid / F_hat refit (write-up Alg. 3 lines 2-6; reuse if too few arrivals)
    refit = idx_arr.size >= cfg.phase_transition.n_min and idx_arr.size >= 2
    if refit:
        myerson = prev.myerson if prev.myerson.known else fit_myerson(v_arr)
        grid = build_valuation_grid(v_arr, myerson, cfg.grid.k_star, cfg.grid.space,
                                    cfg.grid.spacing, cfg.grid.phi_floor)
    else:
        myerson, grid = prev.myerson, prev.grid

    def types_of(idx):
        k = grid.assign(stream.v[idx], myerson) if idx.size else np.zeros(0, np.int64)
        return assign_type_ids(ts, k, stream.priority[idx], stream.hw_idx[idx])

    tid_arr = types_of(idx_arr)
    ok = tid_arr >= 0
    phi_arr = (np.asarray(myerson.phi_job(v_arr, stream.cost_base[idx_arr]))
               if idx_arr.size else np.zeros(0))
    N, mv = _means(tid_arr[ok], n_types, dict(
        v_bar=v_arr[ok], phi_bar=phi_arr[ok]))

    tid_obs = types_of(idx_obs)
    oko = tid_obs >= 0
    o = idx_obs[oko]
    M, md = _means(tid_obs[oko], n_types, dict(
        d_bar=stream.D[o], energy_bar=stream.energy_kwh[o], power_bar=stream.power_kw[o],
        C_elec_bar=stream.C_elec[o], C_carbon_bar=stream.C_carbon[o]))

    nmin = cfg.phase_transition.n_type_min
    n_vfb = n_dfb = n_lr = 0
    if fallback:
        thin_v = (N < nmin) & (prev.v_bar > 0)
        for k in ("v_bar", "phi_bar"):
            mv[k] = np.where(thin_v, getattr(prev, k), mv[k])
        thin_d = (M < nmin) & (prev.d_bar > 0)
        for k in ("d_bar", "energy_bar", "power_bar", "C_elec_bar", "C_carbon_bar"):
            md[k] = np.where(thin_d, getattr(prev, k), md[k])
        n_vfb, n_dfb = int(thin_v.sum()), int(thin_d.sum())
        # last resort: arrived, never observed here or before -> mean over observed
        # jobs with the same hardware profile (else over all observed jobs)
        hole = (N > 0) & (md["d_bar"] <= 0)
        if np.any(hole):
            n_lr = int(hole.sum())
            hw_of_type = np.arange(n_types) % ts.n_hw
            pool = o if o.size else idx_arr
            for k, src in (("d_bar", stream.D), ("energy_bar", stream.energy_kwh),
                           ("power_bar", stream.power_kw), ("C_elec_bar", stream.C_elec),
                           ("C_carbon_bar", stream.C_carbon)):
                hw_sum = np.bincount(stream.hw_idx[pool], weights=src[pool], minlength=ts.n_hw)
                hw_cnt = np.bincount(stream.hw_idx[pool], minlength=ts.n_hw)
                glob = src[pool].mean() if pool.size else 0.0
                hw_mean = np.divide(hw_sum, hw_cnt, out=np.full(ts.n_hw, glob), where=hw_cnt > 0)
                md[k] = np.where(hole, hw_mean[hw_of_type], md[k])

    p = N / max(W, 1)
    L, prio, A_type = type_arrays(ts, grid)
    phi_tilde = compute_phi_tilde(L, myerson)
    v = A_type * md["d_bar"][:, None]
    r_tilde, r_nocarbon, norm_ref, _ = scalarized_reward(
        prio, mv["v_bar"], mv["phi_bar"], md["C_elec_bar"], md["C_carbon_bar"],
        p, v, ds, cfg, common_scale=common_scale, carbon_price=prev.carbon_price,
    )
    lp = solve_lp_rs(p=p, r=r_tilde, v=v, c=ds.capacity, presolve=cfg.lp.presolve)
    tau = lp.tau if lp.tau.size == ds.n_resources else np.zeros(ds.n_resources)

    bids = BidResult(v_rate=v_arr, phi_rate=phi_arr,
                     bid_low=np.zeros_like(v_arr), bid_high=stream.bid_high[idx_arr],
                     cost_base=stream.cost_base[idx_arr])
    st = replace(
        prev, myerson=myerson, grid=grid, bids=bids, p=p, d_bar=md["d_bar"],
        v_bar=mv["v_bar"], phi_bar=mv["phi_bar"], energy_bar=md["energy_bar"],
        power_bar=md["power_bar"], r_tilde=r_tilde, r_nocarbon=r_nocarbon,
        C_elec_bar=md["C_elec_bar"], C_carbon_bar=md["C_carbon_bar"], v=v, L=L,
        phi_tilde=phi_tilde, tau=tau, lam_star=lp.optimum, norm_ref=norm_ref,
        n_assigned=int(ok.sum()), n_ir_rejected=int((~ok).sum()),
    )
    diag = ReinitDiag(
        n_arrivals=int(idx_arr.size), n_observed=int(idx_obs.size), W=int(W),
        grid_refit=bool(refit), n_types_seen=int((N > 0).sum()),
        n_value_fallback=n_vfb, n_dur_fallback=n_dfb, n_last_resort=n_lr,
        lam_hat=float(lp.optimum), tau=tau, lp_status=lp.status,
    )
    return st, diag


def hindsight_lambda(stream, idx, W, prev, ds, cfg, common_scale) -> tuple[float, ReinitDiag]:
    """
    Hindsight benchmark on a window: full information (every arrival's true D
    observed, no fallback, F_hat refit on the window). Returns lam_hat per tick.
    """
    cfg_h = replace(cfg, phase_transition=replace(cfg.phase_transition, n_min=2))
    st, dg = build_window_state(stream, idx, idx, W, prev, ds, cfg_h,
                                common_scale=common_scale, fallback=False)
    return float(st.lam_star), dg

"""
EndPhaseAndReinitialize (write-up Alg. 3) and the hindsight benchmark LP.

Both build a PhaseState from a window of stream jobs with the SAME type terms as Phase 0 (phase0.type_values,
reward.type_reward). The bid grid and the types are built ONCE in Phase 0 and never change (D4, week 33): ReInit
updates only the statistics.
  1. p_j = N_j / W per type (bid bin, column), the empirical arrival rate of the window (0 if no arrival).
  2. Per COLUMN (priority, hardware profile), never per bin (week 33): the hourly cost base c_(q,h) is the mean over
     the window's ARRIVALS (it is known at arrival); the duration, energy, costs and the request of the volume
     (d_bar, energy_bar, power_bar, C_elec_bar, C_carbon_bar, A_type) are the means over the ADMITTED jobs. None
     depends on the bid.
  3. A column with no arrival (for c) or no admitted job (for the rest) keeps its previous values: carry-forward
     (previous phase, then Phase 0). A column never seen since Phase 0 keeps Phase 0's fallback estimates
     (phase0.column_fallback). So every type always has an r~.
  4. sigma is refit on the window's markups if bid.refit_sigma and the window has >= phase_transition.n_min arrivals.
  5. The carbon price s_C is Phase 0's (frozen), so r~ (in $) lives on one scale.
Source codes per column (PhaseState.src_cost / src_dur): 0 = fresh (this window, or H in Phase 0), 1 = carried
forward, 2 = fallback estimate.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ...bid import BidResult, bid_model
from ...lp_oracle import solve_lp_rs
from ...phase0 import PhaseState, assign_type_ids, column_ids, column_means, cols_to_types, type_values
from ...reward import type_reward
from .stream import StreamData

FRESH, CARRY, FALLBACK = 0, 1, 2


@dataclass
class ReinitDiag:
    n_arrivals: int
    n_observed: int
    W: int
    rho: float                 # grid ratio (fixed from Phase 0)
    sigma: float               # sigma of the new state's bid model
    sigma_refit: bool
    n_types_seen: int          # types with an arrival in the window
    n_cost_carry: int          # columns whose cost base c is carried forward
    n_cost_fallback: int       # columns whose cost base c is still Phase 0's fallback estimate
    n_dur_carry: int           # columns whose duration / energy / costs / request are carried forward
    n_dur_fallback: int        # columns whose duration / energy / costs / request are still the fallback estimates
    lam_hat: float
    tau: np.ndarray
    lp_status: str


def _source(has: np.ndarray, prev_src) -> np.ndarray:
    prev_src = np.zeros(has.size, np.int8) if prev_src is None else np.asarray(prev_src)
    return np.where(has, FRESH, np.where(prev_src == FALLBACK, FALLBACK, CARRY)).astype(np.int8)


def build_window_state(
    stream: StreamData, idx_arr: np.ndarray, idx_obs: np.ndarray, W: int,
    prev: PhaseState, ds, cfg,
) -> tuple[PhaseState, ReinitDiag]:
    ts, grid = prev.type_space, prev.grid                 # fixed from Phase 0
    n_types = ts.n_types
    idx_arr = np.asarray(idx_arr, np.int64)
    idx_obs = np.asarray(idx_obs, np.int64)
    v_arr = stream.v[idx_arr]
    c_arr = stream.cost_base[idx_arr]

    refit = cfg.bid.refit_sigma and idx_arr.size >= max(cfg.phase_transition.n_min, 2)
    myerson = bid_model(cfg, v_arr, c_arr) if refit else prev.myerson

    k_arr = grid.assign(v_arr) if idx_arr.size else np.zeros(0, np.int64)
    N = np.bincount(assign_type_ids(ts, k_arr, stream.priority[idx_arr], stream.hw_idx[idx_arr]), minlength=n_types)
    n_cols = ts.n_cols

    def cols_of(idx):
        return column_ids(ts, stream.priority[idx], stream.hw_idx[idx])

    Na, mc = column_means(cols_of(idx_arr), n_cols, dict(c_col=c_arr))
    o = idx_obs
    M, md = column_means(cols_of(o), n_cols, dict(
        d_bar=stream.D[o], energy_bar=stream.energy_kwh[o], power_bar=stream.power_kw[o],
        C_elec_bar=stream.C_elec[o], C_carbon_bar=stream.C_carbon[o], A_type=stream.A[o]))

    has_c, has_d = Na > 0, M > 0                          # carry-forward (previous phase, then Phase 0 / fallback)
    c_col = np.where(has_c, mc["c_col"], prev.c_col)
    for k in md:
        md[k] = np.where(has_d.reshape((-1,) + (1,) * (md[k].ndim - 1)), md[k], prev.col(getattr(prev, k)))
    src_cost, src_dur = _source(has_c, prev.src_cost), _source(has_d, prev.src_dur)

    T = {k: cols_to_types(ts, x) for k, x in md.items()}
    v_bar, phi_bar = type_values(ts, grid, myerson, c_col, md["d_bar"] * cfg.time.tick_seconds / 3600.0)
    p = N / max(W, 1)
    v = T["A_type"] * T["d_bar"][:, None]
    r_tilde, r_nocarbon, _ = type_reward(
        v_bar, phi_bar, T["C_elec_bar"], T["C_carbon_bar"],
        p, cfg, s_C=prev.carbon_price,
    )
    lp = solve_lp_rs(p=p, r=r_tilde, v=v, c=ds.capacity, presolve=cfg.lp.presolve)
    tau = lp.tau if lp.tau.size == ds.n_resources else np.zeros(ds.n_resources)

    phi_arr = np.asarray(myerson.phi_job(v_arr, c_arr)) if idx_arr.size else np.zeros(0)
    bids = BidResult(v_rate=v_arr, phi_rate=phi_arr, cost_base=c_arr)
    st = replace(
        prev, myerson=myerson, bids=bids, p=p, d_bar=T["d_bar"],
        v_bar=v_bar, phi_bar=phi_bar, energy_bar=T["energy_bar"],
        power_bar=T["power_bar"], r_tilde=r_tilde, r_nocarbon=r_nocarbon,
        C_elec_bar=T["C_elec_bar"], C_carbon_bar=T["C_carbon_bar"], v=v, A_type=T["A_type"],
        tau=tau, lam_star=lp.optimum, n_assigned=int(idx_arr.size), c_col=c_col, src_cost=src_cost, src_dur=src_dur,
    )
    diag = ReinitDiag(
        n_arrivals=int(idx_arr.size), n_observed=int(idx_obs.size), W=int(W),
        rho=float(grid.ratio), sigma=float(myerson.sigma), sigma_refit=bool(refit),
        n_types_seen=int((N > 0).sum()),
        n_cost_carry=int((src_cost == CARRY).sum()), n_cost_fallback=int((src_cost == FALLBACK).sum()),
        n_dur_carry=int((src_dur == CARRY).sum()), n_dur_fallback=int((src_dur == FALLBACK).sum()),
        lam_hat=float(lp.optimum), tau=tau, lp_status=lp.status,
    )
    return st, diag


def hindsight_lambda(stream, idx, W, prev, ds, cfg) -> tuple[float, ReinitDiag]:
    """
    Hindsight benchmark on a window: full information (every arrival's true D observed, so the column means are over
    all arrivals; sigma refit on the window; Phase 0's grid). Types without arrivals have p = 0, so they do not enter
    the LP. Returns lam_hat
    per tick.
    """
    cfg_h = replace(cfg, phase_transition=replace(cfg.phase_transition, n_min=2))
    st, dg = build_window_state(stream, idx, idx, W, prev, ds, cfg_h)
    return float(st.lam_star), dg

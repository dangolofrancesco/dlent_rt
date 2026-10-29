"""
Precomputed, immutable view of the online stream (vectorised once).

Bids are exogenous data: they are synthesised ONCE with the same generator as
Phase 0 (seed = run.seed + online.bid_seed_offset) and the same U0 (calibrated on
H), so every model sees the same bids. The generator's phi (sigma = bid.sigma) is
kept ONLY as a diagnostic: the mechanism computes phi with its own bid model,
MyersonModel.phi_job(v, cost_base), whose sigma may be refit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...bid import calibrate_u0
from ...config import Config
from ...data import Dataset
from ...reward import electricity_cost, carbon_kg
from ...phase0 import (
    _synthesize_bids, _compute_job_energy, _compute_job_power, _lookup_grid,
)


@dataclass(frozen=True)
class StreamData:
    n: int
    tick: np.ndarray          # 1-based tick of each arrival (paper time index)
    v: np.ndarray             # bid
    phi_gen: np.ndarray       # generator phi (diagnostic only)
    cost_base: np.ndarray     # reference cost c_j (public function of the features)
    priority: np.ndarray
    hw_idx: np.ndarray
    A: np.ndarray             # (n, |I|) real request (machines)
    D: np.ndarray             # duration in steps (clipped to d_max)
    duration_hours: np.ndarray
    energy_kwh: np.ndarray
    power_kw: np.ndarray
    price: np.ndarray         # $/kWh at request time
    ci: np.ndarray            # kgCO2/kWh at request time
    C_elec: np.ndarray        # $      = E * price
    C_carbon: np.ndarray      # kgCO2  = E * CI


def build_stream(ds: Dataset, cfg: Config) -> StreamData:
    S = ds.stream
    bids = _synthesize_bids(S, cfg, seed=cfg.run.seed + cfg.online.bid_seed_offset, u0=calibrate_u0(ds.H, cfg))
    E = _compute_job_energy(S, cfg)
    P = _compute_job_power(S, cfg)
    price, ci = _lookup_grid(S, ds)
    tick = np.asarray(S.arrival_step, np.int64) + 1
    assert np.all(np.diff(tick) > 0), "at most one arrival per tick"
    return StreamData(
        n=S.n, tick=tick, v=bids.v_rate, phi_gen=bids.phi_rate,
        cost_base=bids.cost_base,
        priority=S.priority, hw_idx=S.hw_idx, A=S.A, D=np.asarray(S.duration_steps, np.int64),
        duration_hours=S.duration_hours, energy_kwh=E, power_kw=P,
        price=np.asarray(price, float), ci=np.asarray(ci, float),
        C_elec=electricity_cost(E, price), C_carbon=carbon_kg(E, ci),
    )

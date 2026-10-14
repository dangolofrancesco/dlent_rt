"""
Calibrate bid.base_utility_u0 (moved out of dlent_rt/phase0.py, audit D5).

U_0 is the single free parameter of bid.base_utility_mode. The value in configs/default.yaml
(23.926) was obtained with this bisection: the smallest U_0 such that the share of POPULATED
Phase-0 types with r_tilde > 0 reaches target_pos_frac = 0.90, at sigma = 0.3. It depends on
bid.gamma1 / bid.gamma2 (they enter cost_base): changing them requires re-running this script.

    dlent_venv/bin/python scripts/calibrate_u0.py            # default config, mode from the config
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dlent_rt.config import Config, default_config
from dlent_rt.data import Dataset, load_dataset
from dlent_rt.phase0 import run_phase0


def calibrate_base_utility_u0(
    ds: Dataset, cfg: Config,
    mode: str,
    sigma: Optional[float] = None,
    target_pos_frac: float = 0.90,
    seed: int = 42,
    lo: float = 0.0, hi: float = 500.0, tol: float = 0.5, max_iter: int = 25,
) -> float:
    """
    Bisection search for the smallest U_0 such that the fraction of
    POPULATED TYPES with r_tilde > 0 (pos_reward_types) reaches
    target_pos_frac, for the given base_utility_mode and sigma.

    Calibrating on the raw per-job phi(v) > 0 fraction instead would be a
    no-op: cost_base enters v (and hence phi(v)) as a pure per-job scale
    factor, so phi(v)'s SIGN depends only on sigma -- never on cost_base's
    magnitude, and therefore never on U_0. U_0 only starts to matter once bids are aggregated into r_tilde
    per type together with W and C, where it shifts V_prof's magnitude
    relative to the other two (already-normalized) objectives.
    """
    def pos_frac(u0: float) -> float:
        cfg_i = replace(cfg, bid=replace(
            cfg.bid, base_utility_u0=u0, base_utility_mode=mode,
            sigma=sigma if sigma is not None else cfg.bid.sigma,
        ), run=replace(cfg.run, seed=seed))
        st_i = run_phase0(ds, cfg_i)
        m_i = st_i.p > 0
        return float((st_i.r_tilde[m_i] > 0).mean()) if np.any(m_i) else 0.0

    if pos_frac(lo) >= target_pos_frac:
        return lo
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if pos_frac(mid) < target_pos_frac:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return hi


if __name__ == "__main__":
    cfg = default_config()
    ds = load_dataset(cfg)
    u0 = calibrate_base_utility_u0(ds, cfg, mode=cfg.bid.base_utility_mode, sigma=cfg.bid.sigma,
                                   seed=cfg.run.seed)
    print(f"base_utility_u0 = {u0:.4f}  (config: {cfg.bid.base_utility_u0}, mode={cfg.bid.base_utility_mode}, "
          f"sigma={cfg.bid.sigma})")

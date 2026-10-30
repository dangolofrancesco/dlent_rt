"""dlent_rt.reward: the single source of the reward terms (week-30 audit, Section 3)."""
import tempfile, warnings
from pathlib import Path

import numpy as np

from dlent_rt import reward
from dlent_rt.data import load_dataset
from dlent_rt.phase0 import run_phase0, assign_type_ids
from dlent_rt.models.core.stream import build_stream
from tests.synthetic import synthetic_config


def _setup():
    with tempfile.TemporaryDirectory() as d, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = synthetic_config(Path(d))
        ds = load_dataset(cfg)
        return cfg, ds, run_phase0(ds, cfg)


def test_job_terms_closed_form():
    cfg, ds, st = _setup()
    S = build_stream(ds, cfg)
    t = reward.job_reward(S, st, cfg.objective)
    lam = cfg.objective
    d = S.duration_hours                                            # the true duration: results only
    phi = np.asarray(st.myerson.phi_job(S.v, S.cost_base)) * d
    np.testing.assert_array_equal(t.phi, phi)
    np.testing.assert_array_equal(t.cv, S.v * d)                   # D1: V = b d, no priority factor
    np.testing.assert_array_equal(t.profit, phi - S.C_elec)
    np.testing.assert_array_equal(S.C_elec, S.energy_kwh * S.price)
    np.testing.assert_array_equal(S.C_carbon, S.energy_kwh * S.ci)
    exp = lam.lambda1 * S.v * d + lam.lambda2 * (phi - S.C_elec) - lam.lambda3 * st.carbon_price * S.C_carbon
    np.testing.assert_allclose(t.R, exp, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(t.R, t.R_nocarbon - lam.lambda3 * t.carbon_usd, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(t.sw, S.v * d - S.C_elec - st.carbon_price * S.C_carbon, rtol=1e-12, atol=1e-12)
    adm = np.arange(S.n) % 3 == 0
    assert reward.social_welfare(t, adm) == float(t.sw[adm].sum())


def test_type_reward_from_type_terms():
    """r~ (k, q, h) = d_(q,h) (L_k + phi_(q,h)(L_k)) - C_elec_(q,h) - s_C K_(q,h): the value from the bin's lower edge,
    the costs from the column (week 33). It is NOT the mean of the job reward over the type's jobs any more."""
    cfg, ds, st = _setup()
    ts = st.type_space
    d_h = st.col(st.d_bar) * cfg.time.tick_seconds / 3600.0
    L = st.grid.lower_edges[:, None]
    np.testing.assert_allclose(st.v_bar.reshape(ts.k_star, ts.n_cols), L * d_h, rtol=1e-12)
    np.testing.assert_allclose(st.phi_bar.reshape(ts.k_star, ts.n_cols),
                               np.asarray(st.myerson.phi_job(L, st.c_col[None, :])) * d_h, rtol=1e-12)
    r, r_nc, info = reward.type_reward(st.v_bar, st.phi_bar, st.C_elec_bar, st.C_carbon_bar, st.p, cfg)
    np.testing.assert_array_equal(r, st.r_tilde)               # Phase 0 IS reward.type_reward
    np.testing.assert_array_equal(r_nc, st.r_nocarbon)
    assert info["s_C"] == st.carbon_price

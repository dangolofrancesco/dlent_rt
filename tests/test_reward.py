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
    phi = np.asarray(st.myerson.phi_job(S.v, S.cost_base))
    np.testing.assert_array_equal(t.phi, phi)
    np.testing.assert_array_equal(t.cv, S.v)                       # D1: V = v, no priority factor
    np.testing.assert_array_equal(t.profit, phi - S.C_elec)
    np.testing.assert_array_equal(S.C_elec, S.energy_kwh * S.price)
    np.testing.assert_array_equal(S.C_carbon, S.energy_kwh * S.ci)
    exp = lam.lambda1 * S.v + lam.lambda2 * (phi - S.C_elec) - lam.lambda3 * st.carbon_price * S.C_carbon
    np.testing.assert_allclose(t.R, exp, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(t.R, t.R_nocarbon - lam.lambda3 * t.carbon_usd, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(t.sw, S.v - S.C_elec - st.carbon_price * S.C_carbon, rtol=1e-12, atol=1e-12)
    adm = np.arange(S.n) % 3 == 0
    assert reward.social_welfare(t, adm) == float(t.sw[adm].sum())


def test_type_reward_is_mean_of_job_reward():
    cfg, ds, st = _setup()
    H, b = ds.H, st.bids
    tid = assign_type_ids(st.type_space, st.grid.assign(b.v_rate), H.priority, H.hw_idx)
    from dlent_rt.phase0 import _compute_job_energy, _lookup_grid
    E = _compute_job_energy(H, cfg)
    price, ci = _lookup_grid(H, ds)
    t = reward.job_terms(b.v_rate, b.cost_base, reward.electricity_cost(E, price),
                         reward.carbon_kg(E, ci), st.myerson, st.carbon_price, cfg.objective)
    N = np.bincount(tid, minlength=st.n_types)
    mean_R = reward.type_mean(tid, N, t.R)
    np.testing.assert_allclose(st.r_tilde[N > 0], mean_R[N > 0], rtol=1e-9, atol=1e-9)
    r, r_nc, info = reward.type_reward(st.v_bar, st.phi_bar, st.C_elec_bar, st.C_carbon_bar, st.p, cfg)
    np.testing.assert_array_equal(r, st.r_tilde)               # Phase 0 IS reward.type_reward
    np.testing.assert_array_equal(r_nc, st.r_nocarbon)
    assert info["s_C"] == st.carbon_price

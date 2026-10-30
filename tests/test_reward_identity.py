"""Reward accounting: type terms vs job data, logged == re-scored, shadow prices.

Since week 33 the type reward is built from the bin's lower edge (value) and the column means (costs, duration,
volume), so it is no longer the mean of the job reward over the type's jobs. Checked here: on the reference state
(every stream arrival observed) the column terms are the column means of the jobs' costs and durations, r~ follows
the formula, and the fluid LP with tau* = 0 is sum_j N_j r_j^+."""
import tempfile, warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from dlent_rt import reward
from dlent_rt.config import default_config, resolve_path
from dlent_rt.data import load_dataset
from dlent_rt.phase0 import run_phase0, assign_type_ids, column_ids, column_means
from dlent_rt.models.theory import TheoryModel
from dlent_rt.models.core.reinit import build_window_state
from dlent_rt.models.core.stream import build_stream
from tests.synthetic import synthetic_config

REAL_SLICE_H_DAYS = 29.5       # real trace, H = 29.5 days, so the stream is short: identity + Phase 0 checks quickly


class _Recording(TheoryModel):
    def run(self):
        self.states = []
        return super().run()

    def _new_phase(self, state, tau_prev, W_prev):
        self.states.append(state)
        super()._new_phase(state, tau_prev, W_prev)


def job_reward(S, st0, cfg):
    """R of every stream job from the single reward module (the online loop's terms)."""
    return reward.job_reward(S, st0, cfg.objective).R


def _synthetic():
    d = tempfile.mkdtemp()
    return synthetic_config(Path(d), n0=400, n_jobs=3000, drift=True)


def _real_slice():
    cfg = default_config()
    if not resolve_path(cfg.data.batch_csv).is_file():
        pytest.skip("real trace not available")
    return replace(cfg, data=replace(cfg.data, h_days=REAL_SLICE_H_DAYS),
                   time=replace(cfg.time, d_cap_hours=2.0)).validate()


def _load(cfg):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = load_dataset(cfg)
        return ds, run_phase0(ds, cfg)


def _assert_tau(st):
    assert np.all(st.tau >= 0)
    assert np.all(st.r_tilde - st.v @ st.tau <= st.r_tilde)


@pytest.mark.parametrize("which", ["synthetic", "real_slice"])
def test_type_job_identity(which):
    cfg = _synthetic() if which == "synthetic" else _real_slice()
    ds, st0 = _load(cfg)
    S = build_stream(ds, cfg)
    T = int(ds.T)
    idx = np.arange(S.n)
    cfg_h = replace(cfg, phase_transition=replace(cfg.phase_transition, n_min=2))
    st_ref, _ = build_window_state(S, idx, idx, T, st0, ds, cfg_h)
    ts = st_ref.type_space
    tid = assign_type_ids(ts, st_ref.grid.assign(S.v), S.priority, S.hw_idx)
    N = np.bincount(tid, minlength=st_ref.n_types).astype(float)
    # the column terms are the column means of the jobs (every arrival observed)
    cid = column_ids(ts, S.priority, S.hw_idx)
    Nc, mc = column_means(cid, ts.n_cols, dict(C_elec=S.C_elec, K=S.C_carbon, D=S.D, c=S.cost_base))
    s = Nc > 0
    np.testing.assert_allclose(st_ref.col(st_ref.C_elec_bar)[s], mc["C_elec"][s], rtol=1e-12)
    np.testing.assert_allclose(st_ref.col(st_ref.C_carbon_bar)[s], mc["K"][s], rtol=1e-12)
    np.testing.assert_allclose(st_ref.col(st_ref.d_bar)[s], mc["D"][s], rtol=1e-12)
    np.testing.assert_allclose(st_ref.c_col[s], mc["c"][s], rtol=1e-12)
    # r~ = V + Pi - s_C K from the type terms
    np.testing.assert_allclose(st_ref.r_tilde, st_ref.v_bar + st_ref.phi_bar - st_ref.C_elec_bar
                               - st0.carbon_price * st_ref.C_carbon_bar, rtol=1e-12, atol=1e-12)
    pos = st_ref.r_tilde > 0
    lhs = float((N[pos] * st_ref.r_tilde[pos]).sum())
    if np.all(st_ref.tau == 0):          # capacity not binding: fluid OPT = sum_j N_j r_j^+
        assert st_ref.lam_star * T == pytest.approx(lhs, rel=1e-9)
    _assert_tau(st0)
    _assert_tau(st_ref)


@pytest.fixture(scope="module")
def synthetic_run():
    cfg = _synthetic()
    ds, st0 = _load(cfg)
    m = _Recording(ds, st0, cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = m.run()
    return cfg, m, res


def test_logged_equals_rescored(synthetic_run):
    cfg, m, res = synthetic_run
    R = job_reward(m.stream, m.state0, cfg)
    i = np.array([a["i"] for a in res.arrivals])
    adm = np.array([a["admit"] for a in res.arrivals])
    logged = np.array([a["R"] for a in res.arrivals])
    assert adm.any()
    np.testing.assert_allclose(logged[adm], R[i[adm]], rtol=1e-12, atol=1e-12)
    assert np.all(logged[~adm] == 0.0)
    assert res.summary["R_total"] == pytest.approx(float(R[i[adm]].sum()), rel=1e-12)


def test_shadow_prices_nonnegative_every_phase(synthetic_run):
    cfg, m, res = synthetic_run
    assert len(m.states) >= 2
    for st in m.states:              # Phase 0 + every ReInit state
        _assert_tau(st)
    assert all(p["tau_cpu"] >= 0 and p["tau_ram"] >= 0 for p in res.phases)

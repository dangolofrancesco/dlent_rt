"""Ablation flags (cfg.ablation, week 30): off by default; each one changes only what it claims."""
import tempfile
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from dlent_rt.config import AblationCfg, Config
from dlent_rt.data import load_dataset
from dlent_rt.models.core.error_params import make_error_params
from dlent_rt.models.theory import TheoryModel
from dlent_rt.phase0 import run_phase0
from tests.synthetic import synthetic_config


@pytest.fixture(scope="module")
def world():
    with tempfile.TemporaryDirectory() as d, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = synthetic_config(Path(d), n0=400, n_jobs=3000, drift=True)
        ds = load_dataset(cfg)
        st = run_phase0(ds, cfg)
        yield cfg, ds, st


def with_ab(cfg, **kw):
    return replace(cfg, ablation=replace(cfg.ablation, **kw)).validate()


class StateLog(TheoryModel):
    def run(self):
        self.states = []
        return super().run()

    def _new_phase(self, state, tau_prev, W_prev):
        self.states.append(state)
        super()._new_phase(state, tau_prev, W_prev)


def test_defaults_off():
    ab = Config().ablation
    assert ab == AblationCfg() and not ab.thinning_off and ab.thinning_delta_power is None
    assert not ab.thinning_eps_D_zero and not ab.fixed_grid


def test_error_params_unchanged_with_flags_off(world):
    cfg, ds, st = world
    ep = make_error_params(st, ds, cfg)
    assert not ep.thin_off and ep.eps_D_thin is None and ep.eps_D_thinning == ep.eps_D
    W, t = int(ds.d_max_steps), 7
    assert ep.p_U(W, t) == (1.0 - ep.xi) / (1.0 + ep.eps_D + ep.eta(W) + ep.eta(t))


def test_A1a_thinning_off_gives_pU_one(world):
    cfg, ds, st = world
    res = TheoryModel(ds, st, with_ab(cfg, thinning_off=True)).run()
    assert all(a["p_U"] == 1.0 and a["U"] for a in res.arrivals)
    assert sum(p["n_thinned"] for p in res.phases) == 0
    ep0, ep1 = make_error_params(st, ds, cfg), make_error_params(st, ds, with_ab(cfg, thinning_off=True))
    assert (ep1.eps_D, ep1.xi, ep1.gamma) == (ep0.eps_D, ep0.xi, ep0.gamma)          # tests' constants unchanged


def test_A1b_A1c_change_only_the_thinning_eps_D(world):
    cfg, ds, st = world
    ep0 = make_error_params(st, ds, cfg)
    b = make_error_params(st, ds, with_ab(cfg, thinning_delta_power=1))
    c = make_error_params(st, ds, with_ab(cfg, thinning_eps_D_zero=True))
    log_I = np.log(ds.n_resources * float(ds.T))                                      # delta = 1/T
    assert b.eps_D_thinning == pytest.approx(2 * np.sqrt(b.xi * log_I) + 2 * b.xi * log_I, rel=1e-12)
    assert b.eps_D_thinning < ep0.eps_D and c.eps_D_thinning == 0.0
    for e in (b, c):                                                                  # Test 2 / radii untouched
        assert e.eps_D == ep0.eps_D and e.delta == ep0.delta
        assert e.eps_test2(500, 300) == ep0.eps_test2(500, 300) and e.eps_test3(300) == ep0.eps_test3(300)
    W, t = int(ds.d_max_steps), 7
    assert c.p_U(W, t) == pytest.approx((1 - c.xi) / (1 + c.eta(W) + c.eta(t)), rel=1e-12)
    assert ep0.p_U(W, t) < b.p_U(W, t) < c.p_U(W, t)
    with pytest.raises(AssertionError):
        with_ab(cfg, thinning_delta_power=1, thinning_eps_D_zero=True)


def test_A3_fixed_grid_keeps_the_phase0_grid(world):
    cfg, ds, st = world
    m0 = StateLog(ds, st, cfg)
    r0 = m0.run()
    assert any(p.get("reinit_grid_refit") for p in r0.phases)                         # default: refit happens
    m1 = StateLog(ds, st, with_ab(cfg, fixed_grid=True))
    r1 = m1.run()
    assert len(m1.states) >= 2
    for s_ in m1.states:
        assert s_.grid is st.grid                                                     # the same grid object
    assert not any(p.get("reinit_grid_refit") for p in r1.phases)

"""quantile_linear normalization: s_C = D_q / K_q, frozen at Phase 0, reward in dollars."""
import tempfile, warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from dlent_rt.data import load_dataset
from dlent_rt.phase0 import (
    run_phase0, scalarized_reward, type_arrays, carbon_scale_from_quantile, _weighted_quantile,
)
from dlent_rt.models.core.reinit import build_window_state
from dlent_rt.models.core.stream import build_stream
from tests.synthetic import synthetic_config


def _with(cfg, **norm):
    return replace(cfg, normalization=replace(cfg.normalization, **norm))


@pytest.fixture(scope="module")
def setup():
    with tempfile.TemporaryDirectory() as d:
        cfg = synthetic_config(Path(d))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = load_dataset(cfg)
            st = run_phase0(ds, cfg)
        yield cfg, ds, st


def _raw(st):
    _, prio, _ = type_arrays(st.type_space, st.grid)
    return prio * st.v_bar, st.phi_bar - st.C_elec_bar, st.C_carbon_bar, prio


# (a) s_C * K_q == D_q, and the Phase-0 state carries s_C
def test_scale_maps_Kq_onto_Dq(setup):
    cfg, ds, st = setup
    assert cfg.normalization.strategy == "quantile_linear"
    info = st.norm_info
    assert info["strategy"] == "quantile_linear" and info["q"] == cfg.normalization.quantile
    assert info["s_C"] > 0
    assert np.isclose(info["s_C"] * info["K_q"], info["D_q"], rtol=1e-12)
    assert np.isclose(info["D_q"], 0.5 * (info["Q_sw"] + info["Q_abs_pi"]), rtol=1e-12)
    assert st.carbon_price == info["s_C"]
    assert st.norm_ref == (1.0, 1.0, 1.0)
    # reward in dollars, no divisor
    sat, prof, carb, _ = _raw(st)
    lam = cfg.objective
    exp = lam.lambda1 * sat + lam.lambda2 * prof - lam.lambda3 * info["s_C"] * carb
    np.testing.assert_allclose(st.r_tilde, exp, rtol=1e-12, atol=1e-15)


# (b) changing q changes only s_C (raw objectives and the non-carbon part are unchanged)
def test_q_changes_only_carbon_price(setup):
    cfg, ds, st = setup
    sat, prof, carb, prio = _raw(st)
    out = {}
    for q in (0.5, 0.75, 0.9, 0.99):
        cfg_q = _with(cfg, quantile=q)
        r, r_nocarbon, ref, info = scalarized_reward(
            prio, st.v_bar, st.phi_bar, st.C_elec_bar, st.C_carbon_bar, st.p, st.v, ds, cfg_q)
        out[q] = (r, r_nocarbon, info)
        lam = cfg.objective
        np.testing.assert_allclose(r_nocarbon, lam.lambda1 * sat + lam.lambda2 * prof, rtol=1e-12)
        np.testing.assert_allclose(r - r_nocarbon, -lam.lambda3 * info["s_C"] * carb, rtol=1e-12, atol=1e-15)
        assert ref == (1.0, 1.0, 1.0)
    revs = [o[1] for o in out.values()]
    assert all(np.array_equal(revs[0], x) for x in revs[1:])
    assert len({o[2]["s_C"] for o in out.values()}) == len(out)
    # the full Phase 0 run agrees with the direct computation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st75 = run_phase0(ds, _with(cfg, quantile=0.75))
    assert np.isclose(st75.carbon_price, out[0.75][2]["s_C"], rtol=1e-12)
    np.testing.assert_array_equal(st75.r_nocarbon, st.r_nocarbon)


# (c) ReInit reuses the frozen Phase-0 s_C
def test_reinit_uses_frozen_carbon_price(setup):
    cfg, ds, st = setup
    S = build_stream(ds, cfg)
    idx = np.arange(min(S.n, 400))
    W = int(S.tick[idx[-1]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new, _ = build_window_state(S, idx, idx, W, st, ds, cfg, float(st.norm_ref[0]))
    assert new.carbon_price == st.carbon_price
    assert new.norm_info == st.norm_info
    sat, prof, carb, _ = _raw(new)
    lam = cfg.objective
    np.testing.assert_allclose(
        new.r_tilde, lam.lambda1 * sat + lam.lambda2 * prof - lam.lambda3 * st.carbon_price * carb,
        rtol=1e-12, atol=1e-15)
    # a fresh scale on the window would differ: the freeze is not a no-op
    fresh = carbon_scale_from_quantile(sat, prof, carb, new.p, cfg.normalization.quantile)
    assert not np.isclose(fresh["s_C"], st.carbon_price)


# (d) the other strategies are numerically unchanged (closed-form reference)
def test_other_strategies_unchanged(setup):
    cfg, ds, _ = setup
    lam = cfg.objective
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st_pj = run_phase0(ds, _with(cfg, strategy="per_job"))
        st_cc = run_phase0(ds, _with(cfg, strategy="common_currency"))
    tax = cfg.normalization.carbon_tax_per_kg

    sat, prof, carb, _ = _raw(st_pj)
    m = st_pj.p > 0
    span = lambda a: max(float(np.abs(a[m]).max()), 1e-12)
    exp = (lam.lambda1 * sat / span(sat) + lam.lambda2 * prof / span(prof)
           - lam.lambda3 * carb / span(carb))
    np.testing.assert_allclose(st_pj.r_tilde, exp, rtol=1e-12, atol=1e-15)
    assert st_pj.carbon_price == tax and st_pj.norm_info is None

    sat, prof, carb, _ = _raw(st_cc)
    s = max(span(sat), span(prof), span(carb * tax))
    exp = lam.lambda1 * sat / s + lam.lambda2 * prof / s - lam.lambda3 * (carb * tax) / s
    np.testing.assert_allclose(st_cc.r_tilde, exp, rtol=1e-12, atol=1e-15)
    assert st_cc.norm_ref == (s, s, s)
    assert st_cc.carbon_price == tax and st_cc.norm_info is None


# (e) weighted quantile on toy examples, hand-computed
def test_weighted_quantile_toy():
    # equal weights: atoms at cw = .125 .375 .625 .875 -> median = 2.5
    assert _weighted_quantile([4, 1, 3, 2], [1, 1, 1, 1], 0.5) == pytest.approx(2.5)
    # w = (1,1,1,5)/8: cw = .0625 .1875 .3125 .6875
    # q = .5 -> 3 + (.5 - .3125)/(.6875 - .3125) = 3.5
    assert _weighted_quantile([1, 2, 3, 4], [1, 1, 1, 5], 0.5) == pytest.approx(3.5)
    # q = .25 -> 2 + (.25 - .1875)/(.3125 - .1875) = 2.5 ; q beyond the last atom -> max
    assert _weighted_quantile([1, 2, 3, 4], [1, 1, 1, 5], 0.25) == pytest.approx(2.5)
    assert _weighted_quantile([1, 2, 3, 4], [1, 1, 1, 5], 0.9) == pytest.approx(4.0)
    # carbon_scale_from_quantile ignores types with p = 0
    info = carbon_scale_from_quantile(np.array([2.0, 4.0, 999.0]), np.array([-2.0, 6.0, 999.0]),
                                      np.array([1.0, 3.0, 999.0]), np.array([0.5, 0.5, 0.0]), 0.5)
    assert info["Q_sw"] == pytest.approx(3.0) and info["Q_abs_pi"] == pytest.approx(4.0)
    assert info["D_q"] == pytest.approx(3.5) and info["K_q"] == pytest.approx(2.0)
    assert info["s_C"] == pytest.approx(1.75)
    with pytest.warns(RuntimeWarning):
        z = carbon_scale_from_quantile(np.ones(2), np.ones(2), np.zeros(2), np.ones(2), 0.9)
    assert z["s_C"] == 0.0


# (f) m(0) = 0: a type with K = 0 has no carbon term
def test_zero_carbon_has_no_carbon_term(setup):
    cfg, ds, st = setup
    _, _, _, prio = _raw(st)
    C0 = st.C_carbon_bar.copy()
    j = int(np.flatnonzero(st.p > 0)[0])
    C0[j] = 0.0
    r, r_nocarbon, _, _ = scalarized_reward(
        prio, st.v_bar, st.phi_bar, st.C_elec_bar, C0, st.p, st.v, ds, cfg,
        carbon_price=st.carbon_price)
    assert r[j] == r_nocarbon[j]

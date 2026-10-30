"""The truthful threshold mechanism on hourly bids (week 33).

* The bid is hourly: it does not depend on the duration.
* The bin sets only the value (through L_k), the column (priority, profile) sets the costs and the volume; the margin
  r~ - tau^T v is increasing in k in every column, so the admitted bins are an upper set (threshold_kstar raises if
  not).
* IR: 0 < p <= b for every admitted job; truthfulness: the hourly payment is the lowest admitted bid, no other bid is
  admitted at a lower hourly payment, a bid below k* is rejected.
* The true duration never enters a decision.
* The grid and the types are fixed from Phase 0; columns without jobs carry forward, then keep the fallback.
"""
import tempfile, warnings
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dlent_rt.bid import synthesize_bids
from dlent_rt.config import default_config
from dlent_rt.data import load_dataset
from dlent_rt.grid import build_valuation_grid
from dlent_rt.mechanism import admit_and_pay, reserve_price, threshold_kstar
from dlent_rt.models.core.reinit import CARRY, FALLBACK, FRESH, build_window_state
from dlent_rt.models.core.stream import build_stream
from dlent_rt.models.theory import TheoryModel
from dlent_rt.phase0 import column_ids, run_phase0
from tests.synthetic import synthetic_config

FACTORS = (0.5, 0.8, 1.25, 2.0)


def grid_of(m, a):
    return m.state0.grid                                                  # the grid is fixed from Phase 0


# --------------------------------------------------------------------------- #
# Hand-made checks
# --------------------------------------------------------------------------- #
def test_bid_does_not_depend_on_duration():
    cfg = default_config()
    n = 50_000
    rng = np.random.default_rng(0)
    cpu, ram, q = rng.uniform(0.001, 0.05, n), rng.uniform(0.001, 0.05, n), rng.integers(1, 6, n).astype(float)

    def jobs(d):
        return SimpleNamespace(A=np.column_stack([cpu, ram]), duration_hours=d, priority=q, n=n)
    b1 = synthesize_bids(jobs(np.full(n, 0.1)), cfg, u0=0.02, seed=7)
    b2 = synthesize_bids(jobs(rng.uniform(0.02, 24.0, n)), cfg, u0=0.02, seed=7)
    np.testing.assert_array_equal(b1.v_rate, b2.v_rate)              # same request and priority -> same bids
    np.testing.assert_array_equal(b1.cost_base, b2.cost_base)


def test_threshold_kstar_and_upper_set():
    m = np.cumsum(np.random.default_rng(0).normal(0.1, 1.0, (12, 40)), axis=0)
    m = np.maximum.accumulate(m, axis=0)                               # increasing in k
    ks = threshold_kstar(m)
    for c in range(m.shape[1]):
        ok = m[:, c] > 0
        assert ks[c] == (np.argmax(ok) + 1 if ok.any() else 0)
    with pytest.raises(ValueError, match="upper set"):
        threshold_kstar(np.array([[-1.0], [2.0], [-0.5], [3.0]]))
    np.testing.assert_array_equal(threshold_kstar(np.array([[-1.0], [-1.0]])), [0])


def _grid():
    return build_valuation_grid(np.exp(np.random.default_rng(1).normal(0, 1, 300)), 16)


@pytest.mark.parametrize("kstar,reserve", [(1, 0.05), (5, 0.01), (5, 3.0), (16, 0.2), (9, 0.0)])
def test_monotone_and_truthful_on_hand_made_jobs(kstar, reserve):
    g = _grid()
    bids = np.r_[0.0, np.geomspace(1e-4, 1e3, 4000)]
    res = [admit_and_pay(b, g, kstar, reserve) for b in bids]
    adm = np.array([b and r for b, r, _, _ in res])
    pay = np.array([p for _, _, _, p in res])
    # monotone: once admitted, every higher bid is admitted
    first = int(np.argmax(adm))
    assert adm.any() and adm[first:].all() and not adm[:first].any()
    # the hourly payment does not depend on the bid and is the lowest admitted bid (the critical value)
    p = pay[adm][0]
    assert np.all(pay[adm] == p)
    assert p == max(g.lower_edges[kstar - 1], reserve)
    assert bids[first] >= p and (first == 0 or bids[first - 1] < p)
    # a bid below k* is rejected
    below = g.assign(bids) < kstar
    assert not adm[below].any()
    # truthful: no other bid is admitted at a lower hourly payment; IR 0 <= p <= b
    for b_true in bids[adm][::400]:
        assert np.all(pay[adm] >= p) and p <= b_true


def test_reserve_is_power_times_price():
    assert reserve_price(2.0, 0.05) == pytest.approx(0.1)               # 2 kW x 0.05 $/kWh = 0.1 $ per hour


# --------------------------------------------------------------------------- #
# On the synthetic world
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def world():
    with tempfile.TemporaryDirectory() as d, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = synthetic_config(Path(d), n0=400, n_jobs=3000, drift=True)
        ds = load_dataset(cfg)
        st = run_phase0(ds, cfg)
        m = TheoryModel(ds, st, cfg)
        res = m.run()
        S = build_stream(ds, cfg)
        idx = np.arange(300)
        new, dg = build_window_state(S, idx, idx[::2], int(S.tick[idx[-1]]), st, ds, cfg)
        return cfg, ds, st, m, res, new, dg


def _check_type_terms(st, cfg):
    ts = st.type_space
    K, nc = ts.k_star, ts.n_cols
    # costs, duration and volume: one value per column, the same in every bin
    for f in ("d_bar", "C_elec_bar", "C_carbon_bar", "energy_bar", "power_bar"):
        x = np.asarray(getattr(st, f)).reshape(K, nc)
        np.testing.assert_array_equal(x, np.broadcast_to(x[0], x.shape))
    v = st.v.reshape(K, nc, -1)
    np.testing.assert_array_equal(v, np.broadcast_to(v[0], v.shape))
    # the value terms depend on the bin only through L_k: r~ = d (L_k + phi_(q,h)(L_k)) - C_elec - s_C K
    d_h = st.col(st.d_bar) * cfg.time.tick_seconds / 3600.0
    L = st.grid.lower_edges[:, None]
    phi = np.asarray(st.myerson.phi_job(L, st.c_col[None, :]))
    r = d_h * (L + phi) - st.col(st.C_elec_bar) - st.carbon_price * st.col(st.C_carbon_bar)
    np.testing.assert_allclose(st.r_tilde.reshape(K, nc), r, rtol=1e-12, atol=1e-12)
    # the margin is increasing in k in every column
    margin = (st.r_tilde - st.v @ st.tau).reshape(K, nc)
    assert np.all(np.diff(margin, axis=0) >= 0)
    threshold_kstar(margin)                                             # never raises


def test_type_terms_and_margin_increasing(world):
    cfg, ds, st, m, res, new, dg = world
    _check_type_terms(st, cfg)
    _check_type_terms(new, cfg)


def test_admitted_bins_upper_set_ir_and_reasons(world):
    cfg, ds, st, m, res, new, dg = world
    A = res.arrivals
    for a in A:
        assert a["k_hat"] == (a["k_star"] > 0 and a["k_bin"] >= a["k_star"])
    adm = [a for a in A if a["admit"]]
    assert adm and all(0 < a["payment_rate"] <= a["v"] for a in adm)               # IR per hour
    assert all(a["payment"] == pytest.approx(a["payment_rate"] * a["duration_hours"]) for a in adm)
    assert res.summary["n_ir_violations"] == 0
    rej = [a["reason"] for a in A if not a["admit"]]
    assert set(rej) <= {"bin below k*", "below reserve", "thinning", "no space"}
    assert all(a["reason"] == "" for a in adm)


def test_truthfulness_spot_check_on_the_run(world):
    """Same state, other bids b x f: no other bid is admitted at a lower hourly payment; a higher bid of an admitted
    job is admitted; a lower bid below k* is rejected."""
    cfg, ds, st, m, res, new, dg = world
    for a in res.arrivals:
        price_ok = a["k_hat"] and a["reserve_ok"]
        crit = max(a["L_kstar"], a["reserve"])                            # the critical hourly bid
        if a["admit"]:
            assert a["payment_rate"] == crit
        for f in FACTORS:
            ok_b, ok_r, _, p = admit_and_pay(a["v"] * f, grid_of(m, a), a["k_star"], a["reserve"])
            ok = ok_b and ok_r
            if ok:
                assert p == crit                                              # never a lower hourly payment
            if price_ok and f > 1:
                assert ok                                                     # a higher bid is never rejected
            if not price_ok and f < 1:
                assert not ok                                                 # a lower bid is never admitted


@pytest.mark.parametrize("oracle", ["frozen", "live"])
def test_true_duration_never_changes_the_decision(world, oracle):
    cfg, ds, st, m, res, new, dg = world
    cfg_o = replace(cfg, oracle=replace(cfg.oracle, kind=oracle))
    rng = np.random.default_rng(5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = TheoryModel(ds, st, cfg_o)
    S = mod.stream
    f = rng.uniform(0.1, 10.0, S.n)                                    # other true durations, same everything else
    S2 = replace(S, D=np.maximum(np.rint(S.D * f), 1).astype(np.int64), duration_hours=S.duration_hours * f,
                 energy_kwh=S.energy_kwh * f, C_elec=S.C_elec * f, C_carbon=S.C_carbon * f)
    for state in (st, new):
        mod.W_prev = int(ds.d_max_steps)
        mod.tau_prev = 0
        mod._new_phase(state, 0, int(ds.d_max_steps))
        for i in range(0, S.n, 7):
            t = int(S.tick[i])
            mod.stream = S
            w1, c1 = mod.want(i, t), dict(mod._cur)
            mod.stream = S2
            w2, c2 = mod.want(i, t), dict(mod._cur)
            assert w1 == w2 and c1["kstar"] == c2["kstar"] and c1["reserve"] == c2["reserve"]
            np.testing.assert_array_equal(c1["pay"], c2["pay"])


def test_grid_fixed_and_columns_carry_forward(world):
    cfg, ds, st, m, res, new, dg = world
    ts = st.type_space
    assert np.all(np.isfinite(st.r_tilde)) and np.all(st.v > 0)
    seen_H = np.bincount(column_ids(ts, ds.H.priority, ds.H.hw_idx), minlength=ts.n_cols) > 0
    assert np.array_equal(st.src_dur == FALLBACK, ~seen_H) and np.array_equal(st.src_cost == FALLBACK, ~seen_H)
    # ReInit on a window: the grid is Phase 0's; columns with no admitted job carry, never-seen columns stay fallback
    assert new.grid is st.grid and new.type_space is st.type_space
    S = build_stream(ds, cfg)
    obs = np.arange(300)[::2]
    has_d = np.bincount(column_ids(ts, S.priority[obs], S.hw_idx[obs]), minlength=ts.n_cols) > 0
    for k in ("d_bar", "C_elec_bar", "C_carbon_bar"):
        np.testing.assert_array_equal(new.col(getattr(new, k))[~has_d], st.col(getattr(st, k))[~has_d])
    assert np.all(new.src_dur[~has_d & (st.src_dur == FALLBACK)] == FALLBACK)
    assert np.all(new.src_dur[~has_d & (st.src_dur == FRESH)] == CARRY)
    assert np.all(new.src_dur[has_d] == FRESH)
    assert dg.n_dur_carry + dg.n_dur_fallback == int((~has_d).sum())
    assert len({p["rho"] for p in res.phases}) == 1                     # the same grid in every phase


# --------------------------------------------------------------------------- #
# The same mechanism in the baselines (static Phase 0, no-learning, Greedy-Pos)
# --------------------------------------------------------------------------- #
def test_baselines_use_the_same_mechanism(world):
    from dlent_rt.models.core.occupancy import Occupancy
    from dlent_rt.models.policy import ThresholdPolicy, mechanism_summary, simulate
    cfg, ds, st, m, res, new, dg = world
    S = m.stream
    pols = {"static": ThresholdPolicy(st, S),
            "no-learning": ThresholdPolicy(st, S, thin=(m.ep, m.u_rand, int(ds.d_max_steps))),
            "Greedy-Pos": ThresholdPolicy(st, S, tau=0.0)}
    out = {}
    for k, pol in pols.items():
        sim = simulate(S, Occupancy(ds.capacity, record=False), pol)
        ms = mechanism_summary(pol, sim, S)
        adm = sim.admit
        assert adm.any(), k
        # the DLENT-RT rule: price part, then thinning, then the gate
        np.testing.assert_array_equal(sim.want, pol.bin_ok & pol.res_ok & pol.U)
        # hourly payment max(L_k*, r), IR, upper set
        np.testing.assert_array_equal(pol.pay_rate[adm], np.maximum(pol.L[adm], pol.reserve[adm]))
        assert ms["payments"] == pytest.approx(float((pol.pay_rate[adm] * S.duration_hours[adm]).sum()))
        assert ms["n_ir_violations"] == 0 and ms["n_upper_set_violations"] == 0, k
        out[k] = (pol, ms)
    # the static k* is DLENT-RT's phase-1 k* (same Phase 0 state, never updated)
    np.testing.assert_array_equal(out["static"][0].kstar_col, threshold_kstar(
        (st.r_tilde - st.v @ st.tau).reshape(st.grid.k_star, len(st.type_space.priorities), st.type_space.n_hw)))
    assert out["static"][1]["rejections"]["thinning"] == 0 and out["Greedy-Pos"][1]["rejections"]["thinning"] == 0
    assert out["no-learning"][1]["rejections"]["thinning"] > 0
    # tau = 0 lowers every margin's cost, so Greedy-Pos's k* is never above the static k*
    gk, sk = out["Greedy-Pos"][0].kstar_col, out["static"][0].kstar_col
    assert np.all((sk == 0) | ((gk > 0) & (gk <= sk)))

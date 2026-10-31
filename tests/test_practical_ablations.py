"""The practical model = ablations of the theoretical model, one config switch each (week 33): thinning off, the live
oracle, bid.sigma_update (phase | cumulative | fixed) and stats.update (replace | by_end_reason). Small synthetic
trace (1,500 jobs) and hand-made jobs: a few seconds."""
import warnings
from dataclasses import replace

import numpy as np
import pytest

from dlent_rt.bid import fit_sigma, log_markups
from dlent_rt.data import load_dataset
from dlent_rt.mechanism import admit_and_pay, reserve_price, threshold_kstar
from dlent_rt.models.core.reinit import CARRY, FRESH, build_window_state, hindsight_cfg
from dlent_rt.models.core.stream import build_stream
from dlent_rt.models.theory import TheoryModel
from dlent_rt.phase0 import column_ids, run_phase0
from tests.synthetic import synthetic_config

STATS = ("d_bar", "energy_bar", "power_bar", "C_elec_bar", "C_carbon_bar", "A_type")


def with_(cfg, **kw):
    """cfg with one switch: thinning_off=, live=, sigma=, stats=."""
    if "thinning_off" in kw:
        cfg = replace(cfg, ablation=replace(cfg.ablation, thinning_off=kw["thinning_off"]))
    if "live" in kw:
        cfg = replace(cfg, oracle=replace(cfg.oracle, kind="live" if kw["live"] else "frozen"))
    if "sigma" in kw:
        cfg = replace(cfg, bid=replace(cfg.bid, sigma_update=kw["sigma"]))
    if "stats" in kw:
        cfg = replace(cfg, stats=replace(cfg.stats, update=kw["stats"]))
    return cfg.validate()


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = synthetic_config(tmp_path_factory.mktemp("abl"), n0=400, n_jobs=1500, drift=True)
        cfg = replace(cfg, logging=replace(cfg.logging, progress_every=0.0, progress_phases=False)).validate()
        ds = load_dataset(cfg)
        st0 = run_phase0(ds, cfg)
        S = build_stream(ds, cfg)
        return cfg, ds, st0, S


def run(world, **kw):
    cfg, ds, st0, S = world
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = TheoryModel(ds, st0, with_(cfg, **kw))
        return m, m.run()


def run_upper_set_violations(res, live=False) -> int:
    """Frozen: within a phase and column, a bin refused by the price rule above one it accepted. Live (k* per job):
    a decision that is not bin >= the job's own k*."""
    if live:
        return sum(int(a["k_hat"] != (a["k_star"] > 0 and a["k_bin"] >= a["k_star"])) for a in res.arrivals)
    seen = {}
    for a in res.arrivals:
        seen.setdefault((a["m"], a["q"], a["hw"]), []).append((a["k_bin"], a["k_hat"]))
    v = 0
    for x in seen.values():
        x = np.array(x)
        lo, hi = x[x[:, 1] == 1, 0], x[x[:, 1] == 0, 0]
        if lo.size and hi.size:
            v += int((hi > lo.min()).sum())
    return v


# --------------------------------------------------------------------------- #
# 1. Thinning off
# --------------------------------------------------------------------------- #
def test_thinning_off_keeps_the_mechanism(world):
    m, res = run(world, thinning_off=True)
    A = res.arrivals
    assert all(a["p_U"] == 1.0 and a["U"] for a in A)
    assert not any(a["reason"] == "thinning" for a in A)
    assert any(a["admit"] for a in A)
    for a in A:                                                  # threshold k*, reserve, then the gate only
        assert a["admit"] == (a["k_hat"] and a["reserve_ok"] and a["fits"])
        if a["admit"]:
            assert a["payment_rate"] == max(a["L_kstar"], a["reserve"]) and 0 < a["payment_rate"] <= a["v"]
    assert res.summary["n_ir_violations"] == 0 and run_upper_set_violations(res) == 0


# --------------------------------------------------------------------------- #
# 2. Live oracle
# --------------------------------------------------------------------------- #
def test_live_oracle_uses_power_dbar_price_never_the_bid(world):
    cfg, ds, st0, S = world
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = TheoryModel(ds, st0, with_(cfg, live=True))
    m._new_phase(st0, 0, int(ds.d_max_steps))
    ts, lam = st0.type_space, cfg.objective
    K, Q, A = ts.k_star, len(ts.priorities), ts.n_hw
    for i in range(0, S.n, 97):
        qi, hw = int(np.searchsorted(ts.priorities, S.priority[i])), int(S.hw_idx[i])
        ids = (np.arange(K) * Q + qi) * A + hw
        E = S.power_kw[i] * st0.col(st0.d_bar).reshape(Q, A)[qi, hw] * cfg.time.tick_seconds / 3600.0
        expect = (st0.r_tilde[ids] - lam.lambda2 * (E * S.price[i] - st0.C_elec_bar[ids])
                  - lam.lambda3 * st0.carbon_price * (E * S.ci[i] - st0.C_carbon_bar[ids])) - st0.v[ids] @ st0.tau
        np.testing.assert_allclose(m._live_scores(i, qi, hw), expect, rtol=1e-12, atol=1e-15)
        # the bid and the true duration never enter the scores
        base = m._live_scores(i, qi, hw)
        m.stream = replace(S, v=S.v * 3.0, duration_hours=S.duration_hours * 5.0, D=S.D * 5,
                           energy_kwh=S.energy_kwh * 5.0, C_elec=S.C_elec * 5.0, C_carbon=S.C_carbon * 5.0)
        np.testing.assert_array_equal(m._live_scores(i, qi, hw), base)
        m.stream = S


def test_live_oracle_truthful_and_ir_on_hand_made_jobs(world):
    """Hand-made jobs: one real job's column, power, price and carbon, with bids from far below to far above. k* and
    the reserve do not move with the bid; admitted iff bin(b) >= k* and b >= reserve; the hourly payment is the same
    critical bid for every admitted bid and never above it; a true duration x 0.1 or x 10 changes nothing."""
    cfg, ds, st0, S = world
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = TheoryModel(ds, st0, with_(cfg, live=True, thinning_off=True))
    m._new_phase(st0, 0, int(ds.d_max_steps))
    g = st0.grid
    n_admitted = 0
    for i in (3, 40, 211, 517, 902):
        t = int(S.tick[i])
        bids = g.bnd_v[0] * np.geomspace(0.5, 2 * g.bnd_v[-1] / g.bnd_v[0], 25)
        crit = set()
        for b in bids:
            for f_d in (1.0, 0.1, 10.0):
                hm = dict(v=S.v.copy(), duration_hours=S.duration_hours.copy(), D=S.D.copy())
                hm["v"][i], hm["duration_hours"][i] = b, S.duration_hours[i] * f_d
                hm["D"][i] = max(int(S.D[i] * f_d), 1)
                m.stream = replace(S, **hm)
                w = m.want(i, t)
                c = m._cur
                if f_d == 1.0:
                    ref = (w, c["kstar"], c["reserve"], c["pay"])
                else:
                    assert (w, c["kstar"], c["reserve"], c["pay"]) == ref        # true duration: no effect
            w, kstar, reserve, pay = ref
            assert reserve == pytest.approx(float(reserve_price(S.power_kw[i], S.price[i])))
            assert w == (kstar > 0 and int(g.assign(b)[0]) >= kstar and b >= reserve)
            if w:
                n_admitted += 1
                assert 0 < pay <= b * (1 + 1e-12)                                # IR
                crit.add(pay)
        assert len(crit) <= 1                                                     # one critical bid per job
    m.stream = S
    assert n_admitted > 0
    m2, res = run(world, live=True)
    assert res.summary["n_ir_violations"] == 0 and run_upper_set_violations(res, live=True) == 0


# --------------------------------------------------------------------------- #
# 3. sigma update
# --------------------------------------------------------------------------- #
def windows(S, n=3, size=250):
    return [np.arange(k * size, (k + 1) * size) for k in range(n)]


def test_sigma_update_on_a_markup_stream(world):
    cfg, ds, st0, S = world
    ws = windows(S)
    W = lambda idx: int(S.tick[idx[-1]] - S.tick[idx[0]] + 1)
    H_lm = log_markups(st0.bids.v_rate, st0.bids.cost_base)
    assert st0.myerson.sigma == pytest.approx(np.std(H_lm))                 # Phase 0: fit on H, in every mode
    for mode in ("phase", "cumulative", "fixed"):
        c = with_(cfg, sigma=mode)
        st, pool = st0, [H_lm]
        for idx in ws:
            pool.append(log_markups(S.v[idx], S.cost_base[idx]))
            st, dg = build_window_state(S, idx, idx, W(idx), st, ds, c,
                                        pooled_log_markups=np.concatenate(pool) if mode == "cumulative" else None)
            if mode == "fixed":
                assert st.myerson.sigma == st0.myerson.sigma and not dg.sigma_refit
            elif mode == "phase":
                assert st.myerson.sigma == pytest.approx(fit_sigma(S.v[idx], S.cost_base[idx]), rel=1e-12)
            else:
                assert st.myerson.sigma == pytest.approx(np.std(np.concatenate(pool)), rel=1e-12)
    with pytest.raises(ValueError):                                         # cumulative needs the pool
        build_window_state(S, ws[0], ws[0], W(ws[0]), st0, ds, with_(cfg, sigma="cumulative"))


@pytest.mark.parametrize("mode", ["fixed", "cumulative", "phase"])
def test_sigma_update_in_the_online_run(world, mode):
    cfg, ds, st0, S = world
    m, res = run(world, sigma=mode)
    ph = res.phases
    assert len(ph) >= 2
    H_lm = log_markups(st0.bids.v_rate, st0.bids.cost_base)
    arr_m = np.array([a["m"] for a in res.arrivals])
    for p in ph[:-1]:
        idx = np.flatnonzero(arr_m <= p["m"])                             # every arrival up to this phase's end
        last = np.flatnonzero(arr_m == p["m"])
        if mode == "fixed":
            assert p["next_sigma"] == st0.myerson.sigma
        elif mode == "cumulative":
            assert p["next_sigma"] == pytest.approx(np.std(np.r_[H_lm, log_markups(S.v[idx], S.cost_base[idx])]),
                                                    rel=1e-12)
        elif last.size >= cfg.phase_transition.n_min:
            assert p["next_sigma"] == pytest.approx(fit_sigma(S.v[last], S.cost_base[last]), rel=1e-12)
    assert res.summary["n_ir_violations"] == 0


# --------------------------------------------------------------------------- #
# 4. statistics by end reason
# --------------------------------------------------------------------------- #
def _stats(st):
    return {"p": st.p, "c_col": st.c_col, **{k: st.col(getattr(st, k)) for k in STATS}}


def test_stats_merge_after_timeout_equals_the_pooled_phases(world):
    cfg, ds, st0, S = world
    w1, w2 = np.arange(0, 300), np.arange(300, 650)
    W1, W2 = int(S.tick[w1[-1]]), int(S.tick[w2[-1]] - S.tick[w1[-1]])
    rep, byr = with_(cfg, stats="replace"), with_(cfg, stats="by_end_reason")
    A, _ = build_window_state(S, w1, w1, W1, st0, ds, rep)
    for reason in ("timeout", "test1"):
        Mg, dg = build_window_state(S, w2, w2, W2, A, ds, byr, reason=reason)
        assert dg.stats_merged
        P, _ = build_window_state(S, np.r_[w1, w2], np.r_[w1, w2], W1 + W2, st0, ds, rep)
        np.testing.assert_allclose(Mg.p, P.p, rtol=1e-12)                   # total arrivals / total time
        ts = st0.type_space
        fresh1 = np.bincount(column_ids(ts, S.priority[w1], S.hw_idx[w1]), minlength=ts.n_cols) > 0
        for k, x in _stats(Mg).items():                                    # columns observed in phase 1
            if k != "p":
                np.testing.assert_allclose(np.asarray(x)[fresh1], np.asarray(_stats(P)[k])[fresh1], rtol=1e-10,
                                           err_msg=k)
        np.testing.assert_array_equal(Mg.n_col_obs, A.n_col_obs + np.bincount(
            column_ids(ts, S.priority[w2], S.hw_idx[w2]), minlength=ts.n_cols))
        threshold_kstar((Mg.r_tilde - Mg.v @ Mg.tau).reshape(ts.k_star, len(ts.priorities), ts.n_hw))   # upper set


@pytest.mark.parametrize("reason", ["test2", "test3"])
def test_stats_after_test2_or_3_are_the_last_phase_only(world, reason):
    cfg, ds, st0, S = world
    w1, w2 = np.arange(0, 300), np.arange(300, 650)
    W1, W2 = int(S.tick[w1[-1]]), int(S.tick[w2[-1]] - S.tick[w1[-1]])
    A, _ = build_window_state(S, w1, w1, W1, st0, ds, with_(cfg, stats="replace"))
    R, _ = build_window_state(S, w2, w2[::2], W2, A, ds, with_(cfg, stats="replace"), reason=reason)
    B, dg = build_window_state(S, w2, w2[::2], W2, A, ds, with_(cfg, stats="by_end_reason"), reason=reason)
    assert not dg.stats_merged
    for k, x in _stats(B).items():
        np.testing.assert_array_equal(x, _stats(R)[k], err_msg=k)
    np.testing.assert_array_equal(B.r_tilde, R.r_tilde)


def test_stats_merge_keeps_empty_columns_and_the_upper_set(world):
    cfg, ds, st0, S = world
    ts = st0.type_space
    w = np.arange(0, 25)                                                    # a short window: some columns empty
    obs = w[::3]
    B, dg = build_window_state(S, w, obs, int(S.tick[w[-1]]), st0, ds, with_(cfg, stats="by_end_reason"),
                               reason="timeout")
    has_d = np.bincount(column_ids(ts, S.priority[obs], S.hw_idx[obs]), minlength=ts.n_cols) > 0
    has_c = np.bincount(column_ids(ts, S.priority[w], S.hw_idx[w]), minlength=ts.n_cols) > 0
    assert (~has_d).any() and (~has_c).any()
    for k in STATS:                                                         # empty columns carry forward
        np.testing.assert_array_equal(B.col(getattr(B, k))[~has_d], st0.col(getattr(st0, k))[~has_d])
    np.testing.assert_array_equal(B.c_col[~has_c], st0.c_col[~has_c])
    np.testing.assert_array_equal(B.n_col_obs[~has_d], st0.n_col_obs[~has_d])
    assert np.all(B.src_dur[has_d] == FRESH) and np.all(B.src_dur[~has_d & (st0.src_dur == FRESH)] == CARRY)
    # a column merged with H's statistics lies between the two (count-weighted)
    both = has_d & (st0.n_col_obs > 0)
    lo = np.minimum(st0.col(st0.d_bar), B.col(B.d_bar))
    assert both.any() and np.all(B.col(B.d_bar)[both] >= lo[both] - 1e-9)
    margin = (B.r_tilde - B.v @ B.tau).reshape(ts.k_star, ts.n_cols)
    assert np.all(np.diff(margin, axis=0) >= 0)
    threshold_kstar(margin.reshape(ts.k_star, len(ts.priorities), ts.n_hw))


def test_stats_by_end_reason_in_the_online_run(world):
    m, res = run(world, stats="by_end_reason")
    for p in res.phases[:-1]:
        assert p["reinit_stats_merged"] == (p["reason"] in ("timeout", "test1"))
    assert res.summary["n_ir_violations"] == 0 and run_upper_set_violations(res) == 0


def test_hindsight_benchmarks_ignore_the_switches(world):
    cfg, ds, st0, S = world
    c = with_(cfg, sigma="fixed", stats="by_end_reason")
    h = hindsight_cfg(c)
    assert h.bid.sigma_update == "phase" and h.stats.update == "replace" and h.phase_transition.n_min == 2
    idx = np.arange(400)
    a, _ = build_window_state(S, idx, idx, int(S.tick[idx[-1]]), st0, ds, hindsight_cfg(cfg))
    b, _ = build_window_state(S, idx, idx, int(S.tick[idx[-1]]), st0, ds, h)
    np.testing.assert_array_equal(a.r_tilde, b.r_tilde)


# --------------------------------------------------------------------------- #
# 5. the ablation runner (plan and configs only; nothing is run)
# --------------------------------------------------------------------------- #
def test_ablation_runner_plan_and_configs():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_ablations as ra
    single = ra.plan(True, False)
    assert [e for _, e in single] == [(k,) for k in ra.ELEMENTS]
    cum = ra.plan(False, True)
    assert [e for _, e in cum] == [ra.CUMULATIVE[:k] for k in range(1, 5)]
    assert ra.CUMULATIVE == ("thinning_off", "stats_by_end_reason", "sigma_cumulative", "live_oracle")
    c = ra.config_for(ra.CUMULATIVE)
    assert c.ablation.thinning_off and c.stats.update == "by_end_reason" and c.bid.sigma_update == "cumulative"
    assert c.oracle.kind == "live"
    base = ra.config_for(())
    assert not base.ablation.thinning_off and base.stats.update == "replace" and base.bid.sigma_update == "phase"
    assert base.oracle.kind == "frozen" and ra.config_for(("sigma_fixed",)).bid.sigma_update == "fixed"
    assert ra.config_for((), seed=7).run.seed == 7


def test_ablation_report_tables_on_fake_runs():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_ablations as ra
    runs = ra.plan(True, True)

    def fake(name, elements, reward):
        return dict(name=name, elements=elements, f=0.35, seed=42, runtime_s=60.0,
                    split={"customer value": 2 * reward, "profit": reward, "carbon": -2 * reward, "reward": reward},
                    social_welfare=0.5 * reward, payments=reward, myerson=dict(expected_profit=reward,
                                                                              realised_profit=0.99 * reward),
                    admitted=100, rejections={r: 1 for r in ra.REASONS}, n_phases=3,
                    end_reasons={"test1": 2, "horizon": 1}, ir_violations=0, upper_set_violations=0,
                    spot_check={"violations": 0})
    res = {n: fake(n, e, 100.0 + k) for k, (n, e) in enumerate(runs)}
    base = dict(theory=dict(reward=100.0), no_learning=90.0, fluid=200.0)
    text = ra.report(runs, 42, res=res, base=base)
    assert "## Each element alone (--single)" in text and "## Added one by one (--cumulative)" in text
    assert "delta of this step $" in text and "% of fluid OPT" in text and "delta vs no-learning $" in text
    row = ra.row(res["cum4_live_oracle"], base)
    assert row["delta vs theoretical $"] == 8.0 and row["% of fluid OPT"] == 54.0

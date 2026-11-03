"""Non-stationary bids (week 34, dlent_rt.scenarios) and the markup model with mu-hat: the scenario changes only the
stream's bids after t*; H and Phase 0 never see it; the no-learning model (FrozenModel) never re-initialises. Small
synthetic trace (3,000 jobs over 36 h, t* = 10 h): a few seconds."""
import warnings
from dataclasses import fields, replace

import numpy as np
import pytest

from dlent_rt import scenarios as sc
from dlent_rt.bid import MyersonModel, bid_model, log_markups
from dlent_rt.data import NoStream, load_dataset
from dlent_rt.models.core.stream import build_stream
from dlent_rt.models.theory import FrozenModel, TheoryModel
from dlent_rt.phase0 import run_phase0
from tests.synthetic import synthetic_config

ENVS = ("s1_bids_fall", "s2_bids_rise", "s6_bids_spread")


def with_scenario(cfg, name, t_star=10.0):
    return replace(cfg, scenario=replace(cfg.scenario, name=name, t_star_hours=t_star)).validate()


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = synthetic_config(tmp_path_factory.mktemp("sc"), n0=400, n_jobs=3000)
        cfg = replace(cfg, logging=replace(cfg.logging, progress_every=0.0, progress_phases=False)).validate()
        ds = load_dataset(cfg)
        return cfg, ds, build_stream(ds, cfg)


def test_mu_hat_and_sigma_hat_are_estimated():
    rng = np.random.default_rng(0)
    c = rng.uniform(0.02, 0.2, 50_000)
    v = c * np.exp(rng.normal(0.3, 0.4, c.size))
    m = bid_model(None, v, c)
    lm = log_markups(v, c)
    assert m.mu == pytest.approx(lm.mean()) and m.sigma == pytest.approx(lm.std())
    assert m.mu == pytest.approx(0.3, abs=0.01) and m.sigma == pytest.approx(0.4, abs=0.01)
    assert MyersonModel.known_markup(0.3).mu == pytest.approx(-0.045)               # the generator


def test_scenario_none_is_the_identity(world):
    cfg, ds, S = world
    assert sc.apply_scenario(S, cfg) is S
    S2 = build_stream(ds, with_scenario(cfg, "none"))
    for f in fields(S):
        np.testing.assert_array_equal(getattr(S2, f.name), getattr(S, f.name))


@pytest.mark.parametrize("name", ENVS)
def test_only_post_change_bids_move(world, name):
    cfg, ds, S = world
    c = with_scenario(cfg, name)
    S2 = build_stream(ds, c)
    post = sc.post_change(S.tick, c)
    assert 100 < post.sum() < S.n - 100
    np.testing.assert_array_equal(S2.v[~post], S.v[~post])                       # bids before t* unchanged
    assert np.all(S2.v[post] != S.v[post])
    for f in fields(S):                                                            # nothing else changes
        if f.name not in ("v", "phi_gen"):
            np.testing.assert_array_equal(getattr(S2, f.name), getattr(S, f.name), err_msg=f.name)
    # after t*: mean and std of ln m match the target within sampling error (4 standard errors)
    lm = log_markups(S2.v[post], S2.cost_base[post])
    tr, n = sc.truth(c), post.sum()
    assert abs(lm.mean() - tr["mu1"]) < 4 * tr["sigma1"] / np.sqrt(n)
    assert abs(lm.std() - tr["sigma1"]) < 4 * tr["sigma1"] / np.sqrt(2 * n)
    if name == "s6_bids_spread":
        assert np.mean(np.exp(lm)) == pytest.approx(1.0, abs=4 * 0.53 / np.sqrt(n))   # E[m] = 1


def test_presets_and_truth():
    from dlent_rt.config import default_config
    cfg = default_config()
    assert cfg.scenario.name == "none" and cfg.scenario.t_star_hours == 240.0
    t = sc.truth(with_scenario(cfg, "s1"))
    assert t["mu1"] == pytest.approx(-0.045 + np.log(0.5)) and t["sigma1"] == 0.3
    assert sc.truth(with_scenario(cfg, "s2"))["mu1"] == pytest.approx(-0.045 + np.log(1.75))
    t6 = sc.truth(with_scenario(cfg, "s6"))
    assert t6["sigma1"] == 0.5 and t6["mu1"] == pytest.approx(-0.125)
    with pytest.raises(ValueError):
        with_scenario(cfg, "s9")


@pytest.mark.parametrize("name", ENVS)
def test_H_and_phase0_identical_under_every_scenario(world, name):
    cfg, ds, S = world
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = with_scenario(cfg, name)
        a = run_phase0(ds, cfg)
        b = run_phase0(load_dataset(c), c)
        g = run_phase0(replace(load_dataset(c), stream=NoStream()), c)            # never reads the stream
    for st in (b, g):
        assert st.u0 == a.u0 and st.myerson == a.myerson and st.carbon_price == a.carbon_price
        np.testing.assert_array_equal(st.r_tilde, a.r_tilde)
        np.testing.assert_array_equal(st.tau, a.tau)
        np.testing.assert_array_equal(st.grid.bnd_v, a.grid.bnd_v)


def test_frozen_model_never_reinitialises(world, monkeypatch):
    import dlent_rt.models.theory as th
    cfg, ds, S = world
    c = replace(with_scenario(cfg, "s1"), ablation=replace(cfg.ablation, thinning_off=True),
                oracle=replace(cfg.oracle, kind="live")).validate()

    def no_reinit(*a, **k):
        raise AssertionError("ReInit called by the no-learning model")
    monkeypatch.setattr(th, "build_window_state", no_reinit)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st0 = run_phase0(ds, c)
        m = FrozenModel(ds, st0, c)
        res = m.run()
    assert len(res.phases) == 1 and res.phases[0]["reason"] == "horizon"
    assert m.state is st0 and res.phases[0]["sigma"] == st0.myerson.sigma        # Phase 0's state all run long
    assert all(a["p_U"] == 1.0 for a in res.arrivals) and res.summary["n_ir_violations"] == 0
    assert any(a["admit"] for a in res.arrivals)


def test_learning_model_follows_the_new_markups(world):
    """S1 on the synthetic trace: a phase that starts after t* uses a mu-hat near the new level."""
    cfg, ds, S = world
    c = with_scenario(cfg, "s1", t_star=8.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = TheoryModel(ds, run_phase0(ds, c), c).run()
    import pandas as pd
    ph = pd.DataFrame(res.phases)
    tab = sc.markup_per_phase(ph, c)
    late = tab[(tab["start h"] >= 8.0 + 2.0)]
    assert len(late), tab
    assert np.all(np.abs(late.mu_hat - late["true mu"]) < 0.15)
    d = sc.detection(ph, c)
    assert set(d) == {"delay_test23_h", "test23", "delay_any_h", "first_reason"}


def test_diagnostics_on_hand_made_phases():
    import pandas as pd
    from dlent_rt.config import default_config
    cfg = with_scenario(default_config(), "s1", t_star=10.0)
    h = 3600
    ph = pd.DataFrame(dict(m=[1, 2, 3, 4], tau_start=[1, 5 * h + 1, 12 * h + 1, 20 * h + 1],
                           tau_end=[5 * h, 12 * h, 20 * h, 30 * h], reason=["timeout", "test1", "test2", "horizon"],
                           sigma=0.3, mu=-0.045, tau_cpu=0.0, tau_ram=1e-3))
    d = sc.detection(ph, cfg)
    assert d["delay_any_h"] == 2.0 and d["first_reason"] == "test1"
    assert d["delay_test23_h"] == 10.0 and d["test23"] == "test2"
    tick = np.arange(1, 25) * h
    pre, post = sc.before_after(np.ones(24), tick, cfg)
    assert pre + post == 24 and post == np.sum(tick - 1 >= 10 * h)
    tab = sc.markup_per_phase(ph, cfg)
    assert list(tab["contains t*"]) == [False, True, False, False]
    assert tab["true mu"].iloc[2] == pytest.approx(-0.045 + np.log(0.5))


@pytest.mark.parametrize("name", ENVS)
def test_post_change_scoring_matches_the_true_markups(world, name):
    """The realised reward of a post-change job uses Phase 0's markup model moved by the scenario: it matches the
    model fitted on the post-change markups (within sampling error); pre-change jobs keep Phase 0's model."""
    cfg, ds, S = world
    c = with_scenario(cfg, name)
    S2 = build_stream(ds, c)
    post = sc.post_change(S2.tick, c)
    m0 = MyersonModel.fitted(log_markups(S.v[post], S.cost_base[post]))        # the pre-change law, fitted
    mp = sc.scoring_model_post(m0, c)
    fit = MyersonModel.fitted(log_markups(S2.v[post], S2.cost_base[post]))     # the same jobs after the change
    assert mp.mu == pytest.approx(fit.mu, abs=1e-9) and mp.sigma == pytest.approx(fit.sigma, rel=1e-9)
    from dlent_rt.reward import job_terms
    st0 = run_phase0(ds, c)
    J = sc.scored_job_terms(S2, st0.myerson, st0.carbon_price, c.objective, c)
    args = (S2.v, S2.cost_base, S2.duration_hours, S2.C_elec, S2.C_carbon)
    pre = job_terms(*args, st0.myerson, st0.carbon_price, c.objective)
    np.testing.assert_array_equal(J.R[~post], pre.R[~post])
    np.testing.assert_array_equal(J.R[post], job_terms(*args, sc.scoring_model_post(st0.myerson, c),
                                                       st0.carbon_price, c.objective).R[post])


def test_none_scoring_is_the_old_scoring(world):
    from dlent_rt.reward import job_reward
    cfg, ds, S = world
    st0 = run_phase0(ds, cfg)
    a, b = sc.scored_job_terms(S, st0.myerson, st0.carbon_price, cfg.objective, cfg), job_reward(S, st0, cfg.objective)
    for f in ("R", "phi", "cv", "profit", "sw"):
        np.testing.assert_array_equal(getattr(a, f), getattr(b, f))


def test_scenario_runner_on_the_synthetic_trace(world, tmp_path, monkeypatch):
    """scripts/run_scenarios.py: the five models of one environment, cached and resumable."""
    import pickle, sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_scenarios as rs
    cfg, ds, S = world
    orig = rs.config_for

    def cfg_for(env, model, seed=None):
        x = orig(env, model, seed)
        return replace(cfg, scenario=replace(x.scenario, t_star_hours=10.0), ablation=x.ablation, oracle=x.oracle,
                       stats=x.stats).validate()
    monkeypatch.setattr(rs, "config_for", cfg_for)
    monkeypatch.setattr(rs, "CACHE", tmp_path)
    assert rs.plan(rs.ENVS, rs.MODELS)[:5] == [("none", m) for m in rs.MODELS]
    p = orig("s1", "practical")
    assert p.ablation.thinning_off and p.oracle.kind == "live" and p.stats.update == "by_end_reason"
    t = orig("s1", "theoretical")
    assert not t.ablation.thinning_off and t.oracle.kind == "frozen" and t.stats.update == "replace"
    n = orig("s1", "no_learning")
    assert n.ablation.thinning_off and n.oracle.kind == "live" and n.scenario.name == "s1_bids_fall"
    for m in rs.MODELS:
        assert rs.run_one("s1_bids_fall", m).endswith("ok")
        assert rs._cached("s1_bids_fall", m, None)
    res = {m: pickle.load(open(rs.path("s1_bids_fall", m), "rb")) for m in rs.MODELS}
    for m in rs.MODELS[1:]:
        r = res[m]
        assert r["ir"] == 0 and r["upper"] == 0 and r["admitted"] > 0, m
        assert r["reward"] == pytest.approx(r["arrays"]["R"].sum())
    assert len(res["no_learning"]["phases"]) == 1 and len(res["practical"]["phases"]) >= 2
    assert res["fluid"]["reward"] == pytest.approx(res["fluid"]["reward_pre"] + res["fluid"]["reward_post"])

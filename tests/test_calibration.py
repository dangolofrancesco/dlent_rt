"""Calibrated thresholds of Tests 2-3 (week 34, tests.threshold = "calibrated"; models.core.calibration): kappa = 1 is
the paper's threshold bit for bit; kappa is chosen on H alone so that the largest ratio on H is the target; no
stream data is read. Small synthetic trace (3,000 jobs over 36 h), practical switches: a few seconds per run."""
import sys
import warnings
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from dlent_rt.config import default_config
from dlent_rt.data import NoStream, load_dataset
from dlent_rt.models.core import calibration as cal
from dlent_rt.models.core.error_params import make_error_params
from dlent_rt.models.theory import TheoryModel
from dlent_rt.phase0 import run_phase0
from tests.synthetic import synthetic_config


def with_tests(cfg, **kw):
    return replace(cfg, tests=replace(cfg.tests, **kw)).validate()


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = synthetic_config(tmp_path_factory.mktemp("cal"), n0=400, n_jobs=3000)
        cfg = replace(cfg, logging=replace(cfg.logging, progress_every=0.0, progress_phases=False),
                      ablation=replace(cfg.ablation, thinning_off=True), oracle=replace(cfg.oracle, kind="live"),
                      stats=replace(cfg.stats, update="by_end_reason")).validate()
        ds = load_dataset(cfg)
        st0 = run_phase0(ds, cfg)
        tr, phases = cal.replay_history(ds, st0, cfg)
        return cfg, ds, st0, tr, phases


def run(ds, st0, cfg):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = TheoryModel(ds, st0, cfg)
        return m, m.run()


def test_kappa_scales_the_thresholds(world):
    cfg, ds, st0, _, _ = world
    ep = make_error_params(st0, ds, cfg)
    k = replace(ep, kappa2=0.01, kappa3=0.002)
    n = np.array([10.0, 3600.0, 86400.0])
    np.testing.assert_allclose(k.eps_test2(3600, n), 0.01 * ep.eps_test2(3600, n), rtol=1e-14)
    np.testing.assert_allclose(k.eps_test3(n), 0.002 * ep.eps_test3(n), rtol=1e-14)
    np.testing.assert_allclose(k.lam_tilde_term(n), 0.002 * ep.lam_tilde_term(n), rtol=1e-14)
    one = replace(ep, kappa2=1.0, kappa3=1.0)
    assert np.array_equal(one.eps_test2(3600, n), ep.eps_test2(3600, n)) and np.array_equal(one.eps_test3(n),
                                                                                             ep.eps_test3(n))


def test_kappa_one_is_the_theoretical_threshold(world):
    """tests.threshold calibrated with kappa = 1: the same run as the theoretical thresholds, bit for bit."""
    cfg, ds, st0, _, _ = world
    _, a = run(ds, st0, cfg)
    m, b = run(ds, st0, with_tests(cfg, threshold="calibrated", kappa_test2=1.0, kappa_test3=1.0))
    assert m.calibration["source"] == "config" and m.ep.kappa2 == m.ep.kappa3 == 1.0
    assert [(p["tau_end"], p["reason"]) for p in a.phases] == [(p["tau_end"], p["reason"]) for p in b.phases]
    for f in ("R", "admit", "payment_rate", "k_star"):
        np.testing.assert_array_equal([x[f] for x in a.arrivals], [x[f] for x in b.arrivals])


def test_kappa_puts_the_largest_ratio_on_H_at_the_target(world):
    cfg, ds, st0, tr, phases = world
    assert phases[-1]["reason"] == "horizon" and phases[-1]["tau_end"] == ds.H_span_steps
    assert tr.tick.size == np.unique(tr.tick).size                      # every H tick traced once
    k = cal.kappas_from_trace(tr, 0.5)
    assert np.max(tr.ratio2(k["kappa2"])) == pytest.approx(0.5, rel=1e-12)
    assert np.max(tr.ratio3(k["kappa3"])) == pytest.approx(0.5, rel=1e-8)
    assert k["kappa2"] < 1 and k["kappa3"] < 1                          # the paper's thresholds are far away
    assert np.max(tr.ratio3(1.0)) == pytest.approx(k["max_ratio3_theory"])
    assert np.all(np.diff([np.max(tr.ratio3(x)) for x in (1e-4, 1e-3, 1e-2, 1e-1, 1.0)]) < 0)


def test_calibration_reads_no_stream_data(world, monkeypatch):
    """kappa from H alone: the same with the stream removed (any access raises), with a scenario that changes the
    stream's bids, and build_stream is never called."""
    import dlent_rt.models.theory as th
    cfg, ds, st0, _, _ = world
    ref = cal.calibrate(ds, st0, with_tests(cfg, threshold="calibrated"))

    def no_stream(*a, **k):
        raise AssertionError("the calibration built the online stream")
    monkeypatch.setattr(th, "build_stream", no_stream)
    blind = cal.calibrate(replace(ds, stream=NoStream()), st0, with_tests(cfg, threshold="calibrated"))
    s2 = replace(cfg, scenario=replace(cfg.scenario, name="s2_bids_rise", t_star_hours=0.0))
    moved = cal.calibrate(ds, st0, with_tests(s2, threshold="calibrated"))
    for d in (blind, moved):
        assert (d["kappa2"], d["kappa3"]) == (ref["kappa2"], ref["kappa3"])
    assert ref["source"] == "H"


def test_calibrated_tests_on_the_stationary_stream(world):
    """The calibrated run: kappa from H, used by the monitor; on this stationary trace the calibrated tests fire at
    most in a few phases (reported), and far smaller kappas make them fire (the switch acts)."""
    cfg, ds, st0, tr, _ = world
    m, res = run(ds, st0, with_tests(cfg, threshold="calibrated"))
    k = cal.kappas_from_trace(tr, 0.5)
    assert (m.ep.kappa2, m.ep.kappa3) == (k["kappa2"], k["kappa3"]) and res.summary["calibration"]["source"] == "H"
    ends = res.summary["phase_end_reasons"]
    n23 = ends["test2"] + ends["test3"]
    print(f"\ncalibrated tests on the synthetic stationary stream: {n23} of {len(res.phases)} phases end by "
          f"Test 2 / 3 ({ends})")
    assert n23 <= len(res.phases) // 2
    _, low = run(ds, st0, with_tests(cfg, threshold="calibrated", kappa_test2=k["kappa2"] / 1e3,
                                    kappa_test3=k["kappa3"] / 1e3))
    e = low.summary["phase_end_reasons"]
    assert e["test2"] + e["test3"] > 0


def test_runner_extra_model_and_old_caches():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_scenarios as rs
    assert "practical_calibrated" not in rs.MODELS and rs.plan(rs.ENVS, rs.MODELS) == [
        (e, m) for e in rs.ENVS for m in rs.MODELS]
    c, p = rs.config_for("s1", "practical_calibrated"), rs.config_for("s1", "practical")
    assert c.tests.threshold == "calibrated" and replace(c, tests=p.tests) == p
    d = asdict(default_config())
    old = asdict(p)
    for f in ("threshold", "kappa_test2", "kappa_test3", "calibration_target"):    # a cache from before the switch
        del old["tests"][f]
    assert rs._same_config(old, asdict(p), d) and not rs._same_config(old, asdict(c), d)

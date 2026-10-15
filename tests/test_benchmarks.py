"""dlent_rt.benchmarks: score / fluid_opt / job_level_opt / regret on the synthetic online run."""
import tempfile, warnings
from pathlib import Path

import numpy as np
import pytest

from dlent_rt.benchmarks import Benchmarks, regret
from dlent_rt.data import load_dataset
from dlent_rt.phase0 import run_phase0
from dlent_rt.models.theory import TheoryModel
from tests.synthetic import synthetic_config


@pytest.fixture(scope="module")
def run():
    with tempfile.TemporaryDirectory() as d, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = synthetic_config(Path(d), n0=400, n_jobs=3000, drift=True)
        ds = load_dataset(cfg)
        st = run_phase0(ds, cfg)
        m = TheoryModel(ds, st, cfg)
        res = m.run()
    return m, res, Benchmarks.from_model(m)


def test_score_and_fluid_match_the_online_loop(run):
    m, res, b = run
    adm = np.zeros(m.stream.n, bool)
    adm[[a["i"] for a in res.arrivals if a["admit"]]] = True
    assert b.score(adm) == pytest.approx(res.summary["R_total"], rel=1e-12)
    assert b.fluid_opt() == pytest.approx(res.summary["bench_global"], rel=1e-12)
    assert regret(b.score(adm), b.fluid_opt()) == pytest.approx(res.summary["regret_global"], rel=1e-9)


def test_job_level_opt_is_feasible_and_bounded(run):
    m, res, b = run
    o = b.job_level_opt(time_limit=60)
    S, x = b.S, o["x"]
    assert np.all(b.R[x] > 0)
    assert o["R_ip"] <= o["R_lp"] + 1e-6 <= o["R_nocap"] + 2e-6
    assert o["R_nocap"] == pytest.approx(float(b.R[b.R > 0].sum()))
    # occupancy of the chosen jobs at every arrival tick <= c (independent of the MILP matrix)
    t, D, A = S.tick[x], S.D[x], S.A[x]
    active = (t[None, :] <= S.tick[:, None]) & (S.tick[:, None] < (t + D)[None, :])
    assert np.all(active.astype(float) @ A <= b.c[None, :] + 1e-9)

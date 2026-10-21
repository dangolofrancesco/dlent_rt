"""Bucketed job-level MILP (Benchmarks.job_opt_bucketed) and its building blocks."""
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest

from dlent_rt.benchmarks import Benchmarks, occupancy_milp, peak_occupancy
from dlent_rt.data import load_dataset
from dlent_rt.phase0 import run_phase0
from tests.synthetic import synthetic_config


def test_bucketed_differs_from_job_level_on_tiny_instance():
    # c = 1 and two overlapping unit jobs: only one fits. Job rewards R = (3, 1), but the jobs'
    # types believe (1.5, 2): the job-level MILP takes job 0, the bucketed one takes job 1.
    t, D, A, c = np.array([1, 2]), np.array([10, 10]), np.ones((2, 1)), np.array([1.0])
    R, r_type = np.array([3.0, 1.0]), np.array([1.5, 2.0])
    job = occupancy_milp(t, D, A, R, c, time_limit=10)
    buck = occupancy_milp(t, D, A, r_type, c, time_limit=10)
    assert job["x"].tolist() == [True, False] and job["ip_obj"] == pytest.approx(3.0)
    assert buck["x"].tolist() == [False, True] and buck["ip_obj"] == pytest.approx(2.0)
    assert R[buck["x"]].sum() == 1.0 <= R[job["x"]].sum()          # realised bucketed <= job OPT
    # without overlap (job 1 arrives after job 0 has left) both are taken by both MILPs
    t2 = np.array([1, 11])
    assert occupancy_milp(t2, D, A, r_type, c, 10)["x"].all()


def test_peak_occupancy_releases_before_admissions():
    # [1, 5) and [5, 9): job 0 leaves at tick 5, when job 1 arrives -> peak 1, not 2
    assert peak_occupancy([1, 5], [4, 4], np.ones((2, 1))).tolist() == [1.0]
    assert peak_occupancy([1, 4], [4, 4], np.ones((2, 1))).tolist() == [2.0]


@pytest.fixture(scope="module")
def bench():
    with tempfile.TemporaryDirectory() as d, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = synthetic_config(Path(d), n0=400, n_jobs=3000, drift=True)
        ds = load_dataset(cfg)
        st = run_phase0(ds, cfg)
        return Benchmarks(ds, st, cfg)


@pytest.mark.parametrize("source", ["phase0", "reference"])
def test_bucketed_variant_A_bounded_by_job_opt(bench, source):
    job = bench.job_level_opt(time_limit=60)
    a = bench.job_opt_bucketed(source, "true", time_limit=60)
    assert a["feasible_true"]                                       # same capacity rows as job OPT
    assert a["R_real"] <= job["R_ip"] * (1 + max(job["gap"], 0)) + 1e-9
    assert a["R_believed"] <= a["R_lp_believed"] + 1e-6
    b = bench.bucket_types(source)
    assert np.all(b["r"][a["x"]] > 0) and np.all(b["seen"][a["x"]])
    if source == "reference":
        assert a["n_unseen_jobs"] == 0                              # the reference is fit on the stream


def test_unseen_phase0_types_have_zero_reward(bench):
    b = bench.bucket_types("phase0")
    st = bench.state0
    unpop = st.p == 0
    assert np.all(st.r_tilde[unpop] == 0)                           # Phase 0 itself gives r~ = 0
    assert np.all(b["r"][~b["seen"]] == 0)


def test_bucketed_variant_B_runs(bench):
    o = bench.job_opt_bucketed("phase0", "type", time_limit=60)
    assert o["R_believed"] <= o["R_lp_believed"] + 1e-6
    assert len(o["peak_true_occ_over_c"]) == 2

"""Benchmarks.ladder (week 32): the whole benchmark ladder from tracked package code, on the synthetic trace."""
import warnings

import numpy as np
import pytest

from dlent_rt.benchmarks import Benchmarks
from dlent_rt.data import load_dataset
from dlent_rt.lp_oracle import solve_lp_rs
from dlent_rt.models.policy import run_static
from dlent_rt.models.theory import TheoryModel
from dlent_rt.phase0 import run_phase0
from tests.synthetic import synthetic_config

ROWS = ["job-level, no capacity", "feasible ceiling", "job-level MILP", "bucketed MILP A (reference buckets)",
        "bucketed MILP A (Phase 0 buckets)", "fluid OPT", "fluid OPT on feasible types", "G2 job greedy",
        "G1 type greedy", "static Phase 0", "static Phase 0 + thinning", "DLENT-RT"]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = synthetic_config(tmp_path_factory.mktemp("lad"), n0=400, n_jobs=1500)
        ds = load_dataset(cfg)
        st = run_phase0(ds, cfg)
        model = TheoryModel(ds, st, cfg)
        res = model.run()
        B = Benchmarks.from_model(model)
        return B, model, res, B.ladder(model, res, time_limit=60.0)


def test_ladder_rows_and_bounds(run):
    B, model, res, lad = run
    r = lad["rows"]
    assert list(r) == ROWS
    assert r["DLENT-RT"] == pytest.approx(res.summary["R_total"], rel=1e-10)
    assert r["feasible ceiling"] <= r["job-level, no capacity"] + 1e-9
    assert r["job-level MILP"] <= r["job-level, no capacity"] + 1e-6
    for k in ("G2 job greedy", "G1 type greedy", "static Phase 0", "static Phase 0 + thinning", "DLENT-RT",
              "job-level MILP"):
        assert r[k] <= r["feasible ceiling"] + 1e-6, k
    assert 0.0 <= lad["g1_decision_error"] <= 1.0


def test_static_thinning_uses_the_cap_and_common_numbers(run):
    B, model, res, lad = run
    p = (1 - model.ep.xi) / (1 + model.ep.eps_D)
    assert lad["p_cap"] == pytest.approx(p)
    sim = run_static(B.S, B.c, B.state0, thin_p=p, u_rand=model.u_rand)
    assert B.score(sim.admit) == lad["rows"]["static Phase 0 + thinning"]


def test_fluid_on_feasible_types(run):
    B, model, res, lad = run
    ref = B.reference_state
    feas = ~np.any(ref.A_type > B.c[None, :], axis=1)
    lp = solve_lp_rs(p=ref.p * feas, r=ref.r_tilde, v=ref.v, c=B.c, presolve=B.cfg.lp.presolve)
    assert lad["rows"]["fluid OPT on feasible types"] == pytest.approx(lp.optimum * B.T, rel=1e-12)
    assert lad["fluid_feasible"]["n_types_removed"] == int((~feas & (ref.p > 0)).sum())

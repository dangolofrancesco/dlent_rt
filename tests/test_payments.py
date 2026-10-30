"""Payments (D6) and the grid edges (D4): the bins cover every bid (bin 1 = [0, L_1), bin K* = [L_K*-1, +inf));
allocation and payment use the same edges; every admitted job pays the hourly p = max(L_k*, reserve), 0 < p <= b (IR),
also when its bid is below the grid's v_min or above its v_max, and p x the hours it runs in total."""
import tempfile, warnings
from pathlib import Path

import numpy as np
import pytest

from dlent_rt.data import load_dataset
from dlent_rt.grid import build_valuation_grid
from dlent_rt.models.theory import TheoryModel
from dlent_rt.phase0 import run_phase0
from tests.synthetic import synthetic_config


def test_bins_cover_every_bid_and_lower_edge_is_at_most_the_bid():
    v = np.exp(np.random.default_rng(0).normal(0, 1, 500))
    g = build_valuation_grid(v, 16)
    bids = np.r_[0.0, 1e-12, v.min() / 3, v, v.max() * 7, 1e12]
    k = g.assign(bids)
    assert k.min() == 1 and k.max() == g.k_star
    assert np.array_equal(g.assign([0.0, v.min() / 3]), [1, 1]) and g.assign(v.max() * 7)[0] == g.k_star
    L = g.lower_edges[k - 1]
    assert g.lower_edges[0] == 0.0 and np.all(L <= bids)
    # interior edges: the same ones that allocation uses
    np.testing.assert_array_equal(g.lower_edges[1:], g.bnd_v[1:g.k_star])
    upper = np.r_[g.bnd_v[1:g.k_star], np.inf][k - 1]
    assert np.all(bids < upper)


@pytest.fixture(scope="module")
def run():
    with tempfile.TemporaryDirectory() as d, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = synthetic_config(Path(d), n0=400, n_jobs=3000, drift=True)
        ds = load_dataset(cfg)
        st = run_phase0(ds, cfg)
        return TheoryModel(ds, st, cfg).run()


def test_every_admitted_job_pays_at_most_its_bid(run):
    adm = [a for a in run.arrivals if a["admit"]]
    assert adm
    assert all(a["ir"] and 0 < a["payment_rate"] <= a["v"] for a in adm)
    assert all(a["payment_rate"] == max(a["L_kstar"], a["reserve"]) for a in adm)
    assert all(a["payment"] == a["payment_rate"] * a["duration_hours"] for a in adm)
    assert run.summary["n_ir_violations"] == 0
    # bids outside the grid's [v_min, v_max] (built on H) arrive in the stream; IR holds for every admitted one
    off = [a for a in run.arrivals if a["below_grid"] or a["above_grid"]]
    assert off
    assert all(a["payment_rate"] <= a["v"] for a in off if a["admit"])
    assert all(a["k_bin"] == 1 for a in off if a["below_grid"])

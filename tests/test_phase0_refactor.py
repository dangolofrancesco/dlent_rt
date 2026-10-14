"""Regression: refactored Phase 0 == original loop implementation (with the E*CI fix applied to both)."""
import tempfile, warnings
from pathlib import Path
import numpy as np
import pytest

from dlent_rt.data import load_dataset
from dlent_rt.phase0 import run_phase0
from tests.synthetic import synthetic_config

old = pytest.importorskip("dlent_rt._phase0_pre_week28")


def test_phase0_refactor_matches_original():
    with tempfile.TemporaryDirectory() as d:
        cfg = synthetic_config(Path(d))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = load_dataset(cfg)
            a = run_phase0(ds, cfg)
            b = old.run_phase0(ds, cfg)
    for f in ["p", "d_bar", "v_bar", "phi_bar", "energy_bar", "power_bar", "r_tilde",
              "r_rev", "C_elec_bar", "C_carbon_bar", "v", "L", "phi_tilde", "tau"]:
        np.testing.assert_allclose(getattr(a, f), getattr(b, f), rtol=1e-10, atol=1e-14, err_msg=f)
    for f in ["lam_star", "gamma", "xi", "eps_D"]:
        assert np.isclose(getattr(a, f), getattr(b, f), rtol=1e-10), f
    assert a.C_carbon_bar[a.p > 0].std() > 0

"""The reward unit of Tests 2-3 (paper definition): r_max = the largest r~ over Phase 0's populated types with
r~ > 0. Negative types (never admitted) and types without jobs do not count."""
from types import SimpleNamespace
import tempfile, warnings
from pathlib import Path

import numpy as np

from dlent_rt.data import load_dataset
from dlent_rt.phase0 import reward_unit, run_phase0
from tests.synthetic import synthetic_config


def test_largest_positive_reward_ignoring_negative_types():
    st = SimpleNamespace(r_tilde=np.array([0.2, -50.0, 0.7, 3.0, -0.1, 0.5]),
                         p=np.array([0.1, 0.4, 0.2, 0.0, 0.3, 0.05]))     # 3.0 has no job (p = 0)
    assert reward_unit(st.r_tilde, st.p) == 0.7                          # not 50 (negative), not 3.0 (unpopulated)
    assert reward_unit(np.array([-1.0, -2.0]), np.array([0.5, 0.5])) == 1.0   # no positive type: unit 1


def test_phase0_state_stores_it_and_the_run_keeps_it():
    with tempfile.TemporaryDirectory() as d, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = synthetic_config(Path(d), n0=400, n_jobs=3000)
        ds = load_dataset(cfg)
        st = run_phase0(ds, cfg)
    pos = (st.p > 0) & (st.r_tilde > 0)
    assert st.r_max == st.r_tilde[pos].max()
    from dlent_rt.models.theory import TheoryModel
    assert TheoryModel(ds, st, cfg).r_unit == st.r_max                  # gamma forced: the monitor's unit

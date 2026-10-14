"""capacity.mode = "from_amax": c_i = a_max,i / xi_target, a_max over the kept jobs (H + stream)."""
import tempfile, warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from dlent_rt.config import AbsoluteCapCfg
from dlent_rt.data import load_dataset
from tests.synthetic import synthetic_config

GOLDEN = Path(__file__).parent / "data" / "golden_capacity_modes.npz"


def _load(mode, **cap):
    with tempfile.TemporaryDirectory() as d, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = synthetic_config(Path(d), n0=400, n_jobs=3000, drift=True)
        return load_dataset(replace(base, capacity=replace(base.capacity, mode=mode, **cap)))


@pytest.fixture(scope="module")
def ds_amax():
    return _load("from_amax", xi_target=0.5)


def test_capacity_is_amax_over_xi_target(ds_amax):                    # (a)
    a_max = np.maximum(ds_amax.H.A.max(axis=0), ds_amax.stream.A.max(axis=0))
    np.testing.assert_allclose(ds_amax.capacity, a_max / 0.5, rtol=1e-15)
    assert ds_amax.capacity.shape == (2,)
    ds_x = _load("from_amax", xi_target=0.2)
    np.testing.assert_allclose(ds_x.capacity, a_max / 0.2, rtol=1e-15)


@pytest.mark.parametrize("mode", ["fraction_of_volume", "fraction_of_peak",
                                  "fraction_of_concurrency_quantile", "absolute"])
def test_other_modes_unchanged(mode):                                 # (b)
    g = np.load(GOLDEN)
    # golden "absolute" was frozen with the YAML's former manual override (cpu 0.4, ram 0.3)
    ds = _load(mode, absolute=AbsoluteCapCfg(cpu=0.4, ram=0.3)) if mode == "absolute" else _load(mode)
    np.testing.assert_allclose(ds.capacity, g[mode], rtol=1e-12)
    assert ds.xi == pytest.approx(float(g[mode + "_xi"][0]), rel=1e-12)


def test_xi_data_equals_xi_target(ds_amax):                           # (c)
    # the loader's xi uses the STREAM a_max (data.py), capacity the H + stream a_max:
    # equality holds when the largest request of the binding resource is in the stream
    # (true on this dataset: drift=True doubles CPU in the second half).
    ratio = ds_amax.a_max / ds_amax.capacity
    i = int(np.argmax(ratio))
    assert ds_amax.stream.A[:, i].max() >= ds_amax.H.A[:, i].max()
    assert ds_amax.xi == pytest.approx(0.5, rel=1e-12)
    assert ratio[i] == pytest.approx(0.5, rel=1e-12)
    assert np.all(ratio <= 0.5 + 1e-12)

"""time.arrival_time_scale: arrivals compressed by k, order / durations / jobs / capacity unchanged."""
from dataclasses import replace

import numpy as np

from dlent_rt.data import load_dataset
from tests.synthetic import synthetic_config


def _ds(tmp_path, k):
    cfg = synthetic_config(tmp_path, n0=400, n_jobs=3000, drift=True)
    return load_dataset(replace(cfg, time=replace(cfg.time, arrival_time_scale=k)).validate())


def test_scale_one_is_off(tmp_path):
    a = _ds(tmp_path, 1.0)
    assert a.arrival_compression is None


def test_compression(tmp_path):
    base, ds = _ds(tmp_path, 1.0), _ds(tmp_path, 3.0)
    assert ds.arrival_compression["k"] == 3.0
    np.testing.assert_array_equal(ds.capacity, base.capacity)          # sized on the original ticks
    for a, b in ((ds.H, base.H), (ds.stream, base.stream)):
        assert a.n == b.n
        np.testing.assert_array_equal(a.duration_steps, b.duration_steps)
        np.testing.assert_array_equal(a.A, b.A)
        assert np.all(np.diff(a.arrival_step) > 0)                     # order kept, one arrival per tick
    abs_H = base.H.arrival_step
    # compressed absolute ticks = floor(original / 3) up to the collision nudges, never earlier
    assert np.all(ds.H.arrival_step >= abs_H // 3)
    assert ds.T < base.T and abs(ds.T - base.T / 3) <= ds.arrival_compression["max_nudge"] + 1
    assert ds.H_span_steps < base.H_span_steps

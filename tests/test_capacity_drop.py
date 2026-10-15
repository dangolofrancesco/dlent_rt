"""capacity.drop_jobs_exceeding_capacity: jobs with A_ij > c_i leave H and the stream, nothing else moves."""
from dataclasses import replace

import numpy as np

from dlent_rt.data import load_dataset
from tests.synthetic import synthetic_config


def _cfg(tmp_path, drop):
    cfg = synthetic_config(tmp_path, n0=400, n_jobs=3000, drift=True)
    cap = replace(cfg.capacity, mode="fraction_of_concurrency_quantile", fraction=0.02,
                  concurrency_quantile=0.7, drop_jobs_exceeding_capacity=drop)
    return replace(cfg, capacity=cap).validate()


def test_drop_jobs_exceeding_capacity(tmp_path):
    off = load_dataset(_cfg(tmp_path, False))
    on = load_dataset(_cfg(tmp_path, True))
    c = off.capacity
    big_H = np.any(off.H.A > c, axis=1)
    big_S = np.any(off.stream.A > c, axis=1)
    assert big_H.any() and big_S.any(), "the test config must have jobs larger than c"
    assert off.capacity_drop is None
    assert on.capacity_drop == dict(n_H=int(big_H.sum()), n_stream=int(big_S.sum()))
    # c, timeline, d_max unchanged; exactly the jobs that fit are kept, in order
    np.testing.assert_array_equal(on.capacity, c)
    assert (on.T, on.H_span_steps, on.d_max_steps) == (off.T, off.H_span_steps, off.d_max_steps)
    for a, b, keep in ((on.H, off.H, ~big_H), (on.stream, off.stream, ~big_S)):
        assert a.n == int(keep.sum())
        np.testing.assert_array_equal(a.A, b.A[keep])
        np.testing.assert_array_equal(a.arrival_step, b.arrival_step[keep])
        np.testing.assert_array_equal(a.hw_idx, b.hw_idx[keep])
        assert not np.any(a.A > c)
    assert on.n_arrivals_stream == on.stream.n
    np.testing.assert_array_equal(on.a_max, on.stream.A.max(axis=0))

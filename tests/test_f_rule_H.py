"""The f rule reads H only (week 33): Phase 0 and the f rule give identical outputs when the stream is NaN or empty,
and nothing on the f rule's path touches the stream (a stream object that raises on access). Small synthetic trace
(1,500 jobs), a few seconds."""
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from dlent_rt.data import NoStream, load_dataset, load_history
from dlent_rt.phase0 import run_phase0
from dlent_rt.theory_eval import f_rule_point, select_f
from tests.synthetic import synthetic_config

COLS = ("CPU", "RAM", "duration", "CPU_hours_real", "priority_tier", "scheduling_class")


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = synthetic_config(tmp_path_factory.mktemp("fH"), n0=400, n_jobs=1500)
        return replace(c, hardware=replace(c.hardware, fit_on="H")).validate()


def outputs(ds, cfg):
    """Everything the f rule and Phase 0 give from H: U0, sigma, grid, D_bar, C, Q, s_C, r_max, tau, the replay."""
    st = run_phase0(ds, cfg)
    return dict(U0=st.u0, sigma=st.myerson.sigma, grid=st.grid.bnd_v, s_C=st.carbon_price, r_max=st.r_max,
                tau=np.asarray(st.tau), r_tilde=st.r_tilde, D_bar=ds.capacity_diag["D_bar"], C=ds.capacity,
                Q=np.asarray(ds.limit_report.quota), point=f_rule_point(ds, cfg))


def assert_same(a, b):
    assert a.keys() == b.keys()
    for k in a:
        np.testing.assert_equal(a[k], b[k], err_msg=k)


def with_csv(cfg, tmp_path, change, name):
    """The config on a copy of the trace whose STREAM rows (arrivals at or after H's end) are changed by `change`."""
    df = pd.read_csv(cfg.data.batch_csv, dtype=str, keep_default_na=False)       # as text: H's rows stay byte-exact
    t = pd.to_datetime(df.arrival_utc)
    H = load_history(cfg).H
    end = pd.Timestamp(H.datetime[0]).tz_localize("UTC") + pd.Timedelta(days=cfg.data.h_days)
    out = change(df, (t >= end).to_numpy())
    p = tmp_path / f"batch_{name}.csv"
    out.to_csv(p, index=False)
    return replace(cfg, data=replace(cfg.data, batch_csv=str(p))).validate()


def test_nan_or_empty_stream_gives_identical_outputs(cfg, tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = outputs(load_history(cfg), cfg)

        def nan(df, s):
            df.loc[s, list(COLS)] = "nan"
            return df
        cfg_nan = with_csv(cfg, tmp_path, nan, "nan")
        assert pd.read_csv(cfg_nan.data.batch_csv)[list(COLS)].isna().any(axis=1).sum() > 100
        assert_same(base, outputs(load_history(cfg_nan), cfg_nan))
        cfg_empty = with_csv(cfg, tmp_path, lambda df, s: df.loc[~s], "empty")
        assert len(pd.read_csv(cfg_empty.data.batch_csv)) < 600
        assert_same(base, outputs(load_history(cfg_empty), cfg_empty))


def test_f_rule_and_phase0_never_read_the_stream(cfg):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds_H = load_history(cfg)
        with pytest.raises(RuntimeError, match="stream was read"):
            ds_H.stream.n
        h = outputs(ds_H, cfg)                         # would raise if Phase 0 or the f rule read the stream
        ds = load_dataset(cfg)
        guarded = replace(ds, stream=NoStream())
        assert_same(outputs(ds, cfg), outputs(guarded, cfg))
        # with the hardware profiles fit on H, the H-only Dataset is the same H as the full one
        np.testing.assert_array_equal(ds_H.H.A, ds.H.A)
        np.testing.assert_array_equal(ds_H.H.hw_idx, ds.H.hw_idx)
        np.testing.assert_array_equal(ds_H.H.arrival_step, ds.H.arrival_step)
        assert_same(h, outputs(ds, cfg))


def test_select_f_rule():
    r = lambda f, tau, full: dict(f=f, tau=tau, full_time=full)
    res = [r(0.3, [0.0, 0.0], 0.25), r(0.4, [0.0, 1e-3], 0.35), r(0.5, [1e-4, 0.0], 0.31), r(0.6, [0.0, 0.0], 0.10)]
    assert select_f(res) == 0.5                        # 0.3 fails (a); 0.5 is 1 point from the band, 0.4 is 5
    res.append(r(0.45, [1e-4, 1e-4], 0.22))
    res.append(r(0.35, [1e-4, 1e-4], 0.26))
    assert select_f(res) == 0.35                       # both inside the band: the tie goes to 25%
    with pytest.raises(ValueError):
        select_f([r(0.3, [0.0, 0.0], 0.25)])

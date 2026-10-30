"""The data path (D3) on the synthetic trace: the gate, the capacity, the bids and the energy use the SAME request
(the real one, not the hardware profile) and the SAME capped effective duration; the job quota is 0.2 x C of the
cluster actually used (fixed point); C = f x D_bar on H; UTC time stamps; the type volume = mean real request x
mean duration."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from dlent_rt.bid import calibrate_u0, cost_base
from dlent_rt.data import load_dataset, mean_demand
from dlent_rt.energy import EnergyParams, compute_energy_kwh
from dlent_rt.models.core.stream import build_stream
from dlent_rt.phase0 import _compute_job_energy, column_ids, column_means, run_phase0
from tests.synthetic import synthetic_config


def _cfg(tmp_path, **time_kw):
    cfg = synthetic_config(tmp_path, n0=400, n_jobs=3000)
    return replace(cfg, time=replace(cfg.time, **time_kw)).validate() if time_kw else cfg


def _frame(cfg):
    df = pd.read_csv(cfg.data.batch_csv)
    return df.assign(t=pd.to_datetime(df["arrival_utc"])).sort_values("t", kind="stable")


def test_one_request_one_duration(tmp_path):
    cfg = _cfg(tmp_path)
    ds = load_dataset(cfg)
    df = _frame(cfg)
    Q = ds.limit_report.quota
    kept = df[(df["CPU"] <= Q[0]) & (df["RAM"] <= Q[1])]
    A = np.concatenate([ds.H.A, ds.stream.A])
    ids = np.concatenate([ds.H.collection_id, ds.stream.collection_id])
    # the quota only; A = the real request (the profile is only the type id)
    assert ds.H.n + ds.stream.n == len(kept) == ds.limit_report.n_after
    assert np.array_equal(A, df.set_index("collection_id").loc[ids, ["CPU", "RAM"]].to_numpy(float))
    assert not np.allclose(A, ds.hw_catalogue.centroids[np.concatenate([ds.H.hw_idx, ds.stream.hw_idx])])
    # one capped duration: hours and steps agree, and both stop at d_max = d_cap
    tick = cfg.time.tick_seconds
    assert ds.d_max_steps == int(np.rint(cfg.time.d_cap_hours * 3600 / tick))
    for J in (ds.H, ds.stream):
        assert J.duration_hours.max() <= ds.d_max_steps * tick / 3600 + 1e-12
        assert J.duration_hours.min() >= cfg.time.duration_floor_hours
        assert np.array_equal(J.duration_steps, np.maximum(np.rint(J.duration_hours * 3600 / tick), 1).astype(np.int64))
    assert (ds.stream.duration_steps == ds.d_max_steps).any()
    # bids and energy use the same A and the same capped duration as the gate (stream)
    S = build_stream(ds, cfg)
    assert S.A is ds.stream.A and np.array_equal(S.D, ds.stream.duration_steps)
    assert np.array_equal(S.duration_hours, ds.stream.duration_hours)
    u0 = calibrate_u0(ds.H, cfg)
    np.testing.assert_allclose(S.cost_base, cost_base(S.A[:, 0], S.A[:, 1], ds.stream.priority, u0, cfg), rtol=1e-12)
    em = cfg.energy_model
    params = EnergyParams(em.machine_cores, em.machine_ram_gb, em.p_core, em.p_gb, em.pue)
    assert np.array_equal(S.energy_kwh, compute_energy_kwh(S.A[:, 0], S.A[:, 1], ds.stream.scheduling_class,
                                                           S.duration_hours, params))
    assert np.array_equal(_compute_job_energy(ds.H, cfg),
                          compute_energy_kwh(ds.H.A[:, 0], ds.H.A[:, 1], ds.H.scheduling_class,
                                             ds.H.duration_hours, params))


def test_capacity_and_quota_fixed_point(tmp_path):
    cfg = _cfg(tmp_path)
    ds = load_dataset(cfg)
    H = ds.H
    md = mean_demand(H.A, H.duration_hours, ds.H_span_steps * cfg.time.tick_seconds / 3600)
    np.testing.assert_allclose(ds.capacity, cfg.capacity.f * md["D_bar"], rtol=1e-12)
    its = ds.limit_report.quota_iterations
    assert np.all(np.asarray(its[-1]["change"]) < cfg.limits.quota_tol)
    # at the fixed point Q = 0.2 C of the cluster actually used (within the 1% tolerance)
    np.testing.assert_allclose(np.asarray(ds.limit_report.quota) / ds.capacity, cfg.limits.quota_share, rtol=0.02)
    assert np.all(np.concatenate([H.A, ds.stream.A]) <= np.asarray(ds.limit_report.quota)[None, :])


def test_quota_collapses_at_tiny_f(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(ValueError, match="collapsed"):
        load_dataset(replace(cfg, capacity=replace(cfg.capacity, f=0.01)).validate())


def test_timestamps_need_a_time_zone(tmp_path):
    cfg = _cfg(tmp_path)
    df = pd.read_csv(cfg.data.batch_csv)
    df["arrival_utc"] = pd.to_datetime(df["arrival_utc"]).dt.tz_localize(None).astype(str)
    df.to_csv(cfg.data.batch_csv, index=False)
    with pytest.raises(ValueError, match="time-zone-aware"):
        load_dataset(cfg)


def test_timestamps_converted_to_utc(tmp_path):
    cfg = _cfg(tmp_path)
    df = pd.read_csv(cfg.data.batch_csv)
    t = pd.to_datetime(df["arrival_utc"]).dt.tz_convert("America/New_York").astype(str)   # same instants
    df.assign(arrival_utc=t).to_csv(cfg.data.batch_csv, index=False)
    ds = load_dataset(cfg)
    first = pd.to_datetime(t).min()
    assert first.utcoffset() == pd.Timedelta(hours=-4)                 # the file is in New York time (EDT)
    assert ds.H.datetime[0] == np.datetime64(first.tz_convert("UTC").tz_localize(None))


def test_effective_duration_used_everywhere(tmp_path):
    cfg = _cfg(tmp_path)
    df = pd.read_csv(cfg.data.batch_csv)
    df["CPU_hours_real"] = df["CPU"] * df["duration"] * np.linspace(0.2, 1.0, len(df))   # <= request x span
    df.to_csv(cfg.data.batch_csv, index=False)
    cfg = replace(cfg, capacity=replace(cfg.capacity, f=5.0)).validate()     # shorter jobs: the quota needs a larger f
    ds = load_dataset(cfg)
    want = (df.set_index("collection_id").eval("CPU_hours_real / CPU")
            .clip(lower=cfg.time.duration_floor_hours, upper=cfg.time.d_cap_hours))
    for J in (ds.H, ds.stream):
        np.testing.assert_allclose(J.duration_hours, want.loc[J.collection_id].to_numpy(), rtol=1e-12)
    assert np.array_equal(build_stream(ds, cfg).duration_hours, ds.stream.duration_hours)


def test_column_means_real_mean():
    cid = np.array([0, 0, 2])
    A = np.array([[0.5, 0.1], [1.5, 0.3], [9.0, 6.0]])
    N, m = column_means(cid, 3, dict(A=A))
    np.testing.assert_array_equal(N, [2, 0, 1])
    np.testing.assert_array_equal(m["A"], [[1.0, 0.2], [0.0, 0.0], [9.0, 6.0]])


def test_phase0_volume_is_column_mean_request_times_mean_duration(tmp_path):
    cfg = _cfg(tmp_path)
    ds = load_dataset(cfg)
    st = run_phase0(ds, cfg)
    cid = column_ids(st.type_space, ds.H.priority, ds.H.hw_idx)
    nc = st.type_space.n_cols
    N = np.bincount(cid, minlength=nc)
    for i in range(ds.n_resources):
        mean = np.bincount(cid, weights=ds.H.A[:, i], minlength=nc)
        np.testing.assert_allclose(st.col(st.A_type)[N > 0, i], mean[N > 0] / N[N > 0], rtol=1e-12)
    d = np.bincount(cid, weights=ds.H.duration_steps, minlength=nc)
    np.testing.assert_allclose(st.col(st.d_bar)[N > 0], d[N > 0] / N[N > 0], rtol=1e-12)
    np.testing.assert_allclose(st.v, st.A_type * st.d_bar[:, None], rtol=1e-12)

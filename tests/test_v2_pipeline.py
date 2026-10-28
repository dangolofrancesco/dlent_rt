"""The data path (week 32) on the synthetic trace: the gate, the capacity, the bids and the energy use the SAME request
(the real one, not the hardware profile) and the SAME capped effective duration; the size limit R_max; UTC time
stamps; the Phase 0 type volume = mean real request of the type x its mean duration."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from dlent_rt.bid import compute_base_utility
from dlent_rt.data import (_assign_real_tick_steps, _occupancy_stats, _OccupancyBasis, _time_weighted_quantile,
                           load_dataset)
from dlent_rt.energy import EnergyParams, compute_energy_kwh
from dlent_rt.models.core.stream import build_stream
from dlent_rt.phase0 import _compute_job_energy, _synthesize_bids, assign_type_ids, run_phase0, type_request
from tests.synthetic import synthetic_config

D_MAX_H = 0.5


def _cfg(tmp_path, **time_kw):
    cfg = synthetic_config(tmp_path, n0=400, n_jobs=3000)
    return replace(cfg, time=replace(cfg.time, d_max_hours=D_MAX_H, **time_kw)).validate()


def _frame(cfg):
    df = pd.read_csv(cfg.data.batch_csv)
    return df.assign(t=pd.to_datetime(df["arrival_utc"])).sort_values("t", kind="stable")


def test_one_request_one_duration(tmp_path):
    cfg = _cfg(tmp_path)
    ds = load_dataset(cfg)
    df = _frame(cfg)
    kept = df[(df[["CPU", "RAM"]] <= cfg.limits.r_max_machines).all(axis=1)]
    A = np.concatenate([ds.H.A, ds.stream.A])
    ids = np.concatenate([ds.H.collection_id, ds.stream.collection_id])
    # the R_max limit only; A = the real request (the profile is only the type id)
    assert ds.H.n + ds.stream.n == len(kept) == ds.limit_report.n_after
    assert np.array_equal(A, df.set_index("collection_id").loc[ids, ["CPU", "RAM"]].to_numpy(float))
    assert not np.allclose(A, ds.hw_catalogue.centroids[np.concatenate([ds.H.hw_idx, ds.stream.hw_idx])])
    # one capped duration: hours and steps agree, and both stop at d_max
    tick = cfg.time.tick_seconds
    assert ds.d_max_steps == int(np.rint(D_MAX_H * 3600 / tick))
    for J in (ds.H, ds.stream):
        assert J.duration_hours.max() <= ds.d_max_steps * tick / 3600 + 1e-12
        assert np.array_equal(J.duration_steps, np.maximum(np.rint(J.duration_hours * 3600 / tick), 1).astype(np.int64))
    assert (ds.stream.duration_steps == ds.d_max_steps).any()
    # capacity: the demand level exceeded 25% of the time, on the same A and durations
    t = kept["t"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
    basis = _OccupancyBasis(arrival_step=_assign_real_tick_steps(t, t[0], tick),
                            duration_steps=np.concatenate([ds.H.duration_steps, ds.stream.duration_steps]), A=A)
    st = _occupancy_stats(basis)
    assert np.array_equal(ds.capacity, _time_weighted_quantile(st["levels"], st["weights"], 0.75))
    # bids and energy use the same A and the same capped duration as the gate (stream)
    S = build_stream(ds, cfg)
    assert S.A is ds.stream.A and np.array_equal(S.D, ds.stream.duration_steps)
    assert np.array_equal(S.duration_hours, ds.stream.duration_hours)
    em, bd = cfg.energy_model, cfg.bid
    cost = (compute_base_utility(ds.stream.priority, bd.base_utility_u0, bd.base_utility_mode)
            + bd.gamma1 * S.A[:, 0] * em.machine_cores + bd.gamma2 * S.A[:, 1] * em.machine_ram_gb) * S.duration_hours
    assert np.allclose(S.cost_base, np.maximum(cost, 1e-9), rtol=1e-12, atol=0)
    params = EnergyParams(em.machine_cores, em.machine_ram_gb, em.p_core, em.p_gb, em.pue)
    assert np.array_equal(S.energy_kwh, compute_energy_kwh(S.A[:, 0], S.A[:, 1], ds.stream.scheduling_class,
                                                           S.duration_hours, params))
    assert np.array_equal(_compute_job_energy(ds.H, cfg),
                          compute_energy_kwh(ds.H.A[:, 0], ds.H.A[:, 1], ds.H.scheduling_class,
                                             ds.H.duration_hours, params))
    assert np.array_equal(_synthesize_bids(ds.stream, cfg, cfg.run.seed + cfg.online.bid_seed_offset).cost_base,
                          S.cost_base)


def test_d_max_defaults_to_the_horizon_ceiling(tmp_path):
    cfg = synthetic_config(tmp_path, n0=400, n_jobs=3000)
    assert cfg.time.d_max_hours is None
    ds = load_dataset(cfg)
    assert ds.d_max_steps == max(int(cfg.time.d_max_horizon_fraction * ds.T), 1)


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
    cfg = replace(cfg, time=replace(cfg.time, d_max_hours=50.0, d_max_horizon_fraction=1.0,
                                    d_max_ratio_abort=1.0)).validate()
    ds = load_dataset(cfg)
    want = df.set_index("collection_id").eval("CPU_hours_real / CPU").clip(lower=cfg.time.duration_floor_hours)
    for J in (ds.H, ds.stream):
        np.testing.assert_allclose(J.duration_hours, want.loc[J.collection_id].to_numpy(), rtol=1e-12)
    assert np.array_equal(build_stream(ds, cfg).duration_hours, ds.stream.duration_hours)


def test_r_max_removes_large_jobs_only(tmp_path):
    cfg = _cfg(tmp_path)
    df = pd.read_csv(cfg.data.batch_csv)
    r = float(df["CPU"].quantile(0.9))
    ds = load_dataset(replace(cfg, limits=replace(cfg.limits, r_max_machines=r)).validate())
    keep = (df[["CPU", "RAM"]] <= r).all(axis=1)
    assert ds.H.n + ds.stream.n == int(keep.sum())
    assert np.concatenate([ds.H.A, ds.stream.A]).max() <= r
    ds_all = load_dataset(replace(cfg, limits=replace(cfg.limits, r_max_machines=None)).validate())
    assert ds_all.H.n + ds_all.stream.n == len(df)


def test_type_request_real_mean():
    prof = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    tid = np.array([0, 0, 2])
    A = np.array([[0.5, 0.1], [1.5, 0.3], [9.0, 6.0]])
    np.testing.assert_array_equal(type_request(tid, A, prof), [[1.0, 0.2], [2.0, 2.0], [9.0, 6.0]])


def test_phase0_type_volume_is_mean_real_request(tmp_path):
    cfg = _cfg(tmp_path)
    ds = load_dataset(cfg)
    st = run_phase0(ds, cfg)
    k = st.grid.assign(st.bids.v_rate, st.myerson)
    tid = assign_type_ids(st.type_space, np.asarray(k), ds.H.priority, ds.H.hw_idx)
    ok = tid >= 0
    N = np.bincount(tid[ok], minlength=st.n_types)
    for i in range(ds.n_resources):
        mean = np.bincount(tid[ok], weights=ds.H.A[ok, i], minlength=st.n_types)
        np.testing.assert_allclose(st.A_type[N > 0, i], mean[N > 0] / N[N > 0], rtol=1e-12)
    np.testing.assert_allclose(st.v, st.A_type * st.d_bar[:, None], rtol=1e-12)

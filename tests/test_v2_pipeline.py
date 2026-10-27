"""Week 32 data path (configs/regime_v2.yaml switches) on the synthetic trace: the gate, the capacity, the bids and the
energy use the SAME request (the real one, not the profile) and the SAME capped duration; old keys still load."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from dlent_rt.bid import compute_base_utility
from dlent_rt.config import load_config
from dlent_rt.data import _occupancy_stats, _OccupancyBasis, _time_weighted_quantile, load_dataset
from dlent_rt.energy import EnergyParams, compute_energy_kwh
from dlent_rt.models.core.stream import build_stream
from dlent_rt.phase0 import _compute_job_energy, _synthesize_bids
from tests.synthetic import synthetic_config

D_MAX_H = 0.5


def _v2(tmp_path, utc=True):
    cfg = synthetic_config(tmp_path, n0=400, n_jobs=3000)
    df = pd.read_csv(cfg.data.batch_csv)
    if utc:
        df["request_time"] = pd.to_datetime(df["request_time"]).dt.tz_localize("UTC").astype(str)
        df.to_csv(cfg.data.batch_csv, index=False)
    return replace(
        cfg,
        data=replace(cfg.data, time_base="utc"),
        time=replace(cfg.time, d_max_hours=D_MAX_H, cap_duration_everywhere=True),
        hardware=replace(cfg.hardware, snap_requests=False),
        outliers=replace(cfg.outliers, policy="none"),
        capacity=replace(cfg.capacity, mode="exceeded_share", exceeded_share=0.25,
                         drop_jobs_exceeding_capacity=False),
    ).validate()


def test_v2_one_request_one_duration(tmp_path):
    cfg = _v2(tmp_path)
    ds = load_dataset(cfg)
    df = pd.read_csv(cfg.data.batch_csv)
    df = df.assign(t=pd.to_datetime(df["request_time"])).sort_values("t", kind="stable")
    A = np.concatenate([ds.H.A, ds.stream.A])
    # no outlier filter; A = the real request (the profile is only the type id)
    assert ds.H.n + ds.stream.n == len(df)
    ids = np.concatenate([ds.H.collection_id, ds.stream.collection_id])
    assert np.array_equal(A, df.set_index("collection_id").loc[ids, ["CPU", "RAM"]].to_numpy(float))
    assert not np.allclose(A, ds.hw_catalogue.centroids[np.concatenate([ds.H.hw_idx, ds.stream.hw_idx])])
    # one capped duration: hours and steps agree, and both stop at d_max
    tick = cfg.time.tick_seconds
    assert ds.d_max_steps == int(np.rint(D_MAX_H * 3600 / tick))
    for J in (ds.H, ds.stream):
        assert J.duration_hours.max() <= ds.d_max_steps * tick / 3600 + 1e-12
        assert np.array_equal(J.duration_steps, np.maximum(np.rint(J.duration_hours * 3600 / tick), 1).astype(np.int64))
    assert (ds.stream.duration_steps == ds.d_max_steps).any()
    # capacity: the level exceeded 25% of the time, on the same A and durations
    basis = _OccupancyBasis(arrival_step=np.concatenate([ds.H.arrival_step, _abs_stream_steps(ds, cfg, df)]),
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
    E = compute_energy_kwh(S.A[:, 0], S.A[:, 1], ds.stream.scheduling_class, S.duration_hours, params)
    assert np.array_equal(S.energy_kwh, E)
    assert np.array_equal(_compute_job_energy(ds.H, cfg),
                          compute_energy_kwh(ds.H.A[:, 0], ds.H.A[:, 1], ds.H.scheduling_class,
                                             ds.H.duration_hours, params))
    assert np.array_equal(_synthesize_bids(ds.stream, cfg, cfg.run.seed + cfg.online.bid_seed_offset).cost_base,
                          S.cost_base)


def _abs_stream_steps(ds, cfg, df):
    """Absolute arrival steps of the stream on H's timeline (load_dataset re-bases the stream to 0)."""
    from dlent_rt.data import _assign_real_tick_steps
    t = np.sort(df["t"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy())
    return _assign_real_tick_steps(t, t[0], cfg.time.tick_seconds)[cfg.data.n0:]


def test_utc_time_base_needs_time_zone(tmp_path):
    cfg = _v2(tmp_path, utc=False)
    with pytest.raises(ValueError, match="time-zone-aware"):
        load_dataset(cfg)


def test_utc_time_base_converts_to_utc(tmp_path):
    cfg = _v2(tmp_path)
    df = pd.read_csv(cfg.data.batch_csv)
    t = pd.to_datetime(df["request_time"]).dt.tz_convert("America/New_York").astype(str)   # same instants
    df.assign(request_time=t).to_csv(cfg.data.batch_csv, index=False)
    ds = load_dataset(cfg)
    first = pd.to_datetime(t).min()
    assert first.utcoffset() == pd.Timedelta(hours=-4)                 # the file is in New York time (EDT)
    assert ds.H.datetime[0] == np.datetime64(first.tz_convert("UTC").tz_localize(None))


def test_old_energy_keys_still_load(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("energy_model:\n  spec_cpu_core: 32\n  spec_ram_gb: 128\n")
    em = load_config(p).energy_model
    assert (em.machine_cores, em.machine_ram_gb) == (32, 128)
    p.write_text("energy_model:\n  spec_cpu_core: 32\n  machine_cores: 32\n")
    with pytest.raises(ValueError, match="old name"):
        load_config(p)


def test_effective_duration_used_everywhere(tmp_path):
    cfg = _v2(tmp_path)
    df = pd.read_csv(cfg.data.batch_csv)
    df["CPU_hours_real"] = df["CPU"] * df["duration"] * np.linspace(0.2, 1.0, len(df))   # <= request x span
    df.to_csv(cfg.data.batch_csv, index=False)
    cfg = replace(cfg, time=replace(cfg.time, duration_mode="effective", d_max_hours=50.0,
                                    d_max_horizon_fraction=1.0, d_max_ratio_abort=1.0)).validate()
    ds = load_dataset(cfg)
    want = df.set_index("collection_id").eval("CPU_hours_real / CPU").clip(lower=cfg.time.duration_floor_hours)
    for J in (ds.H, ds.stream):
        np.testing.assert_allclose(J.duration_hours, want.loc[J.collection_id].to_numpy(), rtol=1e-12)
    S = build_stream(ds, cfg)
    assert np.array_equal(S.duration_hours, ds.stream.duration_hours)


def test_r_max_removes_large_jobs_only(tmp_path):
    cfg = _v2(tmp_path)
    df = pd.read_csv(cfg.data.batch_csv)
    r = float(df["CPU"].quantile(0.9))
    ds = load_dataset(replace(cfg, outliers=replace(cfg.outliers, policy="r_max", r_max_machines=r)).validate())
    keep = (df[["CPU", "RAM"]] <= r).all(axis=1)
    assert ds.H.n + ds.stream.n == int(keep.sum())
    assert np.concatenate([ds.H.A, ds.stream.A]).max() <= r


def test_type_request_real_mean():
    from dlent_rt.phase0 import type_request
    prof = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    tid = np.array([0, 0, 2])
    A = np.array([[0.5, 0.1], [1.5, 0.3], [9.0, 6.0]])
    assert type_request(tid, A, prof, "profile") is prof
    np.testing.assert_array_equal(type_request(tid, A, prof, "real_mean"), [[1.0, 0.2], [2.0, 2.0], [9.0, 6.0]])

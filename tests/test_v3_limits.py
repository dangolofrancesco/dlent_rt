"""Week 32 v3 options (all off in the default config): the duration cap limits.d_cap_hours (truncate, used
everywhere), the job quota limits.job_quota_* (reject before the market), the capacity mode mean_demand."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from dlent_rt.bid import compute_base_utility
from dlent_rt.data import load_dataset, mean_demand
from dlent_rt.energy import EnergyParams, compute_energy_kwh
from dlent_rt.models.core.stream import build_stream
from dlent_rt.reward import carbon_kg
from tests.synthetic import synthetic_config


def _cfg(tmp_path, limits=None, capacity=None):
    cfg = synthetic_config(tmp_path, n0=400, n_jobs=3000)
    cfg = replace(cfg, limits=replace(cfg.limits, r_max_machines=None, **(limits or {})))
    if capacity:
        cfg = replace(cfg, capacity=replace(cfg.capacity, **capacity))
    return cfg.validate()


def test_mean_demand_formula_hand_example():
    # 2 jobs in 10 h: CPU r = (1, 3), RAM m = (2, 0), d = (2 h, 4 h)
    out = mean_demand(np.array([[1.0, 2.0], [3.0, 0.0]]), np.array([2.0, 4.0]), 10.0)
    assert out["lambda_hat"] == pytest.approx(0.2)                     # jobs per hour
    np.testing.assert_allclose(out["mean_rd"], [7.0, 2.0])             # (1*2 + 3*4) / 2, (2*2 + 0*4) / 2
    np.testing.assert_allclose(out["D_bar"], [1.4, 0.4])               # = total work / span = (14, 4) / 10


def test_mean_demand_capacity_uses_H_only(tmp_path):
    cfg = _cfg(tmp_path, capacity=dict(mode="mean_demand", f=0.5))
    ds = load_dataset(cfg)
    span_h = ds.H_span_steps * cfg.time.tick_seconds / 3600
    D_bar = (ds.H.A * ds.H.duration_hours[:, None]).sum(axis=0) / span_h
    np.testing.assert_allclose(ds.capacity, 0.5 * D_bar, rtol=1e-12)
    np.testing.assert_allclose(ds.capacity_diag["D_bar"], D_bar, rtol=1e-12)
    assert ds.capacity_diag["n"] == ds.H.n


def test_quota_rejects_before_the_market(tmp_path):
    cfg0 = _cfg(tmp_path)
    df = pd.read_csv(cfg0.data.batch_csv)
    q_cpu, q_mem = float(df["CPU"].quantile(0.9)), float(df["RAM"].quantile(0.95))
    ds = load_dataset(replace(cfg0, limits=replace(cfg0.limits, job_quota_machines=q_cpu,
                                                   job_quota_mem=q_mem)).validate())
    over = (df["CPU"] > q_cpu) | (df["RAM"] > q_mem)
    assert ds.limit_report.n_quota_rejected == int(over.sum()) > 0
    ids = set(np.concatenate([ds.H.collection_id, ds.stream.collection_id]))
    assert ids == set(df.loc[~over, "collection_id"])                   # neither in H (Phase 0) nor in the stream
    A = np.concatenate([ds.H.A, ds.stream.A])
    assert A[:, 0].max() <= q_cpu and A[:, 1].max() <= q_mem


def test_cap_truncates_and_is_used_everywhere(tmp_path):
    cap = 0.3
    cfg = _cfg(tmp_path, limits=dict(d_cap_hours=cap))
    ds = load_dataset(cfg)
    df = pd.read_csv(cfg.data.batch_csv)
    tick = cfg.time.tick_seconds
    assert ds.H.n + ds.stream.n == len(df)                              # kept, only shortened
    assert ds.d_max_steps == int(np.rint(cap * 3600 / tick))
    S = build_stream(ds, cfg)
    eff = df.set_index("collection_id").eval("CPU_hours_real / CPU").clip(lower=cfg.time.duration_floor_hours)
    d = np.minimum(eff.loc[ds.stream.collection_id].to_numpy(), cap)
    assert (eff > cap).any()
    np.testing.assert_allclose(S.duration_hours, d, rtol=1e-12)
    # 1. cluster occupancy: duration in steps
    assert np.array_equal(S.D, np.maximum(np.rint(d * 3600 / tick), 1).astype(np.int64))
    # 2. bid: hourly rate x d_j
    em, bd = cfg.energy_model, cfg.bid
    rate = (compute_base_utility(ds.stream.priority, bd.base_utility_u0, bd.base_utility_mode)
            + bd.gamma1 * S.A[:, 0] * em.machine_cores + bd.gamma2 * S.A[:, 1] * em.machine_ram_gb)
    np.testing.assert_allclose(S.cost_base, np.maximum(rate * d, 1e-9), rtol=1e-12)
    # 3. energy and carbon cost: power x d_j
    E = compute_energy_kwh(S.A[:, 0], S.A[:, 1], ds.stream.scheduling_class, d,
                           EnergyParams(em.machine_cores, em.machine_ram_gb, em.p_core, em.p_gb, em.pue))
    np.testing.assert_allclose(S.energy_kwh, E, rtol=1e-12)
    np.testing.assert_allclose(S.C_carbon, carbon_kg(E, S.ci), rtol=1e-12)

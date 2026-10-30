"""Greedy policies G1 (TypeGreedyPolicy) and G2 (JobGreedyPolicy) on hand-checkable streams."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from dlent_rt.data import load_dataset
from dlent_rt.models.core.stream import build_stream
from dlent_rt.models.policy import (REJ_NONE, REJ_NOTYPE, REJ_PHYS, REJ_R, REJ_UNSEEN,
                                    JobGreedyPolicy, StaticPolicy, TypeGreedyPolicy, run_policy)
from dlent_rt.phase0 import run_phase0
from dlent_rt.reward import job_reward
from tests.synthetic import synthetic_config


def _stream(n, D=10):
    """n unit jobs (A = 1 on one resource), one arrival per tick, duration D."""
    return SimpleNamespace(n=n, tick=np.arange(1, n + 1), A=np.ones((n, 1)), D=np.full(n, D))


def test_g2_job_reward_and_physical_gate():
    # c = 2: jobs 0 and 2 fill it, job 1 has R <= 0, job 3 wants but does not fit
    pol = JobGreedyPolicy(np.array([5.0, -1.0, 3.0, 4.0]))
    res = run_policy(_stream(4), np.array([2.0]), pol)
    assert res.admit.tolist() == [True, False, True, False]
    assert res.phys_reject.tolist() == [False, False, False, True]
    assert pol.reason.tolist() == [REJ_NONE, REJ_R, REJ_NONE, REJ_PHYS]


def test_g2_release_frees_capacity():
    # D = 1: each job is released before the next arrival, so every positive job fits
    res = run_policy(_stream(3, D=1), np.array([1.0]), JobGreedyPolicy(np.array([1.0, 2.0, 3.0])))
    assert res.admit.all()


def test_g1_cumulative_mean_hand_computed():
    # r0 = (1, -2, 0) with n0 = (1, 1, 0): type 2 is unseen at start
    tid = np.array([0, 0, 1, 0, 2, 2, -1])
    R = np.array([-5.0, 4.0, 3.0, 10.0, 1.0, 7.0, 9.0])
    pol = TypeGreedyPolicy(tid, R, r0=[1.0, -2.0, 0.0], n0=[1, 1, 0])
    res = run_policy(_stream(7), np.array([100.0]), pol)
    # i0: r0=1 > 0 admit;         then r0 = 1 + (-5 - 1)/2 = -2
    # i1: r0=-2 reject;           then r0 = -2 + (4 + 2)/3 = 0
    # i2: r1=-2 reject;           then r1 = -2 + (3 + 2)/2 = 0.5
    # i3: r0=0 reject (<= 0);     then r0 = 0 + 10/4 = 2.5
    # i4: type 2 unseen reject;   then r2 = 1
    # i5: r2=1 > 0 admit;         then r2 = 1 + (7 - 1)/2 = 4
    # i6: no type
    assert res.admit.tolist() == [True, False, False, False, False, True, False]
    assert pol.reason.tolist() == [REJ_NONE, REJ_R, REJ_R, REJ_R, REJ_UNSEEN, REJ_NONE, REJ_NOTYPE]
    np.testing.assert_array_equal(pol.r_dec, [1.0, -2.0, -2.0, 0.0, np.nan, 1.0, np.nan])
    np.testing.assert_array_equal(pol.r, [2.5, 0.5, 4.0])
    assert pol.n.tolist() == [4, 2, 2]


def test_g1_frozen_is_not_updated_and_gate_applies():
    pol = TypeGreedyPolicy(np.array([0, 0, 0]), np.array([-9.0, -9.0, -9.0]), r0=[1.0], n0=[5],
                           update=False)
    res = run_policy(_stream(3), np.array([2.0]), pol)
    assert res.admit.tolist() == [True, True, False]
    assert pol.reason.tolist() == [REJ_NONE, REJ_NONE, REJ_PHYS]
    assert pol.r.tolist() == [1.0] and pol.n.tolist() == [5]


def test_g1_frozen_equals_static_policy_with_zero_tau(tmp_path):
    """G1 on frozen H types == StaticPolicy of Phase 0 with tau = 0 and r~ = 0 for the types with no job in H (G1
    rejects them; Phase 0 gives them fallback estimates) (same types, same gate)."""
    cfg = synthetic_config(tmp_path, n0=400, n_jobs=3000, drift=True)
    ds = load_dataset(cfg)
    st = run_phase0(ds, cfg)
    S = build_stream(ds, cfg)
    R = job_reward(S, st, cfg.objective).R
    g1 = run_policy(S, ds.capacity, TypeGreedyPolicy.from_state(st, S, R, ds.H_span_steps, update=False))
    st0 = replace(st, tau=np.zeros_like(st.tau), r_tilde=np.where(st.p > 0, st.r_tilde, 0.0))
    sp = run_policy(S, ds.capacity, StaticPolicy(st0, S))
    assert np.array_equal(g1.admit, sp.admit) and np.array_equal(g1.phys_reject, sp.phys_reject)
    # H counts recovered from p reproduce Phase 0's type counts
    n0 = np.rint(st.p * ds.H_span_steps)
    assert np.allclose(n0 / ds.H_span_steps, st.p, rtol=0, atol=1e-15)

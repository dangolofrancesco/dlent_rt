"""Policy hook: StaticPolicy through the shared loop == the hand-written static loop of notebook 03."""
import tempfile, warnings
from pathlib import Path

import numpy as np

from dlent_rt.data import load_dataset
from dlent_rt.phase0 import run_phase0, assign_type_ids
from dlent_rt.models.core.occupancy import Occupancy
from dlent_rt.models.core.stream import build_stream
from dlent_rt.models.policy import run_static
from tests.synthetic import synthetic_config


def _manual_static(S, c, st, U):
    j = assign_type_ids(st.type_space, st.grid.assign(S.v, st.myerson), S.priority, S.hw_idx)
    jj = np.maximum(j, 0)
    khat = np.where(j >= 0, st.r_tilde[jj] - st.v[jj] @ st.tau, -np.inf) > 0
    occ = Occupancy(c, record=False)
    adm, phys = np.zeros(S.n, bool), np.zeros(S.n, bool)
    for i in range(S.n):
        t = int(S.tick[i])
        occ.release_until(t)
        if khat[i] and U[i]:
            if occ.fits(S.A[i]):
                occ.add(i, S.A[i], t, int(S.D[i]))
                adm[i] = True
            else:
                phys[i] = True
    return adm, phys


def test_static_policy_matches_manual_loop():
    with tempfile.TemporaryDirectory() as d, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = synthetic_config(Path(d), n0=400, n_jobs=3000, drift=True)
        ds = load_dataset(cfg)
        st = run_phase0(ds, cfg)
    S = build_stream(ds, cfg)
    for c in (ds.capacity, 0.05 * ds.capacity):          # non-binding and binding capacity
        u = np.random.default_rng(0).random(S.n)
        for thin_p in (None, 0.5):
            res = run_static(S, c, st, thin_p=thin_p, u_rand=u)
            U = np.ones(S.n, bool) if thin_p is None else u < thin_p
            adm, phys = _manual_static(S, c, st, U)
            np.testing.assert_array_equal(res.admit, adm)
            np.testing.assert_array_equal(res.phys_reject, phys)
    assert phys.any()                                    # the binding case exercises the gate

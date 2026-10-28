"""policy.unseen_types (week 32): the static FractionPolicy rejects arrivals of types without jobs in H, or (dual_rule)
admits them iff r~_u - tau . v_u > 0, with r~_u, v_u from theory_eval.unseen_type_estimates. Seen types unchanged."""
import warnings

import numpy as np
import pytest

from dlent_rt.config import default_config
from dlent_rt.data import load_dataset
from dlent_rt.models.core.stream import build_stream
from dlent_rt.models.policy import FractionPolicy
from dlent_rt.phase0 import assign_type_ids, run_phase0
from dlent_rt.theory_eval import lp_x, unseen_type_estimates
from tests.synthetic import synthetic_config


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = synthetic_config(tmp_path_factory.mktemp("u"), n0=200, n_jobs=3000, drift=True)
        ds = load_dataset(cfg)
        st = run_phase0(ds, cfg)
        return cfg, ds, st, build_stream(ds, cfg)


def test_default_is_reject():
    assert default_config().policy.unseen_types == "reject"


def test_estimates_keep_seen_types_and_fill_unseen(world):
    cfg, ds, st, S = world
    e = unseen_type_estimates(st, ds, cfg)
    seen = st.p > 0
    np.testing.assert_allclose(e["r"][seen], st.r_tilde[seen], rtol=1e-9)
    np.testing.assert_array_equal(e["v"][seen], st.v[seen])
    assert (~seen).any() and np.all(np.isfinite(e["r"])) and np.all(e["v"][~seen] > 0)
    np.testing.assert_array_equal(e["admit"], (e["r"] - e["v"] @ st.tau) > 0)


def test_dual_rule_only_changes_unseen_arrivals(world):
    cfg, ds, st, S = world
    x, _ = lp_x(st, ds.capacity, cfg)
    e = unseen_type_estimates(st, ds, cfg)
    rej = FractionPolicy(st, S, x, seed=3)
    dual = FractionPolicy(st, S, x, seed=3, admit_unseen=e["admit"])
    j = assign_type_ids(st.type_space, np.asarray(st.grid.assign(S.v, st.myerson)), S.priority, S.hw_idx)
    unseen = (j >= 0) & (st.p[np.maximum(j, 0)] <= 0)
    assert unseen.any()
    w_rej = np.array([rej.want(i, 0) for i in range(S.n)])
    w_dual = np.array([dual.want(i, 0) for i in range(S.n)])
    assert not w_rej[unseen].any()                                            # reject: never
    np.testing.assert_array_equal(w_dual[unseen], e["admit"][j[unseen]])      # dual rule: r~ - tau v > 0
    np.testing.assert_array_equal(w_dual[~unseen], w_rej[~unseen])            # seen types: same thinning

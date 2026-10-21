"""Pareto tools over the objective weights (dlent_rt.pareto, week 30)."""
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import linprog

from dlent_rt import pareto
from dlent_rt.data import load_dataset
from dlent_rt.models.core.stream import build_stream
from dlent_rt.phase0 import run_phase0
from dlent_rt.reward import job_reward
from tests.synthetic import synthetic_config


@pytest.fixture(scope="module")
def phase0():
    with tempfile.TemporaryDirectory() as d, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = synthetic_config(Path(d), n0=400, n_jobs=3000, drift=True)
        ds = load_dataset(cfg)
        st = run_phase0(ds, cfg)
        S = build_stream(ds, cfg)
        return cfg, ds, st, S


def test_linear_combination_reproduces_the_reward(phase0):
    cfg, ds, st, S = phase0
    assert (cfg.objective.lambda1, cfg.objective.lambda2, cfg.objective.lambda3) == (1.0, 1.0, 1.0)
    R = job_reward(S, st, cfg.objective).R
    Rc = pareto.job_components(S, st).combine((1.0, 1.0, 1.0))
    assert np.max(np.abs(Rc - R)) < 0.005 and abs(Rc.sum() - R.sum()) < 0.005       # job level, to the cent
    rt = pareto.type_components(st).combine((1.0, 1.0, 1.0))
    assert np.max(np.abs(rt - st.r_tilde)) < 0.005                                   # type level, to the cent


def test_weight_grid():
    g = pareto.weight_grid()
    assert len(g) == 16                                     # 15 simplex points (step 1/4) + centre (1, 1, 1)
    assert all(abs(sum(l) - 3.0) < 1e-12 for l in g)
    for c in (*pareto.CORNERS, pareto.CENTRE):
        assert c in g
    assert set(pareto.subset7()) <= set(g)


def _toy_fluid():
    rng = np.random.default_rng(0)
    n = 30
    p = rng.uniform(0.01, 0.1, n)
    comp = pareto.Components(CV=rng.uniform(0, 5, n), PR=rng.normal(0, 3, n), CO=rng.uniform(0, 4, n))
    v = rng.uniform(0.1, 1.0, (n, 2))
    c = np.array([0.4, 0.4])                                # binding
    return p, comp, v, c


def test_fluid_corners_are_single_objective_optima():
    p, comp, v, c = _toy_fluid()
    A = (p[:, None] * v).T
    for lam, f in zip(pareto.CORNERS, (comp.CV, comp.PR, -comp.CO)):
        _, tot, _ = pareto.fluid_lp(p, comp, v, c, lam)
        best = -linprog(-(p * f), A_ub=A, b_ub=c, bounds=(0, 1), method="highs").fun
        val = {"CV": tot["CV"], "PR": tot["PR"], "CO": -tot["CO"]}[("CV", "PR", "CO")[lam.index(3.0)]]
        assert val == pytest.approx(best, rel=1e-9, abs=1e-12)
    ideal, nadir = pareto.ideal_nadir_fluid(p, comp, v, c)
    assert np.all(ideal >= nadir)
    for k, w in enumerate(np.eye(3)):                       # Chebyshev with one weight = that objective's optimum
        _, tot, t, _ = pareto.chebyshev_fluid(p, comp, v, c, w, ideal, nadir)
        f = (tot["CV"], tot["PR"], -tot["CO"])[k]
        assert f == pytest.approx(ideal[k], rel=1e-7, abs=1e-9) and t == pytest.approx(1.0, rel=1e-7)


def test_nondominated_and_hypervolume():
    F = np.array([[1, 1, 2], [2, 2, 1], [0.5, 0.5, 0.5], [2, 2, 1]], float)
    assert pareto.nondominated(F).tolist() == [True, True, False, True]
    # union of boxes [0,1]x[0,1]x[0,2] and [0,2]x[0,2]x[0,1]: 2 + 4 - 1 = 5
    assert pareto.hypervolume_3d(F, (0, 0, 0)) == pytest.approx(5.0)
    assert pareto.hypervolume_3d(np.array([[2, 1, 1], [1, 2, 1]], float), (0, 0, 0)) == pytest.approx(3.0)
    assert pareto.hypervolume_3d(F, (5, 5, 5)) == 0.0


def test_chebyshev_finds_the_unsupported_point():
    # non-convex front: (0.4, 0.4) is Pareto-optimal but below the segment between the two corners
    F = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.4, 0.4, 0.0], [0.3, 0.3, 0.0]])
    assert pareto.nondominated(F).tolist() == [True, True, True, False]
    picked_ws = {int(np.argmax(F @ np.array([a, 1 - a, 0.0]))) for a in np.linspace(0, 1, 101)}
    assert 2 not in picked_ws                               # the weighted sum never selects it
    ideal, nadir = F.max(axis=0), F.min(axis=0)
    assert pareto.chebyshev_discrete(F, (0.5, 0.5, 0.0), ideal, nadir) == 2

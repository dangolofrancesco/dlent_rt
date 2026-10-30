"""Online loop: dollar reward (no divisor), s_C frozen across phases, realised reward on Phase 0's bid model."""
import tempfile, warnings
from pathlib import Path

import pytest

from dlent_rt.data import load_dataset
from dlent_rt.phase0 import run_phase0
from dlent_rt.models.theory import TheoryModel
from tests.synthetic import synthetic_config


class _Recording(TheoryModel):
    def run(self):
        self.states = []
        return super().run()

    def _new_phase(self, state, tau_prev, W_prev):
        self.states.append(state)
        super()._new_phase(state, tau_prev, W_prev)


def _run():
    with tempfile.TemporaryDirectory() as d:
        cfg = synthetic_config(Path(d), n0=400, n_jobs=3000, drift=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = load_dataset(cfg)
            st = run_phase0(ds, cfg)
            m = _Recording(ds, st, cfg)
            res = m.run()
    return cfg, m, res


@pytest.fixture(scope="module")
def default_run():
    return _run()


def test_reward_in_dollars_with_frozen_carbon_price(default_run):
    cfg, m, res = default_run
    assert len(m.states) == len(res.phases) >= 2
    assert all(s.carbon_price == m.state0.carbon_price for s in m.states)
    S, (l1, l2, l3), sC = m.stream, m.lam, m.state0.carbon_price
    adm = [a for a in res.arrivals if a["admit"]]
    assert adm
    for a in adm:
        i = a["i"]
        d = S.duration_hours[i]                    # hourly bid and phi x the true hours (results only)
        phi0 = float(m.state0.myerson.phi_job(S.v[i], S.cost_base[i])) * d
        assert a["phi_mech"] == pytest.approx(phi0, rel=1e-12)
        exp = (l1 * S.v[i] * d + l2 * (phi0 - S.C_elec[i])
               - l3 * sC * S.C_carbon[i])
        assert a["R"] == pytest.approx(exp, rel=1e-12, abs=1e-12)
    # the reward is unbounded: nothing clips it to [-1, 1]
    assert max(abs(a["R"]) for a in adm) > 0.1

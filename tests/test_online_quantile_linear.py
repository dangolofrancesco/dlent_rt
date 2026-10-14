"""Online loop under quantile_linear: dollar reward (no divisor), s_C frozen across phases, IR rejects."""
import tempfile, warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
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


def _run(known_distribution=True, **grid):
    with tempfile.TemporaryDirectory() as d:
        cfg = synthetic_config(Path(d), n0=400, n_jobs=3000, drift=True)
        cfg = replace(cfg, bid=replace(cfg.bid, known_distribution=known_distribution))
        if grid:
            cfg = replace(cfg, grid=replace(cfg.grid, **grid))
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
    assert m.state0.norm_info["strategy"] == "quantile_linear"
    assert len(m.states) == len(res.phases) >= 2
    assert all(s.carbon_price == m.state0.carbon_price for s in m.states)
    S, (l1, l2, l3), sC = m.stream, m.lam, m.state0.carbon_price
    adm = [a for a in res.arrivals if a["admit"]]
    assert adm
    for a in adm:
        i = a["i"]
        exp = (l1 * S.priority[i] * S.v[i] + l2 * (a["phi_mech"] - S.C_elec[i])
               - l3 * sC * S.C_carbon[i])
        assert a["R"] == pytest.approx(exp, rel=1e-12, abs=1e-12)
    # the reward is unbounded: nothing clips it to [-1, 1]
    assert max(abs(a["R"]) for a in adm) > 1.0
    assert not any(a["ir_reject"] for a in res.arrivals)       # v-space grid: every bid has a type


def test_ir_reject_never_indexes_last_type():
    # phi-space grid needs one phi(v) for every job: legacy pooled-fit ablation only
    cfg, m, res = _run(known_distribution=False, space="phi")
    ir = [a for a in res.arrivals if a["ir_reject"]]
    assert ir, "phi-space grid on lognormal bids should IR-reject the phi < floor tail"
    for a in ir:
        assert a["type"] == -1 and not a["k_hat"] and not a["admit"] and a["R"] == 0.0
        assert np.isnan(a["r_tilde"]) and np.isnan(a["shadow_cost"])
    ph = res.phases
    assert ph[0]["tau_start"] == 1 and ph[-1]["tau_end"] == res.summary["T"]

"""
Calibrated thresholds of Tests 2 and 3 (week 34; cfg.tests.threshold = "calibrated").

The paper's thresholds (Appendix A) are out of reach by 2-3 orders of magnitude on our data
(reports/week34/test_margins.md). The calibrated variant multiplies them by one constant per test:
  * Test 2: eps~_Test2 x kappa2;
  * Test 3: every confidence term x kappa3 (eps~_Test3, and eps_A' + eps_B' inside lambda~, eq. 7).
The kappas are chosen ONCE, in Phase 0, from H alone: H's jobs (build_history_stream, Phase 0's own bids) are
replayed through the no-learning model (Phase 0's state for the whole replay; the run's own switches, e.g. thinning
off and the live oracle). Phases end as online by Test 1 or timeout, never by Tests 2-3, and restart with Phase 0's
state (no ReInit). Each test's LHS / threshold ratio is traced at every tick; kappa is set so that the largest ratio
on H equals tests.calibration_target (0.5: a safety factor 2).
  * Test 2: ratio = LHS / eps~_Test2, so kappa2 = max ratio (kappa = 1) / target.
  * Test 3: ratio = LHS / (lambda~(u-1) + eps~_Test3), per window start s of the geometric grid. Between two
    arrivals the LHS and the RHS are non-increasing, so each gap is summarised as the monitor's screen does: LHS at
    the first tick / RHS at the last (an upper bound on the gap's ratios). lambda~ depends on kappa3 through a
    running minimum, so kappa3 solves max ratio(kappa3) = target by root finding (the ratio decreases in kappa3).
No stream data is read: only ds.H, Phase 0's state and the run's constants (capacity, d_max and the horizon T in
the confidence level delta, both known in advance).
"""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
from scipy.optimize import brentq


class ThresholdTrace:
    """What the monitor records for the calibration (PhaseMonitor(trace=...)), at kappa = 1: per committed tick its
    phase (tau_prev), lam(tau) and the lambda~ term eps_A' + eps_B'; per gap Test 2's largest LHS / threshold ratio;
    per gap and window start s, Test 3's screen pair (LHS at the first tick, eps~_Test3 at the last)."""

    def __init__(self):
        self._phase, self._tick, self._lam, self._c = [], [], [], []
        self._r2, self._r2_tick = [], []
        self._t3_lhs, self._t3_e3, self._t3_pos, self._t3_tick = [], [], [], []
        self._n = 0

    def add_gap(self, tau_prev, taus, lam, c, r2, t3):
        n = int(taus.size)
        self._phase.append(np.full(n, int(tau_prev), np.int64))
        self._tick.append(np.asarray(taus, np.int64))
        self._lam.append(np.asarray(lam, float))
        self._c.append(np.broadcast_to(np.asarray(c, float), (n,)).copy())
        self._n += n
        end = self._n - 1                          # the gap's last committed tick: lambda~(u-1) of the screen
        if np.isfinite(r2):
            self._r2.append(float(r2))
            self._r2_tick.append(int(taus[-1]) + 1)
        for lhs, e3 in t3 or ():
            self._t3_lhs.append(lhs)
            self._t3_e3.append(e3)
            self._t3_pos.append(end)
            self._t3_tick.append(int(taus[-1]) + 1)

    def finish(self) -> "ThresholdTrace":
        cat = lambda x, d: np.concatenate(x) if x else np.zeros(0, d)
        self.phase, self.tick = cat(self._phase, np.int64), cat(self._tick, np.int64)
        self.lam, self.c = cat(self._lam, float), cat(self._c, float)
        self.r2, self.r2_tick = np.asarray(self._r2, float), np.asarray(self._r2_tick, np.int64)
        self.t3_lhs, self.t3_e3 = np.asarray(self._t3_lhs, float), np.asarray(self._t3_e3, float)
        self.t3_pos, self.t3_tick = np.asarray(self._t3_pos, np.int64), np.asarray(self._t3_tick, np.int64)
        return self

    def lam_tilde(self, kappa3: float) -> np.ndarray:
        """lambda~ at every traced tick: the running min over the phase of lam + kappa3 (eps_A' + eps_B')."""
        x = self.lam + kappa3 * self.c
        out = np.empty_like(x)
        cut = np.flatnonzero(np.diff(self.phase)) + 1
        for a, b in zip(np.r_[0, cut], np.r_[cut, x.size]):
            out[a:b] = np.minimum.accumulate(x[a:b])
        return out

    def ratio2(self, kappa2: float = 1.0) -> np.ndarray:
        return self.r2 / kappa2

    def ratio3(self, kappa3: float = 1.0) -> np.ndarray:
        return self.t3_lhs / (self.lam_tilde(kappa3)[self.t3_pos] + kappa3 * self.t3_e3)


def kappas_from_trace(tr: ThresholdTrace, target: float) -> dict:
    """kappa2, kappa3 such that the largest LHS / threshold ratio on the trace is `target`, for each test."""
    m2 = float(np.max(tr.r2)) if tr.r2.size else np.nan
    if not m2 > 0:
        raise ValueError(f"Test 2's LHS is never positive on H (max ratio {m2}): kappa2 is undefined")
    k2 = m2 / target
    if not tr.t3_lhs.size or not np.max(tr.t3_lhs) > 0:
        raise ValueError("Test 3's LHS is never positive on H: kappa3 is undefined")
    f = lambda lk: float(np.max(tr.ratio3(np.exp(lk)))) - target
    lo, hi = np.log(1e-12), np.log(1e6)
    if f(lo) < 0 or f(hi) > 0:
        raise ValueError("kappa3 outside [1e-12, 1e6]")
    k3 = float(np.exp(brentq(f, lo, hi, xtol=1e-12, rtol=1e-10)))
    return dict(kappa2=float(k2), kappa3=k3, max_ratio2_theory=m2, max_ratio3_theory=float(np.max(tr.ratio3(1.0))))


def replay_history(ds, state0, cfg):
    """H replayed through the no-learning model with the run's switches (Tests 2-3 traced, never ending a phase).
    Returns (ThresholdTrace, list of H's phases as dicts)."""
    from ..theory import TheoryModel
    from .monitor import PhaseMonitor
    from .stream import build_history_stream

    c = replace(cfg, tests=replace(cfg.tests, threshold="theoretical", kappa_test2=None, kappa_test3=None),
                logging=replace(cfg.logging, progress_every=0.0, progress_phases=False))

    class HistoryReplay(TheoryModel):
        kind = "calibration replay"

        def _new_phase(self, state, tau_prev, W_prev):
            super()._new_phase(state, tau_prev, W_prev)
            self.mon = PhaseMonitor(self.ep, state.r_tilde, state.v, self.c, self.a_max_eff, tau_prev, W_prev,
                                    self.d_max, self.cfg, r_unit=self.r_unit, trace=trace, tests23=False)

        def _end_phase(self, tau_m, reason, reinit=True):
            phases.append(dict(m=self.m, tau_start=self.tau_prev + 1, tau_end=int(tau_m), W_prev=self.W_prev,
                               reason=reason, n_arrivals=len(self.ph_arr), n_admitted=len(self.ph_adm)))
            if reinit:                              # no learning: the next phase starts from Phase 0's state again
                self.m += 1
                self._new_phase(self.state0, tau_m, tau_m - self.tau_prev)

    trace, phases = ThresholdTrace(), []
    SH = build_history_stream(ds, state0, c)
    m = HistoryReplay(ds, state0, c, stream=SH)
    m._summary = lambda: {}
    m.ds = replace(ds, T=int(ds.H_span_steps))      # the replay's horizon: H's span (run() ends the last phase there)
    m.run()
    return trace.finish(), phases


def calibrate(ds, state0, cfg) -> dict:
    """kappa2, kappa3 of the run: tests.kappa_test2 / kappa_test3 if given, else from the H replay."""
    t = cfg.tests
    out = dict(kappa2=t.kappa_test2, kappa3=t.kappa_test3, target=t.calibration_target, source="config")
    if t.kappa_test2 is not None and t.kappa_test3 is not None:
        return out
    t0 = time.perf_counter()
    tr, phases = replay_history(ds, state0, cfg)
    k = kappas_from_trace(tr, t.calibration_target)
    out.update(kappa2=t.kappa_test2 if t.kappa_test2 is not None else k["kappa2"],
               kappa3=t.kappa_test3 if t.kappa_test3 is not None else k["kappa3"], source="H",
               max_ratio2_theory=k["max_ratio2_theory"], max_ratio3_theory=k["max_ratio3_theory"],
               H_phases=[(p["tau_end"], p["reason"]) for p in phases], replay_s=time.perf_counter() - t0)
    return out

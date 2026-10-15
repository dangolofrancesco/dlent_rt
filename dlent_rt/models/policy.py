"""
Policy hook (week-30 Stage D): ONE event-driven loop for every admission policy, with the
same physical-capacity gate, so policies differ only in their admission rule.

    simulate(stream, occ, policy), for each arrival i in arrival order (tick t):
        policy.before_arrival(i, t)     e.g. DLENT-RT scans Tests 1-3 / restarts phases up to t
        occ.release_until(t)            physical releases of the jobs that finished
        want  = policy.want(i, t)       the policy's admission wish (oracle, thinning, ...)
        fits  = occ.fits(A_i)           physical gate: U + A_i <= c
        admit = want and fits           if admitted: occ.add(i, A_i, t, D_i)
        policy.record(i, t, want, fits, admit)

Policies:
  * TheoryModel (DLENT-RT, dlent_rt.models.theory): oracle on the phase's type reward and
    shadow prices, Bernoulli thinning, Tests 1-3 and ReInit.
  * StaticPolicy: the FROZEN (r~, tau, v) of one PhaseState (e.g. Phase 0 or the full-stream
    hindsight state), want = 1[r~_j - tau^T v_j > 0] (optionally thinned with the online run's
    common random numbers); no learning, no tests, no restarts.
  * Greedy policies: TO BE ADDED (not implemented). Two myopic first-come rules, under the
    same gate, to measure what the shadow prices (and learning) add:
      G1  type-level reward only: admit when the arrival's type reward r~_j > 0 (no tau);
      G2  informed: admit when the job's own R_job > 0, with the known bid distribution F
          (phi_job of the known markup lognormal, Phase 0's frozen s_C).
    Each plugs in by implementing the three methods of Policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from ..phase0 import PhaseState, assign_type_ids
from .core.occupancy import Occupancy


class Policy(Protocol):
    def before_arrival(self, i: int, t: int) -> None: ...
    def want(self, i: int, t: int) -> bool: ...
    def record(self, i: int, t: int, want: bool, fits: bool, admit: bool) -> None: ...


@dataclass
class SimResult:
    admit: np.ndarray            # (n,) bool: admitted
    want: np.ndarray             # (n,) bool: the policy wanted to admit
    phys_reject: np.ndarray      # (n,) bool: wanted but blocked by the physical gate


def simulate(stream, occ: Occupancy, policy: Policy) -> SimResult:
    """Run `policy` over every arrival of `stream` with the physical-capacity gate of `occ`."""
    n = stream.n
    admit, want_a, phys = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
    for i in range(n):
        t = int(stream.tick[i])
        policy.before_arrival(i, t)
        occ.release_until(t)
        want = bool(policy.want(i, t))
        fits = occ.fits(stream.A[i])
        ok = want and fits
        if ok:
            occ.add(i, stream.A[i], t, int(stream.D[i]))
        policy.record(i, t, want, fits, ok)
        admit[i], want_a[i], phys[i] = ok, want, want and not fits
    return SimResult(admit=admit, want=want_a, phys_reject=phys)


class StaticPolicy:
    """Frozen type oracle of one PhaseState: want_i = 1[r~_j - tau^T v_j > 0] for the arrival's type
    j under that state's grid; thin_p = None -> no thinning, else want_i &= u_rand[i] < thin_p."""

    def __init__(self, state: PhaseState, stream, thin_p: Optional[float] = None,
                 u_rand: Optional[np.ndarray] = None):
        k = state.grid.assign(stream.v, state.myerson)
        j = assign_type_ids(state.type_space, k, stream.priority, stream.hw_idx)
        jj = np.maximum(j, 0)
        self.khat = np.where(j >= 0, state.r_tilde[jj] - state.v[jj] @ state.tau, -np.inf) > 0
        if thin_p is None:
            self.U = np.ones(stream.n, bool)
        else:
            assert u_rand is not None, "thinning needs the common random numbers u_rand"
            self.U = np.asarray(u_rand) < thin_p

    def before_arrival(self, i: int, t: int) -> None:
        pass

    def want(self, i: int, t: int) -> bool:
        return bool(self.khat[i] and self.U[i])

    def record(self, i: int, t: int, want: bool, fits: bool, admit: bool) -> None:
        pass


def run_static(stream, capacity, state: PhaseState, thin_p: Optional[float] = None,
               u_rand: Optional[np.ndarray] = None) -> SimResult:
    """StaticPolicy of `state` through the shared event-driven loop (no occupancy log)."""
    return simulate(stream, Occupancy(capacity, record=False),
                    StaticPolicy(state, stream, thin_p=thin_p, u_rand=u_rand))

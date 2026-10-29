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
  * FractionPolicy: the static Phase 0 policy (D6 benchmark): each arrival admitted with its type's LP-RS fraction
    x_j; types with no job in H by the dual rule with fallback estimates (D4).
  * NoLearningPolicy: the no-learning model (D5) = the Phase 0 state forever: FractionPolicy, then the
    theoretical model's Bernoulli thinning of phase 1 (p_U(W^(0), t)), never re-estimated.
  * Greedy policies (practical only: code kept, not run in the model check): two myopic first-come
    rules under the same gate. No tau, no LP, no thinning, no tests.
      G1 TypeGreedyPolicy  type-level: want = 1[r~_j > 0] for the arrival's type j (Phase-0 grid
          and types, never refit); r~_j starts at Phase 0's value (H) and is the cumulative
          running mean of R_job over H and ALL stream arrivals seen so far (admitted or not),
          updated after each decision. A type never observed is rejected ("unseen type").
      G2 JobGreedyPolicy   informed: want = 1[R_job > 0], R_job of dlent_rt.reward (phi_job under
          the known F, Phase 0's frozen s_C, the job's realised energy). No types.
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
        k = state.grid.assign(stream.v)
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


# --------------------------------------------------------------------------- #
# Greedy policies (week 30)
# --------------------------------------------------------------------------- #
REJ_NONE, REJ_R, REJ_UNSEEN, REJ_NOTYPE, REJ_PHYS = 0, 1, 2, 3, 4
REJ_NAMES = {REJ_R: "r<=0", REJ_UNSEEN: "unseen type", REJ_NOTYPE: "no type (IR)", REJ_PHYS: "physical"}


class TypeGreedyPolicy:
    """
    G1: want_i = 1[r~_j > 0] for the arrival's type j = tid[i]. r~ starts at r0 with n0
    observations per type (Phase 0: r0 = r~ from H, n0 = H counts) and, if update, absorbs
    every arrival after its decision: n_j += 1, r~_j += (R_i - r~_j) / n_j (cumulative mean
    of R_job, no discounting). A type with n_j = 0 is rejected as "unseen type"; tid = -1
    (no type) as "no type (IR)". Logged per arrival: reason (REJ_*), r_dec (r~_j used, NaN
    when unseen / no type).
    """

    def __init__(self, tid: np.ndarray, R: np.ndarray, r0: np.ndarray, n0: np.ndarray,
                 update: bool = True):
        self.tid = np.asarray(tid, np.int64)
        self.R = np.asarray(R, float)
        self.r = np.array(r0, float)
        self.n = np.array(n0, np.int64)
        self.update = update
        n = self.tid.size
        self.reason = np.zeros(n, np.int8)
        self.r_dec = np.full(n, np.nan)

    @classmethod
    def from_state(cls, state: PhaseState, stream, R: np.ndarray, H_span_steps: int,
                   update: bool = True) -> "TypeGreedyPolicy":
        """Phase-0 types (grid of `state`, not refit), r0 = state.r_tilde, n0 = H counts
        (p_j = N_j / H_span_steps, as in Phase 0)."""
        k = state.grid.assign(stream.v)
        tid = assign_type_ids(state.type_space, k, stream.priority, stream.hw_idx)
        n0 = np.rint(np.asarray(state.p) * H_span_steps).astype(np.int64)
        return cls(tid, R, state.r_tilde, n0, update=update)

    def before_arrival(self, i: int, t: int) -> None:
        pass

    def want(self, i: int, t: int) -> bool:
        j = self.tid[i]
        if j < 0:
            self.reason[i] = REJ_NOTYPE
            return False
        if self.n[j] == 0:
            self.reason[i] = REJ_UNSEEN
            return False
        self.r_dec[i] = self.r[j]
        if self.r[j] <= 0:
            self.reason[i] = REJ_R
            return False
        return True

    def record(self, i: int, t: int, want: bool, fits: bool, admit: bool) -> None:
        if want and not fits:
            self.reason[i] = REJ_PHYS
        j = self.tid[i]
        if self.update and j >= 0:
            self.n[j] += 1
            self.r[j] += (self.R[i] - self.r[j]) / self.n[j]


class JobGreedyPolicy:
    """G2: want_i = 1[R_i > 0] on the job's own reward R_job (no types)."""

    def __init__(self, R: np.ndarray):
        self.R = np.asarray(R, float)
        self.reason = np.zeros(self.R.size, np.int8)

    def before_arrival(self, i: int, t: int) -> None:
        pass

    def want(self, i: int, t: int) -> bool:
        if self.R[i] > 0:
            return True
        self.reason[i] = REJ_R
        return False

    def record(self, i: int, t: int, want: bool, fits: bool, admit: bool) -> None:
        if want and not fits:
            self.reason[i] = REJ_PHYS


def run_policy(stream, capacity, policy) -> SimResult:
    """Any Policy through the shared event-driven loop (no occupancy log)."""
    return simulate(stream, Occupancy(capacity, record=False), policy)


class FractionPolicy:
    """Static policy with the LP's admission FRACTIONS: want_i = 1[u_i < x_j] for the arrival's type j under
    `state`'s grid (u_i uniform, fixed seed). x_j in [0, 1] is the LP-RS solution of that state. No learning, no
    tests; the physical gate is applied by simulate."""

    def __init__(self, state: PhaseState, stream, x: np.ndarray, seed: int = 0,
                 admit_unseen: Optional[np.ndarray] = None):
        """admit_unseen: per-type bool of the dual rule (theory_eval.unseen_type_estimates), used for the arrivals
        whose type has no job in the state (p = 0); None = reject them."""
        k = state.grid.assign(stream.v)
        j = assign_type_ids(state.type_space, k, stream.priority, stream.hw_idx)
        jj = np.maximum(j, 0)
        seen = (j >= 0) & (np.asarray(state.p)[jj] > 0)
        self.x_job = np.where(seen, np.asarray(x, float)[jj], 0.0)
        self.unseen = (j >= 0) & ~seen
        self.dual = (np.zeros(stream.n, bool) if admit_unseen is None
                     else self.unseen & np.asarray(admit_unseen, bool)[jj])
        self.u = np.random.default_rng(seed).random(stream.n)

    def before_arrival(self, i: int, t: int) -> None:
        pass

    def want(self, i: int, t: int) -> bool:
        return bool(self.dual[i] or self.u[i] < self.x_job[i])

    def record(self, i: int, t: int, want: bool, fits: bool, admit: bool) -> None:
        pass


class NoLearningPolicy(FractionPolicy):
    """The no-learning model: FractionPolicy of the Phase 0 state, then Bernoulli thinning with p_U(W0, t) of the
    theoretical model's first phase (tau^(0) = 0, W^(0) = W0 = d_max) and its common random numbers u_rand. No
    tests, no ReInit: the Phase 0 state (grid, sigma = bid.sigma, r~, tau, x) is kept for the whole stream."""

    def __init__(self, state: PhaseState, stream, x: np.ndarray, ep, u_rand: np.ndarray, W0: int,
                 seed: int = 0, admit_unseen: Optional[np.ndarray] = None):
        super().__init__(state, stream, x, seed=seed, admit_unseen=admit_unseen)
        self.ep, self.u_rand, self.W0 = ep, np.asarray(u_rand), int(W0)

    def want(self, i: int, t: int) -> bool:
        return super().want(i, t) and bool(self.u_rand[i] < self.ep.p_U(self.W0, t))

"""
Nonstationarity tests of DLENT (Zhang & Cheung 2025, Algorithm 1 lines 8-15),
evaluated EXACTLY at every tick, including the ~99% null ticks, while only
doing work at arrivals.

Time conventions (paper): ticks are 1-based; phase m starts at tau_prev + 1,
tau_bar = tau_prev + d_max. At tick u the tests use information up to u-1 and
are active for u > tau_bar + 1. If a test fails (or u > tau_prev + 2 W_prev)
the phase ends at tau^(m) = u - 1 and tick u is the first tick of phase m+1.

Between two arrivals the window counts are constant, so for every tick:
  * lam(tau) = lam_hat^(tau_prev+1 : tau) comes from the exact piecewise
    reconstruction of the parametric LP (window_lp.WindowLP);
  * Test 2's running sum and lambda~ (eq. 7) are vectorised cumulative ops;
  * Test 1's LHS only increases at admissions and is non-increasing between
    them, so its maximum over a gap is at the first active tick -> one check;
  * Test 3: for a fixed s the LHS lam_hat^(s+1:u-1) is non-increasing in u
    within a gap and the RHS lambda~(u-1) + eps_T3 is non-increasing too, so
    "LHS(first u) <= RHS(last u)" certifies the whole gap (1 LP per s); only
    if that screen fails we reconstruct the gap exactly, tick by tick.
Windows (s+1 : u-1) that contain no arrival have lam_hat = 0 and cannot fail.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .error_params import ErrorParams
from .occupancy import WindowOccupancy
from .window_lp import WindowLP


@dataclass
class ScanResult:
    fail_tick: Optional[int]           # u*: phase ends at u*-1 (None = no failure in the gap)
    reason: str = ""                   # "test1" | "test2" | "test3" | "timeout"
    diag: dict = field(default_factory=dict)   # values at the last tick of the gap


def geometric_offsets(base: int, max_off: int) -> np.ndarray:
    offs = [0]
    k = 1
    while k <= max_off:
        offs.append(k)
        k *= base
    return np.array(offs, dtype=np.int64)


class PhaseMonitor:
    def __init__(self, ep: ErrorParams, r_tilde, v, capacity, a_max_eff,
                 tau_prev: int, W_prev: int, d_max: int, cfg):
        if cfg.test3.s_grid != "geometric":
            raise NotImplementedError(
                "test3.s_grid='full' needs O(n) LPs per tick; the exact event-driven "
                "monitor supports the geometric grid only.")
        self.ep = ep
        self.c = np.asarray(capacity, float)
        self.a_max_eff = np.asarray(a_max_eff, float)
        self.tau_prev, self.W_prev = int(tau_prev), int(W_prev)
        self.tau_bar = self.tau_prev + int(d_max)
        self.base = cfg.test3.geometric_base
        self.wlp = WindowLP(r_tilde, v, capacity, tol=cfg.tests.piecewise_tol)
        self.fac = 1.0 - ep.eps_D - ep.xi
        self.arr_ticks: list[int] = []
        self.arr_types: list[int] = []
        self.R: dict[int, float] = {}
        self.t1 = WindowOccupancy(len(self.c))
        self.last = self.tau_prev            # ticks <= last are committed
        self.S2 = 0.0                        # sum_{tau=tau_bar+1}^{last} term(tau)
        self.lam_tilde = np.inf              # min_{tau_prev < tau <= last} (lam + eps_A' + eps_B')
        self.sum_lam = 0.0                   # sum_{tau_prev < tau <= last} lam(tau)  (diagnostic)

    # ------------------------------------------------------------------ #
    def register_arrival(self, t: int, type_id: int, R: float,
                         admitted: bool, job: int, A: np.ndarray, D: int):
        assert t == self.last + 1, "scan() must commit all ticks before the arrival"
        self.arr_ticks.append(int(t))
        self.arr_types.append(int(type_id))
        self.R[int(t)] = float(R)
        if admitted and t >= self.tau_bar + 1:
            self.t1.add(job, A, t, D)

    def _counts(self, s_excl: int, hi_incl: int):
        """Type counts of phase arrivals with s_excl < tick <= hi_incl."""
        tk = np.asarray(self.arr_ticks, np.int64)
        lo = np.searchsorted(tk, s_excl, side="right")
        hi = np.searchsorted(tk, hi_incl, side="right")
        ty = np.asarray(self.arr_types[lo:hi], np.int64)
        ty = ty[ty >= 0]                     # IR rejects (type -1) belong to no type
        if ty.size == 0:
            return None
        return np.unique(ty, return_counts=True)

    # ------------------------------------------------------------------ #
    def scan(self, b: int) -> ScanResult:
        """
        Advance through ticks: compute lam(tau) for tau in [last+1, b-1] and run
        the tests for u in [last+2, b]. b is the next arrival tick (or T+1).
        """
        ep = self.ep
        lo, hi = self.last + 1, b - 1
        if hi < lo:
            return ScanResult(None)
        taus = np.arange(lo, hi + 1, dtype=np.int64)
        n_tau = taus - self.tau_prev

        cnt = self._counts(self.tau_prev, lo)
        if cnt is None:
            lam = np.zeros(taus.size)
        else:
            self.wlp.set_counts(*cnt)
            lam = self.wlp.lam_on_integers(int(n_tau[0]), int(n_tau[-1]))

        R = np.zeros(taus.size)
        R[0] = self.R.get(lo, 0.0)
        active_tau = taus >= self.tau_bar + 1
        term = np.where(active_tau, lam * self.fac - R, 0.0)
        S2_c = self.S2 + np.cumsum(term)
        lt_c = np.minimum.accumulate(
            np.minimum(self.lam_tilde, lam + ep.lam_tilde_term(n_tau)))

        # checks at u = tau + 1, u in [lo+1, b]
        u = taus + 1
        fails: list[tuple[int, str]] = []

        timeout = self.tau_prev + 2 * self.W_prev + 1       # first u with u > tau_prev + 2W
        if lo + 1 <= timeout <= b:
            fails.append((timeout, "timeout"))

        act = u > self.tau_bar + 1
        lhs2 = thr2 = None
        if np.any(act):
            den = (u - self.tau_bar - 1).astype(float)
            lhs2 = np.divide(S2_c, den, out=np.zeros_like(S2_c), where=den > 0)
            thr2 = ep.eps_test2(self.W_prev, n_tau)          # eps~_Test2(u-1), n = u-1-tau_prev
            bad = act & (lhs2 > thr2)
            if np.any(bad):
                fails.append((int(u[np.argmax(bad)]), "test2"))

            u0 = int(u[np.argmax(act)])                       # first active tick of the gap
            val1 = self.t1.value_at(u0)
            if np.any(val1 > self.c - self.a_max_eff + 1e-12):
                fails.append((u0, "test1"))

            f3, m3 = self._test3(u0, b, lo, lam, lt_c, taus)
            if f3 is not None:
                fails.append((f3, "test3"))
        else:
            m3 = np.nan

        diag = dict(lam_phase=float(lam[-1]), lam_tilde=float(lt_c[-1]),
                    test2_lhs=float(lhs2[-1]) if lhs2 is not None else np.nan,
                    test2_thr=float(thr2[-1]) if thr2 is not None else np.nan,
                    test3_screen_margin=float(m3))

        if fails:
            u_star, reason = min(fails)
            k = u_star - 1 - lo                               # commit taus[0..k] (tau <= u*-1)
            self._commit(S2_c[k], lt_c[k], lam[: k + 1].sum(), u_star - 1)
            return ScanResult(u_star, reason, diag)
        self._commit(S2_c[-1], lt_c[-1], lam.sum(), hi)
        return ScanResult(None, "", diag)

    def _commit(self, S2, lt, lam_sum, last):
        self.S2, self.lam_tilde, self.last = float(S2), float(lt), int(last)
        self.sum_lam += float(lam_sum)

    # ------------------------------------------------------------------ #
    def _test3(self, u0: int, b: int, lo: int, lam, lt_c, taus):
        """First failing u in [u0, b] for the geometric s-grid, and screen margin."""
        ep = self.ep
        offs = geometric_offsets(self.base, max(b - 2 - self.tau_prev, 0))
        first_fail, margin = None, np.inf
        us = np.arange(u0, b + 1)
        lt_u = lt_c[us - 1 - taus[0]]                                  # lambda~(u-1)
        for off in offs:
            s = self.tau_prev + int(off)
            if s >= lo or s > u0 - 2:          # window without arrivals (lam=0) / not yet valid
                continue
            n_u = us - 1 - s
            rhs = lt_u + ep.eps_test3(n_u)
            if s == self.tau_prev:             # phase window: lam already known per tick
                lhs = lam[us - 1 - taus[0]]
                margin = min(margin, float(rhs[-1] - lhs[0]))
                bad = lhs > rhs
            else:
                cnt = self._counts(s, lo)
                if cnt is None:
                    continue
                self.wlp.set_counts(*cnt)
                lhs0 = self.wlp.lam(float(n_u[0]))
                margin = min(margin, float(rhs[-1] - lhs0))
                if lhs0 <= rhs[-1]:            # monotone screen certifies the whole gap
                    continue
                lhs = self.wlp.lam_on_integers(int(n_u[0]), int(n_u[-1]))
                bad = lhs > rhs
            if np.any(bad):
                uf = int(us[np.argmax(bad)])
                first_fail = uf if first_fail is None else min(first_fail, uf)
        return first_fail, (margin if np.isfinite(margin) else np.nan)

"""Literal tick-by-tick reference implementation of the tests (no event-driven tricks)."""
import numpy as np
from dlent_rt.lp_oracle import solve_lp_rs
from dlent_rt.models.core.monitor import ScanResult, geometric_offsets


class BruteMonitor:
    def __init__(self, ep, r_tilde, v, capacity, a_max_eff, tau_prev, W_prev, d_max, cfg, r_unit=1.0):
        self.r_unit = float(r_unit)
        self.ep, self.r, self.v = ep, np.asarray(r_tilde) / self.r_unit, np.asarray(v)
        self.c, self.amax = np.asarray(capacity, float), np.asarray(a_max_eff, float)
        self.tau_prev, self.W_prev, self.tau_bar = tau_prev, W_prev, tau_prev + d_max
        self.base = cfg.test3.geometric_base
        self.fac = 1 - ep.eps_D - ep.xi
        self.ticks, self.types, self.R, self.adm = [], [], {}, []
        self.last = tau_prev
        self.sum_lam = 0.0
        class _W: n_solves = 0
        self.wlp = _W()
        self._cache = {}

    def register_arrival(self, t, type_id, R, admitted, job, A, D):
        assert t == self.last + 1
        self.ticks.append(t); self.types.append(type_id); self.R[t] = R / self.r_unit
        if admitted and t >= self.tau_bar + 1:
            self.adm.append((t, np.asarray(A, float), D))

    def lamhat(self, s, t):            # window [s+1 : t]
        key = (s, t)
        if key not in self._cache:
            n = t - s
            p = np.zeros(len(self.r))
            for tk, ty in zip(self.ticks, self.types):
                if s < tk <= t and ty >= 0:
                    p[ty] += 1.0 / n
            self._cache[key] = solve_lp_rs(p, self.r, self.v, self.c, presolve=False).optimum if p.any() else 0.0
        return self._cache[key]

    def scan(self, b):
        ep = self.ep
        for u in range(self.last + 2, b + 1):
            reason = None
            if u > self.tau_prev + 2 * self.W_prev:
                reason = "timeout"
            elif u > self.tau_bar + 1:
                S1 = sum((A for (tau, A, D) in self.adm if tau <= u - 1 and D >= u - tau),
                         np.zeros_like(self.c))
                t1 = np.any(S1 > self.c - self.amax + 1e-12)
                lhs2 = sum(self.lamhat(self.tau_prev, tau) * self.fac - self.R.get(tau, 0.0)
                           for tau in range(self.tau_bar + 1, u)) / (u - self.tau_bar - 1)
                t2 = lhs2 > ep.eps_test2(self.W_prev, u - 1 - self.tau_prev)
                lt = min(self.lamhat(self.tau_prev, tau) + ep.lam_tilde_term(tau - self.tau_prev)
                         for tau in range(self.tau_prev + 1, u))
                t3 = False
                for off in geometric_offsets(self.base, u - 2 - self.tau_prev):
                    s = self.tau_prev + int(off)
                    if s > u - 2: continue
                    if self.lamhat(s, u - 1) > lt + ep.eps_test3(u - 1 - s):
                        t3 = True
                reason = "test1" if t1 else "test2" if t2 else "test3" if t3 else None
            if reason:
                self.last = u - 1
                return ScanResult(u, reason, {})
        self.last = b - 1
        return ScanResult(None, "", {})

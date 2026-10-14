"""
Error parameters of DLENT, transcribed from Zhang & Cheung (2025),
"Allocation of Reusable Resources in Nonstationary Environments",
Mathematics of Operations Research, Appendix A, eqs. (A.1)-(A.13).

Every function takes the WINDOW LENGTH n = t - s of the window [s+1 : t]
(vectorised over numpy arrays). Nothing here is tuned: the only deviations
from the paper are explicit and switchable:
  * gamma / xi may be forced (cfg.constants.*_override);
  * eps_E (A.8) and eps_F (A.9) are 0 in the Exact-LP variant
    (cfg.tests.drop_dual_learning_terms, week 17): they bound the error of the
    stochastic dual-learning Algorithm 2, which we replace by the exact LP-RS.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ErrorParams:
    gamma: float
    xi: float
    c_min: float
    n_res: int          # |I|
    T: int
    delta: float
    drop_EF: bool = True
    # provenance (logged)
    gamma_data: float = float("nan")
    xi_data: float = float("nan")

    # log terms
    @property
    def log_I(self) -> float:    # log(|I|/delta)
        return float(np.log(self.n_res / self.delta))

    @property
    def log_T(self) -> float:    # log(T/delta)
        return float(np.log(self.T / self.delta))

    @property
    def log_1(self) -> float:    # log(1/delta)
        return float(np.log(1.0 / self.delta))

    # (A.1)-(A.3), (A.10), (A.11)
    def eps_Ap(self, n):
        n = np.asarray(n, float)
        g2c = 2.0 * self.gamma ** 2 / self.c_min
        return g2c * np.sqrt(2.0 / n * self.log_I) + g2c / n * self.log_I

    def eps_Bp(self, n):
        n = np.asarray(n, float)
        return self.gamma * np.sqrt(2.0 / n * self.log_T)

    def eps_Cp(self, n):
        n = np.asarray(n, float)
        return (2.0 * self.gamma * np.sqrt(2.0 / n * self.log_1)
                + 2.0 * self.gamma / n * self.log_1)

    def eps_G(self, n):
        n = np.asarray(n, float)
        return self.gamma * np.sqrt(2.0 / n * self.log_1)

    def eta(self, n):
        n = np.asarray(n, float)
        return self.gamma / (self.c_min * np.sqrt(n))

    # (A.7)
    @property
    def eps_D(self) -> float:
        return 2.0 * np.sqrt(self.xi * self.log_I) + 2.0 * self.xi * self.log_I

    # (A.8), (A.9): Exact-LP variant only
    def eps_E(self, W_prev: int, W_cur: int) -> float:
        if self.drop_EF:
            return 0.0
        raise NotImplementedError(
            "eps_E (A.8) belongs to the dual-learning Algorithm 2, which DLENT-RT "
            "replaces by the exact LP-RS; set tests.drop_dual_learning_terms=true.")

    def eps_F(self, W_prev: int, W_cur: int) -> float:
        if self.drop_EF:
            return 0.0
        raise NotImplementedError("see eps_E")

    # thinning (Algorithm 1, line 16) with eps~_U from Appendix A
    def eps_U(self, W_prev: int, t_minus_tau: int) -> float:
        return float(self.eps_D + self.eps_E(W_prev, 0)
                     + self.eta(W_prev) + self.eta(t_minus_tau))

    def p_U(self, W_prev: int, t_minus_tau: int) -> float:
        return float((1.0 - self.xi) / (1.0 + self.eps_U(W_prev, t_minus_tau)))

    # Test 2 (A.12): eps~_Test2^(m)(t) with CI^(1)(tau_{m-1}+1 : t), n = t - tau_{m-1}
    def CI1(self, n):
        n = np.asarray(n, float)
        return (2.0 * self.eps_Cp(n) + self.eps_G(n)
                + 2.0 * self.gamma * self.eta(n) + 4.0 * self.gamma / np.sqrt(n))

    def eps_test2(self, W_prev: int, n):
        return (self.gamma * self.eps_E(W_prev, 0) + self.eps_F(W_prev, 0)
                + self.gamma * self.eta(W_prev) + 4.0 * self.gamma / np.sqrt(W_prev)
                + self.CI1(n))

    # Test 3 (A.13), n = t - s
    def eps_test3(self, n):
        n = np.asarray(n, float)
        return (self.eps_Ap(n) + self.eps_Bp(n) + self.eps_Cp(n)
                + 2.0 * self.gamma ** 2 / np.sqrt(n))

    # eq. (7): lambda~ = min_tau { lam^(s+1:tau) + eps_A' + eps_B' }
    def lam_tilde_term(self, n):
        return self.eps_Ap(n) + self.eps_Bp(n)

    def as_dict(self) -> dict:
        return dict(gamma=self.gamma, xi=self.xi, gamma_data=self.gamma_data,
                    xi_data=self.xi_data, c_min=self.c_min, n_res=self.n_res,
                    T=self.T, delta=self.delta, eps_D=self.eps_D,
                    drop_EF=self.drop_EF)


def make_error_params(state, ds, cfg) -> ErrorParams:
    """Build ErrorParams from Phase 0 output + config overrides."""
    T = int(ds.T)
    delta = cfg.confidence.delta
    if delta is None:
        delta = 1.0 / (T ** cfg.confidence.delta_power)
    gamma = cfg.constants.gamma_override if cfg.constants.gamma_override is not None else state.gamma
    xi = cfg.constants.xi_override if cfg.constants.xi_override is not None else state.xi
    return ErrorParams(
        gamma=float(gamma), xi=float(xi), c_min=float(np.min(ds.capacity)),
        n_res=int(ds.n_resources), T=T, delta=float(delta),
        drop_EF=cfg.tests.drop_dual_learning_terms,
        gamma_data=float(state.gamma), xi_data=float(state.xi),
    )

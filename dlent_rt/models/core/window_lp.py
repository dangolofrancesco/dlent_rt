"""
LP-RS on a window [s+1 : t] and its exact evaluation between arrivals.

Reformulation. With n = t - s and N_j = #arrivals of type j in the window,
p_hat_j = N_j / n. Substituting z_j = n * p_hat_j * x_j = N_j * x_j:

    lam_hat(n; N) = (1/n) * g(n; N),
    g(n; N) = max { r~^T z : V^T z <= n * c,  0 <= z_j <= N_j }.

Only column BOUNDS (N) and the row RHS (n c) change from window to window, so
one persistent HiGHS model per phase is hot-started (dual simplex keeps the
basis across bound changes).

Between two arrivals N is fixed and only n moves. g(.; N) is concave and
piecewise affine in n (parametric RHS along the ray c), and by LP duality

    lam_hat(n) = min_{y >= 0} [ y^T c + (1/n) sum_j N_j (r~_j - y^T v_j)^+ ],

which is non-increasing in n. We reconstruct g exactly with the
Eisner-Severance (1976) sandwich method: supporting lines from the optimal
duals at the two ends; if they coincide g is affine, otherwise solve at their
intersection and recurse. #LP solves = O(#breakpoints). HiGHS per-row ranging
is not used because the RHS moves along a DIRECTION (both rows at once),
which single-row ranging does not cover.

Ref: M. J. Eisner and D. G. Severance (1976), "Mathematical techniques for
efficient record segmentation in large shared databases", J. ACM 23(4).
"""
from __future__ import annotations

from dataclasses import dataclass

import highspy
import numpy as np


@dataclass
class Piece:
    n_lo: float
    n_hi: float
    a: float        # g(n) = a + b n on [n_lo, n_hi]
    b: float


class WindowLP:
    def __init__(self, r: np.ndarray, V: np.ndarray, c: np.ndarray,
                 tol: float = 1e-9):
        r = np.asarray(r, float)
        V = np.asarray(V, float)
        self.c = np.asarray(c, float)
        self.n_res = len(self.c)
        self.tol = tol
        # types with r~ <= 0 never enter an optimal solution (V >= 0, z >= 0)
        self.cols = np.where(r > 0)[0]
        self.col_of_type = np.full(len(r), -1, dtype=np.int64)
        self.col_of_type[self.cols] = np.arange(len(self.cols))
        self.r = r[self.cols]
        self.V = V[self.cols]
        self.n_solves = 0
        self._ub = np.zeros(len(self.cols))
        self._h = None
        if len(self.cols):
            self._build()

    def _build(self):
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("presolve", "off")      # keep the basis for hot starts
        inf = highspy.kHighsInf
        nc = len(self.cols)
        lp = highspy.HighsLp()
        lp.num_col_, lp.num_row_ = nc, self.n_res
        lp.sense_ = highspy.ObjSense.kMaximize
        lp.col_cost_ = self.r.tolist()
        lp.col_lower_ = [0.0] * nc
        lp.col_upper_ = [0.0] * nc
        lp.row_lower_ = [-inf] * self.n_res
        lp.row_upper_ = [0.0] * self.n_res
        A = self.V.T                                    # (|I|, nc), dense is fine: |I| = 2
        starts, idx, val = [0], [], []
        for j in range(nc):
            nz = np.nonzero(A[:, j])[0]
            idx.extend(nz.tolist()); val.extend(A[nz, j].tolist())
            starts.append(len(idx))
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = starts, idx, val
        lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = nc, self.n_res
        h.passModel(lp)
        self._h = h

    # ------------------------------------------------------------------ #
    def set_counts(self, type_ids: np.ndarray, counts: np.ndarray):
        """Set z_j <= N_j (sparse input: type ids with positive counts)."""
        if self._h is None:
            return
        ub = np.zeros(len(self.cols))
        col = self.col_of_type[np.asarray(type_ids, np.int64)]
        keep = col >= 0
        np.add.at(ub, col[keep], np.asarray(counts, float)[keep])
        changed = np.nonzero(ub != self._ub)[0]
        if changed.size:
            self._h.changeColsBounds(
                changed.size, changed.astype(np.int32),
                np.zeros(changed.size), ub[changed])
            self._ub = ub

    def g(self, n: float) -> tuple[float, float]:
        """(g(n), slope) with slope = y*^T c, a supergradient of g at n."""
        if self._h is None or not np.any(self._ub > 0):
            return 0.0, 0.0
        h = self._h
        for i in range(self.n_res):
            h.changeRowBounds(i, -highspy.kHighsInf, float(n * self.c[i]))
        h.run()
        self.n_solves += 1
        st = h.getModelStatus()
        if st != highspy.HighsModelStatus.kOptimal:
            raise RuntimeError(f"LP-RS window solve failed: {h.modelStatusToString(st)}")
        sol = h.getSolution()
        z = np.asarray(sol.col_value)
        y = np.abs(np.asarray(sol.row_dual))
        return float(self.r @ z), float(y @ self.c)

    def lam(self, n: float) -> float:
        return self.g(n)[0] / n

    def tau(self, n: float) -> np.ndarray:
        """Shadow prices of the current window at length n (after g(n))."""
        return np.abs(np.asarray(self._h.getSolution().row_dual)) if self._h else np.zeros(self.n_res)

    # ------------------------------------------------------------------ #
    def pieces(self, n_lo: float, n_hi: float) -> list[Piece]:
        """Exact affine pieces of g on [n_lo, n_hi] (Eisner-Severance)."""
        g1, s1 = self.g(n_lo)
        if n_hi <= n_lo:
            return [Piece(n_lo, n_lo, g1 - s1 * n_lo, s1)]
        g2, s2 = self.g(n_hi)
        out: list[Piece] = []
        self._rec(n_lo, g1, s1, n_hi, g2, s2, out, depth=0)
        return out

    def _close(self, x: float, y: float) -> bool:
        return abs(x - y) <= self.tol * max(1.0, abs(x), abs(y))

    def _rec(self, n1, g1, s1, n2, g2, s2, out, depth):
        a1, a2 = g1 - s1 * n1, g2 - s2 * n2
        if self._close(a1 + s1 * n2, g2):           # L1 passes through (n2, g2)
            out.append(Piece(n1, n2, a1, s1)); return
        if self._close(a2 + s2 * n1, g1):
            out.append(Piece(n1, n2, a2, s2)); return
        if s1 - s2 <= self.tol * max(1.0, abs(s1), abs(s2)) or n2 - n1 < 1e-9:
            # equal supporting slopes => g affine on [n1, n2] up to solver
            # tolerance: use the chord through the two exact end values
            b = (g2 - g1) / (n2 - n1) if n2 > n1 else s1
            out.append(Piece(n1, n2, g1 - b * n1, b)); return
        if depth > 200:
            raise RuntimeError("Eisner-Severance did not converge (non-concave g?)")
        ns = (a2 - a1) / (s1 - s2)                   # intersection of L1 and L2
        gs, ss = self.g(ns)
        if self._close(gs, a1 + s1 * ns):            # single breakpoint at ns
            out.append(Piece(n1, ns, a1, s1)); out.append(Piece(ns, n2, a2, s2)); return
        self._rec(n1, g1, s1, ns, gs, ss, out, depth + 1)
        self._rec(ns, gs, ss, n2, g2, s2, out, depth + 1)

    def lam_on_integers(self, n_lo: int, n_hi: int) -> np.ndarray:
        """lam_hat(n) for n = n_lo..n_hi (n_lo >= 1), exact."""
        n = np.arange(n_lo, n_hi + 1, dtype=float)
        if self._h is None or not np.any(self._ub > 0):
            return np.zeros_like(n)
        ps = self.pieces(float(n_lo), float(n_hi))
        edges = np.array([p.n_hi for p in ps])
        k = np.minimum(np.searchsorted(edges, n, side="left"), len(ps) - 1)
        a = np.array([p.a for p in ps])[k]
        b = np.array([p.b for p in ps])[k]
        return (a + b * n) / n

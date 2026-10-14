"""
Event-driven resource occupancy U_i(t).

A job admitted at tick tau with duration D (steps) occupies ticks
tau, ..., tau + D - 1 and is released at tick tau + D (IP-C convention,
Zhang & Cheung 2025, Sec. 3.5: 1(D >= t - tau + 1)). U only changes at
admissions and releases, so we never iterate over ticks.
"""
from __future__ import annotations

import heapq

import numpy as np


class Occupancy:
    def __init__(self, capacity: np.ndarray, record: bool = True):
        self.c = np.asarray(capacity, float)
        self.U = np.zeros_like(self.c)
        self._heap: list[tuple[int, int]] = []     # (release_tick, job_idx)
        self._A: dict[int, np.ndarray] = {}
        self.record = record
        self.events: list[tuple] = []              # (tick, +1/-1, job, U_0, U_1, ...)

    def release_until(self, t: int) -> list[int]:
        """Release every job with release tick <= t. Returns released job ids."""
        out = []
        while self._heap and self._heap[0][0] <= t:
            rt, j = heapq.heappop(self._heap)
            self.U -= self._A.pop(j)
            self.U[np.abs(self.U) < 1e-15] = 0.0
            out.append(j)
            if self.record:
                self.events.append((rt, -1, j, *self.U.tolist()))
        return out

    def fits(self, A: np.ndarray, buffer: np.ndarray | float = 0.0) -> bool:
        return bool(np.all(self.U + A <= self.c - buffer + 1e-12))

    def add(self, j: int, A: np.ndarray, t: int, D: int):
        A = np.asarray(A, float)
        self.U += A
        self._A[j] = A
        heapq.heappush(self._heap, (int(t + D), int(j)))
        if self.record:
            self.events.append((t, +1, j, *self.U.tolist()))

    @property
    def n_active(self) -> int:
        return len(self._heap)


class WindowOccupancy:
    """
    Test 1 left-hand side (Algorithm 1, eq. (8)):
        sum_{tau = tau_bar+1}^{t-1} A_{j(tau)} 1(D(tau) >= t - tau)
    i.e. only jobs ADMITTED in the current phase after the warm-up, counted
    while tau + D >= t. Reset at every phase start.
    """
    def __init__(self, n_res: int):
        self.S = np.zeros(n_res)
        self._heap: list[tuple[int, int]] = []
        self._A: dict[int, np.ndarray] = {}

    def add(self, j: int, A: np.ndarray, tau: int, D: int):
        self.S += A
        self._A[j] = np.asarray(A, float)
        heapq.heappush(self._heap, (int(tau + D), int(j)))   # counted while tau + D >= t

    def value_at(self, t: int) -> np.ndarray:
        while self._heap and self._heap[0][0] < t:
            _, j = heapq.heappop(self._heap)
            self.S -= self._A.pop(j)
        self.S[np.abs(self.S) < 1e-15] = 0.0
        return self.S.copy()

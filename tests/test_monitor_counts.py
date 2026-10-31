"""PhaseMonitor keeps the phase's arrivals in growing numpy arrays (week 33 fix): _counts gives the same type counts as
the old list-based version, across the buffer growth, and no longer copies a Python list of the whole phase per call
(the old cost grew with the phase's arrivals, about 45% of a long phase's time)."""
import numpy as np

from dlent_rt.models.core.monitor import PhaseMonitor


def _bare_monitor() -> PhaseMonitor:
    """A PhaseMonitor with only the state register_arrival and _counts use (no LP, no tests)."""
    m = object.__new__(PhaseMonitor)
    m._n, m._tk, m._ty = 0, np.empty(1024, np.int64), np.empty(1024, np.int64)
    m.R, m.r_unit, m.last, m.tau_bar = {}, 1.0, 0, 10 ** 9
    return m


def _old_counts(ticks, types, s_excl, hi_incl):
    tk = np.asarray(ticks, np.int64)
    lo, hi = np.searchsorted(tk, s_excl, side="right"), np.searchsorted(tk, hi_incl, side="right")
    ty = np.asarray(types[lo:hi], np.int64)
    ty = ty[ty >= 0]
    return None if ty.size == 0 else np.unique(ty, return_counts=True)


def test_counts_match_the_list_version_across_growth():
    rng = np.random.default_rng(0)
    m, ticks, types = _bare_monitor(), [], []
    t = 0
    for _ in range(5000):                                  # > 4 doublings of the 1,024 buffer
        t += int(rng.integers(1, 20))
        j = int(rng.integers(-1, 40))                      # -1: an arrival of no type
        m.last = t - 1
        m.register_arrival(t, j, 1.0, False, 0, np.zeros(2), 1)
        ticks.append(t); types.append(j)
    assert m._n == 5000 and m._tk.size >= 5000
    for s, h in [(0, t), (0, 1), (t // 3, 2 * t // 3), (t - 50, t), (t, t + 5), (-5, 0)]:
        a, b = m._counts(s, h), _old_counts(ticks, types, s, h)
        if b is None:
            assert a is None
        else:
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_array_equal(a[1], b[1])
    assert not hasattr(m, "arr_ticks") and not hasattr(m, "arr_types")

"""
Test margins (week 30, read-only): for every phase of a DLENT-RT run and every test, the maximum
over the checked ticks of (statistic / threshold) and the tick where it is reached.

    dlent_venv/bin/python notebooks/src/test_margins_check.py [default|binding|both]

The tests are the monitor's (dlent_rt/models/core/monitor.py). RecordingMonitor.scan is a
VERBATIM copy of PhaseMonitor.scan plus logging, so the run must reproduce the plain run exactly;
this is checked against cached results (tests/data/snapshot_real.npz for the default config, the
week-30 capacity sweep row for the binding config). No threshold is changed.

Ratios (> 1 means the test fails at that tick):
  Test 1   max_i LHS1_i(u0) / (c_i - a_max_eff_i), at the first active tick u0 of each gap
           (exact: the LHS is non-increasing inside a gap).
  Test 2   LHS2(u) / eps~_Test2(u-1) on every active tick u (exact).
  Test 3   lambda_hat(s+1:u-1) / (lambda~(u-1) + eps_Test3(u-1-s)) over the geometric s-grid. Exact
           on every tick for s = tau_prev and whenever the monitor reconstructs the gap; otherwise
           the monitor's monotone screen LHS(first u) / RHS(last u), an UPPER bound on the gap's
           maximum (column "T3 bound only").
  timeout  W / (2 W_prev) at the end of the phase (the timeout fires at W = 2 W_prev).
Ticks are restricted to the phase (u <= u*, the failing tick, when a test fires in the gap).

"default" = the COMMITTED configs/default.yaml (git HEAD), "binding" = the same with capacity
fraction_of_concurrency_quantile 0.3 / 0.7, drop_jobs_exceeding_capacity on, default grid.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import dlent_rt.models.theory as theory_mod
from dlent_rt.config import load_config
from dlent_rt.data import load_dataset
from dlent_rt.models.core.monitor import PhaseMonitor, ScanResult, geometric_offsets
from dlent_rt.models.core.error_params import make_error_params
from dlent_rt.phase0 import run_phase0

warnings.simplefilter("ignore")
OUT = ROOT / "reports" / "week30" / "tables"


class RecordingMonitor(PhaseMonitor):
    """PhaseMonitor whose scan / _test3 also log the test ratios (logic copied verbatim)."""
    LOG: list = []

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.rec = dict(t1=(-np.inf, None), t2=(-np.inf, None), t3=(-np.inf, None), t3_bound=False)
        RecordingMonitor.LOG.append(self)

    def _upd(self, key, ratio, tick, bound=False):
        if ratio > self.rec[key][0]:
            self.rec[key] = (float(ratio), int(tick))
            if key == "t3":
                self.rec["t3_bound"] = bound

    def scan(self, b: int) -> ScanResult:
        ep = self.ep
        self._t3_pending = []
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

        u = taus + 1
        fails: list[tuple[int, str]] = []

        timeout = self.tau_prev + 2 * self.W_prev + 1
        if lo + 1 <= timeout <= b:
            fails.append((timeout, "timeout"))

        act = u > self.tau_bar + 1
        lhs2 = thr2 = None
        r1 = r2 = None
        if np.any(act):
            den = (u - self.tau_bar - 1).astype(float)
            lhs2 = np.divide(S2_c, den, out=np.zeros_like(S2_c), where=den > 0)
            thr2 = ep.eps_test2(self.W_prev, n_tau)
            bad = act & (lhs2 > thr2)
            if np.any(bad):
                fails.append((int(u[np.argmax(bad)]), "test2"))
            r2 = (lhs2 / thr2, act)

            u0 = int(u[np.argmax(act)])
            val1 = self.t1.value_at(u0)
            if np.any(val1 > self.c - self.a_max_eff + 1e-12):
                fails.append((u0, "test1"))
            r1 = (float(np.max(val1 / (self.c - self.a_max_eff))), u0)

            f3, m3 = self._test3(u0, b, lo, lam, lt_c, taus)
            if f3 is not None:
                fails.append((f3, "test3"))
        else:
            m3 = np.nan

        diag = dict(lam_phase=float(lam[-1]), lam_tilde=float(lt_c[-1]),
                    test2_lhs=float(lhs2[-1]) if lhs2 is not None else np.nan,
                    test2_thr=float(thr2[-1]) if thr2 is not None else np.nan,
                    test3_screen_margin=float(m3))

        u_end = min(fails)[0] if fails else b                 # last tick of this phase checked here
        if r1 is not None and r1[1] <= u_end:
            self._upd("t1", *r1)
        if r2 is not None:
            m = r2[1] & (u <= u_end)
            if np.any(m):
                k = int(np.argmax(np.where(m, r2[0], -np.inf)))
                self._upd("t2", r2[0][k], u[k])
        for ratio, tick, bound in self._t3_pending:
            if tick <= u_end:
                self._upd("t3", ratio, tick, bound)

        if fails:
            u_star, reason = min(fails)
            k = u_star - 1 - lo
            self._commit(S2_c[k], lt_c[k], lam[: k + 1].sum(), u_star - 1)
            return ScanResult(u_star, reason, diag)
        self._commit(S2_c[-1], lt_c[-1], lam.sum(), hi)
        return ScanResult(None, "", diag)

    def _test3(self, u0: int, b: int, lo: int, lam, lt_c, taus):
        ep = self.ep
        self._t3_pending = []
        offs = geometric_offsets(self.base, max(b - 2 - self.tau_prev, 0))
        first_fail, margin = None, np.inf
        us = np.arange(u0, b + 1)
        lt_u = lt_c[us - 1 - taus[0]]
        for off in offs:
            s = self.tau_prev + int(off)
            if s >= lo or s > u0 - 2:
                continue
            n_u = us - 1 - s
            rhs = lt_u + ep.eps_test3(n_u)
            if s == self.tau_prev:
                lhs = lam[us - 1 - taus[0]]
                margin = min(margin, float(rhs[-1] - lhs[0]))
                bad = lhs > rhs
                k = int(np.argmax(lhs / rhs))
                self._t3_pending.append((float(lhs[k] / rhs[k]), int(us[k]), False))
            else:
                cnt = self._counts(s, lo)
                if cnt is None:
                    continue
                self.wlp.set_counts(*cnt)
                lhs0 = self.wlp.lam(float(n_u[0]))
                margin = min(margin, float(rhs[-1] - lhs0))
                if lhs0 <= rhs[-1]:
                    self._t3_pending.append((float(lhs0 / rhs[-1]), int(us[0]), True))
                    continue
                lhs = self.wlp.lam_on_integers(int(n_u[0]), int(n_u[-1]))
                bad = lhs > rhs
                k = int(np.argmax(lhs / rhs))
                self._t3_pending.append((float(lhs[k] / rhs[k]), int(us[k]), False))
            if np.any(bad):
                uf = int(us[np.argmax(bad)])
                first_fail = uf if first_fail is None else min(first_fail, uf)
        return first_fail, (margin if np.isfinite(margin) else np.nan)


def committed_default() -> Path:
    """configs/default.yaml as committed (git HEAD), written to a temporary file."""
    txt = subprocess.check_output(["git", "-C", str(ROOT), "show", "HEAD:configs/default.yaml"], text=True)
    p = Path(tempfile.mkdtemp()) / "default_head.yaml"
    p.write_text(txt)
    return p


def configs():
    base = load_config(committed_default())
    binding = replace(base, capacity=replace(base.capacity, mode="fraction_of_concurrency_quantile",
                                             fraction=0.3, concurrency_quantile=0.7,
                                             drop_jobs_exceeding_capacity=True)).validate()
    return {"default": base, "binding": binding}


def thresholds(ep, ds, n_list):
    """Threshold values at representative window lengths n (W_prev = n for Test 2)."""
    c = np.asarray(ds.capacity, float)
    rows = []
    for n in n_list:
        rows.append({"n": n, "eta(n)": float(ep.eta(n)), "eps_A'(n)": float(ep.eps_Ap(n)),
                     "eps_B'(n)": float(ep.eps_Bp(n)), "eps_C'(n)": float(ep.eps_Cp(n)),
                     "eps_G(n)": float(ep.eps_G(n)), "CI1(n)": float(ep.CI1(n)),
                     "eps_Test2 (W_prev=n, n)": float(ep.eps_test2(n, n)),
                     "eps_Test3(n)": float(ep.eps_test3(n))})
    return pd.DataFrame(rows).set_index("n")


def run(name, cfg):
    t0 = time.time()
    RecordingMonitor.LOG = []
    ds = load_dataset(cfg)
    st = run_phase0(ds, cfg)
    orig = theory_mod.PhaseMonitor
    theory_mod.PhaseMonitor = RecordingMonitor
    try:
        model = theory_mod.TheoryModel(ds, st, cfg)
        res = model.run()
    finally:
        theory_mod.PhaseMonitor = orig
    ph = res.phases
    assert len(RecordingMonitor.LOG) == len(ph)
    rows = []
    for m, (p, mon) in enumerate(zip(ph, RecordingMonitor.LOG), start=1):
        r = mon.rec
        rows.append({
            "phase": m, "start": int(p["tau_start"]), "end": int(p["tau_end"]), "W": int(p["W"]),
            "W_prev": int(p["W_prev"]), "fired": p["reason"],
            "T1 max ratio": r["t1"][0], "T1 tick": r["t1"][1],
            "T2 max ratio": r["t2"][0], "T2 tick": r["t2"][1],
            "T3 max ratio": r["t3"][0], "T3 tick": r["t3"][1], "T3 bound only": r["t3_bound"],
            "timeout ratio W/(2 W_prev)": p["W"] / (2 * p["W_prev"]),
        })
    df = pd.DataFrame(rows).set_index("phase")
    df = df.replace([-np.inf], np.nan)
    ep = model.ep
    # without the overrides: gamma, xi of Phase 0 (data)
    cfg_no = replace(cfg, constants=replace(cfg.constants, xi_override=None, gamma_override=None))
    ep_no = make_error_params(st, ds, cfg_no)
    n_list = [int(ds.d_max_steps), 100_000, int(ds.T)]
    c = np.asarray(ds.capacity, float)
    info = dict(config=name, capacity=f"{cfg.capacity.mode} fraction={cfg.capacity.fraction} "
                                        f"q={cfg.capacity.concurrency_quantile} drop={cfg.capacity.drop_jobs_exceeding_capacity}",
                c=c.tolist(), a_max=np.asarray(ds.a_max).tolist(), d_max=int(ds.d_max_steps), T=int(ds.T),
                delta=ep.delta, c_min=ep.c_min, gamma=ep.gamma, xi=ep.xi, eps_D=ep.eps_D,
                fac=1 - ep.eps_D - ep.xi, p_U_cap=(1 - ep.xi) / (1 + ep.eps_D),
                T1_threshold=(c - model.a_max_eff).tolist(),
                gamma_data=ep_no.gamma, xi_data=ep_no.xi, eps_D_data=ep_no.eps_D,
                fac_data=1 - ep_no.eps_D - ep_no.xi, p_U_cap_data=(1 - ep_no.xi) / (1 + ep_no.eps_D),
                T1_threshold_data=(c - np.asarray(ds.a_max)).tolist(),
                R_total=res.summary["R_total"], reasons=res.summary["phase_end_reasons"],
                seconds=time.time() - t0)
    return df, info, thresholds(ep, ds, n_list), thresholds(ep_no, ds, n_list), res


def check_against_cache(name, res):
    if name == "default":
        snap = np.load(ROOT / "tests" / "data" / "snapshot_real.npz")
        same = np.isclose(res.summary["R_total"], snap["on_R_total"][0], rtol=0, atol=0) and \
            len(res.phases) == len(snap["on_phases"])
        return f"vs tests/data/snapshot_real.npz: R_total and phase count {'IDENTICAL' if same else 'DIFFER'}"
    sw = pd.read_csv(OUT / "table08_capacity_grid_sweep_full.csv", index_col=[0, 1])
    row = sw.loc[(0.3, "default (k*=128, hw=32)")]
    same = np.isclose(res.summary["R_total"], row["R DLENT-RT $"], rtol=1e-12) and \
        len(res.phases) == int(row["phases"])
    return f"vs capacity sweep (0.3, default grid): R_total and phase count {'IDENTICAL' if same else 'DIFFER'}"


def main(which):
    cf = configs()
    names = ["default", "binding"] if which == "both" else [which]
    for name in names:
        df, info, thr, thr_no, res = run(name, cf[name])
        print(f"=== {name}: {info['capacity']}  ({info['seconds']:.0f} s)")
        print(check_against_cache(name, res))
        for k, v in info.items():
            if k not in ("config", "capacity", "seconds"):
                print(f"  {k:<20} {v}")
        print("thresholds with the overrides:")
        print(thr.to_string(float_format=lambda x: f"{x:.4g}"))
        print("thresholds WITHOUT the overrides (Phase-0 gamma, xi):")
        print(thr_no.to_string(float_format=lambda x: f"{x:.4g}"))
        print(df.to_string(float_format=lambda x: f"{x:.4g}"))
        df.to_csv(OUT / f"test_margins_{name}.csv")
        thr.to_csv(OUT / f"test_thresholds_{name}.csv")
        thr_no.to_csv(OUT / f"test_thresholds_{name}_no_overrides.csv")
        print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "both")

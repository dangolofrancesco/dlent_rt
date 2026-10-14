"""
Regression snapshot (week 30): Phase 0 outputs + online run summary, compared BIT-IDENTICALLY.

    dlent_venv/bin/python -m tests.regression_snapshot save  synthetic   # -> tests/data/snapshot_synthetic.npz
    dlent_venv/bin/python -m tests.regression_snapshot check synthetic
    dlent_venv/bin/python -m tests.regression_snapshot save  real        # real trace (minutes)

Phase 0: p, r_tilde, v, tau, s_C, util_opt (= sum_j p_j v_ij x*_j / c_i), lambda_star.
Online:  phase table (start, end, reason, arrivals, admitted, k_hat, thinned, phys rejects, R, tau per phase),
         admission counts per decision reason, R_total.
"""
from __future__ import annotations

import sys, tempfile, warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

from dlent_rt.config import default_config
from dlent_rt.data import load_dataset
from dlent_rt.lp_oracle import solve_lp_rs
from dlent_rt.phase0 import run_phase0
from dlent_rt.models.theory import TheoryModel

DATA = Path(__file__).parent / "data"
PHASE_COLS = ["tau_start", "tau_end", "n_arrivals", "n_admitted", "n_khat", "n_thinned", "n_phys_reject",
              "R", "tau_cpu", "tau_ram", "lam_hat_start"]
REASONS = ("test1", "test2", "test3", "timeout", "horizon")


def _config(which: str, tmp: Path):
    if which == "synthetic":
        from tests.synthetic import synthetic_config
        return synthetic_config(tmp, n0=400, n_jobs=3000, drift=True)
    cfg = default_config()
    return replace(cfg, bid=replace(cfg.bid, strategy="lognormal", sigma=0.3)).validate()


def snapshot(which: str) -> dict:
    with tempfile.TemporaryDirectory() as d, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = _config(which, Path(d))
        ds = load_dataset(cfg)
        st = run_phase0(ds, cfg)
        lp = solve_lp_rs(p=st.p, r=st.r_tilde, v=st.v, c=ds.capacity, presolve=cfg.lp.presolve)
        x = np.zeros(st.n_types)
        x[lp.support_idx] = lp.x
        res = TheoryModel(ds, st, cfg).run()
    A, ph = res.arrivals, res.phases
    dec = np.array([[a["admit"], a["ir_reject"], a["k_hat"], a["U"], a["phys_reject"]] for a in A], bool)
    return {
        "p0_p": st.p, "p0_r_tilde": st.r_tilde, "p0_v": st.v, "p0_tau": st.tau,
        "p0_s_C": np.array([st.carbon_price]), "p0_util_opt": (st.p * x) @ st.v / ds.capacity,
        "p0_lambda_star": np.array([st.lam_star]),
        "on_phases": np.array([[float(p[k]) for k in PHASE_COLS] for p in ph]),
        "on_reasons": np.array([sum(p["reason"] == r for p in ph) for r in REASONS]),
        "on_decisions": np.array([dec[:, 0].sum(), dec[:, 1].sum(), (~dec[:, 2]).sum(),
                                  (dec[:, 2] & ~dec[:, 3]).sum(), dec[:, 4].sum()]),
        "on_R_total": np.array([res.summary["R_total"]]),
        "on_R": np.array([a["R"] for a in A]),
        "on_admit": dec[:, 0],
    }


def path(which: str) -> Path:
    return DATA / f"snapshot_{which}.npz"


def compare(a: dict, b: dict) -> list[str]:
    """Bit-identical comparison (NaN == NaN). Returns the keys that differ."""
    bad = []
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            bad.append(f"{k}: missing")
            continue
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.shape != y.shape or not np.array_equal(x, y, equal_nan=x.dtype.kind == "f"):
            n = int((x != y).sum()) if x.shape == y.shape else -1
            bad.append(f"{k}: differs ({n} entries)")
    return bad


if __name__ == "__main__":
    mode, which = sys.argv[1], sys.argv[2]
    snap = snapshot(which)
    if mode == "save":
        np.savez_compressed(path(which), **snap)
        print(f"saved {path(which)}  R_total={snap['on_R_total'][0]:.10g}  phases={len(snap['on_phases'])}")
    else:
        bad = compare(snap, dict(np.load(path(which))))
        print("BIT-IDENTICAL" if not bad else "DIFF:\n  " + "\n  ".join(bad))
        sys.exit(1 if bad else 0)

"""
Capacity from a target load factor rho (week 30, report only; no default changed).

    dlent_venv/bin/python scripts/capacity_from_rho.py scan [rho ...]       # Phase 0 / fluid / G2 / G1 / static / MILP
    dlent_venv/bin/python scripts/capacity_from_rho.py dlent rho [rho ...]  # DLENT-RT with the test margins

rho_i = sum_{j: r~_j > 0} p_j v_ij / c_i, exactly what the Phase 0 fluid LP sees (v_ij = A_ij d_bar_j, p_j per tick),
the same rho for CPU and RAM: c_i = load_i / rho. load(c) depends on c through drop_jobs_exceeding_capacity (ON) and,
through the kept jobs, s_C and r~; so c is found by fixed-point iteration c <- load(c) / rho with
capacity.mode = "absolute" (capacity.absolute.cpu / ram are in the trace's normalised units, the units of A:
1.0 CPU = 64 cores, 1.0 RAM = 256 GB; data._size_capacity). load(c) only changes when the drop set changes (A takes
the 32 snapped profile values), so the iteration either reaches an exact fixed point or cycles; both are reported.

Base: the committed configs/default.yaml (git show HEAD into a temporary file), seed 42; xi / gamma stay forced.
Rows reuse scripts/capacity_regime.run_regime (same code path as the regime table).
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "notebooks" / "src"))

from dlent_rt.data import load_dataset
from dlent_rt.phase0 import run_phase0

OUT = ROOT / "reports" / "week30" / "small_cluster"
RHOS = (1.1, 1.25, 1.5, 2.0)
MAX_WORKERS = 4                                    # ~1 GB per DLENT-RT run


def absolute_cfg(base, c, drop=True):
    cap = replace(base.capacity, mode="absolute", drop_jobs_exceeding_capacity=drop,
                  absolute=replace(base.capacity.absolute, cpu=float(c[0]), ram=float(c[1])))
    return replace(base, capacity=cap).validate()


def profitable_load(cfg):
    ds = load_dataset(cfg)
    st = run_phase0(ds, cfg)
    return (st.p * (st.r_tilde > 0)) @ st.v, ds, st


def find_c(base, rho, max_iter=40):
    """Fixed point c = load(c) / rho; start from the load with no capacity binding (the default cluster)."""
    load0, _, _ = profitable_load(base)                 # committed default (from_amax, no filter)
    c = load0 / rho
    hist = []
    for it in range(1, max_iter + 1):
        load, ds, st = profitable_load(absolute_cfg(base, c))
        c_new = load / rho
        drop = ds.capacity_drop
        hist.append({"iter": it, "c_cpu": c[0], "c_ram": c[1], "load_cpu": load[0], "load_ram": load[1],
                     "rho_cpu": load[0] / c[0], "rho_ram": load[1] / c[1], "dropped_H": drop["n_H"],
                     "dropped_stream": drop["n_stream"]})
        if np.allclose(c_new, c, rtol=1e-12, atol=0):
            return c, hist, "fixed point"
        prev = [h for h in hist[:-1] if np.allclose((h["c_cpu"], h["c_ram"]), c_new, rtol=1e-12, atol=0)]
        if prev:                                       # cycle: keep the smaller-capacity end
            return np.minimum(c, c_new), hist, f"cycle of length {it - prev[0]['iter']}"
        c = c_new
    return c, hist, "no convergence"


def scan_worker(rho):
    warnings.simplefilter("ignore")
    from capacity_regime import committed, run_regime
    base = committed()
    t0 = time.time()
    c, hist, how = find_c(base, rho)
    cfg = absolute_cfg(base, c)
    row = run_regime(f"rho={rho}", cfg, dict(dlent_R=np.nan, dlent_reasons="not run", dlent_src="not run"), base)
    load, ds, st = profitable_load(cfg)
    row.update({"rho": rho, "convergence": how, "iterations": len(hist),
                "rho achieved CPU": load[0] / c[0], "rho achieved RAM": load[1] / c[1],
                "capacity.absolute.cpu": float(c[0]), "capacity.absolute.ram": float(c[1]),
                "find_c s": None, "total s": time.time() - t0})
    return row, hist


def dlent_worker(rho, c):
    warnings.simplefilter("ignore")
    import test_margins_check as tm
    from capacity_regime import committed
    base = committed()
    cfg = absolute_cfg(base, np.asarray(c, float))
    df, info, thr, thr_no, res = tm.run(f"rho={rho}", cfg)
    ph = res.phases
    return dict(rho=rho, margins=df.reset_index().to_dict("list"), reasons=res.summary["phase_end_reasons"],
                tau=[(p["tau_cpu"], p["tau_ram"]) for p in ph], R=res.summary["R_total"],
                thinning=int(sum(p["n_thinned"] for p in ph)), phys=int(sum(p["n_phys_reject"] for p in ph)),
                admitted=int(res.summary["n_admitted"]), seconds=info["seconds"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "scan"
    if mode == "scan":
        rhos = [float(x) for x in sys.argv[2:]] or list(RHOS)
        with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(rhos)), mp_context=mp.get_context("spawn")) as pool:
            out = list(pool.map(scan_worker, rhos))
        rows = pd.DataFrame([r for r, _ in out]).set_index("regime")
        hist = pd.concat([pd.DataFrame(h).assign(rho=r["rho"]) for r, h in out])
        rows.to_csv(OUT / "rho_scan.csv")
        hist.to_csv(OUT / "rho_iterations.csv", index=False)
        pd.set_option("display.width", 250)
        print(rows.T.to_string())
        print(hist.to_string())
    elif mode == "dlent":
        rows = pd.read_csv(OUT / "rho_scan.csv", index_col=0)
        rhos = [float(x) for x in sys.argv[2:]]
        cs = {r: (float(rows.loc[f"rho={r}", "capacity.absolute.cpu"]), float(rows.loc[f"rho={r}", "capacity.absolute.ram"]))
              for r in rhos}
        with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(rhos)), mp_context=mp.get_context("spawn")) as pool:
            res = list(pool.map(dlent_worker, rhos, [cs[r] for r in rhos]))
        (OUT / "rho_dlent.json").write_text(json.dumps(res, indent=1, default=float))
        for o in res:
            print(f"=== rho = {o['rho']}: R = {o['R']:,.2f}  reasons {o['reasons']}  thinning {o['thinning']}  "
                  f"phys {o['phys']}  admitted {o['admitted']}  ({o['seconds']:.0f} s)")
            print(pd.DataFrame(o["margins"]).to_string())


if __name__ == "__main__" and (len(sys.argv) < 2 or sys.argv[1] != "curve"):
    main()


def rho_curve(n=48, lo=-3.0, hi=0.0):
    """rho_i(c) = load_i(c) / c_i along c = alpha * load0 (load0 = the default cluster's profitable load), filter ON,
    alpha log-spaced in [10^lo, 10^hi]: the largest rho the filtered trace can reach, and what it costs."""
    from capacity_regime import committed
    from dlent_rt.benchmarks import Benchmarks
    from dlent_rt.models.core.stream import build_stream
    base = committed()
    load0, ds0, st0 = profitable_load(base)
    R0 = Benchmarks(ds0, st0, base, stream=build_stream(ds0, base)).R
    Rpos0 = float(np.maximum(R0, 0).sum())
    rows = []
    for a in np.logspace(lo, hi, n):
        c = a * load0
        try:
            load, ds, st = profitable_load(absolute_cfg(base, c))
        except Exception as e:                        # every job dropped
            rows.append({"alpha": a, "c_cpu": c[0], "c_ram": c[1], "error": type(e).__name__})
            continue
        R = Benchmarks(ds, st, absolute_cfg(base, c), stream=build_stream(ds, absolute_cfg(base, c))).R
        rows.append({"alpha": a, "c_cpu": c[0], "c_ram": c[1], "c CPU (cores)": c[0] * 64, "c RAM (GB)": c[1] * 256,
                     "rho CPU": load[0] / c[0], "rho RAM": load[1] / c[1],
                     "|H| kept": ds.H.n, "|stream| kept": ds.stream.n,
                     "stream jobs dropped %": 100 * ds.capacity_drop["n_stream"] / (ds.capacity_drop["n_stream"] + ds.stream.n),
                     "stream R_job+ kept $": float(np.maximum(R, 0).sum()),
                     "stream R_job+ removed % (vs no filter)": 100 * (1 - float(np.maximum(R, 0).sum()) / Rpos0),
                     "tau_0": str(np.round(st.tau, 5).tolist())})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "rho_curve.csv", index=False)
    return df


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "curve":
    warnings.simplefilter("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    print(rho_curve().to_string(float_format=lambda x: f"{x:,.4g}"))

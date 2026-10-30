"""
The long computations of notebooks/01_model_check.ipynb, cached in reports/week33/cache/ (the notebook reads the
cache; without it, it recomputes the same things itself).

    dlent_venv/bin/python scripts/model_check_cache.py select_f  # the f rule on H (scripts/select_f.py) -> f_scan_H.pkl
    dlent_venv/bin/python scripts/model_check_cache.py bench     # the stream models at the configured f -> benchmarks_f.pkl
    dlent_venv/bin/python scripts/model_check_cache.py disc      # MILP_true vs MILP_bin, 7 days, 3,600 s each
                                                                 # (scripts/discretization_milp.py) -> discretization_milp.pkl
    dlent_venv/bin/python scripts/model_check_cache.py theory    # one theoretical DLENT-RT run -> theory_run.pkl
                                                                 # (+ IR / upper set / truthfulness spot check)
    dlent_venv/bin/python scripts/model_check_cache.py all       # select_f and bench, then disc and theory in two processes

Every step uses configs/default.yaml as it is on disk (bench, disc and theory at its capacity.f).
"""
from __future__ import annotations

import multiprocessing as mp
import pickle
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CACHE = ROOT / "reports" / "week33" / "cache"


def select_f() -> dict:
    sys.path.insert(0, str(ROOT / "scripts"))
    import select_f as sf
    return sf.main()


def bench() -> dict:
    """fluid OPT, fluid on feasible types, the static policy, the no-learning model and the window LPs on the stream
    at the configured f (theory_eval.evaluate_f). Not part of the f rule."""
    from dlent_rt import theory_eval as te
    from dlent_rt.config import default_config
    cfg = default_config()
    t = time.perf_counter()
    r = te.evaluate_f(cfg, cfg.capacity.f, seed=cfg.run.seed)
    r["runtime_s"] = time.perf_counter() - t
    _save(r, "benchmarks_f.pkl")
    return r


def disc() -> dict:
    sys.path.insert(0, str(ROOT / "scripts"))
    import discretization_milp as dm
    return dm.main(days=7.0, time_limit=3600.0)


def theory() -> dict:
    from dlent_rt.benchmarks import enlarge_for_job_models
    from dlent_rt.config import default_config
    from dlent_rt.data import load_dataset
    from dlent_rt.models.theory import TheoryModel
    from dlent_rt.phase0 import run_phase0
    sys.path.insert(0, str(ROOT / "scripts"))
    from truthful_batch import _Recording, run_numbers, spot_check     # the same checks as the 48 h batch
    cfg = default_config()
    ds = load_dataset(cfg)
    st0 = run_phase0(ds, cfg)
    _, _, enl = enlarge_for_job_models(ds, st0, cfg)          # the job-level check (enlargement off by default)
    t = time.perf_counter()
    m = _Recording(ds, st0, cfg)
    res = m.run()
    runtime = time.perf_counter() - t
    adm = [a["i"] for a in res.arrivals if a["admit"]]
    checks = dict(run=run_numbers(m, res, cfg), spot_check=spot_check(m, res, cfg.run.seed),
                  myerson=dict(expected_profit=float(m.J.profit[adm].sum()),
                               realised_profit=float(res.summary["profit_realised_total"]),
                               payments=float(res.summary["payments_total"]),
                               C_elec=float(m.stream.C_elec[adm].sum())))
    r = dict(arrivals=pd.DataFrame(res.arrivals), phases=pd.DataFrame(res.phases), summary=res.summary,
             runtime_s=runtime, job_level_check=enl, r_unit=m.r_unit, f=cfg.capacity.f, checks=checks,
             R_job=m.J.R, tick=m.stream.tick)
    _save(r, "theory_run.pkl")
    return r


def _save(obj, name):
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(CACHE / name, "wb") as fh:
        pickle.dump(obj, fh)
    print(f"saved {CACHE / name}", flush=True)


def _run(name):
    warnings.simplefilter("ignore")
    return {"select_f": select_f, "bench": bench, "disc": disc, "theory": theory}[name]()


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what == "all":
        select_f()
        bench()
        with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as pool:
            list(pool.map(_run, ["disc", "theory"]))
    else:
        _run(what)

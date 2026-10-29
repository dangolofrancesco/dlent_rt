"""
The long computations of notebooks/01_model_check.ipynb, cached in reports/week33/cache/ (the notebook reads the
cache; without it, it recomputes the same things itself).

    dlent_venv/bin/python scripts/model_check_cache.py scan      # the f scan (~5 min) -> f_scan.pkl
    dlent_venv/bin/python scripts/model_check_cache.py week      # one-week bucketed MILP, 3,600 s -> week_milp.pkl
    dlent_venv/bin/python scripts/model_check_cache.py theory    # one theoretical DLENT-RT run -> theory_run.pkl
    dlent_venv/bin/python scripts/model_check_cache.py all       # scan, then week and theory in two processes

Every step uses configs/default.yaml as it is on disk (week and theory at its capacity.f).
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


def scan() -> dict:
    from dlent_rt import theory_eval as te
    from dlent_rt.config import default_config
    cfg = default_config()
    out = []
    for f in te.F_SCAN:
        t = time.perf_counter()
        r = te.evaluate_f(cfg, f, seed=cfg.run.seed)
        if f != cfg.capacity.f:
            r.pop("occupancy")                 # keep the occupancy series of the configured f only
        r["runtime_s"] = time.perf_counter() - t
        out.append(r)
        print(f"f = {f}: >= 95% full {100 * r['static']['full_time']:.1f}% of the time", flush=True)
    res = dict(results=out, selected=te.select_f(out), config_f=cfg.capacity.f)
    _save(res, "f_scan.pkl")
    print(f"fixed rule picks f = {res['selected']}; configs/default.yaml has f = {cfg.capacity.f}")
    return res


def week() -> dict:
    from dlent_rt import theory_eval as te
    from dlent_rt.config import default_config
    cfg = default_config()
    r = te.window_models(cfg, days=(0.0, 7.0), milp_time_limit=3600.0, tail_hours=1.0)
    _save(r, "week_milp.pkl")
    return r


def theory() -> dict:
    from dlent_rt.benchmarks import enlarge_for_job_models
    from dlent_rt.config import default_config
    from dlent_rt.data import load_dataset
    from dlent_rt.models.theory import TheoryModel
    from dlent_rt.phase0 import run_phase0
    cfg = default_config()
    ds = load_dataset(cfg)
    st0 = run_phase0(ds, cfg)
    _, _, enl = enlarge_for_job_models(ds, st0, cfg)          # the job-level check (enlargement off by default)
    t = time.perf_counter()
    m = TheoryModel(ds, st0, cfg)
    res = m.run()
    r = dict(arrivals=pd.DataFrame(res.arrivals), phases=pd.DataFrame(res.phases), summary=res.summary,
             runtime_s=time.perf_counter() - t, job_level_check=enl, r_unit=m.r_unit, f=cfg.capacity.f)
    _save(r, "theory_run.pkl")
    return r


def _save(obj, name):
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(CACHE / name, "wb") as fh:
        pickle.dump(obj, fh)
    print(f"saved {CACHE / name}", flush=True)


def _run(name):
    warnings.simplefilter("ignore")
    return {"scan": scan, "week": week, "theory": theory}[name]()


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what == "all":
        scan()
        with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as pool:
            list(pool.map(_run, ["week", "theory"]))
    else:
        _run(what)

"""
Non-stationary bids (week 34; reports/week34/scenarios_expectations.md): the five models in every environment.

  environments  none, s1_bids_fall, s6_bids_spread, s2_bids_rise (in this order; scenario.t_star_hours = 240 h)
  models        fluid        fluid OPT: hindsight LPs on the changed stream, before and after t* (an upper bound)
                greedy_pos   Greedy-Pos: DLENT-RT's mechanism on Phase 0's state with tau = 0, no learning
                no_learning  the practical model frozen at Phase 0 (models.theory.FrozenModel: thinning off, live
                             oracle, no test, no ReInit)
                practical    C3 = thinning off + live oracle + stats.update by_end_reason (the proposed model)
                theoretical  the default config (thinning on, frozen oracle, stats.update replace)

Same seed (42), f, data, bids and thinning draws everywhere; Phase 0 is the same in every environment (only the
stream changes). Each run is its own process, cached in reports/week34/cache/scenarios/<env>/<model>.pkl and skipped
if cached with the same config (resumable).

    caffeinate -i dlent_venv/bin/python scripts/run_scenarios.py --workers 2
    dlent_venv/bin/python scripts/run_scenarios.py --dry-run
    dlent_venv/bin/python scripts/run_scenarios.py --scenario s1 --models fluid greedy_pos     # a subset
    dlent_venv/bin/python scripts/run_scenarios.py --report-only                                # summary table
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pickle
import sys
import time
import traceback
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
CACHE = ROOT / "reports" / "week34" / "cache" / "scenarios"
ENVS = ("none", "s1_bids_fall", "s6_bids_spread", "s2_bids_rise")
MODELS = ("fluid", "greedy_pos", "no_learning", "practical", "theoretical")
LONG = ("practical", "theoretical")
REASONS = ("bin below k*", "below reserve", "thinning", "no space")


def config_for(env: str, model: str, seed=None):
    from dlent_rt.config import default_config
    from dlent_rt.scenarios import resolve
    cfg = default_config()
    cfg = replace(cfg, scenario=replace(cfg.scenario, name=resolve(env)))
    if seed is not None:
        cfg = replace(cfg, run=replace(cfg.run, seed=int(seed)))
    if model in ("practical", "no_learning"):
        cfg = replace(cfg, ablation=replace(cfg.ablation, thinning_off=True), oracle=replace(cfg.oracle, kind="live"))
    if model == "practical":
        cfg = replace(cfg, stats=replace(cfg.stats, update="by_end_reason"))
    return cfg.validate()


def path(env, model) -> Path:
    return CACHE / env / f"{model}.pkl"


def _cached(env, model, seed) -> bool:
    p = path(env, model)
    if not p.is_file():
        return False
    with open(p, "rb") as fh:
        return pickle.load(fh)["config"] == asdict(config_for(env, model, seed))


# --------------------------------------------------------------------------- #
# The runs
# --------------------------------------------------------------------------- #
def _setup(cfg):
    from dlent_rt.data import load_dataset
    from dlent_rt.phase0 import run_phase0
    ds = load_dataset(cfg)
    return ds, run_phase0(ds, cfg)


def _fluid(cfg) -> dict:
    """Hindsight LP-RS on the changed stream, over [0, t*) and [t*, T) separately (the markups differ), and on the
    whole stream; each with Phase 0's grid, full information."""
    from dlent_rt.models.core.reinit import build_window_state, hindsight_cfg
    from dlent_rt.models.core.stream import build_stream
    from dlent_rt.scenarios import post_change, t_star_step
    ds, st0 = _setup(cfg)
    S = build_stream(ds, cfg)
    T, ts = int(ds.T), t_star_step(cfg)
    post = post_change(S.tick, cfg)
    out = {}
    for key, idx, W in (("pre", np.flatnonzero(~post), ts), ("post", np.flatnonzero(post), T - ts),
                        ("whole", np.arange(S.n), T)):
        st, _ = build_window_state(S, idx, idx, W, st0, ds, hindsight_cfg(cfg))
        out[key] = dict(value=float(st.lam_star) * W, tau=np.asarray(st.tau).tolist(), mu=st.myerson.mu,
                        sigma=st.myerson.sigma, n=int(idx.size), W=int(W))
    return dict(reward=out["pre"]["value"] + out["post"]["value"], reward_pre=out["pre"]["value"],
                reward_post=out["post"]["value"], whole=out["whole"]["value"], segments=out, T=T, t_star=ts)


def _per_arrival(J, S, admit, pay_rate, phi_belief):
    """Per admitted arrival (0 if rejected): reward R; expected profit phi d - C_elec with the TRUE markup model
    (profit_exp, the scoring of J) and with the model the run used at that arrival (profit_belief, its phase's
    (mu-hat, sigma-hat)); realised profit payment - C_elec (profit_real)."""
    a = np.asarray(admit, bool)
    pay = np.where(a, np.nan_to_num(pay_rate) * S.duration_hours, 0.0)
    belief = np.asarray(phi_belief, float) * S.duration_hours - S.C_elec
    return dict(R=np.where(a, J.R, 0.0), profit_exp=np.where(a, J.profit, 0.0),
                profit_belief=np.where(a, belief, 0.0), profit_real=np.where(a, pay - S.C_elec, 0.0),
                payment=pay, admit=a)


def _greedy(cfg) -> dict:
    from dlent_rt.benchmarks import Benchmarks
    from dlent_rt.models.core.occupancy import Occupancy
    from dlent_rt.models.policy import ThresholdPolicy, mechanism_summary, simulate
    ds, st0 = _setup(cfg)
    B = Benchmarks(ds, st0, cfg)
    S = B.S
    pol = ThresholdPolicy(st0, S, tau=0.0)
    sim = simulate(S, Occupancy(B.c, record=False), pol)
    ms = mechanism_summary(pol, sim, S)
    arr = _per_arrival(B.terms, S, sim.admit, pol.pay_rate, st0.myerson.phi_job(S.v, S.cost_base))
    return dict(arrays=dict(tick=np.asarray(S.tick), v=np.asarray(S.v), k_bin=np.asarray(pol.k), **arr),
                rejections=ms["rejections"], ir=ms["n_ir_violations"], upper=ms["n_upper_set_violations"],
                spot=np.nan, phases=None, sw=float(B.terms.sw[sim.admit].sum()), grid=st0.grid)


def _recording(cls):
    class Rec(cls):
        def run(self):
            self.states = []
            return super().run()

        def _new_phase(self, state, tau_prev, W_prev):
            self.states.append(state)
            super()._new_phase(state, tau_prev, W_prev)
    return Rec


def _online(cfg, frozen: bool) -> dict:
    from dlent_rt.models.theory import FrozenModel, TheoryModel
    from run_ablations import spot_check, upper_set
    ds, st0 = _setup(cfg)
    m = _recording(FrozenModel if frozen else TheoryModel)(ds, st0, cfg)
    res = m.run()
    S, A = m.stream, res.arrivals
    admit = np.array([a["admit"] for a in A], bool)
    rate = np.array([a["payment_rate"] for a in A], float)
    ph_of = np.array([a["m"] for a in A]) - 1                     # each arrival's phase -> the state it used
    phi_b = np.empty(S.n)
    for k, st in enumerate(m.states):
        w = ph_of == k
        phi_b[w] = st.myerson.phi_job(S.v[w], S.cost_base[w])
    arr = _per_arrival(m.J, S, admit, rate, phi_b)
    assert np.allclose(arr["R"], [a["R"] for a in A])
    sm = res.summary
    return dict(arrays=dict(tick=np.asarray(S.tick), v=np.asarray(S.v), k_bin=np.array([a["k_bin"] for a in A]),
                            k_star=np.array([a["k_star"] for a in A]), **arr),
                rejections=dict(Counter(a["reason"] for a in A if not a["admit"])),
                ir=int(sm["n_ir_violations"]), upper=upper_set(m, res), spot=spot_check(m, res, cfg.run.seed)["violations"],
                phases=pd.DataFrame(res.phases), sw=float(sm["SW_total"]), grid=st0.grid,
                end_reasons=sm["phase_end_reasons"])


def run_one(env: str, model: str, seed=None) -> str:
    warnings.simplefilter("ignore")
    cfg = config_for(env, model, seed)
    t = time.perf_counter()
    print(f"[{env}/{model}] start", flush=True)
    try:
        if model == "fluid":
            out = _fluid(cfg)
        else:
            out = _greedy(cfg) if model == "greedy_pos" else _online(cfg, frozen=model == "no_learning")
            a = out["arrays"]
            out.update(reward=float(a["R"].sum()), admitted=int(a["admit"].sum()),
                       payments=float(a["payment"].sum()), profit_exp=float(a["profit_exp"].sum()),
                       profit_belief=float(a["profit_belief"].sum()), profit_real=float(a["profit_real"].sum()))
            out["rejections"] = {r: int(out["rejections"].get(r, 0)) for r in REASONS}
    except Exception:
        print(f"[{env}/{model}] FAILED\n{traceback.format_exc()}", flush=True)
        return f"{env}/{model}: failed"
    out.update(env=env, model=model, config=asdict(cfg), f=cfg.capacity.f, seed=cfg.run.seed,
               runtime_s=time.perf_counter() - t)
    path(env, model).parent.mkdir(parents=True, exist_ok=True)
    with open(path(env, model), "wb") as fh:
        pickle.dump(out, fh)
    print(f"[{env}/{model}] done in {out['runtime_s'] / 60:,.1f} min: reward {out['reward']:,.2f}", flush=True)
    return f"{env}/{model}: ok"


# --------------------------------------------------------------------------- #
def plan(envs, models):
    return [(e, m) for e in envs for m in models]


def summary_table() -> pd.DataFrame:
    rows = []
    for e in ENVS:
        for m in MODELS:
            p = path(e, m)
            if not p.is_file():
                continue
            with open(p, "rb") as fh:
                r = pickle.load(fh)
            rows.append(dict(env=e, model=m, reward=r["reward"], runtime_min=r["runtime_s"] / 60))
    df = pd.DataFrame(rows)
    if len(df):
        fl = df[df.model == "fluid"].set_index("env").reward
        df["% of fluid OPT"] = [100 * r / fl.get(e, np.nan) for e, r in zip(df.env, df.reward)]
    return df


def main(argv=None) -> None:
    from dlent_rt.scenarios import resolve
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", nargs="+", default=list(ENVS), help="environments (none, s1, s6, s2 or full names)")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    ap.add_argument("--workers", type=int, default=1, help="runs in parallel, each its own process")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="rerun cached runs")
    ap.add_argument("--dry-run", action="store_true", help="list the runs, run nothing")
    ap.add_argument("--report-only", action="store_true", help="only the summary table from the cache")
    a = ap.parse_args(argv)
    envs = [resolve(e) for e in a.scenario]
    runs = plan(envs, a.models)
    todo = [(e, m) for e, m in runs if a.force or not _cached(e, m, a.seed)]
    if a.dry_run:
        for e, m in runs:
            print(f"{e:16s} {m:12s} {'to run' if (e, m) in todo else 'cached':7s} {'long' if m in LONG else 'fast'}")
        n_long = sum(m in LONG for _, m in todo)
        print(f"{len(todo)} run(s) to do ({n_long} long), {a.workers} at a time. Long runs: about 3-4 h each; "
              f"no_learning about 0.5 h; fluid and greedy_pos about 1 min.")
        return
    if not a.report_only and todo:
        todo.sort(key=lambda r: (ENVS.index(r[0]), r[1] not in LONG))     # per environment, long runs first
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn")) as pool:
            futs = [pool.submit(run_one, e, m, a.seed) for e, m in todo]
            for f in as_completed(futs):
                print(f.result(), flush=True)
    print(summary_table().to_string(index=False))


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    main()

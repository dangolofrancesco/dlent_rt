"""
The practical model as ablations of the theoretical model. Each element is one config switch on top of
configs/default.yaml (the theoretical model); everything else (seed, f, data) is the same as the theoretical run.

  thinning_off         ablation.thinning_off = true          (p_U = 1)
  stats_by_end_reason  stats.update = "by_end_reason"        (merge after timeout / Test 1, replace after Test 2 / 3)
  sigma_cumulative     bid.sigma_update = "cumulative"       (sigma refit on H's and every phase's markups pooled)
  sigma_fixed          bid.sigma_update = "fixed"            (Phase 0's sigma, never updated)
  live_oracle          oracle.kind = "live"                  (the job's own power x d_bar x price / carbon at arrival)

    dlent_venv/bin/python scripts/run_ablations.py --single               # each element alone (5 runs)
    dlent_venv/bin/python scripts/run_ablations.py --cumulative           # thinning off, + stats by end reason,
                                                                          # + sigma cumulative, + live oracle (4 runs)
    dlent_venv/bin/python scripts/run_ablations.py --single --dry-run     # the runs and the commands, nothing run
    options: --workers N (parallel processes, default 1), --force (rerun cached runs), --seed S, --report-only

Each run is its own process (spawn) and is cached in reports/week33/cache/ablations/<run>.pkl (skipped if the cache
exists and was made with the same config, unless --force). The comparison uses the theoretical run
(reports/week33/cache/theory_run.pkl) and the stream benchmarks at the same f (benchmarks_f.pkl: fluid OPT and the
no-learning model); run scripts/model_check_cache.py theory / bench first. The tables go to stdout and to
reports/week33/ablations_results.md.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pickle
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
CACHE = ROOT / "reports" / "week33" / "cache"
OUT_DIR = CACHE / "ablations"
REPORT = ROOT / "reports" / "week33" / "ablations_results.md"

ELEMENTS = {
    "thinning_off": lambda c: replace(c, ablation=replace(c.ablation, thinning_off=True)),
    "stats_by_end_reason": lambda c: replace(c, stats=replace(c.stats, update="by_end_reason")),
    "sigma_cumulative": lambda c: replace(c, bid=replace(c.bid, sigma_update="cumulative")),
    "sigma_fixed": lambda c: replace(c, bid=replace(c.bid, sigma_update="fixed")),
    "live_oracle": lambda c: replace(c, oracle=replace(c.oracle, kind="live")),
}
CUMULATIVE = ("thinning_off", "stats_by_end_reason", "sigma_cumulative", "live_oracle")
REASONS = ("bin below k*", "below reserve", "thinning", "no space")


def plan(single: bool, cumulative: bool) -> list[tuple[str, tuple[str, ...]]]:
    """(run name, elements) of every run asked for."""
    out = []
    if single:
        out += [(f"single_{e}", (e,)) for e in ELEMENTS]
    if cumulative:
        out += [(f"cum{k}_{CUMULATIVE[k - 1]}", CUMULATIVE[:k]) for k in range(1, len(CUMULATIVE) + 1)]
    return out


def config_for(elements, seed=None):
    from dlent_rt.config import default_config
    cfg = default_config()
    if seed is not None:
        cfg = replace(cfg, run=replace(cfg.run, seed=int(seed)))
    for e in elements:
        cfg = ELEMENTS[e](cfg)
    return cfg.validate()


# --------------------------------------------------------------------------- #
# One run (in its own process)
# --------------------------------------------------------------------------- #
def _kstar_now(m, st, a) -> int:
    """k* of an arrival recomputed on its phase state: the frozen threshold, or the live one (the job's own costs)."""
    from dlent_rt.mechanism import threshold_kstar
    ts = st.type_space
    qi = int(np.searchsorted(ts.priorities, a["q"]))
    if not m.live:
        return int(threshold_kstar((st.r_tilde - st.v @ st.tau).reshape(ts.k_star, len(ts.priorities), ts.n_hw))[qi, a["hw"]])
    m.state = st
    m.dq_h = st.col(st.d_bar).reshape(len(ts.priorities), ts.n_hw) * m.cfg.time.tick_seconds / 3600.0
    return int(threshold_kstar(m._live_scores(a["i"], qi, a["hw"])))


def spot_check(m, res, seed: int, n_each: int = 200) -> dict:
    """The truthfulness spot check of the 48 h batch (truthful_batch.spot_check), with k* recomputed for either oracle:
    200 admitted + 200 rejected arrivals, bids x 0.5, 0.8, 1.25, 2 on the same state."""
    from dlent_rt.mechanism import admit_and_pay
    from truthful_batch import FACTORS
    rng = np.random.default_rng(seed)
    A = res.arrivals
    ia = [k for k, a in enumerate(A) if a["admit"]]
    ir = [k for k, a in enumerate(A) if not a["admit"]]
    pick = list(rng.choice(ia, min(n_each, len(ia)), replace=False)) + list(rng.choice(ir, min(n_each, len(ir)), replace=False))
    bad, mismatch, checked = 0, 0, 0
    for k in pick:
        a = A[k]
        st = m.states[a["m"] - 1]
        ks = _kstar_now(m, st, a)
        mismatch += int(ks != a["k_star"])
        for f in FACTORS:
            checked += 1
            ok_b, ok_r, _, p = admit_and_pay(a["v"] * f, st.grid, ks, a["reserve"])
            adm_f = ok_b and ok_r and a["U"] and a["fits"]
            if a["admit"]:
                bad += int(adm_f and p < a["payment_rate"]) + int(f > 1 and not adm_f)
            else:
                bad += int(adm_f and p < a["v"]) + int(f < 1 and adm_f)
    return {"decisions re-run": checked, "logged k* != recomputed k*": mismatch, "violations": bad}


def upper_set(m, res) -> int:
    """Upper-set violations: in every phase state (a rejected bin above an admitted one in a column), and in the run:
    frozen oracle, a bin refused by the price rule above an accepted one in the same phase and column; live oracle
    (k* per job), an arrival whose decision is not bin >= its own k*."""
    v = 0
    for st in m.states:
        ok = (st.r_tilde - st.v @ st.tau).reshape(st.grid.k_star, -1) > 0
        v += int((ok[:-1] & ~ok[1:]).sum())
    if m.live:
        return v + sum(int(a["k_hat"] != (a["k_star"] > 0 and a["k_bin"] >= a["k_star"])) for a in res.arrivals)
    seen = {}
    for a in res.arrivals:
        seen.setdefault((a["m"], a["q"], a["hw"]), []).append((a["k_bin"], a["k_hat"]))
    for x in seen.values():
        x = np.array(x)
        lo, hi = x[x[:, 1] == 1, 0], x[x[:, 1] == 0, 0]
        if lo.size and hi.size:
            v += int((hi > lo.min()).sum())
    return v


def run_one(name: str, elements: tuple, seed=None) -> dict:
    warnings.simplefilter("ignore")
    from collections import Counter
    from dlent_rt.data import load_dataset
    from dlent_rt.phase0 import run_phase0
    from truthful_batch import _Recording
    cfg = config_for(elements, seed)
    print(f"[{name}] start: {', '.join(elements)}", flush=True)
    ds = load_dataset(cfg)
    st0 = run_phase0(ds, cfg)
    t = time.perf_counter()
    m = _Recording(ds, st0, cfg)
    res = m.run()
    sm = res.summary
    adm = np.array([a["i"] for a in res.arrivals if a["admit"]], np.int64)
    J = m.J
    reasons = Counter(a["reason"] for a in res.arrivals if not a["admit"])
    out = dict(
        name=name, elements=tuple(elements), config=asdict(cfg), f=cfg.capacity.f, seed=cfg.run.seed,
        runtime_s=time.perf_counter() - t,
        split={"customer value": float(J.cv[adm].sum()), "profit": float(J.profit[adm].sum()),
               "carbon": float(-J.carbon_usd[adm].sum()), "reward": float(sm["R_total"])},
        social_welfare=float(sm["SW_total"]), payments=float(sm["payments_total"]),
        myerson=dict(expected_profit=float(J.profit[adm].sum()), realised_profit=float(sm["profit_realised_total"])),
        admitted=int(sm["n_admitted"]), rejections={r: int(reasons.get(r, 0)) for r in REASONS},
        n_phases=int(sm["n_phases"]), end_reasons=sm["phase_end_reasons"],
        phases=pd.DataFrame(res.phases), ir_violations=int(sm["n_ir_violations"]),
        upper_set_violations=upper_set(m, res), spot_check=spot_check(m, res, cfg.run.seed), summary=sm)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"{name}.pkl", "wb") as fh:
        pickle.dump(out, fh)
    print(f"[{name}] done in {out['runtime_s'] / 60:,.0f} min: reward {out['split']['reward']:,.2f}", flush=True)
    return {k: v for k, v in out.items() if k != "phases"}


def _cached(name, elements, seed) -> bool:
    p = OUT_DIR / f"{name}.pkl"
    if not p.is_file():
        return False
    with open(p, "rb") as fh:
        return pickle.load(fh)["config"] == asdict(config_for(elements, seed))


# --------------------------------------------------------------------------- #
# The tables
# --------------------------------------------------------------------------- #
def md(df: pd.DataFrame) -> str:
    """A markdown table (no tabulate dependency); numbers with 4 significant digits."""
    fmt = lambda x: f"{x:,.4g}" if isinstance(x, (float, np.floating)) else str(x)
    head = "| " + " | ".join([str(df.index.name or "")] + [str(c) for c in df.columns]) + " |"
    rows = ["| " + " | ".join([str(i)] + [fmt(v) for v in r]) + " |" for i, r in zip(df.index, df.values)]
    return "\n".join([head, "|" + "---|" * (len(df.columns) + 1), *rows])


def _load(p: Path):
    with open(p, "rb") as fh:
        return pickle.load(fh)


def baselines(f: float) -> dict:
    """Theoretical run (theory_run.pkl), no-learning and fluid OPT (benchmarks_f.pkl) at the same f."""
    th, b = _load(CACHE / "theory_run.pkl"), _load(CACHE / "benchmarks_f.pkl")
    if th["f"] != f or b["f"] != f:
        raise ValueError(f"baselines at f = {th['f']} / {b['f']}, ablations at f = {f}: rerun "
                         f"scripts/model_check_cache.py theory and bench")
    A = th["arrivals"]
    adm = A[A.admit]
    sm = th["summary"]
    return dict(theory=dict(reward=sm["R_total"], customer_value=float(adm.value.sum()), sw=sm["SW_total"],
                            payments=sm["payments_total"], admitted=sm["n_admitted"]),
                no_learning=b["no_learning"]["split"]["reward"], fluid=b["fluid"]["split"]["reward"])


def row(r: dict, base: dict) -> dict:
    s = r["split"]
    return {"elements": " + ".join(r["elements"]), "reward $": s["reward"], "customer value $": s["customer value"],
            "profit $": s["profit"], "carbon $": s["carbon"], "social welfare $": r["social_welfare"],
            "payments $": r["payments"], "Myerson: expected sum(phi d - C_elec) $": r["myerson"]["expected_profit"],
            "Myerson: payments - C_elec $": r["myerson"]["realised_profit"], "admitted": r["admitted"],
            **{f"rejected: {k}": v for k, v in r["rejections"].items()}, "phases": r["n_phases"],
            "end reasons": ", ".join(f"{k} {v}" for k, v in r["end_reasons"].items() if v),
            "IR violations": r["ir_violations"], "upper-set violations": r["upper_set_violations"],
            "spot-check violations": r["spot_check"]["violations"],
            "% of fluid OPT": 100 * s["reward"] / base["fluid"],
            "delta vs theoretical $": s["reward"] - base["theory"]["reward"],
            "delta vs no-learning $": s["reward"] - base["no_learning"], "runtime min": r["runtime_s"] / 60}


def report(runs, seed, res=None, base=None) -> str:
    """The tables of the cached runs (res / base can be passed directly, e.g. by the tests)."""
    if res is None:
        done = [n for n, _ in runs if (OUT_DIR / f"{n}.pkl").is_file()]
        if not done:
            return "no cached ablation run yet"
        res = {n: _load(OUT_DIR / f"{n}.pkl") for n in done}
    done = [(n, res[n]["elements"]) for n, _ in runs if n in res]
    base = base or baselines(next(iter(res.values()))["f"])
    lines = [f"# Practical model: ablations of the theoretical model (f = {next(iter(res.values()))['f']}, "
             f"seed {next(iter(res.values()))['seed']})", "",
             f"Theoretical run: reward {base['theory']['reward']:,.2f} $; no-learning {base['no_learning']:,.2f} $; "
             f"fluid OPT {base['fluid']:,.2f} $.", ""]
    for title, pref in (("Each element alone (--single)", "single_"), ("Added one by one (--cumulative)", "cum")):
        names = [n for n, _ in done if n.startswith(pref)]
        if not names:
            continue
        tab = pd.DataFrame({n: row(res[n], base) for n in names}).T
        if pref == "cum":
            prev = [base["theory"]["reward"]] + [res[n]["split"]["reward"] for n in names[:-1]]
            tab.insert(2, "delta of this step $", [res[n]["split"]["reward"] - p for n, p in zip(names, prev)])
        lines += [f"## {title}", "", md(tab.T), ""]
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--single", action="store_true", help="each element alone on top of the theoretical model")
    ap.add_argument("--cumulative", action="store_true", help="the elements added one by one, in order")
    ap.add_argument("--workers", type=int, default=1, help="runs in parallel (each its own process)")
    ap.add_argument("--seed", type=int, default=None, help="run.seed (default: the configured one)")
    ap.add_argument("--force", action="store_true", help="rerun runs that are already cached")
    ap.add_argument("--dry-run", action="store_true", help="print the runs and the commands, run nothing")
    ap.add_argument("--report-only", action="store_true", help="only the tables, from the cache")
    a = ap.parse_args(argv)
    if not (a.single or a.cumulative):
        ap.error("choose --single and / or --cumulative")
    runs = plan(a.single, a.cumulative)
    todo = [(n, e) for n, e in runs if a.force or not _cached(n, e, a.seed)]
    if a.dry_run:
        for n, e in runs:
            print(f"{n:32s} {' + '.join(e):70s} {'to run' if (n, e) in todo else 'cached'}")
        print(f"{len(todo)} run(s) to do, {a.workers} at a time; each takes about as long as the theoretical run "
              f"(scripts/model_check_cache.py theory) at the same f")
        return
    if not a.report_only and todo:
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn")) as pool:
            list(pool.map(run_one, [n for n, _ in todo], [e for _, e in todo], [a.seed] * len(todo)))
    text = report(runs, a.seed)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text)
    print(text)
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    main()

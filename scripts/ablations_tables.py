"""
The tables of reports/week33/ablations_results.md, from the caches only (nothing is run):
  theory_run.pkl, benchmarks_f.pkl (fluid OPT, static, no-learning), greedy_pos.pkl, ablations/*.pkl,
  discretization_milp.pkl (OPT_true on the first 7 days) and week_fluid.pkl (fluid OPT on that window).

    dlent_venv/bin/python scripts/ablations_tables.py      # -> reports/week33/ablations_numbers.md
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
C = ROOT / "reports" / "week33" / "cache"
OUT = ROOT / "reports" / "week33" / "ablations_numbers.md"
REASONS = ("bin below k*", "below reserve", "thinning", "no space")
SINGLE = (("single_thinning_off", "thinning off"), ("single_live_oracle", "live oracle"),
          ("single_sigma_cumulative", "σ cumulative"), ("single_sigma_fixed", "σ fixed"),
          ("single_stats_by_end_reason", "stats by end reason"))
CUM = (("cum1_thinning_off", "C1 = thinning off"), ("cum2_live_oracle", "C2 = C1 + live oracle"),
       ("cum3_stats_by_end_reason", "C3 = C2 + stats by end reason"))


def _load(p):
    with open(p, "rb") as fh:
        return pickle.load(fh)


def md(df: pd.DataFrame, fmt="{:,.1f}") -> str:
    f = lambda x: (fmt.format(x) if isinstance(x, (float, np.floating)) and np.isfinite(x)
                   else "–" if isinstance(x, (float, np.floating)) or x is None else f"{x:,}" if isinstance(x, (int, np.integer)) else str(x))
    head = "| " + " | ".join([str(df.index.name or "")] + [str(c) for c in df.columns]) + " |"
    return "\n".join([head, "|" + "---|" * (len(df.columns) + 1),
                      *["| " + " | ".join([str(i)] + [f(v) for v in r]) + " |" for i, r in zip(df.index, df.values)]])


def runs() -> tuple[dict, dict]:
    """name -> one dict per model (overall numbers + R per stream arrival); and the context (fluid, ticks, phases)."""
    t, b, g = _load(C / "theory_run.pkl"), _load(C / "benchmarks_f.pkl"), _load(C / "greedy_pos.pkl")
    assert t["f"] == b["f"] == g["f"], "caches at different f"
    tick, Rj = np.asarray(t["tick"]), np.asarray(t["R_job"])
    out = {}
    fl = b["fluid"]["split"]
    out["fluid OPT"] = dict(reward=fl["reward"], cv=fl["customer value"], profit=fl["profit"], carbon=fl["carbon"])
    for key, lab, x in (("static", "static Phase 0", b["static"]), ("greedy", "Greedy-Pos", g),
                        ("no_learning", "no-learning", b["no_learning"])):
        s, m = x["split"], x["mechanism"]
        adm = np.asarray(x["admit"], bool)
        out[lab] = dict(reward=s["reward"], cv=s["customer value"], profit=s["profit"], carbon=s["carbon"],
                        sw=s["social welfare"], pay=m["payments"], adm=x["n_admitted"],
                        **{f"rej {k}": m["rejections"][k] for k in REASONS}, my_exp=s["profit"],
                        my_real=x["profit_payments"], ir=m["n_ir_violations"], up=m["n_upper_set_violations"],
                        R_arr=np.where(adm, Rj, 0.0))
    sm, ck, A = t["summary"], t["checks"], t["arrivals"]
    adm = A.admit.to_numpy()
    cv, prof = float(A.value[adm].sum()), ck["myerson"]["expected_profit"]
    out["DLENT-RT (theoretical)"] = dict(
        reward=sm["R_total"], cv=cv, profit=prof, carbon=sm["R_total"] - cv - prof, sw=sm["SW_total"],
        pay=sm["payments_total"], adm=sm["n_admitted"], **{f"rej {k}": ck["run"]["rejections"][k] for k in REASONS},
        phases=sm["n_phases"], ends=sm["phase_end_reasons"], my_exp=prof, my_real=ck["myerson"]["realised_profit"],
        ir=ck["run"]["IR violations"], up=ck["run"]["upper-set violations (phase states)"]
        + ck["run"]["upper-set violations (run: a bin rejected above an admitted bin, same phase and column)"],
        spot=ck["spot_check"]["violations"], R_arr=np.where(adm, Rj, 0.0))
    for n, lab in SINGLE + CUM:
        p = C / "ablations" / f"{n}.pkl"
        if not p.is_file():
            continue
        r = _load(p)
        assert np.array_equal(r["tick"], tick) and r["f"] == t["f"] and r["seed"] == t["summary"].get("seed", r["seed"])
        s = r["split"]
        out[lab] = dict(reward=s["reward"], cv=s["customer value"], profit=s["profit"], carbon=s["carbon"],
                        sw=r["social_welfare"], pay=r["payments"], adm=r["admitted"],
                        **{f"rej {k}": r["rejections"][k] for k in REASONS}, phases=r["n_phases"], ends=r["end_reasons"],
                        my_exp=r["myerson"]["expected_profit"], my_real=r["myerson"]["realised_profit"],
                        ir=r["ir_violations"], up=r["upper_set_violations"], spot=r["spot_check"]["violations"],
                        R_arr=np.asarray(r["R_arr"], float), reused_from=r.get("reused_from"), runtime=r["runtime_s"] / 60)
    return out, dict(fluid=fl["reward"], tick=tick, phases=t["phases"], theory=sm["R_total"],
                     no_learning=b["no_learning"]["split"]["reward"])


def overall(R, ctx, names) -> pd.DataFrame:
    rows = {}
    for n in names:
        if n not in R:
            continue
        x = R[n]
        e = x.get("ends")
        rows[n] = {"reward $": x["reward"], "% of fluid OPT": 100 * x["reward"] / ctx["fluid"],
                   "Δ vs theoretical $": x["reward"] - ctx["theory"] if n != "fluid OPT" else np.nan,
                   "Δ vs no-learning $": x["reward"] - ctx["no_learning"] if n != "fluid OPT" else np.nan,
                   "customer value $": x["cv"], "profit (φd − C_elec) $": x["profit"], "carbon $": x["carbon"],
                   "social welfare $": x.get("sw", np.nan), "payments $": x.get("pay", np.nan),
                   "admitted jobs": x.get("adm", np.nan),
                   **{f"rejected: {k}": x.get(f"rej {k}", np.nan) for k in REASONS},
                   "phases": x.get("phases", np.nan),
                   "end reasons (Test 1 / timeout / horizon)":
                       f"{e['test1']} / {e['timeout']} / {e['horizon']}" if e else "–",
                   "Myerson: Σ(φd − C_elec) $": x.get("my_exp", np.nan),
                   "Myerson: payments − C_elec $": x.get("my_real", np.nan),
                   "IR violations": x.get("ir", np.nan), "upper-set violations": x.get("up", np.nan),
                   "spot-check violations": x.get("spot", np.nan)}
    return pd.DataFrame(rows)


def per_phase(R, ctx, names) -> pd.DataFrame:
    """Each run's reward over the time windows of the theoretical run's phases."""
    tick, rows = ctx["tick"], []
    for _, p in ctx["phases"].iterrows():
        w = (tick >= p.tau_start) & (tick <= p.tau_end)
        rows.append({"phase": int(p.m), "hours": f"{(p.tau_start - 1) / 3600:.1f}–{p.tau_end / 3600:.1f}",
                     "end (theoretical)": p.reason,
                     **{n: float(R[n]["R_arr"][w].sum()) for n in names if n in R}})
    df = pd.DataFrame(rows).set_index("phase")
    tot = {c: (df[c].sum() if df[c].dtype.kind == "f" else "") for c in df.columns}
    tot["hours"] = f"{ctx['phases'].W.sum() / 3600:.1f} h"
    df.loc["total"] = tot
    return df


def week(R, names) -> pd.DataFrame:
    d, wf = _load(C / "discretization_milp.pkl"), _load(C / "week_fluid.pkl")
    assert d["days"] == wf["days"] == (0.0, 7.0) and d["n_jobs"] == wf["n_jobs"]
    t = _load(C / "theory_run.pkl")
    tick = np.asarray(t["tick"])
    w = (tick > 0) & (tick <= 7 * 86400)                                   # the MILP's window (theory_eval)
    assert int(w.sum()) == d["n_jobs"]
    best, bound = d["true"]["best"], d["true"]["bound"]
    rows = {"fluid OPT on the window (volume inside it)": wf["fluid"],
            "OPT_true, job-level MILP (bound)": bound, "OPT_true, job-level MILP (best)": best,
            "OPT_bin (best)": d["bin"]["best"]}
    rows |= {n: float(R[n]["R_arr"][w].sum()) for n in names if n in R and "R_arr" in R[n]}
    df = pd.DataFrame({"reward over the first 7 days $": rows})
    df["% of OPT_true (best)"] = 100 * df.iloc[:, 0] / best
    df["% of OPT_true (bound)"] = 100 * df.iloc[:, 0] / bound
    return df, int(w.sum())


def main() -> None:
    R, ctx = runs()
    base = ["fluid OPT", "static Phase 0", "Greedy-Pos", "no-learning", "DLENT-RT (theoretical)"]
    single = [lab for _, lab in SINGLE]
    cum = [lab for _, lab in CUM]
    out = ["# Ablation tables (from the caches; scripts/ablations_tables.py)", ""]
    out += ["## All runs, whole stream", "", md(overall(R, ctx, base + single + cum)), ""]
    co = overall(R, ctx, ["DLENT-RT (theoretical)"] + cum + ["static Phase 0", "Greedy-Pos", "fluid OPT"])
    rew = [R[n]["reward"] for n in ["DLENT-RT (theoretical)"] + cum if n in R]
    step = pd.Series([np.nan] + list(np.diff(rew)), index=["DLENT-RT (theoretical)"] + [c for c in cum if c in R])
    co.loc["step Δ $ (this switch)"] = step.reindex(co.columns)
    out += ["## Cumulative", "", md(co), ""]
    out += ["## Per phase (windows of the theoretical run's phases)", "",
            md(per_phase(R, ctx, ["DLENT-RT (theoretical)", "no-learning", "static Phase 0", "Greedy-Pos"] + single + cum)), ""]
    wk, n = week(R, base[1:] + single + cum)
    out += [f"## First 7 days of the stream ({n:,} jobs; the discretization MILP's window)", "",
            md(wk), ""]
    OUT.write_text("\n".join(out))
    print("\n".join(out))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

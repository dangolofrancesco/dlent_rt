"""
Tables of the non-stationary experiment (week 34, Part D) from the caches of scripts/run_scenarios.py; nothing is run.
Used by notebooks/src/02_nonstationary.py and reports/week34/scenarios_results.md.

    dlent_venv/bin/python scripts/scenarios_analysis.py           # every table, as Markdown
"""
from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from run_scenarios import CACHE, ENVS, EXTRA, MODELS, config_for, path  # noqa: E402

LABEL = dict(fluid="fluid OPT", practical="Practical (C3)", practical_calibrated="Practical + calibrated tests",
             greedy_pos="Greedy-Pos", no_learning="No learning", theoretical="Theoretical")
ENV_LABEL = dict(none="none (stationary)", s1_bids_fall="S1 bids fall", s6_bids_spread="S6 bids spread",
                 s2_bids_rise="S2 bids rise")
H_PER_TICK = 1.0 / 3600.0


def load(env, model):
    p = path(env, model)
    if not p.is_file():
        return None
    with open(p, "rb") as fh:
        return pickle.load(fh)


def models_cached():
    return [m for m in MODELS + EXTRA if all(path(e, m).is_file() for e in ENVS)]


def cfg(env):
    return config_for(env, "practical")


def rewards() -> pd.DataFrame:
    """Reward before / after t*, admissions and % of fluid OPT, per environment and model."""
    from dlent_rt.scenarios import before_after
    rows = []
    for e in ENVS:
        fl = load(e, "fluid")
        for m in models_cached():
            r = load(e, m)
            if m == "fluid":
                pre, post, adm = r["reward_pre"], r["reward_post"], (np.nan, np.nan)
            else:
                a = r["arrays"]
                pre, post = before_after(a["R"], a["tick"], cfg(e))
                adm = before_after(a["admit"].astype(float), a["tick"], cfg(e))
            rows.append({"env": e, "model": m, "reward": pre + post, "before t*": pre, "after t*": post,
                         "% OPT": 100 * (pre + post) / fl["reward"], "% OPT before": 100 * pre / fl["reward_pre"],
                         "% OPT after": 100 * post / fl["reward_post"], "admitted before": adm[0],
                         "admitted after": adm[1]})
    return pd.DataFrame(rows)


def learning_gain(rw: pd.DataFrame) -> pd.DataFrame:
    """Practical vs no learning (the value of learning) and practical vs theoretical, after t*."""
    rows = []
    for e in ENVS:
        x = rw[rw.env == e].set_index("model")
        row = {"env": e}
        for m in [m for m in ("practical", "practical_calibrated") if m in x.index]:
            row[f"{m} / no learning, after t*"] = x.loc[m, "after t*"] / x.loc["no_learning", "after t*"]
            row[f"{m} / theoretical, after t*"] = x.loc[m, "after t*"] / x.loc["theoretical", "after t*"]
        row["no learning, after t*"] = x.loc["no_learning", "after t*"]
        rows.append(row)
    return pd.DataFrame(rows)


def detection() -> pd.DataFrame:
    """Per learning model: phase end reasons, the delay from t* to the first phase end (and to a Test 2 / 3 end),
    and to the first phase that starts with mu-hat within 0.1 of the true post-change mu (S1, S2) or sigma-hat within
    0.05 of the true sigma (S6)."""
    from dlent_rt import scenarios as sc
    rows = []
    for e in ENVS:
        c = cfg(e)
        tr, ts = sc.truth(c), sc.t_star_step(c) * H_PER_TICK
        for m in [m for m in ("practical", "practical_calibrated", "theoretical") if m in models_cached()]:
            ph = load(e, m)["phases"]
            d = sc.detection(ph, c)
            tab = sc.markup_per_phase(ph, c)
            after = tab[tab["start h"] >= ts]
            if e == "s6_bids_spread":
                ok = after[np.abs(after.sigma_hat - tr["sigma1"]) < 0.05]
            elif e == "none":
                ok = after.iloc[0:0]
            else:
                ok = after[np.abs(after.mu_hat - tr["mu1"]) < 0.1]
            ends = ph.reason.value_counts().to_dict()
            rows.append({"env": e, "model": m, "phases": len(ph),
                         "ends": ", ".join(f"{k} {v}" for k, v in sorted(ends.items())),
                         "first end after t* (h)": d["delay_any_h"], "by": d["first_reason"],
                         "Test 2/3 delay (h)": d["delay_test23_h"],
                         "adapted phase starts (h after t*)": float(ok["start h"].iloc[0] - ts) if len(ok) else np.nan})
    return pd.DataFrame(rows)


def markups(env, model="practical") -> pd.DataFrame:
    from dlent_rt.scenarios import markup_per_phase
    return markup_per_phase(load(env, model)["phases"], cfg(env))


def myerson() -> pd.DataFrame:
    """Expected profit (phi d - C_elec, admitted jobs) under the true markup law and under the run's own belief, and
    the realised profit (payment - C_elec), before / after t*."""
    from dlent_rt.scenarios import before_after
    rows = []
    for e in ENVS:
        for m in [m for m in models_cached() if m != "fluid"]:
            a = load(e, m)["arrays"]
            row = {"env": e, "model": m}
            for k, name in (("profit_exp", "expected (true)"), ("profit_belief", "expected (belief)"),
                            ("profit_real", "realised")):
                pre, post = before_after(a[k], a["tick"], cfg(e))
                row[f"{name} before"], row[f"{name} after"] = pre, post
            row["belief / realised after"] = row["expected (belief) after"] / row["realised after"]
            rows.append(row)
    return pd.DataFrame(rows)


def bin1() -> pd.DataFrame:
    from dlent_rt.scenarios import bin1_share, post_change
    rows = []
    for e in ENVS:
        r = load(e, "practical")
        a, c = r["arrays"], cfg(e)
        pre = ~post_change(a["tick"], c)
        rows.append({"env": e, "bin 1 share before t*": float(np.mean(np.asarray(r["grid"].assign(a["v"][pre])) == 1)),
                     "bin 1 share after t*": bin1_share(a["v"], a["tick"], r["grid"], c)})
    return pd.DataFrame(rows)


def reward_rate(env, model, window_h=24.0) -> pd.DataFrame:
    """Reward per hour in consecutive windows (for the plots)."""
    a = load(env, model)["arrays"]
    h = (a["tick"] - 1) * H_PER_TICK
    edges = np.arange(0, h.max() + window_h, window_h)
    s, _ = np.histogram(h, bins=edges, weights=a["R"])
    return pd.DataFrame({"hour": edges[:-1] + window_h / 2, "reward per h": s / window_h})


def week33_stationary() -> pd.DataFrame:
    """The stationary runs of week 34 (mu-hat estimated) against week 33's (mu = -sigma^2/2)."""
    w = ROOT / "reports" / "week33" / "cache"
    old = {}
    p = w / "ablations" / "cum3_stats_by_end_reason.pkl"
    if p.is_file():
        old["practical"] = float(pickle.load(open(p, "rb"))["R_arr"].sum())
    p = w / "theory_run.pkl"
    if p.is_file():
        t = pickle.load(open(p, "rb"))
        res = t.get("result", t)
        s = getattr(res, "summary", None) or (res.get("summary") if isinstance(res, dict) else None)
        if s:
            old["theoretical"] = float(s["R_total"])
    p = w / "greedy_pos.pkl"
    if p.is_file():
        g = pickle.load(open(p, "rb"))
        if isinstance(g, dict) and "reward" in g:
            old["greedy_pos"] = float(g["reward"])
    rows = []
    for m, v in old.items():
        new = load("none", m)["reward"]
        rows.append({"model": m, "week 33": v, "week 34": new, "change %": 100 * (new / v - 1)})
    return pd.DataFrame(rows)


def md(df: pd.DataFrame, fmt="{:,.2f}") -> str:
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = [("" if not np.isfinite(x) else fmt.format(x)) for x in d[c]]
    d = d.replace({"env": ENV_LABEL, "model": LABEL}).astype(str)
    lines = ["| " + " | ".join(d.columns) + " |", "|" + "---|" * d.shape[1]]
    lines += ["| " + " | ".join(r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(lines)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    rw = rewards()
    print("## rewards\n", md(rw), "\n")
    print("## learning gain\n", md(learning_gain(rw), "{:,.3f}"), "\n")
    print("## detection\n", md(detection(), "{:,.1f}"), "\n")
    for e in ENVS:
        for m in ("practical", "theoretical"):
            print(f"## markups {e} {m}\n", md(markups(e, m), "{:,.4f}"), "\n")
    print("## myerson\n", md(myerson()), "\n")
    print("## bin 1\n", md(bin1(), "{:,.4f}"), "\n")
    print("## week 33 vs 34\n", md(week33_stationary()), "\n")

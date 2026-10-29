"""
Week 32 (v6): the size_u0 bid (priority only in the customer value). The u0 scan (Phase 0 at f = 0.5 and 1.25),
the f selection with the static policy (dual rule), and the theory summary v6 vs v5 (no MILP). No online model.

    dlent_venv/bin/python scripts/week32/v6_bid.py

Writes reports/week32/v6_bid_numbers.md.
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "week32"))

from limits import lines, out, pct, table                    # noqa: E402
from theory_f050 import summary                              # noqa: E402
from dlent_rt import diagnostics as dg                       # noqa: E402
from dlent_rt import theory_eval as te                       # noqa: E402
from dlent_rt.config import load_config                      # noqa: E402
from dlent_rt.data import load_dataset                       # noqa: E402
from dlent_rt.phase0 import run_phase0                       # noqa: E402

OUT = ROOT / "reports" / "week32"
U0 = (0.0, 3.26, 10.0, 23.93, 50.0)
F_SEL = (0.3, 0.4, 0.5, 0.6, 0.7)


def main() -> None:
    warnings.simplefilter("ignore")
    out("# Week 32: v6 bid (size_u0), all numbers\n")
    c6 = load_config(ROOT / "configs" / "v6_theory.yaml")
    c050 = replace(c6, capacity=replace(c6.capacity, f=0.5)).validate()      # the scans at f = 0.5 and f = 1.25
    c125 = load_config(ROOT / "configs" / "v6_size_bid.yaml")
    d050, d125 = load_dataset(c050), load_dataset(c125)
    assert np.array_equal(d050.H.collection_id, d125.H.collection_id)

    # ---- u0 scan --------------------------------------------------------------------------------------
    rows = []
    for u0 in U0:
        r = {"u0 $/machine-h": u0}
        for tag, cfg, ds in (("f 0.5", c050, d050), ("f 1.25", c125, d125)):
            c = replace(cfg, bid=replace(cfg.bid, u0_per_machine_hour=u0)).validate()
            st = run_phase0(ds, c)
            s = dg.phase0_summary(ds, st, c)
            if tag == "f 0.5":
                sh = dg.term_shares(dg.type_terms(st, c))
                bf = dg.bid_frame(ds, st, c)
                ef = dg.energy_frame(ds, c)
                bph = bf.bid_per_machine_h.dropna()
                r.update({"s_C $/kg": f"{s['s_C']:.4g}", "positive types": pct(s["share_positive_reward_types"]),
                          "positive mass": pct(s["share_positive_reward_mass"]),
                          "|r| share CV / profit / carbon": f"{sh['customer value']:.3f} / {sh['profit']:.3f} / {sh['carbon $']:.3f}",
                          "bid per machine-h mean / p10 / p90 $": f"{bph.mean():.3g} / {bph.quantile(.1):.3g} / {bph.quantile(.9):.3g}",
                          "bid / electricity (median, jobs with electricity > 0)": f"{np.nanmedian(bf.bid / ef.C_elec.where(ef.C_elec > 0)):.4g}"})
            r[f"tau {tag}"] = str([round(x, 6) for x in s["tau"]])
            r[f"LP fill {tag}"] = f"{pct(s['lp_fill'][0])} / {pct(s['lp_fill'][1])}"
        rows.append(r)
    table(pd.DataFrame(rows), "N1. u0 scan (Phase 0, v6; tau and LP fill at f = 0.5 and f = 1.25)")

    # ---- f selection (static policy, dual rule) -------------------------------------------------------
    R = [te.evaluate_f(c050, f, seed=c050.run.seed, windows_h=()) for f in F_SEL]
    rows = [{"f": r["f"], "% time >= 95% full": pct(r["static"]["full_time"]),
             "mean occupancy / C (CPU)": pct(r["static"]["mean_occupancy_over_C"][0]),
             "realised reward $": f"{r['static']['split']['reward']:,.2f}", "fluid OPT $": f"{r['fluid']['split']['reward']:,.2f}",
             "tau Phase 0 LP": str([round(x, 6) for x in r["phase0_lp"]["tau"]])} for r in R]
    table(pd.DataFrame(rows), "N2. f selection on v6 (static Phase 0 policy, dual rule, seed 42)")
    f_sel = te.select_f(R)
    out(f"\nFixed rule (closest to 20-30%, ties to 25%): f = {f_sel:g}. configs/v6_theory.yaml has f = {c6.capacity.f}.")

    # ---- theory summary v6 vs v5 ---------------------------------------------------------------------------
    s5 = summary(load_config(ROOT / "configs" / "v5_theory.yaml"))
    s6a = summary(c050)
    s6 = summary(c6)
    both = s5.rename(columns={"value": "v5_theory (size bid, f 0.5)"}).merge(
        s6a.rename(columns={"value": "v6 (size_u0 bid) at f 0.5"}), on="quantity").merge(
        s6.rename(columns={"value": f"v6_theory (size_u0 bid, f {c6.capacity.f:g})"}), on="quantity")
    table(both, "N3. Theory summary, v5 vs v6 (no MILP)")
    (OUT / "v6_bid_numbers.md").write_text("\n".join(lines) + "\n")
    print(f"SELECTED {f_sel}")


if __name__ == "__main__":
    main()

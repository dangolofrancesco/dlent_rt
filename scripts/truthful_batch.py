"""
Week 33: validation of the truthful threshold mechanism on a small real batch (before any full run).

    dlent_venv/bin/python scripts/truthful_batch.py [hours=48] [out=reports/week33/truthful_batch.json]

Real data, default config (f = 0.5, seed 42): Phase 0 on H, then the THEORETICAL online model on the first `hours`
hours of the stream only. Writes the Phase 0 numbers, the run numbers, the per-column k* vs the jobs' bins, the
payments, the IR / upper-set checks and a truthfulness spot check (200 admitted + 200 rejected arrivals, bids x 0.5,
0.8, 1.25, 2, same state) to a JSON file.
"""
from __future__ import annotations

import json, sys, time, warnings
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

from dlent_rt.config import default_config
from dlent_rt.data import JobArrays, load_dataset
from dlent_rt.bid import PHI_M_FLOOR
from dlent_rt.diagnostics import term_shares, type_terms
from dlent_rt.reward import weighted_quantile
from dlent_rt.mechanism import admit_and_pay, threshold_kstar
from dlent_rt.models.theory import TheoryModel
from dlent_rt.theory_eval import lp_x

FACTORS = (0.5, 0.8, 1.25, 2.0)


class _Recording(TheoryModel):
    """Keeps the PhaseState of every phase (for the spot check: the decision is re-run on the same state)."""

    def run(self):
        self.states = []
        return super().run()

    def _new_phase(self, state, tau_prev, W_prev):
        self.states.append(state)
        super()._new_phase(state, tau_prev, W_prev)


def first_hours(ds, hours: float):
    """The dataset with the stream cut to its first `hours` hours (H, capacity and limits unchanged)."""
    T = int(round(hours * 3600 / ds.tick_seconds))
    S = ds.stream
    keep = S.arrival_step < T
    sub = JobArrays(n=int(keep.sum()), resources=S.resources,
                    **{f: getattr(S, f)[keep] for f in ("priority", "scheduling_class", "duration_hours",
                                                        "duration_steps", "arrival_step", "A", "hw_idx",
                                                        "collection_id", "datetime")})
    return replace(ds, stream=sub, T=T, n_arrivals_stream=sub.n)


def phase0_numbers(ds, st, cfg) -> dict:
    g, b = st.grid, st.bids.v_rate
    pop = st.p > 0
    x, lp = lp_x(st, ds.capacity, cfg)
    tt = type_terms(st, cfg)
    kH = g.assign(b)
    margin = (st.r_tilde - st.v @ st.tau).reshape(g.k_star, st.type_space.n_cols)
    ks = threshold_kstar(margin)
    from dlent_rt.phase0 import column_ids
    cH = column_ids(st.type_space, ds.H.priority, ds.H.hw_idx)
    return {
        "U0 ($/h)": st.u0, "sigma": st.myerson.sigma,
        "grid": {"v_min ($/h)": float(g.bnd_v[0]), "v_max ($/h)": float(g.bnd_v[-1]), "rho": g.ratio,
                 "K*": g.k_star, "bins used by H": int(np.unique(kH).size)},
        "s_C ($/kg)": st.carbon_price, "norm_info": {k: float(v) for k, v in st.norm_info.items()},
        "term shares, p-weighted over H's types": term_shares(tt),
        "term shares, over the LP's admitted mass p x": term_shares(tt, x),
        "term shares, p-weighted over H's types with r~ > 0": term_shares(tt, (st.r_tilde > 0).astype(float)),
        "r~ quantiles over H's types, p-weighted (1, 10, 50, 90, 99 %)":
            [weighted_quantile(st.r_tilde[pop], st.p[pop], q) for q in (0.01, 0.1, 0.5, 0.9, 0.99)],
        "% of H's arrival rate on types with phi(L_k) at the floor": float(
            100 * st.p[pop & (st.phi_bar <= 0.99 * PHI_M_FLOOR * np.repeat(st.c_col[None, :], st.grid.k_star, 0).ravel()
                              * st.d_bar * cfg.time.tick_seconds / 3600)].sum() / st.p[pop].sum()),
        "tau (CPU, RAM)": np.asarray(st.tau).tolist(),
        "LP fill (CPU, RAM)": ((st.p * x) @ st.v / ds.capacity).tolist(),
        "LP optimum ($/h)": float(lp.optimum * 3600 / cfg.time.tick_seconds),
        "populated types": int(pop.sum()),
        "% populated types with r~ > 0": float(100 * (st.r_tilde[pop] > 0).mean()),
        "% of H's arrival rate on types with r~ > 0": float(100 * st.p[pop & (st.r_tilde > 0)].sum() / st.p[pop].sum()),
        "% of H jobs in a bin >= k* of their column (Phase 0 state)": float(100 * ((ks[cH] > 0) & (kH >= ks[cH])).mean()),
        "columns with k* = 0": int((ks == 0).sum()), "columns": int(ks.size),
        "columns with no H job (fallback)": int((st.src_dur == 2).sum()),
        "r_max (Tests 2-3 unit)": st.r_max,
    }


def run_numbers(m, res, cfg) -> dict:
    A, ph = res.arrivals, res.phases
    n = len(A)
    adm = [a for a in A if a["admit"]]
    reasons = Counter(a["reason"] for a in A if not a["admit"])
    # per column: median k* (over the column's arrivals) vs the median bin of the jobs
    by_col = {}
    for a in A:
        by_col.setdefault((a["q"], a["hw"]), []).append((a["k_star"], a["k_bin"]))
    cols = []
    for (q, hw), v in by_col.items():
        k = np.array(v)
        cols.append(dict(q=q, hw=hw, arrivals=len(v), median_kstar=float(np.median(k[:, 0])),
                         median_job_bin=float(np.median(k[:, 1])),
                         share_at_or_above_kstar=float(np.mean((k[:, 0] > 0) & (k[:, 1] >= k[:, 0])))))
    cols.sort(key=lambda c: -c["arrivals"])
    gap = np.array([c["median_job_bin"] - c["median_kstar"] for c in cols])
    w = np.array([c["arrivals"] for c in cols], float)
    # upper set: per phase state and column, the admitted bins (margin > 0) of the state
    up_viol = 0
    for st in m.states:
        ok = (st.r_tilde - st.v @ st.tau).reshape(st.grid.k_star, -1) > 0
        up_viol += int((ok[:-1] & ~ok[1:]).sum())
    # ... and in the run: within a phase and column, a bin with k_hat below a bin without
    seen = {}
    for a in A:
        seen.setdefault((a["m"], a["q"], a["hw"]), []).append((a["k_bin"], a["k_hat"]))
    run_viol = 0
    for v in seen.values():
        v = np.array(v)
        lo_yes = v[v[:, 1] == 1, 0]
        hi_no = v[v[:, 1] == 0, 0]
        if lo_yes.size and hi_no.size:
            run_viol += int((hi_no > lo_yes.min()).sum())
    pr = np.array([a["payment_rate"] for a in adm])
    b = np.array([a["v"] for a in adm])
    return {
        "arrivals": n, "admitted": len(adm), "admitted %": 100 * len(adm) / max(n, 1),
        "rejections": {k: int(reasons.get(k, 0)) for k in ("bin below k*", "below reserve", "thinning", "no space")},
        "phases": len(ph), "end reasons": res.summary["phase_end_reasons"],
        "phase table": [dict(m=p["m"], start_h=(p["tau_start"] - 1) * m._h, end_h=p["tau_end"] * m._h,
                             reason=p["reason"], arrivals=p["n_arrivals"], admitted=p["n_admitted"],
                             tau_cpu=p["tau_cpu"], tau_ram=p["tau_ram"], sigma=p["sigma"],
                             columns_kstar0=p["n_columns_kstar0"]) for p in ph],
        "columns with arrivals": len(cols),
        "median job bin - median k*, over columns (arrival-weighted quantiles 10/50/90)":
            [float(np.quantile(np.repeat(gap, w.astype(int)), q)) for q in (0.1, 0.5, 0.9)],
        "top columns by arrivals": cols[:12],
        "payments": {
            "mean p / b": float(np.mean(pr / b)) if adm else np.nan,
            "median p / b": float(np.median(pr / b)) if adm else np.nan,
            "share set by the reserve (r > L_k*)": float(np.mean([a["payment_by_reserve"] for a in adm])) if adm else np.nan,
            "share with k* = 1 (pay the floor of bin 1, i.e. the reserve)": float(np.mean([a["k_star"] == 1 for a in adm])) if adm else np.nan,
            "total payments ($)": float(np.nansum([a["payment"] for a in adm])),
            "total value of admitted jobs b d ($)": float(sum(a["value"] for a in adm)),
            "total electricity cost of admitted jobs ($)": float(sum(a["C_elec"] for a in adm)),
        },
        "realised reward split ($)": {"customer value": float(sum(m.J.cv[a["i"]] for a in adm)),
                                      "profit (phi d - C_elec)": float(sum(m.J.profit[a["i"]] for a in adm)),
                                      "carbon (s_C K)": float(-sum(m.J.carbon_usd[a["i"]] for a in adm)),
                                      "reward": res.summary["R_total"]},
        "IR violations": int(res.summary["n_ir_violations"]) + int(np.sum(~((pr > 0) & (pr <= b * (1 + 1e-12))))),
        "upper-set violations (phase states)": up_viol,
        "upper-set violations (run: a bin rejected above an admitted bin, same phase and column)": run_viol,
    }


def spot_check(m, res, seed: int, n_each: int = 200) -> dict:
    rng = np.random.default_rng(seed)
    A = res.arrivals
    ia = [k for k, a in enumerate(A) if a["admit"]]
    ir = [k for k, a in enumerate(A) if not a["admit"]]
    pick = list(rng.choice(ia, min(n_each, len(ia)), replace=False)) + list(rng.choice(ir, min(n_each, len(ir)), replace=False))
    S = m.stream
    bad, kstar_mismatch, checked = [], 0, 0
    for k in pick:
        a = A[k]
        st = m.states[a["m"] - 1]
        ts = st.type_space
        qi = int(np.searchsorted(ts.priorities, a["q"]))
        ks = int(threshold_kstar((st.r_tilde - st.v @ st.tau).reshape(ts.k_star, len(ts.priorities), ts.n_hw))[qi, a["hw"]])
        kstar_mismatch += int(ks != a["k_star"])
        _, _, _, p_true = admit_and_pay(a["v"], st.grid, ks, a["reserve"])
        for f in FACTORS:
            checked += 1
            ok_b, ok_r, _, p = admit_and_pay(a["v"] * f, st.grid, ks, a["reserve"])
            adm_f = ok_b and ok_r and a["U"] and a["fits"]          # thinning and the gate do not depend on the bid
            if a["admit"]:
                if adm_f and p < a["payment_rate"]:
                    bad.append(dict(i=a["i"], f=f, kind="admitted at a lower hourly payment", p=p, p_true=a["payment_rate"]))
                if f > 1 and not adm_f:
                    bad.append(dict(i=a["i"], f=f, kind="higher bid rejected"))
            else:
                if adm_f and p < a["v"]:
                    bad.append(dict(i=a["i"], f=f, kind="rejected job admitted below its true bid", p=p, b=a["v"]))
                if f < 1 and adm_f:
                    bad.append(dict(i=a["i"], f=f, kind="lower bid admitted"))
    return {"admitted checked": min(n_each, len(ia)), "rejected checked": min(n_each, len(ir)),
            "decisions re-run": checked, "logged k* != recomputed k*": kstar_mismatch, "violations": len(bad),
            "examples": bad[:10]}


def main(hours: float = 48.0, out: str = "reports/week33/truthful_batch.json") -> None:
    t0 = time.perf_counter()
    cfg = default_config()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = load_dataset(cfg)
        from dlent_rt.phase0 import run_phase0
        st = run_phase0(ds, cfg)
    print(f"Phase 0 done in {time.perf_counter() - t0:.0f} s", flush=True)
    p0 = phase0_numbers(ds, st, cfg)
    ds48 = first_hours(ds, hours)
    m = _Recording(ds48, st, cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = m.run()
    out_d = {"config": {"f": cfg.capacity.f, "seed": cfg.run.seed, "hours": hours, "capacity": ds.capacity.tolist()},
             "phase0": p0, "run": run_numbers(m, res, cfg), "spot check": spot_check(m, res, cfg.run.seed),
             "wall time (s)": time.perf_counter() - t0}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(out_d, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else float(x)))
    print(f"wrote {out} in {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main(*(float(a) if i == 0 else a for i, a in enumerate(sys.argv[1:])))

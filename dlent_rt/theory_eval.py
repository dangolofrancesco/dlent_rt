"""
Type / average ("theoretical") models over a capacity scan. No online model (DLENT-RT) and no job-level benchmark is
run here.

  * fluid OPT                 Benchmarks.fluid_opt: the full-stream LP-RS on the stream's own type rates and means
                              (hindsight), lambda_hat_ref x T.
  * fluid on feasible types   Benchmarks.fluid_opt_feasible_types: the same LP, types with A_type > C removed.
  * bucketed MILP             Benchmarks.job_opt_bucketed (reference buckets, true durations); run only when its
                              constraint matrix is small enough (max_milp_nonzeros), else reported as skipped.
  * static Phase 0            DLENT-RT's mechanism on the Phase 0 state, never updated (models.policy.ThresholdPolicy):
                              threshold k* per column from (r~, tau, v) of Phase 0, hourly reserve, payment
                              max(L_k*, r) x hours; no thinning; real gate.
  * no-learning model         the same, with the theoretical model's thinning p_U(W0, t) (same u_rand).
  * window LPs                the fluid LP on each window of the stream (its own type rates; reinit.build_window_state
                              without fallback), for the share of windows where capacity binds (tau > 0).

    from dlent_rt.theory_eval import evaluate_f, job_distributions
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .benchmarks import Benchmarks
from .data import _occupancy_stats, _OccupancyBasis, load_dataset
from .lp_oracle import solve_lp_rs
from .models.core.occupancy import Occupancy
from .models.core.reinit import build_window_state
from .models.core.error_params import make_error_params
from .models.policy import ThresholdPolicy, mechanism_summary, simulate
from .phase0 import (_compute_job_energy, _lookup_grid, assign_type_ids, run_phase0,
                     type_arrays)
from .reward import social_welfare
from .reward import carbon_kg, electricity_cost, type_reward

FULL = 0.95
QS = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999, 1.0)


# --------------------------------------------------------------------------- #
# LP helpers
# --------------------------------------------------------------------------- #
def unseen_type_estimates(st0, ds=None, cfg=None) -> dict:
    """
    r~ and v of EVERY type of the Phase 0 state and the dual rule admit = 1[r~ - tau . v > 0]. Since week 33 every
    type has an r~ in Phase 0 (value from the bin's lower edge; costs and volume from the column, with the
    hardware-profile fallback for columns with no H job, phase0.column_fallback), so this only reads the state. ds
    and cfg are kept for the old call signature.
    """
    r, v = np.asarray(st0.r_tilde, float), np.asarray(st0.v, float)
    return dict(r=r, v=v, seen=np.asarray(st0.p) > 0, admit=(r - v @ np.asarray(st0.tau, float)) > 0)


def lp_x(state, c, cfg, p=None) -> tuple[np.ndarray, object]:
    """Full admission vector x (n_types) of the LP-RS of `state` at capacity c (p overrides the rates)."""
    p = state.p if p is None else p
    lp = solve_lp_rs(p=p, r=state.r_tilde, v=state.v, c=c, presolve=cfg.lp.presolve)
    x = np.zeros(state.n_types)
    x[lp.support_idx] = lp.x
    return x, lp


def type_split(state, x, T, lam, p=None) -> dict:
    """Expected reward over T ticks of admitting fraction x of each type, split into its three terms [$]."""
    p = state.p if p is None else p
    w = p * x * T
    out = {"customer value": float(w @ (lam.lambda1 * state.v_bar)),
           "profit": float(w @ (lam.lambda2 * (state.phi_bar - state.C_elec_bar))),
           "carbon": float(-(w @ (lam.lambda3 * state.carbon_price * state.C_carbon_bar)))}
    out["reward"] = sum(out.values())
    return out


def admit_counts(x, p) -> dict:
    pop = np.asarray(p) > 0
    xs = np.asarray(x)[pop]
    return dict(fully=int((xs > 1 - 1e-9).sum()), partly=int(((xs > 1e-9) & (xs <= 1 - 1e-9)).sum()),
                rejected=int((xs <= 1e-9).sum()))


def lp_fill(state, x, c, p=None) -> np.ndarray:
    p = state.p if p is None else p
    return (p * x) @ state.v / np.asarray(c, float)


def occupancy_series(tick, D, A, admit):
    """Step function of the admitted jobs' occupancy: (times, levels, weights) from the first to the last arrival."""
    basis = _OccupancyBasis(arrival_step=np.asarray(tick, np.int64)[admit],
                            duration_steps=np.asarray(D, np.int64)[admit], A=np.asarray(A)[admit])
    st = _occupancy_stats(basis)
    t, lev, w = st["times"][:-1], st["levels"], st["weights"]
    keep = t <= tick[-1]
    return t[keep], lev[keep], w[keep]


def milp_nonzeros(B, source="reference") -> tuple[int, int]:
    """(candidate jobs, constraint nonzeros) of Benchmarks.job_opt_bucketed(source, 'true')."""
    S = B.S
    cand = np.flatnonzero(B.bucket_types(source)["r"] > 0)
    t, D = S.tick[cand], S.D[cand]
    last = np.searchsorted(t, t + D, side="left") - 1
    return int(cand.size), int((last - np.arange(cand.size) + 1).sum() * S.A.shape[1])


def window_taus(B, window_ticks: int) -> np.ndarray:
    """tau of the fluid LP on each full window of the stream (its own type rates, no fallback), (n_windows, |I|)."""
    S, cfg = B.S, B.cfg
    cfg_h = replace(cfg, phase_transition=replace(cfg.phase_transition, n_min=2))
    taus = []
    for k in range(int(B.T // window_ticks)):
        idx = np.flatnonzero((S.tick > k * window_ticks) & (S.tick <= (k + 1) * window_ticks))
        if idx.size < 2:
            continue
        st, _ = build_window_state(S, idx, idx, window_ticks, B.state0, B.ds, cfg_h)
        taus.append(np.asarray(st.tau, float))
    return np.array(taus)


# --------------------------------------------------------------------------- #
# One capacity value
# --------------------------------------------------------------------------- #
def static_run(B, st0, cfg, no_learning: bool = False, tau=None) -> dict:
    """
    The static Phase 0 policy on B's stream with the real gate: DLENT-RT's threshold, reserve and payment on the
    Phase 0 state, never updated (models.policy.ThresholdPolicy). no_learning: the no-learning model (+ the
    theoretical model's thinning p_U(W0, t), the same common random numbers u_rand). tau = 0: Greedy-Pos.
    Returns the realised split, the occupancy and the mechanism checks (payments, IR, upper set, reasons).
    """
    S, c = B.S, B.c
    thin = None
    if no_learning:
        ep = make_error_params(st0, B.ds, cfg)
        u_rand = np.random.default_rng(cfg.run.seed + cfg.online.thinning_seed_offset).random(S.n)
        thin = (ep, u_rand, int(B.ds.d_max_steps))
    pol = ThresholdPolicy(st0, S, tau=tau, thin=thin)
    sim = simulate(S, Occupancy(c, record=False), pol)
    mech = mechanism_summary(pol, sim, S)
    adm, terms = sim.admit, B.terms
    _, lev, w = occupancy_series(S.tick, S.D, S.A, adm)
    split = {"customer value": float(terms.cv[adm].sum()), "profit": float(terms.profit[adm].sum()),
             "carbon": float(-terms.carbon_usd[adm].sum())}
    split["reward"] = sum(split.values())
    split["social welfare"] = social_welfare(terms, adm)
    big = np.any(S.A > c[None, :], axis=1)
    return dict(split=split, full_time=float((w * (lev >= FULL * c[None, :]).any(1)).sum() / w.sum()),
                n_admitted=int(adm.sum()), rejected_space=float(sim.phys_reject.mean()), wanted=int(sim.want.sum()),
                oversize_wanted=int((sim.want & big).sum()),
                mean_occupancy_over_C=((lev * w[:, None]).sum(0) / w.sum() / c).tolist(),
                mechanism={k: v for k, v in mech.items() if k != "admit"}, admit=adm,
                profit_payments=float(mech["payments"] - S.C_elec[adm].sum()),
                stream_load_over_C=((S.A * S.duration_hours[:, None] * 3600 / cfg.time.tick_seconds).sum(0)
                                    / B.T / c).tolist(),
                occupancy=(S.tick, S.D, S.A, adm))


def evaluate_f(base_cfg, f: float, seed: int = 0, windows_h=(24, 6), max_milp_nonzeros: int = 0,
               milp_time_limit: float = 600.0, no_learning: bool = True) -> dict:
    """Every type / average model at capacity.f = f (the quota fixed point is at this f, so the jobs change with f).
    The full-stream bucketed MILP runs only if its constraint matrix has <= max_milp_nonzeros nonzeros (0 = never;
    the one-week MILP is window_models)."""
    cfg = replace(base_cfg, capacity=replace(base_cfg.capacity, f=f)).validate()
    ds = load_dataset(cfg)
    st0 = run_phase0(ds, cfg)
    B = Benchmarks(ds, st0, cfg)
    S, c, T, lam = B.S, B.c, B.T, cfg.objective
    em = cfg.energy_model
    res = dict(f=f, C=c.tolist(), C_cores=float(c[0] * em.machine_cores), C_gb=float(c[1] * em.machine_ram_gb),
               quota=list(ds.limit_report.quota), n_H=ds.H.n, n_stream=S.n, T_h=T / 3600)

    # Phase 0 LP (learned on H) and its static policy on the stream (one run per unseen-type option)
    x0, lp0 = lp_x(st0, c, cfg)
    big = np.any(S.A > c[None, :], axis=1)
    res["phase0_lp"] = dict(tau=np.asarray(lp0.tau).tolist(), counts=admit_counts(x0, st0.p),
                            fill=lp_fill(st0, x0, c).tolist(), expected=type_split(st0, x0, T, lam),
                            lam_star_T=float(lp0.optimum) * T)
    st = static_run(B, st0, cfg)
    res["static"] = {k: v for k, v in st.items() if k != "occupancy"}
    res["occupancy"] = st["occupancy"]
    if no_learning:
        nl = static_run(B, st0, cfg, no_learning=True)
        res["no_learning"] = {k: v for k, v in nl.items() if k != "occupancy"}

    # fluid OPT and fluid on feasible types (reference = hindsight state of the whole stream)
    ref = B.reference_state
    xr, lpr = lp_x(ref, c, cfg)
    fluid = type_split(ref, xr, T, lam)
    assert np.isclose(fluid["reward"], B.fluid_opt(), rtol=1e-6, atol=1e-6)
    feas = ~np.any(ref.A_type > c[None, :], axis=1)
    pf = ref.p * feas
    xf, lpf = lp_x(ref, c, cfg, p=pf)
    res["fluid"] = dict(tau=np.asarray(lpr.tau).tolist(), counts=admit_counts(xr, ref.p),
                        fill=lp_fill(ref, xr, c).tolist(), split=fluid)
    res["fluid_feasible"] = dict(tau=np.asarray(lpf.tau).tolist(), counts=admit_counts(xf, pf),
                                 fill=lp_fill(ref, xf, c, p=pf).tolist(), split=type_split(ref, xf, T, lam, p=pf),
                                 n_types_removed=int((~feas & (ref.p > 0)).sum()))
    # bucketed MILP: only when small enough
    n_cand, nnz = milp_nonzeros(B, "reference")
    res["bucketed"] = dict(n_candidates=n_cand, nonzeros=nnz, run=nnz <= max_milp_nonzeros)
    if res["bucketed"]["run"]:
        o = B.job_opt_bucketed("reference", "true", time_limit=milp_time_limit)
        x, terms = o["x"], B.terms
        res["bucketed"].update(split={"customer value": float(terms.cv[x].sum()), "profit": float(terms.profit[x].sum()),
                                      "carbon": float(-terms.carbon_usd[x].sum()), "reward": o["R_real"]},
                               status=o["status"], gap=o["gap"], runtime_s=o["runtime_s"])
    res["oversize"] = dict(stream=int(big.sum()), H=int(np.any(ds.H.A > c[None, :], axis=1).sum()),
                           cpu_hours_share=float((S.A[big, 0] * S.duration_hours[big]).sum()
                                                 / (S.A[:, 0] * S.duration_hours).sum()))
    # window LPs
    res["windows"] = {h: window_taus(B, int(h * 3600 / cfg.time.tick_seconds)) for h in windows_h}
    return res


F_SCAN = (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6)


def tau_positive(r) -> bool:
    """Condition (a) of the f rule: tau > 0 (CPU or RAM) in BOTH the Phase 0 LP and fluid OPT."""
    return bool(np.any(np.asarray(r["phase0_lp"]["tau"]) > 0) and np.any(np.asarray(r["fluid"]["tau"]) > 0))


def band_distance(r, band=(0.20, 0.30)) -> float:
    x = r["static"]["full_time"]
    return max(band[0] - x, 0.0, x - band[1])


def select_f(results, band=(0.20, 0.30), target=0.25):
    """
    The f rule (D3), in priority order:
      (a) REQUIRED: tau > 0 in both the Phase 0 LP and fluid OPT (CPU or RAM);
      (b) among the f that pass (a): the static policy's '% time >= 95% full' closest to the band (0 inside), ties
          to the target 25%.
    """
    ok = [r for r in results if tau_positive(r)]
    if not ok:
        raise ValueError("f rule: no f of the scan has tau > 0 in both the Phase 0 LP and fluid OPT")
    return min(ok, key=lambda r: (band_distance(r, band), abs(r["static"]["full_time"] - target)))["f"]


# --------------------------------------------------------------------------- #
# Job distributions (before / after the 24 h cap and the quota)
# --------------------------------------------------------------------------- #
def job_distributions(cfg) -> dict:
    """Arrays and tables for the 300k jobs and the jobs of H, before and after the cap and the quota."""
    from .config import resolve_path
    col = cfg.data.columns
    df = pd.read_csv(resolve_path(cfg.data.batch_csv))
    t = pd.to_datetime(df[col.datetime]).dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
    cpu, ram = df[col.requested_cpu].to_numpy(float), df[col.requested_ram].to_numpy(float)
    span = df[col.duration_hours].to_numpy(float)
    eff = np.divide(df[col.cpu_hours_real].to_numpy(float), cpu, out=span.copy(), where=cpu > 0)
    ds = load_dataset(cfg)
    Q = np.array(ds.limit_report.quota, float)
    cap = cfg.time.d_cap_hours
    in_H = t < t.min() + np.timedelta64(int(round(cfg.data.h_days * 86400)), "s")
    kept = (cpu <= Q[0]) & (ram <= Q[1])
    d_cap = np.minimum(np.maximum(eff, cfg.time.duration_floor_hours), cap)
    sets = {"all 300k": np.ones(len(df), bool), "H (first 7 days)": in_H}
    rows = []
    for sname, m in sets.items():
        for stage, mm, d in (("before cap + quota", m, eff), ("after cap + quota", m & kept, d_cap)):
            w = cpu[mm] * d[mm]
            for qname, x in (("CPU request (machines)", cpu[mm]), ("RAM request (machines)", ram[mm]),
                             ("duration, effective (h)", eff[mm]), ("duration after the 24 h cap (h)", d_cap[mm]),
                             ("work r*d (machine-h)", cpu[mm] * d[mm])):
                if stage == "before cap + quota" and qname.startswith("duration after"):
                    continue
                rows.append({"set": sname, "stage": stage, "quantity": qname, "count": int(mm.sum()),
                             "mean": float(x.mean()), **{f"p{100 * q:g}" if q < 1 else "max": float(np.quantile(x, q)) for q in QS},
                             **{f"w p{100 * q:g}" if q < 1 else "w max": _wq(x, w, q) for q in QS}})
    dist = pd.DataFrame(rows)
    imp = []
    for sname, m in sets.items():
        tot = (cpu[m] * eff[m]).sum()
        cut = m & (eff > cap)
        rej = m & ~kept
        imp.append({"set": sname, "jobs": int(m.sum()), "CPU-hours (effective)": float(tot),
                    "jobs cut by the 24 h cap": int(cut.sum()), "share of jobs cut": float(cut.sum() / m.sum()),
                    "CPU-hours cut by the cap": float((cpu[cut] * (eff[cut] - cap)).sum() / tot),
                    "jobs removed by the quota": int(rej.sum()), "share of jobs removed": float(rej.sum() / m.sum()),
                    "CPU-hours removed by the quota": float((cpu[rej] * eff[rej]).sum() / tot),
                    "CPU-hours left after both": float((cpu[m & kept] * d_cap[m & kept]).sum() / tot)})
    return dict(dist=dist, impact=pd.DataFrame(imp), cpu=cpu, ram=ram, eff=eff, d_cap=d_cap, in_H=in_H, kept=kept,
                Q=Q, cap=cap)


def _wq(x, w, q):
    o = np.argsort(x, kind="stable")
    cw = np.cumsum(np.asarray(w, float)[o]) / max(np.sum(w), 1e-300)
    return float(np.asarray(x)[o][min(np.searchsorted(cw, q - 1e-12), len(o) - 1)])


# --------------------------------------------------------------------------- #
# Figures (matplotlib; used by notebooks/01_model_check)
# --------------------------------------------------------------------------- #
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e5e4e0"


def _style(ax, xlabel, ylabel=None, log=True):
    if log:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _cdf(ax, x, w, label, color, ls="-"):
    m = x > 0
    o = np.argsort(x[m])
    xs, ws = x[m][o], np.asarray(w, float)[m][o]
    ax.step(xs, np.cumsum(ws) / ws.sum(), where="post", color=color, ls=ls, lw=1.6, label=label)


def fig_cdf_jobs_work(J, which, ax):
    """CDF of CPU request / RAM request / duration over all jobs: share of jobs and share of CPU-hours."""
    x, line, lab = {"cpu": (J["cpu"], J["Q"][0], f"quota Q = {J['Q'][0]:.3f}"),
                    "ram": (J["ram"], J["Q"][1], f"quota Q = {J['Q'][1]:.3f}"),
                    "duration": (J["eff"], J["cap"], f"cap {J['cap']:g} h")}[which]
    _cdf(ax, x, np.ones_like(x), "share of jobs", BLUE)
    _cdf(ax, x, J["cpu"] * J["eff"], "share of CPU-hours", ORANGE, "--")
    ax.axvline(line, color=INK2, ls=":", lw=1)
    ax.text(line, 0.5, f" {lab}", fontsize=8, color=INK2)
    unit = "hours" if which == "duration" else "machines"
    name = {"cpu": "CPU request", "ram": "RAM request", "duration": "effective duration"}[which]
    ax.set_title(f"{name}, all {len(x):,} jobs", loc="left", fontsize=10, color=INK)
    _style(ax, f"{name} ({unit}, log scale)", "cumulative share")
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def fig_hist2d(J, ax):
    m = (J["cpu"] > 0) & (J["eff"] > 0)
    x, y = np.log10(J["cpu"][m]), np.log10(J["eff"][m])
    h = ax.hist2d(x, y, bins=60, cmap="Blues", cmin=1, norm=__import__("matplotlib.colors", fromlist=["LogNorm"]).LogNorm())
    ax.axvline(np.log10(J["Q"][0]), color=ORANGE, ls="--", lw=1.2)
    ax.axhline(np.log10(J["cap"]), color=ORANGE, ls="--", lw=1.2)
    ax.text(np.log10(J["Q"][0]), y.min(), f" Q = {J['Q'][0]:.2f}", color=ORANGE, fontsize=8)
    ax.text(x.min(), np.log10(J["cap"]), " 24 h", color=ORANGE, fontsize=8, va="bottom")
    ax.set_xlabel("CPU request (log10 machines)")
    ax.set_ylabel("effective duration (log10 hours)")
    ax.set_title("Jobs by CPU request and duration (count per cell, log colour)", loc="left", fontsize=10, color=INK)
    return h


def fig_H_vs_month(J, axs):
    for ax, (name, x, unit) in zip(axs, (("CPU request", J["cpu"], "machines"), ("RAM request", J["ram"], "machines"),
                                         ("effective duration", J["eff"], "hours"))):
        _cdf(ax, x, np.ones_like(x), "whole month", BLUE)
        _cdf(ax, x[J["in_H"]], np.ones(J["in_H"].sum()), "H (first 7 days)", ORANGE, "--")
        ax.set_title(name, loc="left", fontsize=10, color=INK)
        _style(ax, f"{unit} (log scale)")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    axs[0].set_ylabel("share of jobs at or below x", color=INK)


MODELS = (("fluid", "fluid OPT (hindsight)"), ("fluid_feasible", "fluid on feasible types"),
          ("phase0_expected", "Phase 0 LP, expected"), ("static", "static Phase 0, realised"))


def model_reward(r, key) -> float:
    if key == "phase0_expected":
        return r["phase0_lp"]["expected"]["reward"]
    return r[key]["split"]["reward"]


def fig_tau_vs_f(results, axs):
    fs = [r["f"] for r in results]
    for ax, i, nm in zip(axs, (0, 1), ("CPU", "RAM")):
        for key, lab, col, ls in (("phase0_lp", "Phase 0 LP", BLUE, "-"), ("fluid", "fluid OPT", ORANGE, "--"),
                                  ("fluid_feasible", "fluid on feasible types", INK2, ":")):
            ax.plot(fs, [r[key]["tau"][i] for r in results], color=col, ls=ls, marker="o", ms=4, lw=1.6, label=lab)
        ax.set_title(f"tau ({nm})", loc="left", fontsize=10, color=INK)
        _style(ax, "f (cluster = f x mean demand of H), log scale", "tau ($ per machine-tick)")
    axs[0].legend(frameon=False, fontsize=8)


def fig_reward_vs_f(results, ax):
    fs = [r["f"] for r in results]
    for (key, lab), col, ls in zip(MODELS, (ORANGE, INK2, BLUE, BLUE), ("--", ":", "-", "-.")):
        ax.plot(fs, [model_reward(r, key) for r in results], color=col, ls=ls, marker="o", ms=4, lw=1.6, label=lab)
    ax.set_title("Reward over the stream ($) by model", loc="left", fontsize=10, color=INK)
    _style(ax, "f, log scale", "reward ($ over the stream)")
    ax.legend(frameon=False, fontsize=8)


def fig_occupancy(r, axs, tick_seconds=1.0):
    st = r["static"]
    tick, D, A, adm = r["occupancy"]
    t, lev, w = occupancy_series(tick, D, A, adm)
    days = (t - t[0]) * tick_seconds / 86400
    for ax, i, nm in zip(axs, (0, 1), ("CPU", "RAM")):
        ax.step(days, lev[:, i], where="post", color=BLUE, lw=0.7)
        ax.axhline(r["C"][i], color=ORANGE, ls="--", lw=1.2)
        ax.text(days[-1], r["C"][i], f" C = {r['C'][i]:.2f}", color=ORANGE, fontsize=8, va="bottom", ha="right")
        ax.set_title(f"{nm}: static Phase 0 policy, f = {r['f']:g} "
                     f"(>= 95% full {100 * st['full_time']:.1f}% of the time)", loc="left", fontsize=10, color=INK)
        _style(ax, "days since the start of the stream", f"{nm} (machines)", log=False)


def fig_tau_windows(r, axs, hours=24):
    taus = r["windows"][hours]
    for ax, i, nm in zip(axs, (0, 1), ("CPU", "RAM")):
        ax.bar(np.arange(len(taus)), taus[:, i], color=BLUE, width=0.8)
        ax.set_title(f"{nm}: tau of the fluid LP per {hours} h window, f = {r['f']:g} "
                     f"(tau > 0 in {100 * (taus[:, i] > 0).mean():.0f}% of windows)", loc="left", fontsize=10, color=INK)
        _style(ax, f"window ({hours} h) since the start of the stream", "tau", log=False)


# --------------------------------------------------------------------------- #
# H vs stream, and the models on one window of the stream (week 32, theory_models_2)
# --------------------------------------------------------------------------- #
def h_vs_stream(B, st0, cfg) -> dict:
    """
    Per Phase 0 type (Phase 0's grid for both): arrival rate (per hour) and mean r*d (CPU machine-hours) in H and in
    the stream. Also the decomposition of 'Phase 0 LP expected' vs 'fluid OPT': the LP on the stream's statistics
    (Phase 0's grid), and Phase 0's x evaluated on those statistics.
    """
    S, T, c, lam = B.S, B.T, B.c, cfg.objective
    per_h = 3600.0 / cfg.time.tick_seconds
    H = B.ds.H
    kH = st0.grid.assign(st0.bids.v_rate)
    jH = assign_type_ids(st0.type_space, np.asarray(kH), H.priority, H.hw_idx)
    jS = assign_type_ids(st0.type_space, np.asarray(st0.grid.assign(S.v)), S.priority, S.hw_idx)
    n = st0.n_types
    rdH, rdS = H.A[:, 0] * H.duration_hours, S.A[:, 0] * S.duration_hours

    def per_type(j, rd, span):
        N = np.bincount(j, minlength=n)
        return N / span * per_h, np.divide(np.bincount(j, weights=rd, minlength=n), N, out=np.zeros(n), where=N > 0)
    lamH, mH = per_type(jH, rdH, B.ds.H_span_steps)
    lamS, mS = per_type(jS, rdS, T)
    both = (lamH > 0) & (lamS > 0)
    w = lamH[both]
    tab = pd.DataFrame({"type": np.flatnonzero(both), "lambda H /h": lamH[both], "lambda stream /h": lamS[both],
                        "mean r*d H": mH[both], "mean r*d stream": mS[both]})
    tab["load H"] = tab["lambda H /h"] * tab["mean r*d H"]
    tab["load stream"] = tab["lambda stream /h"] * tab["mean r*d stream"]
    tab["load gap"] = tab["load stream"] - tab["load H"]
    _, prio, _ = type_arrays(st0.type_space, st0.grid)
    ts = st0.type_space
    ids = tab["type"].to_numpy()
    tab["bid bin"] = ids // (len(ts.priorities) * ts.n_hw) + 1
    tab["priority"] = prio[ids]
    tab["hw profile"] = ids % ts.n_hw
    top = tab.reindex(tab["load gap"].abs().sort_values(ascending=False).index).head(10)
    # decomposition
    cfg_h = replace(cfg, phase_transition=replace(cfg.phase_transition, n_min=2))
    idx = np.arange(S.n)
    sS, _ = build_window_state(S, idx, idx, T, st0, B.ds, cfg_h)
    x0, lp0 = lp_x(st0, c, cfg)
    xS, lpS = lp_x(sS, c, cfg)
    return dict(
        n_types_H=int((lamH > 0).sum()), n_types_stream=int((lamS > 0).sum()), n_types_both=int(both.sum()),
        lambda_ratio=float(np.average(lamS[both] / lamH[both], weights=w)),
        rd_ratio=float(np.average(mS[both & (mH > 0)] / mH[both & (mH > 0)], weights=lamH[both & (mH > 0)])),
        load_H=float((lamH * mH).sum()), load_stream=float((lamS * mS).sum()),
        load_stream_unseen=float((lamS * mS)[lamH == 0].sum()),
        top=top, phase0_expected=type_split(st0, x0, T, lam)["reward"],
        stream_stats_phase0_grid_opt=type_split(sS, xS, T, lam)["reward"],
        phase0_x_on_stream_stats=type_split(sS, x0, T, lam)["reward"],
        fluid_opt=B.fluid_opt(),
        r_tilde_ratio=float(np.average(sS.r_tilde[both] / np.where(st0.r_tilde[both] != 0, st0.r_tilde[both], np.nan),
                                       weights=w)) if both.any() else float("nan"))


def sub_stream(S, mask):
    """The StreamData of the arrivals where mask is True (same ticks)."""
    from dataclasses import fields as dc_fields
    kw = {f.name: getattr(S, f.name)[mask] for f in dc_fields(S) if f.name != "n"}
    return type(S)(n=int(mask.sum()), **kw)


def window_models(cfg, days=(0.0, 7.0), milp_time_limit: float = 3600.0, tail_hours: float | None = 1.0) -> dict:
    """
    The models on the stream arrivals of days [days[0], days[1]) after H, T = the window length (D6):
      fluid (window)          the fluid LP with each type's volume counted only INSIDE the window
                              (mean of min(D_j, window end - t_j) over its jobs);
      fluid feasible (window) the same with the types whose request exceeds C removed;
      fluid (whole volume)    the usual fluid LP (each type's whole volume A x d_bar), for comparison;
      bucketed MILP           reference buckets, true durations, its own HiGHS instance and time limit; capacity
                              checks also every tail_hours after the last arrival.
    """
    import time
    ds = load_dataset(cfg)
    st0 = run_phase0(ds, cfg)
    S = Benchmarks(ds, st0, cfg).S
    t0, t1 = days[0] * 86400 / cfg.time.tick_seconds, days[1] * 86400 / cfg.time.tick_seconds
    m = (S.tick > t0) & (S.tick <= t1)
    Sw = sub_stream(S, m)
    Bw = Benchmarks(ds, st0, cfg, stream=Sw)
    Bw.T = int(t1 - t0)
    t = time.perf_counter()
    fluid_whole = Bw.fluid_opt()
    ref = Bw.reference_state
    j = Bw.bucket_types("reference")["type"]
    inside = np.minimum(Sw.D, Bw.T - (Sw.tick - t0) + 1).astype(float)          # ticks of each job inside the window
    N = np.bincount(j, minlength=ref.n_types)
    d_in = np.divide(np.bincount(j, weights=inside, minlength=ref.n_types), N, out=np.zeros(ref.n_types),
                     where=N > 0)
    v_in = ref.A_type * d_in[:, None]
    feas = ~np.any(ref.A_type > Bw.c[None, :], axis=1)
    lp_w = solve_lp_rs(p=ref.p, r=ref.r_tilde, v=v_in, c=Bw.c, presolve=cfg.lp.presolve)
    lp_wf = solve_lp_rs(p=ref.p * feas, r=ref.r_tilde, v=v_in, c=Bw.c, presolve=cfg.lp.presolve)
    t_lp = time.perf_counter() - t
    n_cand, nnz = milp_nonzeros(Bw, "reference")
    tail = None if tail_hours is None else int(round(tail_hours * 3600 / cfg.time.tick_seconds))
    o = Bw.job_opt_bucketed("reference", "true", time_limit=milp_time_limit, tail_step=tail)
    return dict(f=cfg.capacity.f, days=days, n_jobs=int(Sw.n), C=Bw.c.tolist(),
                fluid=float(lp_w.optimum) * Bw.T, fluid_feasible=float(lp_wf.optimum) * Bw.T,
                fluid_whole_volume=fluid_whole, tail_hours=tail_hours, n_tail_checks=o["n_tail_checks"],
                bucketed_believed=o["R_believed"], bucketed_realised=o["R_real"], bucketed_lp_believed=o["R_lp_believed"],
                bucketed_dual_bound=o["dual_bound"], status=o["status"], gap=o["gap"], milp_runtime_s=o["runtime_s"],
                milp_lp_runtime_s=o["lp_runtime_s"], lp_runtime_s=t_lp, n_candidates=n_cand, nonzeros=o["nonzeros"],
                feasible_true=o["feasible_true"], milp_time_limit=milp_time_limit)

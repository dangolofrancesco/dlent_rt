"""
The f rule (on H only) and the type / average ("theoretical") models at one f. No online model (DLENT-RT) is run here.

  f rule (D3)                 evaluate_f_H / f_rule_point / select_f: Phase 0 on H alone (data.load_history) and the
                              replay of H's jobs through the static policy; no stream job is read.
  discretization cost (D6)    window_discretization: MILP_true vs MILP_bin (benchmarks.discretization_milps) on the
                              first days of the stream.

At one f (evaluate_f), on the stream:
  * fluid OPT                 Benchmarks.fluid_opt: the full-stream LP-RS on the stream's own type rates and means
                              (hindsight), lambda_hat_ref x T. An upper bound.
  * fluid on feasible types   Benchmarks.fluid_opt_feasible_types: the same LP, types with A_type > C removed.
  * static Phase 0            DLENT-RT's mechanism on the Phase 0 state, never updated (models.policy.ThresholdPolicy):
                              threshold k* per column from (r~, tau, v) of Phase 0, hourly reserve, payment
                              max(L_k*, r) x hours; no thinning; real gate.
  * no-learning model         the same, with the theoretical model's thinning p_U(W0, t) (same u_rand).
  * window LPs                the fluid LP on each window of the stream (its own type rates; reinit.build_window_state
                              without fallback), for the share of windows where capacity binds (tau > 0).

    from dlent_rt.theory_eval import evaluate_f_H, select_f, evaluate_f, window_discretization
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .benchmarks import Benchmarks, discretization_milps
from .data import _occupancy_stats, _OccupancyBasis, load_dataset, load_history
from .lp_oracle import solve_lp_rs
from .models.core.occupancy import Occupancy
from .models.core.reinit import build_window_state
from .models.core.stream import build_history_stream, build_stream
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


def evaluate_f(base_cfg, f: float, seed: int = 0, windows_h=(24, 6), no_learning: bool = True) -> dict:
    """Every type / average model on the stream at capacity.f = f (the quota fixed point is at this f, so the jobs
    change with f). Not used by the f rule (that is evaluate_f_H, on H only)."""
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
    res["oversize"] = dict(stream=int(big.sum()), H=int(np.any(ds.H.A > c[None, :], axis=1).sum()),
                           cpu_hours_share=float((S.A[big, 0] * S.duration_hours[big]).sum()
                                                 / (S.A[:, 0] * S.duration_hours).sum()))
    # window LPs
    res["windows"] = {h: window_taus(B, int(h * 3600 / cfg.time.tick_seconds)) for h in windows_h}
    return res


# --------------------------------------------------------------------------- #
# The f rule (D3), on H only
# --------------------------------------------------------------------------- #
F_SCAN = (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6)


def f_rule_point(ds, cfg) -> dict:
    """
    One f of the rule, from H alone: Phase 0 on ds (its H), then H's jobs replayed through the static ThresholdPolicy
    of that Phase 0 state with capacity C (the real gate). Reads only ds.H, ds.capacity and the grid file.
    Returns Phase 0's numbers (U0, sigma, grid, D_bar, C, Q, s_C, r_max, tau, LP fill) and the replay's
    '% of H's time with the cluster >= 95% full' (full_time).
    """
    st0 = run_phase0(ds, cfg)
    SH = build_history_stream(ds, st0, cfg)
    c = np.asarray(ds.capacity, float)
    pol = ThresholdPolicy(st0, SH)
    sim = simulate(SH, Occupancy(c, record=False), pol)
    mech = mechanism_summary(pol, sim, SH)
    _, lev, w = occupancy_series(SH.tick, SH.D, SH.A, sim.admit)
    x0, _ = lp_x(st0, c, cfg)
    g = st0.grid
    return dict(f=float(cfg.capacity.f), C=c.tolist(), D_bar=np.asarray(ds.capacity_diag["D_bar"]).tolist(),
                quota=list(ds.limit_report.quota), n_H=int(ds.H.n), U0=float(st0.u0), sigma=float(st0.myerson.sigma),
                v_min=float(g.bnd_v[0]), v_max=float(g.bnd_v[-1]), rho=float(g.ratio), s_C=float(st0.carbon_price),
                r_max=float(st0.r_max), tau=np.asarray(st0.tau, float).tolist(), lp_fill=lp_fill(st0, x0, c).tolist(),
                full_time=float((w * (lev >= FULL * c[None, :]).any(1)).sum() / w.sum()),
                mean_occupancy_over_C=((lev * w[:, None]).sum(0) / w.sum() / c).tolist(),
                n_admitted=int(sim.admit.sum()), rejections=mech["rejections"])


def evaluate_f_H(base_cfg, f: float) -> dict:
    """f_rule_point at capacity.f = f on the H-only Dataset (data.load_history: no stream job is read)."""
    cfg = replace(base_cfg, capacity=replace(base_cfg.capacity, f=f)).validate()
    return f_rule_point(load_history(cfg), cfg)


def tau_positive(r) -> bool:
    """Condition (a) of the f rule: tau > 0 (CPU or RAM) in the Phase 0 LP (on H)."""
    return bool(np.any(np.asarray(r["tau"]) > 0))


def band_distance(r, band=(0.20, 0.30)) -> float:
    x = r["full_time"]
    return max(band[0] - x, 0.0, x - band[1])


def select_f(results, band=(0.20, 0.30), target=0.25):
    """
    The f rule (D3), on H only (results = evaluate_f_H over the scan), in priority order:
      (a) REQUIRED: tau > 0 (CPU or RAM) in the Phase 0 LP;
      (b) among the f that pass (a): the '% of H's time with the cluster >= 95% full' of H's replay through the static
          policy closest to the band (0 inside), ties to the target 25%.
    """
    ok = [r for r in results if tau_positive(r)]
    if not ok:
        raise ValueError("f rule: no f of the scan has tau > 0 in the Phase 0 LP")
    return min(ok, key=lambda r: (band_distance(r, band), abs(r["full_time"] - target)))["f"]


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


def window_discretization(cfg, days=(0.0, 7.0), time_limit: float = 3600.0) -> dict:
    """
    The discretization cost of the grid on the stream arrivals of days [days[0], days[1]) after H (D6):
    benchmarks.discretization_milps (MILP_true vs MILP_bin, true durations, capacity checked at every arrival), with
    Phase 0's grid, bid model and s_C, lambda = 1. Each MILP on its own HiGHS instance with time_limit seconds.
    """
    ds = load_dataset(cfg)
    st0 = run_phase0(ds, cfg)
    S = build_stream(ds, cfg)
    t0, t1 = days[0] * 86400 / cfg.time.tick_seconds, days[1] * 86400 / cfg.time.tick_seconds
    Sw = sub_stream(S, (S.tick > t0) & (S.tick <= t1))
    lam = replace(cfg.objective, lambda1=1.0, lambda2=1.0, lambda3=1.0)
    out = discretization_milps(Sw, st0, lam, ds.capacity, time_limit=time_limit)
    return dict(f=cfg.capacity.f, seed=cfg.run.seed, days=tuple(days), C=np.asarray(ds.capacity).tolist(),
                **{k: v for k, v in out.items() if k not in ("R_true", "R_bin")})

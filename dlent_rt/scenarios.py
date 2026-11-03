"""
Non-stationary bids (week 34; reports/week34/scenarios_expectations.md). A scenario changes the STREAM's markups once,
at t* (scenario.t_star_hours after the stream start), after the bids are generated:

  s1_bids_fall    m -> 0.5 m            (mu1 = mu0 + ln 0.5, sigma1 = sigma0)
  s2_bids_rise    m -> 1.75 m           (mu1 = mu0 + ln 1.75, sigma1 = sigma0)
  s6_bids_spread  ln m rescaled around its mean to sigma1 = 0.5, mean set so that E[m] = 1 (mu1 = -sigma1^2/2)

for the jobs that arrive at or after t*; then b = c m. Nothing else changes: not c, the arrivals, the requests, the
durations, the prices, the carbon, nor H (Phase 0). The transform is deterministic, so every model and benchmark sees
the same changed stream (models.core.stream.build_stream applies it) with the same random numbers.

The second half of the module turns run results into the scenario diagnostics: detection delay, reward before /
after t*, % of fluid OPT after t*, the markup model per phase against the truth, expected vs realised profit before /
after t*, the share of post-change bids in bin 1.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .bid import MyersonModel

PRESETS = {
    "none": None,
    "s1_bids_fall": dict(kind="scale", k=0.5),
    "s2_bids_rise": dict(kind="scale", k=1.75),
    "s6_bids_spread": dict(kind="spread", sigma1=0.5),
}
ALIASES = {"s1": "s1_bids_fall", "s2": "s2_bids_rise", "s6": "s6_bids_spread"}


def resolve(name: str) -> str:
    n = ALIASES.get(name, name)
    if n not in PRESETS:
        raise ValueError(f"unknown scenario {name!r}: choose one of {sorted(PRESETS) + sorted(ALIASES)}")
    return n


def t_star_step(cfg) -> int:
    """t* in stream steps (arrival_step = tick - 1, 0 at the stream start)."""
    return int(round(cfg.scenario.t_star_hours * 3600.0 / cfg.time.tick_seconds))


def post_change(tick, cfg) -> np.ndarray:
    """True for the arrivals at or after t* (tick is 1-based: arrival_step = tick - 1)."""
    return np.asarray(tick) - 1 >= t_star_step(cfg)


def truth(cfg) -> dict:
    """The true markup parameters before (mu0, sigma0) and after (mu1, sigma1) t*."""
    s0 = float(cfg.bid.sigma)
    mu0 = -0.5 * s0 ** 2
    p = PRESETS[resolve(cfg.scenario.name)]
    if p is None:
        return dict(mu0=mu0, sigma0=s0, mu1=mu0, sigma1=s0)
    if p["kind"] == "scale":
        return dict(mu0=mu0, sigma0=s0, mu1=mu0 + float(np.log(p["k"])), sigma1=s0)
    s1 = float(p["sigma1"])
    return dict(mu0=mu0, sigma0=s0, mu1=-0.5 * s1 ** 2, sigma1=s1)


def transform_log_markups(lm, cfg) -> np.ndarray:
    """ln m after the change, from ln m before it (the generator's ln m ~ N(mu0, sigma0^2))."""
    p = PRESETS[resolve(cfg.scenario.name)]
    lm = np.asarray(lm, float)
    if p is None:
        return lm
    if p["kind"] == "scale":
        return lm + np.log(p["k"])
    tr = truth(cfg)
    return tr["mu1"] + (tr["sigma1"] / tr["sigma0"]) * (lm - tr["mu0"])


def apply_scenario(S, cfg):
    """The stream with the scenario's markups for the arrivals at or after t* (b = c m; phi_gen under the true
    post-change distribution). scenario 'none': S itself."""
    if PRESETS[resolve(cfg.scenario.name)] is None:
        return S
    post = post_change(S.tick, cfg)
    c = np.asarray(S.cost_base, float)
    lm = np.log(np.asarray(S.v, float) / c)
    v = np.asarray(S.v, float).copy()
    v[post] = c[post] * np.exp(transform_log_markups(lm[post], cfg))
    tr = truth(cfg)
    phi = np.asarray(S.phi_gen, float).copy()
    phi[post] = MyersonModel(mu=tr["mu1"], sigma=tr["sigma1"]).phi_job(v[post], c[post])
    return replace(S, v=v, phi_gen=phi)


def scoring_model_post(m0: MyersonModel, cfg) -> MyersonModel:
    """The markup model that scores the realised rewards of the post-change jobs: Phase 0's model m0 (the one that
    scores every pre-change job) moved by the scenario's transform. S1 / S2: mu + ln k; S6: ln m rescaled around the
    generator's mean, so mu' = mu1 + r (mu0_hat - mu0), sigma' = r sigma0_hat, r = sigma1 / sigma0. 'none': m0."""
    p = PRESETS[resolve(cfg.scenario.name)]
    if p is None:
        return m0
    if p["kind"] == "scale":
        return MyersonModel(mu=m0.mu + float(np.log(p["k"])), sigma=m0.sigma)
    tr = truth(cfg)
    r = tr["sigma1"] / tr["sigma0"]
    return MyersonModel(mu=tr["mu1"] + r * (m0.mu - tr["mu0"]), sigma=r * m0.sigma)


def scored_job_terms(S, model0, s_C: float, lam, cfg):
    """reward.job_terms of every stream job, the realised reward of the results: pre-change jobs with Phase 0's markup
    model, post-change jobs with scoring_model_post (the true post-change distribution on the same scale). With
    scenario 'none' this is exactly reward.job_terms with model0."""
    from .reward import JobTerms, job_terms
    args = (S.v, S.cost_base, S.duration_hours, S.C_elec, S.C_carbon)
    pre = job_terms(*args, model0, s_C, lam)
    if PRESETS[resolve(cfg.scenario.name)] is None:
        return pre
    post = job_terms(*args, scoring_model_post(model0, cfg), s_C, lam)
    w = post_change(S.tick, cfg)
    return JobTerms(**{f: np.where(w, getattr(post, f), getattr(pre, f)) for f in JobTerms.__dataclass_fields__})


# --------------------------------------------------------------------------- #
# Diagnostics on run results
# --------------------------------------------------------------------------- #
def detection(phases, cfg) -> dict:
    """Hours from t* to the first phase end by Test 2 or Test 3, and to the first phase end of any kind (not the
    horizon), with the reason. phases: the run's phase table (tau_end in ticks, reason). NaN if none."""
    ts, h = t_star_step(cfg), cfg.time.tick_seconds / 3600.0
    after = phases[(phases.tau_end >= ts) & (phases.reason != "horizon")]
    t23 = after[after.reason.isin(["test2", "test3"])]
    first = after.iloc[0] if len(after) else None
    return dict(delay_test23_h=float((t23.tau_end.iloc[0] - ts) * h) if len(t23) else np.nan,
                test23=str(t23.reason.iloc[0]) if len(t23) else "",
                delay_any_h=float((first.tau_end - ts) * h) if first is not None else np.nan,
                first_reason=str(first.reason) if first is not None else "")


def before_after(values, tick, cfg) -> tuple[float, float]:
    """Sum of per-arrival values before and after t*."""
    post = post_change(tick, cfg)
    v = np.asarray(values, float)
    return float(v[~post].sum()), float(v[post].sum())


def bin1_share(v, tick, grid, cfg) -> float:
    """Share of the post-change bids that fall in bin 1 of the (fixed, Phase 0) grid: never admitted."""
    post = post_change(tick, cfg)
    if not post.any():
        return float("nan")
    return float(np.mean(np.asarray(grid.assign(np.asarray(v)[post])) == 1))


def markup_per_phase(phases, cfg) -> "pd.DataFrame":
    """mu-hat, sigma-hat and tau used in each phase, with the true (mu, sigma) at the phase's start and end."""
    import pandas as pd
    ts, h = t_star_step(cfg), cfg.time.tick_seconds / 3600.0
    tr = truth(cfg)
    out = pd.DataFrame({"phase": phases.m, "start h": (phases.tau_start - 1) * h, "end h": phases.tau_end * h,
                        "end": phases.reason, "mu_hat": phases["mu"] if "mu" in phases else np.nan,
                        "sigma_hat": phases.sigma, "tau CPU": phases.tau_cpu, "tau RAM": phases.tau_ram})
    after = phases.tau_start - 1 >= ts
    out["true mu"] = np.where(after, tr["mu1"], tr["mu0"])
    out["true sigma"] = np.where(after, tr["sigma1"], tr["sigma0"])
    out["contains t*"] = (phases.tau_start - 1 < ts) & (phases.tau_end >= ts)
    return out

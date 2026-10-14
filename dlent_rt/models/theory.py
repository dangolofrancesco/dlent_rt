"""
DLENT-RT, THEORY model (write-up Version P; Zhang & Cheung 2025, Algorithm 1
with the exact LP-RS in place of Algorithm 2).

  * immediate observation: the true D of an admitted job is read at admission;
  * one occupancy counter (+ Test 1's in-phase window counter);
  * vanilla Bernoulli thinning U(t) ~ Bern((1 - xi) / (1 + eps~_U(t)));
  * frozen oracle: k_hat = 1[ r~_j - tau^T v_j > 0 ] with (r~, tau, v) of the phase;
  * physical gate (week-28 decision): an admission that would exceed the
    physical capacity c is rejected and logged as `phys_reject`.

Per arrival at tick t (paper line order):
  1. all ticks since the previous arrival are scanned (tests / timeout, exact
     on null ticks); a failure at u ends the phase at u-1 and re-initialises;
  2. physical releases up to t;
  3. type, oracle, thinning, gate, admission, realised reward.

The realised reward uses the phase state's bid model, phi = myerson.phi_job(v, c_j)
(the same function as Phase 0's and ReInit's type reward). Logged per arrival
(never used in a decision): sigma_used, and threshold_price = the bin floor L_k of
the smallest bid bin k whose type (k, q, a) the frozen oracle would admit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config import Config
from ..data import Dataset
from ..phase0 import PhaseState, assign_type_ids
from .core.error_params import ErrorParams, make_error_params
from .core.monitor import PhaseMonitor
from .core.occupancy import Occupancy
from .core.reinit import build_window_state, hindsight_lambda
from .core.stream import StreamData, build_stream


@dataclass
class RunResult:
    arrivals: list = field(default_factory=list)
    phases: list = field(default_factory=list)
    occupancy_events: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)


class TheoryModel:
    kind = "theory"

    def __init__(self, ds: Dataset, state0: PhaseState, cfg: Config):
        if cfg.scalarization.method != "linear":
            raise ValueError("online admission reward requires linear scalarization")
        if cfg.oracle.kind != "frozen":
            raise NotImplementedError("time_aware oracle: practical-model step")
        self.ds, self.cfg, self.state0 = ds, cfg, state0
        self.stream: StreamData = build_stream(ds, cfg)
        self.ep: ErrorParams = make_error_params(state0, ds, cfg)
        self.c = np.asarray(ds.capacity, float)
        forced_xi = cfg.constants.xi_override is not None
        self.a_max_eff = self.ep.xi * self.c if forced_xi else np.asarray(ds.a_max, float)
        self.d_max = int(ds.d_max_steps)
        rng = np.random.default_rng(cfg.run.seed + cfg.online.thinning_seed_offset)
        self.u_rand = rng.random(self.stream.n)       # common random numbers across ablations
        lam = cfg.objective
        self.lam = (lam.lambda1, lam.lambda2, lam.lambda3)
        self.tax = state0.carbon_price      # $/kgCO2 frozen at Phase 0

    # ------------------------------------------------------------------ #
    def _new_phase(self, state: PhaseState, tau_prev: int, W_prev: int):
        self.state = state
        self.tau_prev, self.W_prev = tau_prev, W_prev
        self.mon = PhaseMonitor(self.ep, state.r_tilde, state.v, self.c, self.a_max_eff,
                                tau_prev, W_prev, self.d_max, self.cfg)
        self.ph_arr: list[int] = []
        self.ph_adm: list[int] = []
        self.ph = dict(R=0.0, n_phys_reject=0, n_khat=0, n_thinned=0,
                       sat=0.0, prof_phi=0.0, prof_pay=0.0, carb_kg=0.0)
        # threshold bin per (q, a): smallest k with r~ - tau^T v > 0 (0 = none)
        ts = state.type_space
        shape = (ts.k_star, len(ts.priorities), ts.n_hw)
        ok = ((state.r_tilde - state.v @ state.tau) > 0).reshape(shape)
        self.k_thr = np.where(ok.any(axis=0), ok.argmax(axis=0) + 1, 0)
        self.L_thr = np.where(self.k_thr > 0,
                              np.asarray(state.L).reshape(shape)[np.maximum(self.k_thr - 1, 0),
                                                                 *np.indices(shape[1:])],
                              np.nan)

    def _end_phase(self, tau_m: int, reason: str, reinit: bool = True):
        W = tau_m - self.tau_prev
        st, S = self.state, self.stream
        arr = np.asarray(self.ph_arr, np.int64)
        adm = np.asarray(self.ph_adm, np.int64)
        if arr.size:
            lam_h, _ = hindsight_lambda(S, arr, W, st, self.ds, self.cfg)
        else:
            lam_h = 0.0
        bench = lam_h * W
        rec = dict(
            m=self.m, tau_start=self.tau_prev + 1, tau_end=tau_m, W=W, W_prev=self.W_prev,
            reason=reason, n_arrivals=int(arr.size), n_admitted=int(adm.size),
            **{k: v for k, v in self.ph.items()},
            lam_hat_start=float(st.lam_star), tau_cpu=float(st.tau[0]),
            sigma=float(st.myerson.sigma), known_distribution=bool(st.myerson.known),
            tau_ram=float(st.tau[1]) if st.tau.size > 1 else np.nan,
            sum_lam_window=self.mon.sum_lam, n_lp_solves=self.mon.wlp.n_solves,
            lam_hindsight=lam_h, bench_hindsight=bench,
            target_paper=(1 - self.ep.eps_D - self.ep.xi) * bench,
            regret_hindsight=bench - self.ph["R"],
            short=bool(arr.size < self.cfg.phase_transition.n_min),
        )
        if reinit:
            new, dg = build_window_state(S, arr, adm, W, st, self.ds, self.cfg)
            rec.update(reinit_grid_refit=dg.grid_refit, reinit_types_seen=dg.n_types_seen,
                       reinit_value_fallback=dg.n_value_fallback,
                       reinit_dur_fallback=dg.n_dur_fallback,
                       reinit_last_resort=dg.n_last_resort, reinit_lp_status=dg.lp_status,
                       next_lam_hat=dg.lam_hat)
        self.res.phases.append(rec)
        if reinit:
            self.m += 1
            self._new_phase(new, tau_m, W)

    def _advance(self, b: int):
        while True:
            sr = self.mon.scan(b)
            self._last_diag = sr.diag or self._last_diag
            if sr.fail_tick is None:
                return
            self._end_phase(sr.fail_tick - 1, sr.reason)

    # ------------------------------------------------------------------ #
    def _decide(self, i: int, t: int) -> dict:
        S, st, ep = self.stream, self.state, self.ep
        k = int(st.grid.assign(S.v[i], st.myerson)[0])
        j = int(assign_type_ids(st.type_space, np.array([k]), S.priority[i:i + 1],
                                S.hw_idx[i:i + 1])[0])
        # j = -1: IR reject (phi-space grid only, k = 0). Never index r_tilde[-1].
        ir_reject = j < 0
        r_j = np.nan if ir_reject else float(st.r_tilde[j])
        shadow = np.nan if ir_reject else float(st.tau @ st.v[j])
        score = -np.inf if ir_reject else r_j - shadow
        k_hat = score > 0
        pU = ep.p_U(self.W_prev, t - self.tau_prev)
        U = bool(self.u_rand[i] < pU)
        want = k_hat and U
        fits = self.occ.fits(S.A[i])
        admit = want and fits
        c_i = float(S.cost_base[i])
        phi = float(st.myerson.phi_job(S.v[i], c_i))
        qi = int(np.searchsorted(st.type_space.priorities, S.priority[i]))
        thr = float(self.L_thr[qi, int(S.hw_idx[i])])
        R = sat = prof = carb = pay = 0.0
        if admit:
            self.occ.add(i, S.A[i], t, int(S.D[i]))
            l1, l2, l3 = self.lam
            sat = float(S.priority[i] * S.v[i])
            prof = phi - float(S.C_elec[i])
            carb = float(S.C_carbon[i])
            R = l1 * sat + l2 * prof - l3 * carb * self.tax
            # payment: phi at the type's bin floor L_k (known distribution: given c_j)
            pay = (float(st.myerson.phi_job(st.L[j], c_i)) if st.myerson.known
                   else float(st.phi_tilde[j]))
            self.ph_adm.append(i)
        self.mon.register_arrival(t, j, R, admit, i, S.A[i], int(S.D[i]))
        self.ph_arr.append(i)
        ph = self.ph
        ph["R"] += R; ph["n_khat"] += int(k_hat); ph["n_thinned"] += int(k_hat and not U)
        ph["n_phys_reject"] += int(want and not fits)
        ph["sat"] += sat; ph["prof_phi"] += prof; ph["carb_kg"] += carb
        ph["prof_pay"] += (pay - float(S.C_elec[i])) if admit else 0.0
        return dict(
            i=i, t=t, m=self.m, type=j, k_bin=k, q=float(S.priority[i]), hw=int(S.hw_idx[i]),
            v=float(S.v[i]), phi_mech=phi, phi_gen=float(S.phi_gen[i]),
            r_tilde=r_j, shadow_cost=shadow, score=score, ir_reject=ir_reject,
            k_hat=k_hat, p_U=pU, U=U, fits=fits, admit=admit, phys_reject=want and not fits,
            R=R, payment=pay, cost_base=c_i, sigma_used=float(st.myerson.sigma),
            threshold_k=int(self.k_thr[qi, int(S.hw_idx[i])]), threshold_price=thr,
            threshold_markup=thr / c_i, D=int(S.D[i]), U_cpu=float(self.occ.U[0]),
            U_ram=float(self.occ.U[1]) if self.occ.U.size > 1 else np.nan,
            n_active=self.occ.n_active, **{k2: v for k2, v in self._last_diag.items()},
        )

    # ------------------------------------------------------------------ #
    def run(self) -> RunResult:
        self.res = RunResult()
        self.occ = Occupancy(self.c)
        self.m = 1
        self._last_diag = {}
        self._new_phase(self.state0, tau_prev=0, W_prev=self.d_max)    # tau^(0)=0, W^(0)=d_max
        S = self.stream
        for i in range(S.n):
            t = int(S.tick[i])
            self._advance(t)
            self.occ.release_until(t)
            self.res.arrivals.append(self._decide(i, t))
        T = int(self.ds.T)
        self._end_phase(T, "horizon", reinit=False)
        self.res.occupancy_events = self.occ.events
        self.res.summary = self._summary()
        return self.res

    def _summary(self) -> dict:
        S, ph = self.stream, self.res.phases
        lam_all, _ = hindsight_lambda(S, np.arange(S.n), int(self.ds.T), self.state0,
                                      self.ds, self.cfg)
        R_tot = float(sum(p["R"] for p in ph))
        bench_ph = float(sum(p["bench_hindsight"] for p in ph))
        short = [p["short"] for p in ph]
        run_short, max_consec = 0, 0
        for s in short:
            run_short = run_short + 1 if s else 0
            max_consec = max(max_consec, run_short)
        n = len(self.res.arrivals)
        a = self.res.arrivals
        return dict(
            model=self.kind, T=int(self.ds.T), n_arrivals=n, n_phases=len(ph),
            phase_end_reasons={r: sum(p["reason"] == r for p in ph)
                               for r in ("test1", "test2", "test3", "timeout", "horizon")},
            n_short_phases=int(sum(short)), max_consecutive_short=int(max_consec),
            n_admitted=int(sum(x["admit"] for x in a)),
            n_khat=int(sum(x["k_hat"] for x in a)),
            n_phys_reject=int(sum(x["phys_reject"] for x in a)),
            R_total=R_tot,
            bench_per_phase=bench_ph, regret_per_phase=bench_ph - R_tot,
            lam_hindsight_global=float(lam_all), bench_global=float(lam_all * self.ds.T),
            regret_global=float(lam_all * self.ds.T - R_tot),
            eps_D=self.ep.eps_D, one_minus_epsD_xi=1 - self.ep.eps_D - self.ep.xi,
            error_params=self.ep.as_dict(),
            a_max_eff=self.a_max_eff.tolist(), a_max_data=np.asarray(self.ds.a_max).tolist(),
            capacity=self.c.tolist(), d_max=self.d_max,
        )

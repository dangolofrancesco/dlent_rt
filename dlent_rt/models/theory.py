"""
DLENT-RT online model (write-up Version P; Zhang & Cheung 2025, Algorithm 1 with the exact LP-RS in place of
Algorithm 2). Theoretical model (the default config):

  * immediate observation: the true D of an admitted job is read at admission;
  * one occupancy counter (+ Test 1's in-phase window counter);
  * frozen oracle: k_hat = 1[ bin(b_j) >= k* ], k* the lowest bin of the arrival's column with r~ - tau^T v > 0,
    with (r~, tau, v) of the phase (D5);
  * Bernoulli thinning AFTER the oracle decision: U(t) ~ Bern((1 - xi) / (1 + eps~_U(t))), admit only if k_hat and U;
  * physical gate: an admission that would exceed the physical capacity c is rejected (`phys_reject`).

Practical-model switches (D5): oracle.kind = "live" scores every bin of the arrival's column with the type value
terms and the job's OWN electricity cost and carbon, estimated at arrival as its power x the column's d_bar x the
price / carbon intensity at arrival (bid-independent); ablation.thinning_off = true sets p_U = 1.

Admission and payment (dlent_rt.mechanism, week 33): a truthful threshold on HOURLY bids. The margin
r~ - tau^T v is increasing in the bin inside a column, so k* = the lowest bin with margin > 0 (frozen: per phase;
live: per job); threshold_kstar raises if the admitted bins are not an upper set. The reserve per hour is
r_j = P_j * price(t_j). admit <=> bin(b) >= k* and b >= r_j and thinning (theory only) and the gate. The hourly
payment is p_j = max(L_{k*}, r_j), the lowest hourly bid that would still be admitted, so 0 < p_j <= b_j; the job
pays p_j x the hours it runs. Every decision uses only the column's estimated duration: the job's true duration is
read only after the decision (occupancy, billing, realised results).
Logged per arrival: k*, the reserve, the hourly and total payment, surplus V - payment, realised profit
payment - C_elec, IR and the rejection reason ("bin below k*", "below reserve", "thinning", "no space").

Per arrival at tick t, through the shared event-driven loop dlent_rt.models.policy.simulate:
  1. before_arrival: all ticks since the previous arrival are scanned (tests / timeout, exact on null ticks); a
     failure at u ends the phase at u-1 and re-initialises (ReInit);
  2. physical releases up to t (simulate);
  3. want: type, oracle, then thinning; gate and admission (simulate); record: reward, payment, monitor, logs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import time

import numpy as np

from ..config import Config
from ..data import Dataset
from ..phase0 import PhaseState, assign_type_ids
from ..mechanism import admit_and_pay, reserve_price, threshold_kstar
from ..reward import job_terms
from .core.error_params import ErrorParams, make_error_params
from .core.monitor import PhaseMonitor
from .core.occupancy import Occupancy
from .core.reinit import build_window_state, hindsight_lambda
from .core.stream import StreamData, build_stream
from .policy import simulate


@dataclass
class RunResult:
    arrivals: list = field(default_factory=list)
    phases: list = field(default_factory=list)
    occupancy_events: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)


class TheoryModel:
    kind = "theory"

    def __init__(self, ds: Dataset, state0: PhaseState, cfg: Config):
        self.ds, self.cfg, self.state0 = ds, cfg, state0
        self.live = cfg.oracle.kind == "live"
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
        # reward unit of Tests 2-3: with gamma forced to 1 the thresholds bound the rewards of admissible actions by 1,
        # so rewards are divided by Phase 0's r_max = the largest positive r~ (fixed for the run); with the data gamma
        # (which contains r_max in $) they stay in $
        self.r_unit = float(state0.r_max) if cfg.constants.gamma_override is not None else 1.0
        # realised job reward terms of every stream job, scored with Phase 0's bid model and s_C: one scale for every
        # model and benchmark (the decisions use each phase's own sigma through r~)
        S = self.stream
        self.J = job_terms(S.v, S.cost_base, S.duration_hours, S.C_elec, S.C_carbon, state0.myerson, self.tax,
                           cfg.objective)

    # ------------------------------------------------------------------ #
    def _new_phase(self, state: PhaseState, tau_prev: int, W_prev: int):
        self.state = state
        self.tau_prev, self.W_prev = tau_prev, W_prev
        self.mon = PhaseMonitor(self.ep, state.r_tilde, state.v, self.c, self.a_max_eff,
                                tau_prev, W_prev, self.d_max, self.cfg, r_unit=self.r_unit)
        self.ph_arr: list[int] = []
        self.ph_adm: list[int] = []
        self.ph = dict(R=0.0, n_phys_reject=0, n_khat=0, n_below_reserve=0, n_thinned=0, cv=0.0, prof_phi=0.0, prof_pay=0.0,
                       carb_kg=0.0, sw=0.0, payments=0.0, surplus=0.0, n_ir_violations=0)
        # threshold: k* per column (q, a) = the lowest bin with margin > 0 (0 = none), frozen oracle
        ts = state.type_space
        self._shape = (ts.k_star, len(ts.priorities), ts.n_hw)
        self.k_star = threshold_kstar((state.r_tilde - state.v @ state.tau).reshape(self._shape))
        # the column's estimated duration in hours (the live oracle's cost estimate)
        self.dq_h = state.col(state.d_bar).reshape(self._shape[1:]) * self.cfg.time.tick_seconds / 3600.0

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
            sigma=float(st.myerson.sigma), rho=float(st.grid.ratio),
            n_columns_kstar0=int((self.k_star == 0).sum()), n_columns=int(self.k_star.size),
            n_cols_cost_carry=int((st.src_cost == 1).sum()), n_cols_cost_fallback=int((st.src_cost == 2).sum()),
            n_cols_dur_carry=int((st.src_dur == 1).sum()), n_cols_dur_fallback=int((st.src_dur == 2).sum()),
            tau_ram=float(st.tau[1]) if st.tau.size > 1 else np.nan,
            sum_lam_window=self.mon.sum_lam, n_lp_solves=self.mon.wlp.n_solves,
            lam_hindsight=lam_h, bench_hindsight=bench,
            target_paper=(1 - self.ep.eps_D - self.ep.xi) * bench,
            regret_hindsight=bench - self.ph["R"],
            short=bool(arr.size < self.cfg.phase_transition.n_min),
        )
        if reinit:
            new, dg = build_window_state(S, arr, adm, W, st, self.ds, self.cfg)
            rec.update(next_rho=dg.rho, next_sigma=dg.sigma, reinit_sigma_refit=dg.sigma_refit,
                       reinit_n_arrivals=dg.n_arrivals, reinit_n_observed=dg.n_observed,
                       reinit_types_seen=dg.n_types_seen,
                       reinit_cost_carry=dg.n_cost_carry, reinit_cost_fallback=dg.n_cost_fallback,
                       reinit_dur_carry=dg.n_dur_carry, reinit_dur_fallback=dg.n_dur_fallback,
                       reinit_lp_status=dg.lp_status, next_lam_hat=dg.lam_hat)
        self.res.phases.append(rec)
        if self.cfg.logging.progress_phases:
            self._progress(f"phase {self.m} ended ({reason}) at {tau_m * self._h:,.1f} h: {rec['n_arrivals']:,} "
                           f"arrivals, {rec['n_admitted']:,} admitted, tau = ({rec['tau_cpu']:.3g}, {rec['tau_ram']:.3g})")
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
    # Policy hook (dlent_rt.models.policy): before_arrival / want / record
    def before_arrival(self, i: int, t: int) -> None:
        self._advance(t)

    def _live_scores(self, i: int, qi: int, hw: int) -> np.ndarray:
        """
        Live oracle: the margin of every bid bin of the job's column (q, a), with the job's own costs estimated at
        arrival: its power x the column's d_bar (hours) x the price / carbon intensity at arrival. No true duration.
        """
        st, S, lam = self.state, self.stream, self.cfg.objective
        K, Q, A = self._shape
        ids = (np.arange(K) * Q + qi) * A + hw
        E = float(S.power_kw[i]) * float(self.dq_h[qi, hw])            # estimated energy [kWh]
        r = (st.r_tilde[ids] - lam.lambda2 * (E * float(S.price[i]) - st.C_elec_bar[ids])
             - lam.lambda3 * self.tax * (E * float(S.ci[i]) - st.C_carbon_bar[ids]))
        return r - st.v[ids] @ st.tau

    def want(self, i: int, t: int) -> bool:
        S, st, ep = self.stream, self.state, self.ep
        k = int(st.grid.assign(S.v[i])[0])
        j = int(assign_type_ids(st.type_space, np.array([k]), S.priority[i:i + 1], S.hw_idx[i:i + 1])[0])
        qi, hw = int(np.searchsorted(st.type_space.priorities, S.priority[i])), int(S.hw_idx[i])
        shadow = float(st.tau @ st.v[j])
        if self.live:       # practical: the type value terms, the job's own estimated costs, for every bin
            scores = self._live_scores(i, qi, hw)
            kstar = int(threshold_kstar(scores))
            r_j = float(scores[k - 1]) + shadow
        else:
            kstar = int(self.k_star[qi, hw])
            r_j = float(st.r_tilde[j])
        score = r_j - shadow
        # 1. the threshold decision, then the hourly reserve price
        reserve = float(reserve_price(S.power_kw[i], S.price[i]))
        k_hat, res_ok, L_k, pay = admit_and_pay(S.v[i], st.grid, kstar, reserve)
        # 2. thinning, after the decision
        pU = ep.p_U(self.W_prev, t - self.tau_prev)
        U = bool(self.u_rand[i] < pU)
        self._cur = dict(k=k, j=j, r_j=r_j, shadow=shadow, score=score, k_hat=k_hat, pU=pU, U=U, kstar=kstar,
                         reserve=reserve, res_ok=res_ok, L_k=L_k, pay=pay)
        return k_hat and res_ok and U

    def record(self, i: int, t: int, want: bool, fits: bool, admit: bool) -> None:
        S, st = self.stream, self.state
        cur = self._cur
        k, j, r_j = cur["k"], cur["j"], cur["r_j"]
        shadow, score, k_hat, pU, U = cur["shadow"], cur["score"], cur["k_hat"], cur["pU"], cur["U"]
        res_ok = cur["res_ok"]
        c_i, v_i = float(S.cost_base[i]), float(S.v[i])
        phi = float(self.J.phi[i])
        reason = ("" if admit else "bin below k*" if not k_hat else "below reserve" if not res_ok
                  else "thinning" if not U else "no space")
        R = cv = prof = carb = sw = 0.0
        pay = pay_rate = surplus = prof_pay = np.nan
        ir = True
        if admit:                                   # occ.add was done by simulate (physical gate)
            cv, prof = float(self.J.cv[i]), float(self.J.profit[i])
            carb, R, sw = float(self.J.carbon_kg[i]), float(self.J.R[i]), float(self.J.sw[i])
            # hourly payment max(L_{k*}, reserve), the lowest hourly bid that would still be admitted, x hours run
            pay_rate = cur["pay"]
            pay = pay_rate * float(S.duration_hours[i])
            surplus, prof_pay = cv - pay, pay - float(S.C_elec[i])
            ir = 0 < pay_rate <= v_i * (1 + 1e-12)
            self.ph_adm.append(i)
        self.mon.register_arrival(t, j, R, admit, i, S.A[i], int(S.D[i]))
        self.ph_arr.append(i)
        if self._next_report is not None and i + 1 >= self._next_report:
            self._progress(f"{100 * (i + 1) / S.n:.0f}% of the arrivals ({i + 1:,} / {S.n:,}), phase {self.m}, "
                           f"{len(self.ph_adm):,} admitted in this phase")
            self._next_report += self._report_step
        ph = self.ph
        ph["R"] += R; ph["n_khat"] += int(k_hat); ph["n_thinned"] += int(k_hat and res_ok and not U)
        ph["n_below_reserve"] += int(k_hat and not res_ok)
        ph["n_phys_reject"] += int(want and not fits)
        ph["cv"] += cv; ph["prof_phi"] += prof; ph["carb_kg"] += carb; ph["sw"] += sw
        if admit:
            ph["payments"] += pay; ph["surplus"] += surplus; ph["prof_pay"] += prof_pay
            ph["n_ir_violations"] += int(not ir)
        self.res.arrivals.append(dict(
            i=i, t=t, m=self.m, type=j, k_bin=k, q=float(S.priority[i]), hw=int(S.hw_idx[i]),
            v=v_i, phi_mech=phi, phi_gen=float(S.phi_gen[i]), oracle="live" if self.live else "frozen",
            r_tilde=r_j, r_tilde_phase=float(st.r_tilde[j]), shadow_cost=shadow, score=score,
            k_hat=k_hat, p_U=pU, U=U, fits=fits, admit=admit, phys_reject=want and not fits,
            R=R, sw=sw, payment_rate=pay_rate, payment=pay, surplus=surplus, value=v_i * float(S.duration_hours[i]),
            duration_hours=float(S.duration_hours[i]), profit_realised=prof_pay, ir=ir, cost_base=c_i,
            C_elec=float(S.C_elec[i]), sigma_used=float(st.myerson.sigma),
            below_grid=v_i < float(st.grid.bnd_v[0]), above_grid=v_i > float(st.grid.bnd_v[-1]),
            k_star=cur["kstar"], L_kstar=cur["L_k"], reserve=cur["reserve"], reserve_ok=res_ok,
            payment_by_reserve=bool(admit and cur["reserve"] > cur["L_k"]), reason=reason, D=int(S.D[i]), U_cpu=float(self.occ.U[0]),
            U_ram=float(self.occ.U[1]) if self.occ.U.size > 1 else np.nan,
            n_active=self.occ.n_active, **{k2: v for k2, v in self._last_diag.items()},
        ))

    # ------------------------------------------------------------------ #
    def _progress(self, msg: str) -> None:
        """One progress line: elapsed wall time, the estimated time left (from the share of arrivals done)."""
        el = time.perf_counter() - self._t0
        done = len(self.res.arrivals) / max(self.stream.n, 1)
        left = f", about {el * (1 - done) / done / 60:,.1f} min left" if 0 < done < 1 else ""
        print(f"[{self.kind} run, {el / 60:,.1f} min{left}] {msg}", flush=True)

    def run(self) -> RunResult:
        self._t0 = time.perf_counter()
        self._h = self.cfg.time.tick_seconds / 3600.0
        pe = self.cfg.logging.progress_every
        self._report_step = max(int(np.ceil(pe * self.stream.n)), 1) if pe and pe > 0 else None
        self._next_report = self._report_step
        self.res = RunResult()
        self.occ = Occupancy(self.c)
        self.m = 1
        self._last_diag = {}
        self._new_phase(self.state0, tau_prev=0, W_prev=self.d_max)    # tau^(0)=0, W^(0)=d_max
        simulate(self.stream, self.occ, self)
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
            R_total=R_tot, SW_total=float(sum(p["sw"] for p in ph)),
            payments_total=float(sum(p["payments"] for p in ph)),
            surplus_total=float(sum(p["surplus"] for p in ph)),
            profit_realised_total=float(sum(p["prof_pay"] for p in ph)),
            n_ir_violations=int(sum(p["n_ir_violations"] for p in ph)),
            n_below_reserve=int(sum(p["n_below_reserve"] for p in ph)),
            share_payment_by_reserve=float(np.mean([x["payment_by_reserve"] for x in self.res.arrivals if x["admit"]]))
            if any(x["admit"] for x in self.res.arrivals) else float("nan"),
            bench_per_phase=bench_ph, regret_per_phase=bench_ph - R_tot,
            lam_hindsight_global=float(lam_all), bench_global=float(lam_all * self.ds.T),
            regret_global=float(lam_all * self.ds.T - R_tot),
            eps_D=self.ep.eps_D, one_minus_epsD_xi=1 - self.ep.eps_D - self.ep.xi,
            error_params=self.ep.as_dict(),
            a_max_eff=self.a_max_eff.tolist(), a_max_data=np.asarray(self.ds.a_max).tolist(),
            capacity=self.c.tolist(), d_max=self.d_max,
        )

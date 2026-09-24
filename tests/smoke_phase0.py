"""
Smoke test for DLENT-RT Phase 0 (new pipeline: on-the-fly bids, energy model,
configurable normalization).

Run:  python -m tests.smoke_phase0  --csv data/batch_may2019_30k.csv  --n0 5000
"""
from __future__ import annotations

import argparse
import dataclasses

import numpy as np
import pandas as pd

from dlent_rt.config import load_config, default_config
from dlent_rt.data import load_dataset
from dlent_rt.phase0 import run_phase0
from dlent_rt.lp_oracle import solve_lp_rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--csv", default="data/batch_may2019_30k.csv")
    ap.add_argument("--grid-csv", default="data/grid_data_may_2019.csv")
    ap.add_argument("--n0", type=int, default=5000)
    ap.add_argument("--capacity-mode", default="fraction_of_concurrency_quantile",
                    choices=["fraction_of_volume", "fraction_of_peak",
                             "fraction_of_concurrency_quantile", "absolute"])
    ap.add_argument("--fraction", type=float, default=0.6)
    ap.add_argument("--concurrency-quantile", type=float, default=0.90)
    ap.add_argument("--semantics", default="real_tick",
                    choices=["real_tick"])
    ap.add_argument("--tick-seconds", type=float, default=1.0)
    # --scalarization / --normalization default to None (not a config value) so
    # that an explicit --config is respected when the user does not also pass
    # these flags. Only override the loaded config when the flag IS given.
    ap.add_argument("--scalarization", default=None,
                    choices=["linear", "chebyshev"])
    ap.add_argument("--bid-strategy", default="uniform",
                    choices=["uniform", "lognormal"])
    ap.add_argument("--normalization", default=None,
                    choices=["per_job", "moving_avg", "common_currency", "utopian"])
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else default_config()
    cfg = dataclasses.replace(
        cfg,
        data=dataclasses.replace(cfg.data, batch_csv=args.csv,
                                 grid_csv=args.grid_csv, n0=args.n0),
        time=dataclasses.replace(cfg.time, step_semantics=args.semantics,
                                 tick_seconds=args.tick_seconds),
        capacity=dataclasses.replace(cfg.capacity, mode=args.capacity_mode,
                                     fraction=args.fraction,
                                     concurrency_quantile=args.concurrency_quantile),
        bid=dataclasses.replace(cfg.bid, strategy=args.bid_strategy),
    )
    if args.scalarization is not None:
        cfg = dataclasses.replace(
            cfg, scalarization=dataclasses.replace(
                cfg.scalarization, method=args.scalarization,
            ),
        )
    if args.normalization is not None:
        cfg = dataclasses.replace(
            cfg, normalization=dataclasses.replace(
                cfg.normalization, strategy=args.normalization,
            ),
        )
    cfg = cfg.validate()

    print("=" * 70)
    print("DLENT-RT smoke test — Phase 0 (new pipeline)")
    print("=" * 70)
    print(f"csv={args.csv}  n0={args.n0}  semantics={args.semantics}")
    print(f"capacity={args.capacity_mode}@{args.fraction}  "
          f"scalarization={cfg.scalarization.method}")
    print(f"bid={args.bid_strategy}  normalization={cfg.normalization.strategy}")
    print()

    ds = load_dataset(cfg)

    # --- time model ---
    print("[time model]")
    print(f"  semantics={ds.step_semantics}  tick_seconds={ds.tick_seconds}")
    print(f"  T = {ds.T:,} steps   (online arrivals = {ds.n_arrivals_stream:,})")
    if ds.step_semantics == "real_tick":
        print(f"  arrival density = {ds.n_arrivals_stream / max(ds.T, 1):.6f}")
        print(f"  H spans {ds.H_span_steps:,} ticks")
    print(f"  d_max = {ds.d_max_steps:,} steps   d_max/T = "
          f"{ds.d_max_steps / max(ds.T, 1):.6f}")
    print()

    # --- dataset ---
    print("[dataset]")
    print(f"  |H|={ds.H.n}  |stream|={ds.stream.n}  T={ds.T:,}  resources={ds.resources}")
    print(f"  capacity={np.round(ds.capacity, 2)}")
    print(f"  a_max={np.round(ds.a_max, 4)}  d_max_steps={ds.d_max_steps}")
    print(f"  xi={ds.xi:.4f}  (target <= {1.0 / np.log(max(ds.T, 3)):.4f})")
    if ds.grid_data is not None:
        print(f"  grid_data: {len(ds.grid_data.timestamps)} hourly records  "
              f"price=[{ds.grid_data.price_min:.4f}, {ds.grid_data.price_max:.4f}] $/kWh  "
              f"CI=[{ds.grid_data.ci_min:.4f}, {ds.grid_data.ci_max:.4f}] kgCO2/kWh")
    else:
        print("  grid_data: NONE (using static defaults)")
    print()

    # --- preprocessing ---
    print("[preprocessing]")
    orep = ds.outlier_report
    print(f"  outlier policy={orep.policy}  before={orep.n_before} "
          f"after={orep.n_after} dropped={orep.n_dropped_total} "
          f"({100 * orep.n_dropped_total / max(orep.n_before, 1):.2f}%)")
    print(f"    by resource={orep.n_dropped_resource}  by duration={orep.n_dropped_duration}")
    print(f"    a_max thresholds={np.round(orep.a_max_thresholds, 4)}  "
          f"d_max hours={orep.d_max_hours_threshold:.2f}")
    print(f"  hardware: {ds.hw_catalogue.n_profiles} profiles "
          f"({ds.hw_catalogue.method})")
    dd = ds.d_max_diag
    print(f"  d_max: candidate={dd['candidate_from_quantile']} "
          f"ceiling={dd['horizon_ceiling']} chosen={dd['chosen']} "
          f"clamped={dd['clamped_by_ceiling']}")
    print()

    # --- stratified-H diagnostic ---
    h_hours = pd.to_datetime(ds.H.datetime).hour
    print("[H stratification]")
    print(f"  H time range: {ds.H.datetime.min()} -> {ds.H.datetime.max()}")
    print(f"  stream starts: {ds.stream.datetime.min()}")
    print(f"  H_span_steps: {ds.H_span_steps:,}")
    print("  Jobs per hour in H:")
    print(pd.Series(h_hours).value_counts().sort_index().to_string())
    print()

    # --- phase 0 ---
    st = run_phase0(ds, cfg)
    mask = st.p > 0
    cons = (st.p[mask, None] * st.v[mask]).sum(axis=0)

    print("[phase 0]")
    print(f"  bid strategy: {cfg.bid.strategy}")
    print(f"  normalization: {cfg.normalization.strategy}")
    print(f"  n_types={st.n_types}  (K*={st.type_space.k_star} x "
          f"|Q|={len(st.type_space.priorities)} x |A|={st.type_space.n_hw})")
    print(f"  types with p>0: {mask.sum()}  (sum of p = {st.p.sum():.6f})")
    print(f"  H jobs assigned: {st.n_assigned}/{ds.H.n}   "
          f"IR-rejected: {st.n_ir_rejected} ({100 * st.n_ir_rejected / max(ds.H.n, 1):.1f}%)")

    # bid diagnostics
    print(f"  bids: v_rate=[{st.bids.v_rate.min():.4f}, {st.bids.v_rate.max():.4f}]  "
          f"phi=[{st.bids.phi_rate.min():.4f}, {st.bids.phi_rate.max():.4f}]")
    phi_neg = (st.bids.phi_rate < 0).mean()
    print(f"  phi<0 fraction: {100 * phi_neg:.1f}%")

    # energy diagnostics
    from dlent_rt.phase0 import _compute_job_energy
    e = _compute_job_energy(ds.H, cfg)
    print(f"  energy: [{e.min():.6f}, {e.max():.4f}] kWh  mean={e.mean():.4f}")

    # marginals
    kq_a = (st.type_space.k_star, len(st.type_space.priorities), st.type_space.n_hw)
    p3 = st.p.reshape(kq_a)
    print(f"  populated k: {(p3.sum(axis=(1, 2)) > 0).sum()}/{kq_a[0]}")
    print(f"  populated q: {(p3.sum(axis=(0, 2)) > 0).sum()}/{kq_a[1]}")
    print(f"  populated a: {(p3.sum(axis=(0, 1)) > 0).sum()}/{kq_a[2]}")

    # reward
    print(f"  gamma={st.gamma:.4g}  xi={st.xi:.4g}  eps_D={st.eps_D:.4g}")
    if st.norm_ref is not None:
        print(f"  norm_ref={tuple(round(x, 4) for x in st.norm_ref)}")
        print(f"  norm_degenerate={st.norm_degenerate}")
    pos_r = (st.r_tilde[mask] > 0).sum()
    print(f"  r_tilde: [{st.r_tilde[mask].min():.4f}, {st.r_tilde[mask].max():.4f}]  "
          f"pos_reward_types={pos_r}/{mask.sum()}")

    # LP
    lp = solve_lp_rs(st.p, st.r_tilde, st.v, ds.capacity)
    if lp.x.size:
        cons_opt = (st.p[lp.support_idx, None] * st.v[lp.support_idx] * lp.x[:, None]).sum(axis=0)
    else:
        cons_opt = np.zeros(len(ds.resources))
    print(f"  LP consumption(x=1)={np.round(cons, 4)}  vs capacity={np.round(ds.capacity, 2)}")
    print(f"  LP consumption(opt)={np.round(cons_opt, 4)}")
    print(f"  util_opt={np.round(cons_opt / ds.capacity, 4)}")
    print(f"  tau={st.tau}  lambda_star={st.lam_star:.4f}")
    print(f"  LP binds: {bool(st.tau.max() > 0)}")
    print()

    # assertions
    assert st.n_types == st.type_space.k_star * len(st.type_space.priorities) * st.type_space.n_hw
    assert mask.sum() > 0, "no types observed"
    assert np.all(st.tau >= 0), "negative shadow prices"
    assert np.all(st.d_bar[mask] >= 1), "duration below one step"
    print("[OK] all assertions passed.")


if __name__ == "__main__":
    main()
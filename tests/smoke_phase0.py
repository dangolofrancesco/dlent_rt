"""
Smoke test for the DLENT-RT first slice (config + data + Phase 0).

Run:  python -m tests.smoke_phase0  --csv path/to/batch_may300k.csv  --n0 5000

Verifies:
  - config loads and validates
  - dataset ingests, H/stream split, hours->steps conversion, capacity, d_max
  - Phase 0 produces a well-formed PhaseState
  - the LP-RS binds and yields non-negative shadow prices when capacity is tight
"""
from __future__ import annotations

import argparse
import dataclasses

import numpy as np

from dlent_rt.config import load_config, default_config
from dlent_rt.data import load_dataset
from dlent_rt.phase0 import run_phase0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="YAML config path (optional)")
    ap.add_argument("--csv", default="data/batch_may300k.csv")
    ap.add_argument("--n0", type=int, default=5000)
    ap.add_argument("--capacity-mode", default="fraction_of_peak",
                    choices=["fraction_of_volume", "fraction_of_peak", "absolute"])
    ap.add_argument("--fraction", type=float, default=0.5)
    ap.add_argument("--scalarization", default="linear",
                    choices=["linear", "chebyshev"])
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else default_config()
    cfg = dataclasses.replace(
        cfg,
        data=dataclasses.replace(cfg.data, batch_csv=args.csv, n0=args.n0),
        capacity=dataclasses.replace(cfg.capacity, mode=args.capacity_mode,
                                     fraction=args.fraction),
        scalarization=dataclasses.replace(cfg.scalarization,
                                          method=args.scalarization),
    ).validate()

    print("=" * 70)
    print("DLENT-RT smoke test — Phase 0")
    print("=" * 70)
    print(f"csv={args.csv}  n0={args.n0}  capacity={args.capacity_mode}"
          f"@{args.fraction}  scalarization={args.scalarization}")
    print()

    ds = load_dataset(cfg)
    print("[dataset]")
    print(f"  |H|={ds.H.n}  |stream|=T={ds.T}  resources={ds.resources}")
    print(f"  capacity={np.round(ds.capacity, 1)}")
    print(f"  a_max={np.round(ds.a_max, 2)}  d_max_steps={ds.d_max_steps}")
    print(f"  xi=max_i a_max/c = {ds.xi:.4f}  (needs <~ 1/log T = "
          f"{1.0/np.log(max(ds.T, 3)):.4f})")
    print(f"  arrivals/hour={ds.arrivals_per_hour:.3f}")
    print(f"  duration_steps: min={ds.stream.duration_steps.min()} "
          f"max={ds.stream.duration_steps.max()} "
          f"mean={ds.stream.duration_steps.mean():.1f}")
    print()
    print("[preprocessing]")
    orep = ds.outlier_report
    print(f"  outlier policy={orep.policy}  before={orep.n_before} "
          f"after={orep.n_after} dropped={orep.n_dropped_total} "
          f"({100*orep.n_dropped_total/max(orep.n_before,1):.2f}%)")
    print(f"    by resource={orep.n_dropped_resource}  by duration={orep.n_dropped_duration}")
    print(f"    a_max thresholds={np.round(orep.a_max_thresholds, 2)}  "
          f"d_max hours={orep.d_max_hours_threshold:.2f}")
    print(f"  hardware: {ds.hw_catalogue.n_profiles} canonical profiles "
          f"({ds.hw_catalogue.method}, log_space={ds.hw_catalogue.log_space})")
    dd = ds.d_max_diag
    print(f"  d_max: quantile-candidate={dd['candidate_from_quantile']} "
          f"ceiling={dd['horizon_ceiling']} chosen={dd['chosen']} "
          f"ratio/T={dd['ratio_to_T']:.4f} clamped={dd['clamped_by_ceiling']}")
    print()

    st = run_phase0(ds, cfg)
    mask = st.p > 0
    cons = (st.p[mask, None] * st.v[mask]).sum(axis=0)
    print("[phase 0]")
    print(f"  n_types={st.n_types}  (K*={st.type_space.k_star} x "
          f"|Q|={len(st.type_space.priorities)} x |A|={st.type_space.n_hw})")
    print(f"  types with p>0: {mask.sum()}")
    print(f"  grid ratio={st.grid.ratio:.4f}")
    print(f"  gamma={st.gamma:.4g}  xi={st.xi:.4g}  eps_D={st.eps_D:.4g}")
    if st.norm_ref is not None:
        print(f"  utopian z*=(sat,prof,carb)="
              f"{tuple(round(x, 3) for x in st.norm_ref)}")
    print(f"  r_tilde range: [{st.r_tilde[mask].min():.4f}, "
          f"{st.r_tilde[mask].max():.4f}]")
    print(f"  LP consumption(x=1)={np.round(cons, 1)}  vs capacity="
          f"{np.round(ds.capacity, 1)}")
    print(f"  shadow prices tau={st.tau}")
    print(f"  lambda_star (LP optimum)={st.lam_star:.4f}")
    print()

    # assertions
    assert st.n_types == st.type_space.k_star * len(st.type_space.priorities) \
        * st.type_space.n_hw
    assert mask.sum() > 0, "no types observed"
    assert np.all(st.tau >= 0), "negative shadow prices"
    assert st.grid.bnd_phi[0] < st.grid.bnd_phi[-1], "degenerate grid"
    assert np.all(st.d_bar[mask] >= 1), "duration below one step"
    print("[OK] all smoke-test assertions passed.")


if __name__ == "__main__":
    main()

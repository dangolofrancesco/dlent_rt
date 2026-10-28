"""
Lambda sensitivity (week 30, read-only): does each objective weight still change the decisions?
Frozen Phase 0 state on H of the COMMITTED configs/default.yaml (git HEAD); no online run.

    dlent_venv/bin/python notebooks/src/lambda_sensitivity_check.py

R_job = l1*q*v + l2*(phi_job - C_elec) - l3*s_C*K   (dlent_rt.reward.job_terms)
r~_j  = the same on the per-type means (dlent_rt.reward.type_reward), s_C frozen.

1. s_C vs lambda: recomputed with s_C=None for every lambda and compared with the baseline.
2. Magnitudes of q*v, phi_job, C_elec, s_C*K on H and on the stream: mean, mean |.|, p10/p50/p90,
   share = mean|term| / sum over the four terms of mean|term|.
3. One-at-a-time sweep lambda_i in {0, 0.5, 1, 2, 5} (others 1), vs baseline (1, 1, 1):
   (a) share of POPULATED types with r~ > 0; (b) populated types whose 1[r~ > 0] flips;
   (c) H jobs whose 1[R_job > 0] flips; (d) on the admitted set = H jobs of types with r~ > 0:
   mean R_job and mean of the weighted components l1*q*v, l2*(phi - C_elec), -l3*s_C*K;
   (e) fluid LP-RS objective lambda* on the Phase 0 state (p, r~(lambda), v, c), per tick and * T.
4. s_C and the carbon share for normalization.quantile q in {0.75, 0.90, 0.95} (baseline lambda).
Also: decision flips when the electricity term (C_elec = 0) or the carbon term (K = 0) is removed.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dlent_rt.config import load_config
from dlent_rt.data import load_dataset
from dlent_rt.lp_oracle import solve_lp_rs
from dlent_rt.models.core.stream import build_stream
from dlent_rt.phase0 import (_compute_job_energy, _lookup_grid, assign_type_ids, run_phase0,
                             type_arrays)
from dlent_rt.reward import carbon_price, carbon_kg, electricity_cost, job_terms, type_mean, type_reward

warnings.simplefilter("ignore")
OUT = ROOT / "reports" / "week30" / "tables"
LAM_VALUES = (0.0, 0.5, 1.0, 2.0, 5.0)
Q_VALUES = (0.75, 0.90, 0.95)


def committed_default():
    txt = subprocess.check_output(["git", "-C", str(ROOT), "show", "HEAD:configs/default.yaml"], text=True)
    p = Path(tempfile.mkdtemp()) / "default_head.yaml"
    p.write_text(txt)
    return load_config(p)


def h_jobs(ds, st, cfg):
    """Per-job quantities of H exactly as Phase 0 builds them (checked against its type means)."""
    H = ds.H
    price, ci = _lookup_grid(H, ds)
    E = _compute_job_energy(H, cfg)
    C_elec, K = electricity_cost(E, price), carbon_kg(E, ci)
    v, c = st.bids.v_rate, st.bids.cost_base
    k = st.grid.assign(v, st.myerson)
    tid = assign_type_ids(st.type_space, np.asarray(k), H.priority, H.hw_idx)
    ok = tid >= 0
    N = np.bincount(tid[ok], minlength=st.n_types)
    assert np.allclose(type_mean(tid[ok], N, C_elec[ok]), st.C_elec_bar, rtol=1e-12, atol=0)
    assert np.allclose(type_mean(tid[ok], N, K[ok]), st.C_carbon_bar, rtol=1e-12, atol=0)
    return dict(q=H.priority.astype(float), v=v, c=c, C_elec=C_elec, K=K, tid=tid)


def magnitudes(q, v, phi, C_elec, K, s_C):
    terms = {"q*v": q * v, "phi_job": phi, "C_elec": C_elec, "s_C*K": s_C * K}
    mabs = {k: float(np.mean(np.abs(x))) for k, x in terms.items()}
    tot = sum(mabs.values())
    return pd.DataFrame([{"term": k, "mean": float(np.mean(x)), "mean |.|": mabs[k],
                          "p10": float(np.percentile(x, 10)), "p50": float(np.percentile(x, 50)),
                          "p90": float(np.percentile(x, 90)), "share of mean |reward| %": 100 * mabs[k] / tot}
                         for k, x in terms.items()]).set_index("term")


def main():
    cfg = committed_default()
    ds = load_dataset(cfg)
    st = run_phase0(ds, cfg)
    s_C = st.carbon_price
    pop = st.p > 0
    _, prio, _ = type_arrays(st.type_space, st.grid)
    J = h_jobs(ds, st, cfg)
    phi_H = np.asarray(st.myerson.phi_job(J["v"], J["c"]), float)
    S = build_stream(ds, cfg)
    phi_S = np.asarray(st.myerson.phi_job(S.v, S.cost_base), float)

    print(f"config: committed configs/default.yaml  capacity={cfg.capacity.mode}  |H|={ds.H.n:,}  "
          f"|stream|={S.n:,}  s_C={s_C:.6g} $/kgCO2  populated types {int(pop.sum())} / {st.n_types}")

    # 2. magnitudes
    mag_H = magnitudes(J["q"], J["v"], phi_H, J["C_elec"], J["K"], s_C)
    mag_S = magnitudes(S.priority.astype(float), S.v, phi_S, S.C_elec, S.C_carbon, s_C)
    print("\nmagnitudes on H [$]:\n" + mag_H.to_string(float_format=lambda x: f"{x:.4g}"))
    print("\nmagnitudes on the stream [$]:\n" + mag_S.to_string(float_format=lambda x: f"{x:.4g}"))

    def r_types(lam_cfg):
        return type_reward(prio, st.v_bar, st.phi_bar, st.C_elec_bar, st.C_carbon_bar, st.p, lam_cfg)

    def R_H(lam):
        return job_terms(J["q"], J["v"], J["c"], J["C_elec"], J["K"], st.myerson, s_C, lam)

    base_lam = cfg.objective
    r0, _, info0 = r_types(cfg)
    assert np.allclose(r0, st.r_tilde, rtol=1e-12, atol=1e-12)
    dec0 = r0 > 0
    Rj0 = R_H(base_lam).R > 0
    rows = []
    for name in ("lambda1", "lambda2", "lambda3"):
        for val in LAM_VALUES:
            lam = replace(base_lam, **{name: val})
            cfg_l = replace(cfg, objective=lam)
            r, _, info = r_types(cfg_l)                         # s_C recomputed (s_C=None)
            assert info["s_C"] == s_C, "s_C depends on lambda"
            dec = r > 0
            T = R_H(lam)
            adm = (J["tid"] >= 0) & dec[np.maximum(J["tid"], 0)]
            lp = solve_lp_rs(p=st.p, r=r, v=st.v, c=ds.capacity, presolve=cfg.lp.presolve)
            l1, l2, l3 = lam.lambda1, lam.lambda2, lam.lambda3
            rows.append({
                "lambda": name, "value": val, "s_C (recomputed)": info["s_C"],
                "(a) types r~>0 %": 100 * dec[pop].mean(),
                "(b) types flipped": int((dec != dec0)[pop].sum()),
                "(b) types flipped %": 100 * (dec != dec0)[pop].mean(),
                "(c) H jobs R_job>0 flipped %": 100 * ((T.R > 0) != Rj0).mean(),
                "(d) admitted H jobs": int(adm.sum()),
                "(d) mean R_job $": float(T.R[adm].mean()) if adm.any() else np.nan,
                "(d) mean l1*q*v $": float(l1 * T.cv[adm].mean()) if adm.any() else np.nan,
                "(d) mean l2*(phi-C_elec) $": float(l2 * T.profit[adm].mean()) if adm.any() else np.nan,
                "(d) mean -l3*s_C*K $": float(-l3 * T.carbon_usd[adm].mean()) if adm.any() else np.nan,
                "(e) fluid lambda* $/tick": float(lp.optimum),
                "(e) fluid lambda* * T $": float(lp.optimum) * ds.T,
            })
    sweep = pd.DataFrame(rows).set_index(["lambda", "value"])
    print("\none-at-a-time sweep (baseline (1,1,1); flips vs baseline; types = populated types):")
    print(sweep.to_string(float_format=lambda x: f"{x:.4g}"))

    # term removal at baseline lambda: electricity (C_elec = 0) and carbon (K = 0)
    trows = []
    for term in ("C_elec", "s_C*K"):
        Ce = np.zeros_like(st.C_elec_bar) if term == "C_elec" else st.C_elec_bar
        Kb = np.zeros_like(st.C_carbon_bar) if term == "s_C*K" else st.C_carbon_bar
        r, _, _ = type_reward(prio, st.v_bar, st.phi_bar, Ce, Kb, st.p, cfg, s_C=s_C)
        Cj = np.zeros_like(J["C_elec"]) if term == "C_elec" else J["C_elec"]
        Kj = np.zeros_like(J["K"]) if term == "s_C*K" else J["K"]
        T = job_terms(J["q"], J["v"], J["c"], Cj, Kj, st.myerson, s_C, base_lam)
        trows.append({"term removed": term, "types flipped": int(((r > 0) != dec0)[pop].sum()),
                      "types flipped %": 100 * ((r > 0) != dec0)[pop].mean(),
                      "H jobs R_job>0 flipped": int(((T.R > 0) != Rj0).sum()),
                      "H jobs R_job>0 flipped %": 100 * ((T.R > 0) != Rj0).mean()})
    tdf = pd.DataFrame(trows).set_index("term removed")
    print("\nterm removed (baseline lambda, s_C frozen):")
    print(tdf.to_string(float_format=lambda x: f"{x:.4g}"))
    tdf.to_csv(OUT / "lambda_term_removal.csv")

    # 4. quantile of the normalization
    qrows = []
    _, prio_t, _ = type_arrays(st.type_space, st.grid)
    for qv in Q_VALUES:
        cfg_q = replace(cfg, normalization=replace(cfg.normalization, quantile=qv))
        r, _, info = r_types(cfg_q)
        sC = info["s_C"]
        carb = sC * J["K"]
        T = job_terms(J["q"], J["v"], J["c"], J["C_elec"], J["K"], st.myerson, sC, base_lam)
        mabs = [np.mean(np.abs(x)) for x in (T.cv, T.phi, J["C_elec"], carb)]
        qrows.append({"q": qv, "s_C $/kgCO2": sC, "D_q $": info["D_q"], "K_q kg": info["K_q"],
                      "carbon share of mean |reward| % (H)": 100 * mabs[3] / sum(mabs),
                      "sum s_C*K / sum (q*v + phi - C_elec) % (H)": 100 * carb.sum() / T.R_nocarbon.sum(),
                      "types r~>0 %": 100 * (r > 0)[pop].mean(),
                      "types flipped vs q=0.90": int(((r > 0) != dec0)[pop].sum())})
    qdf = pd.DataFrame(qrows).set_index("q")
    print("\nnormalization quantile (baseline lambda):")
    print(qdf.to_string(float_format=lambda x: f"{x:.4g}"))

    mag_H.to_csv(OUT / "lambda_magnitudes_H.csv")
    mag_S.to_csv(OUT / "lambda_magnitudes_stream.csv")
    sweep.to_csv(OUT / "lambda_sweep.csv")
    qdf.to_csv(OUT / "lambda_quantile.csv")


if __name__ == "__main__":
    main()

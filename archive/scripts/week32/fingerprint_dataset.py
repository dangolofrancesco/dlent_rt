"""
Bit-identity check of the data path (week 32, step 2): no model is run, only load_dataset and the per-job inputs
of Phase 0 / the stream (bids, energy, power, grid lookup).

    dlent_venv/bin/python scripts/week32/fingerprint_dataset.py save  OUT.npz    # with the old code
    dlent_venv/bin/python scripts/week32/fingerprint_dataset.py check OUT.npz    # with the new code: must be equal

Configs: the committed configs/default.yaml (git HEAD) and configs/regime_small_cluster.yaml.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from capacity_regime import committed                                              # noqa: E402
from dlent_rt.config import load_config                                            # noqa: E402
from dlent_rt.data import load_dataset                                             # noqa: E402
from dlent_rt.phase0 import _compute_job_energy, _compute_job_power, _lookup_grid, _synthesize_bids  # noqa: E402


def fingerprint() -> dict:
    out = {}
    for name, cfg in (("default", committed()), ("regime", load_config(ROOT / "configs" / "regime_small_cluster.yaml"))):
        ds = load_dataset(cfg)
        out[f"{name}/capacity"] = ds.capacity
        out[f"{name}/scalars"] = np.array([ds.d_max_steps, ds.T, ds.H_span_steps, ds.xi, ds.n_arrivals_stream], float)
        out[f"{name}/a_max"] = ds.a_max
        out[f"{name}/centroids"] = ds.hw_catalogue.centroids
        for part, J, seed in (("H", ds.H, cfg.run.seed), ("S", ds.stream, cfg.run.seed + cfg.online.bid_seed_offset)):
            for k in ("A", "hw_idx", "duration_hours", "duration_steps", "arrival_step", "priority", "collection_id"):
                out[f"{name}/{part}/{k}"] = np.asarray(getattr(J, k))
            b = _synthesize_bids(J, cfg, seed=seed)
            out[f"{name}/{part}/v"], out[f"{name}/{part}/cost_base"] = b.v_rate, b.cost_base
            out[f"{name}/{part}/energy"] = _compute_job_energy(J, cfg)
            out[f"{name}/{part}/power"] = _compute_job_power(J, cfg)
            pr, ci = _lookup_grid(J, ds)
            out[f"{name}/{part}/price"], out[f"{name}/{part}/ci"] = np.asarray(pr), np.asarray(ci)
    return out


def main(mode: str, path: str) -> None:
    warnings.simplefilter("ignore")
    fp = fingerprint()
    if mode == "save":
        np.savez(path, **fp)
        print(f"saved {len(fp)} arrays -> {path}")
        return
    ref = dict(np.load(path))
    bad = [k for k in sorted(set(fp) | set(ref))
           if k not in fp or k not in ref or fp[k].shape != ref[k].shape or not np.array_equal(fp[k], ref[k])]
    print(f"{len(ref)} arrays compared; {len(bad)} differ" + ("" if not bad else ": " + ", ".join(bad)))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

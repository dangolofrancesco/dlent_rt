import json, tempfile, warnings
from pathlib import Path
from dlent_rt.models.runner import run_online
from tests.synthetic import synthetic_config


def test_theory_run_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        cfg = synthetic_config(Path(d), n0=400, n_jobs=3000, drift=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = run_online(cfg, out_dir=Path(d) / "out")
        s = json.loads((Path(d) / "out" / "summary.json").read_text())
        assert s["n_arrivals"] == len(r.arrivals) > 0
        assert s["n_admitted"] > 0
        # phases tile [1, T] without gaps
        ph = r.phases
        assert ph[0]["tau_start"] == 1 and ph[-1]["tau_end"] == s["T"]
        assert all(a["tau_end"] + 1 == b["tau_start"] for a, b in zip(ph, ph[1:]))
        # physical capacity never exceeded
        import pandas as pd
        occ = pd.read_parquet(Path(d) / "out" / "occupancy.parquet")
        assert (occ["U_cpu"] <= s["capacity"][0] + 1e-9).all()
        assert (occ["U_ram"] <= s["capacity"][1] + 1e-9).all()

"""
Build the job file data/batch_v2_30k.csv from the raw outputs of scripts/data/fetch_v2.py (week 32, step 2).
It replaces scripts/data/data_generator.py for the new path; data_generator.py stays as it was, so the old file
data/batch_may2019_30k.csv can still be rebuilt.

    dlent_venv/bin/python scripts/data/build_v2.py                                        # 30k
    dlent_venv/bin/python scripts/data/build_v2.py data/raw/v2_300k data/batch_v2_300k.csv   # 300k

One row per job (instances = the job's instances that have usage rows, i.e. that ran):
  arrival_utc         2019-05-01 07:00 UTC + (first usage start - 600 s), with its time zone (+00:00)
  priority_tier       Google's tiers: free <= 99, best-effort batch 100-115, mid 116-119, production 120-359,
                      monitoring >= 360 -> 1..5 (pd.cut, bin edges -1, 99, 115, 119, 359, inf)
  CPU, RAM            request = SUM over the instances of their first request (machines; 1.0 = the largest machine)
  CPU_peak, RAM_peak  real peak: the largest sum of the requests of the instances running at the same moment
                      (instance i runs in [its first usage start, its last usage end))
  CPU_hours_real, RAM_hours_real   sum over the instances of request x own running span (hours)
  duration            first start to last end of the job, hours
  n_instances, n_instances_no_request, request_changed (some instance shows > 1 request value),
  n_update_events (UPDATE_PENDING / UPDATE_RUNNING rows), first_start_us, last_end_us
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "v2"
OUT = ROOT / "data" / "batch_v2_30k.csv"
T0_UTC = pd.Timestamp("2019-05-01 07:00", tz="UTC")     # trace time t = 600 s (Google doc, table "Traces")
START_US = 600 * 10**6
TIER_BINS = [-1, 99, 115, 119, 359, np.inf]


def real_peak(inst: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Per job: max over time of the summed request of its running instances (ends before starts at a tie)."""
    s = inst[["collection_id", "first_start"] + cols].rename(columns={"first_start": "t"})
    e = inst[["collection_id", "last_end"] + cols].rename(columns={"last_end": "t"})
    e[cols] = -e[cols]
    ev = pd.concat([s.assign(o=1), e.assign(o=0)], ignore_index=True)
    ev = ev.sort_values(["collection_id", "t", "o"], kind="stable")
    run = ev.groupby("collection_id", sort=False)[cols].cumsum()
    return run.groupby(ev["collection_id"]).max().add_suffix("_peak")


def main(raw: Path = RAW, out: Path = OUT) -> None:
    inst = pd.read_csv(raw / "instances.csv.gz")
    jobs = pd.read_csv(raw / "jobs.csv")
    inst["no_request"] = inst["cpus"].isna()
    inst[["cpus", "memory"]] = inst[["cpus", "memory"]].fillna(0.0)
    inst["changed"] = (inst["n_cpu_values"] > 1) | (inst["n_ram_values"] > 1)
    inst["span_h"] = (inst["last_end"] - inst["first_start"]) / 3.6e9
    g = inst.groupby("collection_id")
    job = pd.DataFrame({
        "first_start_us": g["first_start"].min(), "last_end_us": g["last_end"].max(),
        "CPU": g["cpus"].sum(), "RAM": g["memory"].sum(),
        "CPU_hours_real": (inst["cpus"] * inst["span_h"]).groupby(inst["collection_id"]).sum(),
        "RAM_hours_real": (inst["memory"] * inst["span_h"]).groupby(inst["collection_id"]).sum(),
        "n_instances": g.size(), "n_instances_no_request": g["no_request"].sum(),
        "request_changed": g["changed"].any(), "n_update_events": g["n_update_events"].sum(),
    })
    job = job.join(real_peak(inst.rename(columns={"cpus": "CPU", "memory": "RAM"}), ["CPU", "RAM"]))
    job = job.join(jobs.set_index("collection_id")[["priority", "scheduling_class"]], how="inner").reset_index()
    job["arrival_utc"] = T0_UTC + pd.to_timedelta(job["first_start_us"] - START_US, unit="us")
    job["duration"] = (job["last_end_us"] - job["first_start_us"]) / 3.6e9
    job["priority_tier"] = pd.cut(job["priority"], bins=TIER_BINS, labels=[1, 2, 3, 4, 5]).astype(float)
    assert job["first_start_us"].min() >= START_US
    assert np.all(job["CPU_peak"] <= job["CPU"] * (1 + 1e-9)) and np.all(job["RAM_peak"] <= job["RAM"] * (1 + 1e-9))
    cols = ["collection_id", "arrival_utc", "priority", "priority_tier", "scheduling_class", "CPU", "RAM",
            "CPU_peak", "RAM_peak", "CPU_hours_real", "RAM_hours_real", "duration", "n_instances",
            "n_instances_no_request", "request_changed", "n_update_events", "first_start_us", "last_end_us"]
    job = job[cols].sort_values(["arrival_utc", "collection_id"], kind="stable").reset_index(drop=True)
    job.to_csv(out, index=False)
    print(f"{len(job)} jobs -> {out}")


if __name__ == "__main__":
    import sys
    a = sys.argv[1:]
    main(ROOT / a[0], ROOT / a[1]) if len(a) == 2 else main()

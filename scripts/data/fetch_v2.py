"""
New fetch (week 32, step 2): a uniform, reproducible sample of JOBS of cell a that start and end inside the trace.

    dlent_venv/bin/python scripts/data/fetch_v2.py dry    # dry run of the two table scans: GB and estimated cost
    dlent_venv/bin/python scripts/data/fetch_v2.py run    # dry run again, refuse above MAX_GB, then run
    dlent_venv/bin/python scripts/data/fetch_v2.py run 300000 data/raw/v2_300k   # sample size and output folder

    dlent_venv/bin/python scripts/data/fetch_v2.py reuse 300000 data/raw/v2_300k _script<id>   # see below

The sample is the N smallest fingerprints, so a larger N with the same SALT is a SUPERSET of a smaller one.
"reuse": run only the sample and the two per-sample SELECTs of SCRIPT on the TEMP tables (iu, ie, pop) that an
earlier run left in its hidden dataset _script<id> (kept 24 h). The trace tables do not change, so the result equals
a full run; the population is not re-downloaded (copy population.csv.gz from that run).

Definitions (Google cluster-usage traces v3):
  * job = collection_type 0 (alloc sets, type 1, are left out). Only collections with >= 1 row in instance_usage.
  * trace time t is in microseconds; t = 600 s is 1 May 2019 00:00 PDT = 07:00 UTC, so
    clock time UTC = 2019-05-01 07:00 UTC + (t - 600 s). The trace window is [600 s, 600 s + 31 days].
  * border job: first usage start < 600 s (started before the trace) OR last usage end > window end - 300 s (still
    running at the end; 300 s = one usage window, the doc's maximum window length).
  * sample: the N non-border jobs with at least one instance request and the smallest
    FARM_FINGERPRINT(SALT || collection_id) (ties broken by collection_id). Same SALT -> same sample.
  * job priority and scheduling class = those of the job's FIRST event in collection_events (by time, then type).
    Week 32 fix: the old ANY_VALUE over instance_events picked an arbitrary value, and instance_events often carries
    priority 0 on some events of a job whose collection priority is, e.g., 200.
  * instance request = its FIRST event with a non-null request (by time); "changed" = the instance shows more than one
    distinct CPU or RAM request value; UPDATE_PENDING / UPDATE_RUNNING rows are counted too.

The two table scans run once, inside one BigQuery script, into TEMP tables; the small SELECTs read those.
Outputs (data/raw/v2/): population.csv.gz (every job of cell a), instances.csv.gz (every instance of the sample),
jobs.csv (sample: priority, scheduling class), meta.json (bytes, salt, query time).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google.cloud import bigquery

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "v2"
DS = "`google.com:google-cluster-data.clusterdata_2019_a`"
SALT = "dlent-rt-v2-2019a"
N_SAMPLE = 30000
START_US = 600 * 10**6
END_US = START_US + 31 * 86400 * 10**6
WINDOW_US = 300 * 10**6
MAX_GB = 500.0
USD_PER_TIB = 6.25      # BigQuery on-demand list price (assumption; the first 1 TiB per month is free)

SCAN_USAGE = f"""
    SELECT collection_id, instance_index,
           MIN(start_time) AS first_start, MAX(end_time) AS last_end, COUNT(*) AS n_windows
    FROM {DS}.instance_usage
    WHERE collection_type = 0
    GROUP BY collection_id, instance_index
"""

SCAN_EVENTS = f"""
    SELECT collection_id, instance_index,
           ARRAY_AGG(IF(resource_request.cpus IS NULL, NULL,
                        STRUCT(resource_request.cpus AS cpus, resource_request.memory AS memory))
                     IGNORE NULLS ORDER BY time LIMIT 1)[SAFE_OFFSET(0)] AS first_req,
           COUNT(DISTINCT resource_request.cpus) AS n_cpu_values,
           COUNT(DISTINCT resource_request.memory) AS n_ram_values,
           COUNTIF(type IN (9, 10)) AS n_update_events
    FROM {DS}.instance_events
    WHERE collection_type = 0
    GROUP BY collection_id, instance_index
"""

SCRIPT_TMPL = """
CREATE TEMP TABLE iu AS {SCAN_USAGE};
CREATE TEMP TABLE ie AS {SCAN_EVENTS};
CREATE TEMP TABLE req AS
    SELECT iu.collection_id, COUNT(*) AS n_inst_ran, COUNTIF(ie.first_req.cpus IS NOT NULL) AS n_inst_with_req,
           SUM(ie.first_req.cpus) AS cpu_sum, SUM(ie.first_req.memory) AS ram_sum,
           MIN(iu.first_start) AS first_start, MAX(iu.last_end) AS last_end
    FROM iu LEFT JOIN ie USING (collection_id, instance_index)
    GROUP BY iu.collection_id;
CREATE TEMP TABLE pop AS
    SELECT r.*, n.n_inst_events,
           (r.first_start < {START_US} OR r.last_end > {END_US_MINUS_W}) AS border,
           FARM_FINGERPRINT(CONCAT('{SALT}', CAST(r.collection_id AS STRING))) AS fp
    FROM req r LEFT JOIN (SELECT collection_id, COUNT(*) AS n_inst_events FROM ie GROUP BY collection_id) n
      USING (collection_id);
CREATE TEMP TABLE samp AS
    SELECT collection_id FROM pop
    WHERE NOT border AND n_inst_with_req > 0
    ORDER BY fp, collection_id
    LIMIT {n_sample};
SELECT * FROM pop;
SELECT iu.collection_id, iu.instance_index, iu.first_start, iu.last_end, iu.n_windows,
       ie.first_req.cpus AS cpus, ie.first_req.memory AS memory,
       ie.n_cpu_values, ie.n_ram_values, ie.n_update_events
FROM iu JOIN samp USING (collection_id) LEFT JOIN ie USING (collection_id, instance_index);
SELECT s.collection_id,
       ARRAY_AGG(STRUCT(ce.priority, ce.scheduling_class) ORDER BY ce.time, ce.type LIMIT 1)[OFFSET(0)].priority AS priority,
       ARRAY_AGG(STRUCT(ce.priority, ce.scheduling_class) ORDER BY ce.time, ce.type LIMIT 1)[OFFSET(0)].scheduling_class
           AS scheduling_class,
       COUNT(DISTINCT ce.priority) AS n_priority_values
FROM samp s JOIN {DS}.collection_events ce USING (collection_id)
GROUP BY s.collection_id;
"""
OUTPUTS = ("population.csv.gz", "instances.csv.gz", "jobs.csv")


def dry(client: bigquery.Client) -> float:
    total = 0.0
    for name, sql in (("instance_usage scan", SCAN_USAGE), ("instance_events scan", SCAN_EVENTS)):
        gb = client.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
                          ).total_bytes_processed / 1e9
        total += gb
        print(f"  {name}: {gb:.1f} GB  (~${gb * 1e9 / 2**40 * USD_PER_TIB:.2f})")
    print(f"  total of the two scans: {total:.1f} GB (~${total * 1e9 / 2**40 * USD_PER_TIB:.2f} at "
          f"${USD_PER_TIB}/TiB). The SELECTs on the TEMP tables add their size (not known before the run).")
    return total


def script(n_sample: int) -> str:
    return SCRIPT_TMPL.format(DS=DS, SCAN_USAGE=SCAN_USAGE, SCAN_EVENTS=SCAN_EVENTS, START_US=START_US,
                              END_US_MINUS_W=END_US - WINDOW_US, SALT=SALT, n_sample=n_sample)


PRIORITY_SQL = f"""
    SELECT collection_id,
           ARRAY_AGG(STRUCT(priority, scheduling_class) ORDER BY time, type LIMIT 1)[OFFSET(0)].priority AS priority,
           ARRAY_AGG(STRUCT(priority, scheduling_class) ORDER BY time, type LIMIT 1)[OFFSET(0)].scheduling_class
               AS scheduling_class,
           COUNT(DISTINCT priority) AS n_priority_values
    FROM {DS}.collection_events
    WHERE collection_id IN UNNEST(@ids)
    GROUP BY collection_id
"""


def fix_priority(client: bigquery.Client, out: Path, dry: bool) -> float:
    """Priority and scheduling class of the job's first collection event, for the jobs of out/jobs.csv (the same rule
    as SCRIPT); rewrites out/jobs.csv. Returns the GB scanned."""
    import pandas as pd
    jobs = pd.read_csv(out / "jobs.csv")
    ids = [int(x) for x in jobs.collection_id]
    cfg = bigquery.QueryJobConfig(dry_run=dry, use_query_cache=False,
                                  query_parameters=[bigquery.ArrayQueryParameter("ids", "INT64", ids)])
    job = client.query(PRIORITY_SQL, job_config=cfg)
    if dry:
        return job.total_bytes_processed / 1e9
    fixed = job.to_dataframe()
    assert set(fixed.collection_id) == set(jobs.collection_id)
    fixed.to_csv(out / "jobs.csv", index=False)
    print(f"  jobs.csv: {len(fixed)} rows, billed {job.total_bytes_billed / 1e9:.1f} GB")
    return job.total_bytes_billed / 1e9


def reuse(client: bigquery.Client, n_sample: int, out: Path, dataset: str) -> None:
    """The per-sample SELECTs of SCRIPT on an earlier run's TEMP tables (same SQL, tables fully qualified)."""
    d = f"`{client.project}.{dataset}"
    samp = (f"samp AS (SELECT collection_id FROM {d}.pop` WHERE NOT border AND n_inst_with_req > 0 "
            f"ORDER BY fp, collection_id LIMIT {n_sample})")
    queries = {
        "instances.csv.gz": f"""WITH {samp}
            SELECT iu.collection_id, iu.instance_index, iu.first_start, iu.last_end, iu.n_windows,
                   ie.first_req.cpus AS cpus, ie.first_req.memory AS memory,
                   ie.n_cpu_values, ie.n_ram_values, ie.n_update_events
            FROM {d}.iu` iu JOIN samp USING (collection_id) LEFT JOIN {d}.ie` ie USING (collection_id, instance_index)""",
        "jobs.csv": f"""WITH {samp}
            SELECT s.collection_id, ANY_VALUE(ie.priority) AS priority, ANY_VALUE(ie.scheduling_class) AS scheduling_class,
                   COUNT(DISTINCT ie.priority) AS n_priority_values
            FROM samp s JOIN {d}.ie` ie USING (collection_id)
            GROUP BY s.collection_id""",
    }
    out.mkdir(parents=True, exist_ok=True)
    billed = 0
    for name, q in queries.items():
        job = client.query(q, job_config=bigquery.QueryJobConfig(use_query_cache=False))
        df = job.to_dataframe()
        df.to_csv(out / name, index=False)
        billed += job.total_bytes_billed or 0
        print(f"  {name}: {len(df)} rows")
    (out / "meta.json").write_text(json.dumps(dict(salt=SALT, n_sample=n_sample, start_us=START_US, end_us=END_US,
                                                   window_us=WINDOW_US, reused_dataset=dataset,
                                                   total_bytes_billed=billed), indent=2))
    print(f"  billed {billed / 1e9:.1f} GB")


def main(mode: str, n_sample: int = N_SAMPLE, out: Path = OUT, dataset: str = "") -> None:
    load_dotenv(ROOT / ".env")
    client = bigquery.Client(project=os.environ["GCP_PROJECT_ID"])
    if mode == "reuse":
        reuse(client, n_sample, out, dataset)
        return
    if mode in ("priority-dry", "priority"):
        gb = fix_priority(client, out, dry=mode == "priority-dry")
        print(f"  priority fix: {gb:.1f} GB")
        return
    print("Dry run:")
    total = dry(client)
    if mode != "run":
        return
    if total > MAX_GB:
        sys.exit(f"STOP: {total:.1f} GB > {MAX_GB} GB. Ask before running.")
    t0 = time.time()
    job = client.query(script(n_sample), job_config=bigquery.QueryJobConfig(use_query_cache=False))
    job.result()
    children = sorted(client.list_jobs(parent_job=job.job_id), key=lambda j: j.created)
    selects = [c for c in children if c.statement_type == "SELECT"]
    assert len(selects) == len(OUTPUTS), [c.statement_type for c in children]
    out.mkdir(parents=True, exist_ok=True)
    for c, name in zip(selects, OUTPUTS):
        df = c.to_dataframe()
        df.to_csv(out / name, index=False)
        print(f"  {name}: {len(df)} rows")
    job.reload()
    meta = dict(salt=SALT, n_sample=n_sample, start_us=START_US, end_us=END_US, window_us=WINDOW_US,
                script_job_id=job.job_id, total_bytes_billed=job.total_bytes_billed,
                total_bytes_processed=job.total_bytes_processed, seconds=round(time.time() - t0, 1),
                children=[dict(type=c.statement_type, bytes_billed=c.total_bytes_billed) for c in children])
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  billed {job.total_bytes_billed / 1e9:.1f} GB in {meta['seconds']} s")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "dry", int(a[1]) if len(a) > 1 else N_SAMPLE, ROOT / a[2] if len(a) > 2 else OUT,
         a[3] if len(a) > 3 else "")

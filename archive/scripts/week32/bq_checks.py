"""
Week 32 data audit, task 5: pipeline checks on BigQuery (cell a, the 29,670 collection_ids of data/batch_may2019_30k.csv).

    dlent_venv/bin/python scripts/week32/bq_checks.py dry          # dry run: bytes scanned per query, nothing billed
    dlent_venv/bin/python scripts/week32/bq_checks.py run a b c    # run the named queries (refuses any query > MAX_GB)

Queries
  a  collection_events: collection_type and alloc_collection_id of each collection.
  lost  collection_events for the 330 ids of data/raw/raw_events_30k.csv with no usage row (lost in the merge).
  pop   collection_events, ALL collections of cell a: type, ever scheduled, last event type (fair sample, task 8).
  p0    instance_events, ALL collections of cell a, two columns: instances and event rows per collection.
  b0 instance_events, two columns only (cheap version of b): per collection, number of instances and of event rows.
  b  instance_events, ALL collections of cell a (bytes do not depend on the id filter): per collection, number of instances, number of event rows, the request summed over the FIRST
     event of each instance (what fetch_data.py does, after FIX 1) and over ALL events (the old fetch_30ksample.py), and how many
     instances change their request over time (distinct request values > 1, UPDATE_PENDING / UPDATE_RUNNING rows).
  c  instance_usage: for 500 random jobs (seed 42), each instance's running span [min start_time, max end_time];
     with b's first-event request per instance this gives the real concurrent request over time.
Outputs: reports/week32/bq/<name>.csv (gitignored data, not committed).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "week32" / "bq"
DS = "`google.com:google-cluster-data.clusterdata_2019_a`"
MAX_GB = 50.0
N_SAMPLE_C = 500

COLL = """
        SELECT collection_id,
               ANY_VALUE(collection_type) AS collection_type,
               MAX(alloc_collection_id) AS alloc_collection_id,
               COUNT(*) AS n_events,
               MAX(IF(type = 3, 1, 0)) AS ever_scheduled,
               ARRAY_AGG(type ORDER BY time DESC LIMIT 1)[OFFSET(0)] AS last_type,
               MIN(time) AS first_time
        FROM {DS}.collection_events
        {WHERE}
        GROUP BY collection_id
"""

QUERIES = {
    "lost": COLL.format(DS=DS, WHERE="WHERE collection_id IN UNNEST(@ids)"),
    "pop": COLL.format(DS=DS, WHERE=""),
    "p0": f"""
        SELECT collection_id, COUNT(DISTINCT instance_index) AS n_instances, COUNT(*) AS n_event_rows
        FROM {DS}.instance_events
        GROUP BY collection_id
    """,
    "a": f"""
        SELECT collection_id,
               ANY_VALUE(collection_type) AS collection_type,
               MAX(alloc_collection_id) AS alloc_collection_id,
               COUNT(*) AS n_events
        FROM {DS}.collection_events
        WHERE collection_id IN UNNEST(@ids)
        GROUP BY collection_id
    """,
    "b0": f"""
        SELECT collection_id,
               COUNT(DISTINCT instance_index) AS n_instances,
               COUNT(*) AS n_event_rows
        FROM {DS}.instance_events
        WHERE collection_id IN UNNEST(@ids)
        GROUP BY collection_id
    """,
    "b": f"""
        WITH ev AS (
            SELECT collection_id, instance_index, type,
                   resource_request.cpus AS cpus, resource_request.memory AS memory,
                   ROW_NUMBER() OVER (PARTITION BY collection_id, instance_index
                                      ORDER BY IF(resource_request.cpus IS NULL, 1, 0), time ASC) AS rn
            FROM {DS}.instance_events
        ),
        inst AS (
            SELECT collection_id, instance_index, COUNT(*) AS n_rows,
                   SUM(IF(rn = 1, cpus, 0)) AS cpu_first, SUM(IF(rn = 1, memory, 0)) AS ram_first,
                   SUM(cpus) AS cpu_all, SUM(memory) AS ram_all,
                   COUNT(DISTINCT cpus) AS n_cpu_values, COUNT(DISTINCT memory) AS n_ram_values,
                   COUNTIF(type IN (9, 10)) AS n_update_rows
            FROM ev
            GROUP BY collection_id, instance_index
        )
        SELECT collection_id, COUNT(*) AS n_instances, SUM(n_rows) AS n_event_rows,
               SUM(cpu_first) AS cpu_first, SUM(ram_first) AS ram_first,
               SUM(cpu_all) AS cpu_all_events, SUM(ram_all) AS ram_all_events,
               COUNTIF(n_cpu_values > 1 OR n_ram_values > 1) AS n_inst_request_changed,
               SUM(n_update_rows) AS n_update_rows
        FROM inst
        GROUP BY collection_id
    """,
    "c": f"""
        WITH ev AS (
            SELECT collection_id, instance_index,
                   resource_request.cpus AS cpus, resource_request.memory AS memory,
                   ROW_NUMBER() OVER (PARTITION BY collection_id, instance_index ORDER BY time ASC) AS rn
            FROM {DS}.instance_events
            WHERE collection_id IN UNNEST(@ids) AND resource_request.cpus IS NOT NULL
        ),
        us AS (
            SELECT collection_id, instance_index, MIN(start_time) AS start_time, MAX(end_time) AS end_time,
                   COUNT(*) AS n_windows
            FROM {DS}.instance_usage
            WHERE collection_id IN UNNEST(@ids)
            GROUP BY collection_id, instance_index
        )
        SELECT us.collection_id, us.instance_index, us.start_time, us.end_time, us.n_windows,
               ev.cpus, ev.memory
        FROM us LEFT JOIN ev
          ON ev.collection_id = us.collection_id AND ev.instance_index = us.instance_index AND ev.rn = 1
    """,
}


def ids_for(name: str) -> list[int]:
    if name == "lost":
        ev = pd.read_csv(ROOT / "data" / "raw" / "raw_events_30k.csv", usecols=["collection_id"])["collection_id"]
        us = pd.read_csv(ROOT / "data" / "raw" / "raw_usage_30k.csv", usecols=["collection_id"])["collection_id"]
        return [int(x) for x in sorted(set(ev) - set(us))]
    ids = pd.read_csv(ROOT / "data" / "batch_may2019_30k.csv", usecols=["collection_id"])["collection_id"]
    ids = ids.astype("int64").to_numpy()
    if name == "c":
        ids = np.random.default_rng(42).choice(np.sort(ids), size=N_SAMPLE_C, replace=False)
    return [int(x) for x in ids]


def job_config(name: str, dry: bool) -> bigquery.QueryJobConfig:
    params = [bigquery.ArrayQueryParameter("ids", "INT64", ids_for(name))] if "@ids" in QUERIES[name] else []
    return bigquery.QueryJobConfig(dry_run=dry, use_query_cache=False, query_parameters=params)


def main(argv: list[str]) -> None:
    load_dotenv(ROOT / ".env")
    client = bigquery.Client(project=os.environ["GCP_PROJECT_ID"])
    mode, names = argv[0], (argv[1:] or list(QUERIES))
    for name in names:
        gb = client.query(QUERIES[name], job_config=job_config(name, True)).total_bytes_processed / 1e9
        print(f"[{name}] dry run: {gb:.2f} GB scanned")
        if mode != "run":
            continue
        if gb > MAX_GB:
            print(f"[{name}] SKIPPED: {gb:.1f} GB > {MAX_GB} GB limit")
            continue
        df = client.query(QUERIES[name], job_config=job_config(name, False)).to_dataframe()
        OUT.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(f"[{name}] {len(df)} rows -> {OUT / f'{name}.csv'}")


if __name__ == "__main__":
    main(sys.argv[1:])

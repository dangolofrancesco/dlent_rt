from google.cloud import bigquery
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()

os.makedirs("data/raw", exist_ok=True)
project_id = os.getenv("GCP_PROJECT_ID")

if not project_id:
    raise EnvironmentError("GCP_PROJECT_ID environment variable is not set.")

client = bigquery.Client(project=project_id)

print("Step 1: Sampling of 30.000 unique and valid Job IDs...")
query_jobs = """
    SELECT DISTINCT collection_id
    FROM `google.com:google-cluster-data.clusterdata_2019_a.instance_events`
    WHERE resource_request.cpus IS NOT NULL
    AND RAND() < 0.05
    LIMIT 30000
"""
jobs_df = client.query(query_jobs).to_dataframe()
sampled_job_ids = jobs_df['collection_id'].tolist()
print(f" -> Found {len(sampled_job_ids)} jobs")

job_config = bigquery.QueryJobConfig(
    query_parameters=[
        bigquery.ArrayQueryParameter("job_ids", "INT64", sampled_job_ids)
    ]
)

print("\nStep 2: Extraction and Aggregation of the Events (DEDUPLICATED)...")
# FIX 1: For each (collection_id, instance_index), take ONLY the row with
# the minimum timestamp (first event, typically SUBMIT). This avoids counting
# the same task multiple times if it has undergone SCHEDULE/RESCHEDULE/etc.
# The subquery dedup isolates the first event per task, then the outer query
# aggregates by collection.
query_events = """
    WITH dedup AS (
        SELECT
            collection_id,
            instance_index,
            priority,
            scheduling_class,
            resource_request.cpus AS cpus,
            resource_request.memory AS memory,
            ROW_NUMBER() OVER (
                PARTITION BY collection_id, instance_index
                ORDER BY time ASC
            ) AS rn
        FROM `google.com:google-cluster-data.clusterdata_2019_a.instance_events`
        WHERE collection_id IN UNNEST(@job_ids)
          AND resource_request.cpus IS NOT NULL
    )
    SELECT
        collection_id,
        ANY_VALUE(priority) AS priority,
        ANY_VALUE(scheduling_class) AS scheduling_class,
        SUM(cpus) AS resource_request_cpus,
        SUM(memory) AS resource_request_ram
    FROM dedup
    WHERE rn = 1
    GROUP BY collection_id
"""
events_df = client.query(query_events, job_config=job_config).to_dataframe()
events_df.to_csv("data/raw/raw_events_30k.csv", index=False)
print(f" -> Downloaded {len(events_df)} aggregated jobs (Events, deduplicated).")

print("\nStep 3: Extraction and Aggregation of the Lifecycle Usage...")
# FIX 2: Calculate the CPU*time and MEM*time integral correctly.
# Each row of instance_usage represents a 5-minute window for a task.
# - average_usage.cpus is the average CPU consumption of the task in that window
# - average_usage.memory is the average memory consumption of the task in that window
# - Per ottenere CPU-ore totali del job: somma (cpu * 5/60) su tutte le righe
# - Per la durata: min(start_time) a max(end_time) come prima
query_usage = """
    SELECT
        collection_id,
        MIN(start_time) AS start_time,
        MAX(end_time) AS end_time,
        SUM(average_usage.cpus * 5.0 / 60.0) AS actual_cpu_hours,
        SUM(average_usage.memory * 5.0 / 60.0) AS actual_mem_hours
    FROM `google.com:google-cluster-data.clusterdata_2019_a.instance_usage`
    WHERE collection_id IN UNNEST(@job_ids)
    GROUP BY collection_id
"""
usage_df = client.query(query_usage, job_config=job_config).to_dataframe()
usage_df.to_csv("data/raw/raw_usage_30k.csv", index=False)
print(f" -> Downloaded {len(usage_df)} aggregated jobs (Usage).")

print("\nSampling completed successfully.")
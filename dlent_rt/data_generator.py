import os
import numpy as np
import pandas as pd


class DataGenerator:
    """
    Preprocesses Google Cluster Trace v3 data into a clean workload dataset
    
    Input:  raw_events_30k.csv 
            raw_usage_30k.csv  
    Output: one row per job (collection_id) with request, actual usage, duration
    """
    
    def __init__(self):
        self.raw_jobs = None
        self.processed_jobs = None

    def load_google_traces(
        self, events_path: str, usage_path: str, sample_frac: float = 1.0
    ) -> pd.DataFrame:
        
        print("Loading events...")
        events_df = pd.read_csv(events_path, usecols=[
            'collection_id', 'priority', 'scheduling_class',
            'resource_request_cpus', 'resource_request_ram'
        ])
        events_df['collection_id'] = pd.to_numeric(
            events_df['collection_id'], errors='coerce'
        ).astype('Int64')
        
        # Events are ALREADY aggregated per collection in the fetch step
        # (one row per collection_id, deduplicated by first event per task).
        # No further groupby needed here.
        events_agg = events_df.copy()
        
        if sample_frac < 1.0:
            events_agg = events_agg.sample(frac=sample_frac, random_state=42)

        print("Loading usage...")
        usage_df = pd.read_csv(usage_path, usecols=[
            'collection_id', 'start_time', 'end_time',
            'actual_cpu_hours', 'actual_mem_hours'
        ])
        usage_df['collection_id'] = pd.to_numeric(
            usage_df['collection_id'], errors='coerce'
        ).astype('Int64')
        
        # Duration in hours from wall-clock span
        # Google trace timestamps are in microseconds
        usage_df['duration'] = (
            (usage_df['end_time'] - usage_df['start_time']) / (1e6 * 3600)
        )
        
        print("Merging events + usage...")
        merged = pd.merge(events_agg, usage_df, on='collection_id', how='inner')
        
        # Map raw priorities to 5 standard tiers
        # 1: Best Effort, 2: Batch, 3: Mid-tier, 4: Production, 5: Latency-Critical
        bins = [-1, 99, 115, 119, 350, float('inf')]
        labels = [1, 2, 3, 4, 5]
        merged['priority'] = pd.cut(
            merged['priority'], bins=bins, labels=labels
        ).astype(float)
        
        # Request time = earliest task start, converted to datetime
        merged['request_time'] = (
            pd.to_datetime('2019-05-01')
            + pd.to_timedelta(merged['start_time'], unit='us')
        )
        
        # Rename to clean schema
        merged = merged.rename(columns={
            'resource_request_cpus': 'CPU',
            'resource_request_ram': 'RAM',
        })
        
        # Drop rows with missing critical fields
        merged = merged.dropna(subset=['CPU', 'priority', 'duration'])
        
        # Drop intermediate columns
        merged = merged.drop(columns=['start_time', 'end_time'], errors='ignore')
        
        # Final column order
        final_cols = [
            'collection_id', 'request_time', 'priority', 'scheduling_class',
            'CPU', 'RAM', 'actual_cpu_hours', 'actual_mem_hours', 'duration'
        ]
        merged = merged[final_cols].sort_values('request_time').reset_index(drop=True)
        
        self.raw_jobs = merged
        print(f"Successfully loaded and merged {len(self.raw_jobs)} jobs.")
        return self.raw_jobs

    def generate_batch(self, N: int) -> pd.DataFrame:
        """Sample N jobs chronologically sorted. No synthesis, no energy merge."""
        if self.raw_jobs is None:
            raise ValueError("Must call load_google_traces() first.")
        
        available = len(self.raw_jobs)
        if N > available:
            print(f"Requested N={N} but only {available} available. Using all.")
            N = available
        
        batch = self.raw_jobs.sample(n=N, random_state=42).copy()
        batch = batch.sort_values('request_time').reset_index(drop=True)
        
        self.processed_jobs = batch
        print(f"Batch of {len(batch)} jobs ready.")
        return batch


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    events_path = os.path.join(base_dir, "data", "raw", "raw_events_30k.csv")
    usage_path = os.path.join(base_dir, "data", "raw", "raw_usage_30k.csv")

    if not os.path.exists(events_path) or not os.path.exists(usage_path):
        print("ERROR: Raw CSV files not found. Run fetch_30ksample.py first.")
    else:
        gen = DataGenerator()
        gen.load_google_traces(events_path, usage_path)
        
        print("\n--- Raw Dataset Summary ---")
        print(gen.raw_jobs.describe())
        print(f"\nColumns: {gen.raw_jobs.columns.tolist()}")
        
        batch = gen.generate_batch(N=30000)
        batch.to_csv(os.path.join(base_dir, "data", "batch_may2019_30k.csv"), index=False)
        print(f"\nSaved batch to data/batch_may2019_30k.csv")
        print(batch.head(5))
# DLENT-RT

Online admission of cloud jobs with a learned LP oracle (DLENT, Zhang & Cheung 2025) on the Google cluster trace
2019, cell a, with an additive dollar reward (customer value, profit, carbon).

## How the pipeline works now

1. **Fetch** (`scripts/data/fetch_v2.py`, BigQuery): 30,000 jobs of cell a, chosen uniformly by a fixed hash.
   Only jobs that start and end inside the month. Times are in UTC. Raw output goes to `data/raw/v2/`.
2. **Build** (`scripts/data/build_v2.py`): one row per job in `data/batch_v2_30k.csv`. The request is the SUM of the
   instances' first requests, in machines (1.0 = the largest machine). The file also holds the real CPU-hours.
3. **Data** (`dlent_rt/data.py`, `load_dataset`):
   * jobs larger than R_max machines are removed;
   * each job has one effective duration (real CPU-hours / request), cut at d_max, used everywhere;
   * the cluster is the demand level exceeded 25% of the time;
   * the first n0 jobs are H, the rest is the online stream.
4. **Phase 0** (`dlent_rt/phase0.py`):
   * bids come from the known lognormal markup;
   * energy, price and carbon are read at each job's UTC hour;
   * job types are built from bid bin × priority × hardware profile; the type volume is the mean real request ×
     the mean duration;
   * the carbon price s_C is frozen; the LP-RS gives τ.
5. **Online model** (`dlent_rt/models/theory.py`, run with `dlent_rt/models/runner.py`): DLENT-RT with the
   physical gate, the thinning, the three nonstationarity tests and ReInit.
6. **Benchmarks** (`dlent_rt/benchmarks.py`, `Benchmarks.ladder`): job MILP, bucketed MILP, fluid OPT (also on
   feasible types), G1, G2, static Phase 0 (± thinning) and DLENT-RT, all scored with the same R_job.
7. **Config:** `configs/default.yaml` (one config; the week-32 scope is marked "provisional"). **Tests:**
   `dlent_venv/bin/python -m pytest -q`.

The machine size (64 cores, 256 GB) is a declared assumption, used only for bids and energy. Everything before
week 32 is in `archive/`, and the git tag `pre-cleanup-week32` rebuilds it, with the old data files in
`data/archive/`.

# Week 30: hardware-profile snapping of the resource requests

**Source:** `notebooks/src/snapping_check.py`; CSV `tables/snapping_check.csv`.

## Which A the model uses: the snapped profile value, everywhere

* `dlent_rt/data.py:138–143` (`_build_job_arrays`): `A_raw` is the CSV request, `hw_idx = catalogue.assign(A_raw)`,
  then **`A = catalogue.centroids[hw_idx]`**. `JobArrays.A` is the snapped value.
* Every downstream use reads `JobArrays.A`:
  * capacity sizing (`_size_capacity` basis), the `drop_jobs_exceeding_capacity` filter, a_max and ξ;
  * the physical gate (`stream.A` in `policy.simulate` → `Occupancy.fits`) and Test 1's window;
  * the energy model, through `_compute_job_energy(jobs)` with `jobs.A[:, 0], jobs.A[:, 1]` (`phase0.py:76`), so
    C_elec and K, and hence **R_job**;
  * the bid cost base, through `_synthesize_bids` with `jobs.A` (`phase0.py:67`), so v, φ_job and R_job;
  * the type volume v = A_type·d̄, where A_type is the profile centroid.
* `A_raw` is used only to choose the profile. No quantity in the reward or the gate uses the original request.

## Size of the change

Population: the post-outlier H + stream jobs of the committed default config (25,308 jobs). This is the set the
catalogue is fitted on (`hardware.fit_on = full`). Relative change = |A_snapped − A_raw| / A_raw. A depends on
n_profiles only; k* is the bid grid. Checked: the recomputed snapped A equals the model's `H.A` and `stream.A`
exactly.

| grid | profiles | CPU q0.5 | CPU q0.9 | CPU q0.99 | CPU > 5% | RAM q0.5 | RAM q0.9 | RAM q0.99 | RAM > 5% | any resource > 5% |
|---|---|---|---|---|---|---|---|---|---|---|
| k* 128, 32 profiles (default) | 32 | 15.6% | 39.4% | 76.1% | 82.6% | 17.4% | 43.8% | 84.7% | 76.4% | **95.1%** |
| k* 64, 16 profiles | 16 | 25.7% | 63.2% | 165.3% | 90.0% | 21.3% | 60.5% | 162.2% | 92.2% | **99.0%** |

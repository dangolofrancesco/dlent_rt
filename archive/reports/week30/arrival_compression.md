# Week 30: arrival-time compression (`time.arrival_time_scale`)

**Code:** `dlent_rt/config.py` (`TimeCfg.arrival_time_scale`, default 1.0 = off), `dlent_rt/data.py`
(`_compress_arrival_steps`, `load_dataset`). Test: `tests/test_arrival_compression.py`.
**Check script:** `notebooks/src/arrival_compression_check.py`, CSV `tables/arrival_compression_check.csv`.

## What the option does (k ≠ 1)

* **Arrivals.** Every job's absolute arrival tick, in H and in the stream on their shared timeline, becomes
  floor(tick / k). The order, the durations (in ticks) and the job set are unchanged; no job is added or removed.
* **Collisions.** The event loop allows one arrival per tick: `register_arrival` asserts t = last + 1, and Test 2
  stores the reward per tick (`self.R[t]`), so a second arrival on the same tick would overwrite it. Compression
  therefore applies the trace's existing collision rule from `_assign_real_tick_steps`: a job landing on an occupied
  tick moves to the next free tick. This keeps the order, and keeps H strictly before the stream.
* **Capacity is sized on the ORIGINAL ticks**, so c is the same cluster for every k and only the arrival intensity
  changes. Re-sizing c on the compressed ticks would scale c with the intensity (shown in the intensity report).
* **d_max** (and its 0.05·T ceiling) is computed on the ORIGINAL horizon, so that durations are not clipped by a
  ceiling that shrinks with T. On the real trace the ceiling would not bind anyway for k ≤ 4 (0.05·T ≥ 17,911 > 14,304).
  On the synthetic test trace it did bind at k = 3, which is why the original horizon is used.
* **Off (k = 1)** leaves every array as before: the synthetic and real-trace regression snapshots are bit-identical,
  and the fast suite passes (55; checked on a clean worktree at HEAD plus this change).

## Effect on the real trace

Config: capacity-scan f = 0.2 (q = 0.7, filter on, default grid).

| k | T | d_max | d_max / T | H span | arrivals moved by a collision (H + stream) | max move (ticks) | c (CPU, RAM) | full timeout phases before T | timeout phase ends |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1,432,910 | 14,304 | 0.0100 | 1,245,441 | 0 | 0 | 0.5352, 0.3488 | 5 | 28,608; 85,824; 200,256; 429,120; 886,848 |
| 2 | 716,455 | 14,304 | 0.0200 | 622,721 | 731 | 6 | same | 4 | 28,608; 85,824; 200,256; 429,120 |
| 3 | 477,637 | 14,304 | 0.0299 | 415,147 | 1,131 | 9 | same | 4 | 28,608; 85,824; 200,256; 429,120 |
| 4 | 358,228 | 14,304 | 0.0399 | 311,361 | 1,373 | 10 | same | 3 | 28,608; 85,824; 200,256 |

* The timeout schedule is d_max·(2^(m+1) − 2), with W⁽⁰⁾ = d_max, assuming every phase times out. It does not depend
  on k, so a shorter T leaves fewer full doubling phases.
* d_max / T stays below the 0.10 abort threshold for every k.
* |H| = 8,577 and |stream| = 14,331 at every k.

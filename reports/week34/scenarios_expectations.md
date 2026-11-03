# Experiment — Non-stationary bids (S1, S2, S6)

Why: on our data the environment barely changes, so learning has little to gain and Tests 2 and 3 never fire. These scenarios change the customers' bids during the stream to test learning where it should matter. Experiment only; the theory does not change.

Setting (piecewise stationary): only the stream's markup distribution changes, once, at t* = 240 h after the stream start. Before: ln m ~ N(μ0, σ0²), μ0 = −σ0²/2, σ0 = 0.3. After: ln m ~ N(μ1, σ1²). b = c·m; c, arrivals, requests, durations, prices, carbon and H never change. One change point keeps the total variation V = o(T), as DLENT requires.

Detection: the change falls inside a phase. Test 2 fires when realised reward falls too far below the phase's LP benchmark (market worse); Test 3 when the LP value on recent data rises too far above the bound (market better); Test 1 or a timeout may end the phase first. Test 2 averages the deficit since the phase start, so pre-change data dilute it; larger changes fire sooner.

ReInit: everything is re-estimated on the ended phase's data (p̂, μ̂, σ̂, column durations/costs from admitted jobs, LP, τ). The detecting phase mixes pre- and post-change data, so full adaptation usually takes two phases. The practical model re-estimates μ and σ every phase and adapts; the no-learning model keeps Phase 0's values. With stats by end reason, Test 2/3 → replace (right after a change); Test 1 → merge (wrong after a change).

Scenarios:
- S1 bids fall: m → 0.5·m (μ1 = μ0 + ln 0.5, σ1 = σ0). Expected: with stale estimates the model reads markups ~0.5, below the reserve markup 0.715 (φ = 0), so almost all jobs look negative, k* rises, it rejects almost everything, reward collapses → Test 2 fires. After ReInit μ̂ moves to the new level and admissions recover. No learning never recovers. Learning wins clearly (largest gap). Side effect: halved bids below H's smallest bid fall into bin 1 (never admitted), a limit of the fixed grid.
- S2 bids rise: m → 1.75·m. Expected: stale estimates make every job look valuable, the model over-admits, τ too low, cluster fills with good-not-best jobs. Test 3 should fire, but Test 1 may fire first (then by-end-reason merges old and new statistics, slowing adaptation). After ReInit μ̂ and τ rise. Smaller gain than S1.
- S6 bids spread: σ1 = 0.5, mean markup still 1. Expected: φ is lower at the same markup (m = 1: 0.67 at σ 0.3, 0.48 at σ 0.5); the stale model overestimates φ and admits bins that are too low. Reward falls less than S1, so Test 2 may not fire; the error shows in the Myerson check (expected profit > realised). After ReInit σ̂ → 0.5. Moderate gain; clear gap practical vs no learning.

Summary: S1 → Test 2, learning wins by far; S2 → Test 3 (or Test 1), learning wins slightly; S6 → maybe no test, learning wins moderately, visible in Myerson.

Models in every environment: fluid OPT; Practical (C3); Greedy-Pos; No learning (practical frozen at Phase 0); Theoretical. Practical vs No learning = value of learning; Practical vs Theoretical = value of the practical changes.

Metrics: detection delay; reward before/after t*; % of fluid OPT after t*; μ̂, σ̂, τ per phase; expected vs realised profit; share of post-change bids in bin 1.

# Headroom under uncertainty (line B) -- PRELIMINARY, one scenario

Question: in a setting where the vehicle does not know the density along its track, how much range separates a
well-tuned simple feedback strategy (receding-horizon re-planning, MPC) from a clairvoyant plan? That gap bounds
what ANY non-clairvoyant strategy, learned or not, can still gain. Code: `src/certified_flight/headroom.py`,
`scripts/headroom_study.py`, tests `tests/test_headroom.py`.

## Model and honesty notes
- Same exact point-mass dynamics and path limits as `bound.py`, polar = lower hull of all 298 geometries (no friction),
  coarse parametric time grid. Nominal range 2511.09 km vs the 480-step relaxed bound 2511.03 km.
- Truth: density bias k_rho(s) along the track (AR(1) field, sigma 6 %, correlation length 200 km), constant lift and
  drag biases (k_L = 1.01, k_D = 1.039). Sensors are ideal (dynamic pressure and accelerations measured exactly), which
  favours MPC, so the gaps below are conservative.
- Hard limits (q 80 kPa, n 3 g, heat flux) are checked at every 0.5 s RK stage of the truth; a run with any violation is
  reported infeasible and is NOT comparable on range.
- MPC: re-plans every 15 s, load biases estimated online (they match the truth to 0.1 % after 15 s), first three steps have
  elastic limits, a failed solve is retried once with default IPOPT, then the previous plan is kept.
  3 to 5 of 80 re-plans still fail in every run: the baseline is not yet fully robust, and feasibility is not monotone in
  the back-off b (b = 0.08 with reversion violates, 0.06 and 0.12 do not).
- Strategy "reversion" forecasts ln k_rho decaying with the true correlation length: MPC that knows the statistics.

## Result, scenario sigma 0.06, L 200 km, seed 0 (declared rule: tuned MPC = feasible run with the largest range)

| strategy | range [km] | feasible |
|---|---|---|
| clairvoyant (knows the field and the biases) | 2493.57 | yes |
| nominal plan flown open loop | 2498.51 | no (q exceeds by 18 %) |
| MPC, persistence forecast, best feasible b = 0.10 | 2456.06 | yes |
| MPC, persistence, b <= 0.06 | 2460 to 2468 | no |
| MPC, reversion forecast, best feasible b = 0.06 | 2465.20 | yes |

Gaps to the clairvoyant plan: 37.5 km (1.5 %) for persistence MPC, 28.4 km (1.1 %) for reversion MPC. Knowing the statistics
of the disturbance is worth 9 km; the rest is the price of not seeing the future, which no strategy removes.

## What this does and does not show
- The nominal-state setting leaves about 2 km for learning; with unknown density the spread between simple strategies is
  tens of km, so the uncertain setting is where a learned policy can matter at all.
- It does NOT show that RL can capture it: the clairvoyant bound is unreachable, one scenario is not a statistic, and
  a better simple method (faster re-planning, adaptive back-off) may close more of the gap.
- Next: several seeds and sigma/L settings, finer b grid, replan period 5 s, adaptive back-off; only then an RL comparison.

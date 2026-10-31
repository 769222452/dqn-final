# Held-out-2 evaluation: results under the pre-registered criteria

40 initial states of the test set held-out-2; reference: constant CL 0.025. Gains in km of range at 1200 s; 95% CI = paired percentile bootstrap of the mean (10 000 resamples).

## Primary H1: for every seed of the final method, mean gain > 0 with CI lower bound > 0, both over all states and over the states where the reference completed

- d64_seed17: all states -56.848 km [-177.495, +3.921]; reference completed (40 states) -56.848 km [-177.495, +3.921] -> FAIL
- d64_seed23: all states +3.890 km [+3.128, +4.689]; reference completed (40 states) +3.890 km [+3.128, +4.689] -> PASS
- d64_seed41: all states -0.592 km [-1.298, +0.138]; reference completed (40 states) -0.592 km [-1.298, +0.138] -> FAIL

**H1 overall: FAIL**

## S1: wins >= 33/40 and worst gain >= -1 km, per seed of the final method

- d64_seed17: wins 34/40, worst -2399.138 km -> FAIL
- d64_seed23: wins 39/40, worst -0.233 km -> PASS
- d64_seed41: wins 15/40, worst -5.286 km -> FAIL

## S2: final method (64-64-64-64) minus previous version (16-16), seed-averaged range per state

- mean -22.825 km, CI [-63.250, -2.371], final better in 1/40, worst -804.925 km -> FAIL (the final method is not changed by this outcome)

## S3: one-decision probes, mean A_closed >= mean A_open (1e-4 return units)

- d64_seed17: A_closed -1.262, A_open -1.928 (5/14 pairs) -> PASS
- d64_seed23: A_closed -1.364, A_open -1.386 (6/14 pairs) -> PASS
- d64_seed41: A_closed -3.043, A_open -4.770 (8/14 pairs) -> PASS

## Descriptive (not criteria)

| policy | mean gain km | 95% CI | gain, reference completed | wins | worst km | median km | completed | nominal km |
|---|---|---|---|---|---|---|---|---|
| d64_seed17 | -56.848 | [-177.495, +3.921] | -56.848 [-177.495, +3.921] | 34/40 | -2399.138 | +2.507 | 39/40 | 2449.11 |
| d64_seed23 | +3.890 | [+3.128, +4.689] | +3.890 [+3.128, +4.689] | 39/40 | -0.233 | +3.703 | 40/40 | 2450.97 |
| d64_seed41 | -0.592 | [-1.298, +0.138] | -0.592 [-1.298, +0.138] | 15/40 | -5.286 | -0.827 | 40/40 | 2446.88 |
| b16_seed17 | +4.679 | [+3.697, +5.741] | +4.679 [+3.697, +5.741] | 40/40 | +0.096 | +3.681 | 40/40 | 2448.70 |
| b16_seed23 | +5.228 | [+4.344, +6.197] | +5.228 [+4.344, +6.197] | 40/40 | +1.541 | +4.426 | 40/40 | 2450.56 |
| b16_seed41 | +5.018 | [+4.241, +5.819] | +5.018 [+4.241, +5.819] | 40/40 | +1.118 | +4.504 | 40/40 | 2452.36 |
| old_seed17 | -1.341 | [-2.789, +0.123] | -1.341 [-2.789, +0.123] | 17/40 | -10.437 | -0.878 | 40/40 | 2448.45 |

Reference completed 40/40 flights.

Paired differences (first policy d64_seed17 minus each other one):

- d64_seed17 - d64_seed23: -60.738 km [-181.517, +0.042], first better in 15/40
- d64_seed17 - d64_seed41: -56.256 km [-177.153, +4.685], first better in 33/40
- d64_seed17 - b16_seed17: -61.527 km [-182.323, -0.786], first better in 10/40
- d64_seed17 - b16_seed23: -62.076 km [-182.712, -1.427], first better in 7/40
- d64_seed17 - b16_seed41: -61.866 km [-182.510, -1.255], first better in 7/40
- d64_seed17 - old_seed17: -55.507 km [-176.709, +5.642], first better in 36/40

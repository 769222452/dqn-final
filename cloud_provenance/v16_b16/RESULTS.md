# Held-out-3 evaluation: results under the pre-registered criteria

40 initial states of the test set held-out-3; reference: constant CL 0.025. Ranges and gains in km at 1200 s; 95% CI = paired percentile bootstrap of the mean (10 000 resamples, seed 0). A failed flight counts with its truncated range.

## Primary H1: for every seed of the final method, mean gain > 0 with CI lower bound > 0, both over all states and over the states where the reference completed

- final_seed17: all states +3.070 km [+2.327, +3.829]; reference completed (40 states) +3.070 km [+2.327, +3.829] -> PASS
- final_seed23: all states +5.509 km [+4.935, +6.099]; reference completed (40 states) +5.509 km [+4.935, +6.099] -> PASS
- final_seed41: all states +5.958 km [+4.929, +7.010]; reference completed (40 states) +5.958 km [+4.929, +7.010] -> PASS
- final_seed59: all states -52.923 km [-174.281, +8.438]; reference completed (40 states) -52.923 km [-174.281, +8.438] -> FAIL
- final_seed83: all states +8.445 km [+7.580, +9.320]; reference completed (40 states) +8.445 km [+7.580, +9.320] -> PASS
- final_seed97: all states +8.140 km [+6.963, +9.338]; reference completed (40 states) +8.140 km [+6.963, +9.338] -> PASS
- final_seed101: all states +9.870 km [+8.739, +11.024]; reference completed (40 states) +9.870 km [+8.739, +11.024] -> PASS

**H1 overall: FAIL** (6/7 seeds pass)

## Primary H2: reference policy, single post regressor minus dual-routing post surrogate, range per state

- mean +60.992 km, CI [-0.018, +182.453], single better in 24/40, dual better in 16/40; completed single 40/40, dual 39/40; nominal single 2443.319 km, dual 2443.479 km

**H2: FAIL**

Descriptive, bc_seed17: single minus dual +62.276 km [+0.871, +184.462], single better in 31/40; completed single 40/40, dual 39/40.

## S1: wins >= 33/40 and worst gain >= -1 km, per seed of the final method

- final_seed17: wins 36/40, worst -2.890 km -> FAIL
- final_seed23: wins 40/40, worst +1.596 km -> PASS
- final_seed41: wins 39/40, worst -0.459 km -> PASS
- final_seed59: wins 37/40, worst -2407.185 km -> FAIL
- final_seed83: wins 40/40, worst +3.382 km -> PASS
- final_seed97: wins 40/40, worst +1.327 km -> PASS
- final_seed101: wins 40/40, worst +3.025 km -> PASS

## S2: final method minus the same seeds trained with the dual-routing surrogate (both flown with the single post regressor), seed-averaged range per state

- mean +1.442 km, CI [-24.416, +27.320], final better in 36/40, worst -343.652 km -> FAIL

## S3: one-decision probes, mean A_closed >= mean A_open (1e-4 return units)

- final_seed17: A_closed -2.473, A_open -1.084 (2/14 pairs) -> FAIL
- final_seed23: A_closed -2.841, A_open -2.865 (11/14 pairs) -> PASS
- final_seed41: A_closed -1.136, A_open -1.146 (8/14 pairs) -> PASS
- final_seed59: A_closed -0.715, A_open -1.161 (14/14 pairs) -> PASS
- final_seed83: A_closed -1.653, A_open -2.552 (13/14 pairs) -> PASS
- final_seed97: A_closed -2.516, A_open -3.135 (9/14 pairs) -> PASS
- final_seed101: A_closed -2.155, A_open -3.752 (13/14 pairs) -> PASS

## Descriptive (not criteria)

| policy | mean gain km | 95% CI | gain, reference completed | wins | worst km | median km | completed | nominal km |
|---|---|---|---|---|---|---|---|---|
| final_seed17 | +3.070 | [+2.327, +3.829] | +3.070 [+2.327, +3.829] | 36/40 | -2.890 | +3.161 | 40/40 | 2448.17 |
| final_seed23 | +5.509 | [+4.935, +6.099] | +5.509 [+4.935, +6.099] | 40/40 | +1.596 | +5.472 | 40/40 | 2451.00 |
| final_seed41 | +5.958 | [+4.929, +7.010] | +5.958 [+4.929, +7.010] | 39/40 | -0.459 | +5.584 | 40/40 | 2450.44 |
| final_seed59 | -52.923 | [-174.281, +8.438] | -52.923 [-174.281, +8.438] | 37/40 | -2407.185 | +7.101 | 39/40 | 2452.06 |
| final_seed83 | +8.445 | [+7.580, +9.320] | +8.445 [+7.580, +9.320] | 40/40 | +3.382 | +8.659 | 40/40 | 2453.73 |
| final_seed97 | +8.140 | [+6.963, +9.338] | +8.140 [+6.963, +9.338] | 40/40 | +1.327 | +7.114 | 40/40 | 2450.85 |
| final_seed101 | +9.870 | [+8.739, +11.024] | +9.870 [+8.739, +11.024] | 40/40 | +3.025 | +9.166 | 40/40 | 2452.60 |
| b16_seed17 | +6.199 | [+5.335, +7.082] | +6.199 [+5.335, +7.082] | 40/40 | +1.597 | +5.904 | 40/40 | 2449.19 |
| b16_seed23 | +6.187 | [+5.105, +7.329] | +6.187 [+5.105, +7.329] | 39/40 | -0.423 | +5.352 | 40/40 | 2446.97 |
| b16_seed41 | +5.768 | [+4.630, +6.978] | +5.768 [+4.630, +6.978] | 40/40 | +0.112 | +5.272 | 40/40 | 2445.24 |
| b16_seed59 | +2.126 | [+1.107, +3.086] | +2.126 [+1.107, +3.086] | 32/40 | -6.777 | +2.621 | 40/40 | 2444.79 |
| b16_seed83 | +6.987 | [+5.989, +8.034] | +6.987 [+5.989, +8.034] | 40/40 | +0.440 | +6.622 | 40/40 | 2449.51 |
| b16_seed97 | -54.993 | [-176.757, +6.558] | -54.993 [-176.757, +6.558] | 38/40 | -2416.048 | +4.805 | 39/40 | 2448.45 |
| b16_seed101 | +5.702 | [+4.660, +6.798] | +5.702 [+4.660, +6.798] | 40/40 | +0.317 | +5.000 | 40/40 | 2447.73 |

Seed-averaged range of the final method over the 40 states: 2456.388 km.
Reference completed 40/40 flights.

Paired differences (first policy final_seed17 minus each other one):

- final_seed17 - final_seed23: -2.440 km [-3.281, -1.624], first better in 6/40
- final_seed17 - final_seed41: -2.888 km [-3.850, -1.947], first better in 9/40
- final_seed17 - final_seed59: +55.993 km [-5.478, +177.500], first better in 6/40
- final_seed17 - final_seed83: -5.375 km [-6.148, -4.630], first better in 0/40
- final_seed17 - final_seed97: -5.070 km [-6.304, -3.876], first better in 2/40
- final_seed17 - final_seed101: -6.801 km [-7.829, -5.761], first better in 2/40
- final_seed17 - b16_seed17: -3.129 km [-4.081, -2.202], first better in 7/40
- final_seed17 - b16_seed23: -3.117 km [-4.139, -2.135], first better in 7/40
- final_seed17 - b16_seed41: -2.698 km [-3.769, -1.637], first better in 11/40
- final_seed17 - b16_seed59: +0.943 km [-0.078, +1.973], first better in 25/40
- final_seed17 - b16_seed83: -3.917 km [-4.833, -3.027], first better in 5/40
- final_seed17 - b16_seed97: +58.063 km [-3.504, +179.902], first better in 15/40
- final_seed17 - b16_seed101: -2.632 km [-3.554, -1.723], first better in 9/40

# Improvement attribution report

## Arms

| arm | policy | states | completed | mean range km | mean heat load MJ/m² | median intra-step ptp ° | mean cruise q kPa | non-surrogate decisions | mean gap to bound km |
|---|---|---|---|---|---|---|---|---|---|
| aligned@minimal_first | const0.025 | 21 | 21 | 2511.93 | 1607.4 | 0.009 | 78.8 | 34 | 13.06 (6) |
| aligned@minimal_first | seed101 | 21 | 21 | 2512.01 | 1597.2 | 0.010 | 78.5 | 12 | 12.70 (6) |
| aligned@minimal_first | train_blocks_seed17 | 21 | 21 | 2513.66 | 1609.7 | 0.018 | 79.2 | 17 | 11.19 (6) |
| aligned@minimal_first | train_pi_seed17 | 21 | 21 | 2514.04 | 1607.1 | 0.008 | 78.8 | 24 | 10.45 (6) |
| consistent | const0.025 | 21 | 21 | 2491.88 | 1496.4 | 0.019 | 71.0 | 49 | 33.54 (6) |
| consistent | seed101 | 21 | 21 | 2490.96 | 1491.1 | 0.020 | 70.9 | 10 | 34.78 (6) |
| consistent@minimal_first | const0.025 | 21 | 21 | 2496.00 | 1505.0 | 0.020 | 70.9 | 61 | 29.70 (6) |
| consistent@minimal_first | seed101 | 21 | 21 | 2495.22 | 1499.5 | 0.019 | 71.0 | 10 | 30.39 (6) |
| v16 | const0.025 | 21 | 21 | 2455.38 | 1369.8 | 1.225 | 63.5 | 49 | 70.52 (6) |
| v16 | seed101 | 21 | 21 | 2465.11 | 1398.6 | 1.169 | 64.6 | 18 | 59.04 (6) |
| v16@minimal_first | const0.025 | 21 | 21 | 2456.42 | 1372.4 | 1.259 | 63.4 | 49 | 69.04 (6) |
| v16@minimal_first | seed101 | 21 | 21 | 2467.56 | 1401.5 | 1.176 | 64.4 | 18 | 57.03 (6) |

## Declared comparisons (paired by initial state; a − b)

| comparison | kind | states | mean km [95 % CI] | wins | worst km | completed a/b | Δ heat load MJ/m² | gap closure |
|---|---|---|---|---|---|---|---|---|
| AoA fix: consistent − V16 (const 0.025) | execution | 21 | +36.50 [+35.34, +37.70] | 21 | +32.71 | 21/21 | +126.6 | 0.52 (6) |
| Minimal-first look-ahead: consistent@mf − consistent (const) | execution | 21 | +4.12 [+3.21, +5.01] | 21 | +0.14 | 21/21 | +8.6 | 0.11 (6) |
| Heat-load alignment: aligned@mf − consistent@mf (const) | execution | 21 | +15.93 [+14.69, +17.12] | 21 | +9.69 | 21/21 | +102.4 | 0.56 (6) |
| E1 total execution layer: aligned@mf − V16 (const) | execution | 21 | +56.55 [+54.78, +58.32] | 21 | +49.97 | 21/21 | +237.6 | 0.81 (6) |
| Minimal-first alone on V16: v16@mf − v16 (const) | execution | 21 | +1.04 [+0.08, +2.02] | 12 | -1.85 | 21/21 | +2.6 | 0.02 (6) |
| RL transfer seed17: policy − const on V16 layer (archived result) | rl | 0 | - | - | - | 0/0 | - | - |
| RL transfer seed17: policy − const on aligned@mf | rl | 0 | - | - | - | 0/0 | - | - |
| System seed17: aligned@mf − V16 (same policy) | system | 0 | - | - | - | 0/0 | - | - |
| RL transfer seed23: policy − const on V16 layer (archived result) | rl | 0 | - | - | - | 0/0 | - | - |
| RL transfer seed23: policy − const on aligned@mf | rl | 0 | - | - | - | 0/0 | - | - |
| System seed23: aligned@mf − V16 (same policy) | system | 0 | - | - | - | 0/0 | - | - |
| RL transfer seed41: policy − const on V16 layer (archived result) | rl | 0 | - | - | - | 0/0 | - | - |
| RL transfer seed41: policy − const on aligned@mf | rl | 0 | - | - | - | 0/0 | - | - |
| System seed41: aligned@mf − V16 (same policy) | system | 0 | - | - | - | 0/0 | - | - |
| RL transfer seed59: policy − const on V16 layer (archived result) | rl | 0 | - | - | - | 0/0 | - | - |
| RL transfer seed59: policy − const on aligned@mf | rl | 0 | - | - | - | 0/0 | - | - |
| System seed59: aligned@mf − V16 (same policy) | system | 0 | - | - | - | 0/0 | - | - |
| RL transfer seed83: policy − const on V16 layer (archived result) | rl | 0 | - | - | - | 0/0 | - | - |
| RL transfer seed83: policy − const on aligned@mf | rl | 0 | - | - | - | 0/0 | - | - |
| System seed83: aligned@mf − V16 (same policy) | system | 0 | - | - | - | 0/0 | - | - |
| RL transfer seed97: policy − const on V16 layer (archived result) | rl | 0 | - | - | - | 0/0 | - | - |
| RL transfer seed97: policy − const on aligned@mf | rl | 0 | - | - | - | 0/0 | - | - |
| System seed97: aligned@mf − V16 (same policy) | system | 0 | - | - | - | 0/0 | - | - |
| RL transfer seed101: policy − const on V16 layer (archived result) | rl | 21 | +9.73 [+7.98, +11.63] | 21 | +3.03 | 21/21 | +28.8 | 0.16 (6) |
| RL transfer seed101: policy − const on aligned@mf | rl | 21 | +0.08 [-0.73, +0.84] | 13 | -3.37 | 21/21 | -10.2 | 0.03 (6) |
| System seed101: aligned@mf − V16 (same policy) | system | 21 | +46.90 [+46.23, +47.55] | 21 | +43.63 | 21/21 | +198.6 | 0.78 (6) |
| E2 RL gain, retrained PI seed17 − const (aligned@mf) | rl | 21 | +2.10 [+1.68, +2.46] | 20 | -0.49 | 21/21 | -0.3 | 0.20 (6) |
| E3 coordinated blocks seed17 − PI seed17 (aligned@mf, equal budget) | algorithm | 21 | -0.38 [-0.79, +0.05] | 5 | -2.21 | 21/21 | +2.6 | -0.07 (6) |
| E2 RL gain, retrained PI seed23 − const (aligned@mf) | rl | 0 | - | - | - | 0/0 | - | - |
| E3 coordinated blocks seed23 − PI seed23 (aligned@mf, equal budget) | algorithm | 0 | - | - | - | 0/0 | - | - |
| E2 RL gain, retrained PI seed41 − const (aligned@mf) | rl | 0 | - | - | - | 0/0 | - | - |
| E3 coordinated blocks seed41 − PI seed41 (aligned@mf, equal budget) | algorithm | 0 | - | - | - | 0/0 | - | - |
| E2 RL gain, retrained PI seed59 − const (aligned@mf) | rl | 0 | - | - | - | 0/0 | - | - |
| E3 coordinated blocks seed59 − PI seed59 (aligned@mf, equal budget) | algorithm | 0 | - | - | - | 0/0 | - | - |
| E2 RL gain, retrained PI seed83 − const (aligned@mf) | rl | 0 | - | - | - | 0/0 | - | - |
| E3 coordinated blocks seed83 − PI seed83 (aligned@mf, equal budget) | algorithm | 0 | - | - | - | 0/0 | - | - |
| E2 RL gain, retrained PI seed97 − const (aligned@mf) | rl | 0 | - | - | - | 0/0 | - | - |
| E3 coordinated blocks seed97 − PI seed97 (aligned@mf, equal budget) | algorithm | 0 | - | - | - | 0/0 | - | - |
| E2 RL gain, retrained PI seed101 − const (aligned@mf) | rl | 0 | - | - | - | 0/0 | - | - |
| E3 coordinated blocks seed101 − PI seed101 (aligned@mf, equal budget) | algorithm | 0 | - | - | - | 0/0 | - | - |
| Issue 1, exact overrides seed17 − PI seed17 (aligned@mf) | algorithm | 0 | - | - | - | 0/0 | - | - |
| Issue 1, exact overrides seed23 − PI seed23 (aligned@mf) | algorithm | 0 | - | - | - | 0/0 | - | - |
| Issue 1, exact overrides seed41 − PI seed41 (aligned@mf) | algorithm | 0 | - | - | - | 0/0 | - | - |
| Issue 1, exact overrides seed59 − PI seed59 (aligned@mf) | algorithm | 0 | - | - | - | 0/0 | - | - |
| Issue 1, exact overrides seed83 − PI seed83 (aligned@mf) | algorithm | 0 | - | - | - | 0/0 | - | - |
| Issue 1, exact overrides seed97 − PI seed97 (aligned@mf) | algorithm | 0 | - | - | - | 0/0 | - | - |
| Issue 1, exact overrides seed101 − PI seed101 (aligned@mf) | algorithm | 0 | - | - | - | 0/0 | - | - |

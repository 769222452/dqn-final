# Pre-registration: final method (64-64-64-64 advantage networks) on the second held-out test set

Registered before the final method was trained and before any flight from the held-out-2 initial states or at
the held-out-2 probe times was run. After registration, none of the following is changed: the code, the
training command, the test set, the criteria, the analysis. All outcomes are reported.

## Background and why a second test set

- The first held-out test (registered 2026-10-23 09:59, results 11:45) evaluated B+C with 4-16-16-19 networks.
  All primary criteria passed.
- Afterwards, an offline study on the label archives of the training runs (forward chaining: fit on the labels
  available before an iteration, predict the labels of that iteration; 17 network sizes, 3 seeds, 62
  iterations) showed that deeper networks rank actions better (Spearman 0.53 for 16-16, 0.65 for
  64-64-64-64) and that more than four hidden layers do not help further. This study used training labels
  only, not the held-out tests.
- On this basis the final method uses 64-64-64-64 hidden layers. Everything else is unchanged.
- The first held-out set has been seen by the developer (results of the 16-16 version), so the final method is
  tested on a new set, held-out-2. The decision to adopt 64-64-64-64 is not revisited whatever the outcome.

## Code

Repository snapshot `D:\dqn-final` = cloud `/hy-tmp/v14_final/repo`: 392 files, manifest SHA-256 (sorted
`sha256  path` lines of all files except .git, .cache, __pycache__) 2f10ceb3ccf480a6b798b82c665b2e51a78b4747d08ea2dddd881abb7566f33a. Identical on both machines.
Tests: 30 passed locally; on the cloud machine the certified-flight tests pass (the cross-CPU tolerance of
`test_reference_state` was corrected before registration: differences of 1e-14 relative to the component
magnitudes).

## Training (cloud, Intel Xeon E5-2683 v3, 56 CPUs)

`run_final.sh`: seeds 17, 23, 41, in parallel, each

    python3 repo/scripts/train.py --out runs/seed$s --seed $s --workers 18 --iterations 80

All other settings are the defaults of the proposed method (open-loop continuation, inputs h,V,gamma,t, no
overrides, hidden 64,64,64,64). A run ends by its stop rule or after 80 iterations; its final state is the
policy. All three seeds are reported; no seed is replaced or rerun, except a run that crashes with a software
exception, which is resumed with the same command (resumption is exact) and reported.

## Test set held-out-2 (fixed by code)

`test_set_points('held-out-2')` in `src/rollout_improvement/evaluation.py`: 40 offsets drawn uniformly from
|dh| <= 2000 m, |dV| <= 100 m/s with `default_rng([20261023, 2])`; probe times 228, 264, 312, 444, 612, 696,
1056 s drawn with `default_rng([20261023, 3])` from the 12 s grid without the development and held-out-1
times. Disjoint from the development and held-out-1 sets (unit test). 3 of the 40 states lie inside
|dh| <= 500 m, |dV| <= 20 m/s; quadrant counts 10/12/12/6. Full list: `heldout2_set.json`.

## Policies evaluated (all on the cloud machine, after training)

- d64_seed17, d64_seed23, d64_seed41: the final method (trained as above).
- b16_seed17, b16_seed23, b16_seed41: the previous version (16-16), files of the first pre-registration.
- old_seed17: the earlier method (closed-loop continuation, all inputs, overrides).
- Reference: constant CL 0.025.

## Criteria

Gain = range at 1200 s of the policy minus range of the reference from the same initial state.
95% CI = paired percentile bootstrap of the mean (10 000 resamples, seed 0). A failed flight counts with its
truncated range.

- **Primary H1**: for each of the three seeds of the final method, (a) mean gain over all 40 states > 0 with
  CI lower bound > 0, and (b) mean gain over the states where the reference completed > 0 with CI lower bound
  > 0. Supported only if all three seeds pass.
- S1: for each seed of the final method, wins >= 33/40 and worst gain >= -1 km.
- S2: per state, seed-averaged range of the final method minus seed-averaged range of the 16-16 version: mean
  > 0 with CI lower bound > 0. Reported either way; the final method is not changed by the outcome.
- S3: for each seed of the final method, mean A_closed >= mean A_open over the 14 probes.
- Descriptive: nominal range, completion counts, medians, all per-state values, the earlier method.

The probes start on the nominal trajectory, whose decision states are branched during training; they measure
the reaction to a perturbation, not performance on unseen states.

## Execution

`run_final.sh` runs training, the evaluations (`--test-set held-out-2`, shared reference flights),
`compare.py` and `analyze_heldout2.py` (writes RESULTS.md, verdict.json) without manual steps.

## Frozen files (SHA-256)

- `c9cc8a18aaa9d77587f6881b37da1174506f33f1606b1ffab32f79fd1061fec9`  run_final.sh
- `9c125c95100f1857787f011d4d43239988dbd3175abb43fd20faa1f844ae5683`  analyze_heldout2.py
- `8f131e4127e9e46d00150a8f4409a53e8bd4cc245a004b24758be71fe63c1d68`  heldout2_set.json
- `be2bf5b936fe7862ad03892fe8125e186b3538c7847d60edea2ccf70f40effd8`  policies/bc_seed17.json
- `75ba00eaf9089974608804bc00e5616c826ac568eb5c91fb4d3f34db06b29fa1`  policies/bc_seed23.json
- `45f463b56f25c9a2a0432a59bb9cf1e52b28c754916ce01e9d175ce54a6b42bd`  policies/bc_seed41.json
- `9a6b50a11caa6560c0e0efbd4b53606473f9e3874d064793160cfcbe2519e30d`  policies/old_seed17.json
- `98ea5e750c317da59737ac3bb7176d5055d99e70984b3f8e788ce6611a5354eb`  repo/src/rollout_improvement/evaluation.py
- `5aad3aec7f9a699a2c13119f4030a2224403c0544a846b28320e366c93a98a0e`  repo/src/rollout_improvement/ensemble.py
- `d2dce2bf277ae6f866053f45eb351c51a7773e48d71bd54131d00e519353d7e6`  repo/src/rollout_improvement/training.py
- `fd024198ee57faa392243836583ce354787c0e3fb1a0566b1a8cbdd03b70efb1`  repo/scripts/compare.py

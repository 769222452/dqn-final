# Pre-registration: 16-16 networks with a single post-command regressor, seven seeds, held-out-3

Registered before the final method was trained and before any flight from the 40 held-out-3 initial states or at
the held-out-3 probe times was run. After registration, none of the following is changed: the code, the
commands, the test set, the criteria, the analysis. All outcomes are reported.

## Background

- First pre-registration (held-out-1): 16-16 advantage networks (B+C), seeds 17, 23, 41: all primary criteria
  passed.
- Second pre-registration (held-out-2): 64-64-64-64 networks, chosen from an offline ranking study. H1 FAILED
  (seed17 one failed flight, -2399 km; seed41 mean -0.59 km [-1.30, +0.14]; seed23 +3.89 km). The 16-16
  policies on the same states: +4.68, +5.23, +5.02 km, 40/40 wins each.
- Exploratory post-surrogate analysis on held-out-1 (`_runtime/post_ablation`): the 929-sample post regressor
  alone flew farther than the routing between it and an 85-sample regressor (constant CL 0.025: +0.89 km
  [+0.41, +1.38]; bc_seed17: +1.48 km [+0.77, +2.21]), although the routing was more accurate offline.
- A registration of 64-64-64-64 with the single regressor (`_runtime/heldout3`) was withdrawn before any flight
  after the held-out-2 results were seen (see its NOTES.md).
- Final method registered here: 16-16 networks and the single post regressor; everything else as in the
  repository defaults (open-loop continuation, inputs h, V, gamma, t, no overrides, label decay 0.5 over 7
  policy versions). The choice of 16-16 was informed by held-out-1 and held-out-2; therefore the test set is
  held-out-3, never used. These decisions are not revisited whatever the outcome.
- Seeds: all seven seeds with converged 16-16 runs (17, 23, 41, 59, 83, 97, 101, `/hy-tmp/v13h/runs`). Seeds
  59, 83, 97, 101 were never evaluated on any held-out set.

## Surrogate

`controller.npz` unchanged (sha256 1cb8d962...): same pre head, same 929-sample post regressor, same coverage
checks; the post routing is removed in code (`surrogate.py`). The offline corrector qualification report used by
the look-ahead (`data/lookahead/AB_metrics.json`) is bound to this file and was produced with the routing; the
runtime audits that decide every executed step are independent of the surrogate. With the new code an unpatched
flight reproduces the exploratory single-regressor flight bit for bit (constant CL 0.025 from dh -734 m,
dV +58.7 m/s: G = 1.1442750658596543).

## Code

Repository `D:\dqn-final`, copied unchanged to `/hy-tmp/v14_final/v16_b16/repo`: 388 files, manifest SHA-256
(lines `sha256  path` of all files except .git, .cache, __pycache__, .pytest_cache, sorted by byte order, joined
with LF, no trailing newline; the convention of the earlier registrations)
5d36ff669901a717eda424f055193d1090d3d311a63e41b9ed3e29d4d79fe526. Tests: 30 passed locally.
Dual-routing comparison flights use the registered held-out-2 snapshot `/hy-tmp/v14_final/repo` (manifest
2f10ceb3..., verified after that run).

## Execution (cloud, 56 CPUs), `run_final.sh`, without manual steps

1. H2 flights: constant CL 0.025 and bc_seed17 from the 41 held-out-3 states (nominal + 40), once with this
   repository and once with the held-out-2 snapshot (`post_ablation.py`).
2. Training: the seven seeds in parallel, each
   `python3 repo/scripts/train.py --out runs/seed$s --seed $s --workers 8 --iterations 80`.
   A run ends by its stop rule or after 80 iterations; its final state is the policy. All seeds are reported; no
   seed is replaced or rerun, except a run that crashes with a software exception, which is resumed with the same
   command (resumption is exact) and reported.
3. Evaluation on held-out-3 (`--test-set held-out-3`, shared reference flights) of the 7 final policies and of
   the 7 existing 16-16 policies (trained with the routing surrogate, flown with the single regressor), then
   `compare.py` and `analyze_heldout3.py` (writes RESULTS.md, verdict.json).

## Test set held-out-3 (fixed by code)

`test_set_points('held-out-3')` in `src/rollout_improvement/evaluation.py`: 40 offsets drawn uniformly from
|dh| <= 2000 m, |dV| <= 100 m/s with `default_rng([20261023, 4])`; probe times 180, 192, 504, 564, 732, 744,
996 s drawn with `default_rng([20261023, 5])` from the 12 s grid without the development, held-out-1 and
held-out-2 times. Disjoint from all earlier sets (unit test). 0 of the 40 states lie inside |dh| <= 500 m,
|dV| <= 20 m/s; quadrant counts (+,+)/(+,-)/(-,+)/(-,-) 15/11/9/5. Full list: `heldout3_set.json`.

## Criteria

Gain = range at 1200 s of a policy minus range of the reference from the same initial state. 95% CI = paired
percentile bootstrap of the mean (10 000 resamples, seed 0). A failed flight counts with its truncated range.
The nominal state is reported but not part of the 40 states.

- **Primary H1**: for each of the seven seeds of the final method, (a) mean gain over all 40 states > 0 with CI
  lower bound > 0, and (b) mean gain over the states where the reference completed > 0 with CI lower bound > 0.
  Supported only if all seven seeds pass; the number of passing seeds is reported.
- **Primary H2** (confirmation of the exploratory finding): constant CL 0.025, range with the single regressor
  minus range with the routing, per state: mean > 0 with CI lower bound > 0.
- S1: for each seed of the final method, wins >= 33/40 and worst gain >= -1 km.
- S2: per state, seed-averaged range of the final method minus seed-averaged range of the existing 16-16
  policies of the same seeds (both flown with the single regressor): mean > 0 with CI lower bound > 0.
- S3: for each seed of the final method, mean A_closed >= mean A_open over the 14 probes.
- Descriptive: bc_seed17 single minus routing, nominal ranges, completion counts, medians, per-state values.

## Frozen files (SHA-256)

- `de059547b123cb54435da0348d6a2208c03d5fcc2ed82c31194acc5af01ed0df`  run_final.sh
- `f6ff4f666a7b3dd8547b1b8661941b21aa9e525ba8db84b36a368d9eccf65f59`  analyze_heldout3.py
- `8a22b5ce7c290ea023afbf4eb055a5f9b1348016ea6d1f1b399f174e8ad868f0`  post_ablation.py
- `e63b7acc53f3c36f11649ad9df9009abe643958163c576af9ed736c36997dbe7`  heldout3_set.json
- `be2bf5b936fe7862ad03892fe8125e186b3538c7847d60edea2ccf70f40effd8`  policies/bc_seed17.json
- `75ba00eaf9089974608804bc00e5616c826ac568eb5c91fb4d3f34db06b29fa1`  policies/bc_seed23.json
- `45f463b56f25c9a2a0432a59bb9cf1e52b28c754916ce01e9d175ce54a6b42bd`  policies/bc_seed41.json
- `d8751cfc8c2a94156b7b78f40164f0a4d85bf87c9f062d940a2454bc30c55d91`  policies/bc_seed59.json
- `4e9c9e7c125bc5974bcbe3d4e130ec7ee7843400ba2005f046d0cd67c9ccfe54`  policies/bc_seed83.json
- `72c2e03263eef9dd47b884e03199be024093f7e9d0b80378ea27e9d851591208`  policies/bc_seed97.json
- `df8bddffcea6d26be89e92b4baf985947ad46c250be4ddbf1e03b3c46373030a`  policies/bc_seed101.json
- `b7d8dc8f2e906b2092b6a762c9a845b9dbf1137b6cd2cfb111ea110ea34cfedf`  repo/src/certified_flight/surrogate.py
- `7a349bf2cdf67bd11caa0e4e06986dcb0d4d58bfe73015452d397f56f6e8425b`  repo/src/rollout_improvement/evaluation.py
- `f435c4ae750d4294673907eb6a656da4618b641d91d011ddebe426386249472e`  repo/src/rollout_improvement/ensemble.py
- `3298f966ac66a53085b81c6d1e42344b4abe535db225227c8196c0c20aef933f`  repo/src/rollout_improvement/training.py
- `fd024198ee57faa392243836583ce354787c0e3fb1a0566b1a8cbdd03b70efb1`  repo/scripts/compare.py
- `9e1efc6f00b859237dca9869a56ddc995bea3dec8d194872be74c7f8c0ed6ffc`  repo/scripts/train.py
- `a1267c9e335a71e414adf274e41412e33a93fe4fbf15b90c87cc79c1341a2bc3`  repo/scripts/evaluate.py
- `1cb8d96280679a7e2a3e7d9ddc90c2e2f3159714f60fd19035cf77aca7946f00`  repo/data/surrogate/controller.npz
- `43f893274de86583f2ad0a955262dff47c115ad6bf2b0c86088d7c0d05d06e5d`  repo/data/surrogate/controller.json

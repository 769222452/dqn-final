# Certified rollout policy iteration for a morphing waverider

[中文说明](README_zh.md)

This repository contains the code and input data for learning the lift-coefficient schedule of a morphing
hypersonic waverider over a fixed 1200 s glide, where every executed control step is verified by a safety
certification chain. The policy is improved by rollout-based approximate policy iteration: exact action
advantages are measured by branching deterministic certified simulations, an ensemble of neural networks
generalises them, and a candidate policy is accepted only if its certified return increases.

## Problem

* **Plant.** Three-degree-of-freedom point-mass flight over a spherical Earth (state: altitude h, speed V,
  flight-path angle gamma, downrange s) with angle of attack alpha and a morphing vehicle that blends between
  298 precomputed geometries (8 s transitions). Aerodynamic coefficients are interpolated in Mach number and
  angle of attack from a precomputed database.
* **Mission.** 1200 s from a fixed initial state; the objective is the downrange at 1200 s. The return is
  G = s(1200 s) / 2400 km + 0.10 for a successful mission (no constraint violation, final altitude
  >= 20 km, speed >= 800 m/s, downrange >= 360 km).
* **Constraints.** Altitude 20-80 km, |gamma| <= 0.35 rad, angle of attack -2 to 12 deg with rate
  <= 5 deg/s, dynamic pressure <= 80 kPa, load factor <= 3 g, heat-flux proxy <= 2 MW/m^2.
* **Actions.** Every 12 s the policy selects one of 19 absolute lift-coefficient targets
  (-0.01, 0, 0.01, 0.015, ..., 0.06, 0.07, ..., 0.12). The target is held for four 3 s control steps.

## Certified execution

A 3 s step is executed only after a certificate for that exact state and action passed. For a requested lift
target the chain computes the angle-of-attack and geometry commands with a surrogate of a two-stage optimal
control problem (with an exact-physics baseline when the surrogate is outside its training domain and a full
OCP solution as the last fallback), corrects them under the constraints, checks a 30 s look-ahead for a
recoverable continuation, and integrates the dynamics independently (RK4, 0.1 s audit grid) to confirm every
constraint. When the requested action cannot be certified, the certifiable action closest to it is executed;
if no action can be certified the flight ends as a failure. All learning signals below come from such
certified flights only.

## Method

Policy iteration starts from the constant policy CL = 0.04 (pi_0). In iteration k with policy pi_k:

1. **Exact advantage labels.** At 25 decision times (spacing 48 s, offset rotating by 12 s every iteration)
   the simulation is copied and two alternative actions are held for one decision: pi(s) +- 1 class or
   pi(s) +- 2 classes, alternating between consecutive 96 s blocks of the mission and between sweeps, so
   that two sweeps test all four neighbours of every state. The flight is then continued to the end of the mission by **open-loop replay** of the nominal
   decision sequence of pi_k. The label A(s, a) = G_rest(a) - G_rest(pi(s)) is exact because the simulator is
   deterministic; the unchanged branch is the nominal trajectory itself.
2. **Advantage ensemble.** Five networks (4 inputs, two tanh hidden layers of 16 units, 19 outputs) are
   fitted on bootstrap samples of the labelled states with clipped targets and out-of-bag checkpoint selection.
   All 3000 updates are run; identical network inputs stay in the same bootstrap/OOB group.
   Inputs are the flight state and time (h, V, gamma, t); labels of the last seven policy versions are used
   with weight 0.5^age. `--hidden` sets the layer widths.
3. **Candidates.** pi_k is extended by one network stage: within +-2 classes of the current action it selects
   the action with the largest lower confidence bound (mean - standard deviation) if that bound exceeds a margin
   (six margins between 0.25e-4 and 1.5e-4) and only within +-48 s of labelled decision times.
4. **Monotone acceptance.** Each candidate is flown over the full mission with certification. The best one is
   accepted only if its exact return exceeds G(pi_k); otherwise labels keep accumulating.

Training stops after two consecutive sweeps over all state offsets without any positive advantage, invalid
branch or acceptance, or after three sweeps without acceptance. These are local search/stall conditions,
not proofs of global convergence. The saved `stopped.category` distinguishes `local_search_stationary`,
`stalled` and `budget_limit`; `convergence_proven` is false.

**Baseline (ablation) settings.** The previous variant of the method continues the branches in **closed loop**
(pi re-decides after the alternative action), uses all six features including the policy's own action feedback
(angle of attack and previous CL target) and additionally proposes exact override candidates built from the
positive labels. It is obtained with `--continuation closed-loop --inputs all --overrides`; each switch can be
used separately.

## Repository layout

```
configs/                      mission, vehicle, atmosphere, trajectory and certification settings
data/                         input data, see "Data" below
  geometry_subset/            the 298 candidate geometries and their aerodynamic tables (used in flight)
  surrogate/                  surrogate of the two-stage optimal control problem
  lookahead/                  qualification report of the look-ahead recovery corrector
  initial_state.json          initial state of the mission
  SHA256SUMS                  hashes of all inputs, verified before every flight
archive/                      lossless source data, CSV exports and retired legacy utilities
src/morphing_waverider/       point-mass dynamics, atmosphere and units
src/certified_flight/         certified execution
  build.py                    build_flight(): environment and certifier from the configuration
  environment.py              flight environment (3 s steps, transitions, rewards)
  certification.py            exact-state certificates, certified action pool, certified execution
  surrogate.py, ocp.py        command generation: surrogate, physics baseline, two-stage OCP fallback
  corrector.py, lookahead.py  constraint-aware correction and 30 s look-ahead guard
  reachability.py             conservative 3 s reachability screen of the CL target
  plant.py, aero.py           compiled plant with the aerodynamic interpolation, used by audits
src/rollout_improvement/      learning
  policy.py                   policies as chains of improvement stages
  rollouts.py                 certified flights, branch labels, candidate evaluation
  ensemble.py                 bootstrap ensemble of advantage networks
  training.py                 policy iteration loop and command-line options
  evaluation.py               generalisation tests (held-out and development test sets)
scripts/                      train.py, evaluate.py, compare.py, run_seeds.sh
tests/                        unit tests
```

## Data

Every file in `configs/` and `data/` (except the checksum list itself), the archive, and the
`morphing_waverider` sources are listed in `data/SHA256SUMS`;
each flight verifies them first and stops if any file was changed. The tables mark which files the code loads
during a flight; the other files document where these inputs come from.

**Configuration (`configs/`)**

| File | Content | Used in flight |
|---|---|---|
| `mission.yaml` | 1200 s mission, 3 s control steps, initial state, path limits, terminal gates, return definition; paths of the other configuration files | yes |
| `environment.yaml` | atmosphere model, spherical Earth, terminal altitude, required range (cross-checked against `mission.yaml`) | yes |
| `vehicle.yaml` | mass, path limits, heat-flux proxy, angle-of-attack rate limit and research assumptions | yes |
| `teacher_profile.json` | reference mass and area of the aerodynamic tables; must agree with `mission.yaml` and `vehicle.yaml` | yes (consistency check) |
| `trajectory.yaml` | active OCP objective, target tracking, solver tolerances/budget and numerical scaling; the horizon/grid are in `runtime.json` | yes |
| `runtime.json` | certification chain: surrogate, fallbacks, constraint corrector, 30 s look-ahead, the 19 CL actions, 8 s geometry transition, OCP settings | yes |

**Aerodynamics of the candidate geometries (`data/geometry_subset/`)**

The flight uses one aerodynamic table per geometry: lift, pressure drag and pitching-moment coefficients on 18
Mach numbers (3.5 to 8.25) x 20 angles of attack (-1 to 12 deg), computed with a strip shock-expansion model
(pressure drag only, no friction). For a lift target the code interpolates each geometry's own table (PCHIP in
Mach, then PCHIP in angle of attack), finds the angle of attack that gives the target, and selects the geometry
with the lowest drag there. During an 8 s transition the coefficients of the old and the new geometry are
blended linearly at the same Mach number and angle of attack.

| File | Content | Used in flight |
|---|---|---|
| `aerodynamics/p2_grid.npz` | the tables above as arrays [geometry, Mach, alpha], with validity masks and geometry parameters | **yes, the only aerodynamic data of the flight** |
| `manifest.json` | geometry ids, array shape, SHA-256 of every file; its hash ties the surrogate to this snapshot | yes |
| `action_index.csv`, `candidate_inventory.csv` | index-to-id mapping; shape parameters, volume and thickness ratios to G42, geometry validation records | no |
| `geometries/<id>/native_geometry.mat` | shape of each of the 298 geometries (G42 is the starting geometry of every flight) | no |
| `benchmarks/` | best geometry per condition, mean drag of every fixed geometry over the 108 conditions | no |
| `metadata/`, `provenance/` | grids, reference area and length, model options; scripts and source hashes of the aerodynamic computation | no |
| `figures/` | lowest-drag geometry at each Mach number and lift target | no |
| `validation.json`, `load_database.py`, `README_ZH.txt` | export checks, standalone loader, description of the snapshot | no |

**Certification models**

| File | Content | Used in flight |
|---|---|---|
| `surrogate/controller.npz` | network weights of the surrogate of the two-stage optimal control problem (angle-of-attack commands) | yes |
| `surrogate/controller.json` | its training settings and the SHA-256 of the mission, trajectory, data set and geometry snapshot it was trained for; checked when loading | yes |
| `lookahead/AB_metrics.json` | offline qualification report of the constraint corrector; a flight is built only if the report is complete and passed | yes |
| `lookahead/manifest.json` | hashes of the qualified model and the frozen corrector settings, which must equal `runtime.json` | yes |

**Other inputs**

The three `aero/` files below and the two CSV exports formerly under
`geometry_subset/aerodynamics/` are ZIP members in `archive/retired_assets_20261030.zip`,
under their original `data/` paths. They are retained for research and export use.

| File | Content | Used in flight |
|---|---|---|
| `initial_state.json` | recorded initial state (h, V, gamma, s, alpha) of the mission; must agree with `mission.yaml` | yes |
| `aero/aero_database.csv` | source database of the geometry family: 348 geometries x 13 Mach numbers (3.5 to 9.25) x 15 angles of attack, lift, drag and moment of an earlier off-design model | no |
| `aero/g42_baseline.csv` | the same for the baseline geometry G42 | no |
| `aero/geometry_inventory.csv` | design parameters and origin of the 348 geometries | no |

The Mach support 3.5 to 9.25 recorded in `mission.yaml` describes the source database;
the flight tables end at Mach 8.25 and flights leaving them end as a domain exit.
The unused source CSV paths have been removed from the mission configuration.

The 2026-10-30 cleanup retires unused continuous-geometry, actuator, integrator, logging and
configuration helpers. The active RK4 plant and eight-second transition remain in `certified_flight`.
Only unused configuration keys were removed; active numerical values, model weights, aerodynamic
arrays, all 298 MAT shapes and qualification evidence are retained. Original files are stored in the
archive with per-member SHA-256 hashes; see `archive/README.txt` for restoration into a new directory.
The snapshot and controller input hashes describe the new packaging; this is not a new model qualification.
Source/configuration fingerprints change, so use new run/evaluation directories. Existing checkpoints
and caches require their original working-tree snapshot; fingerprint guards are not bypassed.

## Installation

Python 3.11 is required.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

A C compiler (`gcc` on the PATH, or the path in the environment variable `CERTIFIED_FLIGHT_CC`) is optional:
CasADi functions are then compiled once into `.cache/` for speed. Without a compiler the CasADi virtual
machine evaluates the same expression graphs with identical results.

## Usage

Train with the proposed settings (one process per worker; an iteration performs about 17 000 certified
3 s steps):

```bash
python scripts/train.py --out runs/seed17 --seed 17 --workers 14
```

The run directory contains `state.json` (policy, labels, per-iteration history and experiment fingerprint)
and one folder per iteration. Proofs of branches and all candidates are kept by default; interrupted attempts
are retained separately. A compatible interrupted run is resumed by repeating the command. Legacy runs
without a fingerprint, changed code/inputs, or changed learning settings require a new output directory.
Baseline:

```bash
python scripts/train.py --out runs/baseline_seed17 --seed 17 --continuation closed-loop --inputs all --overrides
```

Several seeds: `SEEDS="17 23 41" bash scripts/run_seeds.sh`.

## Evaluation

A trained policy is flown from perturbed initial states and compared with the constant policy CL = 0.025
flown from the same states; in addition, one decision of its nominal flight is changed by one class and the
flight is continued in closed loop (the policy re-decides) and in open loop (its nominal decisions are
replayed). Four test sets are defined, each with 7 probe times on the 12 s decision grid:

* **held-out-3** (default): the test set of the proposed method. 40 initial-state offsets drawn uniformly
  from |dh| <= 2000 m, |dV| <= 100 m/s from the fixed seed 20261023 (separate random streams for offsets and
  probe times), disjoint from the other sets. It was fixed before the proposed method was trained.
* **held-out-1**, **held-out-2**: generated in the same way from other streams. They were used to test earlier
  configurations of the method (network size, configuration of the post-command surrogate) and their results
  were known when the final configuration was chosen, so they are not clean tests of the proposed method.
* **development**: the nominal state, 12 perturbed states on a grid and 7 probe times. These tests were used
  to diagnose the previous method and to choose the final one, so results on them are biased in its favour
  and are not a measure of generalisation.

No held-out initial state was used for training or development. The probe times of the held-out sets were not
used in any evaluation before, but the probes start on the nominal trajectory, whose decision states are
branched during training; the probes measure the reaction to a perturbation, not performance on unseen states.

```bash
python scripts/evaluate.py runs/seed17/state.json --out evaluation/seed17 --reference-out evaluation/reference
python scripts/evaluate.py runs/seed23/state.json --out evaluation/seed23 --reference-out evaluation/reference
python scripts/compare.py --out evaluation/comparison.json seed17=evaluation/seed17 seed23=evaluation/seed23
```

`--reference-out` lets several policies share the reference flights. `compare.py` reports for each policy the
mean gain over the reference with a paired bootstrap 95% confidence interval, the number of wins and the worst
case (also restricted to the states from which the reference completed the mission, since a failed reference
flight counts with its truncated range), and the paired differences between the first policy and each other
one. `--test-set held-out-2`, `held-out-1` or `development` selects the other test sets.

## Reproducibility

Random numbers are seeded and BLAS runs single-threaded. Numerical regression tests check reproducibility,
but wall-clock limits in certification can change a refusal/fallback when machine load changes, even on the
same platform. Consequently the complete training run is not guaranteed bit-identical under arbitrary load.
Different CPUs can also change the last
digits of floating-point results: after 240 s of flight the states of two machines differed by 1e-14 relative
to the typical magnitude of each component (5 nm of range). Over a training run such differences can change a
discrete decision, so runs on different machines need not be identical. The packages set the BLAS thread
variables when they are imported; code that imports them from its own scripts should do so before importing
numpy (as `scripts/` does).

Every flight first verifies the configuration files, the input data and the physics package against
`data/SHA256SUMS` and stops if any file was modified.

## Integrity and diagnostic protocol

Training protocol `rollout-improvement/6-integrity` binds source/input hashes, numeric software versions,
seed and learning settings. `--iterations` remains the additional budget of this invocation;
`--ignore-stop-rule` explicitly continues a compatible stopped run. This does not migrate old labels.
Evaluation schema `certified-policy-evaluation/2` also binds the policy, initial conditions and continuation
schedule. Old or mismatched cached results are refused and preserved; use a new evaluation directory.

`DONE` now means a successful 1200 s mission; terminal failures retain their actual outcome.
`switches`/`geometry_switches` count geometry transition starts, whereas `cl_action_changes` counts changes
of the executed CL action. Only truly unseen observed-feature/action pairs enter the next-label unseen
diagnostic; seen pairs are reported separately. Ensemble mean-minus-standard-deviation remains an
uncertainty heuristic, not a calibrated confidence bound.

Evaluations retain full flight evidence and expose requested/completed/failed/skipped/error counts.
Missing probe pairs withhold the main probe average; partial statistics are explicitly labelled.
The evaluation command returns exit code 2 when the requested comparison is incomplete, and `compare.py`
refuses incomplete or incompatible summaries. Computation errors can retry into a new attempt directory.
Evidence deletion requires explicit `--discard-evidence` for evaluation or
`--discard-rejected-evidence` for training; preserving evidence uses more disk space.

The physical task is unchanged: pressure drag only, endpoint coefficient blending, discrete alpha
commands, point-mass dynamics and a heat proxy. These checks establish feasibility within that model,
not a validation of vehicle trim, continuous actuator dynamics, structural morphing or thermal protection.

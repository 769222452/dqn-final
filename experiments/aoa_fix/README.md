# Angle-of-attack sawtooth: root cause, fix and experiments

Branch `aoa-fix` = the original V16 source (`cloud-v16-original-20261031`, byte-identical to
`v14_final/v16_b16/repo` of the result archive) plus an **opt-in** inner-loop objective change.
With default settings every V16 code path, number and test is unchanged
(`tests/test_certified_flight.py` reproduces the 240 s V16 reference to 1e-12).

## 1. Root cause (evidence from the archived V16 flights)

Measured with `scripts/analysis/aoa_metrics.py` on the complete archived 1200 s flights of seed 101
(`v16_b16/runs/seed101/iter_000/initial` = constant CL 0.04, `iter_009/candidate_m0.25` = final policy):

| metric | constant CL 0.04 | final policy |
|---|---|---|
| alpha direction reversals (0.5 s steps) | 33 % | 35 % |
| median peak-to-peak within each 3 s step | 0.79° | 1.17° |
| median jump at the 3 s replanning boundary | 0.78° | 1.19° |
| boundary jump opposes the previous step's drift | 99.7 % | 100 % |
| share of detrended alpha variance at 1/3 Hz and harmonics | 36 % | 58 % |
| mean intra-step profile (slots 0..5, deg) | −0.26 −0.23 −0.16 −0.01 +0.19 +0.47 | −0.45 −0.31 −0.19 −0.02 +0.27 +0.68 |

So the oscillation is a deterministic **3 s replanning sawtooth**: alpha ramps up through every 3 s
step and jumps back at the next decision. It exists at constant CL (the learned policy only enlarges it).

Executed alpha equals the post-surrogate output in 400/400 steps (corrector source `NETWORK`); the
surrogate reproduces its teacher, the inner two-stage OCP. Solving that OCP at a recorded state
(t = 150 s) gives a 3 s prefix ramping 5.12°→6.10° (surrogate: 5.20°→6.10°) followed by a jump to about
8.5° for 3–60 s. The OCP is **time-inconsistent**:

- `src/certified_flight/ocp.py`, `tracking_mode: mean_3s`: the commanded CL is tracked only as the
  mean over the first 3 s; the 3–60 s tail is free economic optimisation (drag, heat, range, terminal γ).
  The plan ramps toward the economic tail while keeping the 3 s mean; the next solve repeats it.
- The jump from the measured alpha is free: the control-rate cost starts at `i > 0`, and the corrector's
  smoothness term starts at `u[1]-u[0]` (`corrector.py`). Only the 2.5°/0.5 s slew bound limits it.

The deviation is not useless: the executed 3 s-mean CL misses the command by a median 0.0035–0.006,
and this freedom keeps the early descent feasible. Pure equal-lift CL tracking of CL 0.025 fails at
t = 54 s even with the V16 corrector and look-ahead; the V16 inner loop passes. A fix must therefore
remove the inconsistency, **not** the inner loop's economic freedom (strict tracking is what forced the
recovery machinery of the `supported-recovery-latest-20261031` branch).

## 2. Fix (opt-in, `TwoStageOCP` + `PostCorrector`)

`--objective consistent` in `scripts/aoa_gate_experiment.py` sets:

- `tracking_mode: pointwise_horizon`: the commanded CL is penalised pointwise at every RK stage over
  the whole 60 s horizon with one stationary weight (head and tail of the plan share one objective, so a
  re-solve has no incentive to undo the executed prefix). The economic terms are unchanged.
- `boundary_smoothing: true`: the step from the measured alpha is costed like every other command
  change, in the OCP and in the post corrector.
- `horizon_tracking_weight: 20` (**frozen**). Calibrated once on the nominal 0–120 s development flight
  (no held-out state) to match V16's executed CL error; evidence in `calibration/`:

| 0–120 s, CL 0.025 | V16 | weight 5 | **weight 20** | weight 80 |
|---|---|---|---|---|
| intra-step peak-to-peak (median) | 1.21° | 0.13° | 0.20° | 0.22° |
| boundary jump (median) | 1.08° | 0.06° | 0.28° | 0.19° |
| 1/3 Hz power share | 20 % | 6 % | 5 % | 1 % |
| jump-opposes-drift share (0.5 = none) | 92 % | 49 % | 59 % | 56 % |
| executed CL error (median) | 0.0061 | 0.0125 | **0.0068** | 0.0087 |
| range at 120 s | 286.40 km | 285.34 | 285.99 | 286.09 |

All four survived the critical early descent. Development note: in this phase the remaining reversals
(16–25 %) come from a second, safety mechanism — the V16 look-ahead guard adds +0.5…+2° pull-up offsets
(14–18 of 40 early steps; only 1–8 of 400 steps in the archived full V16 flights). The raw reversal rate
therefore mixes both mechanisms and is reported but not used as a pass criterion; the sawtooth-specific
criteria below were fixed before any full-mission gate or held-out flight was evaluated.

The gate experiment uses the **true OCP** as inner controller (no surrogate), so it isolates the
objective change. The V16 corrector, the 30 s look-ahead (future candidate policies use the unchanged
V16 surrogate) and the 0.1/0.5 s audits are unchanged; a refused CL request falls back to the nearest
certifiable action, as in V16 certification.

## 3. Pre-declared criteria (`scripts/analysis/aoa_report.py`)

Gate (nominal state; constant CL 0.025 and final seed-101 policy):
- G1 sawtooth removed: 1/3 Hz power share ≤ 0.25 × V16, jump-opposes-drift share ≤ 0.65, median
  intra-step peak-to-peak ≤ 0.5 × V16;
- G2 the consistent objective completes the 1200 s mission;
- G3 |range(consistent) − range(V16 objective)| ≤ 0.5 %.

Held-out-3 (nominal + 40 states, paired by state; constant CL 0.025 and the 7 V16 policies):
- H1 range kept: consistent − V16-objective range, mean ≥ −0.5 % of the V16 mean, CI upper bound ≥ 0;
- H2 learned gain kept: per seed, gain over constant CL 0.025 under the consistent objective has a
  95 % CI lower bound > 0 (paired bootstrap, 10 000 resamples, seed 0, as in V16);
- H3 sawtooth removed (G1 thresholds) on every completed flight pair.
Failed flights count with their truncated range. All outcomes are reported.

**Addendum: surrogate pipeline.** This was declared after the local gate and before any refitted
surrogate was fitted or flown.
- H1–H3 are evaluated on the `heldout` stage, i.e. the production surrogate pipeline: arm (i) is the V16
  surrogate with the V16 objective, arm (ii) is the refitted surrogate with the consistent objective.
- Q1: the refitted surrogate must pass the V16 P1 rule before any held-out flight is run. The rule is
  zero independent dense-audit failures of corrected plans on validation and test labels, which come from
  flights never used for fitting.
- Q2: arm (i) must reproduce the archived V16 held-out-3 ranges, since it is the identical code path.
  Any difference above 1e-6 km is reported as a reproduction failure.
- Fidelity (`fidelity` stage, true OCP) is descriptive: the student−teacher range and sawtooth
  differences of both arms. It is not a pass criterion. The V16 surrogate itself is 13.1 km short of its
  own teacher at the nominal state (2443.32 vs 2456.45 km), so arm (ii) is judged against arm (i), not
  against the OCP gate numbers.

**Amendment (before any held-out result and before the V16-objective gate flights finished).** The first
two full gate flights (consistent objective) reduced within-step alpha swings to ~0.002°. Reversal and
jump-opposition shares computed over every non-zero change then measure solver round-off (moves of
~1e-3°), not a sawtooth. Both are now computed only over changes above a 0.01° noise floor
(`NOISE_DEG` in `aoa_metrics.py`); jump opposition is evaluated over steps whose jump and preceding drift
both exceed the floor (`None` = no such step, i.e. no sawtooth). V16 baselines are unaffected by this
(jump-opposition 99.7 % / 100 % over 398 / 399 steps). No threshold value was changed.

## 4. Why the true OCP for the gate, and the surrogate for everything else

V16 replaces the repeated OCP solves by a kernel-ridge surrogate. The sawtooth lives in that
surrogate's training labels: executed alpha is the surrogate output in 400/400 steps and the surrogate
reproduces its OCP teacher (section 1). The shipped `data/surrogate/controller.npz` was fitted to
V16-objective labels, so it cannot express a new objective; flying the fix with it would only reproduce
the sawtooth. A new surrogate needs new teacher labels, i.e. OCP solves, anyway. Hence:

1. **Gate (true OCP, 4 flights).** Cheap causal check that the *objective* removes the sawtooth without
   losing range, before paying for labels, refit and qualification. It separates "the objective works"
   from "the surrogate approximates it well".
2. **Surrogate refit (V16 design, V16 model class).** Teacher labels of the consistent objective on 1,500
   recorded V16 contexts → fit → P1 qualification → one DAgger round on development states → final fit and
   qualification (section 9).
3. **Held-out-3 with the production pipeline (surrogates, 656 flights).** Arm (i): V16 surrogate + V16
   objective, i.e. the archived V16 configuration (must reproduce the archived ranges; checked in the
   report). Arm (ii): refitted surrogate + consistent objective. The fallback chain (physics baseline on
   OOD, then true OCP), corrector, look-ahead and audits are unchanged and use the arm's own objective.
4. **Fidelity (true OCP, 24 flights).** Nominal + 5 held-out states, constant CL 0.025 and seed 101, both
   objectives: surrogate-vs-teacher range and sawtooth differences of both arms.

Measured costs (one core): surrogate 1200 s flight ≈ 1 min (V16 arm, 55 s locally); true-OCP flight
25–50 min; teacher label ≈ 6 s. Held-out-3 with the true OCP every decision would be ≈ 800 CPU-h
(`heldout-ocp` stage, optional) instead of ≈ 15 CPU-h.

## 5. How to run (cloud server)

Requirements: Linux, Python 3.11, `gcc` (CasADi functions are compiled once into `.cache/`), about
2 GB RAM per parallel job. Use at most one job per physical core: the V16 corrector (2 s) and look-ahead
(30 s) have wall-clock budgets, so oversubscription can change results.

```bash
git clone --single-branch --branch aoa-fix https://github.com/769222452/dqn-final.git dqn-aoa
cd dqn-aoa
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export JOBS=$(( $(nproc) - 2 ))
bash scripts/run_aoa_experiments.sh test        # all tests incl. bit-exact V16 reference; must pass
bash scripts/run_aoa_experiments.sh smoke       # ~3 min: OCP and surrogate inner loops, 2 labels
bash scripts/run_aoa_experiments.sh gate        # [true OCP] 4 nominal flights, ~1 h
bash scripts/run_aoa_experiments.sh report      # check PROCEED_TO_HELDOUT=True for both policies
bash scripts/run_aoa_experiments.sh surrogate   # labels, fit r0, qualify, 26 DAgger flights, final fit+qualify
bash scripts/run_aoa_experiments.sh heldout     # [surrogate] 656 flights
bash scripts/run_aoa_experiments.sh fidelity    # [true OCP] 24 flights (~10-20 h wall with 24+ cores)
bash scripts/run_aoa_experiments.sh report
```

Run inside `tmux`/`nohup`. Every stage resumes: finished flights (`summary.json`) and finished labels
are skipped; fitted/qualified models are kept (delete `data/surrogate_consistent*` and
`data/lookahead_consistent*` to refit). The `surrogate` stage stops if a qualification fails
(`P1_physical_gate_passed=false`); do not run `heldout` then — send the qualification reports back.
Subsets: `POLICIES="const0.025 seed101"`, `STATES=10`, `OBJECTIVES=consistent`.

## 6. What to send back

```bash
tar czf aoa_results.tgz runs/aoa/report.md runs/aoa/report.json runs/aoa/*.jobs runs/aoa/*.progress \
  runs/aoa/surrogate data/surrogate_consistent* data/lookahead_consistent* \
  $(find runs/aoa -name summary.json -o -name setup.json -o -name steps.jsonl) \
  $(find runs/aoa -name '*.log' -size -2M)
```

plus `nproc`, CPU model and `pip freeze`. `steps.jsonl` holds every executed alpha command, so all
metrics can be recomputed.

## 7. Files

- `src/certified_flight/ocp.py`, `src/certified_flight/corrector.py`: opt-in objective options.
- `src/certified_flight/objectives.py`: the two objectives and their runtime configurations.
- `src/certified_flight/contracts.py`: the new options enter the control-contract digest only when they
  differ from their defaults (V16 digest unchanged), so a surrogate cannot be flown with the wrong objective.
- `src/certified_flight/lookahead.py`: `boundary_smoothing` is a qualified corrector setting.
- `src/certified_flight/build.py`: `build_flight(..., config=...)` for a non-default runtime configuration.
- `src/certified_flight/distill.py`: teacher labels and the V16 kernel-ridge fit.
- `scripts/aoa_gate_experiment.py`: one flight (`--inner ocp|surrogate`, objective, policy, offset).
- `scripts/aoa_label_contexts.py` (run once; output committed), `scripts/aoa_generate_labels.py`,
  `scripts/aoa_fit_surrogate.py`, `scripts/aoa_qualify_surrogate.py`, `scripts/aoa_dagger_contexts.py`.
- `scripts/run_aoa_experiments.sh`, `scripts/analysis/aoa_jobs.py`: stages and job lists.
- `scripts/analysis/aoa_metrics.py`: oscillation metrics (also for archived V16 flights).
- `scripts/analysis/aoa_report.py`: tables, verdicts, archive reproduction, fidelity.
- `experiments/aoa_fix/surrogate/contexts.jsonl`: 1,500 recorded pre contexts (see section 9).
- `experiments/aoa_fix/v16_heldout3_archive.json`: archived V16 held-out-3 ranges (8 policies × 41 states).
- `experiments/aoa_fix/policies/`: the 7 trained V16 policy chains (chain field of
  `v16_b16/runs/seed*/state.json`), hashes in `manifest.json`.
- `experiments/aoa_fix/calibration/`, `gate_local/`, `surrogate_local/`: local development evidence.
- `tests/test_aoa_objective.py`, `tests/test_aoa_surrogate.py`.

## 8. Local gate result (4-CPU development container, 2026-10-31)

The four gate flights were run once locally before the cloud run (code of commit `e85c8dc` plus
`--horizon-weight 20`; metrics recomputed with the amended noise floor). Evidence: `gate_local/`
(`steps.jsonl` holds every executed command).

| nominal 1200 s | objective | range | intra-step ptp | 1/3 Hz power | CL error (median) |
|---|---|---|---|---|---|
| CL 0.025 | V16 | 2456.45 km | 1.40° | 0.51 | 0.0042 |
| CL 0.025 | consistent | 2475.94 km | 0.002° | 0.0003 | 0.0026 |
| seed 101 | V16 | 2460.65 km | 1.41° | 0.68 | 0.0044 |
| seed 101 | consistent | 2477.81 km | 0.002° | 0.0000 | 0.0026 |

- G1 (sawtooth removed) and G2 (feasible) pass for both policies.
- **G3 as declared fails for both — because the consistent objective flies farther** (+19.5 km,
  +0.79 %; +17.2 km, +0.70 %), outside the two-sided ±0.5 % band. Writing G3 two-sided was a design error;
  its intent was "no range loss". The declared verdict is kept in the report; a separately labelled
  one-sided reading (`G3_no_loss`: consistent ≥ 99.5 % of V16) decides `PROCEED_TO_HELDOUT`. The held-out
  criterion H1 was already one-sided and is unchanged.
- Learned-policy gain over constant CL 0.025 at the nominal state: +4.20 km with the V16 objective,
  +1.87 km with the consistent objective. One state is not evidence either way; H2 on held-out-3 decides.
- Not yet explained: why removing the sawtooth adds ~17–20 km. A plausible but untested mechanism is the
  convexity of drag in alpha (oscillating about a mean costs more drag than holding it). This needs a
  dedicated check before it is claimed.
- These flights use the true OCP inner loop; they are not the archived surrogate flights (constant CL
  0.025 surrogate flight in the V16 archive: 2443.32 km), so V16-surrogate vs OCP differences are a
  separate comparison.

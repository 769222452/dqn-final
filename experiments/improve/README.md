# Larger, attributable performance gains: training stop, execution layer, CL + geometry, coordinated search

The learned V16 policies beat constant CL 0.025 by only +3…+10 km out of ≈2450 km. This study looks at
four suspected causes using the project code, data and physics. It then implements the changes that the
evidence justifies, and tests them while keeping every improvement attributable.

- **Opt-in only.** Every change is opt-in. The defaults still reproduce V16 bit for bit
  (`tests/test_certified_flight.py`), and every aoa-fix test still passes.
- **Safety chain unchanged.** Every certification layer is kept: reachability screen, surrogate
  coverage gate, physics-baseline and true-OCP fallbacks, post corrector, 30 s look-ahead witness, and
  the 0.1/0.5 s audits.
- **Hard limits unchanged.** All mission limits stay hard constraints: h, V, γ, α, α-rate, q ≤ 80 kPa,
  heat flux ≤ 2 MW/m², n ≤ 3 g, Mach support.

## 1. Findings

### 1.1 Headroom: a physical upper bound (`src/certified_flight/bound.py`, `scripts/upper_bound.py`)

The bound is a 1200 s full-horizon OCP that maximises s(1200).
- It uses the exact point-mass dynamics, heat proxy, q and n of `mission.model`, with every path limit
  enforced at every RK stage.
- CL itself is the control, with CD ≥ the lower convex hull of all (CL, CD) pairs of the 298 geometries
  over the alpha support. This is an epigraph built from supporting lines and interpolated in Mach.
- A morph blend is a convex combination of two endpoint (CL, CD) pairs at the same alpha, so it never
  lies below the hull.
- The bound drops the alpha-rate limit, the 8 s morph time, the 3 s/12 s decision structure and every
  certification layer. It therefore bounds **any** admissible flight, whatever the policy, morphing
  strategy, inner loop or surrogate.
- **Accuracy.** Each solution is re-integrated on a 0.1 s grid; the range changes by < 0.05 km and
  constraint margins stay ≥ −1e-5.
- **Caveats.** The hull is sampled at a 0.05° alpha step and a 0.02 Mach step, with a 0.2 % safety
  shift. The hull is about 1 % better in L/D than the best single geometry, so the bound is somewhat
  optimistic.

| nominal state | upper bound |
|---|---|
| all 298 geometries | **2511.03 km** |
| generator's family {S2_MD12, S2_MD8, G42} | 2510.13 km |
| best single geometry S2_MD8 (no morphing) | 2505.03 km |
| start geometry G42 only | 2489.35 km |

**Shape of the optimum.** After the pull-out (0–90 s, a dive to ≈27 km), the optimum **rides the
dynamic-pressure limit**: q = 80.0 kPa, with CL rising from 0.0239 to 0.0250 as V decreases. This is the
classical constrained glide. The max-L/D CL (0.0175–0.020 at M 7–8) would need q > 80 kPa, so the q limit
binds throughout the cruise.

The bounds for the other states are in `bounds/`; held-out-3 states are filled by the cloud run.

### 1.2 Issue 2 — the execution layer, not the policy, sets the cruise

The inner OCP (`ocp.py` build, weights in `configs/trajectory.yaml`) minimises
`w_drag·∫DV/1e9 + w_heat·∫heat/5e7 + … − w_range·s`. The heat normalisation 5e7 is marked as a
placeholder in the config ("占位值").
- **The heat term dominates.** Along the cruise it is ≈0.9, against drag ≈0.12 and CL tracking ≈0.02.
  So it fixes the altitude: q = 68–74 kPa and executed CL ≈ 0.0275, **whatever the policy commands**.
- **Commands below 0.025 don't take effect.** In the 19-constant sweep (`../aoa_fix/cl_sweep_local/`),
  executed CL never drops below ≈0.0275, and commands ≤ 0.02 only oscillate around 0.03. The policy's
  action is a soft bias with almost no authority over the cruise.
- **The mission objective is range only.** Heat flux is a hard limit that is never active in cruise
  (peak 1.62 MW/m² during the pull-out).

**Removing the heat penalty alone (`aligned`) makes things worse:** true-OCP nominal flight
2461.47 km, versus 2475.94 km for `consistent` (the aoa-fix layer).
- **Diagnosis.** Closer to the q limit, the 30 s look-ahead guard forecasts a pressure risk.
- **Mechanism.** With `lookahead_search_order: risk_first`, it re-sorts the positive alpha offsets
  **largest first**. So every ≈30 s it applies a +2° pull-up (CL 0.025 → 0.049 for 3 s), which then
  decays: 59 steps.
- **Fix.** The opt-in `lookahead_offset_order: minimal_first` tries the same offsets smallest first,
  with the same candidate set, audits and witness requirement.

| nominal, constant CL 0.025 | inner loop | range | cruise q | intra-step ptp | integrated heat load | guard offsets |
|---|---|---|---|---|---|---|
| V16 objective | true OCP | 2456.45 km | — | 1.40° | — | 19 |
| consistent (aoa-fix) | true OCP | 2475.94 km | ≈71 kPa | 0.002° | — | 16 |
| aligned | true OCP | 2461.47 km | 72.3 kPa | 0.023° | 1478 MJ/m² | 59 (+2°) |
| **aligned@minimal_first** | true OCP | **2502.61 km** | 79.8 kPa | **0.001°** | 1604 MJ/m² | 19 (all +0.5°) |
| upper bound | relaxed | 2511.03 km | 80.0 kPa | — | — | — |

- **Range.** `aligned@minimal_first` closes 26.7 km of the 35.1 km gap between `consistent` and the
  bound, i.e. 76 %. It is 46 km above the V16 objective.
- **Limits.** Peak heat flux is 1.77 MW/m² (< 2) and peak q is 80.0 kPa: the limit is reached, and every
  step is certified.
- **Cost.** Integrated heat load rises by ≈8 %. The heat load is reported for every arm.

On the production surrogate pipeline, the guard ordering alone (`consistent@minimal_first`) gives:

| | default guard | minimal_first |
|---|---|---|
| constant 0.025 | 2477.36 km | 2484.31 km |
| seed 101 | 2480.41 km | 2482.38 km |

The V16 layer gains only +0.15 / +0.46 km, because V16 cruises at 64 kPa, far from the guard's trigger.

**The two changes interact.** Removing the heat penalty helps only together with the minimal-first
guard; the guard ordering matters only when the inner loop flies close to the q limit.

### 1.3 Issue 3 — geometry and anticipatory morphing: little left to gain (negative result)

- **Bounds.** Morphing is worth at most +6.0 km over the best fixed geometry and +21.7 km over the start
  geometry. The generator's three-geometry family already reaches 2510.13 of 2511.03 km. So anticipatory
  or jointly learned geometry scheduling can add **at most 0.9 km**, under a relaxed bound.
- **Flights.** At the CL actually flown, picking the best geometry at every step would save only
  0.20–0.34 % of drag work (≲1 km) on the aoa-fix flights.
- **Decision.** The pre-declared gate (> 2 km) fails. A joint CL × geometry action space is therefore
  **not** implemented; it would add complexity without measurable benefit. The joint CL–geometry lever is
  really an altitude lever, and that sits in the CL command and the execution layer (1.2).

### 1.4 Issue 1 — why training stops

- **Stop rule.** All 7 V16 seeds stop on "no candidate accepted in 12 iterations"
  (`training.py:stop_reason`). By then each of the 100 decision times has been branched with ±1/±2
  classes in three sweeps. So the policy is a **coordinate-wise local optimum of the 19-level grid**.
- **Sharp optimum.** The median single-decision advantage under the final policy is −0.5 km.
- **Unused positive labels.** Each seed's final-policy labels still hold exact positive advantages of
  +0.28…+1.34 km. Only model-stage candidates (LCB margins in ±48 s windows) were tried, because
  `--overrides` was off in the final run.
- **Early progress only.** Almost all training gain came in iterations 0–2, i.e. finding CL ≈ 0.025 from
  the initial 0.04.
- **The policy compensated for the execution layer.** Seed 101's early manoeuvre (CL 0.04–0.055 for
  ≈150 s) beats CL 0.025 by +3.1 km under the default guard. Under the minimal-first guard it loses
  −1.9 km. So the learned "gain" partly compensated for the guard's over-correction.
- **Code.** `training.py` now records the unrealised positive labels at the stop.

### 1.5 Issue 4 — coordinated multi-decision changes

- **Why single moves fail.** Altitude and specific energy integrate CL, so the return is not separable
  over decisions. One 12 s change of ±0.005 CL mainly excites a phugoid, which gives the −0.5 km median.
  A sustained change (a new glide altitude) can be positive even when every single-decision move is
  negative.
- **Theory.** In pattern-search terms, a mesh-local optimum for coordinate directions need not be
  optimal for block directions.
- **Implementation (opt-in `--blocks 2,4,8`).**
  - A block alternative shifts π's nominal actions by ±k at B consecutive decisions, then the open-loop
    continuation resumes (`rollouts.block_alternatives`).
  - Its exact advantage is measured by forking, as today.
  - Positive block labels become exact, greedily composed, non-overlapping block-override candidates
    (`training.block_override_sets`).
  - Acceptance stays exact and monotone.
  - Training records the certified steps per iteration, so variants are compared **at equal compute**.

## 2. Pre-declared criteria (`comparisons.json`)

These were declared before any held-out-3 flight of the new arms; only nominal development flights had
been seen. Arms are flown on held-out-3 (nominal + 40 states) through the production surrogate pipeline
(`improve` stage). Paired bootstrap 95 % CIs (10 000 resamples, seed 0), as in V16.

- **E1 (execution layer).** `aligned@minimal_first` − `v16` at constant CL 0.025: CI lower bound > 0,
  with no completion loss. The decomposition rows (AoA fix, minimal-first, heat alignment, minimal-first
  alone on V16) are reported as attribution, not as pass/fail criteria.
- **E2 (RL gain on the new layer).** For each retrained seed (same unchanged training code, under
  `aligned@minimal_first`): policy − constant CL 0.025, CI lower bound > 0. The 7 V16 policies flown on the
  new layer are reported as "transfer".
- **E3 (coordinated search).** For each seed: `--overrides --blocks 2,4,8` − baseline PI, both under
  `aligned@minimal_first`, at equal certified-step budget. CI lower bound > 0.
- **Training protocol.** All training variants start from constant CL 0.025 (`--initial-cl 0.025`), the
  best constant on every layer (§1.2 and `../aoa_fix/cl_sweep_local/`). In V16, iterations 0–2 spent
  their gain finding ≈0.025 from 0.04; starting at 0.025 measures only what the learning algorithm adds.
  The other settings are the V16 defaults (25 states every 48 s, 14 candidates, open-loop continuation,
  stop rule). `blocks` uses 7 states every 168 s with `--blocks 2,4,8`: 56 branches per iteration,
  against 50 for PI, i.e. the same budget per iteration. Learning curves are compared over certified
  steps (`history[*].certified_steps`).
- **Safety and reporting.** No completion or safety regression in any arm. Integrated heat load, peak
  q, heat flux and n, fallback counts and the AoA metrics are reported for every arm, together with the
  gap to the per-state upper bound.

## 3. How to run (cloud)

```bash
OBJECTIVE=aligned SAFETY_VARIANT=minimal_first bash scripts/run_aoa_experiments.sh surrogate
bash scripts/run_aoa_experiments.sh improve          # 5 layers x 8 policies x 41 states
OBJECTIVE=aligned SAFETY_VARIANT=minimal_first TRAIN_NAME=pi TRAIN_ARGS="--initial-cl 0.025" \
  bash scripts/run_aoa_experiments.sh train
OBJECTIVE=aligned SAFETY_VARIANT=minimal_first TRAIN_NAME=overrides TRAIN_ARGS="--initial-cl 0.025 --overrides" \
  bash scripts/run_aoa_experiments.sh train
OBJECTIVE=aligned SAFETY_VARIANT=minimal_first TRAIN_NAME=blocks \
  TRAIN_ARGS="--initial-cl 0.025 --overrides --blocks 2,4,8 --states 7 --stride 168 --patience 4" \
  bash scripts/run_aoa_experiments.sh train
ARMS=aligned@minimal_first IMPROVE_POLICIES="$(for n in pi overrides blocks; do for s in 17 23 41 59 83 97 101; do \
  printf 'train_%s_seed%s ' $n $s; done; done)" bash scripts/run_aoa_experiments.sh improve
python scripts/upper_bound.py --out experiments/improve/bounds --states nominal heldout --jobs 8
bash scripts/run_aoa_experiments.sh report-improve   # -> runs/aoa/improve_report/report.md
```

## 4. Results

### 4.1 Local held-out-3 run (4-CPU container, 2026-11-01)

The local run covers the nominal state plus the first 20 held-out-3 states, under the production
surrogate pipeline. The full report is `report_local/report.md`; the bounds cover only 6 of these
states locally. Every one of the 210 flights completed 1200 s, and every executed step passed
certification.

| execution layer | CL 0.025: mean range | seed 101: mean range | heat load (CL 0.025) | intra-step ptp | cruise q | mean gap to bound |
|---|---|---|---|---|---|---|
| v16 (archived configuration) | 2455.38 km | 2465.11 km | 1370 MJ/m² | 1.225° | 63.5 kPa | 70.5 km |
| v16@minimal_first | 2456.42 km | 2467.56 km | 1372 MJ/m² | 1.259° | 63.4 kPa | 69.0 km |
| consistent (aoa-fix) | 2491.88 km | 2490.96 km | 1496 MJ/m² | 0.019° | 71.0 kPa | 33.5 km |
| consistent@minimal_first | 2496.00 km | 2495.22 km | 1505 MJ/m² | 0.020° | 70.9 kPa | 29.7 km |
| **aligned@minimal_first** | **2511.93 km** | **2512.01 km** | 1607 MJ/m² | **0.009°** | 78.8 kPa | **13.1 km** |

Declared comparisons, paired by state (21 states, 95 % paired bootstrap CI):

| comparison | mean [CI] | wins | Δ heat load | gap closure |
|---|---|---|---|---|
| AoA fix: consistent − v16 (CL 0.025) | +36.50 km [+35.34, +37.70] | 21/21 | +127 MJ/m² | 0.52 |
| minimal-first guard: consistent@mf − consistent | +4.12 km [+3.21, +5.01] | 21/21 | +9 | 0.11 |
| heat-load alignment: aligned@mf − consistent@mf | +15.93 km [+14.69, +17.12] | 21/21 | +102 | 0.56 |
| **E1 total execution layer: aligned@mf − v16** | **+56.55 km [+54.78, +58.32]** | **21/21** | +238 | **0.81** |
| minimal-first alone on V16: v16@mf − v16 | +1.04 km [+0.08, +2.02] | 12/21 | +3 | 0.02 |
| RL transfer, seed 101 − constant, V16 layer | +9.73 km [+7.98, +11.63] | 21/21 | +29 | 0.16 |
| **RL transfer, seed 101 − constant, aligned@mf** | **+0.08 km [−0.73, +0.84]** | 13/21 | −10 | 0.03 |
| system, seed 101: aligned@mf − v16 | +46.90 km [+46.23, +47.55] | 21/21 | +199 | 0.78 |

How to read this:
- **E1 is met.** The execution-layer changes add +56.6 km (+2.3 %) on every held-out state, and close
  81 % of the V16 gap to the physical upper bound.
- **The changes are not independent.** Minimal-first alone does little on V16 (+1 km); its value
  appears once the inner loop flies near the q limit. Heat alignment needs the guard fix: without it,
  the true-OCP flight loses 14.5 km.
- **The V16 policy's learned gain disappears on the corrected layer.** On the V16 layer, seed 101 gains
  +9.7 km over the tuned constant; on aligned@minimal_first it gains +0.1 km, with a CI around 0. The
  V16 "learning gain" was mostly the policy compensating for execution-layer losses: the early
  high-CL manoeuvre offsets the guard's over-correction and the heat-biased cruise.
- **The paper can claim** the execution-layer contribution, with a physical bound showing how close it
  comes to optimal. It **cannot** claim that the V16 RL policy adds range on the corrected system.
- **Whether retraining recovers a gain** is E2/E3 (§4.2). At most ≈13 km remain against the
  (optimistic) bound, and most of that sits in the pull-out phase.
- **Price.** Integrated heat load rises by 17 % (238 MJ/m²). Peak heat flux stays ≤ 1.78 MW/m²
  (limit 2.0) and peak q = 80.0 kPa (limit, certified).

### 4.2 Retraining on aligned@minimal_first (local: 1 seed per variant, 10 iterations)

Both variants start from constant CL 0.025 and use the V16 training settings otherwise (§2). PI is the
unchanged V16 algorithm. "Blocks" adds exact overrides and coordinated blocks of 2/4/8 decisions, with 7
states every 168 s, i.e. 14 + 42 branch flights per iteration against PI's 50.
Evidence: `train_local/`, `flights_trained/`; the report is `report_local/report.md`.

| seed 17 | certified steps | stages accepted | nominal gain (training) | held-out gain over constant (21 states) |
|---|---|---|---|---|
| PI | 174,700 | 3 of 10 iterations | +1.70 km | **+2.10 km [+1.68, +2.46]**, 20/21 states |
| blocks + overrides | 163,324 | 8 of 10 iterations | +2.67 km | +1.73 km [+1.40, +2.06], 21/21 states |

Paired comparison at equal budget (E3): blocks − PI = **−0.38 km [−0.79, +0.05]**, blocks better on
5 of 21 states. Both runs completed all 21 flights, and the safety-related quantities are unchanged
(peak heat flux ≤ 1.78 MW/m², no failed or refused steps).

Reading:
- **E2 holds for the PI baseline (one seed).** The unchanged V16 algorithm, retrained on the corrected
  layer, learns a real but small gain: +2.1 km over the tuned constant, i.e. 0.08 % of range and ≈16 %
  of the remaining 13 km gap to the bound. The V16 policy's "+9.7 km" was therefore mostly layer
  compensation (§4.1); the genuinely learnable part on the corrected layer is about a fifth of that.
- **E3 is not met.** The coordinated-block variant found larger gains at the nominal state it trained on
  (+2.67 vs +1.70 km), but they do not carry over to the held-out states. The reason is structural:
  accepted stages are time-keyed overrides, i.e. a learned nominal schedule, which does not adapt to
  perturbed starts. PI's model stages condition on the state and generalise better. So the
  nominal-state number the training loop optimises was misleading for the blocks variant.
- **Issue 1 (premature stop):** no variant stopped early in these 10 iterations; PI stopped
  improving after iteration 3 and blocks kept finding exact improvements until the end. With state-
  independent overrides these are overfit to one trajectory, so more exact labels at one state do
  not help the held-out policy.
- **Open questions, not claims:** a single seed per variant gives no seed variance, and a 21-state
  held-out subset is not the 41-state set. The 7-seed cloud comparison (§3) decides E2/E3 properly.
- **Likely next step if E3 stays negative:** make overrides state-gated (`tol`, already supported by
  `policy._override`) or train blocks as model stages with state inputs, so that coordinated
  improvements generalise. This was not tested.

### 4.3 Total attribution (nominal-start flights are the common ground; held-out means over 21 states)

| source | effect | evidence |
|---|---|---|
| AoA sawtooth fix (consistent objective + refitted surrogate) | **+36.5 km** [+35.3, +37.7] | §4.1 |
| minimal-first look-ahead guard | +4.1 km [+3.2, +5.0] | §4.1 |
| heat-load alignment (needs the guard fix) | +15.9 km [+14.7, +17.1] | §4.1 |
| **execution layer in total** | **+56.5 km** [+54.8, +58.3] | §4.1 |
| RL policy on the corrected layer (PI, 1 seed) | +2.1 km [+1.7, +2.5] | §4.2 |
| coordinated blocks over PI (1 seed) | −0.4 km [−0.8, +0.05] | §4.2 |
| gap left to the physical upper bound | ≈11 km (aligned, PI) | bounds |

So of the ≈ 59 km gained over V16 + constant CL, about 96 % comes from the execution layer and about
4 % from learning. The paper should say so plainly.

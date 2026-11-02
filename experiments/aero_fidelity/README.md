# Does matching the design Mach pay off? A decision experiment for the geometry library

## 1. Why this experiment

The study's premise: a waverider is strongest near its design Mach and loses performance away from it, so
switching to the geometry that is best for the current flight condition should raise the total payoff.
The existing 298-geometry library does not show this.

Measured on the project's own data (`experiments/improve/`, this folder's analysis scripts):

| observation | evidence |
|---|---|
| With pressure drag only, ideal switching gains 1.08 % | `data/geometry_subset/benchmarks/summary.json` |
| Every winning geometry has design Mach 8–15; design Mach 4–7 never wins a condition, even at flight Mach 3.5–5 | same file, `winner_counts` |
| With skin friction added, ideal switching gain drops to 0 %: one geometry (S2_MD12) is best at every Mach | `scripts/analysis/morphing_value.py`, `data/aero_friction/` |
| At equal volume (0.95–1.05 × G42), a design-Mach-14.4 geometry has the highest L/D from Mach 4 to 8 | this folder, analysis in the session notes |
| Physical range bound at the nominal state, pressure + friction: free CL, all 298 geometries **2168.5 km**; free CL, G42 only **2158.1 km** (morphing worth at most 10 km, 0.5 %); best constant CL, all geometries **2133.4 km** (CL scheduling worth 35 km, 1.6 %). Without friction the same bound is 2511 km | `src/certified_flight/bound.py`, `scripts/upper_bound.py --friction data/aero_friction/friction_table.npz [--constant-cl]`, results in `bounds_friction/` |

Two causes, both in how the library was built and evaluated, not in the idea:

1. **The library is not a controlled family.** Its 10 generator inputs vary together (shock angles differ by up
   to 10° at one design Mach, volume 0.85–2.6 × G42), so what is compared is mostly slenderness, not
   design-point matching.
2. **Off-design aerodynamics are computed with a local-inclination method.** The snapshot's "P2 strip
   shock-expansion" model gives every surface panel the state of an attached two-dimensional wedge flow. An
   independent per-panel tangent-wedge + Prandtl–Meyer implementation (`src/certified_flight/friction.py`)
   reproduces the snapshot's CL to 3e-4 and CD to 1 % (`tests/test_aero_fidelity.py`). Such a method has no
   leading-edge shock detachment and no spillage. For a geometry designed for a high Mach and flown at a low
   one, the attached shock it assumes cannot exist, so the model hides exactly the penalty the premise needs.
   The project's own `upstream_matlab/source/tools/v6_aero_physics_validation/p4_shock.py` measures the shock
   mismatch but does not feed it back into the forces.

Skin friction alone does not create the effect (it removes the 1 % that remained); it only makes the
snapshot's L/D physical (8.6–9.7 instead of 15–16; the Küchemann estimate is 5.5–7).

## 2. The experiment

A small family where **only the design Mach differs**, evaluated with Euler CFD, which does capture shock
detachment and spillage at the inviscid level.

**Family** (`family/`; the unmodified generator `upstream_matlab/source/matlab/active/build_geometry_core.m`):
- design Mach M_d ∈ {4, 6, 8};
- leading-edge and planform inputs fixed to those of G42 (c1 = c2 = 1.0331, d1 = 0.11607, design AOA −4°);
- conical shock: a1 = a2 = b1 = b2 = asin(1/M_d) + δ, with one common δ ∈ {−3,…,+4}° for all members;
- exit-deflection coefficient a_x ∈ {0.970, 0.985, 1.000, 1.015, 1.030}, used to match the volume.
  Version 1: 120 generator calls; version 2 (§3b): 180; version 3 (§3c): 189; version 4 (§3d): 29.

**Selection rule, version 1** (`family/check_family.py`; see §3b for the amendment): a member is valid if its surface is watertight with
positive volume. A δ is admissible if all three design Machs have a valid member and the chosen members'
volumes (per M_d the one closest to the median volume V* of the a_x = 1 members) are within ±3 % of V*. The
admissible δ with the smallest maximum deviation wins. If none is admissible the experiment stops with
`NO_ADMISSIBLE_FAMILY`; the rule is not relaxed.

**Flow cases:** the 3 members × flight Mach {4, 6, 8} × alpha {2, 4, 6}° = 27 SU2 Euler cases at the altitude
where q = 50 kPa for each Mach (`scripts/cfd/make_su2_cases.py`), medium mesh. Grid check: one member at one
Mach on coarse, medium and fine meshes.

**Viscous drag** is the same function for both inviscid models (`friction.friction_coefficients`, Eckert
reference temperature, laminar/turbulent/transition, T_w = 1000 K, Re_tr = 5e6), evaluated from each
model's own edge state. The tangent-wedge model is evaluated on the same files for the comparison.

**Equal-lift comparison** (`scripts/cfd/collect_su2.py`): per member and Mach, a quadratic fit of
CD_total(CL) through the three alphas, evaluated at CL = 0.03, 0.04, 0.05 (inside the fitted range only);
the ranking is by total L/D.

## 3. Decision criteria (declared before any CFD result)

**The premise holds** if, for at least two of the three CL targets, the Euler + friction L/D is highest for
the M_d = 4 member at flight Mach 4 **and** for the M_d = 8 member at flight Mach 8, each leading the worst
family member at that condition by at least 3 %.

Reported alongside, whatever the outcome:
- the same ranking from the tangent-wedge model (expected: no design-Mach dependence, i.e. the model defect);
- the ideal switching gain over the best single member, for Euler and for the wedge model;
- grid convergence of CL and CD between mesh levels;
- convergence status of every case (residual and iteration count), because unconverged cases are excluded
  from the comparison and listed.

**If the premise holds:** the next stage is a design-point-optimal family (M_d = 4…8 in steps of 0.5, same
constraints), with continuous morph paths checked by generating the intermediate geometries
(`upstream_matlab/source/results/v5_multimach_geometry_db_v2/code/stage4_morph.py`), a multi-fidelity
aerodynamic database (tangent wedge + shock-mismatch correction calibrated on an Euler design of
experiments), and a re-run of the flight pipeline.

**If it does not:** the report states that, in this geometry class, design-point matching is not worth much
at the inviscid level, and the case for morphing has to rest on other mechanisms (trim, heating, a wider
Mach range, payload volume), which the point-mass model does not contain.

## 3b. Result of family version 1 and protocol amendment 1

**Version 1 was run first** (120 generator calls, `results_family_v1/`, evidence branch
`aero-family-results-20261102`, commit `2acaaa5`):
- 105 generated, 15 failed with the generator's own `V5:GeometryDomain` error (δ = −3° for MD6 and MD8, and
  δ = +2…+4° at a_x = 0.970 for MD6 and MD8).
- The check stopped on a base-outline error. Cause found afterwards: **every one of the 105 members has
  intersecting upper and lower skins at the tail.** The project's own frozen native-mesh audit, ported in
  `family/native_audit.py` (verified: G42 → VALID; FAM_MD6_d+0_ax1.000 → INVALID, 196 triangle-pair
  intersections, tail minimum thickness −0.035 m), classifies **0 of 105 as VALID**, 105 as INVALID.
- Why: the generator builds the upper skin as a straight free-stream line from the leading edge
  (`build_geometry_core.m`, lines 474–476), while the lower skin follows the streamlines behind the shock and
  drifts sideways. The drift at the base is 0.09–0.125 m for the library's valid members and 0.02–0.39 m,
  growing with δ, for version 1. With four equal shock angles the lower edge crosses the upper edge.
  The generator is not at fault; version 1 chose a parametrisation outside its valid region.
- Outcome under the declared rule: **NO_ADMISSIBLE_FAMILY**. Nothing was relaxed.

**Amendment 1** (declared before any CFD result exists; no CFD has been run on any family member):
- Design variable: the shock is described by the mean angle offset δ and the ratio r = b/a of the lower to the
  upper shock angle pair, a1 = a2 = a, b1 = b2 = r·a with mean(a, r·a) = asin(1/M_d) + δ, r ∈ {0.50, 0.65,
  0.80}, δ ∈ {−2, 0, 2, 4, 6}°; plus the angle pattern of the validated reference geometry G42 scaled to the
  same mean angle. a_x ∈ {0.97, 1.00, 1.03}. 180 generator calls (`family_jobs.py --version 2`).
- Validity: the frozen native-mesh audit must classify the member VALID (this replaces "watertight and positive
  volume", which is implied but weaker), and a watertight closed surface must be constructible.
- Selection rule: unchanged in form. A candidate is a design key shared by the three design Machs (now
  `pair_d·_r·` or `g42_d·` instead of δ alone); volumes within ±3 % of V*; smallest maximum deviation wins;
  no relaxation; `NO_ADMISSIBLE_FAMILY` is a valid outcome.
- The decision criteria in §3 are unchanged.
- Not known in advance: whether version 2 has any admissible member. The largest shock angles (up to 31.7° for
  MD4 with r = 0.5) are far above the 20° upper bound used for the project's validated members and are
  expected to fail at generation; that is a boundary of the design space, not an error.

## 3c. Result of family version 2 and protocol amendment 2

**Version 2 was run** (180 generator calls, evidence branch `aero-family-v2-results-20261102`, copy of the summary in
`results_family_v2/`): 110 generated, 70 failed at generation (`V5:GeometryDomain`, invalid shock or isentropic
streamline), and the frozen native-mesh audit classified 37 VALID, 73 INVALID. Outcome under the declared rule:
**NO_ADMISSIBLE_FAMILY**. Nothing was relaxed. (An authorised MATLAB-side fix for heterogeneous job records, the
`iscell` branch in `make_family.m`, was needed to start the run; it does not touch the generator or any input.)

What the data show:
- All 37 VALID members have r = 0.5 except 3 with r = 0.65 (MD8, d = -2). No member with r = 0.8 or with the G42
  shock-angle pattern was VALID.
- Only `pair_d+2_r0.5`, `pair_d+4_r0.5`, `pair_d+6_r0.5` have a VALID member at all three design Machs; their best
  volume deviations were 59.5 %, 27.4 % and 15.6 %.
- The cause is the volume knob, not the family: over a_x = 0.97...1.03 the enclosed volume changes by about 35 %
  (e.g. MD6, `pair_d+6_r0.5`: 1.395 to 1.882 m3), so three a_x values cannot land inside +-3 %.

**Amendment 2** (declared after seeing the version-2 audit, before any CFD result; no CFD has been run on any member):
- Shock parametrisations: exactly the three keys above (those with a VALID member at every design Mach in version 2;
  `tests/test_aero_fidelity.py` re-derives them from `results_family_v2/family_audit.json`).
- Volume knob: a_x = 0.90, 0.91, ..., 1.10 (21 values), 189 generator calls (`family_jobs.py --version 3`).
- Target volume V*: the enclosed volume of G42, 1.018814 m3 (equal payload volume), replacing the median of the
  a_x-closest-to-1 members. The median rule would put V* at a value the three design Machs cannot reach together
  for these keys; the G42 volume is also the physically meaningful reference.
- Unchanged: validity = frozen audit VALID plus a constructible watertight surface; the tolerance +-3 %; the
  candidate with the smallest maximum deviation wins; no relaxation; `NO_ADMISSIBLE_FAMILY` remains a valid outcome;
  all decision criteria of section 3.
- Not known in advance: whether a_x outside 0.97...1.03 stays VALID. Generation failures or INVALID audits at the ends
  of the range are possible, in which case the outcome is again `NO_ADMISSIBLE_FAMILY` and is reported as such.
- Limitation to carry into any write-up: the family is a one-sided design (r = 0.5, lower-surface shock angle half
  the upper one), because that is where this generator produces audit-VALID members. Conclusions apply to that class.

## 3d. Result of family version 3 and protocol amendment 3

**Version 3 was run** (189 generator calls, evidence branch `aero-family-v3-results-20261102`, copy in
`results_family_v3/`): 106 generated, all 106 audit-VALID, 83 failed at generation (74 invalid isentropic streamline,
9 characteristic iteration cap). Outcome under the declared rule: **NO_ADMISSIBLE_FAMILY**, best volume deviations
38.5 % (`pair_d+2`), 32.6 % (`pair_d+4`), 66.7 % (`pair_d+6`). Nothing was relaxed.

Why: at one shock offset delta, a_x moves the volume only inside the window where the generator works, and the
windows of the three design Machs do not intersect. Valid volume ranges [m3] (G42 = 1.019):

| delta | MD4 | MD6 | MD8 | gap between MD8 minimum and MD4 maximum |
|---|---|---|---|---|
| +2 | 0.10-0.63 | 0.57-1.21 | 0.91-1.34 | 0.28 |
| +4 | 0.39-1.18 | 1.05-1.59 | 1.35-1.71 | 0.17 |
| +6 | 0.77-1.60 | 1.40-1.94 | 1.70-2.04 | 0.10 |

so "one common delta for all three members" cannot reach a common volume near G42, which is a property of the
requirement, not of the volume knob. The common delta was never part of the scientific question (design Mach is).

**Amendment 3** (declared after the version-3 audit, before any CFD result; chosen from volumes and validity only,
never from aerodynamic results):
- Each design Mach gets its own shock offset: the weakest delta of the audited set {+2, +4, +6} whose valid a_x window
  reaches G42's volume. From the version-3 audit: MD4 +4 deg, MD6 +2 deg, MD8 +2 deg. Shock angle pairs (upper/lower,
  deg): MD4 24.6/12.3, MD6 15.5/7.7, MD8 12.2/6.1 (r = 0.5 as before).
- a_x scanned finely around the match (`family_jobs.py --version 4`, 29 generator calls): MD4 1.020-1.060 (step 0.005),
  MD6 1.000-1.040 (0.005), MD8 0.975-1.000 (0.0025). The closest version-3 members are MD4 -0.6 %, MD6 -1.9 %,
  MD8 +3.4 % from G42's volume; MD8 needs the finer a_x.
- Selection (`check_family.select_per_mach`): per design Mach, the VALID member closest to V* = 1.018814 m3; the family
  is admissible only if all three are within +-3 %. Validity (frozen audit), tolerance and all decision criteria of
  section 3 are unchanged.
- Confound to state in any write-up: the three members differ in design Mach, in shock offset (+4, +2, +2 deg, i.e.
  0.28, 0.21, 0.28 of the Mach angle) and slightly in a_x. They are equal in volume, planform inputs, length and
  design AOA. This is a volume-constrained design-point family, not a one-variable sweep.
- Not known in advance: whether a_x between the audited points stays VALID and reaches +-3 % at MD8.

## 3e. Result of family version 4: the declared test is NOT met, and its data are compromised

**Family selected** (version 4, 29/29 generated and audit-VALID): MD4 `pair_d+4_r0.5` a_x = 1.040 (volume -0.60 % from G42),
MD6 `pair_d+2_r0.5` a_x = 1.025 (+0.89 %), MD8 `pair_d+2_r0.5` a_x = 0.985 (-0.02 %). All within +-3 %.
**CFD:** 27 medium cases plus 2 grid-check cases (coarse, fine) = 29 cases, all met the frozen convergence rule
(cfd-mesh 656 s, cfd-run 3077 s on 10 cores). Copies of the summary, decision and audit are in `results_family_v4/`.

**Formal outcome of the declared criterion (section 3): premise NOT met, 0 of 3 CL targets.** The ideal switching gain is
0 % for both Euler and the wedge model (MD6 best throughout in the filtered set).

**This outcome cannot be read as a refutation of the premise, because the primary measurement fails basic checks:**
1. 12 of 29 Euler cases have a NEGATIVE pressure drag (down to CDp = -0.0040). Steady inviscid supersonic flow without
   thrust cannot have negative drag, so the Euler CD contains an error of the size of the physical drag (0.002-0.008).
2. The equal-lift L/D values are therefore meaningless in places (+58, -80 at CL = 0.03, M8): they divide by a total CD
   near zero or negative. Two total-CD entries after the quadratic fit are non-positive.
3. Grid check (FAM_MD6, M6, alpha 4): CL 0.0354 / 0.0369 / 0.0381 (coarse / medium / fine), pressure CD -0.0038 /
   +0.00007 / -0.0005. CL changes 3 %; CD changes sign.
4. FAM_MD8 at M8 lacks the CL = 0.05 target (alpha range too small for it), so the filtered switching-gain statistic does not
   cover all three members.

**What the data say about the source of the error** (computed from the archived tables, not part of the declared test):
- Euler minus wedge pressure drag is nearly independent of alpha for every (member, Mach) pair: standard deviation over
  alpha = 0.0001 to 0.0003, while the offset itself is +0.0023 ... -0.0048.
- The alpha-dependence of the drag agrees: increase of CDp from alpha 2 to 6 is Euler / wedge = 0.00902 / 0.00824 (MD4, M4),
  0.00571 / 0.00603 (MD6, M6), 0.00320 / 0.00349 (MD8, M8), i.e. within 1-12 %. CL agrees within 0.001-0.007.
- The offset scales like (open-base area / S_ref) times a pressure coefficient that grows with Mach (it divides to -0.04 at
  M6 and -0.07 at M8 using the base-cap area of each member). A pressure on the closed base cap above the free-stream value
  would produce exactly this; it is not physical for an aft-facing face and must come from how the open tail is closed and
  meshed (the base cap is part of the monitored `body` marker).
- This is a diagnosis from the integrated coefficients; the per-surface force split was not written, so it is NOT yet verified.

**Consequence for the protocol.** The decision criterion was declared in terms of total Euler CD. Changing it now, after seeing
that it is not met, would be a post-hoc change. The honest handling is:
- the primary test stands as reported: not met, with its data-quality defect documented here;
- a follow-up experiment (version 5 of the CFD stage, NOT a new geometry family) is declared before it is run: the same three
  geometries and 27 + 2 cases, with the base cap as its own marker excluded from force monitoring so that CL and CD
  are the skin forces only (the usual forebody convention; the base is not predicted by an Euler model), the old total minus the new skin
  force giving the base term as a check of the diagnosis above. Its criteria are those of section 3 applied to the skin-only
  forces, and its result is reported as exploratory, following a primary test whose measurement was flawed.

## 4. Known limits

- Euler is inviscid: no viscous interaction, no boundary-layer displacement at the leading edge, no
  transition prediction. Viscous drag is an add-on estimate.
- Real-gas effects and wall-temperature coupling are not modelled; the hypersonic range here (Mach 4–8)
  keeps them modest.
- Three members at three Machs is a decision experiment, not a database: it can falsify the premise or
  justify the larger effort, not size the benefit.
- The tangent-wedge viscous edge state is the panel's own inviscid state; a boundary-layer solver would
  differ by some percent. This affects both models alike.

## 4b. Known issue: volume meshing of some geometries (read before running `cfd-mesh`)

Found in a local dry run with stand-in geometries from the 298 library (the new family has not been generated
here, since MATLAB is unavailable locally):
- gmsh Frontal 3D (algorithm 4) crashes (segmentation fault) on these surfaces; it is not used.
- For one stand-in (V2_MD8_018) at the medium level, gmsh re-meshing of the surface produced "overlapping
  facets on surface 4" with both HXT and Delaunay, for every edge-size scale tried, and the result was an empty
  mesh. Two others (V2M_MD4_044 at coarse/medium/fine, V2_MD6_008 at medium) meshed fine (97k / 328k / 1.25M tetrahedra).
- `scripts/cfd/make_su2_cases.py` now treats an empty mesh as a failure and, if re-meshing fails, falls back to
  using the validated STL triangulation as the surface mesh (`remesh_surface=false`, recorded in `mesh_info.json`
  and `case.json`). **This fallback has not been tested yet.** If `cfd-mesh` fails for a family member, send the
  error list it prints; do not change the geometry.
- Because the surface resolution differs between the two paths, the grid-convergence check is only meaningful
  among meshes with the same `remesh_surface` value; `collect_su2.py` lists the value per case.

## 5. How to run (cloud server)

See `CODEX.md`. In short: `bash experiments/aero_fidelity/run_decision.sh all`, which stops at the first
failed stage and prints what it needs (MATLAB with the Optimization and Symbolic Math toolboxes, SU2, MPI).

## 6. Files

- `family/family_jobs.py`, `family/make_family.m`, `family/check_family.py`, `family/native_audit.py`: the family, its
  selection and the ported frozen audit. `results_family_v1/`: the first run and its outcome.
- `run_decision.sh`, `CODEX.md`: stages and agent instructions.
- `../../scripts/cfd/make_su2_cases.py`, `collect_su2.py`: surfaces, meshes, SU2 inputs; comparison and decision.
- `../../src/certified_flight/friction.py`, `../../scripts/build_friction_table.py`, `../../data/aero_friction/`:
  viscous drag and its table for the 298 snapshot geometries.
- `../../scripts/analysis/morphing_value.py`, `../../src/certified_flight/bound.py`: the evidence in §1.
- `../../upstream_matlab/`: the historical generator and aerodynamic sources (verified by `SHA256SUMS`).

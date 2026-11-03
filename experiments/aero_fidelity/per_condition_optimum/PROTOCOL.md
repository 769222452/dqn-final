# Per-condition optimal shapes and the ideal morphing gain: protocol (written before any run)

Status: **protocol only.** Nothing in this folder has been run. The optimiser is the owner's own MATLAB code
(`upstream_matlab/supplemental/mat_code/source/`, byte-preserved, unmodified; the driver is
`surrogate_assisted_ga_wave_rider_6.m`, which sits in `upstream_matlab/source/data/source_teacher/`). Any modified copy goes in a
separate folder together with a recorded diff; the archived files are never edited.

## What the source code does today (read from the archived files, not executed)
- Objective: `L_D = Force_z/Force_x` from `force_lower.m` and `force_upper.m`, which sum pressure x area x normal.
  **Pressure only, no skin friction.** Design point: `Ma = 15`, `AOA = -4`, `c1 = c2 = 1.0331` fixed in the driver.
- Second objective: `Vol_eff` from `vol_eff_sim.m`: at 50 sections between 30 % and 100 % of the length, the ellipse that fits inside the
  upper and lower contour, integrated along x. This is **usable (inscribed) volume**, not the enclosed volume `Vol` (`volume.m`).
- Search: Latin-hypercube start, Gaussian-process surrogate, NSGA-II on the surrogate, expected-hypervolume infill.
  Saved run: 1090 evaluated points, 52 Pareto designs (L/D 5.4-11.2).
- `compare_results52.txt`: the library's P2 aerodynamic model reproduces the generator's design-point L/D of those 52 designs within
  -3 % to +0.6 %.

## Decisions (made with the owner)
| | decision | reason |
|---|---|---|
| D1 objective | **includes skin friction**, the model used in every earlier comparison (tangent wedge + friction, which agrees with Euler to 3-8 % in L/D); a pressure-only run is kept as a control | optimise and judge with the same model |
| D2 volume | **`Vol_eff` is the constraint**, target = `Vol_eff` of G42, tolerance +-3 %; enclosed volume is reported alongside | payload space is what matters; equal enclosed volume does not mean equal usable volume |
| D3 operating points | flight Mach {4, 5, 6, 7, 8} x CL {0.03, 0.04, 0.05} = 15 points, free stream at q = 50 kPa; no extrapolation outside the validated range | inside the database support (3.5-8.25) and the Euler-checked range |
| D4 fixed-shape baseline | the best single shape found by **the same optimiser with the same budget and seeds**, maximising the mean L/D over the 15 points weighted by mission time at each Mach; equal weights as a sensitivity run | the baseline must be optimised too, or the gain is inflated |

**Ideal morphing gain** = (L/D of the shape optimised for one operating point) / (L/D of the best fixed shape at that point) - 1,
averaged over the points that share a CL, and reported per CL. It is an upper bound: it ignores the 8 s morph time and any weight cost.

## Prerequisites (to be done in order, by the owner on the MATLAB machine)
0. Check on G42: `vol_eff_sim` on `native_geometry.mat` gives the same value as the second output of `waveRiderEval` for G42's ten inputs
   (confirms the input format), and record G42's `Vol_eff` (the D2 target). If they differ, stop and report.
1. Decide how friction enters the objective (inside MATLAB, or by calling the validated Python module). The route is the owner's and the
   advisor's decision; the model must be the one named in D1.
2. For each flight condition the four shock angles must exceed the local Mach angle; the search box is set relative to the Mach angle
   (earlier family runs: the audit-VALID members had a lower/upper shock-angle ratio near 0.35-0.5), not copied from the Mach-15 box 3-8.
   The values are fixed before the first run and written next to the results.
3. Every returned shape goes through the frozen native-mesh audit (`experiments/aero_fidelity/family/native_audit.py`); only VALID
   shapes count.

## What is fixed before any run, and reported afterwards
- Budget and stopping rule (hypervolume change below a stated threshold for a stated number of generations), identical for the
  per-condition runs and the fixed-shape runs.
- At least 5 independent seeds per optimisation problem; every seed is reported, not only the best.
- Whether the optimum lies on the boundary of the search box (and which side).
- The convergence curve of each run, the number of generator failures, and the MATLAB and toolbox versions.

## Decision rule (declared now)
Using the median over seeds, and the spread between seeds as the uncertainty:
- gain below 3 % at two or more of the three CL values: **no detectable morphing gain** in this class;
- gain between 3 % and 8 % at two or more CL values: **within the model uncertainty** (wedge + friction against Euler is 3-8 % in L/D);
  reported as unconfirmed;
- gain above 8 % at two or more CL values: **supported by the model**; Euler confirmation is then required before it is relied on.
Euler is deferred on purpose: the wedge + friction model agrees with Euler to a few percent and ranks shapes identically in all tests so far.
This rule is not changed after seeing results; the data, including negative outcomes, are reported in full.

## Limits to state in any write-up
- The result applies to this generator, this search space and these 15 operating points.
- Friction is an engineering estimate (Eckert reference temperature, wall 1000 K, transition Reynolds number 5e6), not viscous CFD.
- The gain is for ideal switching and has no morph-time or weight cost.

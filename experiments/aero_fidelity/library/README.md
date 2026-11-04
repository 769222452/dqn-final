# Design-point library: recipe and plan (not yet run)

Goal: a geometry library whose construction a reviewer can follow, and whose members differ by design Mach, not by an
arbitrary mix of ten generator inputs. Status: **plan only**. Stage A has a job list; nothing here has been executed, and the
decision experiment (`../README.md` section 3e/3f) has to say first whether design-Mach matching pays at all.

## Recipe (what the family experiments established)
- Generator: the unmodified `upstream_matlab/.../build_geometry_core.m`; length 5 m.
- Fixed to G42: c1 = c2 = 1.0331, d1 = 0.11607, design AOA = -4 deg.
- Shock: upper pair a, lower pair b = r a with r = 0.5. Other ratios (0.65, 0.8) and the G42 pattern gave no audit-VALID member in
  family version 2. Mean shock angle = asin(1/M_d) + delta.
- Volume: equal to G42's enclosed volume 1.0188 m3, +-3 %, chosen with a_x (valid window roughly 0.95 to 1.06; the volume changes by
  about 35 % over it) and, where the window does not reach, delta.
- Validity gate for every member: the project's frozen native-mesh audit = VALID, plus a constructible watertight surface.
- Design variable: M_d = 4.0, 4.5, ..., 8.0 (9 members). The shock offset is a consequence of the volume constraint
  (family version 4: +4, +2, +2 deg at M_d = 4, 6, 8), and has to be stated as such in any write-up.

## Stages
1. **Stage A, volume map** (`library_jobs.py`): 9 M_d x delta 1..6 x a_x {0.95, 0.975, 1.0, 1.025, 1.05} = 270 generator calls,
   about 10 minutes of MATLAB. Audit and volume of each member via `family/check_family.py`.
2. **Stage B, member generation** (to write once the map exists): per M_d interpolate (delta, a_x) to the target volume, rule: the
   weakest delta whose valid a_x window reaches the target; generate, audit, accept if within +-3 %.
3. **Morph paths** between neighbours: intermediate geometries generated and audited with the method of
   `upstream_matlab/source/results/v5_multimach_geometry_db_v2/code/stage4_morph.py`; a member pair is "morphable" only if every
   intermediate passes the audit.
4. **Aerodynamics, multi-fidelity**: Euler with skin-only forces (`make_su2_cases.py --base-marker`) for 9 members x Mach
   {4,5,6,7,8} x alpha {2,3,4,5,6} = 225 cases, about 7 hours on the 14-core server at the measured 51 min per 27 cases;
   tangent-wedge + friction (`src/certified_flight`) as the low-fidelity model, corrected to the Euler set.
5. **Re-run the flight pipeline** (upper bound, execution layer, policies) on the new database.

## Honest limits
- r = 0.5 is where this generator produces VALID members; conclusions apply to that class of waverider.
- Equal volume is a requirement of the study (payload), not a law; other volumes would give another library.
- Stage 4 depends on stage 3's morph validity and on whether the decision experiment supports design-Mach matching.

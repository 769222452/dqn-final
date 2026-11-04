# Aerodynamic database and geometry generation code

Code-only extract of everything that generates the geometries and the aerodynamic database, plus the tools that audit and evaluate them.
Created on 2026-11-04. The full project (flight control, learning, tests, data) is on the `project-latest` branch; the results of every
check are on `evidence-archive`.

## Contents

| path | what it is |
|---|---|
| `upstream_matlab/source/matlab/active/` | the geometry generator (`build_geometry_core.m`, osculating-cone method) and the off-design aerodynamic evaluation (`offdesign_aero_v4.m`) used for the 298-geometry library |
| `upstream_matlab/source/` (rest), `SHA256SUMS`, `verify_snapshot.py` | the archived snapshot (203 files) of the MATLAB geometry database, with the audit and validation tools and a SHA-256 manifest |
| `upstream_matlab/supplemental/waveRiderEval/` | the original `waveRiderEval.m` (design-point L/D and effective volume), byte-preserved |
| `upstream_matlab/supplemental/mat_code/` | the complete `D:\mat\.code` directory (127 files, byte-preserved): generator, force and volume functions, the surrogate-assisted NSGA-II optimiser `surrogate_assisted_ga_wave_rider_6.m` is in `upstream_matlab/source/data/source_teacher/` |
| `experiments/aero_fidelity/family/` | controlled-family job lists (versions 1 to 6), the MATLAB batch (`make_family.m`) and the frozen native-mesh audit port |
| `experiments/aero_fidelity/library/`, `optimum/`, `per_condition_optimum/` | design-point library recipe, optimal-family evaluation and the per-condition optimum protocol |
| `experiments/aero_fidelity/README.md`, `CODEX.md`, `run_decision.sh` | the declared criteria, amendments and the stage runner |
| `scripts/cfd/` | SU2 case generation (`make_su2_cases.py`) and collection (`collect_su2.py`) |
| `scripts/build_friction_table.py`, `src/certified_flight/friction.py` | skin-friction model (validated against Euler) and its table builder |

## Running

The MATLAB parts run on their own (MATLAB R2023a with the Optimization and Symbolic Math toolboxes). The Python tools import
`certified_flight` and read `data/` of the project: check this branch's folders out into a `project-latest` working tree, or copy them over.
`upstream_matlab/` is verified with `python upstream_matlab/verify_snapshot.py`; its original 203-file manifest is unchanged.

## History

Two parents are merged in (`matlab-geometry-database-20261101` and `matlab-optimization-20261103`), so the history of the generator import,
the full MATLAB directory upload and the experiment code stays reachable.

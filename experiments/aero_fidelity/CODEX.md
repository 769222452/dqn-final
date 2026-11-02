# Instructions for the agent running the design-point decision experiment

Goal: build a small controlled family of waverider geometries, run Euler CFD on it, and report whether the
design-Mach-matching premise holds, using the criteria in `README.md` §3. Do not change a criterion, the
selection rule, the family definition or any generator source. Do not substitute tools silently: when
something is missing, stop and report it.

Machine: 14 cores. Use `NP=7 JOBS=2` (two cases in parallel, seven MPI ranks each) unless `nproc` says
otherwise. Keep the machine otherwise idle.

## Steps

1. Get the branch: `git fetch origin aoa-fix && git checkout aoa-fix && git pull`. If you commit anything,
   push it to a separate branch; do not push to `aoa-fix`.
2. Python environment: `python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt gmsh`.
   Check: `python -m pytest -q tests/test_aero_fidelity.py` must pass.
3. Preflight: `bash experiments/aero_fidelity/run_decision.sh preflight`. It checks the integrity of
   `upstream_matlab/` (SHA-256), MATLAB with the Optimization and Symbolic Math toolboxes, SU2_CFD (>= 7),
   mpirun, and the Python packages.
   - Missing SU2: try `conda install -c conda-forge su2`; if that is not possible, build SU2 from source.
     Report which one you used and its version.
   - Missing MATLAB: stop and report. The `family` stage can be run on another machine (the command is
     printed by preflight); copy the resulting folder back to `runs/aero_fidelity/family`.
4. Run the stages one at a time and read the output of each before the next:
   ```bash
   export NP=7 JOBS=2 SLICES=4
   bash experiments/aero_fidelity/run_decision.sh family      # version 3: 189 generator calls, resumable
   bash experiments/aero_fidelity/run_decision.sh check       # selection rule; may legitimately end in NO_ADMISSIBLE_FAMILY
   bash experiments/aero_fidelity/run_decision.sh cfd-mesh
   nohup bash experiments/aero_fidelity/run_decision.sh cfd-run > runs/aero_fidelity/cfd_run.log 2>&1 &
   bash experiments/aero_fidelity/run_decision.sh collect
   ```
5. Versions 1 and 2 of the family already ran (`results_family_v1/`, `results_family_v2/`): no admissible member.
   Version 3 (protocol amendment 2, README §3c) is the one to run now. `NO_ADMISSIBLE_FAMILY` is a result, not a failure to work around: send `family_audit.json` back and stop.
   Do not widen the scan, change δ values or relax the ±3 % volume tolerance.
6. Convergence: a case is usable if its `history.csv` ends with the density residual below 1e-5 or the
   lift and drag Cauchy criteria met. If cases do not converge, report their ids and residuals. You may
   rerun them with `--iters` increased through `make_su2_cases.py`, but say so. Do not drop cases to change
   the outcome.
7. Send back (as a single archive): `runs/aero_fidelity/results/` (`decision.json`, `decision.txt`,
   `family_audit.json`), `runs/aero_fidelity/family_selected/` (the three `.mat` files with
   `select.json`), every `case.json`, `case.cfg`, `history.csv` and `su2.log` under `runs/aero_fidelity/cfd/`
   (not the meshes or restart files), `runs/aero_fidelity/family/*/status.json` and
   `runs/aero_fidelity/family/matlab_*.log`, plus `nproc`, `lscpu`, `pip freeze`, the MATLAB and SU2 versions,
   and `git rev-parse HEAD`.

## What to expect

Generator calls take about 2 to 10 seconds each; 189 of them over four processes take roughly 5 to 10 minutes. The check audits
each valid candidate with the project's frozen native-mesh audit (about 12 s each) and takes some minutes.
SU2 on the medium mesh (about 330 thousand cells) takes about one to two hours per case with seven ranks; the
fine mesh (about 1.25 million cells) several times that. The full set is 27 medium cases plus the two
grid-check cases. Report timings.

Report the outcome of the declared criterion exactly as `decision.json` states it, together with the wedge
model's ranking and the grid-convergence numbers. Do not interpret beyond that; the analysis is done
afterwards.

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
   bash experiments/aero_fidelity/run_decision.sh family      # version 4: 29 generator calls, resumable
   bash experiments/aero_fidelity/run_decision.sh check       # selection rule; may legitimately end in NO_ADMISSIBLE_FAMILY
   bash experiments/aero_fidelity/run_decision.sh cfd-mesh
   nohup bash experiments/aero_fidelity/run_decision.sh cfd-run > runs/aero_fidelity/cfd_run.log 2>&1 &
   bash experiments/aero_fidelity/run_decision.sh collect
   ```
5. Versions 1 to 3 of the family already ran (`results_family_v1/` to `results_family_v3/`): no admissible member.
   Version 4 (protocol amendment 3, README §3d) is the one to run now. `NO_ADMISSIBLE_FAMILY` is a result, not a failure to work around: send `family_audit.json` back and stop.
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

Generator calls take about 2 to 10 seconds each; 29 of them over four processes take about one minute. The check audits
each valid candidate with the project's frozen native-mesh audit (about 12 s each) and takes some minutes.
SU2 on the medium mesh (about 330 thousand cells) takes about one to two hours per case with seven ranks; the
fine mesh (about 1.25 million cells) several times that. The full set is 27 medium cases plus the two
grid-check cases. Report timings.

Report the outcome of the declared criterion exactly as `decision.json` states it, together with the wedge
model's ranking and the grid-convergence numbers. Do not interpret beyond that; the analysis is done
afterwards.

## CFD stage version 5 (skin-only forces; README section 3f)

Run only after the family selection of version 4 exists (`results_family_v4/` documents it; copy its `family_selected/`
folder with the three `.mat` files and `select.json` from your version-4 run). No MATLAB is needed.

```bash
git fetch origin aoa-fix && git checkout aoa-fix && git pull          # report git rev-parse HEAD
export NP=7 JOBS=2 BASE_MARKER=1 OUT=runs/aero_fidelity_v5
mkdir -p $OUT && cp -r runs/aero_fidelity/family_selected $OUT/       # the version-4 selection, unchanged
bash experiments/aero_fidelity/run_decision.sh cfd-mesh
nohup bash experiments/aero_fidelity/run_decision.sh cfd-run > $OUT/cfd_run.log 2>&1 &
bash experiments/aero_fidelity/run_decision.sh collect
```

Check after cfd-mesh that each `mesh.su2` has three markers (body, base, farfield) and that `mesh_info.json` lists
`base_surfaces` >= 1 (gmsh may split the planar base cap into more than one patch; every patch counted there lies in the
plane x = x_max by construction, and all of them form the single marker `base`; FAM_MD4 medium has 2); if a mesh differs from the version-4 cell count (FAM_MD6 coarse: 93 085), report it. Do not change any
setting. Send back `case.json`, `case.cfg`, `history.csv`, `su2.log` of every case, `decision.json/txt`, timings, and the
convergence status, to a new branch `aero-family-v5-results-20261102`.

## Optimal-family test (README section 3h)

MATLAB runs on the user's computer as before (two short batches); everything else on the server. Do not change any value.

```bash
git fetch origin aoa-fix && git checkout aoa-fix && git pull                     # report git rev-parse HEAD
export NP=7 JOBS=2 SLICES=4
# stage A: map, 420 generator calls, then audit (mode map: status MAP_ONLY is the expected result)
OUT=runs/aero_opt_a FAMILY_VERSION=5 bash experiments/aero_fidelity/run_decision.sh family
OUT=runs/aero_opt_a bash experiments/aero_fidelity/run_decision.sh check
# stage B: interpolated a_x from the audited map, then audit
OUT=runs/aero_opt_b FAMILY_VERSION=6 FAMILY_ARGS="--map $PWD/runs/aero_opt_a/family_selected/family_audit.json" \
    bash experiments/aero_fidelity/run_decision.sh family
OUT=runs/aero_opt_b bash experiments/aero_fidelity/run_decision.sh check
# declared evaluation (tangent wedge + friction)
python experiments/aero_fidelity/optimum/evaluate_optimum.py runs/aero_opt_b/family \
    --audit runs/aero_opt_b/family_selected/family_audit.json --out runs/aero_opt/results
# Euler confirmation of the winners (only if optimum_decision.json has status EVALUATED)
mkdir -p runs/aero_opt/cfd_root && cp -r runs/aero_opt/results/confirm runs/aero_opt/cfd_root/family_selected
OUT=runs/aero_opt/cfd_root BASE_MARKER=1 ALPHAS="2 4 6 8" GRID_CHECK=0 bash experiments/aero_fidelity/run_decision.sh cfd-mesh
nohup env OUT=runs/aero_opt/cfd_root bash experiments/aero_fidelity/run_decision.sh cfd-run > runs/aero_opt/cfd_run.log 2>&1 &
# after cfd-run has finished (do NOT run the 'collect' stage here):
python experiments/aero_fidelity/optimum/evaluate_optimum.py runs/aero_opt_b/family \
    --audit runs/aero_opt_b/family_selected/family_audit.json --out runs/aero_opt/results --confirm-cfd runs/aero_opt/cfd_root/cfd
```

If the MATLAB batches run on another machine, run the `family` stage there exactly as for versions 1-4 (job lists from
`family_jobs.py --version 5`, then `--version 6 --map ...` with the stage-A audit copied over). Send back to a new branch
`aero-opt-results-20261103`: both `family_audit.json`, all `status.json` and `matlab_*.log`, the stage-B `.mat` files of VALID
members, `runs/aero_opt/results/` (optimum_decision.json, optimum_table.json, euler_confirmation.json, confirm/), every
case.json/case.cfg/history.csv/su2.log of the confirmation run, timings and convergence status.

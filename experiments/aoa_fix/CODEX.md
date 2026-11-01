# Instructions for the agent running the AoA experiments on the cloud server

Run exactly this revision and these commands. Do not write your own pass/fail or stage-permission logic:
`scripts/analysis/aoa_report.py` is the only decision rule, and `scripts/run_aoa_experiments.sh` already
enforces it.

1. Check out the head of `aoa-fix`, at or after the commit that added this file:
   `git fetch origin aoa-fix && git checkout aoa-fix && git pull`.
   If you have local commits (for example `8ae24f3`), push them to a separate branch such as
   `codex-local-<date>` first. Do not merge them into `aoa-fix`.
2. Keep the existing `runs/aoa/gate/` flights. They are skipped, not re-run, if the layout is
   `runs/aoa/gate/<policy>/dh0_dv0/<objective>/summary.json`. If your layout differs, re-run the gate.
3. The machine must be otherwise idle. Set `JOBS` to at most the number of physical cores minus 2: the
   post corrector (2 s) and look-ahead (30 s) have wall-clock budgets, so CPU contention changes results.
4. Run everything:

   ```bash
   export JOBS=$(( $(nproc --all) / 2 - 2 ))   # physical cores - 2 on a 2-thread-per-core machine; adjust
   nohup bash scripts/run_aoa_experiments.sh all > runs/aoa_all.log 2>&1 &
   ```

   `all` runs these stages: test → gate (resumes) → gate check → surrogate → heldout (656 flights) →
   fidelity (24 true-OCP flights) → report.
   - The gate check prints the declared two-sided G3 and the documented one-sided `G3_no_loss` reading
     (README sections 3 and 8). The held-out stage runs only if `PROCEED_TO_HELDOUT` is true for both
     gate policies.
   - The declared G3 failing because of a range gain is expected. It is not a reason to stop.
5. If a stage stops (test failure, gate check false, qualification failure), do not work around it.
   Send the logs back.
6. Send back the tarball of README section 6, plus `runs/aoa_all.log`, `nproc`, `lscpu`, `pip freeze`,
   `git rev-parse HEAD` and any extra figures or CSVs you produced. Mark extra analyses as additional;
   they do not replace the report.

## Improvement study (experiments/improve/README.md), after the aoa-fix `all` stage

Same rules: no own decision logic, an idle machine, `JOBS` ≤ physical cores − 2. `report-improve` is the
only summary; it reports the pre-declared comparisons in `experiments/improve/comparisons.json`.

```bash
export OBJECTIVE=aligned SAFETY_VARIANT=minimal_first
bash scripts/run_aoa_experiments.sh surrogate                 # aligned surrogate: labels, r0, DAgger, final
unset OBJECTIVE SAFETY_VARIANT
bash scripts/run_aoa_experiments.sh improve                   # 5 execution layers x 8 policies x 41 states
python scripts/upper_bound.py --out experiments/improve/bounds --states nominal heldout --jobs "$JOBS"
export OBJECTIVE=aligned SAFETY_VARIANT=minimal_first WORKERS=8
TRAIN_NAME=pi bash scripts/run_aoa_experiments.sh train       # baseline PI, 7 seeds (V16 settings)
TRAIN_NAME=overrides TRAIN_ARGS="--overrides" bash scripts/run_aoa_experiments.sh train
TRAIN_NAME=blocks TRAIN_ARGS="--overrides --blocks 2,4,8 --states 7 --stride 168 --patience 4" \
  bash scripts/run_aoa_experiments.sh train                   # same branch budget per iteration as PI
unset OBJECTIVE SAFETY_VARIANT
ARMS=aligned@minimal_first IMPROVE_POLICIES="$(for n in pi overrides blocks; do for s in 17 23 41 59 83 97 101; \
  do printf 'train_%s_seed%s ' $n $s; done; done)" bash scripts/run_aoa_experiments.sh improve
bash scripts/run_aoa_experiments.sh report-improve
```

Send back `runs/aoa/improve_report/`, `runs/aoa/train/*/seed*/state.json` and `seed*.log`,
`runs/aoa/surrogate_aligned/`, `data/surrogate_aligned*`, `data/lookahead_aligned*`, `experiments/improve/bounds/`,
and every `summary.json`, `setup.json` and `steps.jsonl` under `runs/aoa/improve/`.

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

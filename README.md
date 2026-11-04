# Miscellaneous files

Files that belong to none of the other branches (`project-latest`, `aero-database`, `evidence-archive`) and are kept only for the record.
Created on 2026-11-04.

| path | what it is | why it is not a result |
|---|---|---|
| `headroom_pilot/` | one-scenario pilot runs of the headroom study (density field sigma 0.06, correlation length 200 km, seed 0): clairvoyant plan, open-loop plan and receding-horizon runs at several back-offs | debugging runs: the first versions of the receding-horizon controller had failed re-plans and violated the q limit, so their ranges are not comparable; the study's preliminary result is in `experiments/headroom/README.md` of `project-latest` |

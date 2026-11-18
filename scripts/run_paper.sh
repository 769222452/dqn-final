#!/usr/bin/env bash
# Round 2 (EXPERIMENTS.md, amendment 5), in one command: the main library F8D2000B with the full plan, the
# sensitivity library F8D2000 with the reduced plan, then all tables and figures (runs/r2/paper/). Resumable.
#
#   WORKERS=$(nproc) nohup bash scripts/run_paper.sh > run_paper.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
export PY WORKERS=${WORKERS:-$(nproc)}
for lib in F8D2000B F8D2000; do
  LIBRARY=$lib bash scripts/run_all.sh
done
PYTHONPATH=src $PY -m target_rl.paper 2>&1 | tee runs/r2/paper.log
echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) paper outputs in runs/r2/paper/"

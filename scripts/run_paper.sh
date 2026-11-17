#!/usr/bin/env bash
# Everything the paper needs, in one command (EXPERIMENTS.md): the main library with the full plan, the
# sensitivity libraries with the reduced plan, then all tables and figures (runs/paper/). Resumable.
#
#   WORKERS=$(nproc) nohup bash scripts/run_paper.sh > run_paper.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
export PY WORKERS=${WORKERS:-$(nproc)}
for lib in F8D2000B F8D2000 L298; do
  LIBRARY=$lib bash scripts/run_all.sh
done
PYTHONPATH=src $PY -m target_rl.paper 2>&1 | tee runs/paper.log
echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) paper outputs in runs/paper/"

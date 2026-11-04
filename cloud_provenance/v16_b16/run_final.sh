#!/usr/bin/env bash
# Pre-registered run (see PREREGISTRATION.md): Post comparison, training of 7 seeds, evaluation on held-out-3.
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
DUAL=../repo   # this folder is /hy-tmp/v14_final/v16_b16; ../repo is the registered held-out-2 snapshot
SEEDS="17 23 41 59 83 97 101"
mkdir -p logs runs eval ablation

# H2: dual-routing post surrogate (held-out-2 snapshot) vs single post regressor (this repository)
python3 -u post_ablation.py repo ablation/single_reference reference 54 > logs/ablation_single_reference.log 2>&1
python3 -u post_ablation.py $DUAL ablation/dual_reference reference 54 > logs/ablation_dual_reference.log 2>&1
python3 -u post_ablation.py repo ablation/single_bc17 policies/bc_seed17.json 54 > logs/ablation_single_bc17.log 2>&1
python3 -u post_ablation.py $DUAL ablation/dual_bc17 policies/bc_seed17.json 54 > logs/ablation_dual_bc17.log 2>&1
echo "post comparison finished $(date -Is)" | tee logs/stage.log

for s in $SEEDS; do
  python3 -u repo/scripts/train.py --out runs/seed$s --seed $s --workers 8 --iterations 80 > logs/train_seed$s.log 2>&1 &
done
wait
echo "training finished $(date -Is)" | tee -a logs/stage.log
sha256sum runs/seed*/state.json | tee -a logs/stage.log

evaluate() {  # name state
  python3 -u repo/scripts/evaluate.py "$2" --out eval/$1 --test-set held-out-3 --reference-out eval/reference \
    --workers 54 > logs/eval_$1.log 2>&1
}
for s in $SEEDS; do evaluate final_seed$s runs/seed$s/state.json; done
for s in $SEEDS; do evaluate b16_seed$s policies/bc_seed$s.json; done
echo "evaluation finished $(date -Is)" | tee -a logs/stage.log

args=()
for s in $SEEDS; do args+=(final_seed$s=eval/final_seed$s); done
for s in $SEEDS; do args+=(b16_seed$s=eval/b16_seed$s); done
python3 repo/scripts/compare.py --out comparison.json "${args[@]}" > logs/compare.log 2>&1
python3 analyze_heldout3.py > logs/analyze.log 2>&1
echo "done $(date -Is)" | tee -a logs/stage.log

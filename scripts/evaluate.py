"""Evaluate a trained policy. Example: python scripts/evaluate.py runs/seed17/state.json --out runs/seed17/evaluation"""
import os

for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'   # single-threaded BLAS: results are reproducible bit for bit

import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

from rollout_improvement.evaluation import main  # noqa: E402

if __name__ == '__main__':
    mp.freeze_support()
    sys.exit(main())

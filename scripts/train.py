"""Train a policy. Example: python scripts/train.py --out runs/seed17 --seed 17"""
import os

for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'   # single-threaded BLAS: results are reproducible bit for bit

import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

from rollout_improvement.training import main  # noqa: E402

if __name__ == '__main__':
    mp.freeze_support()
    sys.exit(main())

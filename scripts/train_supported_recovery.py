"""Run one explicitly budgeted, isolated action-support experiment."""
import os

for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import multiprocessing as mp
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from supported_recovery.training import main

if __name__ == '__main__':
    mp.freeze_support()
    sys.exit(main())

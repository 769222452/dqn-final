"""Rollout-based policy iteration for the certified lift-coefficient schedule of a morphing waverider."""
import os

# Single-threaded BLAS keeps flights and network fits reproducible bit for bit. This only takes effect if
# numpy has not been imported before this package.
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

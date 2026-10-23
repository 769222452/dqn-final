"""Certified flight of the morphing waverider.

Public API::

    env, certifier = build_flight(folder, label, initial_offset=None)
    pool = certifier.pool(preference)            # preference: array of 19 scores
    reward, done, info = certifier.commit(pool, action)
    branch = certifier.fork(folder)

CL_ACTIONS lists the 19 absolute CL targets in action order. OracleRequired is
raised when an action cannot be executed safely.
"""
import os

# Single-threaded BLAS keeps the floating-point results reproducible.
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

from .build import build_flight  # noqa: E402
from .certification import CL_ACTIONS  # noqa: E402
from .surrogate import OracleRequired  # noqa: E402

__all__ = ['build_flight', 'CL_ACTIONS', 'OracleRequired']

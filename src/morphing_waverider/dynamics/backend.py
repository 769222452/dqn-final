"""Numeric backend shim so the same physics code runs on NumPy floats/arrays
(simulation) and on CasADi symbols (trajectory optimization).

Every physics function in dynamics/ and aero/ is written against this module
instead of importing numpy or casadi directly."""
from __future__ import annotations

from typing import Any

import numpy as np

try:  # CasADi is a core dependency but keep the import guarded for docs/tools.
    import casadi as ca

    _CA_TYPES: tuple[type, ...] = (ca.SX, ca.MX, ca.DM)
except ImportError:  # pragma: no cover
    ca = None
    _CA_TYPES = tuple()


def is_symbolic(*args: Any) -> bool:
    return any(isinstance(a, _CA_TYPES) for a in args)


def _mod(*args: Any):
    return ca if is_symbolic(*args) else np


def exp(x):
    return _mod(x).exp(x)


def log(x):
    return _mod(x).log(x)


def sqrt(x):
    return _mod(x).sqrt(x)


def sin(x):
    return _mod(x).sin(x)


def cos(x):
    return _mod(x).cos(x)


def tanh(x):
    return _mod(x).tanh(x)


def power(x, p):
    if is_symbolic(x, p):
        return ca.power(x, p)
    return np.power(x, p)


def fmax(a, b):
    if is_symbolic(a, b):
        return ca.fmax(a, b)
    return np.maximum(a, b)


def fmin(a, b):
    if is_symbolic(a, b):
        return ca.fmin(a, b)
    return np.minimum(a, b)


def clip(x, lo, hi):
    return fmin(fmax(x, lo), hi)


def if_else(cond, a, b):
    if is_symbolic(cond, a, b):
        return ca.if_else(cond, a, b)
    return np.where(cond, a, b)


def softplus(x, beta: float = 1.0):
    """Smooth max(x, 0), evaluated as max(y, 0) + log1p(exp(-|y|)) with y = beta*x so that
    neither branch overflows for large |y| (plain log1p(exp(y)) is inf for y > ~709)."""
    if is_symbolic(x):
        y = beta * x
        return (ca.fmax(y, 0.0) + ca.log1p(ca.exp(-ca.fabs(y)))) / beta
    x = np.asarray(x, dtype=float)
    return np.logaddexp(0.0, beta * x) / beta


def smooth_floor(x, floor: float, sharpness: float = 20.0):
    """Smooth approximation of max(x, floor) with C^inf continuity."""
    return floor + softplus(x - floor, beta=sharpness / max(abs(floor), 1e-9))


def stack(items):
    """Column-stack scalars into a vector: casadi.vertcat or np.array."""
    if is_symbolic(*items):
        return ca.vertcat(*items)
    return np.array([float(i) if np.ndim(i) == 0 else i for i in items], dtype=float)


def to_float(x) -> float:
    if is_symbolic(x):
        return float(ca.DM(x))
    return float(x)

"""Global seeding for Python / NumPy / PyTorch (if present). Gymnasium envs and the
DummyGenerator take an explicit seed via their own APIs and should be seeded from the
same master seed by the caller (see scripts/)."""
from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int, deterministic_torch: bool = True) -> np.random.Generator:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # torch is optional
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
    return np.random.default_rng(seed)

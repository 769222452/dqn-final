"""Disturbance scenarios (same generator as the original fast model, certified_flight.fastflight/headroom).

A scenario fixes: the initial offset (|dh| <= 2000 m, |dV| <= 100 m/s, uniform), the lift and drag biases
(1 + sigma clip(z, +-2)), and a mean-one multiplicative density field along the track (AR(1) in downrange,
step 5 km). The same seed gives the same draws at every level. A case is a scenario plus a target.
"""
from dataclasses import dataclass
import math

import numpy as np

from . import config

FIELD_STEP_M = 5e3
FIELD_LENGTH_M = 3.2e6


@dataclass(frozen=True)
class Scenario:
    level: str
    seed: int
    dh: float
    dv: float
    k_lift: float
    k_drag: float


@dataclass(frozen=True)
class Target:
    name: str
    range_km: float
    speed_mps: float


def scenario(level, seed, cfg=None):
    cfg = cfg or config.load()
    d = cfg['disturbance']
    sigma, corr_km, aero = d['levels'][level]
    rng = np.random.default_rng([int(seed), 7])
    dh = float(np.round(rng.uniform(-d['initial_dh_m'], d['initial_dh_m']), 0))
    dv = float(np.round(rng.uniform(-d['initial_dv_mps'], d['initial_dv_mps']), 1))
    zl, zd = rng.standard_normal(2)
    k = lambda z: float(1+aero*np.clip(z, -2., 2.))
    return Scenario(level, int(seed), dh, dv, k(zl), k(zd))


def density_field(sc, cfg=None):
    """(s grid [m], density ratio) along the track."""
    cfg = cfg or config.load()
    sigma, corr_km, _ = cfg['disturbance']['levels'][sc.level]
    s = np.arange(0, FIELD_LENGTH_M+FIELD_STEP_M, FIELD_STEP_M)
    if sigma <= 0:
        return s, np.ones_like(s)
    rng = np.random.default_rng(sc.seed)
    a = np.exp(-FIELD_STEP_M/(corr_km*1e3))
    z = np.empty(len(s))
    z[0] = rng.standard_normal()
    for i in range(1, len(s)):
        z[i] = a*z[i-1]+np.sqrt(1-a*a)*rng.standard_normal()
    return s, np.exp(sigma*z-sigma**2/2)


class Field:
    """Fast linear interpolation of the density ratio on the uniform 5 km grid."""

    def __init__(self, sc, cfg=None):
        self.s, k = density_field(sc, cfg)
        self.k = k.tolist()
        self.n = len(self.k)

    def __call__(self, s):
        f = s/FIELD_STEP_M
        if f <= 0:
            return self.k[0]
        i = int(math.floor(f))
        if i >= self.n-1:
            return self.k[-1]
        u = f-i
        return (1-u)*self.k[i]+u*self.k[i+1]

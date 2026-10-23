"""Scalar fast path for the speed of sound of the layered atmosphere of morphing_waverider.

``Atmosphere.speed_of_sound`` evaluates all seven layers through the generic
CasADi/NumPy backend (nested ``if_else``). For finite real scalars this wrapper
performs the same IEEE operations in the same order (clamp, one affine layer
formula, product, sqrt), so results are bit-identical; arrays, symbols,
non-finite values and every other method are delegated to the wrapped object.
"""
import numpy as np
from morphing_waverider.dynamics.atmosphere import _LAYERS, _H_TOP

_SCALARS = (float, int, np.float64)


class FastAtmosphere:
    def __init__(self, atmosphere):
        if isinstance(atmosphere, FastAtmosphere):
            atmosphere = atmosphere.base
        self.base = atmosphere
        c = atmosphere.cfg
        self._lo, self._hi = c.h_min_valid, c.h_max_valid
        self._gR = c.gamma_gas * c.R_gas      # same left-to-right product as Atmosphere.speed_of_sound
        self._T = list(atmosphere._layer_T)

    def __getattr__(self, name):
        # Only called for missing attributes; guard so copy/pickle (which build
        # the object without __init__) cannot recurse through self.base.
        if name == 'base' or name.startswith('__'):
            raise AttributeError(name)
        return getattr(self.base, name)

    def temperature_scalar(self, h):
        h = min(max(h, self._lo), self._hi)
        n = len(_LAYERS)
        for i in range(n - 1):
            if h < _LAYERS[i + 1][0]:
                h_b, L = _LAYERS[i]
                dh = h - h_b
                return self._T[i] + 0.0 * dh if abs(L) < 1e-12 else self._T[i] + L * dh
        h_b, L = _LAYERS[n - 1]
        return self._T[n - 1] + L * (min(h, _H_TOP) - h_b)

    def speed_of_sound(self, h):
        if type(h) in _SCALARS and np.isfinite(h):
            return np.float64(np.sqrt(self._gR * self.temperature_scalar(float(h))))
        return self.base.speed_of_sound(h)

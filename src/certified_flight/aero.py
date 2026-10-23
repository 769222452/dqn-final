"""Aerodynamic database of the candidate geometries: separable PCHIP, Mach first, alpha second.

There is no interpolation across geometry indices. Transition coefficients are the
convex combination of the two endpoint queries at the SAME flight Mach and alpha.
Numeric and CasADi forms share this definition.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import casadi as ca
import numpy as np
from scipy.interpolate import PchipInterpolator


def _pchip_edge(h0, h1, m0, m1):
    d = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    mask = np.sign(d) != np.sign(m0)
    mask2 = (np.sign(m0) != np.sign(m1)) & (np.abs(d) > 3.0 * np.abs(m0))
    return np.where(mask, 0.0, np.where(~mask & mask2, 3.0 * m0, d))


def pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Node derivatives identical to scipy PchipInterpolator, without object construction."""
    y = np.asarray(y, dtype=float)
    hk = np.diff(x).reshape((-1,) + (1,) * (y.ndim - 1))
    mk = np.diff(y, axis=0) / hk
    if y.shape[0] == 2:
        return np.stack([mk[0], mk[0]])
    smk = np.sign(mk)
    cond = (smk[1:] != smk[:-1]) | (mk[1:] == 0) | (mk[:-1] == 0)
    w1 = 2.0 * hk[1:] + hk[:-1]
    w2 = hk[1:] + 2.0 * hk[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        whmean = (w1 / mk[:-1] + w2 / mk[1:]) / (w1 + w2)
        inner = np.where(cond, 0.0, 1.0 / whmean)
    d0 = _pchip_edge(hk[0], hk[1], mk[0], mk[1])
    dn = _pchip_edge(hk[-1], hk[-2], mk[-1], mk[-2])
    return np.concatenate([d0[None], inner, dn[None]], axis=0)


def _safe(m):
    return ca.if_else(m == 0, 1.0, m)


def _edge(h0, h1, m0, m1):
    d = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    return ca.if_else(d * m0 <= 0, 0.0,
                      ca.if_else((m0 * m1 <= 0) * (ca.fabs(d) > 3.0 * ca.fabs(m0)), 3.0 * m0, d))


class DomainExit(ValueError):
    """Mach or alpha outside the tabulated support."""


class AeroDatabase:
    """CL and pressure CD of every candidate geometry on a common Mach/alpha grid."""

    def __init__(self, directory: Path, verify=True):
        self.directory = Path(directory).resolve()
        manifest = json.loads((self.directory / 'manifest.json').read_text(encoding='utf-8'))
        if manifest['status'] != 'VERIFIED':
            raise ValueError('Unverified candidate snapshot')
        if verify:
            for item in manifest['files']:
                p = (self.directory / item['path']).resolve()
                if not p.is_relative_to(self.directory):
                    raise ValueError('Manifest path escapes snapshot')
                if hashlib.sha256(p.read_bytes()).hexdigest() != item['sha256']:
                    raise ValueError(f'Snapshot hash mismatch: {item["path"]}')
        with np.load(self.directory / 'aerodynamics/p2_grid.npz', allow_pickle=False) as z:
            self.ids = z['geometry_ids'].astype(str).tolist()
            self.M = z['mach_grid'].astype(float)
            self.A = np.deg2rad(z['alpha_deg_grid'].astype(float))
            self.values = np.stack([z['CL'], z['CD_pressure']], axis=-1).astype(float)
            self.valid = z['usable_p2'].astype(bool)
        if self.ids != manifest['geometry_ids'] or list(self.values.shape[:3]) != manifest['tensor_shape']:
            raise ValueError('Snapshot axes or identities changed')
        # The snapshot is complete. Refuse holes rather than interpolate through them.
        if not self.valid.all() or not np.isfinite(self.values).all() or np.any(self.values[..., 1] <= 0):
            raise ValueError('Incomplete/invalid table: segmented interpolation is required')
        self.index = {name: i for i, name in enumerate(self.ids)}
        # [geometry, polynomial_power, Mach_interval, alpha_node, coefficient]
        self.c = np.moveaxis(PchipInterpolator(self.M, self.values, axis=1, extrapolate=False).c, 2, 0)
        self.ncoef = int(np.prod(self.c.shape[1:]))
        self.parameters = np.stack([c.transpose(0, 2, 3, 1).ravel(order='F') for c in self.c])
        self.symbolic = self._symbolic()

    def check(self, mach, alpha=None):
        if not np.isfinite(mach) or not self.M[0] <= mach <= self.M[-1]:
            raise DomainExit(f'Mach={mach}, support={self.M[[0, -1]].tolist()}')
        if alpha is not None and (not np.isfinite(alpha) or not self.A[0]-1e-12 <= alpha <= self.A[-1]+1e-12):
            raise DomainExit(f'alpha_deg={np.rad2deg(alpha)}')

    def curves(self, mach, indices=None):
        """Coefficient curves over the alpha nodes at one Mach: [geometry, alpha_node, (CL, CD)]."""
        self.check(mach)
        i = int(np.clip(np.searchsorted(self.M, mach, side='right')-1, 0, len(self.M)-2))
        c = self.c if indices is None else self.c[np.atleast_1d(indices)]
        d = mach-self.M[i]
        return ((c[:, 0, i]*d+c[:, 1, i])*d+c[:, 2, i])*d+c[:, 3, i]

    def curves_slice(self, mach, indices, start, stop, channel):
        """Bit-identical to ``curves(mach, indices)[:, start:stop, channel]``.

        Slices the alpha nodes/channel before the same elementwise Horner steps,
        instead of evaluating every node and coefficient for every geometry.
        """
        self.check(mach)
        i = int(np.clip(np.searchsorted(self.M, mach, side='right')-1, 0, len(self.M)-2))
        c = self.c[:, :, i, start:stop, channel]
        if indices is not None:
            c = c[np.atleast_1d(indices)]
        d = mach-self.M[i]
        return ((c[:, 0]*d+c[:, 1])*d+c[:, 2])*d+c[:, 3]

    def coefficients_k(self, k, mach, alpha):
        """(CL, CD) of geometry k at (Mach, alpha)."""
        self.check(mach, alpha)
        y = self.curves(mach, [k])[0]
        d = pchip_slopes(self.A, y)
        i = int(np.clip(np.searchsorted(self.A, alpha, side='right')-1, 0, len(self.A)-2))
        h = self.A[i+1]-self.A[i]
        z = alpha-self.A[i]
        slope = (y[i+1]-y[i])/h
        out = y[i]+d[i]*z+(3*slope-2*d[i]-d[i+1])*z*z/h+(d[i]+d[i+1]-2*slope)*z**3/h**2
        return float(out[0]), float(out[1])

    def equal_lift(self, mach, cl_des):
        """Memoised on the exact (Mach, CL) doubles; returns private copies."""
        key = (float(mach), float(cl_des))
        cache = self.__dict__.setdefault('_equal_lift_cache', {})
        hit = cache.get(key)
        if hit is None:
            hit = self._equal_lift(mach, cl_des)
            if len(cache) >= 4096:
                cache.clear()
            cache[key] = hit
        return tuple(v.copy() for v in hit)

    def _equal_lift(self, mach, cl_des):
        """Alpha giving cl_des for every geometry, the CD there (inf if unreachable) and feasibility."""
        if not np.isfinite(cl_des):
            raise ValueError('Nonfinite lift target')
        y = self.curves(mach)
        if not np.all(np.diff(y[..., 0], axis=1) > 0):
            raise ValueError('Nonmonotone interpolated lift curve: branch-wise inversion required')
        feasible = (y[:, 0, 0] <= cl_des) & (cl_des <= y[:, -1, 0])
        # Vectorised PCHIP bisection for all candidates; no extrapolation.
        pc = PchipInterpolator(self.A, y, axis=1, extrapolate=False)
        lo = np.full(len(self.ids), self.A[0])
        hi = np.full(len(self.ids), self.A[-1])
        rows = np.arange(len(self.ids))
        for _ in range(38):
            mid = (lo+hi)/2
            j = np.clip(np.searchsorted(self.A, mid, side='right')-1, 0, len(self.A)-2)
            c = pc.c[:, j, rows, 0]
            d = mid-self.A[j]
            lift = ((c[0]*d+c[1])*d+c[2])*d+c[3]
            lo = np.where(lift < cl_des, mid, lo)
            hi = np.where(lift >= cl_des, mid, hi)
        alpha = (lo+hi)/2
        j = np.clip(np.searchsorted(self.A, alpha, side='right')-1, 0, len(self.A)-2)
        c = pc.c[:, j, rows, 1]
        d = alpha-self.A[j]
        cd = ((c[0]*d+c[1])*d+c[2])*d+c[3]
        return alpha, np.where(feasible, cd, np.inf), feasible

    def generate(self, mach, CL_des, current=None):
        """Geometry generator: the minimum-CD candidate that reaches CL_des at this Mach.

        Ties within 1e-12 keep the current target geometry.
        """
        alpha, cd, ok = self.equal_lift(mach, CL_des)
        if not ok.any():
            return {'status': 'CL_UNREACHABLE', 'geometry_index': None, 'feasible_count': 0}
        score = cd
        k = int(np.argmin(score))
        if current is not None and ok[current] and abs(score[current]-score[k]) <= 1e-12:
            k = int(current)
        cl_check, cd_check = self.coefficients_k(k, mach, alpha[k])
        return {'status': 'MATCHED', 'geometry_index': k, 'geometry_id': self.ids[k],
                'CL_des': float(CL_des), 'alpha_rad': float(alpha[k]),
                'CL': cl_check, 'CD': cd_check, 'CL_abs_error': abs(cl_check-CL_des),
                'feasible_count': int(ok.sum()), 'selection': 'MIN_CD_AT_CL'}

    def _symbolic(self):
        """CasADi PCHIP evaluation with the coefficients of one geometry as a parameter vector."""
        M = ca.MX.sym('M')
        a = ca.MX.sym('alpha')
        c = ca.MX.sym('coefficients', self.ncoef)
        im = ca.fmin(len(self.M)-2, ca.fmax(0, ca.low(ca.DM(self.M), M)))
        dx = M-ca.MX(self.M)[im]
        ia = ca.fmin(len(self.A)-2, ca.fmax(0, ca.low(ca.DM(self.A), a)))
        h = ca.MX(np.diff(self.A))[ia]
        z = a-ca.MX(self.A)[ia]
        hm = ca.MX(np.diff(self.A))[ca.fmax(0, ia-1)]
        hp = ca.MX(np.diff(self.A))[ca.fmin(len(self.A)-2, ia+1)]

        def slope(hl, hr, ml, mr):
            w1, w2 = 2*hr+hl, hr+2*hl
            wh = (w1/_safe(ml)+w2/_safe(mr))/(w1+w2)
            return ca.if_else(ml*mr <= 0, 0, 1/_safe(wh))

        def node(j, off):
            # Only four neighbouring alpha nodes are needed for the two PCHIP
            # slopes. Gathering before evaluation avoids computing all slopes.
            base = im*(4*len(self.A)*2)+4*(off+j)
            return ((c[base]*dx+c[base+1])*dx+c[base+2])*dx+c[base+3]

        out = []
        for off in (0, len(self.A)):
            ym, y0, y1, yp = [node(j, off) for j in (ca.fmax(0, ia-1), ia, ia+1, ca.fmin(len(self.A)-1, ia+2))]
            sm, s, sp = (y0-ym)/hm, (y1-y0)/h, (yp-y1)/hp
            d0 = ca.if_else(ia == 0, _edge(h, hp, s, sp), slope(hm, h, sm, s))
            d1 = ca.if_else(ia == len(self.A)-2, _edge(h, hm, s, sm), slope(h, hp, s, sp))
            out.append(y0+d0*z+(3*s-2*d0-d1)*z*z/h+(d0+d1-2*s)*z**3/h**2)
        # During NLP iterations expressions extend their edge polynomial; hard RK-stage
        # domain constraints and independent numeric audits reject such candidate plans.
        return ca.Function('pchip', [M, a, c], [ca.vertcat(*out)],
                           {'is_diff_in': [True, True, False]})

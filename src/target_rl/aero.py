"""Equal-lift drag tables of the geometry library (F8D2000, original P2 surface-pressure model).

For every Mach node of the library and every CL of the table grid:

  CDmin, CDmax   lowest / highest pressure drag over the geometries that reach that CL inside the alpha support
  CDfix          drag of the fixed geometry (configs/task.yaml aero.fixed_geometry)

The CL of a geometry is followed on its monotone branch from the lowest alpha (PCHIP in alpha, step
aero.alpha_step_deg). Between nodes the tables are interpolated linearly in Mach and CL. The morphing
action w in [0, 1] gives CD = CDmin + w (CDmax - CDmin): a proxy for "choose a geometry with this much drag at
this lift" (the 8 s morph transition is not modelled). Tables are cached in runs/cache/.
"""
import hashlib
import math
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from . import config


def _cd_at_cl(alpha, cl_curve, cd_curve, cls):
    out = np.full(len(cls), np.nan)
    k = int(np.argmax(cl_curve))
    c, d = cl_curve[:k+1], cd_curve[:k+1]
    keep = np.r_[True, np.diff(np.maximum.accumulate(c)) > 0]
    c, d = c[keep], d[keep]
    inside = (cls >= c[0]) & (cls <= c[-1])
    out[inside] = np.interp(cls[inside], c, d)
    return out


def build_tables(cfg=None):
    cfg = cfg or config.load()
    a = cfg['aero']
    path = config.ROOT/a['library']
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != a['sha256']:
        raise ValueError(f'aerodynamic library checksum mismatch: {path}')
    z = np.load(path, allow_pickle=False)
    ids = [str(x) for x in z['geometry_ids']]
    if a['fixed_geometry'] not in ids:
        raise ValueError(f"fixed geometry {a['fixed_geometry']} not in the library")
    if not z['usable_p2'].all():
        raise ValueError('library contains unusable grid points')
    fix = ids.index(a['fixed_geometry'])
    M = z['mach_grid'].astype(float)
    A = np.deg2rad(z['alpha_deg_grid'].astype(float))
    lo, hi, step = a['cl_table']
    cls = np.round(np.arange(lo, hi+step/2, step), 6)
    alpha = np.deg2rad(np.arange(np.rad2deg(A[0]), np.rad2deg(A[-1])+1e-9, a['alpha_step_deg']))
    alpha = np.clip(alpha, A[0], A[-1])
    tmin = np.full((len(M), len(cls)), np.nan)
    tmax = np.full_like(tmin, np.nan)
    tfix = np.full_like(tmin, np.nan)
    count = np.zeros_like(tmin, dtype=int)
    for i in range(len(M)):
        y = PchipInterpolator(A, np.stack([z['CL'][:, i], z['CD_pressure'][:, i]], -1), axis=1)(alpha)
        cd = np.array([_cd_at_cl(alpha, y[g, :, 0], y[g, :, 1], cls) for g in range(len(ids))])
        ok = np.isfinite(cd)
        count[i] = ok.sum(0)
        with np.errstate(all='ignore'):
            tmin[i] = np.where(count[i] > 0, np.nanmin(np.where(ok, cd, np.inf), 0), np.nan)
            tmax[i] = np.where(count[i] > 0, np.nanmax(np.where(ok, cd, -np.inf), 0), np.nan)
        tfix[i] = cd[fix]
    return dict(mach=M, cl=cls, cd_min=tmin, cd_max=tmax, cd_fix=tfix, count=count, sha256=digest,
                fixed=a['fixed_geometry'], n_geometries=len(ids))


def tables(cfg=None):
    cfg = cfg or config.load()
    cache = config.ROOT/'runs'/'cache'/f"aero_{cfg['aero']['sha256'][:12]}_{cfg['aero']['fixed_geometry']}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return {k: (z[k] if z[k].ndim else z[k].item()) for k in z.files}
    t = build_tables(cfg)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(cache.name+'.tmp.npz')
    np.savez(tmp, **t)
    tmp.replace(cache)
    return t


def _edge_fill(t, ok):
    out = np.array(t, float)
    for i in range(out.shape[0]):
        idx = np.flatnonzero(ok[i])
        out[i, :idx[0]] = out[i, idx[0]]
        out[i, idx[-1]+1:] = out[i, idx[-1]]
        out[i, idx[0]:idx[-1]+1] = np.interp(np.arange(idx[0], idx[-1]+1), idx, out[i, idx])
    return out


class Drag:
    """CD(M, CL, w) for morphing, or the fixed geometry's CD(M, CL); CL range available at each Mach."""

    def __init__(self, morph=True, cfg=None):
        t = tables(cfg)
        self.morph = morph
        self.M = np.asarray(t['mach'], float)
        self.cl = np.asarray(t['cl'], float)
        self.cl0, self.dcl = float(self.cl[0]), float(self.cl[1]-self.cl[0])
        if morph:
            self.lo, self.hi = np.asarray(t['cd_min']), np.asarray(t['cd_max'])
        else:
            self.lo = self.hi = np.asarray(t['cd_fix'])
        ok = np.isfinite(self.lo) & np.isfinite(self.hi)
        self.cl_range = np.array([[self.cl[r].min(), self.cl[r].max()] for r in ok])   # per Mach node
        # outside the available CL interval hold the edge value (CL is clipped to cl_limits before use; this only
        # covers RK sub-stages whose Mach differs slightly from the clipping Mach)
        self.lo, self.hi = _edge_fill(self.lo, ok), _edge_fill(self.hi, ok)
        self.Ml, self.Mh = list(self.M), len(self.M)

    def _node(self, mach):
        M = self.M
        mach = min(max(mach, M[0]), M[-1])
        i = int(np.searchsorted(M, mach, side='right'))-1
        i = min(max(i, 0), len(M)-2)
        return i, (mach-M[i])/(M[i+1]-M[i])

    def cl_limits(self, mach):
        """CL interval available at this Mach (inner of the two neighbouring nodes)."""
        i, _ = self._node(mach)
        return max(self.cl_range[i, 0], self.cl_range[i+1, 0]), min(self.cl_range[i, 1], self.cl_range[i+1, 1])

    def cd(self, mach, cl, w=0.):
        i, u = self._node(mach)
        f = (cl-self.cl0)/self.dcl
        j = int(math.floor(f))
        j = min(max(j, 0), len(self.cl)-2)
        v = f-j
        lo = (1-u)*((1-v)*self.lo[i, j]+v*self.lo[i, j+1])+u*((1-v)*self.lo[i+1, j]+v*self.lo[i+1, j+1])
        if not self.morph or w == 0.:
            return float(lo)
        hi = (1-u)*((1-v)*self.hi[i, j]+v*self.hi[i, j+1])+u*((1-v)*self.hi[i+1, j]+v*self.hi[i+1, j+1])
        return float(lo+w*(hi-lo))

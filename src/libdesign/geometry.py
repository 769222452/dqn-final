"""Fixed-leading-edge geometry generator (the F8D2000 method, datasets/F8D2000/config.json) and validity checks.

A shape = parent mesh + z displacements of the upper and lower surfaces. With t = streamwise column index/(n-1),
c = max(0, 5 - x_LE)/5 per spanwise row and e = y_LE/max(y_LE), the F8D2000 basis is

    B = c [t^2 (1-t)^3/0.03456, t^3 (1-t)^2/0.03456, t^2 (3-2t), 16 t^2 (1-t)^2 (2 e^2 - 1)]

(front, rear, tail, span). Every basis function vanishes on the leading-edge column (t = 0) and on the collapsed
tip row (c = 0), so the 3D leading edge, the x/y grid, the planform and the length stay those of the parent.
Native z points down: a positive coefficient moves the surface down.

Extra bases studied here (same boundary behaviour, PREREGISTRATION-independent design study):
    flap   c t^4               trailing-edge ramp (a deflected flap)
    twist  c t^2 (2 e - 1)     spanwise linear incidence change
"""
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[2]
PARENTS = ('G42', 'F2K_01233', 'F2K_00021')
F8D_NAMES = ('front', 'rear', 'tail', 'span')
F8D_LIMITS = dict(upper=(0.03, 0.03, 0.04, 0.025), lower=(0.025, 0.025, 0.035, 0.02))


def load_mat(path):
    m = sio.loadmat(path)
    lo = np.stack([m['wav_x'], m['wav_y'], m['wav_z']], -1)
    up = np.stack([m['wav_up_x'], m['wav_up_y'], m['wav_up_z']], -1)
    return lo, up


def parent(name):
    return load_mat(ROOT/'data'/'F8D2000_source'/'references'/name/'native_geometry.mat')


def basis(surface, names=F8D_NAMES):
    """[n_basis, nrow, ncol] for one surface mesh."""
    n = surface.shape[1]
    t = np.arange(n)/(n-1)
    xle, yle = surface[:, 0, 0], surface[:, 0, 1]
    c = np.maximum(0., 5.-xle)/5.
    e = yle/np.max(yle)
    T, C, E = t[None, :], c[:, None], e[:, None]
    lib = dict(front=C*T**2*(1-T)**3/0.03456, rear=C*T**3*(1-T)**2/0.03456, tail=C*T**2*(3-2*T),
               span=C*16*T**2*(1-T)**2*(2*E**2-1), flap=C*T**4, twist=C*T**2*(2*E-1))
    B = np.array([lib[k] for k in names])
    B[:, :, 0] = 0.
    B[:, 0, :] = 0.
    return B


def deform(par, upper_coef, lower_coef, names=F8D_NAMES):
    lo, up = par
    lo2, up2 = lo.copy(), up.copy()
    up2[..., 2] += np.tensordot(np.asarray(upper_coef, float), basis(up, names), 1)
    lo2[..., 2] += np.tensordot(np.asarray(lower_coef, float), basis(lo, names), 1)
    return lo2, up2


def thickness(lo, up):
    """Lower minus upper native z at the lower-surface (x, y) points (upper z interpolated linearly on its own
    Delaunay triangulation in x, y; points outside the upper planform are NaN)."""
    from scipy.interpolate import LinearNDInterpolator
    pts = up[..., :2].reshape(-1, 2)
    keep = np.unique(np.round(pts, 12), axis=0, return_index=True)[1]
    f = LinearNDInterpolator(pts[keep], up[..., 2].reshape(-1)[keep])
    return lo[..., 2]-f(lo[..., 0], lo[..., 1])


def volume(lo, up):
    """Vehicle volume (both halves): thickness integrated over the lower-surface triangles in (x, y)."""
    from scipy.spatial import Delaunay
    th = np.nan_to_num(np.clip(thickness(lo, up), 0, None)).reshape(-1)
    pts = lo[..., :2].reshape(-1, 2)
    keep = np.unique(np.round(pts, 12), axis=0, return_index=True)[1]
    tri = Delaunay(pts[keep])
    P, T = pts[keep][tri.simplices], th[keep][tri.simplices]
    a = .5*np.abs((P[:, 1, 0]-P[:, 0, 0])*(P[:, 2, 1]-P[:, 0, 1])-(P[:, 2, 0]-P[:, 0, 0])*(P[:, 1, 1]-P[:, 0, 1]))
    return 2*float((a*T.mean(1)).sum())


class Fixed:
    """Precomputed (x, y) structure of a parent: deformations change z only, so the interpolation weights of the
    upper surface at the lower-surface points and the planform triangles are computed once."""

    def __init__(self, par):
        from scipy.spatial import Delaunay
        lo, up = par
        pts = up[..., :2].reshape(-1, 2)
        self.keep_up = np.unique(np.round(pts, 12), axis=0, return_index=True)[1]
        tri = Delaunay(pts[self.keep_up])
        q = lo[..., :2].reshape(-1, 2)
        s = tri.find_simplex(q)
        self.inside = s >= 0
        T = tri.transform[s[self.inside]]
        b = np.einsum('nij,nj->ni', T[:, :2], q[self.inside]-T[:, 2])
        self.bary = np.c_[b, 1-b.sum(1)]
        self.vert = self.keep_up[tri.simplices[s[self.inside]]]
        ptl = lo[..., :2].reshape(-1, 2)
        self.keep_lo = np.unique(np.round(ptl, 12), axis=0, return_index=True)[1]
        tl = Delaunay(ptl[self.keep_lo])
        P = ptl[self.keep_lo][tl.simplices]
        self.tri_area = .5*np.abs((P[:, 1, 0]-P[:, 0, 0])*(P[:, 2, 1]-P[:, 0, 1])-(P[:, 2, 0]-P[:, 0, 0])*(P[:, 1, 1]-P[:, 0, 1]))
        self.tri_idx = self.keep_lo[tl.simplices]
        self.shape = lo.shape[:2]
        mask = np.zeros(self.shape, bool)
        mask[1:-1, 1:] = True
        self.inner = mask.reshape(-1)[self.inside]

    def thickness(self, lo, up):
        zu = up[..., 2].reshape(-1)
        th = np.full(self.shape[0]*self.shape[1], np.nan)
        th[self.inside] = lo[..., 2].reshape(-1)[self.inside]-(self.bary*zu[self.vert]).sum(1)
        return th

    def check(self, lo, up, min_gap=0.):
        th = self.thickness(lo, up)[self.inside]
        return bool(np.all(np.isfinite(th)) and np.min(th[self.inner]) >= min_gap)

    def volume(self, lo, up):
        th = np.nan_to_num(np.clip(self.thickness(lo, up), 0, None))
        return 2*float((self.tri_area*th[self.tri_idx].mean(1)).sum())


def check(lo, up, min_gap=0.):
    """Simple validity screen: finite, no streamwise fold-back, upper above lower away from the leading edge."""
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(up))):
        return False, 'nonfinite'
    for s in (lo, up):
        mid = (s[:-1]+s[1:])/2
        if np.any(np.diff(mid[:, :, 0], axis=1) < -1e-12):
            return False, 'foldback'
    th = thickness(lo, up)
    inner = th[1:-1, 1:]
    inner = inner[np.isfinite(inner)]
    if np.min(inner) < min_gap:
        return False, f'thickness {np.min(inner):.4f} m'
    return True, 'ok'

"""Geometric metrics computed from the stored native surfaces only.

No aerodynamic, performance or trajectory quantity is computed anywhere in
this task. Everything here is pure mesh geometry in the program coordinate
system of build_geometry_core (x downstream, y spanwise half model, z toward
the shock), with the native 50x50 grids exactly as MATLAB saved them.

Volume needs a closed surface. The native output is an open-base half skin, so
the base ring is capped by a triangle fan between the lower and upper tail
curves for the volume integral only. This is recorded as
volume_closure = BASE_CAP_ADDED_FOR_VOLUME_ONLY and changes no stored surface.
"""
from __future__ import annotations

import numpy as np
from scipy.io import loadmat


def load_surfaces(mat_path):
    m = loadmat(mat_path)
    lower = np.stack([m["wav_" + a] for a in ("x", "y", "z")], -1).astype(float)
    upper = np.stack([m["wav_up_" + a] for a in ("x", "y", "z")], -1).astype(float)
    return lower, upper


def _tris(grid):
    m, n = grid.shape[:2]
    ids = np.arange(m * n).reshape(m, n)
    a = ids[:-1, :-1].ravel()
    b = ids[1:, :-1].ravel()
    c = ids[1:, 1:].ravel()
    d = ids[:-1, 1:].ravel()
    faces = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    return grid.reshape(-1, 3)[faces]


def _projected_area(grid):
    t = _tris(grid)
    return float(np.abs(np.cross(t[:, 1, :2] - t[:, 0, :2], t[:, 2, :2] - t[:, 0, :2])).sum() / 2)


def _signed_volume(tris):
    return float(np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum() / 6)


def _base_cap(lower, upper):
    """Triangle fan between the two tail curves, closing the open base."""
    lo = lower[:, -1, :]
    up = upper[:, -1, :]
    n = min(len(lo), len(up))
    quads = []
    for i in range(n - 1):
        quads.append([lo[i], lo[i + 1], up[i + 1]])
        quads.append([lo[i], up[i + 1], up[i]])
    return np.asarray(quads) if quads else np.empty((0, 3, 3))


def metrics(mat_path):
    lower, upper = load_surfaces(mat_path)
    allpts = np.concatenate([lower.reshape(-1, 3), upper.reshape(-1, 3)])
    x0, x1 = float(allpts[:, 0].min()), float(allpts[:, 0].max())
    length = x1 - x0
    half_span = float(lower[:, :, 1].max())
    out = {
        "length": length,
        "half_span": half_span,
        "span": 2 * half_span,
        "aspect_ratio_span_over_length": (2 * half_span / length) if length > 0 else None,
        "x_range": [x0, x1],
        "lower_z_range": [float(lower[:, :, 2].min()), float(lower[:, :, 2].max())],
        "upper_z_range": [float(upper[:, :, 2].min()), float(upper[:, :, 2].max())],
        "leading_edge_max_gap": float(np.linalg.norm(lower[:, 0] - upper[:, 0], axis=1).max()),
        "lower_shape": list(lower.shape[:2]),
        "upper_shape": list(upper.shape[:2]),
    }
    # Thickness on the native grid rows: both surfaces share the leading edge
    # and the same row count, so row-wise comparison is a mesh-level diagnostic
    # (the audit's section thickness remains the binding geometric check).
    t = lower[:, :, 2] - upper[:, :, 2]
    out["grid_thickness_min"] = float(t.min())
    out["grid_thickness_max"] = float(t.max())
    out["grid_thickness_mean"] = float(t.mean())
    out["planform_area_half"] = _projected_area(lower)
    out["planform_area_full"] = 2 * out["planform_area_half"]
    tris = np.concatenate([_tris(lower), _tris(upper), _base_cap(lower, upper)])
    half = abs(_signed_volume(tris))
    out["volume_half"] = half
    out["volume_full"] = 2 * half
    out["volume_closure"] = "BASE_CAP_ADDED_FOR_VOLUME_ONLY"
    sp = out["planform_area_full"]
    out["volumetric_efficiency"] = (out["volume_full"] ** (2 / 3) / sp) if sp > 0 else None
    out["volumetric_efficiency_definition"] = "V_full**(2/3) / S_planform_full"
    out["wetted_area_lower_half"] = float(
        np.linalg.norm(np.cross(_tris(lower)[:, 1] - _tris(lower)[:, 0],
                                _tris(lower)[:, 2] - _tris(lower)[:, 0]), axis=1).sum() / 2)
    out["wetted_area_upper_half"] = float(
        np.linalg.norm(np.cross(_tris(upper)[:, 1] - _tris(upper)[:, 0],
                                _tris(upper)[:, 2] - _tris(upper)[:, 0]), axis=1).sum() / 2)
    le = lower[:, 0, :]
    dx = le[-1, 0] - le[0, 0]
    dy = le[-1, 1] - le[0, 1]
    out["leading_edge_mean_sweep_deg"] = float(np.degrees(np.arctan2(abs(dx), abs(dy)))) if dy else None
    out["base_height_at_symmetry"] = float(abs(lower[-1, -1, 2] - upper[-1, -1, 2]))
    return out

"""Teacher label tables: separable PCHIP (Mach first, then alpha), no extrapolation.

Every table is built only from real Teacher labels (V2 database = V2_TEACHER, stage-2
ledger = NEW_TEACHER). Interpolated values are never written back as labels.
"""
from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np
from scipy.interpolate import PchipInterpolator

from common import AERO_DB, add_paths

add_paths()
from morphing_waverider.closed_loop.aero_tables import pchip_slopes  # noqa: E402


class DomainExit(ValueError):
    pass


class Table:
    def __init__(self, name, labels, machs, alphas):
        """labels: {(M, alpha_deg): (CL, CD)}; every requested node must be present and PASS."""
        self.name = name
        self.machs = np.asarray(sorted(machs), float)
        self.alphas = np.asarray(sorted(alphas), float)
        grid = np.empty((len(self.machs), len(self.alphas), 2))
        for i, m in enumerate(self.machs):
            for j, a in enumerate(self.alphas):
                v = labels.get((float(m), float(a)))
                if v is None or not np.all(np.isfinite(v)):
                    raise KeyError(f"{name}: missing/failed label M={m} alpha={a}")
                grid[i, j] = v
        self.grid = grid
        self.cm = PchipInterpolator(self.machs, grid, axis=0, extrapolate=False).c
        self._fine = {}

    def alpha_nodes(self, mach):
        mach = float(mach)
        if not (self.machs[0] - 1e-12 <= mach <= self.machs[-1] + 1e-12):
            raise DomainExit(f"{self.name}: Mach={mach}")
        im = min(int(np.searchsorted(self.machs, mach, side="right") - 1), len(self.machs) - 2)
        dx = mach - self.machs[im]
        c = self.cm[:, im]
        return ((c[0] * dx + c[1]) * dx + c[2]) * dx + c[3]

    def coefficients(self, mach, alpha_rad, eta=None):
        a = float(alpha_rad) * 180.0 / np.pi
        if not (self.alphas[0] - 1e-12 <= a <= self.alphas[-1] + 1e-12):
            raise DomainExit(f"{self.name}: alpha_deg={a}")
        nodes = self.alpha_nodes(mach)
        da = np.diff(self.alphas)
        d = pchip_slopes(self.alphas, nodes)
        ia = min(int(np.searchsorted(self.alphas, a, side="right") - 1), len(self.alphas) - 2)
        z = a - self.alphas[ia]
        h = da[ia]
        v0, v1 = nodes[ia], nodes[ia + 1]
        s = (v1 - v0) / h
        d0, d1 = d[ia], d[ia + 1]
        o = v0 + d0 * z + (3 * s - 2 * d0 - d1) * z * z / h + (d0 + d1 - 2 * s) * z ** 3 / (h * h)
        return float(o[0]), float(o[1]), None

    def curve(self, mach, step=0.01):
        """Dense alpha curve at one Mach: (alpha_deg, CL, CD) via exact PCHIP in alpha."""
        nodes = self.alpha_nodes(mach)
        af = np.arange(self.alphas[0], self.alphas[-1] + 1e-9, step)
        p = PchipInterpolator(self.alphas, nodes, axis=0, extrapolate=False)(af)
        return af, p[:, 0], p[:, 1]

    def iso_cl(self, mach, cl, step=0.01):
        """(alpha_deg, CD) at the lowest alpha where CL(alpha) reaches cl; None if not reachable."""
        af, c, d = self.curve(mach, step)
        k = np.nonzero(np.diff(np.sign(c - cl)) != 0)[0]
        if c[0] == cl:
            return float(af[0]), float(d[0])
        if not len(k):
            return None
        i = k[0]
        w = (cl - c[i]) / (c[i + 1] - c[i])
        return float(af[i] + w * (af[i + 1] - af[i])), float(d[i] + w * (d[i + 1] - d[i]))


def read_v2_labels(geometry_ids=None, machs=None):
    """{geometry_id: {(M, alpha): (CL, CD)}} from V2 aero_database.csv + g42_baseline.csv."""
    out = defaultdict(dict)
    meta = {}
    want = set(geometry_ids) if geometry_ids else None
    mset = {float(m) for m in machs} if machs else None
    for name in ("aero_database.csv", "g42_baseline.csv"):
        with open(AERO_DB / name, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                g = r["geometry_id"]
                if want is not None and g not in want:
                    continue
                m = float(r["flight_mach"])
                if mset is not None and m not in mset:
                    continue
                if r["status"] != "PASS":
                    continue
                out[g][(m, float(r["alpha_deg"]))] = (float(r["CL"]), float(r["CD"]))
                meta.setdefault(g, {"geometry_hash": r["geometry_hash"], "design_mach": r["design_mach"],
                                    "volume_full": float(r["volume_full"]),
                                    "grid_thickness_max": float(r["grid_thickness_max"])})
    return out, meta


def labels_from_ledger(committed, geometry_hash):
    """{(M, alpha): (CL, CD)} for PASS points of one geometry from the stage-2 ledger."""
    out = {}
    for (h, m, a), p in committed.items():
        if h == geometry_hash and p.get("status") == "PASS" and p.get("CL") is not None:
            out[(m, a)] = (float(p["CL"]), float(p["CD"]))
    return out

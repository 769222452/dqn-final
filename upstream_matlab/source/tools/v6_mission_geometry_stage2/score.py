"""Mission-weighted iso-CL drag scoring (same rule for V2 and new geometries)."""
from __future__ import annotations

import numpy as np

from aerotab import Table
from common import BANDS, OUT, SCREEN_ALPHA, SCREEN_MACH, read_csv


def scored_conditions():
    rows = read_csv(OUT / "mission_design_conditions.csv")
    out = []
    for r in rows:
        m = float(r["Mach"])
        if r["scored"] != "True" or not (SCREEN_MACH[0] <= m <= SCREEN_MACH[-1]):
            continue
        out.append({"id": r["condition_id"], "band": r["band"], "M": m, "CL": float(r["CL_required"]),
                    "w": float(r["drag_work_weight"]), "q": float(r["q_pa"]), "V": float(r["V_mps"])})
    return out


def screen_table(name, labels):
    return Table(name, labels, SCREEN_MACH, SCREEN_ALPHA)


def evaluate(table, conds):
    """Per-condition iso-CL (alpha, CD); None where CL is not reachable in alpha range."""
    return {c["id"]: table.iso_cl(c["M"], c["CL"]) for c in conds}


def scores(per_cond, ref_cond, conds):
    """J (drag-work ratio vs reference) overall and per band; NaN if any condition unreachable."""
    res = {}
    for key, sel in [("ALL", conds)] + [(b, [c for c in conds if c["band"] == b]) for b in BANDS]:
        if not sel:
            continue
        num = den = 0.0
        ok = True
        for c in sel:
            v, r = per_cond.get(c["id"]), ref_cond.get(c["id"])
            if v is None or r is None:
                ok = False
                break
            num += c["w"] * v[1]
            den += c["w"] * r[1]
        res[key] = num / den if ok and den > 0 else float("nan")
        res[key + "_reachable"] = ok
    return res

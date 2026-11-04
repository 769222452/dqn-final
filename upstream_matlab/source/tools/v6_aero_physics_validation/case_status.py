"""Add per-phase execution status columns to VALIDATION_CASE_MATRIX.csv (derived from the phase outputs).

Idempotent: only the status_* columns are (re)written; all other columns are kept as produced by p0_geometry.py.
"""
from __future__ import annotations

import json

import common as C

STATUS_COLS = ["status_P1_design_pressure", "status_P2_low_order", "status_P3_drag_components",
               "status_P4_shock_screening", "status_CFD", "cfd_config"]


def main():
    cases = C.read_csv(C.OUT / "VALIDATION_CASE_MATRIX.csv")
    p1 = {r["geometry_id"] for r in C.read_csv(C.OUT / "DESIGN_POINT_PRESSURE_COMPARISON.csv") if r["level"] == "GEOMETRY"}
    p2 = {r["case_id"] for r in C.read_csv(C.OUT / "LOW_ORDER_MODEL_COMPARISON.csv")}
    p3 = {r["case_id"] for r in C.read_csv(C.OUT / "DRAG_COMPONENT_SENSITIVITY.csv")}
    p4 = {r["case_id"] for r in C.read_csv(C.OUT / "SHOCK_MISMATCH_SCREENING.csv")}
    cfd = json.loads((C.OUT / "CFD_AVAILABILITY.json").read_text(encoding="utf-8"))
    solver_ok = bool(cfd.get("real_cfd_solver_available", False))
    out = []
    for c in cases:
        r = {k: v for k, v in c.items() if k not in STATUS_COLS}
        cid, design = c["case_id"], c["kind"] == "DESIGN_POINT"
        r["status_P1_design_pressure"] = ("DONE" if c["geometry_id"] in p1 else "MISSING") if design else "N/A"
        r["status_P2_low_order"] = "DONE" if cid in p2 else ("N/A" if design else "MISSING")
        r["status_P3_drag_components"] = "DONE" if cid in p3 else ("N/A (design point has no mission q/Re)" if design else "MISSING")
        r["status_P4_shock_screening"] = "DONE" if cid in p4 else "MISSING"
        cfg = C.OUT / "cfd_inputs" / C.SHORT.get(c["geometry_id"], c["geometry_id"]) / f"su2_{cid}.cfg"
        if not cfg.exists():
            cfg = next((p for p in (C.OUT / "cfd_inputs").rglob(f"su2_{cid}.cfg")), None)
        r["cfd_config"] = str(cfg.relative_to(C.OUT)).replace("\\", "/") if cfg else ""
        r["status_CFD"] = ("NOT_RUN (no solver; config prepared)" if cfg else "NOT_RUN (no solver; not in first CFD batch)") \
            if not solver_ok else "PENDING"
        out.append(r)
    C.write_csv(C.OUT / "VALIDATION_CASE_MATRIX.csv", out)
    from collections import Counter
    for k in STATUS_COLS[:-1]:
        print(k, dict(Counter(r[k] for r in out)))


if __name__ == "__main__":
    main()

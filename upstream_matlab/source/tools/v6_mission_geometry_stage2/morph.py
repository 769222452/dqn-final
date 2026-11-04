"""P3: continuous morph paths with natively generated intermediates.

A path is the straight line in the generator input space p(s) = (1-s) p_A + s p_C with
p = (Ma_design, a1, a2, b1, b2, d1, a_x). Every node geometry is produced by the native
MATLAB generator and the frozen audit; nothing is interpolated in MAT coordinates.
Crossing Ma_design is therefore allowed only when every node on the line generates VALID.

Refinement (all evidence is Teacher):
  level 0 nodes s = 0, .25, .5, .75, 1 -> Teacher MISSION-grid tables;
  level k: for each segment generate its midpoint, Teacher-label it and compare the real
  midpoint table with the linear blend of its neighbours (the rule used by the 1200 s
  simulation). The midpoint always becomes a node; segments whose residual exceeds the
  tolerance are refined again (max depth 3). Reported residuals are those of the final level.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import geometry
import teacher_pool
from aerotab import labels_from_ledger, read_v2_labels
from common import MISSION_ALPHA, MISSION_MACH, OUT, PARAMS, ACTUATOR_RATE_PATH_PER_S, dump, load

CL_TOL = 0.02     # |CL_mid - CL_blend| / max(|CL_mid|, 0.02)
CD_TOL = 0.02     # |CD_mid - CD_blend| / CD_mid
CHECK_MACH = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
CHECK_ALPHA = [2.0, 4.0, 6.0, 8.0, 10.0]
MAX_DEPTH = 3

_V2_BY_HASH = None


def v2_labels_by_hash():
    global _V2_BY_HASH
    if _V2_BY_HASH is None:
        lab, meta = read_v2_labels(machs=MISSION_MACH)
        _V2_BY_HASH = {meta[g]["geometry_hash"]: (g, lab[g]) for g in lab}
    return _V2_BY_HASH


def mission_labels(geometry_hash, committed=None):
    """Teacher labels on the MISSION grid: stage-2 ledger, plus V2 labels of an identical hash."""
    committed = committed if committed is not None else teacher_pool.committed()
    lab = {}
    v2 = v2_labels_by_hash().get(geometry_hash)
    if v2:
        lab.update({k: v for k, v in v2[1].items() if k[1] in MISSION_ALPHA})
    lab.update(labels_from_ledger(committed, geometry_hash))
    return lab


def mission_complete(geometry_hash, committed=None):
    lab = mission_labels(geometry_hash, committed)
    return all((m, a) in lab for m in MISSION_MACH for a in MISSION_ALPHA)


def teacher_mission(recs, workers=3, log=print, deadline=None):
    committed = teacher_pool.committed()
    tasks = []
    for r in recs:
        if r.get("classification") != "VALID":
            continue
        have = mission_labels(r["geometry_hash"], committed)
        for m in MISSION_MACH:
            al = [a for a in MISSION_ALPHA if (m, a) not in have]
            if al:
                tasks.append({"geometry_id": r["candidate_id"], "geometry_hash": r["geometry_hash"],
                              "geometry_mat": r["native_geometry_mat"].replace("\\", "/"), "M_flight": m,
                              "alpha_deg": al})
    if tasks:
        return teacher_pool.run(tasks, n_workers=workers, log=log, deadline=deadline)
    return {"requests": 0}


def point(pA, pC, s):
    return (1 - s) * np.asarray(pA, float) + s * np.asarray(pC, float)


def node_candidate(path_name, pA, pC, s):
    p = point(pA, pC, s)
    return geometry.make_candidate(float(p[0]), p[1:], "morph_path_node", path=path_name, s=float(s))


def _gen(cands, stage, log):
    geometry.generate(cands, stage, sessions=min(3, len(cands)), log=log)
    return {c["candidate_id"]: r for c, r in zip(cands, geometry.audit(cands, workers=6, log=log))}


def _grid(lab, machs=CHECK_MACH, alphas=CHECK_ALPHA):
    return np.array([[lab[(m, a)] for a in alphas] for m in machs])  # (M, A, 2)


def blend_residual(lab_l, lab_r, lab_m, w=0.5):
    gl, gr, gm = _grid(lab_l), _grid(lab_r), _grid(lab_m)
    b = (1 - w) * gl + w * gr
    cl = np.abs(gm[..., 0] - b[..., 0]) / np.maximum(np.abs(gm[..., 0]), 0.02)
    cd = np.abs(gm[..., 1] - b[..., 1]) / gm[..., 1]
    return float(cl.max()), float(cd.max())


def build_path(name, endA, endC, *, workers=3, log=print, deadline=None, scenario_limits=None, ref=None):
    """endA/endC: dict(Ma_design, z, label). Returns the path record (also written to morph/<name>.json)."""
    t0 = time.time()
    pA = np.r_[endA["Ma_design"], endA["z"]]
    pC = np.r_[endC["Ma_design"], endC["z"]]
    nodes = {}  # s -> record

    def add(svals, level):
        cands = [node_candidate(name, pA, pC, s) for s in svals]
        recs = _gen(cands, f"morph_{name}", log)
        for s, c in zip(svals, cands):
            r = recs[c["candidate_id"]]
            r["s"] = float(s)
            r["level"] = level
            nodes[float(s)] = r
        teacher_mission([nodes[float(s)] for s in svals], workers, log, deadline)

    add([0.0, 0.25, 0.5, 0.75, 1.0], 0)
    segments = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
    checks = []
    for depth in range(1, MAX_DEPTH + 1):
        if not segments:
            break
        if any(nodes[s].get("classification") != "VALID" for s in nodes):
            break
        mids = [(a + b) / 2 for a, b in segments]
        add(mids, depth)
        committed = teacher_pool.committed()
        nxt = []
        for (a, b), mid in zip(segments, mids):
            rec = {"segment": [a, b], "mid": mid, "depth": depth}
            if nodes[mid].get("classification") != "VALID":
                rec["status"] = "MIDPOINT_NOT_VALID"
                checks.append(rec)
                continue
            la, lb, lm = (mission_labels(nodes[x]["geometry_hash"], committed) for x in (a, b, mid))
            try:
                rcl, rcd = blend_residual(la, lb, lm)
            except KeyError:
                rec["status"] = "MISSING_TEACHER_LABELS"
                checks.append(rec)
                continue
            rec.update({"cl_rel_residual_max": rcl, "cd_rel_residual_max": rcd,
                        "status": "PASS" if rcl <= CL_TOL and rcd <= CD_TOL else "REFINE"})
            checks.append(rec)
            if rec["status"] == "REFINE":
                nxt += [(a, mid), (mid, b)]
        segments = nxt
    return finish(name, endA, endC, nodes, checks, segments, t0, scenario_limits, ref)


def finish(name, endA, endC, nodes, checks, unresolved, t0, scenario_limits=None, ref=None):
    order = sorted(nodes)
    committed = teacher_pool.committed()
    out_nodes = []
    prev = None
    cum_max = cum_rms = 0.0
    for s in order:
        r = nodes[s]
        n = {"s": s, "level": r["level"], "candidate_id": r["candidate_id"], "Ma_design": r["Ma_design"],
             **{p: r[p] for p in PARAMS}, "generation_status": r.get("generation_status"),
             "classification": r.get("classification"), "geometry_hash": r.get("geometry_hash"),
             "native_geometry_mat": r.get("native_geometry_mat"), "volume_full": r.get("volume_full"),
             "grid_thickness_max": r.get("grid_thickness_max"), "span": r.get("span"), "length": r.get("length")}
        if ref:
            if n["volume_full"] is not None:
                n["volume_ratio_G42"] = n["volume_full"] / ref["volume_full"]
                n["thickness_ratio_G42"] = n["grid_thickness_max"] / ref["grid_thickness_max"]
        n["teacher_mission_grid_complete"] = bool(r.get("geometry_hash")) and mission_complete(r["geometry_hash"], committed)
        if prev is not None and prev.get("native_geometry_mat") and n["native_geometry_mat"]:
            d = geometry.shape_distance(prev["native_geometry_mat"], n["native_geometry_mat"])
            n["from_previous"] = d
            if d.get("compatible_topology"):
                cum_max += d["max_displacement_m"]
                cum_rms += d["rms_displacement_m"]
        n["cum_max_displacement_m"] = cum_max
        n["cum_rms_displacement_m"] = cum_rms
        out_nodes.append(n)
        prev = n
    all_valid = all(n["classification"] == "VALID" for n in out_nodes)
    teach = all(n["teacher_mission_grid_complete"] for n in out_nodes)
    final_checks = [c for c in checks if c.get("status") in ("PASS", "REFINE")]
    worst_cl = max((c["cl_rel_residual_max"] for c in final_checks), default=None)
    worst_cd = max((c["cd_rel_residual_max"] for c in final_checks), default=None)
    end_d = None
    if out_nodes[0]["native_geometry_mat"] and out_nodes[-1]["native_geometry_mat"]:
        end_d = geometry.shape_distance(out_nodes[0]["native_geometry_mat"], out_nodes[-1]["native_geometry_mat"])
    scen = {}
    if scenario_limits and ref:
        for k, lim in scenario_limits.items():
            scen[k] = all(n.get("volume_ratio_G42", 0) >= lim["volume_ratio_min"] - 1e-12 and
                          n.get("thickness_ratio_G42", 0) >= lim["thickness_ratio_min"] - 1e-12 for n in out_nodes)
    status = ("VALIDATED" if all_valid and teach and not unresolved else
              "VALID_GEOMETRY_AERO_RESIDUAL_ABOVE_TOL" if all_valid and teach else
              "TEACHER_INCOMPLETE" if all_valid else "BROKEN_INVALID_INTERMEDIATE")
    transition_s = 1.0 / ACTUATOR_RATE_PATH_PER_S
    rec = {"path": name, "status": status, "endpoint_A": endA, "endpoint_C": endC,
           "crosses_design_mach": abs(endA["Ma_design"] - endC["Ma_design"]) > 1e-9,
           "parametrisation": "straight line in native generator inputs (Ma_design, a1, a2, b1, b2, d1, a_x); "
                              "c1=c2=1.0331, AOA=-4 fixed; every node regenerated natively",
           "nodes": out_nodes, "n_nodes": len(out_nodes), "blend_checks": checks,
           "unresolved_segments": unresolved,
           "worst_final_cl_rel_residual": worst_cl, "worst_final_cd_rel_residual": worst_cd,
           "tolerances": {"CL_rel": CL_TOL, "CD_rel": CD_TOL, "check_mach": CHECK_MACH, "check_alpha": CHECK_ALPHA},
           "endpoint_shape_distance": end_d,
           "path_length_sum_max_displacement_m": cum_max, "path_length_sum_rms_displacement_m": cum_rms,
           "actuator": {"rate_path_per_s": ACTUATOR_RATE_PATH_PER_S, "full_transition_s": transition_s,
                        "mean_max_point_speed_mps": cum_max * ACTUATOR_RATE_PATH_PER_S,
                        "mean_rms_point_speed_mps": cum_rms * ACTUATOR_RATE_PATH_PER_S,
                        "note": "0.1 path/s is a research assumption; the physical deformation it implies is "
                                "reported as surface-point displacement speed (native grid index correspondence)"},
           "scenario_satisfied_all_nodes": scen,
           "wall_s": time.time() - t0}
    dump(OUT / "morph" / f"{name}.json", rec)
    return rec

"""Stage 4: morph paths inside each design Mach, validated with real geometries.

An edge between two VALID geometries of the same Ma_design is a straight line
in the 6-D design vector. For every edge this stage generates the intermediate
geometries at rho = 0.25, 0.50, 0.75 through the same confirmed MATLAB core and
audits each one with the same frozen rules. An edge is CONNECTED only when
every intermediate geometry is really generated and really classified VALID.
No edge is ever marked reachable on the basis of interpolation alone.

Edge set per design Mach: the minimum spanning tree of the VALID nodes in
normalised design space, plus the shortest additional non-tree edges, so that
connectivity is tested rather than assumed.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import mmg
from mmg import OUT, PARAMS, DesignSpace, dump

RHOS = (0.25, 0.50, 0.75)
EXTRA_EDGES_PER_MACH = 3
NODE_COLUMNS = ["geometry_id", "candidate_id", "Ma_design", "design_space", *PARAMS,
                "degree_connected", "degree_tested", "component_id"]
EDGE_COLUMNS = ["edge_id", "Ma_design", "from_geometry_id", "to_geometry_id", "edge_type",
                "normalised_distance", "intermediate_count", "generated_count", "valid_count",
                "status", "first_failure_rho", "first_failure_reason"]
VALIDATION_COLUMNS = ["edge_id", "Ma_design", "rho", "candidate_id", "from_geometry_id",
                      "to_geometry_id", *PARAMS, "generation_status", "classification",
                      "geometry_hash", "min_section_thickness", "surface_intersection_count",
                      "self_intersection_count", "failure_reason"]


def normalise(space, z):
    lo, hi = space.extended_lower, space.extended_upper
    return (np.asarray(z, float) - lo) / (hi - lo)


def mst_edges(points):
    """Prim's algorithm on the complete graph; returns (i, j, distance)."""
    n = len(points)
    if n < 2:
        return []
    inside = {0}
    edges = []
    while len(inside) < n:
        best = None
        for i in inside:
            for j in range(n):
                if j in inside:
                    continue
                d = float(np.linalg.norm(points[i] - points[j]))
                if best is None or d < best[2]:
                    best = (i, j, d)
        edges.append(best)
        inside.add(best[1])
    return edges


def extra_edges(points, chosen, count):
    n = len(points)
    taken = {tuple(sorted((i, j))) for i, j, _ in chosen}
    pool = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in taken:
                continue
            pool.append((i, j, float(np.linalg.norm(points[i] - points[j]))))
    pool.sort(key=lambda e: e[2])
    return pool[:count]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--audit-only", action="store_true")
    args = ap.parse_args()
    space = DesignSpace()
    space.verify()

    valid = mmg.read_csv(OUT / "valid_geometry_inventory.csv")
    by_mach = defaultdict(list)
    for r in valid:
        by_mach[float(r["Ma_design"])].append(r)

    edges, cands = [], []
    for ma in sorted(by_mach):
        nodes = sorted(by_mach[ma], key=lambda r: r["geometry_id"])
        pts = np.asarray([normalise(space, [float(r[p]) for p in PARAMS]) for r in nodes])
        tree = mst_edges(pts)
        extra = extra_edges(pts, tree, EXTRA_EDGES_PER_MACH)
        for kind, group in (("MST", tree), ("EXTRA", extra)):
            for i, j, d in group:
                a, b = nodes[i], nodes[j]
                eid = f"E_MD{ma:g}_{a['geometry_id'].split('_')[-1]}_{b['geometry_id'].split('_')[-1]}_{kind}"
                za = np.asarray([float(a[p]) for p in PARAMS])
                zb = np.asarray([float(b[p]) for p in PARAMS])
                edge = {"edge_id": eid, "Ma_design": ma, "from_geometry_id": a["geometry_id"],
                        "to_geometry_id": b["geometry_id"], "edge_type": kind,
                        "normalised_distance": d, "intermediate_count": len(RHOS),
                        "from_z": za.tolist(), "to_z": zb.tolist(), "intermediates": []}
                for rho in RHOS:
                    z = (1 - rho) * za + rho * zb
                    cid = mmg.candidate_id(ma, z, prefix="MORPH")
                    edge["intermediates"].append({"rho": rho, "candidate_id": cid})
                    cands.append({"candidate_id": cid, "Ma_design": ma, "z": z.tolist(),
                                  "source": "morph_intermediate", "seed_design_index": None,
                                  "path_id": eid, "path_k": rho,
                                  "from_geometry_id": a["geometry_id"],
                                  "to_geometry_id": b["geometry_id"]})
                edges.append(edge)

    # The same interpolated point can be shared by two edges; keep one case dir.
    unique, seen = [], set()
    for c in cands:
        if c["candidate_id"] in seen:
            continue
        seen.add(c["candidate_id"])
        unique.append(c)
    for i, c in enumerate(unique, start=1):
        c["job_index"] = i
        c["design_space"] = space.label(c["z"])
        c["case_dir"] = str(OUT / "cases" / c["candidate_id"])
        c["inputs10"] = space.inputs(c["Ma_design"], c["z"])
    dump(OUT / "generation" / "stage4_candidates.json", unique)
    dump(OUT / "generation" / "stage4_edges.json", edges)
    print(f"[stage4] {len(edges)} edges, {len(unique)} intermediate geometries "
          f"{dict(Counter(e['Ma_design'] for e in edges))}", flush=True)

    if not args.audit_only:
        mmg.generate(space, unique, "stage4")
    results = mmg.audit_parallel(unique, workers=args.workers)
    audit_by_id = {r["candidate_id"]: r for r in results}
    gen_by_id = {}
    for c in unique:
        p = Path(c["case_dir"]) / "generation_status.json"
        gen_by_id[c["candidate_id"]] = mmg.load(p) if p.exists() else {"status": "NOT_RUN"}

    validation_rows, edge_rows = [], []
    degree_connected = Counter()
    degree_tested = Counter()
    adjacency = defaultdict(set)
    for e in edges:
        gen = val = 0
        first_fail = None
        for item in e["intermediates"]:
            cid = item["candidate_id"]
            a = audit_by_id.get(cid, {})
            g = gen_by_id.get(cid, {})
            cls = a.get("classification") or ("GENERATION_FAILED" if g.get("status") != "GENERATED"
                                              else None)
            if g.get("status") == "GENERATED":
                gen += 1
            if cls == "VALID":
                val += 1
            elif first_fail is None:
                first_fail = (item["rho"], (a.get("failure_reason") or g.get("error_message")))
            c = next(x for x in unique if x["candidate_id"] == cid)
            validation_rows.append({
                "edge_id": e["edge_id"], "Ma_design": e["Ma_design"], "rho": item["rho"],
                "candidate_id": cid, "from_geometry_id": e["from_geometry_id"],
                "to_geometry_id": e["to_geometry_id"],
                **{p: float(c["z"][k]) for k, p in enumerate(PARAMS)},
                "generation_status": g.get("status"), "classification": cls,
                "geometry_hash": a.get("geometry_hash"),
                "min_section_thickness": a.get("min_section_thickness"),
                "surface_intersection_count": a.get("surface_intersection_count"),
                "self_intersection_count": a.get("self_intersection_count"),
                "failure_reason": a.get("failure_reason") or g.get("error_message")})
        status = "CONNECTED" if val == len(RHOS) else "BLOCKED"
        degree_tested[e["from_geometry_id"]] += 1
        degree_tested[e["to_geometry_id"]] += 1
        if status == "CONNECTED":
            degree_connected[e["from_geometry_id"]] += 1
            degree_connected[e["to_geometry_id"]] += 1
            adjacency[e["from_geometry_id"]].add(e["to_geometry_id"])
            adjacency[e["to_geometry_id"]].add(e["from_geometry_id"])
        edge_rows.append({k: e[k] for k in ("edge_id", "Ma_design", "from_geometry_id",
                                            "to_geometry_id", "edge_type",
                                            "normalised_distance", "intermediate_count")}
                         | {"generated_count": gen, "valid_count": val, "status": status,
                            "first_failure_rho": first_fail[0] if first_fail else None,
                            "first_failure_reason": first_fail[1] if first_fail else None})

    # connected components over CONNECTED edges, per design Mach
    component_of = {}
    comp_id = 0
    for ma in sorted(by_mach):
        for r in sorted(by_mach[ma], key=lambda r: r["geometry_id"]):
            gid = r["geometry_id"]
            if gid in component_of:
                continue
            comp_id += 1
            stack = [gid]
            while stack:
                cur = stack.pop()
                if cur in component_of:
                    continue
                component_of[cur] = comp_id
                stack.extend(adjacency[cur] - set(component_of))

    node_rows = []
    for ma in sorted(by_mach):
        for r in sorted(by_mach[ma], key=lambda r: r["geometry_id"]):
            gid = r["geometry_id"]
            node_rows.append({"geometry_id": gid, "candidate_id": r["candidate_id"],
                              "Ma_design": ma, "design_space": r["design_space"],
                              **{p: float(r[p]) for p in PARAMS},
                              "degree_connected": degree_connected[gid],
                              "degree_tested": degree_tested[gid],
                              "component_id": component_of[gid]})
    unconnected = [r for r in node_rows if r["degree_connected"] == 0]

    mmg.write_csv(OUT / "morph_path_nodes.csv", node_rows, NODE_COLUMNS)
    mmg.write_csv(OUT / "morph_path_edges.csv", edge_rows, EDGE_COLUMNS)
    mmg.write_csv(OUT / "morph_path_validation.csv", validation_rows, VALIDATION_COLUMNS)
    mmg.write_csv(OUT / "unconnected_candidates.csv", unconnected, NODE_COLUMNS)

    per_mach = {}
    for ma in sorted(by_mach):
        e = [r for r in edge_rows if r["Ma_design"] == ma]
        n = [r for r in node_rows if r["Ma_design"] == ma]
        comps = sorted({r["component_id"] for r in n})
        sizes = Counter(r["component_id"] for r in n)
        per_mach[f"{ma:g}"] = {
            "nodes": len(n), "edges_tested": len(e),
            "edges_connected": sum(1 for r in e if r["status"] == "CONNECTED"),
            "edges_blocked": sum(1 for r in e if r["status"] == "BLOCKED"),
            "connected_components": len(comps),
            "largest_component_size": max(sizes.values()) if sizes else 0,
            "isolated_nodes": sum(1 for r in n if r["degree_connected"] == 0)}
    summary = {"utc": mmg.now(), "rhos": list(RHOS),
               "edge_rule": "CONNECTED requires every intermediate geometry GENERATED and VALID",
               "edges_tested": len(edge_rows),
               "edges_connected": sum(1 for r in edge_rows if r["status"] == "CONNECTED"),
               "intermediate_geometries": len(unique),
               "intermediate_counts": dict(Counter(r["classification"] for r in validation_rows)),
               "per_Ma_design": per_mach,
               "cross_Ma_design_edges": "NOT_BUILT (paths are within one Ma_design by design)",
               "unconnected_node_count": len(unconnected)}
    dump(OUT / "morph_path_validation.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

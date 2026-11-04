"""P1b: mission-weighted analysis of all 348 VALID V2 geometries (+ G42 baseline).

Uses only real V2 Teacher labels restricted to the stage-2 screening node set
(Mach 5..8 step 0.5, alpha 0..12 deg) so that V2 and new geometries are scored by the
identical interpolation rule.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from aerotab import read_v2_labels
from common import BANDS, GEO_DB, OUT, PARAMS, SCENARIOS, SCREEN_MACH, dump, read_csv, write_csv
from score import evaluate, scored_conditions, scores, screen_table


def load_v2_inventory():
    inv = {r["geometry_id"]: r for r in read_csv(GEO_DB / "valid_geometry_inventory.csv")}
    nodes = {r["geometry_id"]: r for r in read_csv(GEO_DB / "morph_path_nodes.csv")}
    return inv, nodes


def main():
    conds = scored_conditions()
    labels, meta = read_v2_labels(machs=SCREEN_MACH)
    inv, nodes = load_v2_inventory()
    if len(inv) != 348:
        raise RuntimeError(f"Expected 348 VALID V2 geometries, found {len(inv)}")
    ref = evaluate(screen_table("G42", labels["G42"]), conds)
    gv, gt = meta["G42"]["volume_full"], meta["G42"]["grid_thickness_max"]
    rows = []
    for gid in ["G42"] + sorted(inv):
        per = evaluate(screen_table(gid, labels[gid]), conds)
        s = scores(per, ref, conds)
        r = inv.get(gid, {})
        md = float(r["Ma_design"]) if r else 15.0
        vr, tr = meta[gid]["volume_full"] / gv, meta[gid]["grid_thickness_max"] / gt
        comp = nodes.get(gid, {}).get("component_id")
        row = {"geometry_id": gid, "origin": "V2_DATABASE" if r else "G42_BASELINE",
               "kind": "morph_intermediate" if gid.startswith("V2M") else ("design" if r else "baseline"),
               "Ma_design": md, **({p: float(r[p]) for p in PARAMS} if r else
                                   dict(zip(PARAMS, [5.633749605922896, 7.796675937828777, 3.698465133623518,
                                                     3.0116105313696075, 0.11606537709080691, 0.9944782795122264]))),
               "geometry_hash": meta[gid]["geometry_hash"], "volume_full": meta[gid]["volume_full"],
               "grid_thickness_max": meta[gid]["grid_thickness_max"], "volume_ratio_G42": vr,
               "thickness_ratio_G42": tr,
               "scenario_CONSERVATIVE": vr >= SCENARIOS["CONSERVATIVE"]["volume_ratio_min"] - 1e-12 and
                                        tr >= SCENARIOS["CONSERVATIVE"]["thickness_ratio_min"] - 1e-12,
               "scenario_EXPLORATORY": vr >= SCENARIOS["EXPLORATORY"]["volume_ratio_min"] and
                                       tr >= SCENARIOS["EXPLORATORY"]["thickness_ratio_min"],
               "morph_component": f"V2_MD{md:g}" if r else "V2_MD15 (via hash twin V2_MD15_001)",
               "reachable_from_G42_via_V2_edges": md == 15.0,
               "aero_evidence": "V2_TEACHER (real labels; iso-CL via PCHIP)",
               **{f"J_{k}": v for k, v in s.items() if not k.endswith("_reachable")},
               **{f"reachable_{k}": v for k, v in s.items() if k.endswith("_reachable")}}
        for c in conds:
            v = per.get(c["id"])
            row[f"CD@{c['id']}"] = v[1] if v else ""
        rows.append(row)
    base_cols = [k for k in rows[0] if not k.startswith("CD@")]
    write_csv(OUT / "v2_mission_scores.csv", rows, base_cols)
    write_csv(OUT / "v2_mission_iso_cl_cd.csv", rows, ["geometry_id"] + [f"CD@{c['id']}" for c in conds])

    def best(key, scen=None, md=None, n=5):
        sel = [r for r in rows if np.isfinite(r.get(f"J_{key}", np.nan))
               and (scen is None or r[f"scenario_{scen}"]) and (md is None or r["Ma_design"] == md)]
        return sorted(sel, key=lambda r: r[f"J_{key}"])[:n]

    analysis = {"n_geometries": len(rows), "conditions_scored": len(conds),
                "reference": "G42 (V2 g42_baseline.csv), J = weighted CD / G42 weighted CD", "winners": {}}
    for scen in (None, "EXPLORATORY", "CONSERVATIVE"):
        sk = scen or "UNCONSTRAINED"
        analysis["winners"][sk] = {}
        for key in ["ALL"] + list(BANDS):
            analysis["winners"][sk][key] = [{"geometry_id": r["geometry_id"], "J": r[f"J_{key}"],
                                             "Ma_design": r["Ma_design"], "volume_ratio": r["volume_ratio_G42"],
                                             "thickness_ratio": r["thickness_ratio_G42"]} for r in best(key, scen)]
    fam = defaultdict(dict)
    for md in sorted({r["Ma_design"] for r in rows}):
        for key in ["ALL"] + list(BANDS):
            b = best(key, md=md, n=1)
            fam[f"MD{md:g}"][key] = {"geometry_id": b[0]["geometry_id"], "J": b[0][f"J_{key}"]} if b else None
    analysis["best_per_design_mach_family"] = fam
    cross = {}
    for scen in ("UNCONSTRAINED", "EXPLORATORY", "CONSERVATIVE"):
        w = {b: (analysis["winners"][scen][b][0]["geometry_id"] if analysis["winners"][scen][b] else None) for b in BANDS}
        cross[scen] = {"band_winners": w, "distinct_winners": len(set(v for v in w.values() if v)),
                       "winner_families": {b: next((r["Ma_design"] for r in rows if r["geometry_id"] == v), None)
                                           for b, v in w.items()}}
    analysis["crossover"] = cross
    analysis["reachability_note"] = ("V2 morph edges exist only inside one Ma_design family (V2 stage 4, 90/90 edges "
                                     "CONNECTED). G42 is the hash twin of V2_MD15_001, so only MD15-family geometries "
                                     "are reachable from G42 through validated V2 edges.")
    dump(OUT / "v2_mission_analysis.json", analysis)
    print(json.dumps({"crossover": cross, "families": fam}, indent=1, default=str))


if __name__ == "__main__":
    main()

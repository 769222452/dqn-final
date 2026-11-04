"""Pre-run identity check: Teacher sources, worker wrapper, 348 VALID geometries, G42.

Writes provenance/identity_check.json and provenance/geometry_identity.csv.
Any geometry whose recomputed hash or MAT does not match is listed in
`quarantined` and excluded from the task queue (never silently mixed).
"""
import json
import shutil

import numpy as np
from scipy.io import loadmat

from common import (ACTIVE, CODE, ROOT, G42_HASH, G42_INPUTS10, G42_MAT, GEO_DB, MODEL_OPTIONS, OFFDESIGN_SHA, OUT,
                    PILOT, PILOT_WORKER_SHA, REFERENCE, V1_DB, dump, load, native_hash, read_csv, sha, write_csv)


def mat_hash(path, inputs10):
    m = loadmat(path)
    lower = np.stack([m["wav_" + a] for a in "xyz"], -1)
    upper = np.stack([m["wav_up_" + a] for a in "xyz"], -1)
    finite = bool(np.isfinite(lower).all() and np.isfinite(upper).all()
                  and not np.iscomplexobj(lower) and not np.iscomplexobj(upper))
    return native_hash(lower, upper, inputs10), finite, list(lower.shape)


def main():
    rep = {}
    # Teacher sources vs V1 identity
    v1 = load(V1_DB / "source_identity.json")
    cur = {}
    for name, h in v1["active_sources"].items():
        p = ACTIVE / name if "/" not in name else None
        if p is not None:
            cur[name] = sha(p) if p.exists() else None
    mism = {k: (v1["active_sources"][k], cur[k]) for k in cur if cur[k] != v1["active_sources"][k]}
    # The only accepted difference vs V1 is the formally promoted CSR1 cone-solver amendment.
    promo = load(ROOT / "results/v5_cone_solver_path_support_csr1/promotion.json")
    explained = {k: v for k, v in mism.items() if k == "comptue_conicalfield.m"
                 and v[0] == promo["previous_active_sha256"] and v[1] == promo["new_active_sha256"]
                 and promo["amendment_entry"]["physics_changed"] is False and all(promo["gates"].values())}
    unexplained = {k: v for k, v in mism.items() if k not in explained}
    rep["teacher_sources_note"] = (
        "V1 (2026-10-02) ran comptue_conicalfield c687a9f2 (Phase 6). CSR1 was promoted 2026-10-05 "
        "(fzero TolX 2e-13 -> eps, physics_changed=false, all gates PASS). The pilot and this build use the "
        "promoted active source; V1 labels are used only for the G42 regression, never mixed into the database.")
    rep["csr1_promotion"] = {"applied_utc": promo["applied_utc"], "gates": promo["gates"],
                             "new_active_sha256": promo["new_active_sha256"]}
    rep["teacher_sources"] = {"checked": len(cur), "mismatch_vs_V1": mism,
                              "mismatch_explained_by_CSR1_promotion": sorted(explained),
                              "mismatch_unexplained": unexplained,
                              "offdesign_aero_v4_sha256": cur.get("offdesign_aero_v4.m"),
                              "offdesign_matches_expected": cur.get("offdesign_aero_v4.m") == OFFDESIGN_SHA}
    rep["active_m_files"] = {p.name: sha(p) for p in sorted(ACTIVE.glob("*.m"))}
    # reference/options vs V1 and pilot
    pid = load(PILOT / "state/identity.json")
    rep["reference_matches_V1"] = (abs(v1["reference"]["area_m2"] - REFERENCE["area_m2"]) == 0
                                   and v1["reference"]["length_m"] == REFERENCE["length_m"]
                                   and v1["reference"]["moment_origin_m"] == REFERENCE["moment_origin_m"])
    rep["options_match_V1"] = v1["model_options"] == MODEL_OPTIONS
    rep["reference_matches_pilot"] = pid["reference"] == REFERENCE
    rep["options_match_pilot"] = pid["model_options"] == MODEL_OPTIONS
    rep["pilot_offdesign_sha_matches"] = pid["offdesign_aero_v4_sha256"] == OFFDESIGN_SHA
    # worker wrapper: byte copy of the pilot worker
    src = PILOT / "code/aero_worker.m"
    dst = CODE / "aero_worker.m"
    if not dst.exists():
        shutil.copyfile(src, dst)
    rep["worker"] = {"pilot_sha": sha(src), "copy_sha": sha(dst),
                     "identical_to_pilot": sha(src) == sha(dst) == PILOT_WORKER_SHA}

    inv = [r for r in read_csv(GEO_DB / "valid_geometry_inventory.csv") if r["classification"] == "VALID"]
    rows, quarantine = [], []
    for r in inv:
        mat = GEO_DB / r["native_geometry_path"]
        case_mat = GEO_DB.parent.parent / r["mat_path"] if r.get("mat_path") else None
        inputs10 = json.loads(r["inputs10"])
        row = {"geometry_id": r["geometry_id"], "Ma_design": r["Ma_design"], "design_space": r["design_space"],
               "source": r["source"], "inventory_geometry_hash": r["geometry_hash"],
               "inventory_mat_sha256": r["mat_sha256"], "native_mat": str(mat), "exists": mat.exists()}
        if mat.exists():
            h, finite, shape = mat_hash(mat, inputs10)
            row.update(native_mat_sha256=sha(mat), recomputed_geometry_hash=h, finite=finite, shape=shape,
                       native_equals_case_mat=(case_mat is not None and case_mat.exists() and sha(case_mat) == sha(mat)))
            row["hash_ok"] = h == r["geometry_hash"]
        else:
            row.update(hash_ok=False, finite=False)
        if not (row["exists"] and row["hash_ok"] and row["finite"]):
            quarantine.append(r["geometry_id"])
        rows.append(row)
    write_csv(OUT / "provenance/geometry_identity.csv", rows)
    hashes = [r["inventory_geometry_hash"] for r in rows]
    gh, gf, _ = mat_hash(G42_MAT, G42_INPUTS10)
    rep["geometries"] = {"valid_in_inventory": len(inv), "hash_ok": sum(r["hash_ok"] for r in rows),
                         "native_equals_case_mat": sum(bool(r.get("native_equals_case_mat")) for r in rows),
                         "unique_hashes": len(set(hashes)), "quarantined": quarantine,
                         "G42_id_in_valid_set": any(r["geometry_id"] == "G42" for r in rows),
                         "G42_hash_twin_in_valid_set": [r["geometry_id"] for r in rows if r["inventory_geometry_hash"] == G42_HASH]}
    rep["G42"] = {"mat": str(G42_MAT), "mat_sha256": sha(G42_MAT), "recomputed_hash": gh,
                  "hash_ok": gh == G42_HASH, "finite": gf}
    ok = (not unexplained and rep["teacher_sources"]["offdesign_matches_expected"] and rep["reference_matches_V1"]
          and rep["options_match_V1"] and rep["reference_matches_pilot"] and rep["options_match_pilot"]
          and rep["worker"]["identical_to_pilot"] and rep["G42"]["hash_ok"])
    rep["IDENTITY_OK"] = ok
    dump(OUT / "provenance/identity_check.json", rep)
    print(json.dumps({k: v for k, v in rep.items() if k != "active_m_files"}, indent=1)[:3000])


if __name__ == "__main__":
    main()

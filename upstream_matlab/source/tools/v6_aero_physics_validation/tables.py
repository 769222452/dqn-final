"""Print compact markdown tables from this round's outputs (used to write the reports)."""
from __future__ import annotations

import csv
import json
import sys

import common as C


def f(x, n=4):
    try:
        return f"{float(x):.{n}g}"
    except (TypeError, ValueError):
        return str(x)


def t_p0():
    m = json.loads((C.WORK / "p0_metrics.json").read_text(encoding="utf-8"))
    ks = ["V2_volume_full", "repro_volume_full", "repro_volume_full_origin_shift", "closed_volume_full",
          "closed_translation_invariance_abs_err", "closed_welded_watertight", "volume_full_xy_grid", "volume_full_veg1_strip",
          "V2_grid_thickness_max", "thickness_max_xy", "thickness_mean_xy", "V2_planform_area_full", "planform_full_xy_grid",
          "planform_full_veg1_le", "Sref_over_own_planform", "base_area_full_exact", "base_area_full_veg1", "base_x_spread",
          "V_over_Splan_V2", "V_over_Splan_closed", "volumetric_efficiency_closed", "max_abs_dx_same_index",
          "max_abs_dy_same_index", "leading_edge_max_gap", "V2_wetted_lower_half", "V2_wetted_upper_half",
          "negative_thickness_area_fraction", "body_lower_panels_with_upward_normal", "body_upper_panels_with_downward_normal"]
    print("| metric | " + " | ".join(C.SHORT[x["geometry_id"]] for x in m) + " |")
    print("|---|" + "---|" * len(m))
    for k in ks:
        print(f"| {k} | " + " | ".join(f(x.get(k)) for x in m) + " |")


def t_p2():
    rows = list(csv.DictReader(open(C.OUT / "LOW_ORDER_MODEL_COMPARISON.csv", encoding="utf-8-sig")))
    print("| case | model | alpha | CD | CD/Teacher | CD/G42 same model | detached |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        if r["kind"] == "MISSION_ISO_CL" and r["CL_target"] in ("0.045",):
            print(f"| {r['case_id']} | {r['model'][:1]} | {f(r['alpha_deg'])} | {f(r['CD'])} | {f(r['CD_over_teacher'])} | "
                  f"{f(r['CD_over_G42_same_model'])} | {f(r['detached_area_fraction'], 2)} |")
    rk = list(csv.DictReader(open(C.WORK / "p2_ranking.csv", encoding="utf-8-sig")))
    for r in rk:
        print({k: v for k, v in r.items() if v})


def t_p3():
    rk = list(csv.DictReader(open(C.WORK / "p3_scenario_ranking.csv", encoding="utf-8-sig")))
    for r in rk:
        print(r["inviscid_model"][:1], r["friction"], r["base"], {k[17:]: f(v) for k, v in r.items() if k.startswith("mean_")}, r["best"])
    rows = list(csv.DictReader(open(C.OUT / "DRAG_COMPONENT_SENSITIVITY.csv", encoding="utf-8-sig")))
    for r in rows:
        if r["CL_target"] == "0.045" and r["inviscid_model"].startswith("A") and r["base"] in ("BASE_0", "BASE_M2"):
            print(r["case_id"], r["friction"], r["base"], f(r["CD_inviscid"]), f(r["CD_friction"]), f(r["CD_base"]),
                  f(r["friction_share"], 3), f(r["base_share"], 3), f(r["LD_total"]), f(r["Re_unit_inf_per_m"], 3),
                  f(r["turbulent_area_frac"], 2))


if __name__ == "__main__":
    {"p0": t_p0, "p2": t_p2, "p3": t_p3}[sys.argv[1]]()

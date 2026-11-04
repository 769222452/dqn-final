"""Condense DRAG_COMPONENT_SENSITIVITY.csv into work/p3_summary.json for the V3 report."""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np

import common as C


def rng(v):
    return [float(np.min(v)), float(np.max(v))] if v else None


def main():
    rows = list(csv.DictReader(open(C.OUT / "DRAG_COMPONENT_SENSITIVITY.csv", encoding="utf-8-sig")))
    rank = list(csv.DictReader(open(C.WORK / "p3_scenario_ranking.csv", encoding="utf-8-sig")))
    out = {"n_rows": len(rows), "share": {}, "LD_total": {}, "Re": {}, "turb_frac": {}, "rank": {}, "cd_over_g42_spread": {}}
    share = defaultdict(list)
    ld = defaultdict(list)
    for r in rows:
        g = C.SHORT[r["geometry_id"]]
        m = r["inviscid_model"][:1]
        share[(m, "friction", r["friction"], r["base"])].append(float(r["friction_share"]))
        share[(m, "base", g, r["base"])].append(float(r["base_share"]))
        share[(m, "friction_geom", g, r["friction"])].append(float(r["friction_share"]))
        ld[(m, g, r["friction"], r["base"])].append(float(r["LD_total"]))
        out["Re"].setdefault(r["Mach"], float(r["Re_unit_inf_per_m"]))
        if r["turbulent_area_frac"]:
            out["turb_frac"].setdefault(f"{r['friction']}_M{r['Mach']}", []).append(float(r["turbulent_area_frac"]))
    out["share"] = {"|".join(k): rng(v) for k, v in share.items()}
    out["LD_total"] = {"|".join(k): rng(v) for k, v in ld.items()}
    out["turb_frac"] = {k: rng(v) for k, v in out["turb_frac"].items()}
    for m in ("A", "C"):
        sub = [r for r in rank if r["inviscid_model"].startswith(m)]
        best = defaultdict(int)
        for r in sub:
            best[r["best"]] += 1
        out["rank"][m] = {"n_scenarios": len(sub), "best_count": dict(best)}
        for g in ("MD9.873", "MD8_020", "MD8_009"):
            v = [float(r[f"mean_CD_over_G42_{g}"]) for r in sub]
            out["cd_over_g42_spread"][f"{m}|{g}"] = rng(v)
    # how much the MD9.873-vs-MD8_009 gap moves (Teacher inviscid) across scenarios
    a = [r for r in rank if r["inviscid_model"].startswith("A")]
    out["gap_MD8_009_minus_MD9.873_A"] = rng([float(r["mean_CD_over_G42_MD8_009"]) - float(r["mean_CD_over_G42_MD9.873"]) for r in a])
    (C.WORK / "p3_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k in ("rank", "cd_over_g42_spread", "gap_MD8_009_minus_MD9.873_A", "Re", "turb_frac"):
        print(k, json.dumps(out[k]))
    for k, v in out["share"].items():
        if k.startswith("A|friction|") or k.startswith("A|base|"):
            print(k, [round(x, 3) for x in v])


if __name__ == "__main__":
    main()

"""Endpoint selection (P2 -> P3), morph path construction (P3) and the 1200 s pilot (P4).

  python pilot.py select            -> endpoints.json
  python pilot.py paths             -> morph/<path>.json, morph_paths.json
  python pilot.py pilot --budget S  -> pilot/<job>.json, fixed_vs_morph_pilot.json
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from common import BANDS, OUT, PARAMS, SCENARIOS, dump, load, read_csv, utc

BAND57 = ("M6-7", "M5-6")


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def band_denominators():
    """G42 drag-work reference per band (exact weights used by score.scores)."""
    from aerotab import read_v2_labels
    from common import SCREEN_MACH
    from score import evaluate, scored_conditions, screen_table
    conds = scored_conditions()
    lab, _ = read_v2_labels(["G42"], machs=SCREEN_MACH)
    ref = evaluate(screen_table("G42", lab["G42"]), conds)
    den = {b: 0.0 for b in BANDS}
    for c in conds:
        den[c["band"]] += c["w"] * ref[c["id"]][1]
    return den


def candidate_pool():
    den = band_denominators()
    rows = []
    for r in read_csv(OUT / "v2_mission_scores.csv"):
        rows.append({**r, "id": r["geometry_id"], "evidence": "V2_TEACHER", "round": "V2"})
    for p in sorted((OUT / "candidates").glob("*_scored.csv")):
        for r in read_csv(p):
            if r.get("classification") == "VALID" and np.isfinite(fnum(r.get("J_ALL"))):
                rows.append({**r, "id": r["candidate_id"], "evidence": "NEW_TEACHER"})
    out = []
    for r in rows:
        j = {k: fnum(r.get(f"J_{k}")) for k in ["ALL", *BANDS]}
        if not np.isfinite(j["ALL"]):
            continue
        j["M5-7"] = (j["M6-7"] * den["M6-7"] + j["M5-6"] * den["M5-6"]) / (den["M6-7"] + den["M5-6"])
        out.append({"id": r["id"], "round": r.get("round"), "evidence": r["evidence"],
                    "source": r.get("source") or r.get("origin"), "Ma_design": float(r["Ma_design"]),
                    "z": [float(r[p]) for p in PARAMS], "geometry_hash": r["geometry_hash"],
                    "volume_ratio_G42": fnum(r["volume_ratio_G42"]),
                    "thickness_ratio_G42": fnum(r["thickness_ratio_G42"]),
                    "scenario": {s: str(r.get(f"scenario_{s}")) == "True" for s in SCENARIOS},
                    "J": j})
    return out, den


def select():
    pool, den = candidate_pool()
    res = {"utc": utc(), "band_reference_drag_work": den, "n_pool": len(pool),
           "n_new": sum(p["evidence"] == "NEW_TEACHER" for p in pool), "scenarios": {}}
    for scen in ("EXPLORATORY", "CONSERVATIVE", "UNCONSTRAINED"):
        sel = [p for p in pool if scen == "UNCONSTRAINED" or p["scenario"][scen]]
        best = {}
        for key in ["ALL", *BANDS, "M5-7"]:
            ranked = sorted(sel, key=lambda p: p["J"][key])
            best[key] = [{"id": p["id"], "evidence": p["evidence"], "Ma_design": p["Ma_design"],
                          "J": p["J"][key], "J_ALL": p["J"]["ALL"], "volume_ratio_G42": p["volume_ratio_G42"],
                          "thickness_ratio_G42": p["thickness_ratio_G42"]} for p in ranked[:5]]
        A = min(sel, key=lambda p: p["J"]["M7-8"])
        C = min(sel, key=lambda p: p["J"]["M5-7"])
        S = min(sel, key=lambda p: p["J"]["ALL"])
        ideal = (A["J"]["M7-8"] * den["M7-8"] + C["J"]["M5-7"] * (den["M6-7"] + den["M5-6"])) / sum(den.values())
        res["scenarios"][scen] = {
            "ranking": best,
            "A_high": _end(A), "C_low_mid": _end(C), "S_best_single": _end(S),
            "complementary": A["id"] != C["id"],
            "static_ideal_two_shape_J_ALL": ideal,
            "best_single_J_ALL": S["J"]["ALL"],
            "static_two_shape_gain_vs_best_single": 1 - ideal / S["J"]["ALL"],
            "note": "static iso-CL drag-work on the stage-1 G42 operating conditions; not a flight result"}
    dump(OUT / "endpoints.json", res)
    return res


def _end(p):
    return {"id": p["id"], "evidence": p["evidence"], "Ma_design": p["Ma_design"], "z": p["z"],
            "geometry_hash": p["geometry_hash"], "J": p["J"], "volume_ratio_G42": p["volume_ratio_G42"],
            "thickness_ratio_G42": p["thickness_ratio_G42"]}


def g42_end():
    from common import G42_INPUTS10
    return {"id": "G42", "Ma_design": 15.0, "z": G42_INPUTS10[1:5] + [G42_INPUTS10[7], G42_INPUTS10[9]]}


def probe_lines():
    """candidate_id -> (far-end V2 id, t) for native points on G42 -> X lines (round-1 probes)."""
    out = {}
    for p in sorted((OUT / "candidates").glob("round*.json")):
        if p.stem.endswith("timing"):
            continue
        for c in load(p):
            if c.get("source") == "cross_family_line_probe":
                out[c["candidate_id"]] = (c["seed"].split("->")[1], float(c["t"]))
    return out


def v2_end(gid):
    r = next(r for r in read_csv(OUT / "v2_mission_scores.csv") if r["geometry_id"] == gid)
    return {"id": gid, "Ma_design": float(r["Ma_design"]), "z": [float(r[p]) for p in PARAMS]}


def path_plan(ep):
    """Pairs needed by the pilot. If both ends of a pair lie on the same native G42 -> X probe line,
    the whole line is built once (sub-paths are then extracted from its nodes)."""
    lines = probe_lines()
    plan = {}

    def need(P, Q, use):
        on = []
        for e in (P, Q):
            if e["id"] == "G42":
                on.append(None)
            elif e["id"] in lines:
                on.append(lines[e["id"]][0])
            else:
                on.append(False)
        targets = {x for x in on if x}
        if False not in on and len(targets) <= 1:
            X = targets.pop() if targets else None
            if X is None:
                return
            key, a, c = f"G42__{X}", g42_end(), v2_end(X)
        else:
            key, a, c = f"{P['id']}__{Q['id']}", P, Q
        plan.setdefault(key, (a, c, []))[2].append(use)

    for scen in ("EXPLORATORY", "CONSERVATIVE", "UNCONSTRAINED"):
        s = ep["scenarios"][scen]
        A, C = s["A_high"], s["C_low_mid"]
        if A["id"] != C["id"]:
            need(A, C, scen)
        if A["id"] != "G42":
            need(g42_end(), A, scen + ":g42_start")
    return plan


def subpath(hA, hC):
    """Nodes between two geometry hashes on any built path, s rescaled to [0, 1] from A to C."""
    for p in sorted((OUT / "morph").glob("*.json")):
        rec = load(p)
        idx = {n["geometry_hash"]: n["s"] for n in rec["nodes"] if n.get("geometry_hash")}
        if hA not in idx or hC not in idx:
            continue
        sA, sC = idx[hA], idx[hC]
        lo, hi = min(sA, sC), max(sA, sC)
        sel = [n for n in rec["nodes"] if lo - 1e-12 <= n["s"] <= hi + 1e-12]
        nodes = sorted(((n["s"] - sA) / (sC - sA), n["geometry_hash"], n["candidate_id"]) for n in sel)
        ok = all(n["classification"] == "VALID" and n["teacher_mission_grid_complete"] for n in sel)
        bad = [c for c in rec["blend_checks"] if lo <= c["mid"] <= hi and c.get("status") not in ("PASS", "REFINE")]
        unresolved = [u for u in rec["unresolved_segments"] if lo <= u[0] and u[1] <= hi]
        st = ("VALIDATED" if ok and not bad and not unresolved else
              "VALID_GEOMETRY_AERO_RESIDUAL_ABOVE_TOL" if ok and not bad else "NOT_VALIDATED")
        return nodes, st, {"parent_path": rec["path"], "s_range_parent": [sA, sC], "n_nodes": len(sel)}
    return None, "NO_PATH", {}


def paths(log=print, deadline=None):
    import morph
    from aerotab import read_v2_labels
    ep = load(OUT / "endpoints.json")
    _, meta = read_v2_labels(["G42"])
    ref = meta["G42"]
    plan = path_plan(ep)
    recs = {}
    for name, (A, C, uses) in plan.items():
        if deadline and time.time() > deadline:
            log(f"[paths] budget reached before {name}")
            break
        p = OUT / "morph" / f"{name}.json"
        if p.exists():
            recs[name] = load(p)
            continue
        log(f"[paths] building {name} for {uses}")
        rec = morph.build_path(name, {"id": A["id"], "Ma_design": A["Ma_design"], "z": A["z"]},
                               {"id": C["id"], "Ma_design": C["Ma_design"], "z": C["z"]},
                               log=log, deadline=deadline, scenario_limits=SCENARIOS, ref=ref)
        rec["used_by"] = uses
        dump(p, rec)
        recs[name] = rec
        log(f"[paths] {name}: {rec['status']} nodes={rec['n_nodes']} "
            f"res CL {rec['worst_final_cl_rel_residual']} CD {rec['worst_final_cd_rel_residual']}")
    # make sure every pilot geometry not on a path has MISSION-grid labels
    import geometry
    extra = []
    for scen in ("EXPLORATORY", "CONSERVATIVE", "UNCONSTRAINED"):
        for role in ("A_high", "C_low_mid", "S_best_single"):
            S = ep["scenarios"][scen][role]
            c = geometry.make_candidate(S["Ma_design"], S["z"], "pilot_fixed_geometry")
            if c["candidate_id"] not in {x["candidate_id"] for x in extra}:
                extra.append(c)
    geometry.generate(extra, "pilot_fixed", sessions=min(3, len(extra)), log=log)
    morph.teacher_mission(geometry.audit(extra, workers=2, log=log), log=log, deadline=deadline)
    summary = {"utc": utc(), "paths": {k: {kk: v.get(kk) for kk in (
        "status", "crosses_design_mach", "n_nodes", "worst_final_cl_rel_residual", "worst_final_cd_rel_residual",
        "path_length_sum_max_displacement_m", "path_length_sum_rms_displacement_m", "actuator",
        "scenario_satisfied_all_nodes", "used_by", "endpoint_shape_distance")} for k, v in recs.items()},
        "plan": {k: [a["id"], c["id"], u] for k, (a, c, u) in plan.items()}}
    dump(OUT / "morph_paths_summary.json", summary)
    return recs


# ----------------------------------------------------------------------------- P4 pilot
def _job(spec):
    import mission_sim as ms
    import morph
    import teacher_pool
    committed = teacher_pool.committed()
    nodes = []
    for s, h, name in spec["nodes"]:
        nodes.append((s, ms.mission_table(name, morph.mission_labels(h, committed))))
    aero = ms.PathAero(nodes, mode="step" if spec["kind"] == "step" else "blend")
    m = ms.mission()
    deadline = time.monotonic() + spec["budget_s"]
    tg = spec.get("t_switch_grid") or [None]
    base = ms.g42_stage1_alpha()
    cache = {}

    def clm(ts):
        if ts not in cache:
            cache[ts] = ms.cl_matched_alpha(m, aero, ms.schedule_fn(spec["kind"], ts or 0.0), scales=(1.0, 1.03))
        return cache[ts]
    # Same seed family for every geometry/strategy: the stage-1 G42 alpha history, and alpha histories
    # that reproduce the stage-1 G42 CL history (x1.00 / x1.03) on this geometry or morph schedule.
    seeds = [("G42_stage1_alpha", base), ("CL_match_G42_x1.00", lambda ts: clm(ts)[0]),
             ("CL_match_G42_x1.03", lambda ts: clm(ts)[1])]
    pdir = "pilot" if spec.get("pass", "A") == "A" else "pilot_B"
    if pdir == "pilot_B":
        # Pass B (same rule for every job): cross-seed with every pass-A best alpha history,
        # at this job's own pass-A switch time (or the full grid if pass A found nothing).
        seeds = [(f"passA:{p.parent.name}", np.load(p)) for p in sorted((OUT / "pilot").glob("*/alpha_0p5s.npy"))]
        own = OUT / "pilot" / f"{spec['job']}.json"
        if own.exists() and load(own).get("t_switch_s") is not None:
            tg = [load(own)["t_switch_s"]]
    res = ms.optimise(m, aero, spec["kind"], deadline, t_switch_grid=tg, knots=(6, 10), maxiter=6, seeds=seeds,
                      log=lambda s: None)
    ctrl = res.pop("_controls", None)
    if ctrl is not None:
        f = ms.simulate(m, aero, ctrl, ms.schedule_fn(spec["kind"], res["t_switch_s"] or 0.0), collect=True)
        d = OUT / pdir / spec["job"]
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "alpha_0p5s.npy", ctrl)
        rows = f["rows"]
        res["trajectory_summary"] = _traj_summary(rows)
        from common import write_csv
        write_csv(d / "trajectory_0p5s.csv", rows)
    res.update({"job": spec["job"], "spec": {k: v for k, v in spec.items() if k != "nodes"},
                "nodes": [{"s": s, "geometry_hash": h, "name": n} for s, h, n in spec["nodes"]]})
    dump(OUT / pdir / f"{spec['job']}.json", res)
    return res


def _traj_summary(rows):
    if not rows:
        return {}
    M = np.array([r["Mach"] for r in rows])
    s = np.array([r["path_s"] for r in rows])
    out = {"mach_start": float(M[0]), "mach_end": float(M[-1]), "mach_min": float(M.min()),
           "mach_max": float(M.max()), "q_max_pa": float(max(r["q_pa"] for r in rows)),
           "alpha_deg_range": [float(np.degrees(min(r["alpha_rad"] for r in rows))),
                               float(np.degrees(max(r["alpha_rad"] for r in rows)))]}
    if s.max() > s.min():
        moving = np.nonzero(np.diff(s) != 0)[0]
        out["morph_active_window_s"] = [float(rows[moving[0]]["time_s"]), float(rows[moving[-1] + 1]["time_s"])]
        out["mach_at_morph_start"] = float(M[moving[0]])
    return out


def pilot_specs(budget):
    ep = load(OUT / "endpoints.json")
    import morph
    import teacher_pool
    committed = teacher_pool.committed()
    specs = {}
    grid = [100.0, 200.0, 300.0, 400.0, 500.0, 650.0, 800.0]

    ok_status = ("VALIDATED", "VALID_GEOMETRY_AERO_RESIDUAL_ABOVE_TOL")

    def fixed(gid, h):
        if not morph.mission_complete(h, committed):
            return None
        return {"job": f"fixed__{gid}", "kind": "fixed", "nodes": [(0.0, h, gid)], "budget_s": budget}

    from aerotab import read_v2_labels
    _, meta = read_v2_labels(["G42"])
    g42h = meta["G42"]["geometry_hash"]
    specs["fixed__G42"] = fixed("G42", g42h)
    skipped = {}
    uses = {}
    for scen in ("EXPLORATORY", "CONSERVATIVE", "UNCONSTRAINED"):
        s = ep["scenarios"][scen]
        A, C, S = s["A_high"], s["C_low_mid"], s["S_best_single"]
        for e in (A, C, S):
            sp = fixed(e["id"], e["geometry_hash"])
            if sp:
                specs.setdefault(sp["job"], sp)
                uses.setdefault(sp["job"], []).append(scen)
            else:
                skipped[f"fixed__{e['id']}"] = "MISSION-grid Teacher labels incomplete"
        n0, st0, info0 = subpath(g42h, A["geometry_hash"]) if A["id"] != "G42" else (None, "SAME", {})
        if A["id"] == C["id"]:
            if n0 is not None and st0 in ok_status:
                job = f"g42start__{A['id']}"
                specs.setdefault(job, {"job": job, "kind": "morph", "nodes": n0, "t_switch_grid": [0.0],
                                       "budget_s": budget, "path_info": info0, "path_status": st0})
                uses.setdefault(job, []).append(scen + ":g42_start")
            continue
        nodes, st, info = subpath(A["geometry_hash"], C["geometry_hash"])
        if nodes is None or st not in ok_status:
            skipped[f"{scen}:morph {A['id']}->{C['id']}"] = f"path status {st}"
            continue
        name = f"{A['id']}__{C['id']}"
        for job, spec in ((f"step__{name}", {"kind": "step", "nodes": [nodes[0], nodes[-1]]}),
                          (f"morph__{name}", {"kind": "morph", "nodes": nodes})):
            specs.setdefault(job, {"job": job, **spec, "t_switch_grid": grid, "budget_s": budget,
                                   "path_info": info, "path_status": st})
            uses.setdefault(job, []).append(scen)
        if A["id"] != "G42":
            if n0 is not None and st0 in ok_status:
                job = f"morphg42__{name}"
                specs.setdefault(job, {"job": job, "kind": "morph_g42",
                                       "nodes": n0 + [(1.0 + s_, h, n) for s_, h, n in nodes[1:]],
                                       "t_switch_grid": grid, "budget_s": budget,
                                       "path_info": [info0, info], "path_status": [st0, st]})
                uses.setdefault(job, []).append(scen + ":g42_start")
            else:
                skipped[f"morphg42__{name}"] = f"G42->A path status {st0}"
    for k, v in specs.items():
        if v:
            v["used_by"] = uses.get(k, ["reference"])
    return {k: v for k, v in specs.items() if v}, skipped


def pilot(budget, workers, log=print, pass_="A", skip=()):
    specs, skipped = pilot_specs(budget)
    specs = {k: v for k, v in specs.items() if k not in skip}
    for s in specs.values():
        s["pass"] = pass_
    pdir = OUT / ("pilot" if pass_ == "A" else "pilot_B")
    log(f"[pilot {pass_}] {len(specs)} jobs: {list(specs)}; skipped {skipped}")
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_job, s): k for k, s in specs.items() if not (pdir / f"{k}.json").exists()}
        for k in specs:
            if (pdir / f"{k}.json").exists():
                results[k] = load(pdir / f"{k}.json")
        for f in as_completed(futs):
            k = futs[f]
            try:
                results[k] = f.result()
                log(f"[pilot] {k}: {results[k]['status']} {results[k].get('range_m', float('nan')) / 1e3:.2f} km")
            except Exception as e:  # noqa: BLE001
                results[k] = {"status": "JOB_ERROR", "error": repr(e)}
                log(f"[pilot] {k}: ERROR {e!r}")
    name = "pilot_results_raw.json" if pass_ == "A" else "pilot_results_raw_B.json"
    dump(OUT / name, {"utc": utc(), "pass": pass_, "budget_s_per_job": budget, "results": results,
                      "skipped": skipped})
    return results, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "paths", "pilot"])
    ap.add_argument("--budget", type=float, default=600)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--deadline-min", type=float, default=None)
    ap.add_argument("--pass", dest="pass_", choices=["A", "B"], default="A")
    ap.add_argument("--skip", default="", help="comma-separated job names already running elsewhere")
    a = ap.parse_args()
    log = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)
    if a.cmd == "select":
        print(json.dumps({k: {kk: (v[kk]["id"] if isinstance(v[kk], dict) and "id" in v[kk] else v[kk])
                              for kk in ("A_high", "C_low_mid", "S_best_single", "complementary",
                                         "static_ideal_two_shape_J_ALL", "best_single_J_ALL",
                                         "static_two_shape_gain_vs_best_single")}
                          for k, v in select()["scenarios"].items()}, indent=1))
    elif a.cmd == "paths":
        dl = time.time() + 60 * a.deadline_min if a.deadline_min else None
        paths(log, dl)
    else:
        pilot(a.budget, a.workers, log, a.pass_, tuple(x for x in a.skip.split(",") if x))


if __name__ == "__main__":
    main()

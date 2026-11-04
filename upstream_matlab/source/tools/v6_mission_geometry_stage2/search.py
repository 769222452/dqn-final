"""P2: mission-driven new geometry search (native MATLAB + frozen audit + Teacher screening).

Rounds:
  round1  local perturbations around V2 role winners, design-Mach continuation of good
          shapes, and native points on the (Ma_design, z) line between the high-Mach
          winner (G42 family) and the low/mid-Mach winners (MD8 family).
  roundN  surrogate-guided proposals (GP on Teacher-scored designs; predictions only
          choose where to generate, never count as evidence) + local refinement.
All candidates are stored with their round in candidates/<round>.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import geometry
import teacher_pool
from aerotab import labels_from_ledger, read_v2_labels
from common import (BANDS, EXT_UPPER, FROZEN_LOWER, OUT, PARAMS, SCENARIOS, SCREEN_ALPHA, SCREEN_MACH, dump, load,
                    normalise, read_csv, write_csv)
from score import evaluate, scored_conditions, scores, screen_table

RNG = np.random.default_rng(20261010)
MD_RANGE = (6.0, 15.0)


def v2_rows():
    return {r["geometry_id"]: r for r in read_csv(OUT / "v2_mission_scores.csv")}


def z_of(r):
    return [float(r[p]) for p in PARAMS]


def clip_z(z):
    return np.clip(np.asarray(z, float), FROZEN_LOWER, EXT_UPPER)


def perturb(z, sigma, n):
    zn = normalise(z)
    out = []
    for _ in range(n):
        p = np.clip(zn + RNG.normal(0, sigma, 6), 0, 1)
        out.append(FROZEN_LOWER + p * (EXT_UPPER - FROZEN_LOWER))
    return out


def round1():
    v2 = v2_rows()
    an = load(OUT / "v2_mission_analysis.json")
    cands = []
    seeds = set()
    for scen in ("UNCONSTRAINED", "EXPLORATORY", "CONSERVATIVE"):
        for band in ["ALL"] + list(BANDS):
            for w in an["winners"][scen][band][:2]:
                seeds.add(w["geometry_id"])
    for fam in an["best_per_design_mach_family"].values():
        for v in fam.values():
            if v and v["J"] < 1.1:
                seeds.add(v["geometry_id"])
    seeds.update(["V2_MD15_007", "V2_MD8_020"])
    seeds.discard("V2_MD15_001")
    for gid in sorted(seeds):
        r = v2[gid]
        for z in perturb(z_of(r), 0.035, 4):
            cands.append(geometry.make_candidate(float(r["Ma_design"]), z, "local_perturbation", seed=gid))
    # design-Mach continuation of the strongest shapes
    for gid, mds in (("G42", (8.0, 10.0, 12.0)), ("V2_MD15_007", (10.0, 12.0)),
                     ("V2M_MD8_051", (9.0, 10.0, 12.0, 15.0)), ("V2_MD8_020", (9.0, 10.0, 12.0, 15.0)),
                     ("V2_MD8_009", (10.0, 12.0)), ("V2M_MD8_006", (10.0, 12.0))):
        for md in mds:
            cands.append(geometry.make_candidate(md, z_of(v2[gid]), "design_mach_continuation", seed=gid))
    # native points on the joint (Ma_design, z) line G42 -> MD8 winners
    g = v2["G42"]
    for gid in ("V2M_MD8_051", "V2_MD8_020", "V2_MD8_009"):
        r = v2[gid]
        for t in (0.25, 0.5, 0.75):
            md = (1 - t) * 15.0 + t * float(r["Ma_design"])
            z = (1 - t) * np.array(z_of(g)) + t * np.array(z_of(r))
            cands.append(geometry.make_candidate(md, z, "cross_family_line_probe", seed=f"G42->{gid}", t=t))
    return dedupe(cands)


def dedupe(cands):
    seen, out = set(), []
    for c in cands:
        if c["candidate_id"] not in seen and c["design_space"] != "OUT_OF_RANGE":
            seen.add(c["candidate_id"])
            out.append(c)
    return out


def teacher_tasks(recs, machs=SCREEN_MACH, alphas=SCREEN_ALPHA):
    tasks = []
    for r in recs:
        if r.get("classification") != "VALID":
            continue
        for m in machs:
            tasks.append({"geometry_id": r["candidate_id"], "geometry_hash": r["geometry_hash"],
                          "geometry_mat": r["native_geometry_mat"].replace("\\", "/"), "M_flight": float(m),
                          "alpha_deg": [float(a) for a in alphas]})
    return tasks


def score_new(recs):
    conds = scored_conditions()
    v2lab, meta = read_v2_labels(["G42"], machs=SCREEN_MACH)
    ref = evaluate(screen_table("G42", v2lab["G42"]), conds)
    gv, gt = meta["G42"]["volume_full"], meta["G42"]["grid_thickness_max"]
    committed = teacher_pool.committed()
    out = []
    for r in recs:
        r = dict(r)
        if r.get("classification") == "VALID" and r.get("volume_full") is not None:
            r["volume_ratio_G42"] = r["volume_full"] / gv
            r["thickness_ratio_G42"] = r["grid_thickness_max"] / gt
            for s, d in SCENARIOS.items():
                r[f"scenario_{s}"] = (r["volume_ratio_G42"] >= d["volume_ratio_min"] - 1e-12 and
                                      r["thickness_ratio_G42"] >= d["thickness_ratio_min"] - 1e-12)
            lab = labels_from_ledger(committed, r["geometry_hash"])
            need = [(float(m), float(a)) for m in SCREEN_MACH for a in SCREEN_ALPHA]
            fails = [k for k in need if k not in lab]
            r["teacher_labels_pass"] = len(need) - len(fails)
            r["teacher_labels_missing_or_failed"] = len(fails)
            if not fails:
                per = evaluate(screen_table(r["candidate_id"], lab), conds)
                for k, v in scores(per, ref, conds).items():
                    r[("J_" + k) if not k.endswith("_reachable") else ("reachable_" + k)] = v
                r["aero_evidence"] = "NEW_TEACHER (real labels; iso-CL via PCHIP)"
            else:
                r["aero_evidence"] = "INCOMPLETE_TEACHER_LABELS"
        out.append(r)
    return out


def run_round(name, cands, sessions=3, workers=3, log=print):
    dump(OUT / "candidates" / f"{name}.json", cands)
    t0 = time.time()
    geometry.generate(cands, name, sessions=sessions, log=log)
    t1 = time.time()
    recs = geometry.audit(cands, workers=6, log=log)
    t2 = time.time()
    stats = teacher_pool.run(teacher_tasks(recs), n_workers=workers, log=log)
    t3 = time.time()
    scored = score_new(recs)
    for r in scored:
        r["round"] = name
    write_csv(OUT / "candidates" / f"{name}_scored.csv", scored)
    timing = {"round": name, "candidates": len(cands), "generation_s": t1 - t0, "audit_s": t2 - t1,
              "teacher_s": t3 - t2, "teacher": stats,
              "generated": sum(r["generation_status"] == "GENERATED" for r in recs),
              "valid": sum(r.get("classification") == "VALID" for r in recs)}
    dump(OUT / "candidates" / f"{name}_timing.json", timing)
    log(json.dumps(timing))
    return scored


def all_scored():
    rows = []
    for p in sorted((OUT / "candidates").glob("*_scored.csv")):
        rows += read_csv(p)
    return rows


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def gp_fit(X, y, ls=0.15, noise=1e-4):
    mu = y.mean()
    K = np.exp(-0.5 * ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1) / ls ** 2) + noise * np.eye(len(X))
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y - mu))
    return X, L, alpha, mu, ls


def gp_predict(model, Xs):
    X, L, alpha, mu, ls = model
    k = np.exp(-0.5 * ((Xs[:, None, :] - X[None, :, :]) ** 2).sum(-1) / ls ** 2)
    m = mu + k @ alpha
    v = np.linalg.solve(L, k.T)
    s = np.sqrt(np.maximum(1e-12, 1 - (v ** 2).sum(0)))
    return m, s


def design_vec(md, z):
    return np.r_[(float(md) - MD_RANGE[0]) / (MD_RANGE[1] - MD_RANGE[0]), normalise(z)]


def from_vec(x):
    md = MD_RANGE[0] + x[0] * (MD_RANGE[1] - MD_RANGE[0])
    return round(float(md), 3), FROZEN_LOWER + np.clip(x[1:], 0, 1) * (EXT_UPPER - FROZEN_LOWER)


def propose(name, per_target=3, n_local=2):
    """Surrogate-guided proposals per (band, scenario) using all Teacher-scored designs."""
    v2 = list(v2_rows().values())
    new = all_scored()
    pts = []
    for r in v2:
        md = float(r["Ma_design"])
        if MD_RANGE[0] <= md <= MD_RANGE[1]:
            pts.append((md, z_of(r), r, "V2"))
    for r in new:
        if r.get("classification") == "VALID" and np.isfinite(fnum(r.get("J_ALL"))):
            pts.append((float(r["Ma_design"]), z_of(r), r, "NEW"))
    # validity classifier data: every generation attempt (V2 failures + stage 2)
    tried = [(float(r["Ma_design"]), z_of(r), r.get("classification") == "VALID") for r in new]
    for r in read_csv(OUT.parent / "v5_multimach_geometry_db_v2/failed_candidates.csv"):
        tried.append((float(r["Ma_design"]), z_of(r), False))
    for r in v2:
        tried.append((float(r["Ma_design"]), z_of(r), True))
    TX = np.array([design_vec(md, z) for md, z, _ in tried if MD_RANGE[0] <= md <= MD_RANGE[1]])
    Ty = np.array([v for md, z, v in tried if MD_RANGE[0] <= md <= MD_RANGE[1]], float)
    cands = []
    for scen in ("EXPLORATORY", "CONSERVATIVE"):
        sel = [p for p in pts if str(p[2].get(f"scenario_{scen}")) == "True"]
        for key in ["ALL"] + list(BANDS):
            data = [(design_vec(md, z), fnum(r.get(f"J_{key}"))) for md, z, r, _ in sel]
            data = [(x, y) for x, y in data if np.isfinite(y)]
            if len(data) < 5:
                continue
            X = np.array([d[0] for d in data])
            y = np.array([d[1] for d in data])
            model = gp_fit(X, y)
            order = np.argsort(y)[:3]
            samples = []
            for i in order:
                samples.append(np.clip(X[i] + RNG.normal(0, 0.06, (1500, 7)), 0, 1))
            S = np.concatenate(samples)
            m, s = gp_predict(model, S)
            # validity likelihood from nearest tried designs (kNN, k=7)
            d = ((S[:, None, :] - TX[None, :, :]) ** 2).sum(-1)
            nn = np.argsort(d, axis=1)[:, :7]
            pvalid = Ty[nn].mean(1)
            lcb = m - 1.0 * s
            ok = pvalid >= 0.5
            idx = np.argsort(np.where(ok, lcb, np.inf))
            chosen = []
            for i in idx:
                if not ok[i] or len(chosen) >= per_target:
                    break
                if all(np.linalg.norm(S[i] - S[j]) > 0.05 for j in chosen):
                    chosen.append(i)
            for i in chosen:
                md, z = from_vec(S[i])
                cands.append(geometry.make_candidate(md, z, "surrogate_proposal", target=f"{scen}:{key}",
                                                     predicted_J=float(m[i]), predicted_sd=float(s[i]),
                                                     validity_knn=float(pvalid[i]),
                                                     evidence_status="SURROGATE_PREDICTION_ONLY"))
            # local refinement around current best real design for this target
            best_x = X[order[0]]
            for _ in range(n_local):
                md, z = from_vec(np.clip(best_x + RNG.normal(0, 0.03, 7) * np.r_[0.3, np.ones(6)], 0, 1))
                cands.append(geometry.make_candidate(md, z, "local_refinement", target=f"{scen}:{key}"))
    return dedupe(cands)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["regression", "round1", "propose", "rescore"])
    ap.add_argument("--name", default="round2")
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    log = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)
    if args.command == "regression":
        print(json.dumps(geometry.g42_regression(log), indent=1, default=str))
    elif args.command == "round1":
        run_round("round1", round1(), args.sessions, args.workers, log)
    elif args.command == "propose":
        run_round(args.name, propose(args.name), args.sessions, args.workers, log)
    elif args.command == "rescore":
        for p in sorted((OUT / "candidates").glob("round*.json")):
            if p.stem.endswith("timing"):
                continue
            cands = load(p)
            recs = [geometry.summary(c) for c in cands]
            scored = score_new(recs)
            for r in scored:
                r["round"] = p.stem
            write_csv(OUT / "candidates" / f"{p.stem}_scored.csv", scored)


if __name__ == "__main__":
    main()

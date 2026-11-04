"""Stage 4: fixed selected geometries -> nearby, common cross-Mach and dense-alpha (equal-lift) wedge-model evaluations (real calls, one MATLAB process)."""
import time
from common import OUT, cfg, read_csv, write_csv, run_matlab_eval

c = cfg(); sel = read_csv(OUT / "screening/selected_unique_geometries.csv")
Mx, Ax = c["cross_mach"]["M_grid"], c["cross_mach"]["alpha_deg_grid"]
Md, Ad = c["equal_lift"]["M_grid"], c["equal_lift"]["alpha_dense_deg"]
nb = c["nearby"]
pts = {}
def add(g, M, a): pts[(g["geometry_id"], round(M, 6), round(a, 6))] = {"geometry_id": g["geometry_id"], "mat_path": g["mat_path"], "Mach": round(M, 6), "alpha_deg": round(a, 6)}
for g in sel:
    for M in Mx:
        for a in Ax: add(g, M, a)
    for M in Md:
        for a in Ad: add(g, M, a)
    for cid in g["conditions"].split(";"):                     # nearby points explicitly requested, even if already in the grids above
        M = float(cid[1:5]); a = float(cid.split("_A")[1])
        for dM, da in ((0, 0), (-nb["dM"], 0), (nb["dM"], 0), (0, -nb["dalpha_deg"]), (0, nb["dalpha_deg"])): add(g, M + dM, a + da)
q = list(pts.values()); write_csv(OUT / "work/selected_queries.csv", q)
print("geometries", len(sel), "unique queries", len(q), flush=True); t = time.time()
(OUT / "logs/selected_eval_matlab.log").unlink(missing_ok=True)
run_matlab_eval(OUT / "work/selected_queries.csv", OUT / "work/selected_results.csv", OUT / "logs/selected_eval_matlab.log")
print("done wall_s", round(time.time() - t, 1), flush=True)

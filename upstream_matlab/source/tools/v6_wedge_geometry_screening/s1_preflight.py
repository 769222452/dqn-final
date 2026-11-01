"""Stage 1 preflight: 3 geometries x 3 conditions through the existing wedge model, measure time."""
import time
from common import OUT, read_csv, write_csv, run_matlab_eval

pool = {r["geometry_id"]: r for r in read_csv(OUT / "screening/candidate_pool.csv")}
pick = ["G42"] + [next(r["geometry_id"] for r in pool.values() if abs(float(r["Ma_design"]) - m) < 1e-9 and r["geometry_id"] != "G42") for m in (8.0, 15.0)]
conds = [(6.0, 6.0), (5.0, 4.0), (8.0, 10.0)]
q = [{"geometry_id": g, "mat_path": pool[g]["mat_path"], "Mach": m, "alpha_deg": a} for g in pick for m, a in conds]
write_csv(OUT / "work/preflight_queries.csv", q)
t = time.time()
print(run_matlab_eval(OUT / "work/preflight_queries.csv", OUT / "work/preflight_results.csv", OUT / "logs/preflight_matlab.log"))
print("wall s", round(time.time() - t, 1))
for r in read_csv(OUT / "work/preflight_results.csv"):
    print(r["geometry_id"], r["M_flight"], r["alpha_deg"], r["CL"], r["CD_pressure"], r["model_valid"], r["detached_area_fraction"], r["error"])

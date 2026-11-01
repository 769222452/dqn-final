"""Stage 2: every audited-VALID pool geometry x 49 centre conditions through the existing wedge model (one MATLAB process)."""
import time
from common import OUT, cfg, read_csv, write_csv, run_matlab_eval

c = cfg(); pool = [r for r in read_csv(OUT / "screening/candidate_pool.csv") if r["geometry_validity"] == "VALID"]
q = [{"geometry_id": r["geometry_id"], "mat_path": r["mat_path"], "Mach": m, "alpha_deg": a}
     for r in pool for m in c["flight_grid"]["M_flight"] for a in c["flight_grid"]["alpha_deg"]]
write_csv(OUT / "work/pool_queries.csv", q)
print("queries", len(q), flush=True); t = time.time()
(OUT / "logs/pool_eval_matlab.log").unlink(missing_ok=True)
run_matlab_eval(OUT / "work/pool_queries.csv", OUT / "work/pool_results.csv", OUT / "logs/pool_eval_matlab.log")
print("done wall_s", round(time.time() - t, 1), flush=True)

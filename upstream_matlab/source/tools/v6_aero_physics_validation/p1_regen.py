"""P1a: regenerate the four geometries with the diagnostic MATLAB copy and prove identity with the stored MATs.

    .venv\\Scripts\\python.exe -B tools\\v6_aero_physics_validation\\p1_regen.py
"""
from __future__ import annotations

import json
import subprocess
import time

import numpy as np
from scipy.io import loadmat

import common as C

DIAG_DIR = C.HERE / "matlab_diag"
REGEN = C.WORK / "regen"
KEYS = ("wav_x", "wav_y", "wav_z", "wav_up_x", "wav_up_y", "wav_up_z")


def run_matlab(jobs):
    REGEN.mkdir(parents=True, exist_ok=True)
    jf = REGEN / "jobs.json"
    jf.write_text(json.dumps(jobs), encoding="utf-8")
    code = f"addpath('{DIAG_DIR.as_posix()}'); run_diag_batch('{jf.as_posix()}','{DIAG_DIR.as_posix()}');"
    t0 = time.time()
    with open(REGEN / "matlab.log", "w", encoding="utf-8") as log:
        r = subprocess.run([str(C.MATLAB), "-batch", code], cwd=str(REGEN), stdout=log, stderr=subprocess.STDOUT,
                           creationflags=subprocess.CREATE_NO_WINDOW, timeout=1800)
    return r.returncode, time.time() - t0


def main():
    jobs = [{"geometry_id": g, "inputs10": C.inputs10(g), "out_mat": (REGEN / f"{C.SHORT[g]}_diag.mat").as_posix()}
            for g in C.ORDER]
    if not all((REGEN / f"{C.SHORT[g]}_diag.mat").exists() for g in C.ORDER):
        rc, dt = run_matlab(jobs)
        print("matlab rc", rc, "seconds", round(dt, 1))
    out = []
    for g in C.ORDER:
        p = REGEN / f"{C.SHORT[g]}_diag.mat"
        m = loadmat(p)
        st = C.load_mat(C.GEOMS[g]["mat"])
        rec = {"geometry_id": g, "regen_mat": str(p.relative_to(C.ROOT)), "matlab_version": str(m["matlab_version"][0])}
        diffs = {k: float(np.abs(np.asarray(m[k], float) - st[k]).max()) for k in KEYS}
        rec["max_abs_diff"] = diffs
        rec["bitwise_identical"] = all(np.array_equal(np.asarray(m[k], float), st[k]) for k in KEYS)
        lo = np.stack([m["wav_x"], m["wav_y"], m["wav_z"]], -1)
        up = np.stack([m["wav_up_x"], m["wav_up_y"], m["wav_up_z"]], -1)
        rec["regen_geometry_hash"] = C.native_hash(lo, up, C.inputs10(g))
        lo0, up0 = C.lower_upper(st)
        rec["stored_geometry_hash"] = C.native_hash(lo0, up0, C.inputs10(g))
        rec["hash_identical"] = rec["regen_geometry_hash"] == rec["stored_geometry_hash"]
        out.append(rec)
        print(g, rec["bitwise_identical"], rec["hash_identical"], max(diffs.values()))
    C.dump(C.WORK / "p1_regeneration_identity.json", out)


if __name__ == "__main__":
    main()

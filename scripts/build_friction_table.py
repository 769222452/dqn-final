"""Skin-friction table of the 298 candidate geometries (src/certified_flight/friction.py).

    python scripts/build_friction_table.py --out data/aero_friction/friction_table.npz --jobs 4

Axes: [geometry (action index order of the snapshot), Mach (snapshot grid), alpha (snapshot grid), altitude].
Values: CL_f, CD_f referred to the free-stream dynamic pressure and SREF; plus wetted area and length.
Atmosphere: the mission atmosphere (configs/environment.yaml). Wall temperature 1000 K, transition at
edge Re_x = 5e6 (the defaults of provenance/offdesign_aero_v7.m).
"""
import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from certified_flight import friction as F  # noqa: E402
from certified_flight.aero import AeroDatabase  # noqa: E402
from certified_flight.mission import Mission  # noqa: E402

ALTITUDES_M = (18e3, 22e3, 26e3, 30e3, 34e3, 40e3)
SNAPSHOT = ROOT/'data/geometry_subset'


def _atm():
    atm = Mission(ROOT, 'configs/mission.yaml').model(None).atm
    rows = []
    for h in ALTITUDES_M:
        a = float(atm.speed_of_sound(h))
        rows.append((a*a/(F.GAMMA*F.R_AIR), float(atm.density(h))))
    return rows


def job(args):
    gid, mach, alpha, mode = args
    p = F.load_panels(SNAPSHOT/'geometries'/gid/'native_geometry.mat')
    atm = _atm()
    out = np.zeros((len(mach), len(alpha), len(ALTITUDES_M), 2))
    for i, M in enumerate(mach):
        for j, a in enumerate(alpha):
            state = F.edge_state(p, M, a)
            for k, (T, rho) in enumerate(atm):
                out[i, j, k] = F.friction_coefficients(p, M, a, T, rho, mode=mode, state=state)
    return gid, out, float(2*p.area.sum()), float(p.s2.max())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--mode', default='transition', choices=('laminar', 'turbulent', 'transition'))
    ap.add_argument('--limit', type=int, help='first N geometries only (smoke test)')
    a = ap.parse_args(argv)
    db = AeroDatabase(SNAPSHOT)
    ids = db.ids[:a.limit]
    table = np.zeros((len(ids), len(db.M), len(db.A), len(ALTITUDES_M), 2))
    wet = np.zeros(len(ids))
    length = np.zeros(len(ids))
    index = {g: i for i, g in enumerate(ids)}
    with mp.get_context('spawn').Pool(a.jobs) as pool:
        for n, (gid, out, w, L) in enumerate(pool.imap_unordered(job, [(g, db.M, db.A, a.mode) for g in ids]), 1):
            table[index[gid]], wet[index[gid]], length[index[gid]] = out, w, L
            if n % 25 == 0 or n == len(ids):
                print(f'{n}/{len(ids)} geometries', flush=True)
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, geometry_ids=np.array(ids), mach_grid=db.M, alpha_rad_grid=db.A,
                        altitude_m_grid=np.array(ALTITUDES_M), CL_friction=table[..., 0], CD_friction=table[..., 1],
                        wetted_area_m2=wet, length_m=length)
    meta = dict(schema='aero-friction-table/1', mode=a.mode, wall_temperature_K=F.TW_K, Re_transition=F.RE_TR,
                reference_area_m2=F.SREF, edge_state='tangent wedge / Prandtl-Meyer per panel (P1 of offdesign_aero_v7.m)',
                correlations='Eckert reference temperature; laminar 0.664/sqrt(Re*), turbulent 0.0592/Re*^0.2, cell-averaged',
                snapshot_manifest_sha256=hashlib.sha256((SNAPSHOT/'manifest.json').read_bytes()).hexdigest(),
                friction_module_sha256=hashlib.sha256((ROOT/'src/certified_flight/friction.py').read_bytes()).hexdigest(),
                table_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), geometries=len(ids))
    path.with_suffix('.json').write_text(json.dumps(meta, indent=1)+'\n', encoding='utf-8')
    print(json.dumps(meta))


if __name__ == '__main__':
    mp.freeze_support()
    main()

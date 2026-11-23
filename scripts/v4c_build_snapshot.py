"""Build the certified-chain snapshot of the F8D2000B library (F8D2000 + 91 brake-ladder shapes, fixed G42 leading edge).

    python scripts/v4c_build_snapshot.py            # writes data/geometry_f8d2000b/{aerodynamics/p2_grid.npz, manifest.json}

Same Mach / alpha grids and the same P2 pressure-only coefficients as the target_rl tables (data/F8D2000, data/BrakeLadder);
the format is the one certified_flight.aero.AeroDatabase reads (manifest status VERIFIED, per-file SHA-256, geometry order =
sorted ids). The script refuses a library the chain cannot use: holes, non-positive drag, or a lift curve that is not
strictly increasing in alpha at any tabulated Mach or at the Mach midpoints (AeroDatabase.equal_lift needs monotone lift).
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'data'/'geometry_f8d2000b'
SOURCES = [ROOT/'data'/'F8D2000'/'p2_grid.npz', ROOT/'data'/'BrakeLadder'/'p2_grid.npz']


def main():
    parts = [np.load(p, allow_pickle=False) for p in SOURCES]
    M, A = parts[0]['mach_grid'], parts[0]['alpha_deg_grid']
    for z in parts[1:]:
        assert np.array_equal(z['mach_grid'], M) and np.array_equal(z['alpha_deg_grid'], A), 'grids differ'
    ids = np.concatenate([z['geometry_ids'].astype(str) for z in parts])
    CL = np.concatenate([z['CL'] for z in parts])
    CD = np.concatenate([z['CD_pressure'] for z in parts])
    usable = np.concatenate([z['usable_p2'] for z in parts])
    order = np.argsort(ids)
    ids, CL, CD, usable = ids[order], CL[order], CD[order], usable[order]
    assert len(set(ids)) == len(ids), 'duplicate geometry ids'
    if not usable.all() or not np.isfinite(CL).all() or not np.isfinite(CD).all() or (CD <= 0).any():
        sys.exit('library has holes or non-positive drag')
    # lift must increase strictly with alpha on every Mach curve the chain may evaluate (grid and midpoints)
    Mq = np.sort(np.r_[M, .5*(M[1:]+M[:-1])])
    curves = PchipInterpolator(M, CL, axis=1)(Mq)                 # [geometry, Mach, alpha]
    bad = np.argwhere(~(np.diff(curves, axis=2) > 0).all(axis=2))
    if len(bad):
        sys.exit(f'{len(bad)} (geometry, Mach) lift curves are not strictly increasing, e.g. {ids[bad[0][0]]} at Mach {Mq[bad[0][1]]}')
    (OUT/'aerodynamics').mkdir(parents=True, exist_ok=True)
    npz = OUT/'aerodynamics'/'p2_grid.npz'
    np.savez_compressed(npz, geometry_ids=ids, mach_grid=M, alpha_deg_grid=A, CL=CL, CD_pressure=CD, usable_p2=usable)
    files = [dict(path='aerodynamics/p2_grid.npz', size_bytes=npz.stat().st_size,
                  sha256=hashlib.sha256(npz.read_bytes()).hexdigest())]
    manifest = dict(schema='v4c-f8d2000b-snapshot/1', status='VERIFIED', created_utc=datetime.now(timezone.utc).isoformat(),
                    geometry_count=int(len(ids)), geometry_ids=ids.tolist(), tensor_shape=list(CL.shape),
                    tensor_axis_order=['geometry (sorted id)', 'mach_grid', 'alpha_deg_grid'], model='P2 pressure only',
                    sources=[dict(path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in SOURCES],
                    interpretation='F8D2000B library (2000 shapes + 91 brake-ladder shapes) for the certified chain (v4c)',
                    files=files)
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=1)+'\n', encoding='utf-8')
    print(f'{len(ids)} geometries, tensor {list(CL.shape)} -> {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()

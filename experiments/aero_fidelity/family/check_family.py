"""Audit the generated family and apply the declared selection rule (experiments/aero_fidelity/README.md).

    python experiments/aero_fidelity/family/check_family.py runs/aero_fidelity/family \
        --out runs/aero_fidelity/family_selected

Per generated job: watertight closed surface (scripts/cfd/make_su2_cases.closed_surface), positive enclosed
volume, finite coordinates, length and half span; tangent-wedge CL/CD at the design Mach as a sanity check.

Selection rule (declared before any CFD result):
1. candidate delta: every design Mach has at least one valid geometry at that delta;
2. per candidate delta, target volume V* = median over design Mach of the volume at the a_x closest to 1.0;
   for each design Mach the valid a_x with the volume closest to V* is taken;
3. the delta is admissible only if all three volumes are within +-3 % of V*; among admissible deltas the
   smallest maximum deviation wins (ties: smallest |delta|).
If no delta is admissible the script stops with status NO_ADMISSIBLE_FAMILY; the rule is not relaxed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'/'cfd'))

from certified_flight import friction as F  # noqa: E402
import make_su2_cases as cfd  # noqa: E402

TOL = .03


def audit(mat):
    V, T = cfd.closed_surface(mat)
    closed = cfd.check_closed(T)
    vol = float(np.sum(np.einsum('ij,ij->i', V[T[:, 0]], np.cross(V[T[:, 1]], V[T[:, 2]])))/6)
    ok = bool(np.isfinite(V).all() and closed['open_edges'] == 0 and closed['nonmanifold_edges'] == 0 and vol > 0)
    return dict(valid=ok, volume_m3=vol, length_m=float(np.ptp(V[:, 0])), half_span_m=float(V[:, 1].max()),
                height_m=float(np.ptp(V[:, 2])), **closed)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('family')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    fam = Path(a.family)
    jobs = json.loads((fam/'jobs.json').read_text())
    rows = []
    for j in jobs:
        d = fam/j['job_id']
        st = json.loads((d/'status.json').read_text()) if (d/'status.json').exists() else dict(status='MISSING')
        r = dict(j, status=st.get('status'), error=st.get('error_message'))
        if st.get('status') == 'GENERATED':
            r.update(audit(d/'native_geometry.mat'))
            if r['valid']:
                p = F.load_panels(d/'native_geometry.mat')
                r['design_tangent_wedge_CL_CD_alpha4'] = F.pressure_coefficients(p, j['Ma_design'], np.deg2rad(4.))
        else:
            r['valid'] = False
        rows.append(r)
    machs = sorted({r['Ma_design'] for r in rows})
    deltas = sorted({r['delta_deg'] for r in rows})
    options = []
    for dl in deltas:
        per = {M: [r for r in rows if r['Ma_design'] == M and r['delta_deg'] == dl and r['valid']] for M in machs}
        if not all(per.values()):
            continue
        ref = [min(v, key=lambda r: abs(r['a_x']-1.))['volume_m3'] for v in per.values()]
        target = float(np.median(ref))
        chosen = {M: min(v, key=lambda r: abs(r['volume_m3']-target)) for M, v in per.items()}
        dev = max(abs(c['volume_m3']/target-1) for c in chosen.values())
        options.append(dict(delta_deg=dl, target_volume_m3=target, max_volume_deviation=dev, admissible=dev <= TOL,
                            chosen={str(M): c['job_id'] for M, c in chosen.items()}))
    adm = [o for o in options if o['admissible']]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = dict(rule=__doc__.split('Selection rule')[1].strip(), generated=sum(r['status'] == 'GENERATED' for r in rows),
                   valid=sum(r['valid'] for r in rows), jobs=len(rows), options=options)
    if not adm:
        summary['status'] = 'NO_ADMISSIBLE_FAMILY'
    else:
        best = min(adm, key=lambda o: (o['max_volume_deviation'], abs(o['delta_deg'])))
        summary.update(status='SELECTED', selected=best)
        ids = []
        for M, jid in best['chosen'].items():
            src = fam/jid/'native_geometry.mat'
            gid = f'FAM_MD{float(M):g}'
            (out/gid).mkdir(exist_ok=True)
            shutil.copy2(src, out/gid/'native_geometry.mat')
            ids.append(gid)
            member = {k: v for k, v in next(r for r in rows if r['job_id'] == jid).items() if k != 'error'}
            member['sha256'] = hashlib.sha256(src.read_bytes()).hexdigest()
            summary.setdefault('members', {})[gid] = member
        (out/'select.json').write_text(json.dumps(dict(cfd_geometries=ids, source='check_family.py'), indent=1))
    (out/'family_audit.json').write_text(json.dumps(dict(summary, rows=rows), indent=1, default=float))
    print(json.dumps({k: v for k, v in summary.items() if k != 'rule'}, indent=1, default=float))
    return 0 if adm else 3


if __name__ == '__main__':
    sys.exit(main())

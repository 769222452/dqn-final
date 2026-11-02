"""Audit the generated family and apply the declared selection rule (experiments/aero_fidelity/README.md).

    python experiments/aero_fidelity/family/check_family.py runs/aero_fidelity/family \
        --out runs/aero_fidelity/family_selected [--jobs 4]

Per generated job (every failure is caught and recorded with a reason; one bad member never stops the check):
1. the project's frozen native-mesh audit (native_audit.py): VALID / INVALID / AMBIGUOUS;
2. only for VALID members: a watertight closed surface can be built (scripts/cfd/make_su2_cases.closed_surface)
   and has positive volume.
A member is `valid` only if both hold.

Selection rule (declared before any CFD result; protocol amendment 1 only enlarges the design variable):
1. a candidate is a design key shared by the three design Machs (version 1: delta; version 2: the shock
   parametrisation `design_key`, e.g. pair_d+2_r0.65 or g42_d+0);
2. per candidate, V* = median over design Mach of the volume of the valid member whose a_x is closest to 1.0;
   per design Mach the valid member with the volume closest to V* is taken;
3. the candidate is admissible only if every chosen volume is within +-3 % of V*; among admissible
   candidates the smallest maximum deviation wins (ties: the key that sorts first).
Protocol amendment 2 (version-3 jobs carry `target_volume_m3`, the enclosed volume of G42): V* is that target
instead of the median in step 2; steps 1 and 3, the tolerance and the frozen audit are unchanged.
If no candidate is admissible the script stops with status NO_ADMISSIBLE_FAMILY; the rule is not relaxed.
"""
import argparse
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'/'cfd'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

TOL = .03


def audit_one(args):
    """Audit one generated job folder; never raises."""
    job, folder = args
    r = dict(job, status='MISSING', valid=False, reason=None)
    sfile = Path(folder)/job['job_id']/'status.json'
    if sfile.exists():
        st = json.loads(sfile.read_text(encoding='utf-8-sig'))
        r.update(status=st.get('status'), error=st.get('error_message'))
    if r['status'] != 'GENERATED':
        r['reason'] = 'GENERATION_FAILED' if r['status'] == 'GENERATION_FAILED' else 'MISSING_OUTPUT'
        return r
    mat = Path(folder)/job['job_id']/'native_geometry.mat'
    try:
        from native_audit import native_audit
        a = native_audit(mat)
        r.update(frozen_audit=a['classification'], audit_reason=a['failure_reason'],
                 surface_intersections=a.get('surface_intersection_count'), tail_min_thickness=a.get('tail_min_thickness'))
        if a['classification'] != 'VALID':
            r['reason'] = 'FROZEN_AUDIT_'+a['classification']
            return r
    except Exception as exc:                                    # noqa: BLE001
        r['reason'] = 'AUDIT_ERROR: '+repr(exc)[:200]
        return r
    try:
        import make_su2_cases as cfd
        V, T = cfd.closed_surface(mat)
        c = cfd.check_closed(T)
        vol = float(np.sum(np.einsum('ij,ij->i', V[T[:, 0]], np.cross(V[T[:, 1]], V[T[:, 2]])))/6)
        r.update(volume_m3=vol, length_m=float(np.ptp(V[:, 0])), **c)
        if c['open_edges'] or c['nonmanifold_edges'] or not vol > 0:
            r['reason'] = 'NOT_WATERTIGHT_OR_NONPOSITIVE_VOLUME'
            return r
    except Exception as exc:                                    # noqa: BLE001
        r['reason'] = 'CLOSED_SURFACE_ERROR: '+repr(exc)[:200]
        return r
    r['valid'] = True
    return r


def select(rows, machs=(4., 6., 8.), tol=TOL, target=None):
    """The declared selection rule; rows need Ma_design, design_key, a_x, valid, volume_m3 (pure function).

    target=None: V* is the median volume of the a_x-closest-to-1 members (versions 1 and 2).
    target=V:    V* = V (version 3, the enclosed volume of G42)."""
    keys = sorted({r['design_key'] for r in rows})
    options = []
    for key in keys:
        per = {M: [r for r in rows if r['design_key'] == key and r['Ma_design'] == M and r['valid']] for M in machs}
        if not all(per.values()):
            options.append(dict(design_key=key, admissible=False, reason='a design Mach has no valid member',
                                valid_per_mach={str(M): len(v) for M, v in per.items()}))
            continue
        if target is None:
            ref = [min(v, key=lambda r: abs(r['a_x']-1.))['volume_m3'] for v in per.values()]
            tv = float(np.median(ref))
        else:
            tv = float(target)
        chosen = {M: min(v, key=lambda r: abs(r['volume_m3']-tv)) for M, v in per.items()}
        dev = max(abs(c['volume_m3']/tv-1) for c in chosen.values())
        options.append(dict(design_key=key, target_volume_m3=tv, max_volume_deviation=float(dev), admissible=bool(dev <= tol),
                            chosen={str(M): c['job_id'] for M, c in chosen.items()},
                            valid_volume_range_per_mach={str(M): [min(r['volume_m3'] for r in v), max(r['volume_m3'] for r in v)]
                                                         for M, v in per.items()}))
    adm = [o for o in options if o['admissible']]
    best = min(adm, key=lambda o: (o['max_volume_deviation'], o['design_key'])) if adm else None
    return options, best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('family')
    ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', type=int, default=4)
    a = ap.parse_args(argv)
    fam = Path(a.family)
    jobs = json.loads((fam/'jobs.json').read_text())
    for j in jobs:                                              # version-1 job lists have no design_key
        j.setdefault('design_key', 'd%+g' % j['delta_deg'])
    with get_context('spawn').Pool(a.jobs) as pool:
        rows = pool.map(audit_one, [(j, str(fam)) for j in jobs], chunksize=1)
    targets = {j.get('target_volume_m3') for j in jobs}
    assert len(targets) == 1, 'jobs of one family must share one target volume (or none)'
    options, best = select(rows, target=targets.pop())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    reasons = {}
    for r in rows:
        reasons[r['reason'] or 'VALID'] = reasons.get(r['reason'] or 'VALID', 0)+1
    summary = dict(jobs=len(rows), generated=sum(r['status'] == 'GENERATED' for r in rows), valid=sum(r['valid'] for r in rows),
                   outcomes=reasons, options=options, rule=__doc__.split('Selection rule')[1].strip())
    if best is None:
        summary['status'] = 'NO_ADMISSIBLE_FAMILY'
    else:
        summary.update(status='SELECTED', selected=best, members={})
        ids = []
        for M, jid in best['chosen'].items():
            src = fam/jid/'native_geometry.mat'
            gid = f'FAM_MD{float(M):g}'
            (out/gid).mkdir(exist_ok=True)
            shutil.copy2(src, out/gid/'native_geometry.mat')
            ids.append(gid)
            member = {k: v for k, v in next(r for r in rows if r['job_id'] == jid).items() if k != 'error'}
            member['sha256'] = hashlib.sha256(src.read_bytes()).hexdigest()
            summary['members'][gid] = member
        (out/'select.json').write_text(json.dumps(dict(cfd_geometries=ids, source='check_family.py', design_key=best['design_key']), indent=1))
    (out/'family_audit.json').write_text(json.dumps(dict(summary, rows=rows), indent=1, default=float))
    print(json.dumps({k: v for k, v in summary.items() if k not in ('rule', 'options', 'members')}, indent=1, default=float))
    print('candidates:', [(o['design_key'], o['admissible'], round(o.get('max_volume_deviation', float('nan')), 3)) for o in options][:60])
    return 0 if best else 3


if __name__ == '__main__':
    sys.exit(main())

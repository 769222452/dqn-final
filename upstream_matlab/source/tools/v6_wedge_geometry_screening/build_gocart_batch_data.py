"""Generate DATA ONLY for GoCart_Batch_Win7 from the real V6 handoff (STEP + related_flight_cases.csv).
Static scripts (bat/vbs/ini/README) are authored in the package folder and are never touched here.
No CFD is run and no aero value is created: the case table has flight conditions only.
"""
import csv, hashlib, json, shutil, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/v6_wedge_geometry_screening'
PKG = OUT / 'GoCart_Batch_Win7'
HANDOFF = OUT / 'UG_CFD_HANDOFF'
SREF, LREF = '6.931920786884801', '5'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


(PKG / 'geometries').mkdir(parents=True, exist_ok=True)
rows = []
info_all = {}
for d in sorted(p for p in HANDOFF.iterdir() if p.is_dir()):
    gid = d.name
    info = json.loads((d / 'geometry_info.json').read_text(encoding='utf-8'))
    dst = PKG / 'geometries' / (gid + '.step')
    shutil.copyfile(d / 'geometry.step', dst)
    h = sha(dst)
    info_all[gid] = {'geometry_hash': info['geometry_hash'], 'step_sha256': h, 'Ma_design': info['Ma_design']}
    for r in csv.DictReader((d / 'related_flight_cases.csv').open(encoding='utf-8-sig', newline='')):
        M, a = float(r['M_flight']), float(r['alpha_flight_deg'])
        cid = '%s__M%s_A%s' % (gid, ('%.2f' % M).replace('.', 'p'), ('%05.2f' % a).replace('.', 'p'))
        rows.append(dict(case_id=cid, step_file='geometries/%s.step' % gid, mach='%.2f' % M, alpha_deg='%.2f' % a,
                         geometry_id=gid, step_sha256=h, role=r['role'], Ma_design='%s' % info['Ma_design'],
                         Sref_m2=SREF, Lref_m=LREF, length_unit='mm'))
prio = ['S2_MD14.004_4b1ceeb42b', 'S2_MD15_029eb6f5fd', 'G42']
rows.sort(key=lambda r: (prio.index(r['geometry_id']) if r['geometry_id'] in prio else 3, r['geometry_id'], float(r['mach']), float(r['alpha_deg'])))
if len(rows) != 343 or len({r['case_id'] for r in rows}) != 343 or len(info_all) != 31:
    raise RuntimeError('handoff changed: expected 31 geometries / 343 unique cases')
cols = list(rows[0])


def write_csv(p, rs):
    with open(p, 'w', encoding='ascii', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator='\r\n'); w.writeheader(); w.writerows(rs)


write_csv(PKG / 'cases_full_343.csv', rows)
test = [next(r for r in rows if r['case_id'] == i) for i in ('S2_MD14.004_4b1ceeb42b__M6p50_A07p00', 'S2_MD14.004_4b1ceeb42b__M6p50_A08p00')]
write_csv(PKG / 'cases.csv', test)
for r in rows:
    assert (PKG / r['step_file']).exists()
print('geometries', len(info_all), 'full cases', len(rows), 'default test cases', len(test))

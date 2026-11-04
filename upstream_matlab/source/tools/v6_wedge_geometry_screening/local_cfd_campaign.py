"""CFD campaign: several geometries x Mach x alpha with SU2 Euler (resumable). usage: local_cfd_campaign.py <threads> <geom_id> [<geom_id> ...]
One mesh per geometry (shared by all flow conditions). Results -> results/.../local_cfd_campaign/<geom>/<case>/result.json
"""
import csv, json, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(r'D:\v6-mission-morph-core')
OUT = ROOT / 'results/v6_wedge_geometry_screening'
CAMP = OUT / 'local_cfd_campaign'
PY = str(ROOT / 'tools_cfd/venv/Scripts/python.exe')
TOOLS = ROOT / 'tools/v6_wedge_geometry_screening'
MACHS = [5.0, 6.5, 8.0]
ALPHAS = [3.0, 5.0, 7.0]
thr = sys.argv[1]
geoms = sys.argv[2:]


def sh(cmd, env=None, log=None):
    e = dict(os.environ, **(env or {}))
    return subprocess.run(cmd, env=e, capture_output=True, text=True, errors='replace')


def make_mesh(g):
    mdir = CAMP / g / 'mesh'
    if (mdir / 'mesh.su2').exists() and (mdir / 'mesh_info.json').exists():
        return mdir
    for sizes in ('0.03,0.1,0.9', '0.035,0.11,0.9', '0.04,0.14,1.0', '0.045,0.15,1.0'):
        if mdir.exists():
            shutil.rmtree(mdir)
        p = sh([PY, '-B', str(TOOLS / 'local_cfd_mesh.py'), str(OUT / 'UG_CFD_HANDOFF' / g), str(mdir), 'medium'], {'MESH_SIZES': sizes})
        if (mdir / 'mesh.su2').exists() and (mdir / 'mesh_info.json').exists():
            info = json.loads((mdir / 'mesh_info.json').read_text()); info['sizes_used'] = sizes
            (mdir / 'mesh_info.json').write_text(json.dumps(info, indent=1))
            return mdir
        print(g, 'mesh failed with', sizes, flush=True)
    return None


def parse(cdir):
    r = [{k.strip().strip('"'): v.strip() for k, v in x.items()} for x in csv.DictReader(open(cdir / 'history.csv'))]
    x = r[-1]; last = r[-100:]
    sp = lambda k: max(float(y[k]) for y in last) - min(float(y[k]) for y in last)
    return dict(spread_CL100=sp('CL(BODY)'), spread_CD100=sp('CD(BODY)'), iters=int(x['Inner_Iter']) + 1, rms_rho=float(x['rms[Rho]']), CL=float(x['CL(BODY)']), CD=float(x['CD(BODY)']),
                CD_base=float(x['CD(BASE)']), CL_base=float(x['CL(BASE)']))


for g in geoms:
    mdir = make_mesh(g)
    if mdir is None:
        (CAMP / g).mkdir(parents=True, exist_ok=True)
        (CAMP / g / 'MESH_FAILED.txt').write_text('all sizes failed')
        continue
    info = json.loads((mdir / 'mesh_info.json').read_text())
    for M in MACHS:
        for a in ALPHAS:
            cdir = CAMP / g / f'M{M:.1f}_A{a:.0f}'
            if (cdir / 'result.json').exists():
                continue
            if cdir.exists():
                shutil.rmtree(cdir)
            cdir.mkdir(parents=True)
            env = {'SU2_FLUX': 'HLLC', 'SU2_MESH': '../mesh/mesh.su2'}
            t0 = time.time()
            sh([PY, '-B', str(TOOLS / 'local_cfd_run.py'), str(cdir), str(M), str(a), '600', thr, 'first'], env)
            shutil.copyfile(cdir / 'restart_flow.dat', cdir / 'solution_flow.dat')
            sh([PY, '-B', str(TOOLS / 'local_cfd_run.py'), str(cdir), str(M), str(a), '700', thr, 'second', 'restart'], env)
            try:
                res = parse(cdir)
            except Exception as e:
                res = {'error': str(e)}
            res.update(geometry_id=g, M=M, alpha=a, tets=info['tets'], mesh_sizes=info.get('sizes_used'), wall_s=round(time.time() - t0, 1))
            for junk in ('restart_flow.dat', 'solution_flow.dat', 'restart_flow.csv', 'surface_flow.csv'):
                (cdir / junk).unlink(missing_ok=True)
            (cdir / 'result.json').write_text(json.dumps(res, indent=1))
            print(g, M, a, {k: res.get(k) for k in ('CL', 'CD', 'rms_rho', 'iters', 'wall_s')}, flush=True)

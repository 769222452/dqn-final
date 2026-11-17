"""Read-only validation of all published files, geometries and 720,000 aero rows."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def surfaces(native):
    return [np.stack([native[prefix+axis] for axis in 'xyz'],axis=-1)
            for prefix in ['wav_','wav_up_']]


def basis(surface):
    t=np.broadcast_to(np.linspace(0,1,surface.shape[1]),surface.shape[:2])
    chord=np.maximum(0,5-surface[:,0,0])[:,None]/5
    eta=np.clip(surface[:,0,1]/np.max(surface[:,0,1]),0,1)[:,None]
    b=np.stack([t**2*(1-t)**3/.03456,t**3*(1-t)**2/.03456,
                t**2*(3-2*t),16*t**2*(1-t)**2*(2*eta**2-1)],axis=-1)*chord[...,None]
    b[:,0,:]=0
    b[0,:,:]=0
    return b


def main():
    manifest={}
    for line in (ROOT/'SHA256SUMS').read_text(encoding='utf-8').splitlines():
        expected, relative=line.split('  ',1)
        require(relative not in manifest, f'Duplicate manifest entry: {relative}')
        manifest[relative]=expected
        require(hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==expected,
                f'File hash mismatch: {relative}')
    actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()
            and '__pycache__' not in p.parts and p.name!='SHA256SUMS' and p.suffix!='.pyc'}
    require(actual==set(manifest),'Manifest file set mismatch')
    ids=read(ROOT/'selected_ids.json')
    require(len(ids)==len(set(ids))==2000,'Candidate count or uniqueness mismatch')
    require({p.name for p in (ROOT/'geometries').iterdir() if p.is_dir()}==set(ids),'Geometry folder mismatch')
    with (ROOT/'action_index.csv').open(encoding='utf-8-sig',newline='') as f:
        actions=list(csv.DictReader(f))
    require([r['geometry_id'] for r in actions]==ids,'Action IDs mismatch')
    require([int(r['action_index']) for r in actions]==list(range(2000)),'Action index mismatch')
    for row in actions:
        require(row['mat_path']==f"geometries/{row['geometry_id']}/native_geometry.mat",'MAT path mismatch')
    cfg=read(ROOT/'config.json')
    parents={g:surfaces(loadmat(ROOT/f'references/{g}/native_geometry.mat')) for g in cfg['parents']}
    bases={g:[basis(s) for s in geo] for g,geo in parents.items()}
    geometry_hashes=set()
    max_le_error=0.0
    with np.load(ROOT/'aerodynamics/p2_grid.npz',allow_pickle=False) as archive:
        # Decode each compressed tensor once before checking 720,000 CSV rows.
        z={key:archive[key] for key in archive.files}
        require(z['geometry_ids'].tolist()==ids,'Aero ID order mismatch')
        require(z['CL'].shape==(2000,18,20),'Aero tensor dimensions mismatch')
        require(z['usable_p2'].all() and z['model_valid'].all(),'Unusable P2 points found')
        mat=loadmat(ROOT/'aerodynamics/p2_grid.mat',simplify_cells=True)
        require(mat['geometry_ids'].tolist()==ids,'MAT aero ID order mismatch')
        for k in ['CL','CD_pressure','Cm_pressure','model_valid','usable_p2','mach_grid','alpha_deg_grid']:
            require(np.array_equal(z[k],mat[k]),f'NPZ/MAT mismatch: {k}')
        for k in ['CL','CD_pressure','Cm_pressure']:
            require(np.isfinite(z[k]).all(),f'Nonfinite {k}')
        require((z['CD_pressure']>0).all(),'Nonpositive drag')
        mi={float(m):i for i,m in enumerate(z['mach_grid'])}
        ai={float(a):i for i,a in enumerate(z['alpha_deg_grid'])}
        expected={(m,a) for m in mi for a in ai}
        for index,gid in enumerate(ids):
            native=loadmat(ROOT/f'geometries/{gid}/native_geometry.mat',simplify_cells=True)
            geo=surfaces(native)
            h=hashlib.sha256()
            parent=str(native['parent_geometry_id'])
            coefficients=np.asarray(native['shape_coefficients_m'])
            require(coefficients.shape==(8,),'Coefficient shape mismatch')
            require((abs(coefficients)<=np.array(cfg['parameter_abs_limits_m'])+1e-14).all(),'Parameter bound mismatch')
            params=read(ROOT/f'geometries/{gid}/shape_parameters.json')
            require(np.array_equal(coefficients,[params[k] for k in cfg['parameter_names']]),'Parameter JSON mismatch')
            for si,surface in enumerate(geo):
                require(surface.shape==(50,50,3) and surface.dtype==np.float64 and np.isfinite(surface).all(),f'Invalid surface: {gid}')
                reconstructed=parents[parent][si].copy()
                offset=4 if si==0 else 0
                reconstructed[:,:,2]+=np.einsum('ijk,k->ij',bases[parent][si],coefficients[offset:offset+4])
                require(np.allclose(surface,reconstructed,rtol=0,atol=3e-14),f'Reconstruction mismatch: {gid}')
                err=float(np.linalg.norm(surface[:,0,:]-parents['G42'][si][:,0,:],axis=1).max())
                max_le_error=max(max_le_error,err)
                require(err<=1e-10,f'Leading edge mismatch: {gid}')
                h.update(surface.tobytes())
            require(h.hexdigest() not in geometry_hashes,f'Duplicate coordinates: {gid}')
            geometry_hashes.add(h.hexdigest())
            certificate=read(ROOT/f'geometries/{gid}/certificate.json')
            require(certificate['classification']=='VALID' and certificate['geometry_id']==gid,'Invalid certificate')
            with (ROOT/f'aerodynamics/{gid}.csv').open(encoding='utf-8-sig',newline='') as f:
                rows=list(csv.DictReader(f))
            require(len(rows)==360,'Incomplete CSV')
            keys={(float(r['M_flight']),float(r['alpha_deg'])) for r in rows}
            require(keys==expected,f'CSV axes mismatch: {gid}')
            for row in rows:
                require(row['geometry_id']==gid,'CSV ID mismatch')
                require(row['model_valid'].lower() in ['1','true'],'CSV validity mismatch')
                for key in ['CL','CD_pressure','Cm_pressure']:
                    require(float(row[key])==float(z[key][index,mi[float(row['M_flight'])],ai[float(row['alpha_deg'])]]),
                            f'CSV/NPZ coefficient mismatch: {gid}/{key}')
            if (index+1)%500==0:
                print(f'Checked {index+1}/2000 geometries, reconstruction and complete CSV tables',flush=True)
    with np.load(ROOT/'evaluation/equal_lift_new_2000.npz',allow_pickle=False) as z:
        require(z['geometry_ids'].tolist()==ids,'Selection ID mismatch')
        cd=z['CD_pressure']; scores=cd.mean(axis=(1,2)); fi=int(scores.argmin())
        require(np.array_equal(cd.argmin(axis=0),z['winner_index']),'Winner index mismatch')
        summary=read(ROOT/'evaluation/summary.json')
        gain=100*(1-cd.min(axis=0).mean()/scores[fi])
        require(ids[fi]==summary['best_fixed_geometry'],'Best fixed geometry mismatch')
        require(abs(gain-summary['reduction_vs_best_fixed_percent'])<1e-12,'Gain mismatch')
    print(json.dumps(dict(status='PASS',files_hashed=len(manifest),valid_unique_geometries=len(geometry_hashes),
        reconstructed_geometries=2000,complete_aerodynamic_points=720000,
        native_geometry_precision='float64',max_leading_edge_error_m=max_le_error,
        best_fixed_geometry=ids[fi],static_equal_lift_reduction_percent=gain),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()

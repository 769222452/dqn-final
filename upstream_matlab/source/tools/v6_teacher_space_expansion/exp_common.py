"""Isolated campaign state; existing Teacher, audit and aerodynamic data are read-only."""
from __future__ import annotations
import hashlib, importlib.util, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[2]
TOOLS = Path(__file__).resolve().parent
OUT = ROOT / 'results/v6_teacher_space_expansion'
OLD = ROOT / 'results/v6_wedge_geometry_screening'
PARAMS = ['a1','a2','b1','b2','d1','a_x']
INPUTS = ['Ma_design','a1','a2','b1','b2','c1','c2','d1','AOA_design_prog','a_x']
FEATURES = ['Ma_design'] + PARAMS
CLMIN = 0.026150265744858465
CLTARGETS = [CLMIN,0.0398,0.05350025264833205]
SREF = 6.931920786884801
AB = ['S2_MD14.004_4b1ceeb42b','S2_MD15_029eb6f5fd']
LOWER = np.array([4.,3.,3.,3.,3.,.02,.9])
UPPER = np.array([15.,20.,20.,20.,20.,.18,1.1])
FROZEN_M = np.arange(5.,8.01,.5)
FROZEN_A = np.arange(4.,10.01,1.)

def utc(): return datetime.now(timezone.utc).isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def clean(v):
    if isinstance(v,dict): return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple,np.ndarray)): return [clean(x) for x in v]
    if isinstance(v,(np.integer,np.floating,np.bool_)): v=v.item()
    if isinstance(v,float) and not np.isfinite(v): return None
    if isinstance(v,Path): return v.as_posix()
    return v
def dump(p,v):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.tmp{os.getpid()}')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(clean(v),f,ensure_ascii=False,indent=2,allow_nan=False); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,p)
def csv(p,df):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_name(p.name+f'.tmp{os.getpid()}')
    pd.DataFrame(df).to_csv(tmp,index=False); os.replace(tmp,p)
def event(kind,**kw):
    OUT.mkdir(parents=True,exist_ok=True)
    r={'utc':utc(),'event':kind,**kw}
    with (OUT/'events.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(clean(r),ensure_ascii=False)+'\n'); f.flush(); os.fsync(f.fileno())
    print(json.dumps(clean(r),ensure_ascii=False),flush=True)
def surfaces(p):
    m=loadmat(p)
    return tuple(np.stack([m[pre+x] for x in 'xyz'],-1) for pre in ['wav_','wav_up_'])
def coordinate_hash(p):
    h=hashlib.sha256(b'native-coordinates-only-v1\0')
    for x in surfaces(p):
        x=np.asarray(x,dtype='<f8',order='C'); h.update(str(x.shape).encode()); h.update(x.tobytes())
    return h.hexdigest()
def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
def audit_module():
    d=str(ROOT/'tools/geometry_audit')
    if d not in sys.path: sys.path.insert(0,d)
    return module('exp_native_scan',ROOT/'tools/geometry_audit/scan.py')
def closed_metrics(p):
    # Exact current p0b_volume_rescreen.closed_volume calculation, using original helpers.
    d=str(ROOT/'tools/v6_aero_physics_validation')
    if d not in sys.path: sys.path.append(d)
    gm=module('exp_p0_geometry',ROOT/'tools/v6_aero_physics_validation/p0_geometry.py')
    lo,up=surfaces(p); parts=gm.closed_half_hull(lo,up)
    t=np.concatenate([parts[k] for k in ['lower','upper_reversed','base','tip','symmetry']])
    v=gm.signed_volume(t); err=abs(v-gm.signed_volume(t,origin=(3.,-2.,5.)))
    return {'closed_volume_full':2*abs(v),'closure_translation_error':err,
            'grid_thickness_max':float((lo[...,2]-up[...,2]).max()),
            'definition':'p0b_volume_rescreen.closed_volume; native closed half hull x2; original grid thickness'}
def old_pool(): return pd.read_csv(OUT/'baseline/candidate_pool.csv',float_precision='round_trip')
def pool():
    rows=[old_pool()]
    records=[load(p) for p in sorted((OUT/'cases').glob('*/record.json'))]
    if records: rows.append(pd.DataFrame(records))
    return pd.concat(rows,ignore_index=True)
def gate(p):
    return (p.geometry_validity.eq('VALID') & p.volume_ratio_G42.ge(.85) & p.thickness_ratio_G42.ge(.5)
            & ~p.get('duplicate_of',pd.Series('',index=p.index)).fillna('').ne(''))
def true(s): return s.astype(str).str.lower().isin(['true','1','1.0'])
def frozen_pairs(): return [(float(m),float(a)) for m in FROZEN_M for a in FROZEN_A]
def protect_manifest():
    paths=list((ROOT/'matlab/active').glob('*.m'))+list((ROOT/'results/v6_matlab_no_new_cfd/code/matlab').glob('*.m'))
    paths+=list((ROOT/'tools/geometry_audit').glob('*.py'))
    paths += [ROOT/'tools/v6_wedge_geometry_screening/wedge_eval.m',ROOT/'tools/v6_aero_physics_validation/p0_geometry.py',
              ROOT/'tools/v6_aero_physics_validation/p0b_volume_rescreen.py',ROOT/'results/v5_geometry_audit/classification_rules.json',
              ROOT/'tools/v6_mission_geometry_stage2/common.py',ROOT/'configs/mission_1200s.yaml',
              ROOT/'data/geometries/candidate_identity_index.csv',OLD/'work/pool_results.csv',OLD/'screening/candidate_pool.csv',
              OLD/'screening/top5_by_condition.csv',OLD/'config/screening_config.json']
    return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}
def verify_sources():
    expected=load(OUT/'baseline/source_hashes.json')
    extra=OUT/'baseline/additional_source_hashes.json'
    if extra.exists(): expected.update(load(extra))
    changed=[p for p,h in expected.items() if not (ROOT/p).exists() or sha(ROOT/p)!=h]
    if changed: raise RuntimeError(f'Protected sources changed: {changed}')

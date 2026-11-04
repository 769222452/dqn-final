"""Read-only source inventory; freezes audit rules before any new geometry call."""
from pathlib import Path
import hashlib, json, re, subprocess
from datetime import datetime, timezone
import numpy as np
from scipy.io import loadmat

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/v5_geometry_audit'
TOOLS=ROOT/'tools/geometry_audit'
PARAMS=['a1','a2','b1','b2','d1','a_x']
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def protected():
    roots=[Path(p) for p in ['D:/final-DQN','D:/final-DQN-v3','D:/final-DQN-v4','D:/final-DQN-physics-v4','D:/mat','D:/final-DQN-v5']]
    files={}
    for root in roots:
        if not root.exists(): continue
        for p in root.rglob('*'):
            if not p.is_file() or any(x in {'.git','__pycache__','.pytest_cache','.venv','node_modules'} for x in p.parts): continue
            if OUT in p.parents or TOOLS in p.parents: continue
            files[str(p)]=sha(p)
    return {'roots':{str(r):r.exists() for r in roots},'files':files}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'classification_rules.json').exists():
        raise RuntimeError('Rules already frozen; do not silently replace source provenance or rules')
    snap=protected(); dump(OUT/'protected_files_before.json',snap)
    rules={
      'version':'native-triangle-audit-1','frozen_utc':datetime.now(timezone.utc).isoformat(),
      'length_tolerance':'max(128*float64_eps*L, 1e-9*L), L=maximum coordinate extent',
      'area_tolerance':'1e-14 * L**2 (triangle area)',
      'thickness_negative_tolerance':'same length tolerance; no centimetre-scale tolerance',
      'native_triangulation':'Each native quad (a,b,c,d) split (a,b,c),(a,c,d); no smoothing or regridding. Opposite diagonal is a sensitivity check.',
      'allowed_degeneracy':'Only zero-area faces on row 0 when all row-0 vertices coincide with the common leading-edge tip; source explicitly collapses outermost row. All raw degeneracies still counted.',
      'allowed_contact':'Only the actual shared upper/lower leading-edge polyline; self pairs may touch at their shared native vertices/edge only. No tail or interior penetration exclusion.',
      'section_method':'Intersect native triangles with x plane; compare piecewise-linear yz curves on common y domain using union of actual breakpoints. No extrapolation. t=lower_program_z-upper_program_z.',
      'VALID':'Finite real complete native surfaces; no unapproved degeneracy or intersection/self-intersection; single-valued graph with nonnegative common-domain thickness; connected leading edge and symmetry conditions. Documented open base is allowed.',
      'INVALID':'Confirmed nonallowed surface intersection or self-intersection, nonfinite/complex generated coordinates, or unapproved degenerate interior faces. Thickness alone never establishes INVALID.',
      'AMBIGUOUS':'Negative thickness without actual intersection, unresolved graph/connection/normal diagnostics, or insufficient surface semantics.',
      'GENERATION_FAILED':'No complete native geometry returned, MATLAB error, or supervisor hard timeout.',
      'boundary_semantics':'Native half-skin, shared leading edge; y=0 is mirror seam, not upper/lower closure; no base cap is supplied => OPEN_BASE.',
      'normal_diagnostic':'Adjacent nondegenerate normals with negative dot product; positive XY oriented triangle areas plus simple projected boundary certify a piecewise-linear single-valued graph.',
      'discontinuity_diagnostic':'Native edge length greater than 20 times the median positive edge length, diagnostic only (mesh is intentionally nonuniform).',
      'hard_timeout_s':120,'max_consecutive_timeouts':3,'workers':0,
      'pilot_design_indices':[1,2,3], 'scan_order':'stable ascending pareto_front[:,0], then original row',
      'figure_selection':'All INVALID; at least three VALID if available; at least one AMBIGUOUS if available.',
      'limits':'Floating-point narrow phase, not exact arithmetic. Conclusions apply to frozen native discretization, not unobserved continuous surfaces or fabrication.'}
    dump(OUT/'classification_rules.json',rules)
    source=Path('D:/mat/.code/wave_rider_optimization_final.mat')
    mat=loadmat(source)
    z=np.asarray(mat['design_vars']); pf=np.asarray(mat['pareto_front'])
    assert z.shape==(52,6) and pf.shape==(52,2)
    optimizer=Path('D:/mat/.code/surrogate_assisted_ga_wave_rider_6.m')
    code=optimizer.read_text(encoding='utf-8-sig')
    ev=code.split('function [success, L_D, vol_frac] = evaluatePoint(x)',1)[1].split('function hv',1)[0]
    fixed={k:float(re.search(r'\b'+k+r'\s*=\s*([-+\d.]+)\s*;',ev)[1]) for k in ['Ma','c1','c2','AOA']}
    mapping={k:int(re.search(r'\b'+k+r'\s*=\s*x\((\d)\)',ev)[1]) for k in ['angle_shock_a1','angle_shock_a2','angle_shock_b1','angle_shock_b2','d1','a_xishu_1']}
    spec=json.loads((ROOT/'results/v5_teacher/design_spec.json').read_text(encoding='utf-8-sig'))
    assert fixed['Ma']==spec['Ma_design'] and fixed['c1']==spec['c1'] and fixed['c2']==spec['c2'] and fixed['AOA']==spec['AOA_design_prog_deg']
    bounds={key:[float(v) for v in re.search(r'\b'+key+r'\s*=\s*\[([^]]+)\]',code)[1].replace(',',' ').split()] for key in ['lb','ub']}
    candidates=[]
    for root in [Path('D:/mat'),ROOT]:
        for p in root.rglob('wave_rider_optimization_final.mat'):
            m=loadmat(p); candidates.append({'path':str(p),'sha256':sha(p),'fields':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in m.items() if not k.startswith('__')}})
    active={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'teacher_matlab/active').glob('*.m')}
    original={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'teacher_matlab/original').glob('*.m')}
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'source_path':str(source),'source_sha256':sha(source),'same_name_search_roots':['D:/mat',str(ROOT)],'same_name_matches':candidates,'fields':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in mat.items() if not k.startswith('__')},'parameter_columns':[{'column_1based':i+1,'name':p,'min':float(z[:,i].min()),'max':float(z[:,i].max()),'source_bounds':[bounds['lb'][i],bounds['ub'][i]]} for i,p in enumerate(PARAMS)],'fixed_parameters':fixed,'mapping_from_evaluatePoint':mapping,'fixed_source':str(optimizer),'optimizer_sha256':sha(optimizer),'rules_sha256':sha(OUT/'classification_rules.json'),'active_source_hashes':active,'original_source_hashes':original,'git_head':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),'old_selected_design_spec':spec,'old_objectives':'Stored pareto_front only, never recomputed; columns L_D and Vol_eff confirmed from evaluatePoint and objective construction.'}
    dump(OUT/'source_manifest.json',manifest)
    jobs=[]
    for idx,row in enumerate(np.argsort(pf[:,0],kind='stable')):
        a1,a2,b1,b2,d1,ax=map(float,z[row])
        inputs=[fixed['Ma'],a1,a2,b1,b2,fixed['c1'],fixed['c2'],d1,fixed['AOA'],ax]
        jobs.append({'design_index':idx+1,'original_row_index':int(row)+1,'original_row_index_0based':int(row),'parameters':dict(zip(PARAMS,map(float,z[row]))),'inputs10':inputs,'stored_L_D':float(pf[row,0]),'stored_Vol_eff':float(pf[row,1]),'case_dir':str(OUT/'cases'/f'design_{idx+1:02d}')})
    dump(OUT/'jobs.json',jobs)
    print(json.dumps({'shape':list(z.shape),'fixed':fixed,'protected_files':len(snap['files']),'rules_sha256':manifest['rules_sha256'],'selected45':jobs[44]},ensure_ascii=False))
if __name__=='__main__': main()

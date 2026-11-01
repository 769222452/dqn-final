"""Real Teacher reproducibility followed by 12 saved new candidates, audited individually."""
from __future__ import annotations
from exp_common import *
from runtime import Worker, CampaignLock, StopRequested
from initialize import initialize

def make_case(x,phase,index,strategy,**kw):
    x=np.asarray(x,dtype=float); full=[x[0],*x[1:5],1.0331,1.0331,x[5],-4.,x[6]]
    key=hashlib.sha256(np.asarray(full,dtype='<f8').tobytes()).hexdigest()[:12]
    gid=f'EXP_{index:05d}_{key}'
    return {'geometry_id':gid,'inputs10':full,**dict(zip(INPUTS,full)),'phase':phase,'attempt_index':index,'strategy':strategy,'case_dir':(OUT/'cases'/gid).as_posix(),'mat_path':(OUT/'cases'/gid/'native_geometry.mat').relative_to(ROOT).as_posix(),**kw}

def process_case(worker,c,known=None):
    d=Path(c['case_dir']); record=d/'record.json'
    if record.exists(): return load(record)
    worker.generate(c)
    scan=audit_module(); job={'design_index':c['attempt_index'],'original_row_index':None,'parameters':{k:c[k] for k in PARAMS},'inputs10':c['inputs10'],'case_dir':d.as_posix(),'stored_L_D':None,'stored_Vol_eff':None}
    if (d/'audit.json').exists(): au=load(d/'audit.json')
    else: au=scan.audit(job); dump(d/'audit.json',au)
    generation=load(d/'generation_status.json')
    r={**c,'geometry_validity':au['classification'],'geometry_hash':au.get('geometry_hash'),'failure_reason':au['failure_reason'],'volume_ratio_G42':None,'thickness_ratio_G42':None,'duplicate_of':'','coordinate_hash':None,'generation_status':generation['status'],'execution_interrupted':bool(generation.get('interrupted',False))}
    if au.get('generation_status')=='GENERATED' and not any(au.get('numeric_checks',{}).get(k,1) for k in ['nan_count','inf_count','complex_count']):
        met=closed_metrics(d/'native_geometry.mat'); gm=load(OUT/'baseline/G42_metrics.json')
        r.update(met); r.update(volume_ratio_G42=met['closed_volume_full']/gm['closed_volume_full'],thickness_ratio_G42=met['grid_thickness_max']/gm['grid_thickness_max'])
        r['coordinate_hash']=coordinate_hash(d/'native_geometry.mat'); dump(d/'metrics.json',met)
        if known is None: known=pool().dropna(subset=['coordinate_hash']).set_index('coordinate_hash').geometry_id.to_dict()
        r['duplicate_of']=known.get(r['coordinate_hash'],'')
    r['hard_gate_pass']=bool(r['geometry_validity']=='VALID' and r['volume_ratio_G42'] is not None and r['volume_ratio_G42']>=.85 and r['thickness_ratio_G42']>=.5 and not r['duplicate_of'])
    dump(d/'geometry_checkpoint.json',r)
    if r['hard_gate_pass']:
        data=worker.evaluate(c['geometry_id'],ROOT/c['mat_path'],frozen_pairs(),d/'frozen49.csv')
        r['aero_valid_points']=int((true(data.model_valid)&np.isfinite(data.CL)&np.isfinite(data.CD_pressure)&data.CD_pressure.gt(0)).sum())
        r['eligible_points']=int((true(data.model_valid)&data.CL.ge(CLMIN)&data.CD_pressure.gt(0)).sum())
    r['completed_utc']=utc(); dump(record,r)
    event('CANDIDATE_COMMITTED',candidate=c['geometry_id'],phase=c['phase'],attempt_index=c['attempt_index'],classification=r['geometry_validity'],volume_ratio=r['volume_ratio_G42'],thickness_ratio=r['thickness_ratio_G42'],duplicate=r['duplicate_of'],eligible_points=r.get('eligible_points',0))
    return r

def reproducibility(worker):
    p=old_pool().set_index('geometry_id'); results=[]
    for gid in ['G42',*AB]:
        dest=OUT/'reproducibility'/gid; dest.mkdir(parents=True,exist_ok=True)
        if (dest/'verification.json').exists():
            r=load(dest/'verification.json')
            if not r['passed']: raise RuntimeError(f'Reproducibility failed: {gid}')
            results.append(r); continue
        row=p.loc[gid]; inputs=[float(row[k]) for k in INPUTS]
        c={'geometry_id':'REPRO_'+gid,'phase':'reproducibility','case_dir':dest.as_posix(),'inputs10':inputs}
        worker.generate(c)
        scan=audit_module(); au=scan.audit({'design_index':0,'original_row_index':None,'parameters':{k:row[k] for k in PARAMS},'inputs10':inputs,'case_dir':dest.as_posix(),'stored_L_D':None,'stored_Vol_eff':None})
        dump(dest/'audit.json',au)
        old=surfaces(ROOT/row.mat_path); new=surfaces(dest/'native_geometry.mat')
        delta=max(float(np.max(np.abs(a-b))) for a,b in zip(old,new))
        results_p2=worker.evaluate(gid,dest/'native_geometry.mat',[(5.,5.),(6.5,7.),(8.,10.)],dest/'p2_reproduction.csv')
        frozen=pd.read_csv(OUT/'baseline/frozen49.csv'); matched=results_p2.merge(frozen,on=['geometry_id','M_flight','alpha_deg'],suffixes=('_new','_old'))
        aero_delta=max(float(np.max(abs(matched[k+'_new']-matched[k+'_old']))) for k in ['CL','CD_pressure'])
        r={'geometry_id':gid,'max_coordinate_abs_difference':delta,'audit_class':au['classification'],'native_hash_exact_match':au.get('geometry_hash')==row.geometry_hash,'max_aero_abs_difference':aero_delta,'passed':au['classification']=='VALID' and delta<1e-10 and aero_delta<1e-10}
        dump(dest/'verification.json',r); results.append(r); event('REPRODUCIBILITY',**r)
        if not r['passed']: raise RuntimeError(f'Reproducibility failed: {gid}')
    dump(OUT/'reproducibility/summary.json',{'passed':all(x['passed'] for x in results),'teacher_calls':len(results),'checks':results})

def preflight(worker):
    path=OUT/'preflight/candidates.json'
    if not path.exists():
        p=old_pool(); ids=['G42',*AB,'S2_MD8_b38c94c34e','S2_MD9.898_5e403aae01','S2_MD11.5_bf14cea04c']
        seeds=p.set_index('geometry_id').loc[ids]
        rng=np.random.default_rng(load(OUT/'config.json')['random_seed']); cases=[]
        for j in range(12):
            seed=seeds.iloc[j%len(seeds)]; x=seed[FEATURES].to_numpy(float)
            scales=np.array([.07,.04,.04,.04,.04,.0015,.0015])*(1 if j<6 else 2)
            x=np.clip(x+rng.normal(size=7)*scales,LOWER,UPPER)
            cases.append(make_case(x,'preflight',j+1,'preflight_local',parent_geometry=seed.name))
        dump(path,cases); dump(OUT/'preflight/rng_after.json',rng.bit_generator.state)
    cases=load(path)
    for c in cases:
        if (OUT/'STOP_REQUESTED').exists(): raise StopRequested()
        process_case(worker,c)
    rec=[load(Path(c['case_dir'])/'record.json') for c in cases]
    nvalid=sum(r['geometry_validity']=='VALID' for r in rec); ngate=sum(r['hard_gate_pass'] for r in rec)
    np2=sum(r.get('eligible_points',0)>0 for r in rec)
    report={'teacher_attempts':12,'native_VALID':nvalid,'unique_hard_gate_pass':ngate,'with_eligible_P2_points':np2,'passed':nvalid>0 and ngate>0 and np2>0,'criterion':'All 12 real attempts accounted for; at least one unique geometry passes original audit, hard gates and P2, plus all known-geometry reproduction checks pass. Invalid proposals are retained as research outcomes.'}
    dump(OUT/'preflight/summary.json',report); event('PREFLIGHT_COMPLETE',**report)
    if not report['passed']: raise RuntimeError('Preflight produced no usable real candidate')

def main():
    with CampaignLock():
        initialize(); verify_sources(); w=Worker()
        try: reproducibility(w); preflight(w)
        finally: w.close()
if __name__=='__main__': main()

"""Freeze real input evidence and derive the mission domain from successful saved trajectories."""
from __future__ import annotations
import yaml
from exp_common import *

def mission():
    cfg=yaml.safe_load((ROOT/'configs/mission_1200s.yaml').read_text(encoding='utf-8'))
    trajectories=[]; evidence=[]
    paths=sorted((ROOT/'results/fixed_geometry_baselines').glob('*/trajectory_0p5s.csv'))
    for path in paths:
        meta=load(path.parent/'best.json')
        if meta.get('outcome')!='SUCCESS': continue
        df=pd.read_csv(path)
        if abs(df.time_s.iloc[0])>1e-9 or abs(df.time_s.iloc[-1]+.5-1200)>1e-9: raise ValueError(f'Trajectory duration: {path}')
        df['alpha_deg']=np.degrees(df.alpha_rad); df['trajectory_id']=path.parent.name
        df['dt_s']=np.diff(np.r_[df.time_s.to_numpy(),1200.])
        trajectories.append(df[['trajectory_id','time_s','dt_s','Mach','alpha_deg','h_m','V_mps','CL_actual','q_pa']])
        evidence.append({'trajectory_id':path.parent.name,'path':path.relative_to(ROOT),'sha256':sha(path),'success_evidence':path.parent/'best.json','success_sha256':sha(path.parent/'best.json'),'model':'historical V2 pressure-only'})
    # This additional verified 1200 s flight reaches M~3.8 and must not be silently discarded.
    path=ROOT/'results/v5_long_horizon_mach_diagnostic_1200s/trajectory_0p5s.csv'
    meta=load(path.parent/'run_summary.json')
    if meta.get('mission_success_existing_semantics') and meta['t_end_s']==1200:
        d=pd.read_csv(path).rename(columns={'time':'time_s','h':'h_m','V':'V_mps','CL':'CL_actual','q':'q_pa'})
        d['dt_s']=np.diff(np.r_[d.time_s.to_numpy(),1200.]); d=d[d.dt_s>0].copy()
        d['trajectory_id']='historical_G42_1200s'; trajectories.append(d[trajectories[0].columns])
        evidence.append({'trajectory_id':'historical_G42_1200s','path':path.relative_to(ROOT),'sha256':sha(path),'success_evidence':path.parent/'run_summary.json','success_sha256':sha(path.parent/'run_summary.json'),'model':'historical Teacher pressure-only'})
    df=pd.concat(trajectories,ignore_index=True)
    if not np.isfinite(df.select_dtypes('number')).all().all(): raise ValueError('Nonfinite mission samples')
    if not all(abs(x.dt_s.sum()-1200)<1e-9 for _,x in df.groupby('trajectory_id')): raise ValueError('Incomplete time weights')
    support_m=cfg['model']['support_mach']; support_a=cfg['model']['support_alpha_deg']
    from support_evidence import inspect_support
    existing=inspect_support(); dump(OUT/'mission/support_evidence.json',existing)
    support_m=[max(support_m[0],existing['P2_table_M_support'][0]),min(support_m[1],existing['P2_table_M_support'][1])]
    support_a=[max(support_a[0],existing['P2_table_alpha_support'][0]),min(support_a[1],existing['P2_table_alpha_support'][1])]
    # 0.25 Mach and one degree margins, rounded outward, clipped only to documented model support.
    mlo=max(support_m[0],np.floor((df.Mach.min()-.25)*4)/4)
    mhi=min(support_m[1],np.ceil((df.Mach.max()+.25)*4)/4)
    alo=max(support_a[0],np.floor(df.alpha_deg.min()-1))
    ahi=min(support_a[1],np.ceil(df.alpha_deg.max()+1))
    mbase=np.arange(mlo,mhi+.001,.5); abase=np.arange(alo,ahi+.001,1.)
    mq=np.quantile(df.Mach,[.05,.95]); aq=np.quantile(df.alpha_deg,[.05,.95])
    mdense=np.arange(np.floor(mq[0]*4)/4,np.ceil(mq[1]*4)/4+.001,.25)
    adense=np.arange(np.floor(aq[0]*2)/2,np.ceil(aq[1]*2)/2+.001,.5)
    ms=np.unique(np.r_[mbase,mdense,FROZEN_M,mhi]); aa=np.unique(np.r_[abase,adense,FROZEN_A,ahi])
    ms=ms[(ms>=mlo)&(ms<=mhi)]; aa=aa[(aa>=alo)&(aa<=ahi)]
    info={'trajectories':evidence,'samples':len(df),'total_weight_s':float(df.dt_s.sum()),
          'envelope':{k:{'min':float(df[k].min()),'p05':float(df[k].quantile(.05)),'median':float(df[k].median()),'p95':float(df[k].quantile(.95)),'max':float(df[k].max())} for k in ['Mach','alpha_deg','CL_actual','h_m','V_mps','q_pa']},
          'M_grid':ms,'alpha_grid':aa,'support_M':support_m,'support_alpha':support_a,
          'support_evidence':'mission/support_evidence.json; intersection of mission config and existing P2 table support; candidate-specific validity remains mandatory',
          'margin_clipped':{'Mach_low':mlo>df.Mach.min()-.25,'Mach_high':mhi<df.Mach.max()+.25,'alpha_low':alo>df.alpha_deg.min()-1,'alpha_high':ahi<df.alpha_deg.max()+1},
          'reference_trajectory':'V2_MD15_007','integration':'left-endpoint saved 0.5 s controls/states, all 2400 intervals = 1200 s; pressure drag impulse N s, not drag work or range gain',
          'CL_targets':np.unique(np.r_[CLTARGETS,df.CL_actual.quantile([.25,.5,.75]).to_numpy()])}
    csv(OUT/'mission/samples.csv',df); dump(OUT/'mission/domain.json',info)
    ref=df[df.trajectory_id.eq(info['reference_trajectory'])]; csv(OUT/'mission/reference.csv',ref)
    mi=np.argmin(abs(df.Mach.to_numpy()[:,None]-ms),axis=1)
    ai=np.argmin(abs(df.alpha_deg.to_numpy()[:,None]-aa),axis=1)
    hist=pd.DataFrame({'M_flight':ms[mi],'alpha_deg':aa[ai],'weight_s':df.dt_s})
    csv(OUT/'mission/frequency.csv',hist.groupby(['M_flight','alpha_deg'],as_index=False).weight_s.sum())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    h=axes[0].hist2d(df.Mach,df.alpha_deg,bins=[np.arange(3.5,8.76,.25),np.arange(-2,12.51,.5)],weights=df.dt_s,cmap='viridis')
    fig.colorbar(h[3],ax=axes[0],label='Accumulated flight time [s]'); axes[0].set(xlabel='Flight Mach',ylabel='Angle of attack [deg]')
    for name,d in df.groupby('trajectory_id'): axes[1].plot(d.time_s,d.Mach,label=name)
    axes[1].set(xlabel='Time [s]',ylabel='Flight Mach'); axes[1].legend(fontsize=7)
    (OUT/'figures').mkdir(exist_ok=True); fig.savefig(OUT/'figures/mission_domain.png',dpi=160); plt.close(fig)
    return info

def initialize():
    if (OUT/'baseline/initialized.json').exists(): verify_sources(); return
    for d in ['baseline','mission','cases','reproducibility','preflight','aero','metrics','reports','models','figures','workers']: (OUT/d).mkdir(parents=True,exist_ok=True)
    hashes=protect_manifest(); dump(OUT/'baseline/source_hashes.json',hashes)
    p=pd.read_csv(OLD/'screening/candidate_pool.csv'); assert len(p)==453 and p.geometry_hash.nunique()==453
    scan=audit_module(); g42=p[p.geometry_id.eq('G42')].iloc[0]; gm=closed_metrics(ROOT/g42.mat_path)
    for i,r in p.iterrows():
        mp=ROOT/r.mat_path; au=load(mp.with_name('audit.json')); lo,up=surfaces(mp)
        if au['classification']!='VALID' or scan.native_hash(lo,up,au['inputs10'])!=r.geometry_hash: raise RuntimeError(f'Identity/audit mismatch: {r.geometry_id}')
        met=closed_metrics(mp)
        if not np.isclose(met['closed_volume_full']/gm['closed_volume_full'],r.volume_ratio_G42,rtol=1e-10,atol=1e-12): raise ValueError(f'Volume mismatch: {r.geometry_id}')
        if not np.isclose(met['grid_thickness_max']/gm['grid_thickness_max'],r.thickness_ratio_G42,rtol=1e-10): raise ValueError('Thickness mismatch')
        for name,val in zip(INPUTS,au['inputs10']): p.loc[i,name]=val
        p.loc[i,'coordinate_hash']=coordinate_hash(mp); p.loc[i,'mat_sha256']=sha(mp)
    p['phase']='baseline'; p['duplicate_of']=''
    # Identity audit includes inputs; coordinate-only identity additionally catches relabelled geometry.
    dup=p.duplicated('coordinate_hash'); p.loc[dup,'duplicate_of']=p.loc[dup,'coordinate_hash'].map(p.drop_duplicates('coordinate_hash').set_index('coordinate_hash').geometry_id)
    csv(OUT/'baseline/candidate_pool.csv',p); dump(OUT/'baseline/G42_metrics.json',gm)
    raw=pd.read_csv(OLD/'work/pool_results.csv'); assert len(raw)==453*49
    assert raw.groupby('geometry_id').size().eq(49).all()
    csv(OUT/'baseline/frozen49.csv',raw)
    counts=p.assign(design_bin=np.floor(p.Ma_design).astype(int),hard_gate=gate(p)).groupby('design_bin').agg(existing=('geometry_id','size'),hard_gate_pass=('hard_gate','sum'))
    csv(OUT/'baseline/design_mach_coverage.csv',counts.reset_index())
    domain=mission()
    config={'schema':'teacher-space-expansion/1','created_utc':utc(),'batch_size':100,'max_teacher_attempts':5000,'max_wall_time_hours':48,'matlab_concurrency':1,'checkpoint_every_candidate':True,'random_seed':20261014,
            'lower':LOWER,'upper':UPPER,'features':FEATURES,'bounds_source':'tools/v6_mission_geometry_stage2/common.py FROZEN_LOWER/EXT_UPPER; design Mach observed native VALID [4,15]; no new boundary extension',
            'fixed':{'c1':1.0331,'c2':1.0331,'AOA_design_prog':-4.},'ranking':{'CL_min':CLMIN,'volume_min':.85,'thickness_min':.5,'Sref':SREF,'Lref':5.,'mode':'P2','friction':'none','base_pressure_ratio':None,'metric':'raw CL/CD_pressure, original tie break CD then geometry_id'},
            'formal':{'core_effective_winners':6,'AB_share_max_exclusive':.60,'new_effective_winners':3,'adjacent_margin_min':.005,'equal_lift_effective_winners':4},'preflight_attempts':12}
    if not (OUT/'config.json').exists(): dump(OUT/'config.json',config)
    dump(OUT/'baseline/initialized.json',{'utc':utc(),'old_geometries':len(p),'hard_gate_pass':int(gate(p).sum()),'coordinate_duplicates':int(dup.sum()),'frozen_points':len(raw),'sources_verified':True})
    event('INITIALIZED',old_geometries=len(p),hard_gate_pass=int(gate(p).sum()),domain_M=[min(domain['M_grid']),max(domain['M_grid'])],domain_alpha=[min(domain['alpha_grid']),max(domain['alpha_grid'])])

if __name__=='__main__': initialize()

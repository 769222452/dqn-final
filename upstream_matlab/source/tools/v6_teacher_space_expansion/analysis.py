"""Physical rankings, independent CORE42/equal-lift metrics, and research outputs."""
from __future__ import annotations
from exp_common import *
from evaluation import all_aero,usable,equal_table,oracle

def rank_fixed(data,p):
    d=data.merge(p[['geometry_id','geometry_hash','phase','geometry_validity','volume_ratio_G42','thickness_ratio_G42','duplicate_of']],on='geometry_id',validate='many_to_one')
    d['eligible']=usable(d)&gate(d)&d.CL.ge(CLMIN)
    d['L_over_D']=d.CL/d.CD_pressure
    d['rank']=np.nan
    order=d[d.eligible].sort_values(['M_flight','alpha_deg','L_over_D','CD_pressure','geometry_id'],ascending=[True,True,False,True,True])
    d.loc[order.index,'rank']=order.groupby(['M_flight','alpha_deg']).cumcount()+1
    return d

def winners(ranked):
    keys=['M_flight','alpha_deg']; one=ranked[ranked['rank'].eq(1)].copy()
    two=ranked[ranked['rank'].eq(2)][keys+['geometry_id','L_over_D']].rename(columns={'geometry_id':'runner_up','L_over_D':'runner_up_LD'})
    one=one.merge(two,on=keys,how='left'); one['delta_LD']=(one.L_over_D-one.runner_up_LD)/one.runner_up_LD
    one['advantage_class']=pd.cut(one.delta_LD,[-np.inf,.001,.005,.01,np.inf],labels=['extremely_weak','weak','noticeable_low_order','priority_CFD'],right=False)
    return one

def adjacent_ids(w,axes=('M_flight','alpha_deg'),margin='delta_LD',threshold=.005,axis_values=None):
    if w.empty: return []
    vals=[sorted(w[x].unique()) for x in axes] if axis_values is None else axis_values
    coords={x:{v:i for i,v in enumerate(vs)} for x,vs in zip(axes,vals)}; selected=[]
    for gid,g in w[w[margin].ge(threshold)].groupby('geometry_id'):
        pts={(coords[axes[0]][r[axes[0]]],coords[axes[1]][r[axes[1]]]) for _,r in g.iterrows()}
        if any((i+di,j+dj) in pts for i,j in pts for di,dj in [(1,0),(-1,0),(0,1),(0,-1)]): selected.append(gid)
    return selected

def distribution(w,weight=None):
    if w.empty: return {'unique_top1':0,'AB_share':None,'N_eff':None,'shares':{}}
    counts=w.groupby('geometry_id')[weight].sum() if weight else w.geometry_id.value_counts()
    shares=counts/counts.sum()
    return {'unique_top1':len(shares),'AB_share':float(shares.reindex(AB,fill_value=0).sum()),'N_eff':float(1/np.sum(shares**2)),'shares':shares.to_dict()}

def rank_equal(eq):
    eq=eq.copy(); eq['rank']=np.nan
    ok=eq[eq.status.eq('OK')].sort_values(['M_flight','CL_target','CD_at_target','geometry_id'])
    eq.loc[ok.index,'rank']=ok.groupby(['M_flight','CL_target']).cumcount()+1
    top=eq[eq['rank'].eq(1)].copy()
    second=eq[eq['rank'].eq(2)][['M_flight','CL_target','geometry_id','CD_at_target']].rename(columns={'geometry_id':'runner_up','CD_at_target':'runner_up_CD'})
    top=top.merge(second,on=['M_flight','CL_target'],how='left')
    top['delta_CD']=(top.runner_up_CD-top.CD_at_target)/top.runner_up_CD
    return eq,top

def summarize(tag,with_oracle=False):
    p=pool(); data=all_aero(); r=rank_fixed(data,p)
    fixed=r[r.M_flight.isin(FROZEN_M)&r.alpha_deg.isin(FROZEN_A)].copy()
    win=winners(fixed); core=win[win.alpha_deg.ge(5)].copy()
    # Full frozen 49 remains complete for every eligible new geometry.
    expected=p[gate(p)].geometry_id
    counts=fixed[fixed.geometry_id.isin(expected)].groupby('geometry_id').size()
    if len(counts)!=len(expected) or not counts.eq(49).all(): raise RuntimeError('Incomplete frozen49 comparison')
    old=rank_fixed(data[data.geometry_id.isin(p[p.phase.eq('baseline')].geometry_id)],p[p.phase.eq('baseline')])
    oldwin=winners(old[old.M_flight.isin(FROZEN_M)&old.alpha_deg.isin(FROZEN_A)])
    core=core.merge(oldwin[['M_flight','alpha_deg','L_over_D']].rename(columns={'L_over_D':'old_best_LD'}),on=['M_flight','alpha_deg'])
    core['improvement_vs_old']=(core.L_over_D-core.old_best_LD)/core.old_best_LD
    eq,eqw=rank_equal(equal_table(data,p[gate(p)],machs=load(OUT/'mission/domain.json')['M_grid']))
    baseeq,baseeqw=rank_equal(equal_table(data,p[p.phase.eq('baseline')&gate(p)],machs=load(OUT/'mission/domain.json')['M_grid']))
    csv(OUT/f'metrics/{tag}_baseline_equal_lift.csv',baseeq)
    nominal=adjacent_ids(core,axis_values=[list(FROZEN_M),list(np.arange(5.,10.01))])
    domain=load(OUT/'mission/domain.json')
    eq_axes=[domain['M_grid'],domain['CL_targets']]
    eqnom=adjacent_ids(eqw,axes=('M_flight','CL_target'),margin='delta_CD',axis_values=eq_axes)
    verified_path=OUT/'verification/latest.json'; verification=load(verified_path) if verified_path.exists() else {}
    current_ids=set(p[gate(p)].geometry_id)
    verified_current=verification.get('pool_ids_sha256')==hashlib.sha256('\n'.join(sorted(current_ids)).encode()).hexdigest()
    verified=verification.get('stable_core_winners',[]) if verified_current else []
    eqverified=verification.get('stable_equal_winners',[]) if verified_current else []
    all_nominal_checked=verified_current and verification.get('all_new_nominal_core_checked',False) and verification.get('all_new_nominal_equal_checked',False)
    newids=set(p[p.phase.ne('baseline')].geometry_id)
    # Legacy selected points and an in-progress refinement may cover only a
    # subset of competitors. Published task-domain rankings use the shared grid.
    mission=r[r.M_flight.isin(domain['M_grid'])&r.alpha_deg.isin(domain['alpha_grid'])].copy()
    allwin=winners(mission); freq=pd.read_csv(OUT/'mission/frequency.csv')
    weighted=allwin.merge(freq,on=['M_flight','alpha_deg'])
    coverage=float(weighted.weight_s.sum()/freq.weight_s.sum()) if len(weighted) else 0.
    stats={'utc':utc(),'tag':tag,'committed_teacher_candidates':int(p.phase.ne('baseline').sum()),
           'native_VALID_new':int((p.phase.ne('baseline')&p.geometry_validity.eq('VALID')).sum()),'new_unique_hard_gate_pass':int((p.phase.ne('baseline')&gate(p)).sum()),
           'total_registered':len(p),'baseline_CORE42':distribution(oldwin[oldwin.alpha_deg.ge(5)]),'CORE42':distribution(core),
           'core_winners_with_adjacent_0p5pct_margins':nominal,'core_effective_verified':verified,'new_effective_verified':sorted(set(verified)&newids),
           'new_nominal_core_winners':sorted(set(core.geometry_id)&newids),'equal_lift':distribution(eqw),'baseline_equal_lift':distribution(baseeqw),'equal_lift_adjacent_0p5pct_winners':eqnom,'equal_lift_verified_winners':eqverified,
           'new_equal_lift_verified_winners':sorted(set(eqverified)&newids),'new_equal_lift_nominal_winners':sorted(set(eqw.geometry_id)&newids),
           'weighted_task_distribution':distribution(weighted,'weight_s'),'weighted_grid_coverage_fraction':coverage,
           'max_core_improvement_vs_old':float(core.improvement_vs_old.max()),'mean_core_improvement_vs_old':float(core.improvement_vs_old.mean()),
           'near_ties_core_below_0p5pct':int(core.delta_LD.lt(.005).sum()),'verification_current':verified_current,'all_new_nominal_diagnostics_complete':bool(all_nominal_checked),
           'formal_target_reached':all_nominal_checked and len(verified)>=6 and distribution(core)['AB_share']<.60 and len(set(verified)&newids)>=3 and len(eqverified)>=4 and len(set(eqverified)&newids)>=2}
    stats['nominal_target_ready_for_verification']=len(nominal)>=6 and stats['CORE42']['AB_share']<.60 and len(set(nominal)&newids)>=3 and len(eqnom)>=4 and len(set(eqnom)&newids)>=2
    from evaluation import mission_pairs
    need=set(mission_pairs()); completeness={gid:need.issubset(set(zip(g.M_flight,g.alpha_deg))) for gid,g in data.groupby('geometry_id')}
    stats['mission_grid_complete_geometries']=sum(completeness.get(gid,False) for gid in expected)
    stats['mission_grid_required_geometries']=len(expected)
    stats['equal_and_task_comparison_complete']=stats['mission_grid_complete_geometries']==len(expected)
    stats['formal_target_reached'] &= stats['equal_and_task_comparison_complete']
    stats['nominal_target_ready_for_verification'] &= stats['equal_and_task_comparison_complete']
    if stats['equal_and_task_comparison_complete']:
        from sensitivity import task_grid_sensitivity
        stats['grid_sensitivity']=task_grid_sensitivity(tag,data,p[gate(p)],core,allwin,eqw)
    stats['milestone_nominal']=stats['CORE42']['unique_top1']>=4 and stats['CORE42']['AB_share']<.75 and len(set(adjacent_ids(core,threshold=0,axis_values=[list(FROZEN_M),list(np.arange(5.,10.01))]))-set(AB))>=2
    for name,d in [('frozen49_all_scores',fixed),('frozen49_top1',win),('CORE42_top1',core),('equal_lift',eq),('equal_lift_top1',eqw),('mission_all_scores',mission),('mission_top1',allwin)]: csv(OUT/f'metrics/{tag}_{name}.csv',d)
    csv(OUT/'merged_candidate_inventory.csv',p)
    if with_oracle: stats['oracle']=oracle(data,p[gate(p)],tag)
    dump(OUT/f'metrics/{tag}_summary.json',stats); dump(OUT/'metrics/latest.json',stats)
    make_figures(tag,win,eqw,core,oldwin)
    return stats

def make_figures(tag,win,eqw,core,oldwin):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    for name,d,x,y in [('Mach_alpha',win,'M_flight','alpha_deg'),('Mach_CL',eqw,'M_flight','CL_target')]:
        if d.empty: continue
        gids=sorted(d.geometry_id.unique()); d=d.copy(); d['color']=d.geometry_id.map({g:i for i,g in enumerate(gids)})
        fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
        for i,g in enumerate(gids):
            s=d[d.geometry_id.eq(g)]; ax.scatter(s[x],s[y],s=70,marker='s',label=g,color=plt.cm.tab20(i%20))
        ax.set(xlabel='Flight Mach',ylabel='Angle of attack [deg]' if y=='alpha_deg' else 'Required CL',title='Raw pressure-only P2 winner; near ties remain nominal')
        ax.legend(fontsize=6,bbox_to_anchor=(1.02,1),loc='upper left'); fig.savefig(OUT/f'figures/{tag}_{name}.png',dpi=160); plt.close(fig)
    before=oldwin[oldwin.alpha_deg.ge(5)].geometry_id.value_counts()/42
    after=core.geometry_id.value_counts()/42; gids=sorted(set(before.index)|set(after.index))
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained'); x=np.arange(len(gids))
    ax.bar(x-.2,before.reindex(gids,fill_value=0),.4,label='453 baseline'); ax.bar(x+.2,after.reindex(gids,fill_value=0),.4,label='Current merged pool')
    ax.set_xticks(x,gids,rotation=30,ha='right',fontsize=7); ax.set_ylabel('CORE42 Top1 share'); ax.legend(); fig.savefig(OUT/f'figures/{tag}_top1_shares.png',dpi=160); plt.close(fig)

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--tag',default='current'); ap.add_argument('--oracle',action='store_true'); args=ap.parse_args()
    print(json.dumps(clean(summarize(args.tag,args.oracle)),ensure_ascii=False,indent=2))

"""Verify adjacent winning regions against every eligible competitor, with cached real P2."""
from __future__ import annotations
from exp_common import *
from evaluation import ensure_grid,all_aero,equal_table
from analysis import rank_fixed,winners,rank_equal,adjacent_ids


def witness_queues(table,axes,axis_values,margin,new_ids):
    """Strong adjacent pairs first; all new nominal winners receive a diagnostic.

    A geometry qualifies after one adjacent region is verified. Other winning
    regions are not implicitly certified. Failed pairs advance to the next pair.
    """
    coords=[{float(v):i for i,v in enumerate(values)} for values in axis_values]
    queues={}; provenance={}
    for gid,g in table.groupby('geometry_id'):
        by_cell={(coords[0][float(r[axes[0]])],coords[1][float(r[axes[1]])]):int(i) for i,r in g.iterrows()}
        pairs=[]
        for (i,j),a in by_cell.items():
            for neighbor in [(i+1,j),(i,j+1)]:
                if neighbor in by_cell:
                    b=by_cell[neighbor]; margins=table.loc[[a,b],margin]
                    score=float(margins.min()) if margins.notna().all() else -np.inf
                    pairs.append((score,float(margins.mean()),a,b))
        pairs.sort(key=lambda x:(-x[0],-x[1],x[2],x[3]))
        substantial=[(a,b) for score,_,a,b in pairs if score>=.005]
        if substantial:
            queues[gid]=substantial; kind='ADJACENT_0P5PCT_REGION'
        elif gid in new_ids:
            queues[gid]=[(pairs[0][2],pairs[0][3])] if pairs else [(int(g[margin].fillna(-np.inf).idxmax()),)]
            kind='NEW_NOMINAL_DIAGNOSTIC_ONLY'
        else:
            continue
        provenance[gid]={'kind':kind,'nominal_points':len(g),'substantial_pairs':len(substantial),'ordered_row_indices':queues[gid]}
    return queues,provenance


def neighborhood_plan(core,eq,core_indices,eq_indices,domain):
    mlo,mhi=min(domain['M_grid']),max(domain['M_grid'])
    alo,ahi=min(domain['alpha_grid']),max(domain['alpha_grid'])
    pairs=set(); cn={}; en={}; densea=np.arange(alo,ahi+.001,.25)
    for i in core_indices:
        r=core.loc[i]
        points=[(float(m),float(a)) for m in [r.M_flight-.125,r.M_flight,r.M_flight+.125] for a in [r.alpha_deg-.25,r.alpha_deg,r.alpha_deg+.25] if mlo<=m<=mhi and alo<=a<=ahi]
        cn[i]={'geometry_id':r.geometry_id,'M_flight':float(r.M_flight),'alpha_deg':float(r.alpha_deg),'points':points}
        pairs.update(points)
    for i in eq_indices:
        r=eq.loc[i]; nm=[float(m) for m in [r.M_flight-.125,r.M_flight,r.M_flight+.125] if mlo<=m<=mhi]
        en[i]={'geometry_id':r.geometry_id,'M_flight':float(r.M_flight),'CL_target':float(r.CL_target),'M_check':nm}
        pairs.update((m,float(a)) for m in nm for a in densea)
    return cn,en,pairs,densea


def check_neighborhood(lookup,gid,points,margin):
    evidence=[]
    for point in points:
        row={'coordinates':point,'rank_retained':False}
        if point in lookup.index:
            r=lookup.loc[point]
            row.update(winner=r.geometry_id,runner_up=r.runner_up,margin=float(r[margin]),rank_retained=bool(r.geometry_id==gid))
        evidence.append(row)
    margins=[r['margin'] for r in evidence if 'margin' in r and np.isfinite(r['margin'])]
    return {'stable_all_neighbors':bool(evidence) and all(r['rank_retained'] for r in evidence),'points_checked':len(evidence),'minimum_local_winner_margin':min(margins,default=None),'neighbor_evidence':evidence}


def verify_winners(worker,stats,check):
    p=pool(); eligible=p[gate(p)]; new_ids=set(p[p.phase.ne('baseline')].geometry_id)
    core=pd.read_csv(OUT/f"metrics/{stats['tag']}_CORE42_top1.csv",float_precision='round_trip').reset_index(drop=True)
    eq=pd.read_csv(OUT/f"metrics/{stats['tag']}_equal_lift_top1.csv",float_precision='round_trip').reset_index(drop=True)
    domain=load(OUT/'mission/domain.json'); core_axes=[list(FROZEN_M),list(np.arange(5.,10.01))]; eq_axes=[domain['M_grid'],domain['CL_targets']]
    cq,cp=witness_queues(core,('M_flight','alpha_deg'),core_axes,'delta_LD',new_ids)
    eqqueue,ep=witness_queues(eq,('M_flight','CL_target'),eq_axes,'delta_CD',new_ids)
    # Spend the search budget first on potentially effective regions. Near-tie
    # diagnostics remain required for final success, and are never promoted to
    # effective winners merely because their diagnostics have been deferred.
    full_nominal_diagnostics=bool(stats['nominal_target_ready_for_verification'])
    deferred=[[],[]]
    if not full_nominal_diagnostics:
        for k,(queues,provenance) in enumerate([(cq,cp),(eqqueue,ep)]):
            deferred[k]=sorted(g for g in queues if provenance[g]['kind']=='NEW_NOMINAL_DIAGNOSTIC_ONLY')
            for gid in deferred[k]: del queues[gid]
    pool_hash=hashlib.sha256('\n'.join(sorted(eligible.geometry_id)).encode()).hexdigest()
    folder=OUT/'verification'/pool_hash[:16]
    plan={'utc':utc(),'pool_ids_sha256':pool_hash,'all_pool_count':len(eligible),'minimum_original_margin':.005,'core':cp,'equal':ep,
          'verification_mode':'FULL_NOMINAL_DIAGNOSTICS' if full_nominal_diagnostics else 'QUALIFYING_REGIONS_DURING_SEARCH',
          'deferred_nominal_core_ids':deferred[0],'deferred_nominal_equal_ids':deferred[1],
          'scope':'Every selected point compares the complete eligible pool. Sequential adjacent region witnesses, strongest minimum original margin first; try other substantial pairs if needed. During search, near-tie nominal diagnostics may be deferred. Before formal success, every new nominal CORE42 and equal-CL winner must receive a local diagnostic. No claim about untested regions.'}
    dump(folder/'plan.json',plan); dump(OUT/'verification/plan.json',plan)
    positions=[{g:0 for g in q} for q in [cq,eqqueue]]; finished=[set(),set()]; records={}; eqrecords={}; round_no=0
    while True:
        selected=[{g:q[positions[k][g]] for g,q in queues.items() if g not in finished[k] and positions[k][g]<len(q)} for k,queues in enumerate([cq,eqqueue])]
        if not any(selected): break
        check(); round_no+=1
        ci=sorted({i for pair in selected[0].values() for i in pair}); ei=sorted({i for pair in selected[1].values() for i in pair})
        cn,en,pairs,densea=neighborhood_plan(core,eq,ci,ei,domain)
        dump(folder/f'round_{round_no:03d}_plan.json',{'utc':utc(),'selected_row_indices':selected,'core':cn,'equal':en,'query_points_per_geometry':len(pairs),'all_pool_count':len(eligible)})
        event('REFINEMENT_ROUND',round=round_no,core_regions=len(selected[0]),equal_regions=len(selected[1]),query_points_per_geometry=len(pairs),pool_count=len(eligible))
        ensure_grid(worker,eligible,sorted(pairs),'verification',check)
        data=all_aero(); refined=winners(rank_fixed(data,p)); lookup=refined.set_index(['M_flight','alpha_deg'])
        for i,n in cn.items():
            records[i]={**{k:v for k,v in n.items() if k!='points'},'source_row':i,**check_neighborhood(lookup,n['geometry_id'],n['points'],'delta_LD')}
        if en:
            ms=sorted({m for n in en.values() for m in n['M_check']})
            targets=sorted({c for n in en.values() for c in [n['CL_target']*.99,n['CL_target'],n['CL_target']*1.01] if c>=CLMIN})
            eqdense,eqwin=rank_equal(equal_table(data[data.alpha_deg.isin(densea)],eligible,ms,targets,alpha_grid=densea))
            elook=eqwin.set_index(['M_flight','CL_target'])
            csv(folder/f'round_{round_no:03d}_dense_equal_lift.csv',eqdense)
            for i,n in en.items():
                points=[(m,cl) for m in n['M_check'] for cl in [n['CL_target']*.99,n['CL_target'],n['CL_target']*1.01] if cl>=CLMIN]
                eqrecords[i]={**{k:v for k,v in n.items() if k!='M_check'},'source_row':i,**check_neighborhood(elook,n['geometry_id'],points,'delta_CD')}
        for k,results in enumerate([records,eqrecords]):
            for gid,indices in selected[k].items():
                if all(results[i]['stable_all_neighbors'] for i in indices): finished[k].add(gid)
                positions[k][gid]+=1
        dump(folder/'in_progress.json',{'utc':utc(),'round':round_no,'positions':positions,'finished':list(map(sorted,finished)),'core_results':list(records.values()),'equal_results':list(eqrecords.values())})
    corechecks=pd.DataFrame(records.values()); eqchecks=pd.DataFrame(eqrecords.values())
    stable_core=adjacent_ids(core.merge(corechecks,on=['geometry_id','M_flight','alpha_deg']).query('stable_all_neighbors'),axis_values=core_axes) if len(corechecks) else []
    stable_equal=adjacent_ids(eq.merge(eqchecks,on=['geometry_id','M_flight','CL_target']).query('stable_all_neighbors'),axes=('M_flight','CL_target'),margin='delta_CD',axis_values=eq_axes) if len(eqchecks) else []
    report={'utc':utc(),'pool_ids_sha256':pool_hash,'pool_count':len(eligible),'stable_core_winners':stable_core,'stable_equal_winners':stable_equal,
            'verification_mode':plan['verification_mode'],'deferred_nominal_core_ids':deferred[0],'deferred_nominal_equal_ids':deferred[1],
            'criterion':'At least two neighboring original points with >=0.5% raw margin over every eligible competitor; rank retained across local M/alpha or M/CL refinements, with refined margins separately disclosed',
            'scope':plan['scope'],'plan_directory':str(folder),'rounds':round_no,'all_new_nominal_core_checked':set(stats['new_nominal_core_winners']).issubset({r['geometry_id'] for r in records.values()}),
            'all_new_nominal_equal_checked':set(stats['new_equal_lift_nominal_winners']).issubset({r['geometry_id'] for r in eqrecords.values()}),
            'CL_solver':'PCHIP on complete valid contiguous alpha segments; no extrapolation','core_results':list(records.values()),'equal_results':list(eqrecords.values())}
    if full_nominal_diagnostics and (not report['all_new_nominal_core_checked'] or not report['all_new_nominal_equal_checked']): raise RuntimeError('New nominal winner refinement is incomplete')
    dump(folder/'completed.json',report); dump(OUT/'verification/latest.json',report)
    event('REFINEMENT_COMPLETE',core_stable=stable_core,equal_stable=stable_equal,rounds=round_no)
    return report

"""Unmodified P2 evaluations and conservative, bracketed equal-lift inversion."""
from __future__ import annotations
from scipy.interpolate import PchipInterpolator
from exp_common import *
from runtime import StopRequested, BudgetReached

KEY=['geometry_id','M_flight','alpha_deg']

def all_aero():
    pieces=[pd.read_csv(OUT/'baseline/frozen49.csv',float_precision='round_trip')]
    # Same frozen options and reference; these were computed by the same wedge_eval.
    old_selected=OLD/'work/selected_results.csv'
    if old_selected.exists(): pieces.append(pd.read_csv(old_selected,float_precision='round_trip'))
    for path in sorted((OUT/'cases').glob('*/frozen49.csv')): pieces.append(pd.read_csv(path,float_precision='round_trip'))
    for path in sorted((OUT/'aero').glob('*/*.csv')):
        if not path.name.endswith(('.queries.csv','.tmp.csv')): pieces.append(pd.read_csv(path,float_precision='round_trip'))
    d=pd.concat(pieces,ignore_index=True)
    # Only the baseline/new identity registry participates (old selected sometimes has aliases).
    d=d[d.geometry_id.isin(pool().geometry_id)]
    return d.drop_duplicates(KEY,keep='first')

def usable(d): return true(d.model_valid)&np.isfinite(d.CL)&np.isfinite(d.CD_pressure)&d.CD_pressure.gt(0)

def ensure_grid(worker,geometries,pairs,tag,check=lambda:None):
    have=all_aero(); lookup={gid:set(zip(g.M_flight,g.alpha_deg)) for gid,g in have.groupby('geometry_id')}
    for _,r in geometries.iterrows():
        check()
        missing=sorted(set(pairs)-lookup.get(r.geometry_id,set()))
        if not missing: continue
        signature=hashlib.sha256(json.dumps(missing).encode()).hexdigest()[:12]
        dest=OUT/'aero'/r.geometry_id/f'{tag}_{signature}.csv'
        worker.evaluate(r.geometry_id,ROOT/r.mat_path,missing,dest)
        event('P2_GRID_COMMITTED',geometry=r.geometry_id,tag=tag,new_points=len(missing))

def mission_pairs():
    domain=load(OUT/'mission/domain.json')
    return [(float(m),float(a)) for m in domain['M_grid'] for a in domain['alpha_grid']]

def equal_lift(g,target):
    """PCHIP CL(alpha) roots, then CD(alpha); never bridge an invalid row or extrapolate.

    Multiple valid lift branches are allowed; the lowest physical CD branch is used.
    The caller supplies the complete evaluated alpha grid, including invalid rows.
    """
    g=g.sort_values('alpha_deg').drop_duplicates('alpha_deg')
    a=g.alpha_deg.to_numpy(float); cl=g.CL.to_numpy(float); cd=g.CD_pressure.to_numpy(float)
    ok=usable(g).to_numpy(); indices=np.flatnonzero(ok); solutions=[]
    for ix in np.split(indices,np.where(np.diff(indices)>1)[0]+1):
        if not len(ix): continue
        exact=ix[np.isclose(cl[ix],target,rtol=0,atol=1e-13)]
        solutions += [(float(cd[k]),float(a[k])) for k in exact]
        if len(ix)<2 or target<cl[ix].min() or target>cl[ix].max(): continue
        lift=PchipInterpolator(a[ix],cl[ix],extrapolate=False)
        drag=PchipInterpolator(a[ix],cd[ix],extrapolate=False)
        for root in lift.solve(float(target),extrapolate=False):
            if np.isfinite(root) and a[ix[0]]<=root<=a[ix[-1]]:
                value=float(drag(root))
                if np.isfinite(value) and value>0: solutions.append((value,float(root)))
    if not solutions: return {'status':'INFEASIBLE_NO_VALID_BRACKET','CD_at_target':np.nan,'alpha_at_target':np.nan}
    drag,alpha=min(solutions)
    return {'status':'OK','CD_at_target':drag,'alpha_at_target':alpha}

def equal_table(data,geometries,machs=None,targets=None,alpha_grid=None):
    domain=load(OUT/'mission/domain.json'); targets=targets if targets is not None else domain['CL_targets']
    d=data[data.geometry_id.isin(geometries.geometry_id)].copy()
    # Use a common alpha grid so denser legacy tables cannot change only one contender's interpolation.
    alpha_grid=domain['alpha_grid'] if alpha_grid is None else alpha_grid
    d=d[d.alpha_deg.isin(alpha_grid)]
    if machs is not None: d=d[d.M_flight.isin(machs)]
    rows=[]
    for (gid,m),g in d.groupby(['geometry_id','M_flight'],sort=True):
        g=g.drop_duplicates('alpha_deg').set_index('alpha_deg').reindex(alpha_grid).reset_index()
        for cl in targets: rows.append({'geometry_id':gid,'M_flight':float(m),'CL_target':float(cl),**equal_lift(g,cl)})
    return pd.DataFrame(rows)

def mission_cd_for_geometry(g,ref):
    """Mach PCHIP of CL and CD at each alpha, then bracketed alpha inversion.

    Invalid/missing grid nodes split interpolation intervals in both dimensions.
    Full mission domain grid is required; exact saved 0.5s states retain their weights.
    """
    domain=load(OUT/'mission/domain.json'); ms=np.array(domain['M_grid']); aa=np.array(domain['alpha_grid'])
    grid=pd.MultiIndex.from_product([ms,aa],names=['M_flight','alpha_deg'])
    g=g.drop_duplicates(['M_flight','alpha_deg']).set_index(['M_flight','alpha_deg']).reindex(grid).reset_index()
    valid=usable(g).to_numpy().reshape(len(ms),len(aa))
    CL=g.CL.to_numpy().reshape(valid.shape); CD=g.CD_pressure.to_numpy().reshape(valid.shape)
    mt=ref.Mach.to_numpy(); clt=np.full((len(ref),len(aa)),np.nan); cdt=clt.copy()
    for j in range(len(aa)):
        ix=np.flatnonzero(valid[:,j])
        for run in np.split(ix,np.where(np.diff(ix)>1)[0]+1):
            if not len(run): continue
            if len(run)==1:
                take=np.isclose(mt,ms[run[0]],rtol=0,atol=1e-12)
                clt[take,j]=CL[run[0],j]; cdt[take,j]=CD[run[0],j]; continue
            take=(mt>=ms[run[0]])&(mt<=ms[run[-1]])
            clt[take,j]=PchipInterpolator(ms[run],CL[run,j],extrapolate=False)(mt[take])
            cdt[take,j]=PchipInterpolator(ms[run],CD[run,j],extrapolate=False)(mt[take])
    drag=np.full(len(ref),np.nan); alpha=drag.copy()
    for i,target in enumerate(ref.CL_actual.to_numpy()):
        # Required lift may be below the screening floor, including negative initial lift.
        # It is evaluated honestly for the oracle and does not enter fixed-alpha rankings.
        d=pd.DataFrame({'alpha_deg':aa,'CL':clt[i],'CD_pressure':cdt[i],'model_valid':np.isfinite(clt[i])&np.isfinite(cdt[i])})
        r=equal_lift(d,float(target)); drag[i]=r['CD_at_target']; alpha[i]=r['alpha_at_target']
    return drag,alpha

def oracle(data,geometries,tag):
    ref=pd.read_csv(OUT/'mission/reference.csv'); rows=[]; matrix=[]; ids=[]; missing_ids=[]
    required=set(mission_pairs())
    for _,r in geometries.iterrows():
        g=data[data.geometry_id.eq(r.geometry_id)]
        if not required.issubset(set(zip(g.M_flight,g.alpha_deg))):
            # Uncomputed coverage is not a physical infeasibility result.
            missing_ids.append(r.geometry_id)
            rows.append({'geometry_id':r.geometry_id,'full_1200s_feasible':False,'evaluation_status':'P2_GRID_INCOMPLETE','feasible_time_s':np.nan,'J_fixed_Ns':np.nan})
            continue
        # Content hash invalidates cached interpolation when any real P2 evidence changes.
        content=g.sort_values(['M_flight','alpha_deg'])[['M_flight','alpha_deg','CL','CD_pressure','model_valid']].to_csv(index=False).encode()
        key=hashlib.sha256(content+(OUT/'mission/reference.csv').read_bytes()+(OUT/'mission/domain.json').read_bytes()+Path(__file__).read_bytes()).hexdigest()[:16]
        dest=OUT/'oracle_cache'/f'{r.geometry_id}_{key}.npz'; dest.parent.mkdir(exist_ok=True)
        if dest.exists(): z=np.load(dest); drag=z['CD']; alpha=z['alpha']
        else:
            drag,alpha=mission_cd_for_geometry(g,ref)
            tmp=dest.with_suffix('.tmp.npz'); np.savez_compressed(tmp,CD=drag,alpha=alpha); os.replace(tmp,dest)
        complete=bool(np.isfinite(drag).all()); J=float(np.sum(ref.q_pa*SREF*drag*ref.dt_s)) if complete else np.nan
        rows.append({'geometry_id':r.geometry_id,'full_1200s_feasible':complete,'evaluation_status':'COMPLETE_GRID','feasible_time_s':float(ref.dt_s[np.isfinite(drag)].sum()),'J_fixed_Ns':J})
        ids.append(r.geometry_id); matrix.append(drag)
    if not matrix: return {'status':'NO_COMPLETE_GEOMETRY_GRIDS','incomplete_geometry_ids':missing_ids,'geometries_evaluated':0,'full_fixed_feasible_count':0,'oracle_covered_time_s':0}
    D=np.array(matrix); valid=np.isfinite(D); best=np.argmin(np.where(valid,D,np.inf),axis=0)
    mincd=D[best,np.arange(len(ref))]; allcovered=valid.any(axis=0)
    fixed=pd.DataFrame(rows); feasible=fixed[fixed.full_1200s_feasible]
    oracleJ=float(np.sum(ref.q_pa*SREF*mincd*ref.dt_s)) if allcovered.all() else np.nan
    fixedJ=float(feasible.J_fixed_Ns.min()) if len(feasible) else np.nan
    summary={'status':'INCOMPLETE_CANDIDATE_POOL' if missing_ids else ('OK' if np.isfinite(fixedJ) and np.isfinite(oracleJ) else 'FULL_TRAJECTORY_INFEASIBLE'),
             'J_fixed_Ns':fixedJ,'J_oracle_Ns':oracleJ,'eta_potential':(fixedJ-oracleJ)/fixedJ if fixedJ>0 and np.isfinite(oracleJ) else None,
             'best_fixed_geometry':feasible.sort_values('J_fixed_Ns').geometry_id.iloc[0] if len(feasible) else None,
             'oracle_covered_time_s':float(ref.dt_s[allcovered].sum()),'full_fixed_feasible_count':len(feasible),'geometries_evaluated':len(ids),
             'incomplete_geometry_ids':missing_ids,'complete_candidate_pool':not missing_ids,
             'scope':'Ideal instantaneous selection on one saved reference trajectory; pressure drag impulse only; no range, OCP, DQN or continuous-morph claim',
             'infeasibility_rule':'No dropping early/negative-CL states; any uncovered interval makes the full 1200 s metric unavailable'}
    trace=ref.copy(); trace['oracle_geometry']=np.where(allcovered,np.array(ids)[best],'INFEASIBLE'); trace['oracle_CD']=mincd
    trace['oracle_cumulative_Ns']=np.cumsum(ref.q_pa*SREF*mincd*ref.dt_s)
    if np.isfinite(fixedJ):
        k=ids.index(summary['best_fixed_geometry']); trace['best_fixed_CD']=D[k]; trace['fixed_cumulative_Ns']=np.cumsum(ref.q_pa*SREF*D[k]*ref.dt_s)
    csv(OUT/f'metrics/{tag}_fixed_drag.csv',fixed); csv(OUT/f'metrics/{tag}_oracle_trace.csv',trace); dump(OUT/f'metrics/{tag}_oracle.json',summary)
    return summary

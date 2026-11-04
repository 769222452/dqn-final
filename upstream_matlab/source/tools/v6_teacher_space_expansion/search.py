"""Adaptive proposal priorities only; no surrogate value enters any physical ranking."""
from __future__ import annotations
from scipy.spatial import cKDTree
from scipy.stats import qmc
from exp_common import *
from evaluation import all_aero
from analysis import rank_fixed,rank_equal
from preflight import make_case

def propose_batch(batch_number,count,start_index,stats,dry_run=False):
    dest=OUT/'batches'/f'batch_{batch_number:04d}'
    statepath=OUT/'search_state.json'
    if not dry_run and (dest/'candidates.json').exists():
        # candidates.json is the commit marker; recover the RNG update if the
        # controller stopped between committing proposals and shared state.
        after=load(dest/'state_after_proposal.json')
        state=load(statepath) if statepath.exists() else dict(after)
        if state.get('proposal_batch',0)<batch_number or not statepath.exists():
            state['rng_state']=after['rng_state']; state['proposal_batch']=batch_number
            dump(statepath,state)
        return load(dest/'candidates.json')
    if not dry_run: dest.mkdir(parents=True,exist_ok=True)
    state=load(statepath) if statepath.exists() else {'rng_state':np.random.default_rng(load(OUT/'config.json')['random_seed']+1).bit_generator.state,'stagnant_batches':0,'best_core_improvement':0.,'best_equal_drag':None}
    rng=np.random.default_rng(); rng.bit_generator.state=state['rng_state']
    p=pool(); d=all_aero(); ranked=rank_fixed(d,p)
    # Interrupted infrastructure calls are logged, but are not negative physical-feasibility labels.
    p=p[~p.get('execution_interrupted',pd.Series(False,index=p.index)).fillna(False).astype(bool)].copy()
    frozen=ranked[ranked.M_flight.isin(FROZEN_M)&ranked.alpha_deg.isin(FROZEN_A)]
    cols=pd.MultiIndex.from_tuples(frozen_pairs(),names=['M_flight','alpha_deg'])
    ld=frozen.pivot(index='geometry_id',columns=['M_flight','alpha_deg'],values='L_over_D').reindex(columns=cols)
    eligible=frozen.pivot(index='geometry_id',columns=['M_flight','alpha_deg'],values='eligible').reindex(columns=cols).fillna(False)
    ld=ld.where(eligible)
    # Per-condition performance relative to the original unmodified best score.
    best=ld.max(0).to_numpy(); Y=ld.to_numpy()/best
    train=p.set_index('geometry_id').loc[ld.index]; X=train[FEATURES].to_numpy(float)
    Xn=(X-LOWER)/(UPPER-LOWER); tree=cKDTree(Xn)
    allX=(p[FEATURES].to_numpy(float)-LOWER)/(UPPER-LOWER); triedtree=cKDTree(allX)
    success=gate(p).astype(float).to_numpy(); success_tree=cKDTree(allX)
    bins=np.arange(4,16); dbin=np.clip(np.floor(p.Ma_design).astype(int),4,15)
    density=np.array([(dbin==m).sum() for m in bins]); wins=np.array([success[dbin==m].sum() for m in bins])
    rate=(wins+2)/(density+4)
    task=load(OUT/'mission/domain.json')['envelope']['Mach']; inmission=(bins>=np.floor(task['min']))&(bins<=np.ceil(task['max']))
    mw=(1+.7*inmission)*np.sqrt(rate)/(1+density)**.35; mw/=mw.sum()
    stagnant=state['stagnant_batches']
    fractions=np.array([.25,.35,.25,.15]) if stagnant<3 else np.array([.40,.20,.25,.15])
    # Pools from QMC global coverage, local evolution, and phenotype-guided candidates.
    sobol=qmc.Sobol(7,scramble=True,seed=int(rng.integers(2**31)))
    u=sobol.random_base2(12); candidates=LOWER+u*(UPPER-LOWER)
    selectedbin=rng.choice(bins,len(candidates),p=mw)
    candidates[:,0]=np.minimum(15.,selectedbin+u[:,0])
    # Empirically feasible angle regions at similar design Mach; retain broad exploration.
    for i in range(len(candidates)):
        if rng.random()<.65:
            near=p[abs(p.Ma_design-candidates[i,0])<1.25]
            if len(near)>=4:
                low=near[PARAMS].quantile(.05).to_numpy(); high=near[PARAMS].quantile(.95).to_numpy()
                span=np.maximum(high-low,(UPPER-LOWER)[1:]*.04)
                candidates[i,1:]=np.clip(low-.15*span+u[i,1:]*1.3*span,LOWER[1:],UPPER[1:])
    # Preserve A/B/G42 and all existing regional top-five candidates as potential parents.
    parents=set(frozen[frozen['rank'].le(5)].geometry_id)|set(AB)|{'G42'}
    eqpath=OUT/f"metrics/{stats['tag']}_equal_lift.csv"
    eq=None
    if eqpath.exists():
        eq=pd.read_csv(eqpath,float_precision='round_trip'); parents|=set(eq[eq['rank'].le(5)].geometry_id)
    parents=p[p.geometry_id.isin(parents)&gate(p)]
    parent_ix=rng.integers(len(parents),size=2048); seeds=parents.iloc[parent_ix][FEATURES].to_numpy(float)
    scale=np.array([.35,.24,.24,.20,.20,.007,.006])*(1+min(stagnant,6)*.12)
    # Explore multiple step lengths, while all clipping stays inside recorded prior bounds.
    local=np.clip(seeds+rng.normal(size=seeds.shape)*scale*rng.choice([.3,1.,3.],(len(seeds),1),p=[.3,.5,.2]),LOWER,UPPER)
    cand=np.r_[candidates,local]; normalized=(cand-LOWER)/(UPPER-LOWER)
    dist,ix=tree.query(normalized,k=min(12,len(X))); weights=np.exp(-dist/np.maximum(dist[:,-1:],1e-5))
    yy=Y[ix]; mask=np.isfinite(yy)
    den=(weights[:,:,None]*mask).sum(1)
    pred=np.divide(np.nansum(weights[:,:,None]*yy,axis=1),den,out=np.full((len(cand),Y.shape[1]),np.nan),where=den>0)
    unc=np.sqrt(np.divide(np.nansum(weights[:,:,None]*(yy-pred[:,None,:])**2,axis=1),den,out=np.zeros_like(pred),where=den>0))
    novelty=triedtree.query(normalized,k=1)[0]
    _,valid_ix=success_tree.query(normalized,k=min(12,len(p)))
    validity_prior=(success[valid_ix].sum(axis=1)+1)/(valid_ix.shape[1]+2)
    abrows=[i for i,gid in enumerate(ld.index) if gid in AB]
    abphen=Y[abrows]; distances=[]
    for v in abphen:
        squared=(pred-v)**2; finite_count=np.isfinite(squared).sum(1)
        distances.append(np.divide(np.nansum(squared,axis=1),finite_count,out=np.full(len(cand),np.inf),where=finite_count>0))
    phenotype_distance=np.min(distances,axis=0)**.5
    phenotype_distance[~np.isfinite(phenotype_distance)]=0.
    core_indices=[i for i,(_,a) in enumerate(frozen_pairs()) if a>=5]
    focus_weights=np.ones(len(core_indices))
    top=frozen[frozen['rank'].eq(1)].set_index(['M_flight','alpha_deg'])
    for j,i in enumerate(core_indices):
        pt=frozen_pairs()[i]
        if pt in top.index and top.loc[pt,'geometry_id'] in AB: focus_weights[j]=2.
    focus_weights/=focus_weights.sum()
    # Equal-lift targets have independent local predictors when covered by real P2 labels.
    eqpred=None; equnc=None
    if eq is not None:
        e=eq[eq.status.eq('OK')].pivot(index='geometry_id',columns=['M_flight','CL_target'],values='CD_at_target').reindex(ld.index)
        ey=e.min(0).to_numpy()/e.to_numpy(); ez=ey[ix]; em=np.isfinite(ez); ed=(weights[:,:,None]*em).sum(1)
        eqpred=np.divide(np.nansum(weights[:,:,None]*ez,axis=1),ed,out=np.full((len(cand),ey.shape[1]),np.nan),where=ed>0)
        equnc=np.sqrt(np.divide(np.nansum(weights[:,:,None]*(ez-eqpred[:,None,:])**2,axis=1),ed,out=np.zeros_like(eqpred),where=ed>0))
    choices=[]; used=set(); strategy_counts=dict.fromkeys(['global_sobol','local_evolution','condition_targeted','aero_complementarity'],0)
    for j in range(count):
        strategy=rng.choice(list(strategy_counts),p=fractions); target=None
        if strategy=='global_sobol':
            score=novelty*(.25+validity_prior); score[len(candidates):]=-np.inf
        elif strategy=='local_evolution':
            score=np.max(np.nan_to_num(pred[:,core_indices]+.75*unc[:,core_indices],nan=-np.inf),axis=1)+.08*novelty
            score[:len(candidates)]=-np.inf
        elif strategy=='condition_targeted':
            if eqpred is not None and rng.random()<.55:
                k=int(rng.integers(eqpred.shape[1])); score=eqpred[:,k]+1.5*equnc[:,k]+.1*novelty; target={'kind':'equal_CL','column':list(e.columns[k])}
            else:
                k=int(rng.choice(core_indices,p=focus_weights)); score=pred[:,k]+1.5*unc[:,k]+.1*novelty; target={'kind':'fixed_alpha','condition':frozen_pairs()[k]}
            score*=.5+.5*validity_prior
        else:
            # Aerodynamic curve distance, with a performance floor; parameter distance alone does not define complementarity.
            quality=np.max(np.nan_to_num(pred,nan=-np.inf),axis=1)
            score=phenotype_distance+.15*novelty+.1*np.nanmax(unc,axis=1)
            score[(quality<.93)|(validity_prior<.25)]=-np.inf
        score=np.nan_to_num(score,nan=-np.inf); score[novelty<1e-7]=-np.inf
        if used: score[list(used)]=-np.inf
        k=int(np.argmax(score))
        if not np.isfinite(score[k]): k=int(np.argmax(np.where(novelty>1e-7,novelty,-np.inf)))
        used.add(k); strategy_counts[strategy]+=1
        c=make_case(cand[k],'search',start_index+j,strategy,batch_number=batch_number,proposal_target=target,proposal_score=float(score[k]) if np.isfinite(score[k]) else None,
                    proposal_warning='Surrogate priority for sampling only. Real MATLAB/audit/P2 required; no reward added to aerodynamic ranking.')
        choices.append(c)
        distance=np.linalg.norm(normalized-normalized[k],axis=1); novelty=np.minimum(novelty,distance)
    # Save model training data and random state together with the immutable proposals.
    if dry_run: return choices
    for metric in ['frozen49_top1','equal_lift_top1']:
        origin=OUT/f"metrics/{stats['tag']}_{metric}.csv"
        if origin.exists():
            csv(dest/f'incumbent_{metric}.csv',pd.read_csv(origin,float_precision='round_trip'))
    np.savez_compressed(dest/'surrogate.npz',X=X,Y=Y,geometry_ids=ld.index.to_numpy(str),condition_M=np.array(frozen_pairs())[:,0],condition_alpha=np.array(frozen_pairs())[:,1],feature_lower=LOWER,feature_upper=UPPER)
    state['rng_state']=rng.bit_generator.state; state['proposal_batch']=batch_number
    dump(dest/'state_after_proposal.json',state)
    dump(dest/'strategy.json',{'counts':strategy_counts,'stagnant_batches':stagnant,'design_mach_bins':bins,'sampling_weights':mw,'historical_validity_rate':rate,'selection_rule':'Adaptive nearest-neighbor kernel with uncertainty and QMC; only a candidate recommender'})
    dump(dest/'candidates.json',choices); dump(statepath,state)
    event('BATCH_PROPOSED',batch=batch_number,count=count,strategies=strategy_counts)
    return choices

def update_search_state(stats):
    state=load(OUT/'search_state.json')
    batch_key=stats['tag'].removesuffix('_verified')
    applied=state.get('metrics_applied_batches',[])
    # Refinement can stop after the batch's metrics have already been applied.
    # Resuming that batch must not increment stagnation a second time.
    if batch_key in applied or state.get('last_metrics','').removesuffix('_verified')==batch_key:
        return state
    previous=state.get('best_core_improvement',0.); improved=stats.get('max_core_improvement_vs_previous_batch',0.)>.001 or stats['max_core_improvement_vs_old']>previous+.001
    eq=pd.read_csv(OUT/f"metrics/{stats['tag']}_equal_lift_top1.csv")
    drag={f'{r.M_flight:.6g}|{r.CL_target:.12g}':r.CD_at_target for _,r in eq.iterrows()}
    old=state.get('best_equal_drag') or {}
    eqgain=max([(old[k]-v)/old[k] for k,v in drag.items() if k in old and old[k]>0],default=0.)
    state['stagnant_batches']=0 if improved or eqgain>.001 else state.get('stagnant_batches',0)+1
    state['best_core_improvement']=max(previous,stats['max_core_improvement_vs_old']); state['best_equal_drag']=drag
    state['last_metrics']=stats['tag']; state['metrics_applied_batches']=[*applied,batch_key]
    dump(OUT/'search_state.json',state)
    return state

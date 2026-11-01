"""Use real common-grid P2 evidence to quantify nominal winner and interpolation sensitivity."""
from __future__ import annotations
from exp_common import *
from evaluation import equal_table

def task_grid_sensitivity(tag,data,geometries,core,fixed_top,equal_fine_top):
    from analysis import rank_equal
    domain=load(OUT/'mission/domain.json'); ms=set(domain['M_grid']); aa=set(domain['alpha_grid'])
    lookup=fixed_top.set_index(['M_flight','alpha_deg']); details=[]
    for _,r in core.iterrows():
        required=[(float(m),float(a)) for m in [r.M_flight-.25,r.M_flight,r.M_flight+.25] for a in [r.alpha_deg-.5,r.alpha_deg,r.alpha_deg+.5]]
        available=[pt for pt in required if pt[0] in ms and pt[1] in aa and pt in lookup.index]
        retained=[]; margins=[]; rivals=[]
        for pt in available:
            q=lookup.loc[pt]; retained.append(q.geometry_id==r.geometry_id); margins.append(float(q.delta_LD)); rivals.append(str(q.geometry_id))
        details.append({'geometry_id':r.geometry_id,'M_flight':r.M_flight,'alpha_deg':r.alpha_deg,'core_margin':r.delta_LD,'checked_points':len(available),'requested_points':len(required),'missing_points':json.dumps(sorted(set(required)-set(available))),
                        'rank_retained_at_all_computed_neighbors':bool(retained and all(retained)),'all_nine_points_available':len(available)==9,'retained_fraction':float(np.mean(retained)) if retained else None,'minimum_neighbor_winner_margin':min(margins,default=None),'neighbor_winners':'|'.join(sorted(set(rivals)))})
    detail=pd.DataFrame(details); csv(OUT/f'metrics/{tag}_task_grid_rank_stability.csv',detail)
    coarse_alpha=[a for a in domain['alpha_grid'] if abs(a-round(a))<1e-12]
    coarse,coarsetop=rank_equal(equal_table(data,geometries,machs=domain['M_grid'],alpha_grid=coarse_alpha))
    compare=equal_fine_top[['M_flight','CL_target','geometry_id','CD_at_target','delta_CD']].merge(coarsetop[['M_flight','CL_target','geometry_id','CD_at_target']],on=['M_flight','CL_target'],suffixes=('_fine','_coarse'),how='outer')
    both=compare.CD_at_target_fine.notna()&compare.CD_at_target_coarse.notna()
    compare['same_winner']=compare.geometry_id_fine.eq(compare.geometry_id_coarse)&both
    compare['fine_vs_coarse_best_CD_relative']=(compare.CD_at_target_fine-compare.CD_at_target_coarse)/compare.CD_at_target_coarse
    csv(OUT/f'metrics/{tag}_equal_grid_sensitivity.csv',compare)
    summary={'comparison_pool_count':len(geometries),'core_centers':len(detail),'full_nine_neighbor_centers':int(detail.all_nine_points_available.sum()),
             'full_nine_rank_stable_centers':int((detail.all_nine_points_available&detail.rank_retained_at_all_computed_neighbors).sum()),
             'equal_lift_comparable_points':int(both.sum()),'equal_lift_changed_winner_points':int((both&~compare.same_winner).sum()),
             'equal_lift_max_abs_relative_CD_change':float(compare.fine_vs_coarse_best_CD_relative.abs().max()),
             'definition':'All eligible geometries use the same real P2 grid. Fixed-alpha neighbors at dM=0.25/dalpha=0.5; missing points explicitly listed. Equal CL compares common integer-alpha PCHIP to common mission-grid PCHIP. Diagnostic evidence, not the formal dM=0.125/dalpha=0.25 winner verification.'}
    dump(OUT/f'metrics/{tag}_sensitivity.json',summary)
    return summary

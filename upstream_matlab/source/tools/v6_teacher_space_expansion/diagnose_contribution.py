"""Attribute a completed batch's saved aerodynamic rankings without running physics."""
from __future__ import annotations
import argparse
import re
from exp_common import *


def diagnose(tag, previous_tag):
    match=re.fullmatch(r'batch_(\d+)(?:_verified)?',tag)
    if not match or not re.fullmatch(r'[A-Za-z0-9_-]+',previous_tag):
        raise ValueError('Use saved batch metric tags')
    number=int(match.group(1)); sources={}
    def read(name):
        path=OUT/name; sources[name]=sha(path)
        return pd.read_csv(path,float_precision='round_trip')
    stats_path=OUT/f'metrics/{tag}_summary.json'; stats=load(stats_path)
    sources[str(stats_path.relative_to(OUT))]=sha(stats_path)
    tables={name:read(f'metrics/{tag}_{name}.csv') for name in ['frozen49_top1','CORE42_top1','equal_lift_top1','mission_top1','fixed_drag','oracle_trace']}
    old=read(f'metrics/{previous_tag}_mission_top1.csv')
    freq=read('mission/frequency.csv')
    p=pool(); batch=p[p.batch_number.eq(number)&gate(p)].copy(); ids=set(batch.geometry_id)
    frozen=read(f'metrics/{tag}_frozen49_all_scores.csv')
    pool_ids=set(frozen.loc[gate(frozen),'geometry_id'])
    if len(pool_ids)!=stats['mission_grid_required_geometries'] or not ids.issubset(pool_ids):
        raise RuntimeError('Batch registry does not match the saved comparison pool')
    if not stats['equal_and_task_comparison_complete']:
        raise RuntimeError('Contribution attribution requires a completed common task grid')
    for r in batch.itertuples():
        path=Path(r.case_dir)/'record.json'; sources[str(path.relative_to(OUT))]=sha(path)
    task=tables['mission_top1'].merge(freq,on=['M_flight','alpha_deg'],how='left',validate='one_to_one')
    task['weight_s']=task.weight_s.fillna(0)
    trace=tables['oracle_trace']; fixed=tables['fixed_drag'].set_index('geometry_id')
    rows=[]
    for r in batch.itertuples():
        gid=r.geometry_id; f=fixed.loc[gid]
        rows.append({'geometry_id':gid,'strategy':r.strategy,'Ma_design':float(r.Ma_design),
                     'volume_ratio_G42':float(r.volume_ratio_G42),'thickness_ratio_G42':float(r.thickness_ratio_G42),
                     'frozen49_top1_points':int(tables['frozen49_top1'].geometry_id.eq(gid).sum()),
                     'CORE42_top1_points':int(tables['CORE42_top1'].geometry_id.eq(gid).sum()),
                     'equal_CL_top1_points':int(tables['equal_lift_top1'].geometry_id.eq(gid).sum()),
                     'task_grid_top1_points':int(task.geometry_id.eq(gid).sum()),
                     'task_grid_weight_s':float(task.loc[task.geometry_id.eq(gid),'weight_s'].sum()),
                     'oracle_selected_time_s':float(trace.loc[trace.oracle_geometry.eq(gid),'dt_s'].sum()),
                     'full_1200s_fixed_feasible':bool(true(pd.Series([f.full_1200s_feasible])).iloc[0]),
                     'feasible_time_s':float(f.feasible_time_s),'J_fixed_Ns':float(f.J_fixed_Ns)})
    per_geometry=pd.DataFrame(rows)
    new_points=task[task.geometry_id.isin(ids)].drop(columns='weight_s')
    prior=old[['geometry_id','M_flight','alpha_deg','L_over_D']].rename(columns={'geometry_id':'geometry_id_previous','L_over_D':'L_over_D_previous'})
    changed=new_points.merge(prior,on=['M_flight','alpha_deg'],how='left',validate='one_to_one')
    if changed.L_over_D_previous.isna().any(): raise RuntimeError('Previous task grid is missing a comparison point')
    changed['relative_LD_gain_vs_previous_winner']=changed.L_over_D/changed.L_over_D_previous-1
    if changed.relative_LD_gain_vs_previous_winner.lt(-1e-12).any(): raise RuntimeError('Expanded-pool best score decreased')
    result={'utc':utc(),'tag':tag,'previous_tag':previous_tag,'comparison_pool_count':len(pool_ids),
            'new_hard_gate_pass':len(batch),
            'new_frozen49_top1_points':int(tables['frozen49_top1'].geometry_id.isin(ids).sum()),
            'new_CORE42_top1_points':int(tables['CORE42_top1'].geometry_id.isin(ids).sum()),
            'new_equal_CL_top1_points':int(tables['equal_lift_top1'].geometry_id.isin(ids).sum()),
            'new_task_grid_top1_points':len(changed),
            'new_task_grid_weight_s':float(task.loc[task.geometry_id.isin(ids),'weight_s'].sum()),
            'task_frequency_total_weight_s':float(freq.weight_s.sum()),
            'new_oracle_selected_time_s':float(trace.loc[trace.oracle_geometry.isin(ids),'dt_s'].sum()),
            'new_full_1200s_fixed_feasible_count':sum(r['full_1200s_fixed_feasible'] for r in rows),
            'largest_task_grid_LD_gain_vs_previous':float(changed.relative_LD_gain_vs_previous_winner.max()) if len(changed) else None,
            'incomplete_fixed_trajectory_candidates':[r for r in rows if not r['full_1200s_fixed_feasible']],
            'source_sha256':sources,'scope':'Saved complete common-grid snapshot. Fixed-alpha, equal-CL and reference-oracle contributions are separate; frequency weights aggregate all input trajectories. No ranking or feasibility rule changes.',
            'Teacher_or_P2_calls':0,'formal_goal_proved':False}
    folder=OUT/'batches'/f'batch_{number:04d}'
    csv(folder/'aerodynamic_contribution.csv',per_geometry)
    csv(folder/'new_task_grid_winning_points.csv',changed)
    dump(folder/'aerodynamic_contribution.json',result)
    return {k:v for k,v in result.items() if k!='source_sha256'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag',required=True);parser.add_argument('--previous-tag',required=True)
    args=parser.parse_args();print(json.dumps(clean(diagnose(args.tag,args.previous_tag)),ensure_ascii=False,indent=2))

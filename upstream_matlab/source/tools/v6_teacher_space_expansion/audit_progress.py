"""Read-only completion audit against live processes and committed scientific artifacts."""
from __future__ import annotations
import psutil
from exp_common import *
from evaluation import all_aero,mission_pairs

def audit_progress():
    owner=load(OUT/'owner.json') if (OUT/'owner.json').exists() else {}; active=[]
    for proc in psutil.process_iter(['pid','name','cmdline','create_time','cpu_times']):
        try:
            command=' '.join(proc.info['cmdline'] or []).replace('\\','/')
            if 'v6_teacher_space_expansion' in command and proc.info['name'].lower() in ['matlab.exe','python.exe'] and ('run.py' in command or 'exp_worker(' in command):
                active.append({'pid':proc.pid,'name':proc.info['name'],'create_time':proc.info['create_time'],'cpu_seconds':sum(proc.info['cpu_times'][:2])})
        except (psutil.NoSuchProcess,psutil.AccessDenied): continue
    p=pool(); d=all_aero(); expected=set(mission_pairs()); complete=[]
    for gid,g in d.groupby('geometry_id'):
        if expected.issubset(set(zip(g.M_flight,g.alpha_deg))): complete.append(gid)
    feasible=p[gate(p)]; pending=sorted(set(feasible.geometry_id)-set(complete))
    latest=load(OUT/'metrics/latest.json'); status=load(OUT/'status.json')
    snapshot_path=OUT/f"metrics/{latest['tag']}_frozen49_all_scores.csv"
    snapshot_ids=set()
    if snapshot_path.exists():
        snapshot=pd.read_csv(snapshot_path,float_precision='round_trip')
        snapshot_ids=set(snapshot.loc[gate(snapshot),'geometry_id'])
    snapshot_matches_pool=snapshot_ids==set(feasible.geometry_id)
    source_ok=True
    try: verify_sources()
    except RuntimeError: source_ok=False
    new=p[p.phase.ne('baseline')]; all_calls=list((OUT/'cases').glob('*/started.json')); repro=list((OUT/'reproducibility').glob('*/started.json'))
    cases_complete=all((Path(r.case_dir)/'audit.json').exists() and (Path(r.case_dir)/'generation_status.json').exists() and (not gate(pd.DataFrame([r])).iloc[0] or (Path(r.case_dir)/'frozen49.csv').exists()) for _,r in new.iterrows())
    requirements={
      'existing_453_identity_and_original_ranking':{'proved':len(old_pool())==453 and (OUT/'validation.json').exists(),'evidence':'baseline/candidate_pool.csv, validation.json'},
      'source_formulas_reference_and_frozen_data_preserved':{'proved':source_ok,'evidence':'baseline/source_hashes.json, baseline/additional_source_hashes.json'},
      'real_successful_1200s_working_domain':{'proved':(OUT/'mission/support_evidence.json').exists(),'evidence':'mission/samples.csv, domain.json, support_evidence.json'},
      'known_geometry_reproduction_and_12_real_preflight':{'proved':load(OUT/'reproducibility/summary.json')['passed'] and load(OUT/'preflight/summary.json')['passed'],'evidence':'reproducibility/summary.json, preflight/summary.json'},
      'every_committed_candidate_audited_and_checkpointed':{'proved':cases_complete,'evidence':'cases/*/{candidate,generation_status,audit,record}.json'},
      'task_grid_for_entire_feasible_pool':{'proved':not pending,'missing_geometry_count':len(pending),'evidence':'aero/*/*.csv'},
      'batched_adaptive_search_executed':{'proved':bool(list((OUT/'batches').glob('*/completed.json'))),'evidence':'batches/*/{candidates,strategy,metrics,completed}.json'},
      'metrics_snapshot_matches_current_pool':{'proved':snapshot_matches_pool,'snapshot_tag':latest['tag'],'snapshot_pool_count':len(snapshot_ids),'current_pool_count':len(feasible),'evidence':str(snapshot_path)},
      'formal_CORE42_stable_diversity_goal':{'proved':snapshot_matches_pool and len(latest.get('core_effective_verified',[]))>=6 and latest['CORE42']['AB_share']<.6 and len(latest.get('new_effective_verified',[]))>=3,'evidence':'metrics/latest.json, verification/latest.json'},
      'independent_equal_CL_regions_with_new_geometries':{'proved':snapshot_matches_pool and len(latest.get('equal_lift_verified_winners',[]))>=4 and len(latest.get('new_equal_lift_verified_winners',[]))>=2,'evidence':'metrics/*_equal_lift.csv, verification/latest.json'},
      'whole_pool_1200s_oracle':{'proved':snapshot_matches_pool and bool(latest.get('oracle',{}).get('complete_candidate_pool',False)) and not pending,'evidence':'metrics/*_oracle.json, *_oracle_trace.csv, *_fixed_drag.csv'},
      'grid_and_ranking_sensitivity':{'proved':snapshot_matches_pool and 'grid_sensitivity' in latest and not pending,'evidence':'metrics/*_sensitivity.json, *_task_grid_rank_stability.csv'},
      'terminal_stop_condition_reached_and_final_deliverables':{'proved':not active and status['status'] in ['TARGET_REACHED_VERIFIED','TARGET_NOT_REACHED'],'evidence':'status.json, reports/FINAL_REPORT_ZH.md, gocart_candidates.csv'}
    }
    result={'utc':utc(),'status':status['status'],'live_processes':active,'total_Teacher_calls_started':len(all_calls)+len(repro),'new_candidates_committed':len(new),'new_unique_hard_gate_pass':int(gate(new).sum()),'task_grid_complete_feasible_geometries':len(feasible)-len(pending),'task_grid_required_feasible_geometries':len(feasible),'requirements':requirements,'goal_completion_proved':False,'completion_note':'Scientific target is not guaranteed; budget-end delivery is a permitted outcome only after a separate final artifact and requirement audit.'}
    dump(OUT/'progress_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['requirements']},ensure_ascii=False,indent=2))
    return result
if __name__=='__main__': audit_progress()

"""Resume the single-process real Teacher campaign until verified success or protection limits."""
from __future__ import annotations
import argparse,traceback
from datetime import datetime,timedelta,timezone
from exp_common import *
from initialize import initialize
from preflight import reproducibility,preflight,process_case
from runtime import Worker,CampaignLock,StopRequested,BudgetReached
from evaluation import all_aero,ensure_grid,mission_pairs,oracle
from analysis import summarize
from reporting import write_report
from search import propose_batch,update_search_state
from verification import verify_winners

def actual_calls():
    return len(list((OUT/'cases').glob('*/started.json')))+len(list((OUT/'reproducibility').glob('*/started.json')))

def baseline_ready_for_resume():
    summary=OUT/'metrics/baseline_plus_preflight_summary.json'
    reference=OUT/'metrics/baseline453_oracle.json'
    if not summary.exists() or not reference.exists(): return False
    s=load(summary); r=load(reference)
    return bool(s.get('equal_and_task_comparison_complete') and r.get('complete_candidate_pool'))

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--resume',action='store_true')
    ap.add_argument('--max-teacher-attempts',type=int); ap.add_argument('--max-wall-time-hours',type=float)
    ap.add_argument('--batch-size',type=int); ap.add_argument('--extend-hours',type=float,help='Explicitly grant additional wall budget from now; use only for a user-authorized next stage')
    args=ap.parse_args()
    with CampaignLock():
        initialize(); verify_sources(); cfg=load(OUT/'config.json')
        prior_termination=OUT/'termination.json'
        if prior_termination.exists():
            archive=OUT/'termination_history'; archive.mkdir(exist_ok=True)
            os.replace(prior_termination,archive/f'termination_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")}.json')
        for arg,key in [(args.max_teacher_attempts,'max_teacher_attempts'),(args.max_wall_time_hours,'max_wall_time_hours'),(args.batch_size,'batch_size')]:
            if arg is not None:
                if arg<=0: raise ValueError(f'{key} must be positive')
                cfg[key]=arg
        if args.extend_hours:
            cfg['deadline_override_utc']=(datetime.now(timezone.utc)+timedelta(hours=args.extend_hours)).isoformat()
        dump(OUT/'config.json',cfg)
        deadline=datetime.fromisoformat(cfg.get('deadline_override_utc')) if cfg.get('deadline_override_utc') else datetime.fromisoformat(cfg['created_utc'])+timedelta(hours=cfg['max_wall_time_hours'])
        def check():
            if (OUT/'STOP_REQUESTED').exists(): raise StopRequested((OUT/'STOP_REQUESTED').read_text(encoding='utf-8-sig'))
            if datetime.now(timezone.utc)>=deadline: raise BudgetReached('48h/configured wall budget exhausted')
        w=None; stats=None
        try:
            check(); w=Worker(); reproducibility(w); preflight(w)
            baseline_ready=baseline_ready_for_resume()
            if baseline_ready:
                latest=OUT/'metrics/latest.json'
                stats=load(latest if latest.exists() else OUT/'metrics/baseline_plus_preflight_summary.json')
                event('BASELINE_REUSED',metrics_tag=stats['tag'],reason='Completed initial baseline is retained; resume current search batch without overwriting historical baseline metrics')
            else:
                stats=summarize('preflight',False); write_report(stats,'RUNNING_BASELINE_P2')
            # Every eligible old candidate and all preflight survivors receive the same task grid.
            p=pool(); e=p[gate(p)].copy(); e['priority']=np.where(e.geometry_id.isin(['G42',*AB]),0,1)
            ensure_grid(w,e.sort_values(['priority','geometry_id']),mission_pairs(),'mission',check)
            verify_sources()
            if not baseline_ready:
                stats=summarize('baseline_plus_preflight',True)
                if not (OUT/'metrics/baseline453_oracle.json').exists():
                    old=p[p.phase.eq('baseline')&gate(p)]
                    oracle(all_aero(),old,'baseline453')
                event('BASELINE_COMPLETE',core_winners=stats['CORE42']['unique_top1'],AB_share=stats['CORE42']['AB_share'],oracle=stats.get('oracle'))
            write_report(stats,'RUNNING_SEARCH')
            batch_number=1
            while True:
                check()
                batchdir=OUT/'batches'/f'batch_{batch_number:04d}'
                if (batchdir/'completed.json').exists():
                    stats=load(batchdir/'metrics.json'); batch_number+=1; continue
                if actual_calls()>=cfg['max_teacher_attempts']: raise BudgetReached('Teacher attempt budget exhausted')
                started=[load(f) for f in (OUT/'cases').glob('*/candidate.json')]
                next_index=max([r['attempt_index'] for r in started],default=12)+1
                remaining=cfg['max_teacher_attempts']-actual_calls()
                cases=propose_batch(batch_number,min(cfg['batch_size'],remaining),next_index,stats)
                for c in cases:
                    check()
                    exists=(Path(c['case_dir'])/'started.json').exists()
                    if not exists and actual_calls()>=cfg['max_teacher_attempts']: raise BudgetReached('Teacher attempt budget exhausted')
                    r=process_case(w,c)
                    if r['hard_gate_pass']:
                        ensure_grid(w,pd.DataFrame([r]),mission_pairs(),'mission',check)
                    dump(OUT/'checkpoint.json',{'utc':utc(),'batch_number':batch_number,'last_candidate':c['geometry_id'],'teacher_calls':actual_calls(),'deadline_utc':deadline.isoformat()})
                verify_sources(); tag=f'batch_{batch_number:04d}'; stats=summarize(tag,True)
                cids={c['geometry_id'] for c in cases}; batch=pool(); batch=batch[batch.geometry_id.isin(cids)]
                stats['batch']={'attempts':len(cases),'native_VALID':int(batch.geometry_validity.eq('VALID').sum()),'unique_hard_gate_pass':int(gate(batch).sum()),'aero_with_valid_points':int(batch.get('aero_valid_points',pd.Series(dtype=float)).fillna(0).gt(0).sum()),'generation_success_rate':sum(load(Path(c['case_dir'])/'generation_status.json')['status']=='GENERATED' for c in cases)/len(cases),'geometry_legal_rate':float(batch.geometry_validity.eq('VALID').mean()),'hard_gate_rate':float(gate(batch).mean())}
                evaluated=int(gate(batch).sum())*49
                stats['batch'].update(P2_evaluated_points=evaluated,P2_valid_points=int(batch.get('aero_valid_points',pd.Series(dtype=float)).fillna(0).sum()),P2_eligible_points=int(batch.get('eligible_points',pd.Series(dtype=float)).fillna(0).sum()))
                stats['batch']['P2_valid_fraction']=stats['batch']['P2_valid_points']/evaluated if evaluated else None
                stats['batch']['outcomes_by_strategy']={name:{'attempts':len(g),'native_VALID':int(g.geometry_validity.eq('VALID').sum()),'unique_hard_gate_pass':int(gate(g).sum())} for name,g in batch.groupby('strategy')}
                before=batchdir/'incumbent_frozen49_top1.csv'
                if before.exists():
                    previous=pd.read_csv(before,float_precision='round_trip'); current=pd.read_csv(OUT/f'metrics/{tag}_CORE42_top1.csv',float_precision='round_trip')
                    change=current.merge(previous[['M_flight','alpha_deg','L_over_D']],on=['M_flight','alpha_deg'],suffixes=('_new','_previous'))
                    change['relative_improvement']=(change.L_over_D_new-change.L_over_D_previous)/change.L_over_D_previous
                    csv(batchdir/'improvement_vs_incumbent.csv',change)
                    stats['max_core_improvement_vs_previous_batch']=float(change.relative_improvement.max())
                    stats['mean_core_improvement_vs_previous_batch']=float(change.relative_improvement.mean())
                eq=pd.read_csv(OUT/f'metrics/{tag}_equal_lift_top1.csv'); middle=eq[np.isclose(eq.M_flight,6.5)&np.isclose(eq.CL_target,.0398)]
                stats['best_equal_drag']=float(middle.CD_at_target.iloc[0]) if len(middle) else None
                state=update_search_state(stats); stats['stagnant_batches']=state['stagnant_batches']
                # Validate emerging substantial winners even before all diversity targets are met.
                newids=set(pool().query("phase != 'baseline'").geometry_id)
                emerging=(set(stats['core_winners_with_adjacent_0p5pct_margins'])|set(stats['equal_lift_adjacent_0p5pct_winners']))&newids
                if stats['nominal_target_ready_for_verification'] or emerging:
                    dump(batchdir/'metrics_pending_verification.json',{**stats,'snapshot_utc':utc(),'batch_status':'REFINEMENT_IN_PROGRESS','verification_completed':False})
                    write_report(stats,'RUNNING_WINNER_REFINEMENT'); verify_winners(w,stats,check)
                    batch_details={k:v for k,v in stats.items() if k in ['batch','best_equal_drag','stagnant_batches','max_core_improvement_vs_previous_batch','mean_core_improvement_vs_previous_batch']}
                    stats=summarize(tag+'_verified',True); stats.update(batch_details)
                dump(batchdir/'metrics.json',stats); dump(batchdir/'completed.json',{'utc':utc(),'tag':stats['tag'],'teacher_calls_total':actual_calls()})
                write_report(stats,'TARGET_REACHED_VERIFIED' if stats['formal_target_reached'] else 'RUNNING_SEARCH')
                event('BATCH_COMPLETE',batch=batch_number,teacher_calls=actual_calls(),unique_new=stats['new_unique_hard_gate_pass'],core=stats['CORE42'],effective=stats['core_effective_verified'],equal_lift=stats['equal_lift'],formal_target_reached=stats['formal_target_reached'])
                if stats['formal_target_reached']: break
                batch_number+=1
        except (StopRequested,BudgetReached) as exc:
            status=('CHECKPOINT_RESTART' if str(exc).startswith('Internal checkpoint') else 'STOPPED_BY_USER') if isinstance(exc,StopRequested) else 'TARGET_NOT_REACHED'
            event(status,reason=str(exc),teacher_calls=actual_calls())
            if (OUT/'preflight/summary.json').exists():
                stats=summarize('checkpoint_final',isinstance(exc,BudgetReached)); write_report(stats,status)
            dump(OUT/'termination.json',{'utc':utc(),'status':status,'reason':str(exc),'teacher_calls':actual_calls(),'resume_command':r'.\.venv\Scripts\python.exe -B tools\v6_teacher_space_expansion\run.py --resume','budget_extension_example':r'.\.venv\Scripts\python.exe -B tools\v6_teacher_space_expansion\run.py --resume --extend-hours 48 --max-teacher-attempts 10000'})
        except Exception as exc:
            event('INFRASTRUCTURE_ERROR',error=str(exc),traceback=traceback.format_exc())
            dump(OUT/'termination.json',{'utc':utc(),'status':'INFRASTRUCTURE_ERROR','error':str(exc),'traceback':traceback.format_exc()})
            if stats: write_report(stats,'INFRASTRUCTURE_ERROR')
            raise
        finally:
            if w: w.close()
            verify_sources()

if __name__=='__main__': main()

"""Observe committed verification coverage without sending any solver request."""
from __future__ import annotations
import argparse
import time
import psutil
from exp_common import *
from evaluation import all_aero
from verification import neighborhood_plan


def snapshot():
    progress_path=OUT/'verification_progress.json'
    previous=load(progress_path) if progress_path.exists() else {}
    plan=load(OUT/'verification/plan.json')
    folder=OUT/'verification'/plan['pool_ids_sha256'][:16]
    previous_hash=previous.get('pool_ids_sha256',previous.get('verified_pool_ids_sha256'))
    if previous_hash and previous_hash!=plan['pool_ids_sha256']:
        archive=OUT/'verification'/previous_hash[:16]/'progress_snapshot.json'
        if not archive.exists(): dump(archive,previous)
    round_path=sorted(folder.glob('round_*_plan.json'))[-1]
    round_bytes=round_path.read_bytes(); current=json.loads(round_bytes)
    tag=load(OUT/'status.json')['metrics_tag']
    core=pd.read_csv(OUT/f'metrics/{tag}_CORE42_top1.csv',float_precision='round_trip').reset_index(drop=True)
    equal=pd.read_csv(OUT/f'metrics/{tag}_equal_lift_top1.csv',float_precision='round_trip').reset_index(drop=True)
    selected=current['selected_row_indices']
    indices=[sorted({i for pair in group.values() for i in pair}) for group in selected]
    _,_,pairs,_=neighborhood_plan(core,equal,*indices,load(OUT/'mission/domain.json'))
    if len(pairs)!=current['query_points_per_geometry']:
        raise RuntimeError('Reconstructed query set does not match the saved round')
    p=pool(); ids=set(p.loc[gate(p),'geometry_id'])
    if hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()!=plan['pool_ids_sha256']:
        return {'utc':utc(),'state':'POOL_CHANGED','note':'Wait for a plan for the expanded pool; no completion claim.'}
    data=all_aero(); complete=[]
    for gid,g in data[data.geometry_id.isin(ids)].groupby('geometry_id'):
        if pairs.issubset(set(zip(g.M_flight,g.alpha_deg))): complete.append(gid)
    owner=load(OUT/'owner.json'); controller_live=False; processes=[]
    expected={owner['pid']:owner['create_time']}
    for item in previous.get('processes',[]):
        expected[item['pid']]=item.get('creation',item.get('create_time'))
    # A gracefully resumed controller may own a newly launched MATLAB worker.
    # Discover it from the registered worker path and actual command line rather
    # than continuing to report only the exited worker from an earlier snapshot.
    for ready in (OUT/'workers').glob('*/ready.json'):
        try:
            proc=psutil.Process(load(ready)['pid'])
            command=' '.join(proc.cmdline()).replace('\\','/').lower()
            if 'matlab' in proc.name().lower() and ready.parent.as_posix().lower() in command:
                expected[proc.pid]=proc.create_time()
        except psutil.NoSuchProcess: pass
    for pid,created in expected.items():
        try:
            proc=psutil.Process(pid)
            if created is not None and abs(proc.create_time()-created)<.01:
                processes.append({'pid':pid,'creation':proc.create_time(),'cpu_seconds':sum(proc.cpu_times()[:2])})
                if pid==owner['pid']: controller_live=True
        except psutil.NoSuchProcess: pass
    result=load(folder/'completed.json') if (folder/'completed.json').exists() else None
    state='VERIFICATION_COMPLETED' if result else ('P2_GRID_COMPLETE_RANK_CHECK_PENDING' if len(complete)==len(ids) else 'VERIFICATION_IN_PROGRESS')
    record={'utc':utc(),'state':state,'metrics_tag':tag,'round':int(round_path.stem.split('_')[1]),
            'verification_mode':plan.get('verification_mode','FULL_NOMINAL_DIAGNOSTICS'),
            'pool_count':len(ids),'pool_ids_sha256':plan['pool_ids_sha256'],
            'query_points_per_geometry':len(pairs),'fully_evaluated_geometries':len(complete),
            'fully_evaluated_geometry_ids':sorted(complete),'pending_geometries':len(ids)-len(complete),
            'plan_sha256':hashlib.sha256(round_bytes).hexdigest(),
            'teacher_calls':len(list((OUT/'cases').glob('*/started.json')))+len(list((OUT/'reproducibility').glob('*/started.json'))),
            'processes':processes,'controller_live':controller_live,'formal_target_proven':False,
            'verification_result_completed':bool(result),
            'note':'Coverage of committed raw P2 files for this exact pool and round. Complete coverage alone does not certify rank retention; this observer never starts or restarts a solver.'}
    if result:
        record.update(core_effective_verified=result['stable_core_winners'],equal_lift_verified_winners=result['stable_equal_winners'],verification_result_utc=result['utc'],all_new_nominal_diagnostics_complete=bool(result['all_new_nominal_core_checked'] and result['all_new_nominal_equal_checked']))
    if round_path.read_bytes()!=round_bytes or load(OUT/'verification/plan.json')['pool_ids_sha256']!=plan['pool_ids_sha256']:
        return {'utc':utc(),'state':'PLAN_CHANGED','note':'Repeat observation against the new plan.'}
    dump(progress_path,record)
    return record


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seconds',type=float,default=0,help='Bounded observation duration; zero takes one snapshot')
    ap.add_argument('--interval',type=float,default=40)
    args=ap.parse_args()
    if args.seconds<0 or not 1<=args.interval<=60: ap.error('Use a nonnegative duration and a 1-60 second interval')
    until=time.monotonic()+args.seconds
    while True:
        result=snapshot()
        print(json.dumps({k:v for k,v in result.items() if k not in ['fully_evaluated_geometry_ids','note']},ensure_ascii=False),flush=True)
        remaining=until-time.monotonic()
        if remaining<=0 or result['state'] in ['VERIFICATION_COMPLETED','POOL_CHANGED'] or result.get('controller_live') is False: break
        time.sleep(min(args.interval,remaining))

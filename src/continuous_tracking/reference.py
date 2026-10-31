"""Fresh fixed-CL reference through the unchanged surrogate certification path."""
from pathlib import Path
import time

import numpy as np

from certified_flight import CL_ACTIONS, OracleRequired, build_flight
from rollout_improvement.identity import experiment_identity
from rollout_improvement.policy import preference
from .experiment import write_json


def run_reference(out, cl=.025, max_decisions=400):
    if cl not in CL_ACTIONS or not 1 <= max_decisions <= 400:
        raise ValueError('Invalid fixed-CL reference request')
    out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    env,cert=build_flight(out/'run','continuous_tracking_legacy_reference')
    write_json(out/'identity.json',experiment_identity())
    action=CL_ACTIONS.index(cl)
    rows=[]
    reason='MAX_DECISIONS'
    for step in range(max_decisions):
        tick=time.perf_counter()
        before=env.alpha
        pool=cert.pool(preference(cl))
        if not pool['mask'][action]:
            reason='REQUESTED_CL_NOT_CERTIFIED'
            break
        try:
            reward,done,info=cert.commit(pool,action)
        except OracleRequired as exc:
            reason='EXECUTION_REFUSED:'+exc.reason
            break
        row=dict(t0=env.t-3,t1=env.t,CL_cmd=info['CL_cmd'],CL_mean=info['executed_CL_mean'],
                 alpha_deg=np.rad2deg(info['plant']['alpha']).tolist(),
                 previous_alpha_deg=float(np.rad2deg(before)),
                 boundary_jump_deg=float(np.rad2deg(info['plant']['alpha'][0]-before)),
                 proof_id=pool['certificates'][action].proof_id,channel=info['channel'],
                 wall_s=time.perf_counter()-tick)
        rows.append(row)
        if done:
            reason=env.last_outcome
        result=dict(protocol='legacy-surrogate-certified-reference',CL=cl,completed_1200s=env.t==1200 and reason=='SUCCESS',
                    stop_reason=reason,time_s=env.t,range_km=float(env.x[3]/1000),reward=env.total_reward,
                    switches=env.switches,steps=len(rows),wall_s=time.perf_counter()-started,rows=rows)
        write_json(out/'result.json',result)
        if step%20==0 or done:
            print(f'Legacy step {step+1}, t={env.t:g}, range={env.x[3]/1000:.6f} km',flush=True)
        if done:
            break
    result=dict(protocol='legacy-surrogate-certified-reference',CL=cl,completed_1200s=env.t==1200 and reason=='SUCCESS',
                stop_reason=reason,time_s=env.t,range_km=float(env.x[3]/1000),reward=env.total_reward,
                switches=env.switches,steps=len(rows),wall_s=time.perf_counter()-started,rows=rows)
    write_json(out/'result.json',result)
    if reason not in ('SUCCESS','MAX_DECISIONS'):
        write_json(out/'stop_state.json',dict(t=env.t,x=env.x,alpha=env.alpha,reason=reason))
    return result

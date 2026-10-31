"""Frozen-state comparisons: recorded execution vs new real OCP, no training."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from certified_flight.corrector import audit_controls, compact_audit
from certified_flight.diagnostics import interval_moments
from continuous_tracking.experiment import DEFAULT_CONFIG, Refused, build, quality, solve_audited, write_json


def run(args):
    args.out.mkdir(parents=True,exist_ok=False)
    ocp,config,identity=build(args.config)
    write_json(args.out/'identity.json',identity)
    flight=json.loads(args.trajectory.read_text(encoding='utf-8'))
    rows=[]
    for t in args.times:
        s=next(s for s in flight['steps'] if s['t0']==t)
        prev=next((s for s in flight['steps'] if s['t0']==t-3),None)
        c=dict(x=prev['x'][-1] if prev else flight['x0'],
               alpha_prev=prev['alpha'][-1] if prev else ocp.m.data['initial_state']['alpha_rad'],
               t=t,source=ocp.db.index[s['src']],target=ocp.db.index[s['tgt']],fraction=s['frac'],
               rate=.125 if s['src']!=s['tgt'] and s['frac']<1 else 0.,CL_des=s['CL_cmd'])
        folder=args.out/f't{t:g}'
        folder.mkdir()
        old=audit_controls(ocp.plant.base,c,s['alpha'])
        old_metrics=interval_moments(old['dense'],target=c['CL_des'])
        row=dict(t=t,CL_cmd=c['CL_des'],context=c,old_alpha_deg=np.rad2deg(s['alpha']).tolist(),
                 old_CL_mean=old_metrics['mean'],old_CL_rmse=old_metrics['rmse'],old_audit=compact_audit(old))
        tick=time.perf_counter()
        try:
            plan,aud,arrays,attempts=solve_audited(ocp,c,None,folder,'new_post')
            q=quality(aud,c['CL_des'],config['tracking_quality'])
            row.update(new_alpha_nodes_deg=np.rad2deg(plan.alpha_nodes[:7]).tolist(),new_audit=aud,
                       new_quality=q,new_CL_mean=aud['grids']['0.1']['CL_mean_first3'],
                       new_CL_rmse=aud['grids']['0.1']['CL_rmse_first3'],
                       boundary_jump_deg=float(np.rad2deg(plan.alpha_nodes[0]-c['alpha_prev'])),
                       solver=plan.summary(),status='AUDITED_PASS')
        except Refused as exc:
            row.update(status=str(exc))
        row['wall_s']=time.perf_counter()-tick
        rows.append(row)
        write_json(args.out/'result.json',dict(source=str(args.trajectory),results=rows,
                    comparison='same post-context, different explicitly declared alpha execution model'))
        print(json.dumps({k:v for k,v in row.items() if k in ('t','status','old_CL_rmse','new_CL_rmse',
                         'old_CL_mean','new_CL_mean','new_quality','boundary_jump_deg','wall_s')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trajectory',type=Path,required=True)
    p.add_argument('--times',type=float,nargs='+',default=[600.,603.])
    p.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    p.add_argument('--out',type=Path,required=True)
    run(p.parse_args())

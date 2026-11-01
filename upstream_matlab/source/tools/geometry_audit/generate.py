"""Serial MATLAB supervisor; hard per-design wall timeout; separate audit budget."""
from pathlib import Path
import argparse,json,subprocess,time
from datetime import datetime,timezone
from prepare import ROOT,OUT,TOOLS,dump,sha

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--start',type=int,default=1); ap.add_argument('--end',type=int,default=52); args=ap.parse_args()
    jobs=json.loads((OUT/'jobs.json').read_text(encoding='utf-8'))
    selected=[j for j in jobs if args.start<=j['design_index']<=args.end]
    manifest=json.loads((OUT/'source_manifest.json').read_text(encoding='utf-8'))
    for p,h in manifest['active_source_hashes'].items(): assert sha(ROOT/p)==h,(p,'source changed')
    budget_path=OUT/'geometry_audit_budget.json'
    budget=json.loads(budget_path.read_text(encoding='utf-8')) if budget_path.exists() else {'authorized_designs':52,'hard_per_design_timeout_s':120,'startup_timeout_s':180,'max_consecutive_timeouts':3,'workers':0,'geometry_calls':0,'aerodynamic_calls':0,'ocp_calls':0,'training_calls':0,'sessions':[],'calls':[]}
    consecutive=0
    while True:
        pending=[j for j in selected if not (Path(j['case_dir'])/'generation_status.json').exists()]
        if not pending: break
        session=len(budget['sessions'])+1
        batch=OUT/f'batch_{session:02d}.json';dump(batch,pending)
        for j in pending: dump(Path(j['case_dir'])/'parameters.json',j)
        code=f"addpath('{TOOLS.as_posix()}'); run_geometry_audit_batch('{batch.as_posix()}','{(ROOT/'teacher_matlab/active').as_posix()}');"
        log=(OUT/f'matlab_session_{session:02d}.log').open('w',encoding='utf-8')
        start=time.monotonic(); p=subprocess.Popen(['D:/matlab/bin/matlab.exe','-batch',code],cwd=str(OUT),stdout=log,stderr=subprocess.STDOUT)
        record={'session':session,'pid':p.pid,'utc':datetime.now(timezone.utc).isoformat(),'designs':[j['design_index'] for j in pending]}
        budget['sessions'].append(record);dump(budget_path,budget)
        timed_out=False;seen=set();watch={}
        while p.poll() is None:
            for j in pending:
                d=Path(j['case_dir']);idx=j['design_index']
                if (d/'started.json').exists() and idx not in watch: watch[idx]=time.monotonic()
                if (d/'generation_status.json').exists() and idx not in seen:
                    s=json.loads((d/'generation_status.json').read_text(encoding='utf-8-sig')); seen.add(idx)
                    budget['calls'].append(s);budget['geometry_calls']=len(budget['calls']);dump(budget_path,budget)
                    print(json.dumps(s,ensure_ascii=False),flush=True);consecutive=0
                elif idx in watch and idx not in seen and time.monotonic()-watch[idx]>120:
                    subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True)
                    s={'design_index':idx,'status':'GENERATION_FAILED','elapsed_s':time.monotonic()-watch[idx],'error_message':'Supervisor hard per-design timeout 120 s','timeout':True}
                    dump(d/'generation_status.json',s); budget['calls'].append(s); budget['geometry_calls']=len(budget['calls']);consecutive+=1;timed_out=True;break
            if timed_out: break
            if not watch and time.monotonic()-start>180:
                subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True)
                raise RuntimeError('MATLAB startup timeout; no geometry call claimed')
            time.sleep(.25)
        p.wait(timeout=30);log.close()
        # Collect last completion if MATLAB exited between supervisor samples.
        for j in pending:
            d=Path(j['case_dir']);idx=j['design_index']
            if (d/'generation_status.json').exists() and not any(c['design_index']==idx for c in budget['calls']):
                s=json.loads((d/'generation_status.json').read_text(encoding='utf-8-sig'));budget['calls'].append(s)
        budget['geometry_calls']=len(budget['calls']);record['wall_s']=time.monotonic()-start;record['returncode']=p.returncode
        budget['geometry_elapsed_s_total']=sum(c.get('elapsed_s',0) for c in budget['calls'])
        budget['pilot_actual_elapsed_s']=[c['elapsed_s'] for c in budget['calls'] if c['design_index'] in [1,2,3]]
        dump(budget_path,budget)
        if consecutive>=3: raise RuntimeError('Three consecutive geometry timeouts; stopped')
        if p.returncode!=0 and not timed_out: raise RuntimeError(f'MATLAB exit {p.returncode}; inspect session log, do not blindly retry')
    print(f'COMPLETED calls={budget["geometry_calls"]}',flush=True)
if __name__=='__main__':main()

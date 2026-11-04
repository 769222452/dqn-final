"""Describe recorded interruptions and recovery intervals without inventing CPU time."""
from __future__ import annotations
from datetime import timedelta
from exp_common import *


def summarize_runtime():
    path=OUT/'events.jsonl'; content=path.read_bytes()
    events=[json.loads(line) for line in content.decode('utf-8').splitlines() if line.strip()]
    ready=[e for e in events if e['event'] in ('MATLAB_READY','MATLAB_REATTACHED')]
    ready.sort(key=lambda e:e['utc'])
    paths=sorted((OUT/'termination_history').glob('*.json'))
    if (OUT/'termination.json').exists(): paths.append(OUT/'termination.json')
    rows=[]; sources={str(path.relative_to(OUT)):hashlib.sha256(content).hexdigest()}
    for path in paths:
        data=path.read_bytes(); termination=json.loads(data.decode('utf-8-sig'))
        sources[str(path.relative_to(OUT))]=hashlib.sha256(data).hexdigest()
        start=datetime.fromisoformat(termination['utc'])
        resumed=next((e for e in ready if datetime.fromisoformat(e['utc'])>=start),None)
        finish=datetime.fromisoformat(resumed['utc']) if resumed else None
        rows.append({'recorded_termination_utc':start.isoformat(),'status':termination['status'],
                     'recorded_reason':termination.get('reason',termination.get('error','')),
                     'next_matlab_ready_utc':finish.isoformat() if finish else None,
                     'seconds_until_matlab_ready':(finish-start).total_seconds() if finish else None,
                     'next_worker':resumed.get('worker') if resumed else None,
                     'termination_record':str(path.relative_to(OUT))})
    rows.sort(key=lambda r:r['recorded_termination_utc'])
    cfg=load(OUT/'config.json'); start=datetime.fromisoformat(cfg['created_utc']);now=datetime.now(timezone.utc)
    deadline=datetime.fromisoformat(cfg['deadline_override_utc']) if cfg.get('deadline_override_utc') else start+timedelta(hours=cfg['max_wall_time_hours'])
    syslog=OUT/'operational/system_events_20261015.json'
    if syslog.exists(): sources[str(syslog.relative_to(OUT))]=sha(syslog)
    summary={'utc':utc(),'campaign_created_utc':start.isoformat(),'campaign_deadline_utc':deadline.isoformat(),
             'elapsed_wall_hours':(now-start).total_seconds()/3600,
             'recorded_termination_count':len(rows),'recorded_matlab_ready_or_reattached_count':len(ready),
             'completed_termination_to_ready_intervals_hours':sum(r['seconds_until_matlab_ready'] for r in rows if r['seconds_until_matlab_ready'] is not None)/3600,
             'unrecovered_terminations':[r['recorded_termination_utc'] for r in rows if r['next_matlab_ready_utc'] is None],
             'rows':rows,'source_sha256':sources,
             'scope':'Intervals from recorded controller termination to the next recorded MATLAB-ready event. These include cleanup/startup and are not a complete CPU utilization or sleep-duration measurement. Wall budget includes these intervals and is not extended.',
             'power_event_evidence':str(syslog.relative_to(OUT)) if syslog.exists() else None,
             'request_timer_documentation':'https://learn.microsoft.com/en-us/windows/win32/api/realtimeapiset/nf-realtimeapiset-queryunbiasedinterrupttime'}
    csv(OUT/'operational/termination_to_ready_intervals.csv',rows)
    dump(OUT/'operational/runtime_history.json',summary)
    return summary


if __name__=='__main__':
    result=summarize_runtime()
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','source_sha256')},ensure_ascii=True,indent=2))

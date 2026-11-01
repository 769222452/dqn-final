"""Live process ownership, immutable request records, and graceful campaign stopping."""
from __future__ import annotations
import os, subprocess, time, uuid, ctypes
import psutil
import pandas as pd
from exp_common import *

class StopRequested(Exception): pass
class BudgetReached(Exception): pass
class WorkerUnavailable(RuntimeError): pass

# Per-request execution limits measure awake time. The independent campaign
# deadline in run.py remains wall-clock based and is never extended by sleep.
_unbiased_time = None
if os.name == 'nt':
    _clock_kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    _unbiased_time = _clock_kernel.QueryUnbiasedInterruptTime
    _unbiased_time.argtypes = [ctypes.POINTER(ctypes.c_ulonglong)]
    _unbiased_time.restype = ctypes.c_int
REQUEST_CLOCK_SOURCE = 'Windows QueryUnbiasedInterruptTime' if _unbiased_time else 'time.monotonic'

def active_time():
    if _unbiased_time is None:
        return time.monotonic()
    value = ctypes.c_ulonglong()
    if not _unbiased_time(ctypes.byref(value)):
        raise ctypes.WinError(ctypes.get_last_error())
    return value.value / 10_000_000.0

class CampaignLock:
    def __enter__(self):
        import msvcrt
        OUT.mkdir(parents=True,exist_ok=True)
        self.f=(OUT/'campaign.lock').open('a+b')
        try:
            self.f.seek(0)
            if not self.f.read(1): self.f.write(b'0'); self.f.flush()
            self.f.seek(0); msvcrt.locking(self.f.fileno(),msvcrt.LK_NBLCK,1)
        except OSError: self.f.close(); raise RuntimeError('Campaign already running (OS file lock held)')
        dump(OUT/'owner.json',{'pid':os.getpid(),'create_time':psutil.Process().create_time(),'utc':utc()})
        return self
    def __exit__(self,*args):
        import msvcrt
        self.f.seek(0); msvcrt.locking(self.f.fileno(),msvcrt.LK_UNLCK,1); self.f.close()

class Worker:
    def __init__(self):
        # Reattach to a live worker after controller interruption; never restart on a stale observation.
        live=[]
        for p in (OUT/'workers').glob('*/ready.json'):
            old=load(p); pid=old.get('pid')
            try:
                proc=psutil.Process(pid)
                if 'matlab' in proc.name().lower() and p.parent.as_posix().lower() in ' '.join(proc.cmdline()).replace('\\','/').lower():
                    live.append((p.parent,pid))
            except psutil.NoSuchProcess: pass
            except psutil.AccessDenied as exc: raise RuntimeError(f'Cannot verify ownership of registered MATLAB PID {pid}') from exc
        if len(live)>1: raise RuntimeError('More than one campaign MATLAB worker is live; resolve ownership before dispatch')
        if live:
            self.dir,self.pid=live[0]; self.proc=None
            self.seq=max([int(p.stem.split('_')[1]) for p in self.dir.glob('request_*.json')],default=0)
            self.log=None
            event('MATLAB_REATTACHED',pid=self.pid,worker=self.dir)
            return
        self.dir=OUT/'workers'/f'{int(time.time())}_{uuid.uuid4().hex[:8]}'
        self.dir.mkdir(parents=True); self.seq=0; self.pid=None
        self.log=(self.dir/'matlab.log').open('w',encoding='utf-8')
        exe=os.environ.get('V6_MATLAB_EXE','D:/matlab/bin/matlab.exe')
        code=f"addpath('{TOOLS.as_posix()}'); exp_worker('{self.dir.as_posix()}','{ROOT.as_posix()}');"
        self.proc=subprocess.Popen([exe,'-batch',code],cwd=ROOT,stdout=self.log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        start=active_time()
        while not (self.dir/'ready.json').exists():
            if self.proc.poll() is not None: raise RuntimeError(f'MATLAB exited: {self.dir}')
            if active_time()-start>300: self.kill(); raise RuntimeError('MATLAB startup timeout')
            time.sleep(.2)
        self.pid=load(self.dir/'ready.json')['pid']; event('MATLAB_READY',pid=self.pid,worker=self.dir,request_clock=REQUEST_CLOCK_SOURCE)
    def alive(self):
        try:
            if self.pid:
                p=psutil.Process(self.pid)
                return 'matlab' in p.name().lower() and self.dir.as_posix().lower() in ' '.join(p.cmdline()).replace('\\','/').lower()
            return self.proc is not None and self.proc.poll() is None
        except psutil.NoSuchProcess: return False
    def call(self,job,timeout=1800):
        self.seq+=1; rid=f'{self.seq:07d}'; req=self.dir/f'request_{rid}.json'; resp=self.dir/f'response_{rid}.json'
        dump(req,job); start=active_time(); wall_start=time.monotonic(); heartbeat_wall=wall_start
        while not resp.exists():
            if not self.alive(): raise WorkerUnavailable(f'MATLAB worker died during {req}')
            elapsed=active_time()-start; wall_now=time.monotonic(); wall_elapsed=wall_now-wall_start
            if elapsed>timeout:
                self.kill(); event('REQUEST_TIMEOUT',request=req,elapsed_s=elapsed,wall_elapsed_s=wall_elapsed,request_clock=REQUEST_CLOCK_SOURCE)
                raise WorkerUnavailable(f'MATLAB explicit per-request execution budget exhausted: {req}')
            if wall_now-heartbeat_wall>30:
                dump(OUT/'heartbeat.json',{'utc':utc(),'owner_pid':os.getpid(),'matlab_pid':self.pid,'request':req,'elapsed_s':elapsed,'wall_elapsed_s':wall_elapsed,'excluded_system_sleep_s':max(0.,wall_elapsed-elapsed),'request_clock':REQUEST_CLOCK_SOURCE})
                heartbeat_wall=wall_now
            time.sleep(.2)
        r=load(resp)
        if r['status']!='PASS': raise RuntimeError(f'MATLAB request failed: {r}')
        return r
    def kill(self):
        owned=[]
        for pid in set(p for p in [self.pid,self.proc.pid if self.proc else None] if p):
            try:
                proc=psutil.Process(pid)
                if 'matlab' in proc.name().lower() and self.dir.as_posix().lower() in ' '.join(proc.cmdline()).replace('\\','/').lower():
                    owned.append(proc)
                    result=subprocess.run(['taskkill','/PID',str(pid),'/T','/F'],capture_output=True)
                    event('MATLAB_TERMINATE_REQUESTED',pid=pid,returncode=result.returncode)
            except psutil.NoSuchProcess: pass
        if owned:
            _,pending=psutil.wait_procs(owned,timeout=15)
            event('MATLAB_TERMINATION_OBSERVED',requested_pids=[p.pid for p in owned],pending_pids=[p.pid for p in pending])
    def restart_after_terminal_failure(self):
        if self.alive(): raise RuntimeError('Refusing restart while original MATLAB is still live')
        old=self.dir
        if self.log: self.log.close()
        self.__init__(); event('MATLAB_RECOVERED',old_worker=old,new_worker=self.dir)
    def close(self):
        (self.dir/'stop').write_text('stop',encoding='utf-8')
        start=active_time()
        while self.alive() and active_time()-start<25: time.sleep(.2)
        if self.alive(): self.kill()
        if self.log: self.log.close()
    def generate(self,case):
        d=Path(case['case_dir']); d.mkdir(parents=True,exist_ok=True)
        if not (d/'candidate.json').exists(): dump(d/'candidate.json',case)
        if not (d/'generation_status.json').exists() and (d/'started.json').exists():
            old=load(d/'started.json')
            if old.get('pid')!=self.pid:
                # The previous process is authoritatively absent (otherwise Worker reattaches).
                # Preserve the attempted call instead of silently executing the same inputs twice.
                dump(d/'generation_status.json',{'status':'GENERATION_FAILED','error_message':'Interrupted prior MATLAB process; saved attempt retained, no automatic duplicate generation','interrupted':True})
        if not (d/'generation_status.json').exists():
            event('TEACHER_DISPATCH',candidate=case['geometry_id'],inputs10=case['inputs10'],phase=case['phase'])
            try:
                self.call({'operation':'generate','case_dir':d.as_posix(),'inputs10':case['inputs10']},timeout=600)
            except WorkerUnavailable as exc:
                if not (d/'started.json').exists(): raise
                if not (d/'generation_status.json').exists():
                    dump(d/'generation_status.json',{'status':'GENERATION_FAILED','interrupted':True,'error_message':str(exc),'note':'Execution failure, not proof that this parameter tuple is physically invalid; no duplicate geometry call'})
                self.restart_after_terminal_failure()
    def evaluate(self,gid,mat,pairs,dest):
        dest=Path(dest)
        if dest.exists(): return pd.read_csv(dest)
        dest.parent.mkdir(parents=True,exist_ok=True)
        q=dest.with_suffix('.queries.csv')
        csv(q,[{'geometry_id':gid,'mat_path':str(mat),'Mach':m,'alpha_deg':a} for m,a in pairs])
        for attempt in range(2):
            try:
                self.call({'operation':'evaluate','query_csv':q.as_posix(),'out_csv':dest.as_posix(),'log_file':dest.with_suffix('.log').as_posix()},timeout=max(600,len(pairs)*3))
                break
            except WorkerUnavailable:
                self.restart_after_terminal_failure()
                if dest.exists(): break
                if attempt==1: raise
                event('P2_INFRASTRUCTURE_RETRY',geometry=gid,output=dest,attempt=attempt+2)
        result=pd.read_csv(dest)
        if len(result)!=len(pairs): raise RuntimeError(f'Incomplete P2 output: {dest}')
        return result

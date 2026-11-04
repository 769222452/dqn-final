"""Regression checks for suspend-aware timeouts and observed process termination.

These control-flow tests never call MATLAB, create geometry, or generate P2 data.
"""
from __future__ import annotations
from types import SimpleNamespace
from unittest.mock import Mock, patch
import runtime
from exp_common import *


def main():
    config_digest=sha(OUT/'config.json')
    dest=OUT/'validation_runtime_timing'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    dest.mkdir(parents=True)
    assert dest.resolve().is_relative_to(OUT.resolve())
    first=runtime.active_time(); runtime.time.sleep(.06); second=runtime.active_time()
    assert 0 < second-first < 2

    worker=runtime.Worker.__new__(runtime.Worker)
    worker.dir=dest/'control_flow_only'; worker.dir.mkdir()
    worker.seq=0; worker.pid=-1
    worker.alive=Mock(return_value=True)
    worker.kill=Mock(side_effect=AssertionError('A sleep gap must not terminate a live worker'))
    def commit_control_response(_):
        dump(worker.dir/'response_0000001.json',{'status':'PASS','operation':'CONTROL_FLOW_TEST_ONLY','no_physical_computation':True})
    # 3 hours of wall-clock jump with only 1 second of awake execution.
    with patch.object(runtime,'OUT',dest), patch.object(runtime,'active_time',side_effect=[100.,101.]), \
         patch.object(runtime.time,'monotonic',side_effect=[200.,11001.]), \
         patch.object(runtime.time,'sleep',side_effect=commit_control_response):
        response=worker.call({'operation':'CONTROL_FLOW_TEST_ONLY'},timeout=10)
    assert response['no_physical_computation']
    worker.kill.assert_not_called()
    heartbeat=load(dest/'heartbeat.json')
    assert heartbeat['elapsed_s']==1 and heartbeat['wall_elapsed_s']==10801
    assert heartbeat['excluded_system_sleep_s']==10800

    expired=runtime.Worker.__new__(runtime.Worker)
    expired.dir=dest/'expired_control_flow_only'; expired.dir.mkdir()
    expired.seq=0; expired.pid=-3; expired.alive=Mock(return_value=True); expired.kill=Mock()
    with patch.object(runtime,'active_time',side_effect=[100.,111.]), \
         patch.object(runtime.time,'monotonic',side_effect=[200.,211.]), patch.object(runtime,'event'):
        try:
            expired.call({'operation':'CONTROL_FLOW_TEST_ONLY'},timeout=10)
        except runtime.WorkerUnavailable:
            pass
        else:
            raise AssertionError('A genuine awake-time timeout must still raise')
    expired.kill.assert_called_once()

    # Termination must be observed before a caller can attempt recovery.
    victim=runtime.Worker.__new__(runtime.Worker)
    victim.dir=dest/'mock_worker'; victim.pid=-2; victim.proc=None
    process=SimpleNamespace(pid=-2,name=lambda:'MATLAB.exe',cmdline=lambda:[victim.dir.as_posix()])
    order=[]
    def terminate(*args,**kwargs):
        order.append('termination_requested'); return SimpleNamespace(returncode=0)
    def confirm(procs,timeout):
        assert procs==[process] and timeout==15
        order.append('termination_observed'); return [process],[]
    with patch.object(runtime.psutil,'Process',return_value=process), \
         patch.object(runtime.subprocess,'run',side_effect=terminate), \
         patch.object(runtime.psutil,'wait_procs',side_effect=confirm), patch.object(runtime,'event'):
        victim.kill()
    assert order==['termination_requested','termination_observed']
    assert sha(OUT/'config.json')==config_digest

    result={'utc':utc(),'status':'PASS','request_clock':runtime.REQUEST_CLOCK_SOURCE,
            'real_clock_advanced_seconds':second-first,
            'wall_sleep_gap_did_not_trigger_request_timeout':True,
            'genuine_awake_timeout_still_terminates_request':True,
            'termination_request_followed_by_process_exit_observation':True,
            'campaign_budget_config_unchanged':True,'campaign_config_sha256':config_digest,
            'runtime_sha256':sha(Path(runtime.__file__)),
            'test_directory':str(dest),
            'scope':'Runtime control-flow simulation only. No MATLAB, geometry, CL/CD, or P2 result is produced.'}
    dump(OUT/'validation_runtime_timing.json',result)
    print(json.dumps(result,ensure_ascii=True,indent=2))


if __name__=='__main__':
    main()

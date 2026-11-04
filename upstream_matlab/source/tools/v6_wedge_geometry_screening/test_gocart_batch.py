"""Isolated MOCK tests of GoCart_Batch_Win7 queue logic (host Windows, cscript/cmd). NO GoCart, NO CFD.
The mock adapter writes made-up numbers ONLY inside temporary test folders (labelled MOCK); they never enter the package.
"""
import csv, json, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / 'results/v6_wedge_geometry_screening/GoCart_Batch_Win7'
WORK = ROOT / 'tools/v6_wedge_geometry_screening/test_work'
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir()
T = WORK / 'pkg'
LOG = []

MOCK = r'''@echo off
setlocal EnableExtensions DisableDelayedExpansion
call "%~1"
if "%MOCKMODE%"=="exit17" exit /b 17
if "%MOCKMODE%"=="sleep" ping -n 30 127.0.0.1 >nul
if "%MOCKMODE%"=="noresult" exit /b 0
set "MACH=%GC_MACH%"
if "%MOCKMODE%"=="wrongmach" set "MACH=9.0"
set "CL=0.0400"
set "CD=0.0030"
set "CONV=YES"
set "KIND=PRESSURE_ONLY"
set "SIG=%GC_REQUEST_SIGNATURE%"
set "HASH=%GC_STEP_SHA256%"
if "%MOCKMODE%"=="nan" set "CL=NaN"
if "%MOCKMODE%"=="negcd" set "CD=-0.003"
if "%MOCKMODE%"=="notconv" set "CONV=NO"
if "%MOCKMODE%"=="badhash" set "HASH=0000000000000000000000000000000000000000000000000000000000000000"
if "%MOCKMODE%"=="badsig" set "SIG=deadbeef"
if "%MOCKMODE%"=="badkind" set "KIND=WEDGE"
if "%MOCKMODE%"=="driftadapter" echo rem drift>>"%GC_PACKAGE_ROOT%\scripts\adapter_gocart.bat"
echo MOCK ONLY - NOT CFD> mock_input.txt
echo MOCK mach=%MACH%>> mock_input.txt
echo MOCK force output - NOT CFD> mock_output.txt
echo MOCK convergence history - NOT CFD> mock_conv.txt
if "%MOCKMODE%"=="noevidence" del mock_conv.txt
(
echo [result]
echo case_id=%GC_CASE_ID%
echo step_sha256=%HASH%
echo mach=%MACH%
echo alpha_deg=%GC_ALPHA_DEG%
echo sref_m2=%GC_SREF_M2%
echo lref_m=%GC_LREF_M%
echo length_unit=mm
echo force_frame=WIND_AXES
echo aero_source=CFD
echo coefficient_kind=%KIND%
echo cl=%CL%
echo cd=%CD%
echo converged=%CONV%
echo solver_name=MOCK_TEST_ONLY
echo solver_version=NOT_REAL_CFD
echo mesh_cells=1000
echo input_signature=%SIG%
echo input_evidence=mock_input.txt
echo raw_output_evidence=mock_output.txt
echo convergence_evidence=mock_conv.txt
) > result.ini
exit /b 0
'''


def rec(name, ok, detail=''):
    LOG.append({'test': name, 'pass': bool(ok), 'detail': detail})
    print('PASS' if ok else 'FAIL', name, detail[:150], flush=True)
    if not ok:
        raise SystemExit(1)


def run(bat, args=(), mode='', timeout=120):
    env = dict(os.environ, NOPAUSE='1', MOCKMODE=mode)
    t = time.time()
    p = subprocess.run(['cmd.exe', '/c', str(T / bat)] + list(args), capture_output=True, text=True, errors='replace', env=env, timeout=timeout, cwd=str(T))
    return p.returncode, p.stdout + p.stderr, time.time() - t


def results():
    p = T / 'results/CFD_RESULTS.csv'
    return list(csv.DictReader(p.open(encoding='utf-8-sig'))) if p.exists() else []


def setcfg(**kw):
    txt = (T / 'config.ini').read_text()
    out = []
    for ln in txt.splitlines():
        k = ln.split('=')[0].strip() if '=' in ln else None
        out.append(f'{k}={kw[k]}' if k in kw else ln)
    (T / 'config.ini').write_text('\n'.join(out) + '\n')


# ---- build isolated copy with 3 real cases (2 same geometry + 1 other geometry)
shutil.copytree(PKG, T, ignore=shutil.ignore_patterns('geometries', 'results', 'logs', 'gocart_environment.txt', 'cases_full_343.csv'))
(T / 'results').mkdir(); (T / 'logs').mkdir()
full = list(csv.DictReader((PKG / 'cases_full_343.csv').open(encoding='ascii')))
pick = [r for r in full if r['case_id'] in ('S2_MD14.004_4b1ceeb42b__M6p50_A07p00', 'S2_MD14.004_4b1ceeb42b__M6p50_A08p00', 'G42__M6p50_A07p00')]
(T / 'geometries').mkdir()
for g in {r['geometry_id'] for r in pick}:
    shutil.copyfile(PKG / 'geometries' / (g + '.step'), T / 'geometries' / (g + '.step'))
with (T / 'cases3.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(pick[0]), lineterminator='\r\n'); w.writeheader(); w.writerows(pick)
ID1, ID2, ID3 = [r['case_id'] for r in pick]

# 1 shipped state
rc, out, _ = run('RUN_ALL.bat', ['dryrun'])
rec('dryrun_passes_and_reports_not_configured', rc == 0 and 'PASS: 2 cases' in out and 'NOT READY TO RUN CFD' in out, out.strip().splitlines()[-4])
rc, out, _ = run('RUN_ONE.bat')
rec('shipped_package_refuses_to_run_one', rc == 2 and 'CONFIG_REQUIRED' in out and not (T / 'results/runs').exists())
rc, out, _ = run('RUN_ALL.bat')
rec('shipped_package_refuses_to_run_all', rc == 2 and not (T / 'results/runs').exists())
rec('shipped_adapter_is_placeholder', 'GOCART_ADAPTER_UNCONFIGURED' in (PKG / 'scripts/adapter_gocart.bat').read_text())

# 2 configure with MOCK adapter
(T / 'scripts/adapter_gocart.bat').write_text(MOCK)
(T / 'fake_dep.txt').write_text('stand-in for solver binary')
(T / 'fake_install').mkdir()
setcfg(configured='YES', physics_reviewed='YES', install_dir=str(T / 'fake_install'), dependency_files=str(T / 'fake_dep.txt'), case_timeout_seconds='60', busy_process_names='no_such_solver.exe')
rc, out, _ = run('RUN_ONE.bat')
rec('mock_case_succeeds', rc == 0 and 'OK    ' + ID1 in out, out.strip().splitlines()[-5])
r = results()
row = [x for x in r if x['case_id'] == ID1][0]
rec('results_csv_fields', row['status'] == 'SUCCEEDED' and row['CL'] == '0.0400' and row['CD'] == '0.0030' and abs(float(row['L_over_D']) - 13.333333) < 1e-5 and row['coefficient_kind'] == 'PRESSURE_ONLY',
    str(row))
rec('results_csv_header_has_required_columns', all(c in r[0] for c in 'case_id,geometry_id,mach,alpha_deg,CL,CD,L_over_D,status,runtime_seconds'.split(',')))

# 3 resume skip, no new attempt
n = len(list((T / 'results/runs').glob('*/attempt_*')))
rc, out, _ = run('RUN_ONE.bat')
rec('resume_skips_valid_success', rc == 0 and 'SKIP' in out and len(list((T / 'results/runs').glob('*/attempt_*'))) == n)

# 4 full-batch gate and run all
setcfg(test_case_limit='2')
rc, out, _ = run('RUN_ALL.bat', ['cases3.csv'])
rec('full_batch_gate_blocks_3_cases', rc == 2 and 'FULL_BATCH_NOT_ALLOWED' in out)
setcfg(allow_full_batch='YES')
rc, out, _ = run('RUN_ALL.bat', ['cases3.csv'])
st = {x['case_id']: x['status'] for x in results()}
rec('run_all_completes_and_skips_done', rc == 0 and set(st.values()) == {'SUCCEEDED'} and len(st) == 3 and 'SKIP  ' + ID1 in out, str(st))

# 5 tamper: change evidence of case 2 -> not trusted, re-run
a2 = sorted((T / 'results/runs' / ID2).glob('attempt_*'))[-1]
(a2 / 'mock_output.txt').write_text('TAMPERED')
rc, out, _ = run('RUN_ALL.bat', ['cases3.csv'])
rec('tampered_evidence_triggers_rerun', rc == 0 and 'RUN   ' + ID2 in out and len(list((T / 'results/runs' / ID2).glob('attempt_*'))) == 2)

# 6 failures (each in a fresh run folder state for the case)
def fail_case(mode, expect):
    cdir = T / 'results/runs' / ID3
    if cdir.exists():
        shutil.rmtree(cdir)
    rc, out, _ = run('RUN_ONE.bat', [ID3, 'cases3.csv'], mode)
    rows = {x['case_id']: x for x in results()}
    rec('reject_' + mode, rc == 1 and rows[ID3]['status'] == 'FAILED' and rows[ID3]['CL'] == '' and expect in rows[ID3]['message'], rows[ID3]['message'])
for mode, exp in [('exit17', 'exit code 17'), ('noresult', 'result.ini missing'), ('wrongmach', 'Mach differs'), ('nan', 'not finite'),
                  ('negcd', 'CD must be'), ('notconv', 'converged=YES'), ('badhash', 'geometry hash'), ('badsig', 'signature'),
                  ('badkind', 'aero_source') if False else ('badkind', 'coefficient_kind'), ('noevidence', 'file missing')]:
    fail_case(mode, exp)

# 7 timeout kills only own tree
shutil.rmtree(T / 'results/runs' / ID3)
setcfg(case_timeout_seconds='3')
rc, out, el = run('RUN_ONE.bat', [ID3, 'cases3.csv'], 'sleep')
rows = {x['case_id']: x for x in results()}
rec('timeout_marks_failed_fast', rc == 1 and 'CASE_TIMEOUT' in rows[ID3]['message'] and el < 25, f'{el:.1f}s')
setcfg(case_timeout_seconds='60')

# 8 failure pauses queue / retry
shutil.rmtree(T / 'results/runs' / ID1); shutil.rmtree(T / 'results/runs' / ID2); shutil.rmtree(T / 'results/runs' / ID3)
os.environ['MOCKMODE'] = ''
rc, out, _ = run('RUN_ALL.bat', ['cases3.csv'], 'exit17')
rows = {x['case_id']: x['status'] for x in results()}
rec('queue_pauses_at_first_failure', rc == 1 and rows[ID1] == 'FAILED' and rows[ID2] == 'PENDING' and rows[ID3] == 'PENDING', str(rows))
rc, out, _ = run('RUN_ALL.bat', ['cases3.csv'], '')
rows = {x['case_id']: x['status'] for x in results()}
rec('failed_not_retried_without_retry_flag', rc == 1 and rows[ID1] == 'FAILED' and 'use RUN_ALL.bat retry' in out, str(rows))
rc, out, _ = run('RUN_ALL.bat', ['cases3.csv', 'retry'], '')
rows = {x['case_id']: x['status'] for x in results()}
rec('retry_recovers_then_all_succeed', rc == 0 and set(rows.values()) == {'SUCCEEDED'}, str(rows))

# 9 workflow change invalidates acceptance
(T / 'case_template/extra_input.txt').write_text('changed template')
rc, out, _ = run('RUN_ALL.bat', ['cases3.csv', 'dryrun'])
rec('template_change_invalidates_previous_success', rc == 0 and '0 already valid' in out, out.strip().splitlines()[-3])
os.remove(T / 'case_template/extra_input.txt')
rc, out, _ = run('RUN_ALL.bat', ['cases3.csv', 'dryrun'])
rec('same_workflow_still_valid', '3 already valid' in out)

# 10 drift during run
for p in (T / 'results/runs').iterdir():
    shutil.rmtree(p)
rc, out, _ = run('RUN_ALL.bat', ['cases3.csv'], 'driftadapter')
rows = {x['case_id']: x['status'] for x in results()}
rec('adapter_drift_stops_before_next_case', rc == 2 and 'WORKFLOW_CHANGED_DURING_RUN' in out and rows[ID2] == 'PENDING', str(rows))
(T / 'scripts/adapter_gocart.bat').write_text(MOCK)

# 11 STEP corruption blocks
step = T / 'geometries/G42.step'
backup = step.read_bytes(); step.write_bytes(backup + b'\nCORRUPT\n')
rc, out, _ = run('RUN_ALL.bat', ['cases3.csv', 'dryrun'])
rec('step_hash_mismatch_blocks', rc == 2 and 'hash differs' in out)
step.write_bytes(backup)

# 12 busy process refuse
setcfg(busy_process_names='cscript.exe')
for p in (T / 'results/runs').iterdir():
    shutil.rmtree(p)
rc, out, _ = run('RUN_ONE.bat')
rec('busy_process_blocks_start_and_nothing_killed', rc == 2 and 'busy_process_names' in out and not list((T / 'results/runs').glob('*/attempt_*')))
setcfg(busy_process_names='no_such_solver.exe')

# 13 stop file
(T / 'STOP_AFTER_CURRENT.txt').write_text('x')
rc, out, _ = run('RUN_ALL.bat', ['cases3.csv'])
rec('stop_file_stops_before_next_case', rc == 3 and not list((T / 'results/runs').glob('*/attempt_*')))
os.remove(T / 'STOP_AFTER_CURRENT.txt')

# 14 lock: second instance refused while the first is running
setcfg(case_timeout_seconds='60')
for p in (T / 'results/runs').iterdir():
    shutil.rmtree(p)
env = dict(os.environ, NOPAUSE='1', MOCKMODE='sleep')
first = subprocess.Popen(['cmd.exe', '/c', str(T / 'RUN_ONE.bat')], env=env, cwd=str(T), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(4)
rc, out, _ = run('RUN_ONE.bat')
rec('second_instance_refused_by_lock', rc == 2 and 'already running' in out, out.strip().splitlines()[0] if out.strip() else '')
subprocess.run(['taskkill', '/PID', str(first.pid), '/T', '/F'], capture_output=True)
time.sleep(1)

# 15 unsafe case id
bad = (T / 'cases3.csv').read_text().replace(ID1, 'bad id;x', 1)
(T / 'cases_bad.csv').write_text(bad)
rc, out, _ = run('RUN_ALL.bat', ['cases_bad.csv', 'dryrun'])
rec('unsafe_case_id_rejected', rc == 2 and 'Unsafe case_id' in out)

# 16 a crashing queue script must not look like success
vb = (T / 'scripts/gc_runner.vbs').read_text()
(T / 'scripts/gc_runner.vbs').write_text(vb.replace('Dim reRe:', 'Call NoSuchProcedure()\r\nDim reRe:', 1))
rc, out, _ = run('RUN_ALL.bat', ['cases3.csv', 'dryrun'])
rec('script_crash_gives_nonzero_exit', rc == 99 and 'crashed' in out, f'rc={rc}')
(T / 'scripts/gc_runner.vbs').write_text(vb)

rep = {'test_type': 'MOCK_ONLY_NOT_CFD', 'host': 'Windows 11 (current PC), cmd + cscript', 'windows7_vm_tested': False,
       'real_gocart_executed': False, 'tests_passed': len(LOG), 'tests': LOG}
(PKG.parent / 'logs/gocart_batch_mock_selftest.json').write_text(json.dumps(rep, indent=1), encoding='utf-8')
shutil.rmtree(WORK)
print('ALL PASS', len(LOG))

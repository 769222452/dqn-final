"""Preserve and package an aborted check, without rerunning or changing any experiment."""
from pathlib import Path
import collections
import csv
import hashlib
import json
import shutil
import subprocess
import zipfile

REPO=Path('/hy-tmp/dqn-aero')
OPS=Path('/hy-tmp/aero_operations_4d211b1')
OUT=REPO/'runs/aero_fidelity'
SHA='4d211b1b9cc282bdf7716c6e03ff5c5156c9d4a7'

def digest(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==SHA
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=REPO,text=True).strip()
assert not (OUT/'family_selected/family_audit.json').exists()
check=(OPS/'check.log').read_text()
assert 'ear clipping failed (non-simple base polygon)' in check
statuses=[json.loads(p.read_text()) for p in sorted((OUT/'family').glob('FAM_*/status.json'))]
assert len(statuses)==120
run=json.loads((OUT/'family/FAMILY_RUN.json').read_text())
counts=dict(collections.Counter(s['status'] for s in statuses))
failures=[s for s in statuses if s['status']!='GENERATED']
errors=dict(collections.Counter(s['error_message'] for s in failures))
summary={'commit':SHA,'branch':'aoa-fix','pytest':'8 passed in 2.33s',
         'family_counts':counts,'family_wall_s':run['wall_s'],
         'family_execution':'Local licensed MATLAB R2023a; CODEX.md section 3 other-machine fallback',
         'worker_exit_codes':run['worker_exit_codes'],'failure_counts':errors,
         'check_status':'ABORTED_WITH_EXCEPTION','check_exit_code':1,
         'check_error':'ear clipping failed (non-simple base polygon)',
         'family_audit_created':False,'select_created':False,'selection_verdict':None,
         'cfd_executed':False,'criteria_or_generator_modified':False}
(OUT/'RUN_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
(OUT/'family_failures.json').write_text(json.dumps(failures,ensure_ascii=False,indent=2)+'\n')
with (OUT/'family_status.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f)
    w.writerow(['job_id','design_mach','status','elapsed_s','nonfinite','error_identifier','error_message'])
    for s in statuses:
        w.writerow([s['job_id'],s['inputs10'][0],s['status'],s['elapsed_s'],s.get('nonfinite',''),s.get('error_identifier',''),s.get('error_message','')])
report=f'''# 外形族实验前半段结果

- 分支：aoa-fix；提交：`{SHA}`。
- 指定测试：8 项全部通过（2.33 s）。
- 云端：Python 3.11.10、SU2 8.5.0（conda-forge）、Open MPI 5.0.11、gmsh 4.15.2。
- 云端无 MATLAB；按 CODEX.md 第 3 条允许的方式，在本机 MATLAB R2023a（9.14.0.2206163）生成，Optimization / Symbolic Math 工具箱许可证均通过验证，再将原始数据传回云端运行 check。
- 云端 nproc（未设置 OpenMP 线程限制时）=80；容器 CPU quota/period=1000000/100000，相当于 10 核，内存上限 16642998272 字节。原脚本设置 OMP_NUM_THREADS=1 后显示 nproc=1，不能视为主机核心数。
- 参数保持 NP=7、JOBS=2、SLICES=4；本次仅启动 4 路本机 MATLAB，未运行 MPI/CFD。
- 120 组全部有记录：生成成功 {counts.get('GENERATED',0)} 组，生成失败 {counts.get('GENERATION_FAILED',0)} 组；外形生成阶段墙钟耗时 {run['wall_s']:.3f} s（{run['wall_s']/60:.3f} min）。四个 MATLAB 进程均正常退出。成功输出的非有限坐标数均为 0。
- 完整失败条目在 `family_failures.json`，全部任务在 `family_status.csv`。

| 原始错误信息 | 组数 |
|---|---:|
'''
for message,n in errors.items(): report+=f'| `{message}` | {n} |\n'
report+='''
## check 结果

原始 check 在闭合底面三角剖分时异常退出，尚未写出任何正式选择判定。`family_audit.json` 和 `select.json` 未生成，`family_selected/` 不存在。
这不是 `NO_ADMISSIBLE_FAMILY` 判定；不能据此宣称不存在满足体积规则的外形族。
没有修改生成器、扫描值、选择规则或 ±3% 容许，没有重跑、跳过失败输入或运行 cfd-mesh / cfd-run / collect。

完整输出：

```text
'''+check+'```\n'
(OUT/'RUN_SUMMARY.md').write_text(report,encoding='utf-8')
metadata=OUT/'operations'
metadata.mkdir(exist_ok=False)
for name in ['environment.txt','pip_freeze.txt','pytest.log','preflight.log','preflight_initial.log','preflight_exit_code.txt',
             'check.log','check_and_package.log','check_and_package.exit','apt_install.log','su2_install.log','python_install.log',
             'cloud_prepare.sh','cloud_finish.py','cloud_package_failure.py']:
    shutil.copy2(OPS/name,metadata/name)
with (metadata/'upstream_verification.log').open('w') as log:
    subprocess.run([str(REPO/'.venv/bin/python'),'verify_snapshot.py'],cwd=REPO/'upstream_matlab',check=True,stdout=log)
for p in (OUT/'family').glob('FAM_*/status.json'):
    s=json.loads(p.read_text())
    if s['status']=='GENERATED': assert s['nonfinite']==0
hashes={p.relative_to(OUT).as_posix():digest(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
(OUT/'SHA256SUMS').write_text(''.join(f'{h}  {name}\n' for name,h in hashes.items()),encoding='utf-8')
archive=OPS/'aero_family_4d211b1_check_failed.zip'
assert not archive.exists()
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
    for p in sorted(OUT.rglob('*')):
        if p.is_file(): z.write(p,arcname=p.relative_to(OUT).as_posix())
record={'zip':str(archive),'sha256':digest(archive),'bytes':archive.stat().st_size,'summary':summary}
(OPS/'delivery_failure.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(record,ensure_ascii=False))

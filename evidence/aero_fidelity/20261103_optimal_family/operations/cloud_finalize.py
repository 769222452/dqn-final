"""Package the optimal-family evidence. Original experiment outputs remain untouched."""
from pathlib import Path
import collections,csv,datetime,hashlib,json,os,shutil,subprocess,zipfile
REPO=Path('/hy-tmp/dqn-aero-opt-f248c4f'); OPS=Path('/hy-tmp/aero_opt_operations_20261103')
SHA='f248c4f7da03a48335cbe1dfca4e2b6e0e58c7da'; PKG=OPS/'delivery_files'
def read(p): return json.loads(p.read_text())
def digest(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def copy(p,dest): dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,dest)
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==SHA
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=REPO,text=True).strip()
stages={p.name[:-11]:read(p) for p in OPS.glob('*_state.json')}
decision_path=REPO/'runs/aero_opt/results/optimum_decision.json'; decision=read(decision_path) if decision_path.exists() else None
assert stages and not any(s['status']=='RUNNING' for s in stages.values())
assert any(s['status']=='FAILED' for s in stages.values()) or (decision and decision['status']!='EVALUATED') or stages.get('confirm',{}).get('status')=='COMPLETED'
assert not PKG.exists(); PKG.mkdir()
families={}
for letter in ('a','b'):
    root=REPO/f'runs/aero_opt_{letter}'; fam=root/'family'
    if not fam.exists(): continue
    selected=root/'family_selected'; audit=read(selected/'family_audit.json') if (selected/'family_audit.json').exists() else None
    if selected.exists(): shutil.copytree(selected,PKG/f'family_{letter}_selected')
    valid={r['job_id'] for r in audit['rows'] if r['valid']} if audit else set()
    statuses=[]
    for p in sorted(fam.rglob('*')):
        if not p.is_file(): continue
        keep=p.name=='status.json' or p.parent==fam
        if letter=='b' and p.name=='native_geometry.mat' and p.parent.name in valid: keep=True
        if keep: copy(p,PKG/f'family_{letter}'/p.relative_to(fam))
        if p.name=='status.json': statuses.append(read(p))
    failures=[s for s in statuses if s['status']!='GENERATED']
    run=read(fam/'FAMILY_RUN.json')
    families[letter]=dict(jobs=len(statuses),generated=sum(s['status']=='GENERATED' for s in statuses),failed=len(failures),
        failure_categories=dict(collections.Counter(s.get('error_identifier','')+' | '+s.get('error_message','') for s in failures)),
        failed_jobs=failures,generation_wall_s=run['wall_s'],audit_summary={k:audit[k] for k in ('jobs','generated','valid','status','outcomes')} if audit else None)
    with (PKG/f'family_{letter}_status.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['job_id','status','elapsed_s','error_identifier','error_message'],extrasaction='ignore'); w.writeheader(); w.writerows(statuses)
res=REPO/'runs/aero_opt/results'
if res.exists(): shutil.copytree(res,PKG/'results')
CFD=REPO/'runs/aero_opt/cfd_root/cfd'
allowed={'case.json','case.cfg','history.csv','su2.log','mesh_info.json','progress.log','cases_medium.txt','done'}
for p in sorted(CFD.rglob('*')):
    if p.is_file() and p.name in allowed: copy(p,PKG/'cfd'/p.relative_to(CFD))
for p in sorted(OPS.iterdir()):
    if p.is_file() and p.suffix in ('.log','.json','.txt','.py','.sh') and p.name!='delivery.json': copy(p,PKG/'operations'/p.name)
env=os.environ.copy(); env['PYTHONDONTWRITEBYTECODE']='1'
with (PKG/'operations/upstream_after.log').open('w') as f:
    subprocess.run(['/hy-tmp/dqn-aero/.venv/bin/python','verify_snapshot.py'],cwd=REPO/'upstream_matlab',env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
for name in ('README.md','CODEX.md','run_decision.sh','family/family_jobs.py','family/make_family.m','family/check_family.py','optimum/evaluate_optimum.py'):
    copy(REPO/'experiments/aero_fidelity'/name,PKG/'frozen_protocol'/name)
for name in ('make_su2_cases.py','collect_su2.py'): copy(REPO/'scripts/cfd'/name,PKG/'frozen_protocol/scripts/cfd'/name)
convergence=read(OPS/'cfd_convergence.json') if (OPS/'cfd_convergence.json').exists() else None
if convergence:
    with (PKG/'cfd_case_status.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['case_id','done','last_residual','density_pass','cauchy_CL_pass','cauchy_CD_pass','converged_by_frozen_rule','observed_case_wall_s'],extrasaction='ignore'); w.writeheader(); w.writerows(convergence['records'])
confirmation=read(res/'euler_confirmation.json') if (res/'euler_confirmation.json').exists() else None
coverage=read(OPS/'confirmation_coverage.json') if (OPS/'confirmation_coverage.json').exists() else None
summary=dict(commit=SHA,source_branch='aoa-fix',source_modified=False,protocol='README 3h optimal-family test',NP=7,JOBS=2,SLICES=4,
    cpu_quota_cores=10,matlab='local R2023a; Optimization and Symbolic Math license checks in every worker',
    tests='21 passed in 53.81s',stages=stages,families=families,decision=decision,euler_confirmation=confirmation,convergence=convergence,
    confirmation_input_audit=coverage,collect_stage_run=False,packaged_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
write(PKG/'RUN_SUMMARY.json',summary)
md=['# 最优外形检验执行记录','',f'冻结提交：`{SHA}`（aoa-fix）。',
    '', '按 README §3h 执行；原生成器、参数、扫描范围、±3% 体积容许和判据保持不变。第一轮 FAMILY_VERSION=5，第二轮 FAMILY_VERSION=6 使用第一轮完整审计。未运行 collect 阶段。',
    '', '本机 MATLAB R2023a；云端 Python 3.11.10、SU2 8.5.0、Open MPI 5.0.11。SLICES=4，NP=7，JOBS=2；容器 CPU 配额 10 核。相关测试 21 passed。',
    '', '## 生成和冻结审计','','| 轮次 | 任务 | 生成成功 | 生成失败 | 生成墙钟/s | 审计有效 | 审计状态 |','|---|---:|---:|---:|---:|---:|---|']
for letter,r in families.items():
    a=r['audit_summary'] or {}; md.append(f'| {letter.upper()} | {r["jobs"]} | {r["generated"]} | {r["failed"]} | {r["generation_wall_s"]:.3f} | {a.get("valid","—")} | {a.get("status","未完成")} |')
md+=['','MAP_ONLY 表示完成审计并形成映射，不表示实验失败。生成失败及审计无效记录全部保留。','', '## 阶段耗时','','| 阶段 | 状态 | 墙钟/s |','|---|---|---:|']
for name,r in stages.items(): md.append(f'| {name} | {r["status"]} | {r.get("wall_s",0):.3f} |')
for letter,r in families.items():
    md+=['',f'## 第 {letter.upper()} 轮生成失败分类','','```json',json.dumps(r['failure_categories'],ensure_ascii=False,indent=2),'```',
        '',f'全部失败 job_id 和原始信息：family_{letter}/generation_failures.json；逐任务记录：family_{letter}/*/status.json。']
if decision:
    md+=['','## 冻结 evaluate_optimum 原始判定','','```json',json.dumps({k:v for k,v in decision.items() if k not in ('rule','best_LD_by_design_mach')},ensure_ascii=False,indent=2),'```']
if convergence:
    md+=['','## Euler 运行及收敛','',f'工况 {convergence["cases"]}，运行完成 {convergence["done"]}，按冻结条件收敛 {convergence["converged"]}。',
        '', '末密度残差为 SU2 rms[Rho] 的 log10 原值；小于 −5，或 CL/CD 两项 Cauchy 都通过，才计为收敛。求解完成与物理判据分开报告。',
        '', '未收敛工况及末残差：','```json',json.dumps(convergence['not_converged'],indent=2),'```',
        '', '每个工况的非物理解点和重构警告、Cauchy 证据、末 40 行日志见 operations/cfd_convergence.json；完整日志在 cfd/。逐工况时间从日志创建到 done 估算，阶段总墙钟单独记录。']
if confirmation:
    md+=['','## Euler 原始复核结果','','```json',json.dumps({k:v for k,v in confirmation.items() if not k.endswith('_mean_LD')},indent=2),'```',
         '', 'euler_confirmation.json 为冻结脚本原始输出。该脚本使用已覆盖 CL 目标的平均值，不自动要求每个成员覆盖全部三个目标，也不筛除未收敛 history。原始输出与可采信证据必须区分。']
if coverage:
    md+=['',f'实际覆盖核对：所有成员、Mach 和模型均覆盖三个指定 CL：{coverage["all_requested_CL_covered"]}；所有 Euler 输入收敛：{coverage["all_inputs_converged"]}。',
        '', 'MD9、Mach 6 的攻角 4°、6°、8° 三个工况未收敛。该组等升力拟合和排名不能计作已通过的 Euler 验证。原始复核 confirmed=false 保持原样；不将运行完成说成完整物理验证。详见 operations/confirmation_coverage.json。',
        '', '本次按冻结协议 GRID_CHECK=0，仅运行 medium 网格，不能据此声称网格收敛。网格构建的原始回退尝试及实际单元数见 operations/cfd-mesh.log 和 mesh_verification.json。']
for name,r in stages.items():
    if r['status']=='FAILED':
        log=OPS/(name+'.log'); md+=['',f'## {name} 失败现场','','```text', '\n'.join(log.read_text(errors='replace').splitlines()[-40:]) if log.exists() else json.dumps(r), '```']
md+=['', '归档包含两轮全部 status、MATLAB 日志和审计，第二轮所有 VALID 成员 MAT、最优外形原始输出及确认外形、全部 CFD 工况配置/history/log、运维时间与哈希。CFD mesh/restart 不入包，云端原始文件不删除。','']
(PKG/'RUN_SUMMARY.md').write_text('\n'.join(md),encoding='utf-8')
hashes={p.relative_to(PKG).as_posix():digest(p) for p in sorted(PKG.rglob('*')) if p.is_file()}
(PKG/'SHA256SUMS').write_text(''.join(f'{h}  {name}\n' for name,h in hashes.items()))
archive=OPS/'aero_opt_f248c4f_results.zip'; assert not archive.exists()
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
    for p in sorted(PKG.rglob('*')):
        if p.is_file(): z.write(p,p.relative_to(PKG).as_posix())
delivery=dict(zip=str(archive),sha256=digest(archive),bytes=archive.stat().st_size,manifest_files=len(hashes),summary=summary)
write(OPS/'delivery.json',delivery)
print(json.dumps({k:v for k,v in delivery.items() if k!='summary'}))

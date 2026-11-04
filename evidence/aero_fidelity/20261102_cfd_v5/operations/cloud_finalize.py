"""Package the frozen CFD-v5 evidence, with the predeclared v4/v5 diagnostic comparisons."""
from pathlib import Path
import csv,datetime,hashlib,json,os,re,shutil,statistics,subprocess,zipfile
REPO=Path('/hy-tmp/dqn-aero-v5-dc39a7f')
OPS=Path('/hy-tmp/aero_operations_v5_20261102')
OUT=REPO/'runs/aero_fidelity_v5'
V4=Path('/hy-tmp/dqn-aero-v4-01589db/runs/aero_fidelity_v4')
SHA='dc39a7f9e7c3d78f8bc66c660866907759362c24'
PKG=OPS/'delivery_files'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def digest(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,obj): p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert not PKG.exists()
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==SHA
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=REPO,text=True).strip()
stages={p.name[:-11]:read(p) for p in OPS.glob('*_state.json')}
assert stages and all(s['status']!='RUNNING' for s in stages.values())
assert 'collect' in stages or any(s['status']=='FAILED' for s in stages.values())
PKG.mkdir()
shutil.copytree(OUT/'family_selected',PKG/'family_selected')
if (OUT/'results').exists(): shutil.copytree(OUT/'results',PKG/'results')
allowed={'case.json','case.cfg','history.csv','su2.log','mesh_info.json','progress.log','cases_medium.txt','cases_coarse.txt','cases_fine.txt','done'}
for p in sorted((OUT/'cfd').rglob('*')):
    if p.is_file() and p.name in allowed:
        dest=PKG/p.relative_to(OUT); dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,dest)
meta=PKG/'operations'; meta.mkdir()
for p in sorted(OPS.iterdir()):
    if p.is_file() and (p.suffix in ('.log','.txt','.py','.sh','.exit') or p.name.endswith(('_state.json','_verification.json')) or p.name in ('accepted_base_deviation.json','base_review.json','SOURCE_ACQUISITION.json')):
        shutil.copy2(p,meta/p.name)
env=os.environ.copy(); env['PYTHONDONTWRITEBYTECODE']='1'
with (meta/'upstream_after.log').open('w') as f:
    subprocess.run(['/hy-tmp/dqn-aero/.venv/bin/python','verify_snapshot.py'],cwd=REPO/'upstream_matlab',env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
protocol=PKG/'frozen_protocol'; protocol.mkdir()
for name in ('README.md','CODEX.md','run_decision.sh'):
    shutil.copy2(REPO/'experiments/aero_fidelity'/name,protocol/name)
for name in ('make_su2_cases.py','collect_su2.py'):
    shutil.copy2(REPO/'scripts/cfd'/name,protocol/name)
case_records=[]
for p in sorted((OUT/'cfd').glob('*/*/M*_a*/case.json')):
    h=p.parent/'history.csv'; log=p.parent/'su2.log'
    r=dict(case_id=p.parent.relative_to(OUT/'cfd').as_posix(),done=(p.parent/'done').exists(),history_exists=h.exists(),last_residual=None)
    if h.exists():
        try:
            rows=list(csv.reader(h.open(newline='')))
            r['last_history']=dict(zip([x.strip().strip('"') for x in rows[0]],[float(x) for x in rows[-1]]))
            r['last_residual']=r['last_history'].get('rms[Rho]')
        except Exception as e: r['history_parse_error']=repr(e)
    text=log.read_text(errors='replace') if log.exists() else ''; lines=text.splitlines()
    r['log_tail']=lines[-40:]
    r['warnings']=[x for x in lines if 'Warning:' in x]
    r['nonphysical_points_warning_counts']=[int(x) for x in re.findall(r'there are (\d+) non-physical points',text)]
    r['nonphysical_reconstructed_states_warning_counts']=[int(x) for x in re.findall(r'Warning: (\d+) reconstructed states',text)]
    r['cauchy_CL_pass']=any('Cauchy[CL]' in x and 'Yes' in x for x in lines)
    r['cauchy_CD_pass']=any('Cauchy[CD]' in x and 'Yes' in x for x in lines)
    r['cauchy_evidence']=[x for x in lines if 'Cauchy[CL]' in x or 'Cauchy[CD]' in x]
    r['solver_all_criteria_satisfied']='All convergence criteria satisfied.' in text
    r['solver_exit_success']='Exit Success (SU2_CFD)' in text
    r['density_residual_below_1e_minus5']=r['last_residual'] is not None and r['last_residual'] < -5
    r['converged_by_frozen_rule']=r['done'] and (r['density_residual_below_1e_minus5'] or (r['cauchy_CL_pass'] and r['cauchy_CD_pass']))
    r['observed_case_wall_s']=None
    if r['done'] and log.exists():
        birth=int(subprocess.check_output(['stat','-c','%W',str(log)],text=True).strip())
        if birth>0: r['observed_case_wall_s']=(p.parent/'done').stat().st_mtime-birth
    r['timing_method']='su2.log filesystem birth (integer seconds) to done timestamp; includes MPI startup/output; approximate'
    case_records.append(r)
write(PKG/'cfd_case_status.json',case_records)
with (PKG/'cfd_case_status.csv').open('w',encoding='utf-8',newline='') as f:
    fields=['case_id','done','last_residual','cauchy_CL_pass','cauchy_CD_pass','density_residual_below_1e_minus5','converged_by_frozen_rule','observed_case_wall_s','nonphysical_points_warning_counts','nonphysical_reconstructed_states_warning_counts']
    w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(case_records)
convergence=dict(converged=sum(c['converged_by_frozen_rule'] for c in case_records),
    not_converged=[dict(case_id=c['case_id'],last_residual=c['last_residual']) for c in case_records if not c['converged_by_frozen_rule']],
    cases_with_nonphysical_point_warning=sum(bool(c['nonphysical_points_warning_counts']) for c in case_records),
    cases_with_nonphysical_reconstruction_warning=sum(bool(c['nonphysical_reconstructed_states_warning_counts']) for c in case_records))
decision_path=PKG/'results/decision.json'; diagnostic=None; decision=None
if decision_path.exists():
    decision=read(decision_path); old=read(V4/'results/decision.json')
    old_by={(c['geometry'],c['level'],c['mach'],c['alpha_deg']):c for c in old['raw_cases']}
    pairs=[]
    for c in decision['raw_cases']:
        key=(c['geometry'],c['level'],c['mach'],c['alpha_deg']); previous=old_by[key]
        base=previous['euler_CD']-c['euler_CD']; offset=previous['euler_CD']-previous['wedge_CD']
        pairs.append(dict(geometry=c['geometry'],level=c['level'],mach=c['mach'],alpha_deg=c['alpha_deg'],
            v4_total_CD=previous['euler_CD'],v5_skin_CD=c['euler_CD'],v4_euler_minus_wedge=offset,
            base_term=base,base_minus_prior_offset=base-offset))
    groups=[]
    for g,m in sorted({(c['geometry'],c['mach']) for c in pairs if c['level']=='medium'}):
        rs=[c for c in pairs if c['geometry']==g and c['mach']==m and c['level']=='medium']
        std=statistics.pstdev(c['base_term'] for c in rs)
        mean_diff=statistics.mean(c['base_minus_prior_offset'] for c in rs)
        max_diff=max(abs(c['base_minus_prior_offset']) for c in rs)
        groups.append(dict(geometry=g,mach=m,alpha_count=len(rs),base_mean=statistics.mean(c['base_term'] for c in rs),base_std_population=std,
            all_base_terms_negative=all(c['base_term']<0 for c in rs),std_below_0_0003=std<0.0003,
            mean_offset_difference=mean_diff,abs_mean_offset_difference_below_0_0005=abs(mean_diff)<0.0005,
            max_abs_pointwise_offset_difference=max_diff,all_pointwise_differences_below_0_0005=max_diff<0.0005))
    grid=[]
    for c in decision['raw_cases']:
        if c['geometry']=='FAM_MD6' and c['mach']==6 and c['alpha_deg']==4:
            grid.append(dict(level=c['level'],CL=c['euler_CL'],CD_pressure=c['euler_CD']))
    by={c['level']:c for c in grid}; d3=None
    if 'medium' in by and 'fine' in by:
        a,b=by['medium'],by['fine']; cl=(b['CL']-a['CL'])/max(abs(b['CL']),1e-12); cd=(b['CD_pressure']-a['CD_pressure'])/max(abs(b['CD_pressure']),1e-12)
        same=a['CL']*b['CL']>0 and a['CD_pressure']*b['CD_pressure']>0
        d3=dict(rel_change_CL=cl,rel_change_CD=cd,same_sign=same,pass_=abs(cl)<0.15 and abs(cd)<0.15 and same,denominator='abs(fine), following frozen collect_su2.grid_convergence')
    nonpositive=[dict(geometry=c['geometry'],level=c['level'],mach=c['mach'],alpha_deg=c['alpha_deg'],CD_pressure=c['euler_CD']) for c in decision['raw_cases'] if c['euler_CD']<=0]
    coverage={}
    for model in ('euler','wedge'):
        rs=[r for r in decision['equal_lift'] if r['model']==model]
        conditions=sorted({(r['mach'],r['CL_target']) for r in rs})
        coverage[model]={}
        for g in sorted({r['geometry'] for r in rs}):
            missing=[dict(mach=m,CL_target=cl) for m,cl in conditions if not any(r['geometry']==g and r['mach']==m and r['CL_target']==cl for r in rs)]
            coverage[model][g]=dict(eligible_for_original_switching_gain=not missing,missing_conditions=missing)
    diagnostic=dict(protocol='README 3f; exploratory, not a replacement of the primary v4 result',
        D1=dict(groups=groups,per_case=pairs,standard_deviation='population SD over three alpha cases per medium geometry/Mach',
            note='Both pointwise and group-mean offset differences reported; the original protocol threshold is 0.0005.'),
        D2=dict(pass_=len(decision['raw_cases'])==29 and not nonpositive,case_count=len(decision['raw_cases']),nonpositive_cases=nonpositive),
        D3=d3,grid_values=grid,original_decision=decision['decision'],original_grid_convergence=decision['grid_convergence'],
        comparison_pair_count=len(pairs),all_cases_converged=convergence['converged']==29,switching_gain_coverage=coverage)
    write(PKG/'results/v5_diagnostics.json',diagnostic)
    with (PKG/'results/v4_v5_force_comparison.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(pairs[0])); w.writeheader(); w.writerows(pairs)
    # Keep the v4 comparison source and its checksum, without altering or rerunning it.
    shutil.copy2(V4/'results/decision.json',PKG/'results/v4_original_decision.json')
    assert digest(OUT/'results/decision.json')==digest(decision_path)
mesh=read(OPS/'mesh_verification.json') if (OPS/'mesh_verification.json').exists() else None
summary=dict(commit=SHA,source_branch='aoa-fix',cfd_stage_version=5,classification='exploratory',NP=7,JOBS=2,BASE_MARKER=1,cpu_quota_cores=10,
    family_reused_from=str(V4/'family_selected'),family_rerun=False,family_copy_verified=read(OPS/'family_copy_verification.json'),
    tests='19 passed',upstream_verified_files=203,source_modified=False,stages=stages,mesh_verification=mesh,
    cfd_case_count=len(case_records),cfd_done_count=sum(c['done'] for c in case_records),convergence=convergence,
    decision=decision['decision'] if decision else None,diagnostics=diagnostic,packaged_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
write(PKG/'RUN_SUMMARY.json',summary)
md=['# CFD 第五版执行记录（探索性实验）','',f'冻结提交：`{SHA}`（aoa-fix）。','',
    '复用第四版三种入选外形，五个文件 SHA-256 逐一一致。本次未运行 MATLAB、family 或 check；未修改冻结源码、求解器设置或判据。',
    '', '环境：Python 3.11.10、SU2 8.5.0、Open MPI 5.0.11、gmsh 4.15.2；实际 CPU 配额 10 核。按文档使用 NP=7、JOBS=2、BASE_MARKER=1。',
    '', '相关测试 19 passed；运行前后 upstream_matlab 203 个文件哈希一致。',
    '', '## 各阶段墙钟','', '| 阶段 | 状态 | 秒 |','|---|---|---:|']
for name,s in stages.items(): md.append(f'| {name} | {s["status"]} | {s.get("wall_s",0):.3f} |')
if mesh:
    md+=['','## 网格核验','','| 网格 | 第四版四面体 | 第五版四面体 | base_surfaces | 标记正确 | 内部网格内容相同 |','|---|---:|---:|---:|---|---|']
    for m in mesh['meshes']: md.append(f'| {m["path"]} | {m["v4"]["tetrahedra"]} | {m["v5"]["tetrahedra"]} | {m["mesh_info"]["base_surfaces"]} | {m["markers_valid"]} | {m["interior_identical"]} |')
    if not mesh['all_base_surfaces_are_one']:
        md+=['','MD4 中网格 base_surfaces=2，其余为 1。两个几何面共同组成同一个连通的 x=5 m 尾部封口，并归入唯一 base 物理标记；用户已明确接受这一冻结脚本的原始输出并授权继续。没有将该字段改写为 1，也没有修改网格或参数。详见 operations/base_review.json 和 accepted_base_deviation.json。']
md+=['','## CFD 收敛证据','',f'生成工况 {len(case_records)}/29，运行完成 {summary["cfd_done_count"]}/29，按原残差或 CL/CD Cauchy 条件收敛 {convergence["converged"]}/29。',
    '',f'未收敛工况及末密度残差：{convergence["not_converged"]}。rms[Rho] 保留 SU2 的 log10 值，小于 -5 才对应密度残差小于 1e-5。',
    '',f'非物理解点警告涉及 {convergence["cases_with_nonphysical_point_warning"]} 个工况；非物理迎风重构警告涉及 {convergence["cases_with_nonphysical_reconstruction_warning"]} 个工况。完整文字保留在 su2.log 和 cfd_case_status.json。',
    '', '逐工况耗时由日志创建到 done 标志估算，起点精度为秒；阶段总墙钟单独计时。']
if decision:
    md+=['','## 冻结 collect 原始判定','','```json',json.dumps(decision['decision'],ensure_ascii=False,indent=2),'```','',
        'decision.json/txt 为冻结脚本原始输出，未筛除工况或改写判定；脚本本身读取所有存在的 history，是否取得收敛证据另见逐案例记录。',
        '',f'理想切换收益 Euler：{decision["switching_gain_euler"]}；切楔模型：{decision["switching_gain_wedge"]}。',
        '', '## 预声明诊断 D1–D3','', '| 构型 | Mach | 底面项均值 | α间标准差 | 全为负 | 均值偏差 | 最大逐点偏差 |','|---|---:|---:|---:|---|---:|---:|']
    for r in diagnostic['D1']['groups']: md.append(f'| {r["geometry"]} | {r["mach"]} | {r["base_mean"]:.9g} | {r["base_std_population"]:.9g} | {r["all_base_terms_negative"]} | {r["mean_offset_difference"]:.9g} | {r["max_abs_pointwise_offset_difference"]:.9g} |')
    md+=['', 'D1 阈值：底面项为负、α 间标准差 <0.0003，与第四版 Euler−切楔偏移之差 <0.0005；均值与逐点差异均提供，不隐去不满足项。',
        '',f'D2（全部 29 个压力 CD >0）：{diagnostic["D2"]["pass_"]}；非正项：{diagnostic["D2"]["nonpositive_cases"]}。',
        '',f'D3（中/细网格同号且 CL、CD 变化绝对值均 <15%）：{diagnostic["D3"]}。',
        '', '第五版为预声明的探索性后续实验。第四版原始判定仍保留；数值收敛与物理正确、网格无关性分别报告。']
    md+=['','## 原切换收益统计的覆盖范围','']
    for model,geometries in diagnostic['switching_gain_coverage'].items():
        for g,r in geometries.items():
            if r['missing_conditions']: md.append(f'- {model} / {g} 缺少 {r["missing_conditions"]}，因此被原脚本从完整条件切换收益计算中排除。')
    md+=['','原始 0% 切换收益按冻结脚本的完整条件筛选口径报告；不能把它解释为三个成员在全部目标 CL 上均已得到比较。decision.json/txt 保持原样。']
for name,s in stages.items():
    if s['status']=='FAILED': md+=['',f'## {name} 失败日志末尾','','```text','\n'.join((OPS/(name+'.log')).read_text(errors='replace').splitlines()[-40:]),'```']
md+=['','压缩包包含所有工况的 case.json、case.cfg、history.csv、su2.log、网格统计、诊断、冻结文档及原始 decision；不含 CFD 网格或 restart。所有云端原始文件保留。','']
(PKG/'RUN_SUMMARY.md').write_text('\n'.join(md),encoding='utf-8')
hashes={p.relative_to(PKG).as_posix():digest(p) for p in sorted(PKG.rglob('*')) if p.is_file()}
(PKG/'SHA256SUMS').write_text(''.join(f'{h}  {n}\n' for n,h in hashes.items()),encoding='utf-8')
archive=OPS/'aero_family_v5_dc39a7f_results.zip'; assert not archive.exists()
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
    for p in sorted(PKG.rglob('*')):
        if p.is_file(): z.write(p,p.relative_to(PKG).as_posix())
record=dict(zip=str(archive),sha256=digest(archive),bytes=archive.stat().st_size,manifest_files=len(hashes),summary=summary)
write(OPS/'delivery.json',record)
print(json.dumps(dict(zip=str(archive),sha256=record['sha256'],bytes=record['bytes'],manifest_files=len(hashes),done=summary['cfd_done_count'],converged=convergence['converged'],decision=summary['decision']),ensure_ascii=False),flush=True)

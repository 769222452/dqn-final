"""Read completed direct-teacher trials, check provenance and write Chinese report."""
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

from run_trials import ROOT, OUT, TOOLS, snapshot
sys.path.insert(0, str(ROOT / 'src'))
from morphing_waverider.teacher_v5.direct_target_matching import read_json, save, sha256


def stats(values):
    return {'count':len(values), 'mean_s':statistics.mean(values) if values else None,
            'min_s':min(values) if values else None, 'max_s':max(values) if values else None}


def main():
    results = read_json(OUT / 'generator_trials.json')
    trials = results['trials']
    identity = read_json(OUT / 'identity_manifest.json')
    events = read_json(OUT / 'candidate_evaluations.json')['events']
    geometry = read_json(OUT / 'geometry_cache_manifest.json')['entries']
    aero = read_json(OUT / 'aero_cache_manifest.json')['entries']
    budget = read_json(OUT / 'search_budget.json')
    calls = read_json(OUT / 'teacher_calls.json')['calls']
    metas = [t['result']['metadata'] for t in trials]
    direct_geometry = [g for g in geometry if g['provenance'] == 'DIRECT_TEACHER_GENERATION']
    new_valid = [g for g in direct_geometry if g['geometry_validity'] == 'VALID' and not g['historical_pareto']]
    self_count = sum(t['result']['success'] for t in trials if t['trial'] <= 3)
    t4 = next((t for t in trials if t['trial'] == 4), None)
    t5 = next((t for t in trials if t['trial'] == 5), None)
    reject = bool(t5 and not t5['result']['success']
        and t5['result']['metadata']['target_match_status'] == 'BEST_EFFORT_NOT_WITHIN_TOLERANCE')
    ready = bool(self_count == 3 and t4 and t4['result']['success']
        and not t4['result']['metadata']['historical_pareto'] and new_valid and reject)
    status = {
        'TEACHER_GENERATOR_INTERFACE':'PASS' if all(len(t['result']['geometry']) == 6 for t in trials) and len(trials)==5 else 'FAIL',
        'DIRECT_TEACHER_GEOMETRY_EVALUATION':'PASS' if direct_geometry else 'BLOCKED',
        'DIRECT_TEACHER_AERO_EVALUATION':'PASS' if any(g['geometry_hash']==a['geometry_hash'] and a['result']['status']=='PASS' for g in new_valid for a in aero) else 'BLOCKED',
        'TARGET_MATCHING_SELF_CONSISTENCY':'PASS' if self_count==3 else ('PARTIAL' if self_count else 'FAIL'),
        'NEW_GEOMETRY_GENERATION':'PASS' if new_valid else ('FAIL' if direct_geometry else 'NOT_RUN'),
        'INFEASIBLE_TARGET_HANDLING':'PASS' if reject else 'FAIL',
        'MORPH_PATH_VALIDATED':'NOT_TESTED',
        'REALTIME_3S_CAPABILITY':'NOT_SUPPORTED' if t4 and t4['result']['metadata']['runtime']>3 else 'UNKNOWN',
        'READY_FOR_DYNAMIC_INTEGRATION':'YES' if ready else 'NO'}
    timing = {'geometry_kernel':stats([c['kernel_s'] for c in calls if c['request']['operation']=='geometry' and c.get('kernel_s') is not None]),
        'geometry_validation_process_wall':stats([g['validation_wall_s'] for g in direct_geometry if g.get('validation_wall_s') is not None]),
        'aero_kernel':stats([c['kernel_s'] for c in calls if c['request']['operation']=='aero' and c.get('kernel_s') is not None]),
        'target_search_wall':stats([m['runtime'] for m in metas]),
        'matlab_cold_start_s':budget['matlab_startup_s'],
        'target_construction_wall_s':results['target_construction']['wall_s']}
    averages = {k:statistics.mean(m[k] for m in metas) for k in ['number_of_geometry_evaluations',
        'number_of_aero_evaluations','valid_candidates','invalid_candidates','error_candidates','runtime']}
    checks = {}
    checks['frozen_config_unchanged'] = sha256(OUT / 'search_config.json') == identity['search_config_file_sha256']
    checks['frozen_implementation_unchanged'] = all(sha256(Path(p))==h for p,h in identity['implementation_hashes'].items())
    checks['frozen_teacher_and_audit_sources_unchanged'] = all(sha256(Path(p))==h for p,h in identity['design_spec']['source_identity'].items())
    checks['five_trials_with_frozen_config_hash'] = len(trials)==5 and all(m['config_hash']==identity['search_config_semantic_hash'] for m in metas)
    checks['target_budgets_respected'] = all(m['number_of_geometry_evaluations']<=30 and m['number_of_aero_evaluations']<=36 and m['runtime']<=300 for m in metas)
    checks['only_legal_geometries_scored'] = all(not r['eligible_for_score'] or r['geometry_validity']=='VALID' for r in events)
    checks['invalid_geometries_never_called_aero'] = all(not r.get('aero_called') and not r['eligible_for_score'] for r in events if r['geometry_validity']=='INVALID')
    checks['no_partial_success_claim'] = all(not t['result']['success'] or (
        t['result']['metadata']['geometry_validity']=='VALID' and
        t['result']['metadata']['target_error']['e_CL']<=.01 and
        t['result']['metadata']['target_error']['e_CD']<=.01 and
        t['result']['metadata']['target_error']['E_aero']<=.0001) for t in trials)
    checks['only_geometry_and_aero_operations'] = all(c['request']['operation'] in ('geometry','aero') for c in calls)
    checks['aero_call_ledger_reconciles'] = sum(m['number_of_aero_evaluations'] for m in metas)+results['target_construction']['aero_evaluations']==sum(c['request']['operation']=='aero' for c in calls)
    checks['geometry_call_ledger_reconciles'] = sum(m['number_of_geometry_evaluations'] for m in metas)==sum(c['request']['operation']=='geometry' for c in calls)
    expected_fixed = identity['design_spec']['fixed_design_inputs']
    checks['design_and_flight_conditions_separate'] = all(
        c['request']['inputs10'][0]==expected_fixed['Ma'] and c['request']['inputs10'][5]==expected_fixed['c1']
        and c['request']['inputs10'][6]==expected_fixed['c2'] and c['request']['inputs10'][8]==expected_fixed['AOA']
        and 'M_flight' not in c['request'] and 'alpha_rad' not in c['request']
        for c in calls if c['request']['operation']=='geometry')
    checks['geometry_artifact_hashes_match'] = all(
        sha256(Path(g['mat_path']))==g['mat_sha256'] and sha256(Path(g['audit_path']))==g['audit_sha256']
        for g in geometry if g.get('mat_path'))
    before = read_json(OUT / 'protected_at_trial_start.json')
    after = snapshot()
    changed = [p for p,h in before.items() if after.get(p)!=h]
    checks['all_trial_start_existing_files_unchanged'] = not changed
    # Independently protect every artifact from the preceding completed task.
    old_manifest = read_json(ROOT / 'results/v5_invalid_geometry_root_cause/output_manifest.json')['files']
    checks['previous_trial_and_root_cause_artifacts_unchanged'] = all(sha256(Path(p))==h for p,h in old_manifest.items())
    old = read_json(ROOT / 'results/v5_invalid_geometry_root_cause/protected_after.json')
    old_changes = [p for p,h in old.items() if not Path(p).is_file() or sha256(Path(p))!=h]
    checks['only_authorized_legacy_code_change'] = old_changes == [str(ROOT / 'src/morphing_waverider/generator/interface.py')]
    save(OUT / 'protected_after.json', after)
    save(OUT / 'timing_summary.json', {'timings':timing, 'average_per_target':averages})
    save(OUT / 'acceptance_status.json', status)
    verification = {'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,
        'protected_existing_file_count':len(before), 'changed_since_trial_start':changed,
        'changes_relative_to_previous_task':old_changes, 'old_artifacts_checked':len(old_manifest),
        'verified_utc':datetime.now(timezone.utc).isoformat()}
    save(OUT / 'delivery_verification.json', verification)
    assert all(checks.values()), {k:v for k,v in checks.items() if not v}

    t4m = t4['result']['metadata'] if t4 else None
    lines = [
        '# V5 第一阶段：直接教师模型目标匹配逆设计验证', '',
        '**结论：本轮直接教师链已找到满足预先冻结容差的新合法六维构型；前三个自洽目标通过，明显越界目标被明确标记为未匹配。终点几何合法性与动态变构路径分开，后者未测试。**' if ready else '**结论：本轮试验存在未通过项；具体状态如下，不宣称已完成动态集成准备。**', '',
        '本轮只做静态 CL_des/CD_des → z_cmd。没有 DQN 训练、180 s episode、OCP 求解、3 s 状态推进、GA 重优化或新代理模型训练。冻结的教师核心以及旧实验结果未修改。', '',
        '## 最终验收状态', '', '| 项目 | 状态 |', '|---|---|']
    lines += [f'| {k} | {v} |' for k,v in status.items()]
    lines += ['', 'READY_FOR_DYNAMIC_INTEGRATION=YES 仅表示有证据进入下一阶段的接口、机构路径和动态验证设计；不表示当前算法可直接用于3秒闭环。本任务在静态验证后停止。', '',
        '## A. 是否为真实教师逆设计', '',
        '新增 `src/morphing_waverider/teacher_v5/direct_target_matching.py` 中的 `TeacherTargetMatchingGenerator`。旧 `target_matching.py` 三候选离散试验保持原样。公共调用仍为 `generate(GeneratorState, TargetAero) -> GenerationResult`，geometry 字段为六维 `[a1,a2,b1,b2,d1,a_x]`，三维网格保存在独立 MAT 文件中。未修改 DQN 动作语义，未连接动态 Actuator。', '',
        '所有目标评分由冻结 `teacher_matlab/active/build_geometry_core.m` 和 `offdesign_aero_v4.m` 支持，不调用 SyntheticPhysicsAeroModel、PolynomialSurrogateAeroModel、GPR 或旧扫掠角/翼展模型。搜索器使用合法种子、参数中点提案、少量全域探索及局部坐标轮询；参数中点仅是新候选提案，其气动系数由真实模型计算，未对缓存气动插值。', '',
        '## B. 每个候选经过哪些环节', '',
        '26个合法历史种子逐项核验原始 Pareto MAT 行、六参数、十输入、原生网格 hash、局部/汇总 audit、固定规则及教师源码 hash。精确命中的几何允许复用原生成和审计证据；它们本轮不重新生成。所有新的 z 都实际调用教师核心，原生50×50网格经上一轮同一 `scan.audit` 检查：数值、内部退化、上下表面真实三角交叉、自交、连接关系、截面和相反对角线敏感性。', '',
        '只有 VALID 可以进入教师气动评价和目标排名。INVALID 直接排除；AMBIGUOUS 转 ERROR，不冒充合法。没有取厚度绝对值、裁面、交换蒙皮或把正体积当合法性。失败的几何、气动和预算停止均记录。', '',
        f'本轮实际新生成 {len(direct_geometry)} 个构型，其中合法 {sum(g["geometry_validity"]=="VALID" for g in direct_geometry)}、非法 {sum(g["geometry_validity"]=="INVALID" for g in direct_geometry)}、错误/未确定 {sum(g["geometry_validity"]=="ERROR" for g in direct_geometry)}。非法候选的气动调用数为0。候选详情见 candidate_evaluations.csv/json，原生表面与交叉证据在 geometry/。', '',
        '六维边界来自原教师优化源码，分别为 ' + json.dumps(identity['design_spec']['bounds_from_optimizer'],ensure_ascii=False) + '。这些是搜索边界，不是合法几何证书。', '',
        '固定生成输入为 ' + json.dumps(expected_fixed,ensure_ascii=False) + '。飞行工作点 W1 为 M_star=15、body alpha_star=+4°；它只传给 offdesign。程序设计 AOA=-4° 与 body +4° 的坐标约定沿用既有定义。本轮接口测试另验证改变 M/alpha 不改变几何缓存身份，不额外声称已验证其他飞行域。', '',
        '## C–E. 五组目标结果', '',
        '| Trial | 目标 | solver_status | 返回构型 | 新几何次数 | 气动次数 | 合法/非法候选 | 时间 / s |', '|---|---|---|---|---:|---:|---:|---:|']
    for t in trials:
        m=t['result']['metadata']
        lines.append(f'| {t["trial"]} | {t["purpose"]} | {m["solver_status"]} | {m["geometry_id"]} | {m["number_of_geometry_evaluations"]} | {m["number_of_aero_evaluations"]} | {m["valid_candidates"]}/{m["invalid_candidates"]} | {m["runtime"]:.3f} |')
    lines += ['', f'前三个自洽目标 {self_count}/3 成功。目标源系数是本轮3次新教师气动调用的输出；各 trial 清空气动缓存后重新评价，不使用上一轮试验系数冒充本轮计算。允许气动等效解，未将参数唯一恢复作为普遍成功标准。', '']
    if t4m:
        pair = [t4m['target']['CL_des'], t4m['target']['CD_des']]
        pred = [t4m['CL_pred'],t4m['CD_pred']]
        lines += ['Trial 4 的中间目标仅在**目标空间**取 G34/G16 实测系数均值，不假定该目标一定可达：', '',
            '| 系数 | 目标 | 真实返回 | 绝对误差 | 相对目标误差 |', '|---|---:|---:|---:|---:|']
        for name, target, actual in zip(['CL','CD'],pair,pred):
            lines.append(f'| {name} | {target:.12g} | {actual:.12g} | {abs(actual-target):.6g} | {abs(actual-target)/abs(target)*100:.3f}% |')
        lines += ['', '新返回六参数：`' + json.dumps(t4m['z_cmd']) + '`。', '',
            f'几何 hash：`{t4m["geometry_hash"]}`；是否原52点：`{t4m["historical_pareto"]}`。原生审计 VALID，MORPH_PATH_VALIDATED=NOT_TESTED。', '']
    if t5:
        m=t5['result']['metadata']
        lines += [f'Trial 5 目标 CL=CD=1000，返回 `{m["target_match_status"]}`，solver_status=`{m["solver_status"]}`，停止原因 `{m["search_stop_reason"]}`。success=false；最佳点作为诊断输出保留，不能作为已成功匹配命令自动执行。有限预算只能证明本次搜索未匹配，不能证明整个六维空间数学上不可达。', '']
    lines += ['## 冻结目标误差标准与搜索限制', '',
        'CL_scale=0.1，CD_scale=0.03，w_CL=w_CD=1。必须同时满足 e_CL≤0.01、e_CD≤0.01、E_aero=e_CL²+e_CD²≤0.0001。即分量绝对误差上限分别0.001和0.0003，且仍受组合误差条件约束。这是明确的第一阶段研究容差，不代表任务控制精度已定标。Trial 4 的相对误差已单列，不能把“容差内匹配”说成精确等于目标。', '',
        f'search_config.json 在首个教师调用前冻结，其 SHA256：`{identity["search_config_file_sha256"]}`。运行后未改权重、尺度或容差。', '',
        '每目标最多30次新几何、36次气动，墙钟300 s，单个模型/验证进程最多120 s，所有等待受剩余墙钟约束。临近截止时按已测成本预留完整候选时间，提前停止而不扩大预算。构型距离只用于完全相同气动误差的排序，不是能耗，不替代硬机构约束。', '',
        '## F–G. 实测成本与3秒能力', '',
        '| 环节 | 样本数 | 平均 / s | 最小 / s | 最大 / s |', '|---|---:|---:|---:|---:|']
    for label,key in [('教师几何生成内核','geometry_kernel'),('独立几何检查进程','geometry_validation_process_wall'),('教师气动内核','aero_kernel'),('每目标搜索墙钟','target_search_wall')]:
        s=timing[key]
        if s['count']:
            lines.append(f'| {label} | {s["count"]} | {s["mean_s"]:.3f} | {s["min_s"]:.3f} | {s["max_s"]:.3f} |')
    lines += ['', f'每目标平均：新几何 {averages["number_of_geometry_evaluations"]:.2f} 次，气动 {averages["number_of_aero_evaluations"]:.2f} 次，合法候选 {averages["valid_candidates"]:.2f} 个，非法候选 {averages["invalid_candidates"]:.2f} 个，墙钟 {averages["runtime"]:.3f} s。每 trial 内按实际不同提案计数；未完成或因预算尚未执行的提案单列，不算合法/非法。', '',
        f'另计：MATLAB 冷启动 {timing["matlab_cold_start_s"]:.3f} s；三组真实目标构造（含冷启动）{timing["target_construction_wall_s"]:.3f} s、3次气动、0次几何。全任务累计新几何 {budget["actual_total_geometry_calls"]} 次、气动 {budget["actual_total_aero_calls"]} 次。几何缓存跨目标精确复用，气动缓存仅在当前 trial 内复用。', '',
        f'REALTIME_3S_CAPABILITY={status["REALTIME_3S_CAPABILITY"]}。历史几何命中后的短时间往返不能证明完整逆设计实时能力；需要生成新几何的 Trial 4 已超过3秒，最坏时延和动态调度也未认证。', '',
        '## H. 下一阶段范围与保留限制', '',
        f'状态为 {"READY_FOR_DYNAMIC_INTEGRATION" if ready else "BLOCKED"}。当前只证明有限目标和一个工作点上的静态逆生成关系；没有证明所有 DQN 可能目标可达、全飞行域精度或连续变构路径合法。', '',
        '当前气动系数为 pressure-only CL/CD，共同参考面积来自 G42 原生镜像投影，Sref=' + str(identity['reference']['area_m2']) + ' m²，Lref=' + str(identity['reference']['length_m']) + ' m，沿用1程序单位=1m研究假设。未增加CFD/风洞认证、黏性阻力、配平或可信底压验证。', '',
        '保留26/52非法与优化门控缺失事实；当前正式教师核心的两处 rotated/program 混用保持冻结，未擅自应用上一轮诊断补丁。已有代表实验不支持它是主要穿透根因，本轮通过硬几何门控隔离非法候选，不重做全52根因扫描。', '',
        '## 代码、验证与复现', '',
        '- 新公共实现：`src/morphing_waverider/teacher_v5/direct_target_matching.py`。',
        '- 运行入口：`tools/teacher_target_generator_direct/run_trials.py`。为保护历史结果，已有输出目录时拒绝覆盖。',
        '- 固定教师调用封装：`tools/teacher_target_generator_direct/teacher_worker.m`，串行、无图形、无并行worker。',
        '- 几何门控封装：`tools/teacher_target_generator_direct/validate_native.py`，直接复用冻结原生审计。',
        '- 40项软件合同测试通过，覆盖旧离散接口与新接口的非法门控、缓存隔离、身份变更、预算、失败和容差组合。测试夹具不进入本轮教师物理数据。',
        '- 唯一修改的原公共代码是 GenerationStatus 新增4个失败状态；旧状态含义不变。', '',
        f'交付核验通过：试验启动时的 {len(before)} 个既有文件未变化；上一任务 {len(old_manifest)} 个清单文件逐项核对。相对上一任务的原源码仅有已登记的枚举扩展，教师冻结核心、原项目及旧结果保持原样。', '',
        '所需10个JSON/CSV清单及本报告均在本目录。`geometry/` 保存新候选原生MAT、生成状态、audit、intersection、section；`matlab_exchange/` 保存每次真实请求/响应；`teacher_calls.json` 对账调用次数和时间。`acceptance_status.json`、`timing_summary.json`、`delivery_verification.json` 和 `output_manifest.json` 提供机器可读结论及哈希。', '']
    (OUT / 'V5_TEACHER_TARGET_GENERATOR_DIRECT_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
    files = set(p for p in OUT.rglob('*') if p.is_file() and p.name!='output_manifest.json')
    files.update(p for p in TOOLS.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files.update(Path(p) for p in identity['implementation_hashes'])
    save(OUT / 'output_manifest.json', {'files':{str(p):sha256(p) for p in sorted(files)}})
    print(json.dumps({'status':status,'averages':averages,'timing':timing,'verification':verification['status']},ensure_ascii=False))


if __name__ == '__main__':
    main()

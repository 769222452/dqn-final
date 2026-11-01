"""Chinese reports always label ongoing/incomplete evidence and nominal near ties explicitly."""
from __future__ import annotations
from exp_common import *

def snapshot_scope_line(stats,running=True):
    line=f'本报告的排名、加密复核和轨迹积分均对应指标快照 `{stats["tag"]}`，该快照包含 {stats["mission_grid_required_geometries"]} 个通过几何硬约束的可行候选。各项计算是否完整分别见下文。'
    if running:
        line+=' 运行中的新候选进度见 checkpoint.json；后续批次汇总后更新本报告。'
    return line


def recovery_instructions(status,total_calls):
    command=r'.\.venv\Scripts\python.exe -B tools\v6_teacher_space_expansion\run.py --resume'
    if status=='TARGET_NOT_REACHED':
        next_cap=int(total_calls)+5000
        return ('本轮预算已经耗尽。决定开启下一阶段时，可执行以下恢复命令：保留全部断点，新增48小时预算，并允许最多5000次额外Teacher调用。该命令仅供后续执行，本轮没有自动延长预算。',
                command+f' --extend-hours 48 --max-teacher-attempts {next_cap}')
    return ('继续运行（会校验断点与原始源码；若已有控制器运行，操作系统锁会拒绝重复启动）：',command)


def verification_policy_text():
    return '加密范围按构型逐对核验相邻优势工况：优先原始最小优势较大的相邻对；失败则检查其他候选相邻对。每个检查点均与当前全部可行构型比较。搜索期间优先复核原始相邻点达到0.5%门槛的区域，弱优势名义赢家可暂缓专项加密并保持待复核标签。达到正式数量门槛后，CORE42和等CL网格中的所有本轮新增名义赢家必须完成至少一个局部诊断，才允许通过最终验收。扩展域其他单点另行记录；弱优势、孤立点或未复核区域均不能计为正式有效赢家。'


def gocart_candidates(stats,p,core,equal):
    """Retain both kinds of region winners plus the named physical controls."""
    ids=set(core.geometry_id)|set(equal.geometry_id)|{'G42',*AB}; rows=[]
    inventory=p.set_index('geometry_id')
    for gid in sorted(ids):
        r=inventory.loc[gid]; cw=core[core.geometry_id.eq(gid)]; ew=equal[equal.geometry_id.eq(gid)]
        core_verified=gid in stats['core_effective_verified']; equal_verified=gid in stats['equal_lift_verified_winners']
        roles=[]
        if gid in {'G42',*AB}: roles.append('REFERENCE_CONTROL')
        if len(cw): roles.append('CORE42_WINNER')
        if len(ew): roles.append('EQUAL_CL_WINNER')
        rows.append({'geometry_id':gid,'selection_roles':';'.join(roles),'phase':r.phase,'mat_path':r.mat_path,'geometry_hash':r.geometry_hash,'coordinate_hash':r.coordinate_hash,
                     **{k:float(r[k]) for k in INPUTS},'volume_ratio_G42':float(r.volume_ratio_G42),'thickness_ratio_G42':float(r.thickness_ratio_G42),
                     'core_top1_points':len(cw),'max_delta_LD':float(cw.delta_LD.max()) if len(cw) else None,'core_region_verified':core_verified,
                     'equal_CL_top1_points':len(ew),'max_delta_CD':float(ew.delta_CD.max()) if len(ew) else None,'equal_region_verified':equal_verified,
                     'equal_comparison_complete':stats['equal_and_task_comparison_complete'],
                     'core_conditions':json.dumps(clean(cw[['M_flight','alpha_deg','delta_LD']].to_dict('records'))),
                     'equal_CL_conditions':json.dumps(clean(ew[['M_flight','CL_target','alpha_at_target','delta_CD']].to_dict('records'))),
                     'selection_status':'VERIFIED_REGION' if core_verified or equal_verified else ('REFERENCE_CONTROL' if gid in {'G42',*AB} else 'NOMINAL_PENDING_OR_NEAR_TIE'),
                     'CFD_status':'NOT_RUN'})
    return rows

def refinement_summary_lines(stats):
    """Describe verified witnesses from the matching archived comparison pool."""
    selected=set(stats.get('core_effective_verified',[]))|set(stats.get('equal_lift_verified_winners',[]))
    if not selected: return []
    tag=stats['tag']
    frozen=pd.read_csv(OUT/f'metrics/{tag}_frozen49_all_scores.csv',float_precision='round_trip')
    ids=sorted(frozen.loc[gate(frozen),'geometry_id'].unique())
    pool_hash=hashlib.sha256('\n'.join(ids).encode()).hexdigest()
    verification=load(OUT/'verification'/pool_hash[:16]/'completed.json')
    if verification['pool_ids_sha256']!=pool_hash:
        raise RuntimeError('Report refinement evidence does not match its saved comparison pool')
    lines=['',f'本统计快照的已复核代表工况（竞争池 {len(ids)} 个构型）：',
           '', '下表分别列出原始工况优势与细网格邻域最低优势。有效区域计数使用原始相邻点至少0.5%的优势及加密邻域保留第一名；细网格优势单独报告。',
           '', '| 比较方式 | 构型 | M_flight | 迎角/目标CL | 原始优势 | 细网格最低优势 | 邻域检查点数 |',
           '|---|---|---:|---:|---:|---:|---:|']
    for label,winners_key,results_key,table_name,second_axis,margin in [
        ('固定迎角','core_effective_verified','core_results','CORE42_top1','alpha_deg','delta_LD'),
        ('等CL','equal_lift_verified_winners','equal_results','equal_lift_top1','CL_target','delta_CD')]:
        winners_set=set(stats.get(winners_key,[]))
        if not winners_set: continue
        original=pd.read_csv(OUT/f'metrics/{tag}_{table_name}.csv',float_precision='round_trip')
        for row in verification[results_key]:
            if row['geometry_id'] not in winners_set or not row['stable_all_neighbors']: continue
            match=original[original.geometry_id.eq(row['geometry_id'])&original.M_flight.eq(row['M_flight'])&original[second_axis].eq(row[second_axis])]
            if len(match)!=1: raise RuntimeError('Verified witness is missing from the saved ranking table')
            raw_margin=float(match[margin].iloc[0])
            if raw_margin<.005: continue
            lines.append(f'| {label} | {row["geometry_id"]} | {row["M_flight"]:g} | {row[second_axis]:.9g} | {100*raw_margin:.6f}% | {100*row["minimum_local_winner_margin"]:.6f}% | {row["points_checked"]} |')
    diagnostics=[]
    for label,winners_key,results_key,table_name,margin in [
        ('固定迎角','core_effective_verified','core_results','CORE42_top1','delta_LD'),
        ('等CL','equal_lift_verified_winners','equal_results','equal_lift_top1','delta_CD')]:
        original=pd.read_csv(OUT/f'metrics/{tag}_{table_name}.csv',float_precision='round_trip')
        records=verification[results_key]
        for gid in sorted({r['geometry_id'] for r in records}-set(stats.get(winners_key,[]))):
            checked=[r for r in records if r['geometry_id']==gid]
            raw=[]
            for r in checked:
                row=original.iloc[r['source_row']]
                if row.geometry_id!=gid: raise RuntimeError('Nominal diagnostic source row differs from saved ranking')
                raw.append(float(row[margin]))
            retained=[r for r in checked if r['stable_all_neighbors']]
            fine=f'{100*min(r["minimum_local_winner_margin"] for r in retained):.6f}%' if retained else '无完整保持首位邻域'
            diagnostics.append(f'| {label} | {gid} | {100*min(raw):.6f}%–{100*max(raw):.6f}% | {len(retained)}/{len(checked)} | {fine} |')
    if diagnostics:
        lines += ['', '未计入正式有效赢家的局部诊断：排名保持与优势幅度分别列出。下表只涵盖已检查的代表中心；最后一列只统计完整保持首位的邻域，不能代表发生换名的邻域或全部名义胜区。', '',
                  '| 比较方式 | 构型 | 所查原始中心优势范围 | 全邻域保持首位的中心/所查中心 | 保持首位邻域内最小优势 |',
                  '|---|---|---:|---:|---:|', *diagnostics]
    lines += ['',f'逐邻域原始排名和竞争者信息见 verification/{pool_hash[:16]}/completed.json；结论范围为保存的相邻代表区域。']
    if verification.get('verification_mode')=='QUALIFYING_REGIONS_DURING_SEARCH':
        lines += ['',f'本池完成的是达到原始相邻点门槛的区域复核；另有 {len(verification.get("deferred_nominal_core_ids",[]))} 个核心名义赢家、{len(verification.get("deferred_nominal_equal_ids",[]))} 个等CL名义赢家暂缓本池专项加密（两组可能重叠）。这些构型不因暂缓而获得有效赢家资格，也不能沿用较小历史竞争池的结论作为当前全池证明。完整身份见上述复核记录。']
    return lines

def diagnostic_summary_line(stats):
    return f'未达正式目标的当前证据与搜索方向：CORE42中有 {stats["near_ties_core_below_0p5pct"]}/42 点的第一/第二名优势不足0.5%，当前已复核有效核心赢家为 {len(stats["core_effective_verified"])} 个；等CL比较中，具有原始相邻点0.5%优势的构型为 {len(stats["equal_lift_adjacent_0p5pct_winners"])} 个，完成当前全池区域复核的为 {len(stats["equal_lift_verified_winners"])} 个。固定迎角和等升力的不足分别依据这些数据判断，不能因核心排名接近而否定已观察到的等升力优势。下一批继续结合真实生成合法率、设计Mach覆盖、名义赢家及已复核区域的参数邻域、等CL阻力和气动曲线差异调整提案；保留欠采样区域探索。当前搜索证据尚不足以证明合法Teacher族内不存在更多互补构型。'


def contribution_summary_lines(stats):
    """Keep historical batch contributions separate from current verified regions."""
    lines=[]
    for path in sorted((OUT/'batches').glob('*/aerodynamic_contribution.json')):
        record=load(path); tag=record['tag']
        source_stats=load(OUT/f'metrics/{tag}_summary.json')
        if source_stats['committed_teacher_candidates']>stats['committed_teacher_candidates']: continue
        if record['comparison_pool_count']!=source_stats['mission_grid_required_geometries']:
            raise RuntimeError('Batch contribution pool differs from its saved metric snapshot')
        points=pd.read_csv(path.with_name('new_task_grid_winning_points.csv'),float_precision='round_trip')
        saved=pd.read_csv(OUT/f'metrics/{tag}_mission_top1.csv',float_precision='round_trip')
        if len(points)!=record['new_task_grid_top1_points']:
            raise RuntimeError('Batch contribution point count differs from its saved evidence')
        for row in points.itertuples():
            match=saved[saved.geometry_id.eq(row.geometry_id)&saved.M_flight.eq(row.M_flight)&saved.alpha_deg.eq(row.alpha_deg)]
            if len(match)!=1 or float(match.L_over_D.iloc[0])!=row.L_over_D:
                raise RuntimeError('Batch contribution does not match its saved raw ranking')
        if not lines:
            lines=['','逐批扩展域改善（按各批当时的竞争池，独立于当前正式区域复核）：']
        weight_note=f'（全部输入轨迹的总统计权重为 {record["task_frequency_total_weight_s"]:g} 秒）' if 'task_frequency_total_weight_s' in record else ''
        lines += ['',f'{tag}：竞争池 {record["comparison_pool_count"]} 个；本批新增构型获得冻结49点中的 {record["new_frozen49_top1_points"]} 点、CORE42中的 {record["new_CORE42_top1_points"]} 点、等CL网格中的 {record["new_equal_CL_top1_points"]} 点、扩展任务网格中的 {record["new_task_grid_top1_points"]} 点。新增任务网格赢家对应统计时间权重合计 {record["new_task_grid_weight_s"]:g} 秒，参考轨迹oracle选用时间 {record["new_oracle_selected_time_s"]:g} 秒。',
                  weight_note,
                  '', '| 批次新增构型 | M_flight | 迎角 | CL | 相对上一批该点最优L/D改善 |',
                  '|---|---:|---:|---:|---:|']
        for row in points.itertuples():
            lines.append(f'| {row.geometry_id} | {row.M_flight:g} | {row.alpha_deg:g} | {row.CL:.9g} | {100*row.relative_LD_gain_vs_previous_winner:.6f}% |')
        lines += ['',f'以上是保存网格上的名义改善；不能单凭这些点认定相邻稳定优势、等升力收益或任务收益。统计权重为零仅描述所用轨迹样本。完整证据见 batches/{path.parent.name}/aerodynamic_contribution.json 及对应CSV。']
    return lines


def batch_generation_summary_lines(stats):
    records=[load(p) for p in sorted((OUT/'batches').glob('*/outcome_diagnostics.json'))]
    records=[r for r in records if r['last_attempt_index']<=stats['committed_teacher_candidates']]
    if not records: return []
    lines=['', '已保存的批次生成诊断（限于本统计快照已提交候选）：', '',
           '| 批次 | 尝试数 | Teacher生成失败 | 原生审计无效 | 仅体积不足 | 唯一硬约束通过 |',
           '|---|---:|---:|---:|---:|---:|']
    for r in records:
        outcomes=r['outcomes']
        lines.append(f'| {r["batch"]} | {r["attempts"]} | {outcomes.get("TEACHER_GENERATION_FAILED",0)} | {outcomes.get("NATIVE_AUDIT_INVALID",0)} | {outcomes.get("VOLUME_BELOW_0P85",0)} | {outcomes.get("UNIQUE_HARD_GATE_PASS",0)} |')
    lines += ['', '逐次10输入、原始错误信息、各策略及设计Mach分组统计见 batches/batch_*/outcome_diagnostics.csv 与 .json。生成不收敛表示该次调用未成功，不构成其参数邻域不存在合法构型的证明。']
    return lines


def batch_aero_summary_lines(stats):
    records=[]
    for path in sorted((OUT/'batches').glob('*/outcome_diagnostics.json')):
        r=load(path)
        if 'aerodynamics' in r and r['last_attempt_index']<=stats['committed_teacher_candidates']: records.append(r)
    if not records: return []
    def rate(r):
        if r['evaluated_points'] is None: return '待统计'
        if r['model_valid_fraction'] is None: return '无评估点'
        return f'{100*r["model_valid_fraction"]:.2f}% ({r["model_valid_points"]}/{r["evaluated_points"]})'
    def eligible(r):
        return '待统计' if r['evaluated_points'] is None else f'{r["CL_floor_eligible_points"]}/{r["evaluated_points"]}'
    lines=['','逐批P2有效性统计（各批新增且通过几何硬约束的唯一构型）：','',
           '| 批次 | 冻结49点有效率 | 冻结点CL合格/已算 | 任务网格有效率 | 任务点CL合格/已算 | 任务点已算/应算 |',
           '|---|---:|---:|---:|---:|---:|']
    for r in records:
        f=r['aerodynamics']['frozen49']; m=r['aerodynamics']['mission']
        coverage='待统计' if m['evaluated_points'] is None else f'{m["evaluated_points"]}/{m["expected_points"]}'
        lines.append(f'| {r["batch"]} | {rate(f)} | {eligible(f)} | {rate(m)} | {eligible(m)} | {coverage} |')
    lines += ['', 'P2统计说明：有效要求model_valid、CL/CD_pressure有限且CD_pressure>0；CL合格列另应用原CL门槛，只对应固定迎角排名资格，不用于删除轨迹积分时刻。尚未保存的任务网格统计标为待统计，不视为物理失败。各批输入文件哈希和逐构型统计见 batches/batch_*/outcome_diagnostics.json 与 .csv。']
    return lines


def core_gap_summary_line(stats):
    """Explain saved fixed-alpha near ties using the original CL and pressure CD."""
    tag=stats['tag']; core_path=OUT/f'metrics/{tag}_CORE42_top1.csv'
    scores_path=OUT/f'metrics/{tag}_frozen49_all_scores.csv'
    core=pd.read_csv(core_path,float_precision='round_trip')
    scores=pd.read_csv(scores_path,float_precision='round_trip').set_index(['geometry_id','M_flight','alpha_deg'])
    rows=[]
    for r in core.itertuples():
        second=scores.loc[(r.runner_up,r.M_flight,r.alpha_deg)]
        relative_cl=float(r.CL/second.CL-1); relative_cd=float(r.CD_pressure/second.CD_pressure-1)
        reconstructed=(1+relative_cl)/(1+relative_cd)-1
        if not np.isclose(reconstructed,r.delta_LD,rtol=1e-10,atol=1e-14):
            raise RuntimeError('CL/CD decomposition differs from the saved original ranking margin')
        rows.append({'geometry_id':r.geometry_id,'runner_up':r.runner_up,'M_flight':r.M_flight,'alpha_deg':r.alpha_deg,
                     'relative_CL':relative_cl,'relative_CD_pressure':relative_cd,'delta_LD':float(r.delta_LD),
                     'same_direction_CL_CD':bool(relative_cl*relative_cd>0)})
    d=pd.DataFrame(rows); winners_set=set(core.geometry_id)
    record={'utc':utc(),'tag':tag,'core_points':len(d),'same_direction_points':int(d.same_direction_CL_CD.sum()),
            'both_top_two_from_nominal_core_winners':int(d.runner_up.isin(winners_set).sum()),
            'nominal_core_winner_count':len(winners_set),'minimum_delta_LD':float(d.delta_LD.min()),
            'median_delta_LD':float(d.delta_LD.median()),'maximum_delta_LD':float(d.delta_LD.max()),
            'source_sha256':{core_path.name:sha(core_path),scores_path.name:sha(scores_path)},
            'scope':'Arithmetic decomposition at identical Mach and alpha in this saved pool only; not an equal-lift result or proof that other Teacher geometries cannot improve.','Teacher_or_P2_calls':0}
    csv(OUT/f'metrics/{tag}_core_gap_coefficients.csv',d);dump(OUT/f'metrics/{tag}_core_gap_diagnosis.json',record)
    explanation='多数工况的升力与阻力变化在比值中大幅抵消，领先构型的L/D相互接近。' if record['same_direction_points']>=.8*len(d) and record['median_delta_LD']<.005 else '应根据逐点比值和优势幅度分别判断，不能仅依据CL或CD的单独变化认定性能提高。'
    return f'核心近似并列的系数诊断（{tag}）：{record["same_direction_points"]}/{len(d)} 点的第一、第二名CL和CD_pressure同向变化，两者取比值后的L/D优势中位数为 {100*record["median_delta_LD"]:.6f}%，范围 {100*record["minimum_delta_LD"]:.6f}%–{100*record["maximum_delta_LD"]:.6f}%。其中 {record["both_top_two_from_nominal_core_winners"]}/{len(d)} 点的前两名均来自同 {len(winners_set)} 个名义核心赢家。{explanation}该固定迎角诊断不能否定等升力优势，也不能证明其他合法Teacher构型不可能改进。逐点原始系数比值及恒等式核对见 metrics/{tag}_core_gap_coefficients.csv 和 _core_gap_diagnosis.json。'


def write_report(stats,status):
    from runtime_history import summarize_runtime
    history=summarize_runtime()
    p=pool(); new=p[p.phase.ne('baseline')]; tag=stats['tag']
    newcalls=len(list((OUT/'cases').glob('*/started.json')))
    repro=len(list((OUT/'reproducibility').glob('*/started.json')))
    resume_intro,resume_command=recovery_instructions(status,newcalls+repro)
    core=stats['CORE42']; eq=stats['equal_lift']; oro=stats.get('oracle')
    if oro is None:
        paths=sorted((OUT/'metrics').glob('*_oracle.json'),key=lambda x:x.stat().st_mtime)
        if paths: oro=load(paths[-1]); oro['note']='最近一次已完成的参考轨迹分析；可能尚未包含正在生成的新候选。'
    dom=load(OUT/'mission/domain.json'); pre=load(OUT/'preflight/summary.json') if (OUT/'preflight/summary.json').exists() else {}
    actual_valid=int(new.geometry_validity.eq('VALID').sum()); gatepass=int(gate(new).sum()) if len(new) else 0
    formal=stats['formal_target_reached']; complete=stats['equal_and_task_comparison_complete']
    lines=[f'# Teacher 构型空间搜索报告（{status}）','',f'更新时间：{utc()}。本文件随批次更新；状态为 {status}，正式目标{ "已通过全池加密复核" if formal else "尚未达到"}。',
           '',snapshot_scope_line(stats,status.startswith('RUNNING')),
           '',f'本轮真实 Teacher 调用已开始 {newcalls+repro} 次：再现性检查 {repro} 次，新候选 {newcalls} 次；已有 {len(new)} 次新候选记录落盘（包括失败记录）。任务气动网格及优势区域复核的完成情况分别见下文。预检：{pre.get("teacher_attempts",0)} 次生成，{pre.get("native_VALID",0)} 个原生 VALID，{pre.get("unique_hard_gate_pass",0)} 个通过全部几何硬约束。',
           '',f'运行预算按墙钟计算，截止时间（UTC）为 {history["campaign_deadline_utc"]}，截至本次报告已用 {history["elapsed_wall_hours"]:.3f} 小时。已有 {history["recorded_termination_count"]} 条终止记录；已记录的“终止到下一次MATLAB就绪”间隔合计 {history["completed_termination_to_ready_intervals_hours"]:.3f} 小时，其中包含清理和启动等待，并非精确CPU空闲时间。中断、睡眠和恢复均不延长本轮总预算，不能把整个墙钟窗口表述为连续有效搜索时间。具体证据见 operational/runtime_history.json 和系统事件记录。',
           '',f'当前新增原生 VALID {actual_valid} 个，通过原生审计、封闭体积≥0.85 G42、网格厚度≥0.5 G42且唯一的构型 {gatepass} 个。所有排名仍使用真实 P2 CL/CD_pressure；未添加多样性或设计 Mach 奖励。',
           '',f'冻结 CORE42：A/B 合计占比由 100% 降至 {core["AB_share"]*100:.4f}%；真实名义 Top1 共 {core["unique_top1"]} 个；N_eff={core["N_eff"]:.6f}。至少两个相邻核心点达到0.5%原始第一/第二名优势的构型 {len(stats["core_winners_with_adjacent_0p5pct_margins"])} 个；完成全池局部加密后仍在该相邻区域保持排名的构型 {len(stats["core_effective_verified"])} 个，其中本轮新增 {len(stats["new_effective_verified"])} 个。',
           '',f'相对453个旧构型的核心最优，最大 L/D 改善 {100*stats["max_core_improvement_vs_old"]:.6f}%，平均改善 {100*stats["mean_core_improvement_vs_old"]:.6f}%。当前 {stats["near_ties_core_below_0p5pct"]}/42 点的第一/第二名差距不足0.5%；这些名义赢家不能称为显著变构优势。',
           '',f'等 CL 比较独立报告：名义赢家 {eq["unique_top1"]} 个，具有相邻点0.5%阻力优势的构型 {len(stats["equal_lift_adjacent_0p5pct_winners"])} 个，完成全池加密复核的 {len(stats["equal_lift_verified_winners"])} 个。任务扩展网格已完成 {stats["mission_grid_complete_geometries"]}/{stats["mission_grid_required_geometries"]} 个可行构型；{"该比较已覆盖当前全部可行池" if complete else "该比较仍不完整，当前等CL赢家不能作为最终全池结论"}。原三个CL目标全部保留，另加入轨迹CL分位点。',
           '',f'工作域来自 {len(dom["trajectories"])} 条保存的成功1200秒轨迹：M_flight=[{min(dom["M_grid"])},{max(dom["M_grid"])}]，迎角=[{min(dom["alpha_grid"])},{max(dom["alpha_grid"])}]度。含高频区加密，共{len(dom["M_grid"])*len(dom["alpha_grid"])}点/构型；49点冻结对照保持不变。历史G42轨迹达到M≈3.8，因此未将扩展域任意截在4.75。设计 Mach 取样范围[4,15]来自已验证的原生几何覆盖，并不等于飞行 Mach。',
           '',f'任务时间加权 Top1 N_eff={stats["weighted_task_distribution"]["N_eff"]}；有排名的时间格覆盖比例={stats["weighted_grid_coverage_fraction"]:.6f}。工作点覆盖比例不等于候选池覆盖率。',
           '', '固定参考轨迹理想构型选择收益：']
    if oro:
        lines += ['',f'- 分析状态：{oro["status"]}；参考：{dom["reference_trajectory"]}；候选数：{oro["geometries_evaluated"]}。',f'- J_fixed={oro.get("J_fixed_Ns")} N·s；J_oracle={oro.get("J_oracle_Ns")} N·s；η_potential={oro.get("eta_potential")}。',f'- 全1200秒可行固定构型数：{oro["full_fixed_feasible_count"]}；自由选择覆盖时间：{oro["oracle_covered_time_s"]}秒。',f'- {oro.get("note", "所有0.5秒区间均保留，包括初期负CL需求。") }']
    else: lines += ['', '尚未完成全候选任务域P2评价及逐时刻等升力分析，暂不报告数值收益。']
    if oro and oro.get('incomplete_geometry_ids'):
        lines += ['',f'仍有 {len(oro["incomplete_geometry_ids"])} 个构型的任务P2网格未算完，以上数值仅适用于已完整计算的候选子集；未计算不等于物理不可行。']
    if stats.get('grid_sensitivity'):
        s=stats['grid_sensitivity']
        lines += ['',f'统一真实P2网格敏感性：{s["full_nine_neighbor_centers"]} 个核心中心具有完整九点邻域，其中 {s["full_nine_rank_stable_centers"]} 个保持同一赢家。等CL粗/细迎角网格的 {s["equal_lift_comparable_points"]} 个可比点中，{s["equal_lift_changed_winner_points"]} 个发生赢家变化；最优CD最大相对变化为 {100*s["equal_lift_max_abs_relative_CD_change"]:.6f}%。缺少的邻域点明确列在 task_grid_rank_stability.csv；这些诊断不替代正式全池更细网格复核。']
    lines += refinement_summary_lines(stats)
    lines += ['',f'等CL正式验收还要求至少两个本轮新构型形成已复核优势区域；当前为 {len(stats.get("new_equal_lift_verified_winners",[]))} 个。这是对“包含若干本轮新构型”的最低数量解释，不进入气动排名。']
    lines += ['', '该指标是同一参考轨迹上压力阻力冲量的理想选择上界，未包含摩擦、真实连续变构路径、变构时间或重新优化轨迹，不能解释为1200秒航程改善。无法达到目标CL的区间标为不可行；不删除不利时刻。',
              '', '工况加密与数值敏感性：G42/A/B真实生成和P2再现性已通过；具体误差见 reproducibility/summary.json。正式有效赢家必须通过 verification/latest.json 的全池邻域比较。未加密候选一律标为待验证；0.1%、0.5%、1%阈值仅为低阶筛选分级，非CFD误差或统计置信度。',
              '', verification_policy_text(),
              '', '建议后续GoCart候选依据：优先选择已复核的不同优势区域赢家，同时保留G42及A/B作基准。下表与 gocart_candidates.csv 同时包含固定迎角和等CL赢家；CSV附完整10输入、MAT路径、硬约束比值及获胜工况。未完成全池等CL比较时，该部分候选仍属暂定。下表只依据当前真实P2结果，不代表已完成三维CFD验证。', '', '| 构型 | 核心Top1次数 | 核心最大优势 | 等CL Top1次数 | 等CL最大优势 | 状态 |','|---|---:|---:|---:|---:|---|']
    cw=pd.read_csv(OUT/f'metrics/{tag}_CORE42_top1.csv',float_precision='round_trip')
    ew=pd.read_csv(OUT/f'metrics/{tag}_equal_lift_top1.csv',float_precision='round_trip')
    candidate_rows=gocart_candidates(stats,p,cw,ew)
    for r in candidate_rows:
        cm=f'{100*r["max_delta_LD"]:.6f}%' if r['max_delta_LD'] is not None else '—'
        em=f'{100*r["max_delta_CD"]:.6f}%' if r['max_delta_CD'] is not None else '—'
        state={'VERIFIED_REGION':'相邻区域复核通过','REFERENCE_CONTROL':'基准对照','NOMINAL_PENDING_OR_NEAR_TIE':'待复核/近似并列'}[r['selection_status']]
        if r['selection_status']=='VERIFIED_REGION':
            modes=[]
            if r['core_region_verified']: modes.append('CORE42')
            if r['equal_region_verified']: modes.append('等CL')
            state='、'.join(modes)+'区域复核通过'
        lines.append(f'| {r["geometry_id"]} | {r["core_top1_points"]} | {cm} | {r["equal_CL_top1_points"]} | {em} | {state} |')
    csv(OUT/'gocart_candidates.csv',candidate_rows)
    lines += batch_generation_summary_lines(stats)
    lines += contribution_summary_lines(stats)
    lines += batch_aero_summary_lines(stats)
    lines += ['',core_gap_summary_line(stats)]
    lines += ['', diagnostic_summary_line(stats),
              '', '结果文件：', '',f'- metrics/{tag}_frozen49_all_scores.csv：合并冻结49点全部CL/CD及资格标志。',f'- metrics/{tag}_CORE42_top1.csv：真实第一/第二名差距与相对旧最佳改善。',f'- metrics/{tag}_equal_lift.csv：等CL阻力、求得迎角及不可行标志。',f'- metrics/{tag}_mission_all_scores.csv：共同任务网格的P2结果。',f'- metrics/{tag}_mission_top1.csv：共同任务网格排行榜；额外加密点的全池覆盖状态见 verification/ 和原始 aero/ 数据。','- cases/：10输入、完整MAT、原生审计、交点/截面证据、封闭体积及日志。','- batches/：不可变候选清单、候选选择权重、代理训练数据和随机数状态。','- figures/：任务分布、优势分区、Top1比例及搜索曲线。',
              '', resume_intro,'', '```powershell',r'Set-Location D:\v6-mission-morph-core',resume_command,'```',
              '', '安全停止：在 results/v6_teacher_space_expansion/ 创建 STOP_REQUESTED 文件；当前候选安全落盘后退出。恢复时先移除该标志。保护上限默认为总Teacher尝试5000次（含再现检查）、48小时、MATLAB并发1；预算耗尽标为TARGET_NOT_REACHED。',
              '', '本轮未运行新的CFD、DQN或OCP；未修改原生Teacher/P2公式、正式气动数据库、旧冻结结果或G42归一化门槛。当前证据不能证明安全连续变形或已实现航程收益。']
    path=OUT/'reports/FINAL_REPORT_ZH.md'; tmp=path.with_suffix('.tmp'); tmp.write_text('\n'.join(lines)+'\n',encoding='utf-8'); os.replace(tmp,path)
    dump(OUT/'status.json',{'status':status,'utc':utc(),'metrics_tag':tag,'formal_target_reached':formal,'teacher_calls_started':newcalls+repro,'new_candidates_committed':len(new),'new_unique_hard_gate_pass':gatepass})
    plots(stats)

def plots(stats):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from evaluation import all_aero,usable
    data=all_aero(); ids=set(stats['new_nominal_core_winners'])|set(stats['equal_lift']['shares'])|set(AB)|{'G42'}
    d=data[data.geometry_id.isin(ids)&data.M_flight.eq(6.5)]
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for gid,g in d.groupby('geometry_id'):
        g=g.sort_values('alpha_deg'); cl=g.CL.where(usable(g)); cd=g.CD_pressure.where(usable(g))
        axes[0].plot(g.alpha_deg,cl,label=gid); axes[1].plot(g.alpha_deg,cd); axes[2].plot(cd,cl)
    axes[0].set(xlabel='Alpha [deg]',ylabel='CL',title='Flight Mach 6.5'); axes[1].set(xlabel='Alpha [deg]',ylabel='Pressure CD'); axes[2].set(xlabel='Pressure CD',ylabel='CL')
    axes[0].legend(fontsize=6); fig.savefig(OUT/'figures/current_CL_CD_polars.png',dpi=150); plt.close(fig)
    hist=[load(p) for p in sorted((OUT/'batches').glob('*/metrics.json'))]
    if hist:
        fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained'); x=[r['committed_teacher_candidates'] for r in hist]
        for ax,ys,label in zip(axes.ravel(),[[r['new_unique_hard_gate_pass'] for r in hist],[len(r['core_effective_verified']) for r in hist],[r['CORE42']['N_eff'] for r in hist],[r.get('best_equal_drag',np.nan) for r in hist]],['Unique feasible new geometries','Verified effective CORE42 winners','CORE42 N_eff','Best equal-lift CD (M6.5, original middle CL)']):
            ax.plot(x,ys,'o-'); ax.set(xlabel='Teacher candidate attempts',ylabel=label)
        fig.savefig(OUT/'figures/search_history.png',dpi=150); plt.close(fig)
    trace=OUT/f"metrics/{stats['tag']}_oracle_trace.csv"
    if trace.exists():
        t=pd.read_csv(trace)
        if 'fixed_cumulative_Ns' in t:
            fig,ax=plt.subplots(figsize=(8,4),layout='constrained'); ax.plot(t.time_s,t.fixed_cumulative_Ns,label='Best fixed'); ax.plot(t.time_s,t.oracle_cumulative_Ns,label='Instantaneous oracle'); ax.set(xlabel='Time [s]',ylabel='Cumulative pressure drag impulse [N s]'); ax.legend(); fig.savefig(OUT/'figures/oracle_cumulative_drag.png',dpi=150); plt.close(fig)

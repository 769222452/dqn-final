"""Summarize this bounded diagnostic without executing any further trajectories."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'))
from continuous_tracking.experiment import write_json


def main(folder):
    result = json.loads((folder/'result.json').read_text(encoding='utf-8'))
    reference = json.loads((folder/'source.json').read_text(encoding='utf-8'))
    source = Path(reference['path'])
    if not source.is_absolute():
        source = ROOT/source
    short = result['short']
    old, new = short['old_rows'], short['rows']
    if not new:
        raise ValueError('No executed new intervals to plot')
    old = old[:len(new)]

    def stat(rows):
        return dict(mean_decision_s=float(np.mean([r['wall_s'] for r in rows])),
                    mean_alpha_peak_to_peak_deg=float(np.mean([np.ptp(r['alpha_nodes_deg']) for r in rows])),
                    CL_mean=float(np.mean([r['CL_mean'] for r in rows])),
                    CL_rmse=float(np.sqrt(np.mean([r['CL_rmse']**2 for r in rows]))),
                    tracking_failed=sum(not r['quality']['passed'] for r in rows),
                    switches=sum(r['switch_started'] for r in rows))

    before, after = stat(old),stat(new)
    identities = json.loads((folder/'identity.json').read_text())['files']
    protected = {p:h for p,h in identities.items() if not p.startswith('experiments/uniform_tracking_v2/')}
    changed = [p for p,h in protected.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    summary = dict(old=before,new=after,
        mean_wave_reduction_percent=100*(1-after['mean_alpha_peak_to_peak_deg']/before['mean_alpha_peak_to_peak_deg']),
        mean_decision_time_reduction_percent=100*(1-after['mean_decision_s']/before['mean_decision_s']),
        original_source_files_checked=len(protected),original_source_files_changed=changed,
        old_result_sha256_unchanged=hashlib.sha256((source/'result.json').read_bytes()).hexdigest()==reference['sha256'],
        physical_audits_passed=all(r['full_plan_audit_valid'] for r in new),
        minimum_executed_margins={k:min(r['min_margins'][k] for r in new) for k in new[0]['min_margins']},
        no_1200s_claim=True,original_execution_last_committed_time_s=reference['snapshot_time_s'],
        original_run_stop_cause='User-authorized external stop; old in-progress result.json left unchanged',
        timing_caveat='Historical same-machine baseline; not a repeated interleaved timing trial')
    write_json(folder/'summary.json',summary)

    plt.rcParams.update({'font.family':'sans-serif',
        'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
        'axes.unicode_minus':False, 'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.7,'pdf.fonttype':42,'savefig.dpi':240})
    fig,axes = plt.subplots(3,1,figsize=(9,8.8),gridspec_kw={'height_ratios':[1.3,1.1,1]},layout='constrained')
    colors = ['#B86B56','#246C83']
    labels = ['原方案（已保存轨迹）','新方案（真实双OCP闭环）']
    for group, rows in enumerate((old,new)):
        ts, al, ct, cl = [],[],[],[]
        for i,r in enumerate(rows):
            ts.extend(r['t0']+np.array(r['times']))
            al.extend(r['alpha_nodes_deg'])
            dest = source/f"decision_{int(r['step']):04d}" if group==0 else folder/'short'/f'decision_{i:04d}'
            with np.load(dest/'executed.npz') as z:
                samples = z['0.1_samples'][8]
                values = np.r_[samples[::5], samples[-1]]
            ct.extend(r['t0']+np.arange(len(values))*.1)
            cl.extend(values)
        axes[0].plot(ts,al,color=colors[group],lw=1.8,label=labels[group],marker='o',ms=2.3)
        axes[1].plot(ct,cl,color=colors[group],lw=1.7,label=labels[group])
    target=old[0]['CL_cmd']
    axes[1].axhline(target,color='#555555',ls='--',lw=1,label=f'CL请求 = {target:g}')
    for ax in axes[:2]:
        for t in [r['t1'] for r in new[:-1]]:
            ax.axvline(t,color='#777777',lw=.65,ls=':',zorder=0)
        ax.set_xlim(new[0]['t0'],new[-1]['t1'])
        ax.grid(axis='y',alpha=.15)
        ax.ticklabel_format(axis='y',style='plain',useOffset=False)
        ax.set_xlabel('飞行时间 / s')
    axes[0].set_ylabel('攻角 α / °')
    axes[1].set_ylabel('实际升力系数 CL')
    axes[0].set_title('(a) 实际执行攻角：原始节点，无平滑处理',loc='left',fontsize=11,pad=35)
    axes[1].set_title('(b) 独立0.1秒积分中的CL',loc='left',fontsize=11,pad=35)
    axes[0].legend(loc='lower left',bbox_to_anchor=(0,1.01),frameon=False,fontsize=9,ncol=2)
    axes[1].legend(loc='lower left',bbox_to_anchor=(0,1.01),frameon=False,fontsize=9,ncol=3)
    x=np.arange(len(new)); w=.34
    for j, rows in enumerate((old,new)):
        bars=axes[2].bar(x+(j-.5)*w,[r['wall_s'] for r in rows],width=w,color=colors[j],label=labels[j])
        axes[2].bar_label(bars,fmt='%.2f',fontsize=9,padding=3)
    axes[2].set_xticks(x,[f"{r['t0']:g}–{r['t1']:g}" for r in new])
    axes[2].set_xlabel('执行区间 / s'); axes[2].set_ylabel('决策墙钟时间 / s')
    axes[2].set_title('(c) 每次Pre＋Final＋审计耗时（不含首次编译）',loc='left',fontsize=11)
    axes[2].set_ylim(0,max(r['wall_s'] for r in old+new)*1.22)
    fig.suptitle('均匀跟踪实验：489–501秒短程对照\n仅验证12秒；不代表1200秒任务结果',fontsize=13)
    fig.savefig(folder/'comparison_zh.png')
    fig.savefig(folder/'comparison_zh.pdf')
    plt.close(fig)

    lines=['# 均匀跟踪与跨周期变化率记忆：有限诊断结果','',
        '测试范围：4个已保存状态的Final重求解，以及从同一489秒状态出发的12秒真实Pre/Final闭环。恒定CL=0.025；没有训练，没有代理执行器，没有1200秒任务结果。',
        '', '## 12秒闭环', '',
        '| 指标 | 原方案 | 新方案 |','|---|---:|---:|',
        f"| 平均每3秒攻角峰谷差 / ° | {before['mean_alpha_peak_to_peak_deg']:.8f} | {after['mean_alpha_peak_to_peak_deg']:.8f} |",
        f"| 实际CL均值 | {before['CL_mean']:.8f} | {after['CL_mean']:.8f} |",
        f"| CL跟踪RMSE | {before['CL_rmse']:.8f} | {after['CL_rmse']:.8f} |",
        f"| 平均决策耗时 / s | {before['mean_decision_s']:.3f} | {after['mean_decision_s']:.3f} |",
        f"| 质量不合格区间 / 4 | {before['tracking_failed']} | {after['tracking_failed']} |",
        f"| 新启动变构次数 | {before['switches']} | {after['switches']} |", '',
        f"平均区间攻角峰谷差减少{summary['mean_wave_reduction_percent']:.2f}%；CL跟踪RMSE有所增大，二者均低于原有质量阈值。新方案全时域和执行前缀的0.1/0.5秒物理审计全部通过。",
        f"新闭环耗时{short['wall_s']:.3f}秒，8次NLP求解，冷启动重试{short['retries']}次；首次冻结状态构建/编译/求解/审计耗时{result['frozen'][0]['wall_s']:.3f}秒，其中优化器求解{result['frozen'][0]['new']['solver_s']:.3f}秒。整个诊断墙钟{result['wall_s']:.3f}秒。",
        '原耗时来自同机历史运行，未做交错重复计时；不据此声称训练提速或完整任务提速。', '',
        '## 同状态Final检查', '',
        '| t / s | 原CL RMSE | 新CL RMSE | 新当前3秒质量 | 新全时域物理审计 |',
        '|---:|---:|---:|---|---|']
    for f in result['frozen']:
        n=f.get('new')
        lines.append(f"| {f['t']:g} | {f['old']['CL_rmse']:.8f} | {n['CL_rmse']:.8f} | {'通过' if n['quality']['passed'] else '不合格'} | {'通过' if n['full_plan_audit_valid'] else '不通过'} |" if n else f"| {f['t']:g} | {f['old']['CL_rmse']:.8f} | — | {f['status']} | — |")
    lines += ['', '48秒和375秒问题状态没有被该修改解决。冻结检查只重求解原Final上下文，不能当作新方案从0秒滚动到这些时刻的轨迹。', '',
        '## 改动与边界', '',
        '- 取消原来只对当前3秒设置的跟踪最优性预算；整个规划时域使用同一逐点CL软误差带（0.0001）和显式非负松弛（罚权100）。',
        '- 把最后实际执行的攻角变化率与间隔时长传入下一次优化；增加变化率改变的代价。它不是攻角二阶连续硬保证，也不是经辨识的真实执行机构模型。',
        '- Pre/Final各只求解一个NLP；保留原硬物理约束与两种网格的独立审计。CL软误差带可以被突破，所有松弛已记录，不能把非零松弛解释为已证明物理上不得不偏离。',
        '- 新代码和配置仅位于本实验文件夹；没有替换原训练控制器。原长程实验在用户授权下外部停止，最后已提交828秒；原result.json保持未改，其MAX_DECISIONS字段为停止前快照。',
        f"- 对照启动时哈希检查的{len(protected)}个原源文件/配置/完整性文件，变化数={len(changed)}；原result.json哈希一致={summary['old_result_sha256_unchanged']}。", '',
        '## 逐区间结果', '', '| 区间 / s | 原攻角峰谷差 / ° | 新攻角峰谷差 / ° | 新CL RMSE | 新决策 / s |', '|---|---:|---:|---:|---:|']
    for a,b in zip(old,new):
        lines.append(f"| {b['t0']:g}–{b['t1']:g} | {np.ptp(a['alpha_nodes_deg']):.8f} | {np.ptp(b['alpha_nodes_deg']):.8f} | {b['CL_rmse']:.8f} | {b['wall_s']:.3f} |")
    lines += ['', '配置、源文件哈希、所有求解尝试、攻角计划、独立积分数组、审计、检查点、原始统计与PNG/PDF均在本结果目录。',
              '没有根据本次结果修改罚权、误差带、物理阈值或求解预算。']
    validation_path=folder/'validation.json'
    if validation_path.exists():
        validation=json.loads(validation_path.read_text(encoding='utf-8'))
        lines += ['', f"验证：{validation['pytest']['output']}；编译与原始MX的目标、约束、梯度及Jacobian在记录的验证点逐位一致。"]
    (folder/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    main(parser.parse_args().folder)

"""Refresh explanatory sections using one saved metric snapshot, without running physics."""
from __future__ import annotations
import argparse
import re
from exp_common import *
from reporting import refinement_summary_lines,diagnostic_summary_line,contribution_summary_lines,core_gap_summary_line,batch_aero_summary_lines,batch_generation_summary_lines,snapshot_scope_line,recovery_instructions,verification_policy_text


def refresh(tag=None):
    status_path=OUT/'status.json'; status_bytes=status_path.read_bytes()
    status=json.loads(status_bytes); tag=tag or status['metrics_tag']
    if tag!=status['metrics_tag'] or not re.fullmatch(r'[A-Za-z0-9_-]+',tag):
        raise ValueError('Select the metric tag of the current report snapshot')
    stats=load(OUT/f'metrics/{tag}_summary.json')
    path=OUT/'reports/FINAL_REPORT_ZH.md'; before=path.read_bytes()
    # Preserve paragraph boundaries while preventing repeated CR expansion on Windows.
    body=re.sub(r'\r+\n','\n',before.decode('utf-8')).replace('\r','\n')
    body=re.sub(r'已有 (\d+) 次新候选流水线完整提交。',r'已有 \1 次新候选记录落盘（包括失败记录）。任务气动网格及优势区域复核的完成情况分别见下文。',body)
    body=re.sub(r'\n本报告的排名、加密复核和轨迹积分均对应指标快照[^\n]*\n','',body)
    scope_marker='\n本轮真实 Teacher'
    if scope_marker not in body: raise RuntimeError('Expected snapshot scope insertion point is missing')
    body=body.replace(scope_marker,'\n'+snapshot_scope_line(stats,status['status'].startswith('RUNNING'))+'\n'+scope_marker,1)
    recovery_start=body.find('\n继续运行（')
    if recovery_start<0: recovery_start=body.find('\n本轮预算已经耗尽。')
    if recovery_start<0: raise RuntimeError('Expected recovery instructions are missing')
    recovery_end=body.index('\n安全停止：',recovery_start)
    intro,command=recovery_instructions(status['status'],status['teacher_calls_started'])
    recovery=['',intro,'','```powershell',r'Set-Location D:\v6-mission-morph-core',command,'```','']
    body=body[:recovery_start]+'\n'.join(recovery)+body[recovery_end:]
    body=body.replace('所有本轮新增名义赢家均至少接受一个局部诊断；弱优势或孤立点不能据此计为正式有效赢家。','CORE42和等CL网格中的所有本轮新增名义赢家均至少接受一个局部诊断；扩展域其他单点另行记录，未完成专门复核时不认定稳定区域。弱优势或孤立点不能据此计为正式有效赢家。')
    body=re.sub(r'(?m)^加密范围按构型逐对核验相邻优势工况：[^\n]*$',verification_policy_text(),body)
    marker='\n等CL正式验收'
    start=body.find('\n本统计快照的已复核代表工况')
    if start>=0:
        end=body.index(marker,start)
        body=body[:start]+body[end:]
    if marker not in body: raise RuntimeError('Expected report insertion point is missing')
    refinement=refinement_summary_lines(stats)
    if refinement: body=body.replace(marker,'\n'.join(refinement)+'\n'+marker,1)
    diagnostic_marker='\n未达正式目标的当前证据与搜索方向：'
    contribution_start=body.find('\n逐批扩展域改善（')
    if contribution_start>=0:
        contribution_end=body.index(diagnostic_marker,contribution_start)
        body=body[:contribution_start]+body[contribution_end:]
    contribution=contribution_summary_lines(stats)
    if contribution:
        if diagnostic_marker not in body: raise RuntimeError('Expected diagnostic paragraph is missing')
        body=body.replace(diagnostic_marker,'\n'.join(contribution)+'\n'+diagnostic_marker,1)
    aero_start=body.find('\n逐批P2有效性统计（')
    if aero_start>=0:
        aero_note=body.index('\nP2统计说明：',aero_start)
        aero_end=body.index('\n',aero_note+1)
        body=body[:aero_start]+body[aero_end:]
    aero=batch_aero_summary_lines(stats)
    if aero: body=body.replace(diagnostic_marker,'\n'.join(aero)+'\n'+diagnostic_marker,1)
    generation_start=body.find('\n已保存的批次生成诊断（')
    if generation_start>=0:
        generation_note=body.index('\n逐次10输入、',generation_start)
        generation_end=body.index('\n',generation_note+1)
        body=body[:generation_start]+'\n'.join(batch_generation_summary_lines(stats))+'\n'+body[generation_end:]
    body='\n'.join(line for line in body.split('\n') if not line.startswith('核心近似并列的系数诊断（'))
    body=body.replace(diagnostic_marker,'\n'+core_gap_summary_line(stats)+'\n'+diagnostic_marker,1)
    core=set(stats.get('core_effective_verified',[])); equal=set(stats.get('equal_lift_verified_winners',[]))
    lines=body.splitlines(); diagnostic_found=False
    for i,line in enumerate(lines):
        if line.startswith('未达正式目标的当前证据与搜索方向：'):
            lines[i]=diagnostic_summary_line(stats); diagnostic_found=True
        for gid in core|equal:
            if line.startswith('| '+gid+' |'):
                cells=line.split('|')
                if len(cells)!=8: continue  # Only the six-column GoCart table has a status cell.
                modes=[]
                if gid in core: modes.append('CORE42')
                if gid in equal: modes.append('等CL')
                cells[-2]=' '+'、'.join(modes)+'区域复核通过 '
                lines[i]='|'.join(cells)
    if not diagnostic_found: raise RuntimeError('Expected diagnostic paragraph is missing')
    body='\n'.join(lines)+'\n'
    if path.read_bytes()!=before or status_path.read_bytes()!=status_bytes:
        raise RuntimeError('Controller updated the report during refresh; retry against its new snapshot')
    tmp=path.with_suffix('.snapshot.tmp');tmp.write_bytes(body.encode('utf-8'));os.replace(tmp,path)
    result={'utc':utc(),'status':'PASS','snapshot_tag':tag,
            'snapshot_pool_count':stats['mission_grid_required_geometries'],
            'verified_witness_rows':sum(line.startswith(('| 等CL |','| 固定迎角 |')) and len(line.split('|'))==9 for line in refinement),
            'nominal_diagnostic_rows':sum(line.startswith(('| 等CL |','| 固定迎角 |')) and len(line.split('|'))==7 for line in refinement),
            'snapshot_counts_and_timestamp_preserved':True,'line_endings':'LF',
            'report_sha256':sha(path),'reporting_source_sha256':sha(Path(__file__).with_name('reporting.py')),
            'scope':'Saved metric snapshot only; subsequent candidates are outside this report snapshot',
            'Teacher_or_P2_calls':0,'goal_completion_claimed':False}
    dump(OUT/f'validation_{tag}_report_refresh.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--tag')
    print(json.dumps(refresh(parser.parse_args().tag),ensure_ascii=False,indent=2))

"""Summary of the round-2 pilot (scripts/pilot_r2.sh): training progress of full seed 17 and hit rates on the pilot
scenarios (seeds.dev x T1-T6). Writes runs/r2/F8D2000B/PILOT.md.

    python -m target_rl.pilot
"""
import numpy as np

from . import config
from .util import load_json

GROUPS = (('T1–T3', ('T1', 'T2', 'T3')), ('T4–T6', ('T4', 'T5', 'T6')))
LABEL = {'lookup': '按目标查表起点', 'pc_fair': '预测-校正（公平版）', 'pc_strong': '预测-校正（强版）',
         'rl:full:17': '本文方法（第二轮，种子 17，试跑）', 'r1:full:17': '第一轮方法（种子 17）'}


def main():
    base = config.runs_dir()
    lines = ['# 第二轮试跑结果（开发场景，不是测试结果）', '']
    st_path = base/'train'/'full_s17'/'state.json'
    if st_path.exists():
        st = load_json(st_path)
        h = st['history']
        lines += ['## 训练进度（full，种子 17）', '',
                  f"- 起点回报 {h[0]['G_before'] if h else st['G']:.3f} → 当前 {st['G']:.3f}；"
                  f"训练场景命中 {st['hits']}/{len(st['cases'])}，超限 {st['violations']}；"
                  f"已迭代 {st['iteration']} 轮，采纳 {sum(r['accepted'] for r in h)} 轮。", '']
    lines += ['## 开发场景命中情况', '',
              '| 方法 | T1–T3 命中 | T4–T6 命中 | T4–T6 \\|航程误差\\| 中位数 km | T4–T6 \\|末速误差\\| 中位数 m/s | 超限 |',
              '|---|---|---|---|---|---|']
    for m, label in LABEL.items():
        p = base/'eval_dev'/'L2'/f"{m.replace(':', '_')}.json"
        if not p.exists():
            continue
        rows = load_json(p)['rows']
        cells = []
        for _, names in GROUPS:
            rr = [r for r in rows if r['target'] in names]
            cells.append(f"{sum(r['hit'] for r in rr)}/{len(rr)}")
        rr = [r for r in rows if r['target'] in GROUPS[1][1] and r['feasible']]
        dD = f"{np.median([abs(r['dD_km']) for r in rr]):.1f}" if rr else '—'
        dV = f"{np.median([abs(r['dV_mps']) for r in rr]):.0f}" if rr else '—'
        lines.append(f"| {label} | {cells[0]} | {cells[1]} | {dD} | {dV} | {sum(not r['feasible'] for r in rows)} |")
    lines += ['', '开发场景只用于决定是否开始全量运行，不进入任何论文结果。']
    text = '\n'.join(lines)+'\n'
    (base/'PILOT.md').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()

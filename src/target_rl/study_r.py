"""Study R (PREREGISTRATION.md amendment 7): evaluation and decision report.

Methods (all flown on the same cases):
  rule_fixed / rule_morph   the start rule of configs/study_r.yaml (no learning), fixed geometry / morphing
  rl_fixed:<seed>           trained relfixed policy (fixed geometry)
  rl_morph:<seed>           trained rel policy (morphing)
  pc_fixed                  predictor-corrector, fixed geometry (range target), cheap settings (guidance section)
  pc_strong                 strong predictor-corrector with morphing, cheap settings: reference only
Sets: validation (seeds.validation x one random target each; decision set) and dev (seeds.dev x T1-T6; reported).
Test scenarios are not used. Writes runs/r/<library>/R_REPORT.md.

    TARGET_RL_CONFIG=configs/study_r.yaml PYTHONPATH=src python -m target_rl.study_r --seeds "17 23" --workers 5
"""
import argparse
import os
import time

import numpy as np

from . import baselines as B, config, guided
from .parallel import pool_map
from .policy import chain_action
from .study_a import cases
from .train import case_from_json, case_to_json
from .util import load_json, save_json

COMPARISONS = (   # (label, reference method, method, decides)
    ('方法消融：强化学习（不变构）− 预测-校正（不变构）', 'pc_fixed', 'rl_fixed', True),
    ('变构消融：强化学习（变构）− 强化学习（不变构）', 'rl_fixed', 'rl_morph', True),
    ('学习的作用（变构）：强化学习 − 起点规则', 'rule_morph', 'rl_morph', False),
    ('学习的作用（不变构）：强化学习 − 起点规则', 'rule_fixed', 'rl_fixed', False),
    ('参照：强化学习（变构）− 强版预测-校正（变构）', 'pc_strong', 'rl_morph', False))
LABEL = {'rule_fixed': '起点规则（不变构）', 'rule_morph': '起点规则（变构）', 'pc_fixed': '预测-校正（不变构）',
         'pc_strong': '强版预测-校正（变构，参照）'}


def train_state(variant, seed):
    p = config.runs_dir()/'train'/f'{variant}_s{seed}'/'state.json'
    return load_json(p) if p.exists() else None


def spec(method):
    """(chain or None, morph) of a method."""
    rule = dict(kind='pfb', **config.load()['start_rule'])
    if method == 'rule_fixed':
        return [dict(rule, morph=False)], False
    if method == 'rule_morph':
        return [dict(rule, morph=True)], True
    if method in ('pc_fixed', 'pc_strong'):
        return None, method == 'pc_strong'
    kind, seed = method.split(':')
    st = train_state('relfixed' if kind == 'rl_fixed' else 'rel', int(seed))
    return st['chain'], kind == 'rl_morph'


def fly_job(kw):
    sc, tg = case_from_json(kw['case'])
    t0 = time.perf_counter()
    if kw['chain'] is None:
        g = config.load()['guidance']
        pc = B.PredictorCorrector(kw['method'])
        pc.pc = dict(pc.pc, cycle_s=g['cycle_s'], newton=g['newton'], newton_first=g['newton_first'],
                     predict_dt_s=g['predict_dt_s'])
        env = pc.fly(sc, tg)
        npred = pc.n_predictions
    else:
        env = guided.make(sc, tg, kw['morph'])
        while not env.done:
            env.step(chain_action(kw['chain'], env.features(extra=True), kw['morph']))
        npred = env.npred
    out = env.summary()
    out.update(wall_s=time.perf_counter()-t0, n_predictions=npred)
    return out


def evaluate(method, which, workers):
    chain, morph = spec(method)
    path = config.runs_dir()/'study_r'/which/f"{method.replace(':', '_')}.json"
    n_chain = None if chain is None else len(chain)
    if path.exists():
        old = load_json(path)
        if old.get('chain_len') == n_chain:
            return old['rows']
    rows = pool_map([('study_r', dict(case=case_to_json(c), chain=chain, morph=morph, method=method))
                     for c in cases(which)], workers)
    save_json(path, dict(method=method, set=which, chain_len=n_chain, rows=rows))
    return rows


def summary(rows):
    ok = [r for r in rows if r['feasible']]
    med = (lambda k: float(np.median([abs(r[k]) for r in ok]))) if ok else (lambda k: float('nan'))
    return dict(n=len(rows), hits=int(sum(r['hit'] for r in rows)), violations=len(rows)-len(ok),
                mean_G=float(np.mean([r['G'] for r in rows])), dD=med('dD_km'), dV=med('dV_mps'),
                wall_s=float(np.median([r['wall_s'] for r in rows])),
                predictions=float(np.median([r['n_predictions'] for r in rows])))


def paired(a, b, n_boot=10000, seed=0):
    d = np.array([y['G']-x['G'] for x, y in zip(a, b)])
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def name(m):
    if m in LABEL:
        return LABEL[m]
    kind, seed = m.split(':')
    return f"强化学习（{'变构' if kind == 'rl_morph' else '不变构'}，种子 {seed}）"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--workers', type=int, default=os.cpu_count() or 1)
    p.add_argument('--seeds', default='17 23')
    a = p.parse_args(argv)
    if config.load().get('guidance', {}).get('kind') != 'rel_pred':
        p.error('study R needs TARGET_RL_CONFIG=configs/study_r.yaml')
    seeds = [int(s) for s in a.seeds.replace(',', ' ').split()]
    seeds = [s for s in seeds if train_state('rel', s) and train_state('relfixed', s)]
    methods = ['pc_fixed', 'rule_fixed', 'rule_morph']+[f'{k}:{s}' for s in seeds for k in ('rl_fixed', 'rl_morph')]+['pc_strong']
    res = {(m, w): evaluate(m, w, a.workers) for w in ('validation', 'dev') for m in methods}
    L = ['# 研究 R 结果（修订 7）', '',
         '强化学习自己飞（每 12 s 调升力系数目标、选变构档位），每 48 s 一次粗预测作为输入。测试场景没有使用。', '', '## 训练', '']
    for s in seeds:
        for v in ('relfixed', 'rel'):
            st = train_state(v, s)
            h = st['history']
            L.append(f"- {'变构' if v == 'rel' else '不变构'}，种子 {s}：起点回报 {h[0]['G_before'] if h else st['G']:.3f} → {st['G']:.3f}；"
                     f"命中 {st['hits']}/{len(st['cases'])}，超限 {st['violations']}；迭代 {st['iteration']} 轮，采纳 "
                     f"{sum(r['accepted'] for r in h)} 轮" + (f"；结束：{st['converged']['reason']}" if st.get('converged') else ''))
    verdict = {}
    for w, title in (('validation', '验证场景（1400–1495，判定用）'), ('dev', '开发场景（2350–2354 × T1–T6，只报告）')):
        L += ['', f'## {title}', '', '| 方法 | 命中 | 超限 | 平均回报 | \\|航程误差\\| 中位数 km | \\|末速误差\\| 中位数 m/s | 单次飞行 s | 预测次数 |'
              + (' T1–T3 | T4–T6 |' if w == 'dev' else ''),
              '|---|---|---|---|---|---|---|---|' + ('---|---|' if w == 'dev' else '')]
        for m in methods:
            s_ = summary(res[(m, w)])
            row = (f"| {name(m)} | {s_['hits']}/{s_['n']} | {s_['violations']} | {s_['mean_G']:.3f} | {s_['dD']:.1f} | "
                   f"{s_['dV']:.0f} | {s_['wall_s']:.2f} | {s_['predictions']:.0f} |")
            if w == 'dev':
                g = lambda ts: sum(r['hit'] for r in res[(m, w)] if r['target'] in ts)
                row += f" {g(('T1', 'T2', 'T3'))}/15 | {g(('T4', 'T5', 'T6'))}/15 |"
            L.append(row)
        L += ['', '配对比较（平均回报差，95% 自助置信区间，场景重抽样）：', '']
        for label, ref, meth, decides in COMPARISONS:
            for s in seeds:
                r_ = ref if ':' not in ref and ref in LABEL else f'{ref}:{s}'
                m_ = f'{meth}:{s}'
                d, lo, hi = paired(res[(r_, w)], res[(m_, w)])
                sa, sb = summary(res[(r_, w)]), summary(res[(m_, w)])
                L.append(f"- {label}（种子 {s}）：{d:+.3f} [{lo:+.3f}, {hi:+.3f}]；命中 {sb['hits']-sa['hits']:+d}，"
                         f"超限 {sb['violations']-sa['violations']:+d}")
                if decides and w == 'validation':
                    verdict.setdefault(label, []).append(lo > 0 and sb['violations'] <= sa['violations'])
    L += ['', '## 判定（修订 7）', '']
    for label, oks in verdict.items():
        L.append(f"- {label}：{'成立' if oks and all(oks) else '不成立'}（每个种子：验证场景置信区间下界 > 0 且超限不更多；"
                 f"{sum(oks)}/{len(oks)} 个种子满足）")
    L += ['', f'平台：{os.name}，并行进程 {a.workers}。']
    text = '\n'.join(L)+'\n'
    out = config.runs_dir()/'R_REPORT.md'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding='utf-8')
    print(text)
    from .parallel import close
    close()


if __name__ == '__main__':
    main()

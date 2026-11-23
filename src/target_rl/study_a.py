"""Study A (PREREGISTRATION.md amendment 6): evaluation and decision report.

Flies the initial policy ('base' = the cheap strong predictor-corrector, offset 0 always) and the trained policy of
every seed ('rl:<seed>') on
  validation  seeds.validation (1400-1495) x one random target each (training-like, never used for training)
  dev         seeds.dev (2350-2354) x T1-T6
at the main disturbance level, then writes runs/a/<library>/A_REPORT.md. The test scenarios are not used.

Decision (amendment 6, fixed before the run): study A passes if, on the validation set, the paired difference of
the mean return (rl - base) has a 95 % bootstrap confidence interval (cases resampled) with lower bound > 0 AND
the trained policy has no more limit violations than the base. Dev results are reported, not used for the decision.

    TARGET_RL_CONFIG=configs/study_a.yaml PYTHONPATH=src python -m target_rl.study_a --workers 5
"""
import argparse
import os
import time

import numpy as np

from . import config, guided, region as RG
from .parallel import pool_map
from .policy import chain_action
from .scenario import scenario
from .train import case_from_json, case_to_json
from .util import load_json, save_json


def cases(which, cfg=None):
    cfg = cfg or config.load()
    lvl = cfg['disturbance']['main_level']
    if which == 'validation':
        return [(scenario(lvl, s), RG.random_target(s, f'val{s}', cfg=cfg))
                for s in config.seed_range(cfg['seeds']['validation'])]
    if which == 'dev':
        return RG.test_cases(cfg=cfg, which='dev')
    raise ValueError(which)


def fly_job(kw):
    sc, tg = case_from_json(kw['case'])
    t0 = time.perf_counter()
    env = guided.make(sc, tg)
    while not env.done:
        env.step(chain_action(kw['chain'], env.features(extra=True), True))
    out = env.summary()
    out.update(wall_s=time.perf_counter()-t0, decisions=int(round(env.t/env.w['decision'])))
    return out


def chain_of(method):
    if method == 'base':
        return [dict(kind='plan', action=guided.plan_action())]
    seed = int(method.split(':')[1])
    path = config.runs_dir()/'train'/f'pcguided_s{seed}'/'state.json'
    if not path.exists():
        raise FileNotFoundError(f'no trained policy for seed {seed}: {path}')
    return load_json(path)['chain']


def evaluate(method, which, workers):
    path = config.runs_dir()/'study_a'/which/f"{method.replace(':', '_')}.json"
    chain = chain_of(method)
    if path.exists():
        old = load_json(path)
        if old.get('chain_len') == len(chain):
            return old['rows']
    cs = cases(which)
    rows = pool_map([('guided', dict(case=case_to_json(c), chain=chain)) for c in cs], workers)
    save_json(path, dict(method=method, set=which, chain_len=len(chain), rows=rows))
    return rows


def summary(rows):
    ok = [r for r in rows if r['feasible']]
    med = (lambda k: float(np.median([abs(r[k]) for r in ok]))) if ok else (lambda k: float('nan'))
    return dict(n=len(rows), hits=int(sum(r['hit'] for r in rows)), violations=int(sum(not r['feasible'] for r in rows)),
                mean_G=float(np.mean([r['G'] for r in rows])), dD=med('dD_km'), dV=med('dV_mps'),
                wall_s=float(np.median([r['wall_s'] for r in rows])),
                predictions=float(np.median([r.get('n_predictions', 0) for r in rows])))


def paired(base, rl, n_boot=10000, seed=0):
    d = np.array([b['G']-a['G'] for a, b in zip(base, rl)])
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(mean=float(d.mean()), lo=float(lo), hi=float(hi))


def training_lines(seed):
    p = config.runs_dir()/'train'/f'pcguided_s{seed}'/'state.json'
    if not p.exists():
        return [f'- 种子 {seed}：没有训练结果']
    st = load_json(p)
    h = st['history']
    g0 = h[0]['G_before'] if h else st['G']
    return [f"- 种子 {seed}：训练场景 {len(st['cases'])} 个；起点回报 {g0:.3f} → {st['G']:.3f}；命中 {st['hits']}，"
            f"超限 {st['violations']}；已迭代 {st['iteration']} 轮，采纳 {sum(r['accepted'] for r in h)} 轮"
            + (f"；已结束：{st['converged']['reason']}" if st.get('converged') else '')]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--workers', type=int, default=os.cpu_count() or 1)
    p.add_argument('--seeds', default='17', help='trained seeds to evaluate (space or comma separated)')
    a = p.parse_args(argv)
    if not guided.enabled():
        p.error('study A needs TARGET_RL_CONFIG=configs/study_a.yaml')
    seeds = [int(s) for s in a.seeds.replace(',', ' ').split()]
    methods = ['base']+[f'rl:{s}' for s in seeds
                        if (config.runs_dir()/'train'/f'pcguided_s{s}'/'state.json').exists()]
    res = {(m, w): evaluate(m, w, a.workers) for w in ('validation', 'dev') for m in methods}
    L = ['# 研究 A 结果（修订 6）', '',
         '起点 = 便宜版强版预测-校正（变构偏移恒为 0）；本文方法 = 起点 + 强化学习调整变构偏移。测试场景没有使用。', '',
         '## 训练', '']
    for s in seeds:
        L += training_lines(s)
    verdicts = []
    for w, title in (('validation', '验证场景（1400–1495，各一个随机目标，判定用）'), ('dev', '开发场景（2350–2354 × T1–T6，只报告）')):
        L += ['', f'## {title}', '',
              '| 方法 | 场景数 | 命中 | 超限 | 平均回报 | \\|航程误差\\| 中位数 km | \\|末速误差\\| 中位数 m/s | 单次飞行 s | 预测次数 |',
              '|---|---|---|---|---|---|---|---|---|']
        for m in methods:
            s_ = summary(res[(m, w)])
            name = '起点' if m == 'base' else f"本文方法（种子 {m.split(':')[1]}）"
            L.append(f"| {name} | {s_['n']} | {s_['hits']} | {s_['violations']} | {s_['mean_G']:.3f} | {s_['dD']:.1f} | "
                     f"{s_['dV']:.0f} | {s_['wall_s']:.2f} | {s_['predictions']:.0f} |")
        for m in methods[1:]:
            d = paired(res[('base', w)], res[(m, w)])
            sb, sr = summary(res[('base', w)]), summary(res[(m, w)])
            L.append(f"\n本文方法（种子 {m.split(':')[1]}）减起点：平均回报差 {d['mean']:+.3f}，95% 置信区间 "
                     f"[{d['lo']:+.3f}, {d['hi']:+.3f}]；命中 {sr['hits']-sb['hits']:+d}，超限 {sr['violations']-sb['violations']:+d}")
            if w == 'validation':
                ok = d['lo'] > 0 and sr['violations'] <= sb['violations']
                verdicts.append((m, ok))
    L += ['', '## 判定（修订 6）', '']
    if not verdicts:
        L.append('还没有训练好的策略。')
    for m, ok in verdicts:
        L.append(f"- 种子 {m.split(':')[1]}：{'通过' if ok else '未通过'}（验证场景平均回报差的 95% 置信区间下界 > 0，且超限不更多）")
    L += ['', f"平台：{os.name}，并行进程 {a.workers}。"]
    text = '\n'.join(L)+'\n'
    out = config.runs_dir()/'A_REPORT.md'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding='utf-8')
    print(text)
    from .parallel import close
    close()


if __name__ == '__main__':
    main()

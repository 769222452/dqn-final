"""Report: hit rates, errors, violations; H1 and G1 of PREREGISTRATION.md section 7.

Groups: T1-T3 (reachable without morphing), T4-T6 (morphing only), R1-R10 (random held-out targets).
Paired differences against pc_fair are taken per (test scenario, target) and their mean's 95% interval comes
from a cluster bootstrap over the 20 test scenarios (10000 resamples, seed 0).

    python -m target_rl.report            # writes runs/report.json and runs/REPORT.md
"""
import argparse

import numpy as np

from . import config
from .util import cluster_bootstrap_ci, load_json, save_json

RUNS = config.ROOT/'runs'
GROUPS = dict(fixed=('T1', 'T2', 'T3'), morph=('T4', 'T5', 'T6'))


def group_of(name):
    for g, names in GROUPS.items():
        if name in names:
            return g
    return 'random'


def load_methods(level):
    out = {}
    for p in sorted((RUNS/'eval'/level).glob('*.json')):
        d = load_json(p)
        out[d['method']] = d['rows']
    return out


def stats(rows):
    if not rows:
        return None
    fe = [r for r in rows if r['feasible']]
    return dict(n=len(rows), hit_rate=float(np.mean([r['hit'] for r in rows])),
                violation_rate=float(np.mean([not r['feasible'] for r in rows])),
                dD_median=float(np.median([abs(r['dD_km']) for r in fe])) if fe else None,
                dD_p90=float(np.quantile([abs(r['dD_km']) for r in fe], .9)) if fe else None,
                dV_median=float(np.median([abs(r['dV_mps']) for r in fe])) if fe else None,
                dV_p90=float(np.quantile([abs(r['dV_mps']) for r in fe], .9)) if fe else None,
                protect_fraction=float(np.mean([r.get('protect_fraction', 0.) for r in rows])))


def paired(rows_a, rows_b, group):
    a = {(r['seed'], r['target']): r for r in rows_a if group_of(r['target']) == group}
    b = {(r['seed'], r['target']): r for r in rows_b if group_of(r['target']) == group}
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    seeds = sorted({k[0] for k in keys})
    clusters = [[float(a[k]['hit'])-float(b[k]['hit']) for k in keys if k[0] == s] for s in seeds]
    d = np.concatenate([np.asarray(c) for c in clusters])
    lo, hi = cluster_bootstrap_ci(clusters)
    return dict(n=len(d), diff=float(d.mean()), ci95=[lo, hi],
                violation_a=float(np.mean([not a[k]['feasible'] for k in keys])),
                violation_b=float(np.mean([not b[k]['feasible'] for k in keys])))


def build(level=None):
    cfg = config.load()
    level = level or cfg['disturbance']['main_level']
    M = load_methods(level)
    table = {m: {g: stats([r for r in rows if group_of(r['target']) == g]) for g in ('fixed', 'morph', 'random')}
             for m, rows in M.items()}
    out = dict(level=level, methods=table, H1={}, comparisons={}, ablations={})
    ref = M.get('pc_fair')
    seeds = cfg['seeds']['training_runs']
    for variant in ('full', 'dist', 'fixed', 'mean'):
        per_seed = {}
        for s in seeds:
            m = f'rl:{variant}:{s}'
            if m in M and ref is not None:
                per_seed[s] = {g: paired(M[m], ref, g) for g in ('fixed', 'morph', 'random')}
                if 'pc_strong' in M:
                    per_seed[s]['morph_vs_pc_strong'] = paired(M[m], M['pc_strong'], 'morph')
                if 'constant' in M:
                    per_seed[s]['morph_vs_constant'] = paired(M[m], M['constant'], 'morph')
        out['comparisons'][variant] = per_seed
        rates = [table[f'rl:{variant}:{s}']['morph']['hit_rate'] for s in seeds if f'rl:{variant}:{s}' in table]
        out['ablations'][variant] = dict(seeds=len(rates), morph_hit_rate_mean=float(np.mean(rates)) if rates else None)
    full = out['comparisons'].get('full', {})
    h1 = {}
    for s, c in full.items():
        p = c['morph']
        h1[s] = dict(diff=p['diff'], ci95=p['ci95'],
                     holds=bool(p['ci95'][0] > 0 and p['violation_a'] <= p['violation_b']))
    out['H1'] = h1
    if len(h1) == len(seeds):
        mean_diff = float(np.mean([v['diff'] for v in h1.values()]))
        n_hold = sum(v['holds'] for v in h1.values())
        out['G1'] = dict(mean_diff=mean_diff, h1_holds=n_hold, threshold=0.20,
                         passed=bool(mean_diff >= 0.20 and n_hold >= 4))
    else:
        out['G1'] = dict(passed=None, note=f'{len(h1)}/{len(seeds)} full-method seeds evaluated')
    return out


def markdown(r):
    pct = lambda v: '—' if v is None else f'{100*v:.0f}%'
    num = lambda v, f='%.1f': '—' if v is None else f % v
    lines = [f"# 评估报告（{r['level']}）", '',
             '命中：无超限，且 |航程误差| ≤ 5 km、|末速误差| ≤ 25 m/s。T1–T3 不变构也能到，T4–T6 只有变构能到，R 为随机留出目标。', '',
             '| 方法 | T1–T3 命中 | T4–T6 命中 | 随机目标命中 | T4–T6 超限率 | T4–T6 末速误差中位数 (m/s) | 保护介入比例 |',
             '|---|---|---|---|---|---|---|']
    for m, g in sorted(r['methods'].items()):
        mo = g['morph'] or {}
        lines.append(f"| {m} | {pct((g['fixed'] or {}).get('hit_rate'))} | {pct(mo.get('hit_rate'))} | "
                     f"{pct((g['random'] or {}).get('hit_rate'))} | {pct(mo.get('violation_rate'))} | "
                     f"{num(mo.get('dV_median'), '%.0f')} | {pct(mo.get('protect_fraction'))} |")
    lines += ['', '## H1：完整方法相对公平版预测-校正（T4–T6 命中率差）', '',
              '| 种子 | 命中率差 | 95% 置信区间 | H1 |', '|---|---|---|---|']
    for s, v in r['H1'].items():
        lines.append(f"| {s} | {100*v['diff']:+.1f} 个百分点 | [{100*v['ci95'][0]:+.1f}, {100*v['ci95'][1]:+.1f}] | "
                     f"{'成立' if v['holds'] else '不成立'} |")
    g = r['G1']
    lines += ['', '## G1', '']
    if g.get('passed') is None:
        lines.append(f"尚未判定：{g.get('note')}")
    else:
        lines.append(f"平均命中率差 {100*g['mean_diff']:+.1f} 个百分点（门槛 +20），H1 成立 {g['h1_holds']}/5 → "
                     f"**{'通过' if g['passed'] else '未通过'}**")
    lines += ['', '## 消融（T4–T6 平均命中率）', '', '| 变体 | 种子数 | 平均命中率 |', '|---|---|---|']
    for v, a in r['ablations'].items():
        lines.append(f"| {v} | {a['seeds']} | {pct(a['morph_hit_rate_mean'])} |")
    return '\n'.join(lines)+'\n'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--level', default=None)
    a = p.parse_args(argv)
    r = build(a.level)
    save_json(RUNS/'report.json', r)
    (RUNS/'REPORT.md').write_text(markdown(r), encoding='utf-8')
    print(markdown(r))


if __name__ == '__main__':
    main()

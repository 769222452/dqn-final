"""v2 evaluation and report (PREREGISTRATION_V2.md): the reliable task domain of morphing and fixed geometry, and what learning adds.

Methods (all flown on the same cases, execution of configs/v2.yaml: altitude guard, library shapes, 8 s transition):
  pc_weak        the old fixed-geometry predictor-corrector (matches the range only; kept for continuity with rounds 1-2)
  pc_fixedlike   strong predictor-corrector (matches range AND speed, 3 phases) with the morph level forced to 0:
                 the fair fixed-geometry baseline (the minimum-drag shape; the static morph gain is ~0.07 %)
  pc_morph       the same strong predictor-corrector with free morphing (reference, not an opponent of the learned policy)
  pc_oracle      the same with the true density ahead and the true biases (full-information reference)
  rule_fixed / rule_morph   the start rule of the learned policy (no learning)
  rl_fixed:<s> / rl_morph:<s>   the trained policies (variants relfixed / rel, seed s)
Sets (region.v2_cases): validation (decision set), test (run once), domain (task-domain map).
Writes runs/v2/<library>/V2_REPORT.md (+ figure).

    TARGET_RL_CONFIG=configs/v2.yaml PYTHONPATH=src python -m target_rl.study_v2 --workers 5 --seeds "17 23"
"""
import argparse
import os
import platform
import sys
import time

import numpy as np

from . import baselines as B, config, guided, region as RG
from .parallel import pool_map
from .policy import chain_action
from .train import case_from_json, case_to_json
from .util import load_json, save_json

LABEL = {'pc_weak': '预测-校正（不变构，只匹配航程，旧基线）', 'pc_fixedlike': '预测-校正（固定外形，匹配距离和末速）',
         'pc_morph': '预测-校正（变构，参照）', 'pc_oracle': '全信息预测-校正（参照）',
         'rule_fixed': '起点规则（不变构）', 'rule_morph': '起点规则（变构）'}
PC_KIND = {'pc_weak': ('pc_fixed', False), 'pc_fixedlike': ('pc_strong_w0', True), 'pc_morph': ('pc_strong', True),
           'pc_oracle': ('pc_oracle', True)}


def train_state(variant, seed):
    p = config.runs_dir()/'train'/f'{variant}_s{seed}'/'state.json'
    return load_json(p) if p.exists() else None


def seeds_available(seeds):
    return [s for s in seeds if train_state('rel', s) and train_state('relfixed', s)]


def name(m):
    if m in LABEL:
        return LABEL[m]
    kind, seed = m.split(':')
    return f"强化学习（{'变构' if kind == 'rl_morph' else '不变构'}，种子 {seed}）"


def spec(method):
    """Job description of a method."""
    rule = dict(kind='pfb', **config.load()['start_rule'])
    if method in PC_KIND:
        k, morph = PC_KIND[method]
        return dict(kind='pc', pc=k, morph=morph, chain=None)
    if method == 'rule_fixed':
        return dict(kind='policy', morph=False, chain=[dict(rule, morph=False)])
    if method == 'rule_morph':
        return dict(kind='policy', morph=True, chain=[dict(rule, morph=True)])
    kind, seed = method.split(':')
    st = train_state('relfixed' if kind == 'rl_fixed' else 'rel', int(seed))
    return dict(kind='policy', morph=(kind == 'rl_morph'), chain=st['chain'])


def fly_job(kw):
    sc, tg = case_from_json(kw['case'])
    sp = kw['spec']
    t0 = time.perf_counter()
    if sp['kind'] == 'pc':
        pc = B.PredictorCorrector(sp['pc'])
        env = pc.fly(sc, tg)
        npred = pc.n_predictions
    else:
        env = guided.make(sc, tg, sp['morph'])
        while not env.done:
            env.step(chain_action(sp['chain'], env.features(extra=True), sp['morph']))
        npred = env.npred
    out = env.summary()
    out.update(wall_s=time.perf_counter()-t0, n_predictions=int(npred))
    return out


def evaluate(method, which, workers):
    sp = spec(method)
    path = config.runs_dir()/'study_v2'/which/f"{method.replace(':', '_')}.json"
    n_chain = None if sp['chain'] is None else len(sp['chain'])
    if path.exists():
        old = load_json(path)
        if old.get('chain_len') == n_chain:
            return old['rows']
    cases = RG.v2_cases(which)
    rows = pool_map([('study_v2', dict(case=case_to_json(c), spec=sp)) for c in cases], workers)
    save_json(path, dict(method=method, set=which, chain_len=n_chain, rows=rows))
    return rows


# ------------------------------------------------------------------------------------------- statistics
def hit(r, k=1.):
    h = config.load()['hit']
    return bool(r['feasible'] and abs(r['dD_km']) <= h['range_km']*k and abs(r['dV_mps']) <= h['speed_mps']*k)


def summary(rows):
    return dict(n=len(rows), succ={k: float(np.mean([hit(r, k) for r in rows])) for k in (1., 2., 4.)},
                viol=int(sum(not r['feasible'] for r in rows)), G=float(np.mean([r['G'] for r in rows])),
                sw=float(np.median([r.get('n_switches', 0) for r in rows])), preds=float(np.median([r['n_predictions'] for r in rows])),
                wall=float(np.median([r['wall_s'] for r in rows])))


def cluster_diff(a, b, scen, n_boot, seed=0):
    """Mean of b - a per case, 95 % bootstrap interval resampling whole scenarios (a scenario carries several cases)."""
    d = np.asarray(b, float)-np.asarray(a, float)
    ids = np.unique(scen)
    sums = np.array([d[scen == i].sum() for i in ids])
    cnt = np.array([(scen == i).sum() for i in ids], float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(ids), (n_boot, len(ids)))
    boots = sums[pick].sum(1)/cnt[pick].sum(1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def strata(rows, c):
    return [r for r in rows if r['target'].startswith(c)]


# ------------------------------------------------------------------------------------------- domain map
def domain_stats(res, methods, p0s):
    grid = RG.domain_grid()
    out = {}
    for m in methods:
        S = {}
        for r in res[m]:
            S.setdefault((r['target_range_km'], r['target_speed_mps']), []).append(hit(r))
        out[m] = {p: float(np.mean(v)) for p, v in S.items()}
    lines = ['| 方法 | 网格平均成功率 | ' + ' | '.join(f'成功率 ≥ {100*p:.0f}% 的格数' for p in p0s) + ' |', '|---|---|' + '---|'*len(p0s)]
    for m in methods:
        sr = out[m]
        lines.append(f"| {name(m)} | {100*np.mean(list(sr.values())):.1f}% | " + ' | '.join(str(sum(v >= p-1e-9 for v in sr.values())) for p in p0s) + ' |')
    return out, lines, grid


def far_end(out, methods, grid, p0=.8):
    Vs = sorted({g[1] for g in grid})
    lines = [f'| 末速要求 m/s | ' + ' | '.join(name(m) for m in methods) + ' |', '|---|' + '---|'*len(methods)]
    for V in Vs:
        cells = []
        for m in methods:
            ok = [g[0] for g in grid if g[1] == V and out[m].get(g, 0.) >= p0-1e-9]
            cells.append(f'{max(ok):.0f} km' if ok else '—')
        lines.append(f'| {V:.0f} | ' + ' | '.join(cells) + ' |')
    return lines


def draw_map(out, methods, grid, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    reg = RG.load_region()
    FX, MH = np.asarray(reg['fixed_hull']), np.asarray(reg['morph_hull'])
    n = len(methods)
    fig, ax = plt.subplots(1, n, figsize=(4.6*n, 4.4), sharey=True, squeeze=False)
    for a, m in zip(ax[0], methods):
        X = [g[0] for g in grid]
        Y = [g[1] for g in grid]
        C = [100*out[m].get(g, 0.) for g in grid]
        sc = a.scatter(X, Y, c=C, cmap='Blues', vmin=0, vmax=100, s=150, marker='s', edgecolor='#444', linewidth=.4)
        for H, ls in ((FX, '--'), (MH, '-')):
            P = np.vstack([H, H[0]])
            a.plot(P[:, 0], P[:, 1], ls, color='#b03a2e', lw=1.1)
        a.set_title(m, fontsize=9)
        a.set_xlabel('target range D (km)')
        a.grid(alpha=.2)
    ax[0][0].set_ylabel('required terminal speed V (m/s)')
    fig.colorbar(sc, ax=ax, label='success rate (%)', shrink=.9)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------------------------------------------- report
def safe_print(text):
    """Print to consoles whose encoding cannot represent every symbol (Windows GBK: the minus sign, >=, x); the file is UTF-8."""
    try:
        print(text)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or 'utf-8'
        print(text.encode(enc, 'replace').decode(enc))


def report(seeds, which, with_domain=True):
    cfg = config.load()
    nb = int(cfg['experiment']['bootstrap'])
    base = ['pc_weak', 'pc_fixedlike', 'pc_morph', 'pc_oracle', 'rule_fixed', 'rule_morph']
    rl = [f'{k}:{s}' for s in seeds for k in ('rl_fixed', 'rl_morph')]
    methods = base+rl
    path = lambda m: config.runs_dir()/'study_v2'/which/f"{m.replace(':', '_')}.json"
    res = {m: load_json(path(m))['rows'] for m in methods if path(m).exists()}
    methods = [m for m in methods if m in res]
    cases = RG.v2_cases(which)
    scen = np.array([sc.seed for sc, _ in cases])
    n_sc = len(np.unique(scen))
    L = [f'# v2 结果报告（{which} 场景）', '',
         f'场景 {n_sc} 个，用例 {len(cases)} 个；扰动 {cfg["disturbance"]["main_level"]}；执行层：高度保护、真实构型就近选、8 s 变构过渡。'
         f'判定容差 {cfg["hit"]["range_km"]:g} km / {cfg["hit"]["speed_mps"]:g} m/s（1×）。区间为整场景重抽样的 95% 自助区间（{nb} 次）。', '']
    # 1 main table
    L += ['## 1. 主结果', '', '| 方法 | 成功率 1× | 2× | 4× | 可到目标* | 只有变构能到* | 超限 | 平均回报 | 构型切换次数/次飞行 | 预测次数 | 单次飞行 s |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    for m in methods:
        s_ = summary(res[m])
        F_, M_ = strata(res[m], 'F'), strata(res[m], 'M')
        f_ = f"{100*np.mean([hit(r) for r in F_]):.1f}%" if F_ else '—'
        m_ = f"{100*np.mean([hit(r) for r in M_]):.1f}%" if M_ else '—'
        L.append(f"| {name(m)} | {100*s_['succ'][1]:.1f}% | {100*s_['succ'][2]:.1f}% | {100*s_['succ'][4]:.1f}% | {f_} | {m_} | {s_['viol']} | "
                 f"{s_['G']:+.3f} | {s_['sw']:.0f} | {s_['preds']:.0f} | {s_['wall']:.2f} |")
    L += ['', '*"可到目标"：名义上固定外形也能到的目标；"只有变构能到"：只有变构的名义可达区域包含的目标（随机目标按此分层，T1–T3 / T4–T6 另列）。', '']
    # 2 conditional
    if 'pc_oracle' in res:
        reach = np.array([hit(r) for r in res['pc_oracle']])
        L += [f'## 2. 可达目标上的条件成功率（全信息参照能打中的 {int(reach.sum())}/{len(reach)} 个目标）', '', '| 方法 | 条件成功率 |', '|---|---|']
        for m in methods:
            L.append(f"| {name(m)} | {100*np.mean(np.array([hit(r) for r in res[m]])[reach]):.1f}% |")
        L.append('')
    # 3 paired comparisons
    H = {m: np.array([hit(r) for r in res[m]], float) for m in methods}
    V = {m: np.array([not r['feasible'] for r in res[m]], float) for m in methods}
    comps = []
    if 'pc_fixedlike' in res and 'pc_morph' in res:
        comps.append(('变构的价值：预测-校正（变构）− 预测-校正（固定外形）', 'pc_fixedlike', 'pc_morph', None))
    for s in seeds:
        a, b = f'rl_fixed:{s}', f'rl_morph:{s}'
        if a in res and b in res:
            comps.append((f'H1 变构消融（种子 {s}）：强化学习变构 − 强化学习不变构', a, b, 'H1'))
        if 'pc_fixedlike' in res and a in res:
            comps.append((f'H2 方法消融（种子 {s}）：强化学习（不变构）− 预测-校正（固定外形）', 'pc_fixedlike', a, 'H2'))
        if 'pc_weak' in res and a in res:
            comps.append((f'方法消融，旧弱基线（种子 {s}）：强化学习（不变构）− 预测-校正（只匹配航程）', 'pc_weak', a, None))
        if 'rule_morph' in res and b in res:
            comps.append((f'学习的作用（变构，种子 {s}）：强化学习 − 起点规则', 'rule_morph', b, None))
        if 'rule_fixed' in res and a in res:
            comps.append((f'学习的作用（不变构，种子 {s}）：强化学习 − 起点规则', 'rule_fixed', a, None))
        if 'pc_morph' in res and b in res:
            comps.append((f'参照：强化学习（变构，种子 {s}）− 预测-校正（变构）', 'pc_morph', b, None))
    L += ['## 3. 配对比较（成功率差，百分点；区间为整场景重抽样的 95% 自助区间）', '', '| 比较 | 差（百分点） | 95% 区间 | 超限数差 |', '|---|---|---|---|']
    verdict = {'H1': [], 'H2': []}
    for lab, a, b, h in comps:
        d, lo, hi = cluster_diff(H[a], H[b], scen[:len(H[a])] if len(scen) == len(H[a]) else scen, nb)
        dv = int(V[b].sum()-V[a].sum())
        L.append(f'| {lab} | {100*d:+.1f} | [{100*lo:+.1f}, {100*hi:+.1f}] | {dv:+d} |')
        if h:
            verdict[h].append((lo > 0 and dv <= 0, lo >= -.05, lab))
    L += ['', '## 4. 判定（预注册 v2 第 7 节的默认设定；待导师确认项见 PREREGISTRATION_V2.md）', '']
    for h, title in (('H1', 'H1 变构消融'), ('H2', 'H2 方法消融')):
        if verdict[h]:
            ok = [v[0] for v in verdict[h]]
            ni = [v[1] for v in verdict[h]]
            L.append(f"- {title}：{'成立' if all(ok) else '不成立'}（每个种子：区间下界 > 0 且超限不更多；{sum(ok)}/{len(ok)} 个种子满足）"
                     + (f"；非劣效（下界 ≥ −5 个百分点）：{sum(ni)}/{len(ni)} 个种子满足" if h == 'H2' else ''))
    # 5 domain
    dom_note = ''
    if with_domain:
        dm = [m for m in ['pc_weak', 'pc_fixedlike', 'pc_morph']+rl if (config.runs_dir()/'study_v2'/'domain'/f"{m.replace(':', '_')}.json").exists()]
        if dm:
            dres = {m: load_json(config.runs_dir()/'study_v2'/'domain'/f"{m.replace(':', '_')}.json")['rows'] for m in dm}
            out, lines, grid = domain_stats(dres, dm, cfg['experiment']['domain_levels'])
            figm = [m for m in dm if m in ('pc_fixedlike', 'pc_morph') or m.endswith(f':{seeds[0]}' if seeds else ':none')]
            fig = config.runs_dir()/'V2_domain_map.png'
            draw_map(out, figm, grid, fig)
            L += ['', '## 5. 距离—末速任务域（扰动下的成功率地图）', '', *lines, '',
                  f'地图：`{fig.name}`（红色实线 / 虚线：变构 / 固定外形的名义可达区域）。', '',
                  '各末速要求下最远可靠距离（成功率 ≥ 80%%，格距 %.0f km）：' % cfg['targets']['v2']['grid']['D'][2], '', *far_end(out, dm, grid)]
    L += ['', f'平台：{platform.system()} / Python {platform.python_version()}，numpy {np.__version__}。', '']
    text = '\n'.join(L)
    out_path = config.runs_dir()/f'V2_REPORT_{which}.md'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding='utf-8')
    safe_print(text)
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--workers', type=int, default=os.cpu_count() or 1)
    p.add_argument('--seeds', default=' '.join(str(s) for s in config.load()['seeds']['training_runs']))
    p.add_argument('--which', default='validation', choices=['validation', 'test'])
    p.add_argument('--no-domain', action='store_true')
    p.add_argument('--report-only', action='store_true')
    a = p.parse_args(argv)
    seeds = seeds_available([int(s) for s in a.seeds.replace(',', ' ').split()])
    base = ['pc_weak', 'pc_fixedlike', 'pc_morph', 'pc_oracle', 'rule_fixed', 'rule_morph']
    rl = [f'{k}:{s}' for s in seeds for k in ('rl_fixed', 'rl_morph')]
    try:
        if not a.report_only:
            for m in base+rl:
                evaluate(m, a.which, a.workers)
            if not a.no_domain:
                for m in ['pc_weak', 'pc_fixedlike', 'pc_morph']+rl:
                    evaluate(m, 'domain', a.workers)
        report(seeds, a.which, with_domain=not a.no_domain)
    finally:
        from .parallel import close
        close()


if __name__ == '__main__':
    main()

"""Paper tables and figures (EXPERIMENTS.md). Reads whatever runs exist and skips missing parts.

Outputs (runs/paper/):
  tables.md + table_*.csv
    T1 main results       (main library, L2)  hit rates by target group, violations, errors, compute per flight
    T2 paired comparisons (main library, L2)  full method vs pc_fair / pc_strong / pc_oracle / constant, per seed,
                                              cluster-bootstrap 95 % CI; H1 and G1
    T3 ablations          (main library, L2)
    T4 library sensitivity (L2)               reachable-area ratio, morphing-only band, hit rates
    T5 generalisation     (main library, L0-L3)
    T6 computation        (main library, L2)  wall time per flight and per decision, predictions per flight
  figures/F1..F7 (.png 300 dpi and .pdf): reachable regions, learning curves, per-target hit rates, hit rate vs
  disturbance level, example trajectories, terminal errors, compute vs hit rate.

    python -m target_rl.paper
"""
import csv
import json
import os

import numpy as np

from . import config
from .report import group_of, paired
from .util import load_json

RUNS = config.ROOT/'runs'
OUT = RUNS/'paper'
GROUPS = (('fixed', 'T1–T3'), ('morph', 'T4–T6'), ('random', 'R1–R10'))
METHOD_LABEL = dict(constant=('常值策略 π₀', 'Constant π₀'), pc_fixed=('预测-校正（不变构）', 'PC (no morphing)'),
                    pc_fair=('预测-校正（公平版）', 'PC (fair)'), pc_strong=('预测-校正（强版）', 'PC (strong)'),
                    pc_oracle=('预测-校正（全信息）', 'PC (oracle)'), full=('本文方法', 'Proposed'),
                    dist=('加扰动输入', '+disturbance inputs'), fixed=('不变构', 'No morphing'),
                    mean=('均值打分', 'Mean score'))
BASE_ORDER = ('constant', 'pc_fixed', 'pc_fair', 'pc_strong', 'pc_oracle')


# ------------------------------------------------------------------------------------------- data
def rows_of(lib, level):
    d = RUNS/lib/'eval'/level
    out = {}
    if d.exists():
        for p in sorted(d.glob('*.json')):
            j = load_json(p)
            out[j['method']] = j['rows']
    return out


def rl_runs(M, variant):
    return sorted((m for m in M if m.startswith(f'rl:{variant}:')), key=lambda m: int(m.split(':')[2]))


def hit_rate(rows, group=None):
    rr = [r for r in rows if group is None or group_of(r['target']) == group]
    return float(np.mean([r['hit'] for r in rr])) if rr else np.nan


def viol_rate(rows):
    return float(np.mean([not r['feasible'] for r in rows])) if rows else np.nan


def med(rows, key, group=None):
    v = [abs(r[key]) for r in rows if r['feasible'] and (group is None or group_of(r['target']) == group)]
    return float(np.median(v)) if v else np.nan


def stat_row(rows):
    return dict(**{g: hit_rate(rows, g) for g, _ in GROUPS}, all=hit_rate(rows), viol=viol_rate(rows),
                dD=med(rows, 'dD_km', 'morph'), dV=med(rows, 'dV_mps', 'morph'),
                wall=float(np.median([r['wall_s'] for r in rows if 'wall_s' in r])) if any('wall_s' in r for r in rows) else np.nan,
                pred=float(np.median([r['n_predictions'] for r in rows if 'n_predictions' in r])) if any('n_predictions' in r for r in rows) else np.nan)


def seed_stats(M, variant):
    """Mean and standard deviation over training seeds of every stat_row entry."""
    runs = rl_runs(M, variant)
    if not runs:
        return None, 0
    S = [stat_row(M[m]) for m in runs]
    return {k: (float(np.nanmean([s[k] for s in S])), float(np.nanstd([s[k] for s in S]))) for k in S[0]}, len(runs)


# ------------------------------------------------------------------------------------------- tables
def pct(v, sd=None):
    if v is None or not np.isfinite(v):
        return '—'
    return f'{100*v:.1f}' if sd is None else f'{100*v:.1f} ± {100*sd:.1f}'


def num(v, f='{:.1f}'):
    return '—' if v is None or not np.isfinite(v) else f.format(v)


class Tables:
    def __init__(self):
        self.md, self.csv = [], {}

    def add(self, key, title, header, rows, note=None):
        self.md += [f'## {title}', '', '| '+' | '.join(header)+' |', '|'+'---|'*len(header)]
        self.md += ['| '+' | '.join(str(c) for c in r)+' |' for r in rows]
        if note:
            self.md += ['', note]
        self.md.append('')
        self.csv[key] = [header]+[[str(c) for c in r] for r in rows]

    def write(self):
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT/'tables.md').write_text('# 论文表格（自动生成）\n\n'+'\n'.join(self.md), encoding='utf-8')
        for k, rows in self.csv.items():
            with (OUT/f'table_{k}.csv').open('w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerows(rows)


def table_main(T, M, lib):
    header = ['方法', 'T1–T3 命中率 %', 'T4–T6 命中率 %', '随机目标命中率 %', '总命中率 %', '超限率 %',
              'T4–T6 |航程误差| 中位数 km', 'T4–T6 |末速误差| 中位数 m/s', '单次飞行耗时 s']
    rows = []
    for m in BASE_ORDER:
        if m in M:
            s = stat_row(M[m])
            rows.append([METHOD_LABEL[m][0], pct(s['fixed']), pct(s['morph']), pct(s['random']), pct(s['all']),
                         pct(s['viol']), num(s['dD']), num(s['dV'], '{:.0f}'), num(s['wall'], '{:.2f}')])
    s, n = seed_stats(M, 'full')
    if s:
        rows.append([f"{METHOD_LABEL['full'][0]}（{n} 个种子）", pct(*s['fixed']), pct(*s['morph']), pct(*s['random']),
                     pct(*s['all']), pct(*s['viol']), num(s['dD'][0]), num(s['dV'][0], '{:.0f}'), num(s['wall'][0], '{:.2f}')])
    T.add('T1_main', f'表 1  主结果（{lib}，L2 扰动，20 个测试场景 × 16 个目标）', header, rows,
          '本文方法为各训练种子的均值 ± 标准差。命中：无超限且 |航程误差| ≤ 5 km、|末速误差| ≤ 25 m/s。')


def table_paired(T, M, lib, report):
    header = ['种子', '对比基线', '目标组', '命中率差（百分点）', '95% 置信区间', '本文方法超限率 %', '基线超限率 %']
    rows = []
    for m in rl_runs(M, 'full'):
        seed = m.split(':')[2]
        for ref in ('pc_fair', 'pc_strong', 'pc_oracle', 'constant'):
            if ref not in M:
                continue
            for g, gl in (('morph', 'T4–T6'), ('fixed', 'T1–T3'), ('random', 'R1–R10')):
                p = paired(M[m], M[ref], g)
                if p:
                    rows.append([seed, METHOD_LABEL[ref][0], gl, f"{100*p['diff']:+.1f}",
                                 f"[{100*p['ci95'][0]:+.1f}, {100*p['ci95'][1]:+.1f}]", pct(p['violation_a']), pct(p['violation_b'])])
    g1 = report.get('G1', {})
    note = ('置信区间：按 20 个测试场景做聚类自助法（10000 次）。'
            + (f"预注册判定：G1 {'通过' if g1.get('passed') else '未通过'}（平均命中率差 {100*g1['mean_diff']:+.1f} 个百分点，"
               f"H1 成立 {g1['h1_holds']}/5）。" if g1.get('passed') is not None else ''))
    T.add('T2_paired', f'表 2  配对比较（{lib}，L2）', header, rows, note)


def table_ablation(T, M, lib):
    header = ['变体', '种子数', 'T1–T3 命中率 %', 'T4–T6 命中率 %', '随机目标命中率 %', '总命中率 %', '超限率 %']
    rows = []
    for v in ('full', 'dist', 'fixed', 'mean'):
        s, n = seed_stats(M, v)
        if s:
            rows.append([METHOD_LABEL[v][0], n, pct(*s['fixed']), pct(*s['morph']), pct(*s['random']), pct(*s['all']),
                         pct(*s['viol'])])
    T.add('T3_ablation', f'表 3  消融（{lib}，L2，均值 ± 标准差）', header, rows)


def table_libraries(T, libs):
    header = ['构型库', '前缘', '构型数', '可达面积比', '只有变构能到的带宽 m/s', 'T4–T6 命中率：公平版 %', '强版 %',
              '本文方法 %', '总命中率：本文方法 %']
    rows = []
    edge = dict(F8D2000B='固定', F8D2000='固定', L298='不固定')
    for lib in libs:
        reg = RUNS/lib/'region.json'
        if not reg.exists():
            continue
        r = load_json(reg)
        bands = [t['band_mps'] for t in r['targets'] if 'band_mps' in t]
        M = rows_of(lib, 'L2')
        s, n = seed_stats(M, 'full')
        n_geo = r.get('n_geometries', {'F8D2000B': 2091, 'F8D2000': 2000, 'L298': 298}.get(lib, '—'))
        rows.append([lib, edge.get(lib, '—'), n_geo, f"{r['area_ratio']:.2f}", f'{min(bands):.0f}–{max(bands):.0f}',
                     pct(hit_rate(M['pc_fair'], 'morph')) if 'pc_fair' in M else '—',
                     pct(hit_rate(M['pc_strong'], 'morph')) if 'pc_strong' in M else '—',
                     pct(*s['morph']) if s else '—', pct(*s['all']) if s else '—'])
    T.add('T4_libraries', '表 4  构型库敏感性（L2）', header, rows)


def table_generalisation(T, lib, levels):
    header = ['方法']+[f'{lv} 总命中率 %' for lv in levels]+[f'{lv} T4–T6 %' for lv in levels]
    rows = []
    Ms = {lv: rows_of(lib, lv) for lv in levels}
    for m in ('constant', 'pc_fair', 'pc_strong', 'pc_oracle', 'full'):
        cells_all, cells_m = [], []
        for lv in levels:
            M = Ms[lv]
            if m == 'full':
                s, n = seed_stats(M, 'full')
                cells_all.append(pct(*s['all']) if s else '—')
                cells_m.append(pct(*s['morph']) if s else '—')
            else:
                cells_all.append(pct(hit_rate(M[m])) if m in M else '—')
                cells_m.append(pct(hit_rate(M[m], 'morph')) if m in M else '—')
        rows.append([METHOD_LABEL[m][0]]+cells_all+cells_m)
    T.add('T5_generalisation', f'表 5  扰动强度泛化（{lib}，训练于 L2，不重新训练）', header, rows,
          'L0：只有初始偏差；L1：密度 σ 3%；L2：σ 6%（训练）；L3：σ 12%。')


def table_compute(T, M, lib):
    header = ['方法', '单次飞行耗时中位数 s', '每次决策耗时 ms', '每次飞行的轨迹预测次数']
    rows = []
    for m in BASE_ORDER:
        if m in M and any('wall_s' in r for r in M[m]):
            s = stat_row(M[m])
            dec = float(np.median([r['wall_s']/max(r.get('decisions', 100), 1) for r in M[m] if 'wall_s' in r]))
            rows.append([METHOD_LABEL[m][0], num(s['wall'], '{:.2f}'), num(1e3*dec, '{:.1f}'), num(s['pred'], '{:.0f}')])
    runs = rl_runs(M, 'full')
    if runs:
        allr = [r for m in runs for r in M[m] if 'wall_s' in r]
        if allr:
            rows.append([METHOD_LABEL['full'][0], num(float(np.median([r['wall_s'] for r in allr])), '{:.2f}'),
                         num(1e3*float(np.median([r['wall_s']/max(r.get('decisions', 100), 1) for r in allr])), '{:.1f}'), '0'])
    T.add('T6_compute', f'表 6  计算量（{lib}，L2；单进程单线程，包括飞行仿真本身）', header, rows,
          '耗时包含 0.5 s 步长的飞行仿真；预测-校正的耗时主要来自轨迹预测。')


# ------------------------------------------------------------------------------------------- figures
def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    cjk = [n for n in ('Noto Sans CJK SC', 'Source Han Sans SC', 'WenQuanYi Zen Hei', 'SimHei', 'Microsoft YaHei') if n in names]
    plt.rcParams['font.sans-serif'] = cjk+['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return plt, (0 if cjk else 1)


def _save(fig, name):
    d = OUT/'figures'
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d/f'{name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(d/f'{name}.pdf', bbox_inches='tight')


COLORS = dict(constant='#999999', pc_fixed='#c7b299', pc_fair='#e9a03b', pc_strong='#d1495b', pc_oracle='#6c5b7b',
              full='#1f6fb4', dist='#2a9d8f', fixed='#8ab17d', mean='#264653')


def fig_regions(libs):
    plt, L = _plt()
    libs = [x for x in libs if (RUNS/x/'region.json').exists()]
    if not libs:
        return
    fig, axes = plt.subplots(1, len(libs), figsize=(4.6*len(libs), 3.8), sharex=True, sharey=True, squeeze=False)
    for ax, lib in zip(axes[0], libs):
        d = load_json(RUNS/lib/'region.json')
        for key, color, lab in (('morph_hull', '#1f6fb4', ('变构可达', 'With morphing')),
                                ('fixed_hull', '#d1495b', ('不变构可达', 'Fixed geometry'))):
            H = np.array(d[key])
            H = np.vstack([H, H[:1]])
            ax.fill(H[:, 0], H[:, 1], color=color, alpha=.18, lw=0)
            ax.plot(H[:, 0], H[:, 1], color=color, lw=1.4, label=lab[L])
        for t in d['targets']:
            ax.plot(t['range_km'], t['speed_mps'], 'o' if t['kind'] == 'fixed_reachable' else 's', color='#222', ms=3.5)
            ax.annotate(t['name'], (t['range_km'], t['speed_mps']), textcoords='offset points', xytext=(3, 3), fontsize=7)
        ax.set_title(f"{lib}  ({['面积比', 'area ratio'][L]} {d['area_ratio']:.2f})", fontsize=9)
        ax.set_xlabel(['1200 s 航程 (km)', 'Range at 1200 s (km)'][L])
    axes[0][0].set_ylabel(['1200 s 末速 (m/s)', 'Speed at 1200 s (m/s)'][L])
    axes[0][0].legend(fontsize=7, frameon=False, loc='upper left')
    _save(fig, 'F1_reachable_regions')
    plt.close(fig)


def fig_learning(lib):
    plt, L = _plt()
    d = RUNS/lib/'train'
    if not d.exists():
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    drawn = False
    for v in ('full', 'dist', 'fixed', 'mean'):
        curves = []
        for p in sorted(d.glob(f'{v}_s*/state.json')):
            st = load_json(p)
            curves.append([h['G_before'] for h in st['history']]+[st['G']])
        if not curves:
            continue
        n = max(len(c) for c in curves)
        A = np.array([c+[c[-1]]*(n-len(c)) for c in curves])
        x = np.arange(n)
        ax.plot(x, A.mean(0), color=COLORS[v], lw=1.6, label=f'{METHOD_LABEL[v][L]} (n={len(curves)})')
        ax.fill_between(x, A.min(0), A.max(0), color=COLORS[v], alpha=.15, lw=0)
        drawn = True
    if not drawn:
        plt.close(fig)
        return
    ax.set_xlabel(['策略迭代轮次', 'Policy-iteration round'][L])
    ax.set_ylabel(['训练场景平均回报', 'Mean training return'][L])
    ax.set_title(['学习曲线（线：种子均值；带：最小到最大）', 'Learning curves (line: mean; band: min-max over seeds)'][L], fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    _save(fig, 'F2_learning_curves')
    plt.close(fig)


def fig_targets(M, lib):
    plt, L = _plt()
    names = [f'T{i}' for i in range(1, 7)]
    methods = [m for m in ('constant', 'pc_fair', 'pc_strong', 'pc_oracle') if m in M]
    if not methods and not rl_runs(M, 'full'):
        return
    fig, ax = plt.subplots(figsize=(8, 3.2))
    series = [(m, [hit_rate([r for r in M[m] if r['target'] == t]) for t in names]) for m in methods]
    runs = rl_runs(M, 'full')
    if runs:
        series.append(('full', [np.mean([hit_rate([r for r in M[m] if r['target'] == t]) for m in runs]) for t in names]))
    w = .8/len(series)
    for k, (m, y) in enumerate(series):
        ax.bar(np.arange(6)+(k-(len(series)-1)/2)*w, 100*np.array(y), w, color=COLORS[m], label=METHOD_LABEL[m][L])
    ax.axvspan(2.5, 5.5, color='#eeeeee', zorder=0)
    ax.set_xticks(range(6), names)
    ax.set_ylabel(['命中率 (%)', 'Hit rate (%)'][L])
    ax.set_title(['各目标命中率（灰底为只有变构能到的目标）', 'Hit rate per target (shaded: morphing-only)'][L], fontsize=9)
    ax.legend(fontsize=7, frameon=False, ncol=3)
    _save(fig, 'F3_hit_rate_per_target')
    plt.close(fig)


def fig_levels(lib, levels):
    plt, L = _plt()
    order = ['L0', 'L1', 'L2', 'L3']
    lv = [x for x in order if x in levels or x == 'L2']
    Ms = {x: rows_of(lib, x) for x in lv}
    if sum(bool(Ms[x]) for x in lv) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, g, title in ((axes[0], None, ('全部目标', 'All targets')), (axes[1], 'morph', ('T4–T6', 'T4–T6'))):
        for m in ('constant', 'pc_fair', 'pc_strong', 'pc_oracle', 'full'):
            y = []
            for x in lv:
                M = Ms[x]
                if m == 'full':
                    runs = rl_runs(M, 'full')
                    y.append(np.mean([hit_rate(M[r], g) for r in runs]) if runs else np.nan)
                else:
                    y.append(hit_rate(M[m], g) if m in M else np.nan)
            ax.plot(range(len(lv)), 100*np.array(y), '-o', ms=3, color=COLORS[m], label=METHOD_LABEL[m][L])
        ax.set_xticks(range(len(lv)), lv)
        ax.set_title(title[L], fontsize=9)
        ax.set_xlabel(['扰动等级（L2 为训练等级）', 'Disturbance level (trained on L2)'][L])
    axes[0].set_ylabel(['命中率 (%)', 'Hit rate (%)'][L])
    axes[0].legend(fontsize=7, frameon=False)
    _save(fig, 'F4_hit_rate_vs_level')
    plt.close(fig)


def fig_traces(lib):
    plt, L = _plt()
    d = RUNS/lib/'traces'/'L2'
    if not d.exists():
        return
    data = {}
    for p in d.glob('*.json'):
        j = load_json(p)
        data[j['method']] = j['traces']
    runs = sorted((m for m in data if m.startswith('rl:full:')), key=lambda m: int(m.split(':')[2]))
    methods = [m for m in ('constant', 'pc_fair', 'pc_strong') if m in data]+runs[:1]
    if not methods:
        return
    cases = sorted({(t['seed'], t['target']) for m in methods for t in data[m]})
    fig, axes = plt.subplots(len(cases), 4, figsize=(13, 2.9*len(cases)), squeeze=False, constrained_layout=True)
    cols = [(1, ('高度 (km)', 'Altitude (km)'), 1e-3), (2, ('速度 (m/s)', 'Speed (m/s)'), 1),
            (5, ('升力系数', 'Lift coefficient'), 1), (6, ('变构程度 w', 'Morph level w'), 1)]
    for i, case in enumerate(cases):
        for m in methods:
            tr = [t for t in data[m] if (t['seed'], t['target']) == case]
            if not tr:
                continue
            A = np.array(tr[0]['trace'], float)
            key = 'full' if m.startswith('rl:') else m
            for j, (c, lab, sc) in enumerate(cols):
                axes[i][j].plot(A[:, 0], A[:, c]*sc, color=COLORS[key], lw=1.1, label=METHOD_LABEL[key][L])
        for j, (c, lab, sc) in enumerate(cols):
            axes[i][j].set_ylabel(lab[L], fontsize=8)
            if i == len(cases)-1:
                axes[i][j].set_xlabel(['时间 (s)', 'Time (s)'][L], fontsize=8)
        axes[i][0].set_title(f'{["场景", "Scenario"][L]} {case[0]}, {case[1]}', fontsize=9, loc='left')
    axes[0][0].legend(fontsize=7, frameon=False)
    _save(fig, 'F5_example_trajectories')
    plt.close(fig)


def fig_errors(M, lib):
    plt, L = _plt()
    fig, ax = plt.subplots(figsize=(5, 4))
    any_ = False
    for m in ('pc_fair', 'pc_strong', 'full'):
        rows = [r for k in (rl_runs(M, 'full') if m == 'full' else [m]) if k in M for r in M[k]
                if group_of(r['target']) == 'morph' and r['feasible']]
        if rows:
            any_ = True
            ax.scatter([r['dD_km'] for r in rows], [r['dV_mps'] for r in rows], s=6, alpha=.5, color=COLORS[m],
                       label=METHOD_LABEL[m][L])
    if not any_:
        plt.close(fig)
        return
    ax.add_patch(__import__('matplotlib.patches', fromlist=['Rectangle']).Rectangle((-5, -25), 10, 50, fill=False, lw=1))
    ax.set_xlim(-100, 100)
    ax.set_ylim(-250, 250)
    ax.set_xlabel(['航程误差 (km)', 'Range error (km)'][L])
    ax.set_ylabel(['末速误差 (m/s)', 'Speed error (m/s)'][L])
    ax.set_title(['T4–T6 终端误差（框为命中容差）', 'Terminal errors, T4–T6 (box: hit tolerance)'][L], fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    _save(fig, 'F6_terminal_errors')
    plt.close(fig)


def fig_compute(M, lib):
    plt, L = _plt()
    pts = []
    for m in BASE_ORDER:
        if m in M and any('wall_s' in r for r in M[m]):
            pts.append((m, stat_row(M[m])['wall'], hit_rate(M[m])))
    runs = rl_runs(M, 'full')
    if runs:
        pts.append(('full', float(np.median([r['wall_s'] for k in runs for r in M[k] if 'wall_s' in r])),
                    float(np.mean([hit_rate(M[k]) for k in runs]))))
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for m, w, h in pts:
        ax.scatter(w, 100*h, s=40, color=COLORS[m])
        ax.annotate(METHOD_LABEL[m][L], (w, 100*h), textcoords='offset points', xytext=(5, 3), fontsize=7)
    ax.set_xscale('log')
    ax.set_xlabel(['单次飞行耗时 (s，对数)', 'Wall time per flight (s, log)'][L])
    ax.set_ylabel(['总命中率 (%)', 'Overall hit rate (%)'][L])
    _save(fig, 'F7_compute_vs_hit_rate')
    plt.close(fig)


# ------------------------------------------------------------------------------------------- main
def main():
    cfg = config.load()
    ex = cfg['experiment']
    lib = ex['main_library']
    libs = [lib]+list(ex['sensitivity_libraries'])
    levels = list(ex['generalization_levels'])
    os.environ['TARGET_RL_LIBRARY'] = lib
    M = rows_of(lib, 'L2')
    from .report import build
    report = build('L2') if M else {}
    T = Tables()
    table_main(T, M, lib)
    table_paired(T, M, lib, report)
    table_ablation(T, M, lib)
    table_libraries(T, libs)
    table_generalisation(T, lib, ['L0', 'L1', 'L2', 'L3'])
    table_compute(T, M, lib)
    T.write()
    for f in (lambda: fig_regions(libs), lambda: fig_learning(lib), lambda: fig_targets(M, lib),
              lambda: fig_levels(lib, levels), lambda: fig_traces(lib), lambda: fig_errors(M, lib),
              lambda: fig_compute(M, lib)):
        f()
    print((OUT/'tables.md').read_text(encoding='utf-8'))
    print('figures:', sorted(p.name for p in (OUT/'figures').glob('*.png')) if (OUT/'figures').exists() else [])


if __name__ == '__main__':
    main()

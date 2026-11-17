"""Figures of the library design study (docs/figures/*.png)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from .paths import OUT, ROOT, load  # noqa: E402

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
FIG = ROOT/'docs'/'figures'
LABEL = dict(front='前部', rear='后部', tail='尾部', span='展向', flap='襟翼', twist='扭转')
INK, MUTED = '#222222', '#777777'
C_LOW, C_HIGH = '#1f6fb4', '#d1495b'


def sweeps(parent='G42'):
    d = load(OUT/'sweep.json')
    fig, axes = plt.subplots(2, 6, figsize=(15, 5.4), sharey=True)
    for i, surf in enumerate(('upper', 'lower')):
        for j, b in enumerate(('front', 'rear', 'tail', 'span', 'flap', 'twist')):
            ax = axes[i, j]
            rows = sorted([r for r in d['rows'] if r['parent'] == parent and r['surface'] == surf and r['basis'] == b],
                          key=lambda r: r['scale'])
            base = np.array([r['cd'] for r in rows if r['scale'] == 0][0], float)
            s = np.array([r['scale'] for r in rows])
            ok = np.array([r['geometry_ok'] for r in rows])
            for (ci, label, color) in ((0, 'CL 0.02', C_LOW), (3, 'CL 0.065', C_HIGH)):
                y = np.array([np.array(r['cd'], float)[1, ci]/base[1, ci] for r in rows])
                ax.plot(s, y, '-', color=color, lw=1.6, label=label)
                ax.plot(s[~ok], y[~ok], 'x', color=color, ms=6)
            ax.axhline(1, color=MUTED, lw=.8)
            ax.axvspan(-1, 1, color='#eeeeee', zorder=0)
            ax.set_title(f"{'上' if surf == 'upper' else '下'}表面·{LABEL[b]}", fontsize=10)
            ax.set_ylim(.6, 2.2)
            if i == 1:
                ax.set_xlabel('系数 / F8D2000 上限')
    axes[0, 0].set_ylabel('同升力阻力 / 母构型')
    axes[1, 0].set_ylabel('同升力阻力 / 母构型')
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle(f'单参数扫描（母构型 {parent}，马赫 7；灰带为 F8D2000 采样范围，× 为几何无效）', fontsize=12)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG/'01_单参数扫描.png', dpi=160)
    plt.close(fig)


def camber(parent='G42'):
    d = load(OUT/'sweep_camber.json')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (ci, title) in zip(axes, ((0, 'CL 0.02'), (3, 'CL 0.065'))):
        for m, color in zip(('front', 'rear', 'flap', 'span', 'tail', 'twist'),
                            ('#1f6fb4', '#d1495b', '#2a9d8f', '#e9a03b', '#6c5b7b', '#999999')):
            rows = sorted([r for r in d['rows'] if r['parent'] == parent and r['mode'] == m], key=lambda r: r['coef_m'])
            base = np.array([r['cd'] for r in rows if r['coef_m'] == 0][0], float)
            c = np.array([r['coef_m'] for r in rows])
            ok = np.array([r['geometry_ok'] for r in rows])
            y = np.array([np.array(r['cd'], float)[1, ci]/base[1, ci] for r in rows])
            ax.plot(c[ok], y[ok], '-o', color=color, ms=3, lw=1.5, label=LABEL[m])
        ax.axhline(1, color=MUTED, lw=.8)
        ax.set_yscale('log')
        ax.set_title(f'马赫 7，{title}', fontsize=10)
        ax.set_xlabel('上下表面同向位移幅值 (m)')
    axes[0].set_ylabel('同升力阻力 / 母构型')
    axes[0].legend(fontsize=8, frameon=False, ncol=2)
    fig.suptitle(f'弯度模态（上下表面一起移动，母构型 {parent}）：只增阻，可作"减速板"', fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG/'02_弯度模态.png', dpi=160)
    plt.close(fig)




def regions():
    import json
    sets = [('F8D2000（前缘固定）', ROOT/'runs'/'F8D2000'/'region.json'),
            ('F8D2000 + 减速板阶梯（前缘固定）', ROOT/'runs'/'F8D2000B'/'region.json'),
            ('L298（前缘不固定）', ROOT/'runs'/'L298'/'region.json')]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True, sharey=True)
    for ax, (title, p) in zip(axes, sets):
        d = json.loads(p.read_text())
        for key, color, label in (('morph_hull', '#1f6fb4', '变构可达'), ('fixed_hull', '#d1495b', '不变构可达')):
            H = np.array(d[key])
            H = np.vstack([H, H[:1]])
            ax.fill(H[:, 0], H[:, 1], color=color, alpha=.18, lw=0)
            ax.plot(H[:, 0], H[:, 1], color=color, lw=1.6, label=label)
        for t in d['targets']:
            ax.plot(t['range_km'], t['speed_mps'], 'o' if t['kind'] == 'fixed_reachable' else 's', color=INK, ms=4)
            ax.annotate(t['name'], (t['range_km'], t['speed_mps']), textcoords='offset points', xytext=(4, 3), fontsize=8)
        ax.set_title(f"{title}\n面积比 {d['area_ratio']:.2f}", fontsize=10)
        ax.set_xlabel('1200 s 航程 (km)')
    axes[0].set_ylabel('1200 s 末速 (m/s)')
    axes[0].legend(fontsize=8, frameon=False, loc='upper left')
    fig.suptitle('定点到达任务的名义可达范围：加入减速板阶梯后，前缘固定的库也有大范围"只有变构才能到"的区域', fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG/'03_可达范围对比.png', dpi=160)
    plt.close(fig)


def bounds():
    d = load(OUT/'pooled.json')
    fig, ax = plt.subplots(figsize=(8, 3.6))
    parents = ('G42', 'F2K_01233', 'F2K_00021')
    w = .12
    for i, (s, k, color) in enumerate([(s, k, c) for s in ('f8d', 'f8d_flap') for k, c in zip((1, 2, 3), ('#9ecae1', '#4292c6', '#08519c') if s == 'f8d' else ('#fcbba1', '#fb6a4a', '#a50f15'))]):
        y = [100*d[f'{p}|{s}|{k:g}']['gain_bound'] for p in parents]
        ax.bar(np.arange(3)+(i-2.5)*w, y, w, color=color, label=f"{'F8D 8 参数' if s == 'f8d' else '+襟翼/扭转'}，{k} 倍范围")
    ax.set_xticks(range(3), parents)
    ax.set_ylabel('静态变构收益上限 (%)')
    ax.axhline(100*0.0041, color=INK, lw=1, ls='--')
    ax.text(2.45, 100*0.0041+.02, 'F8D2000 实际 0.41%', fontsize=8, ha='right')
    ax.legend(fontsize=7.5, frameon=False, ncol=2)
    ax.set_title('按工况优化的"专长构型"相对最佳固定构型的平均减阻上限（12 个工况）', fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG/'04_静态收益上限.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    sweeps()
    camber()
    regions()
    bounds()

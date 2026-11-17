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


if __name__ == '__main__':
    sweeps()
    camber()

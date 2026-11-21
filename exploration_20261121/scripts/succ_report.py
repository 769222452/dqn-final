import json, sys, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from target_rl import region as RG
d = json.load(open(sys.argv[1])); pts = [tuple(p) for p in d['pts']]; rows = d['rows']
REG = RG.load_region(); FXH = np.asarray(REG['fixed_hull']); MH = np.asarray(REG['morph_hull'])
S = {}
for morph, D, V, seed, hit, fail in rows: S.setdefault((morph, D, V), []).append(hit)
sr = {k: np.mean(v) for k, v in S.items()}
cell = 50.*75.
print('grid points %d; cell area %.0f km*m/s' % (len(pts), cell))
for p0 in (0.5, 0.8, 0.9):
    af = sum(sr[(False,)+p] >= p0 for p in pts); am = sum(sr[(True,)+p] >= p0 for p in pts)
    print('reliable domain (success >= %.0f%%): fixed-like %2d cells, morph %2d cells  -> morph / fixed = %s' % (100*p0, af, am, ('%.2f' % (am/af)) if af else 'inf'))
print('mean success over the grid: fixed-like %.1f%%, morph %.1f%%' % (100*np.mean([sr[(False,)+p] for p in pts]), 100*np.mean([sr[(True,)+p] for p in pts])))
inf = [p for p in pts if RG.inside(FXH, p)]; outf = [p for p in pts if not RG.inside(FXH, p)]
for lab, sel in (('points inside the nominal fixed-geometry region', inf), ('points only the morph region contains', outf)):
    print('%-48s (%2d pts): fixed-like %.1f%%, morph %.1f%%' % (lab, len(sel), 100*np.mean([sr[(False,)+p] for p in sel]), 100*np.mean([sr[(True,)+p] for p in sel])))
print('\nfar-end of the reliable domain (success >= 80%) at each terminal speed [km]')
print('%8s | %-12s | %-12s | %s' % ('V*', 'fixed-like', 'morph', 'gain'))
for V in sorted({p[1] for p in pts}):
    f = [p[0] for p in pts if p[1] == V and sr[(False,)+p] >= .8]; m = [p[0] for p in pts if p[1] == V and sr[(True,)+p] >= .8]
    fs = '%.0f' % max(f) if f else '-'; ms = '%.0f' % max(m) if m else '-'
    g = '%+.0f' % (max(m)-max(f)) if (f and m) else ('only morph' if m else '-')
    print('%8.0f | %-12s | %-12s | %s' % (V, fs, ms, g))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
for a, (morph, title) in zip(ax, ((False, 'Fixed geometry (min-drag shape)'), (True, 'Morphing (library shapes, 8 s transition)'))):
    X = [p[0] for p in pts]; Y = [p[1] for p in pts]; C = [sr[(morph,)+p]*100 for p in pts]
    sc = a.scatter(X, Y, c=C, cmap='Blues', vmin=0, vmax=100, s=170, marker='s', edgecolor='#444', linewidth=.4)
    for H, ls, lab in ((FXH, '--', 'nominal fixed-geometry region'), (MH, '-', 'nominal morphing region')):
        P = np.vstack([H, H[0]]); a.plot(P[:, 0], P[:, 1], ls, color='#b03a2e', lw=1.2, label=lab)
    a.set_title(title, fontsize=10); a.set_xlabel('target range D (km)'); a.grid(alpha=.2)
ax[0].set_ylabel('required terminal speed V (m/s)'); ax[0].legend(fontsize=7, loc='lower right')
fig.colorbar(sc, ax=ax, label='success rate (%)', shrink=.9)
fig.savefig(sys.argv[2], dpi=160, bbox_inches='tight'); print('figure', sys.argv[2])

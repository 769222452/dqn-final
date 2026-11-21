import numpy as np, sys
from scipy.interpolate import PchipInterpolator
from target_rl import config, aero
cfg = config.load(); a = dict(cfg['aero'], **config.library(cfg))
paths = a['path'] if isinstance(a['path'], list) else [a['path']]
parts = [np.load(config.ROOT/p, allow_pickle=False) for p in paths]
CL = np.concatenate([q['CL'] for q in parts]); CD = np.concatenate([q['CD_pressure'] for q in parts])
M = parts[0]['mach_grid'].astype(float); A = np.deg2rad(parts[0]['alpha_deg_grid'].astype(float))
alpha = np.deg2rad(np.arange(np.rad2deg(A[0]), np.rad2deg(A[-1])+1e-9, a['alpha_step_deg'])); alpha = np.clip(alpha, A[0], A[-1])
lo, hi, step = a['cl_table']; cls = np.round(np.arange(lo, hi+step/2, step), 6)
W = np.full((len(M), len(cls), len(CL)), np.nan)
for i in range(len(M)):
    y = PchipInterpolator(A, np.stack([CL[:, i], CD[:, i]], -1), axis=1)(alpha)
    cd = np.array([aero._cd_at_cl(alpha, y[g, :, 0], y[g, :, 1], cls) for g in range(len(CL))])   # [geometry, cl]
    for j in range(len(cls)):
        v = cd[:, j]; ok = np.isfinite(v)
        if ok.sum() >= 2:
            ww = (v[ok]-v[ok].min())/(v[ok].max()-v[ok].min()); W[i, j, :ok.sum()] = np.sort(ww)
np.savez(sys.argv[1], M=M, cls=cls, W=W); print('saved', W.shape)

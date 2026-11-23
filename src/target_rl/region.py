"""Target region and test targets (stage 1; PREREGISTRATION.md section 3 and amendment 1).

1. Reachable (range, final speed) at 1200 s on the nominal scenario (level L0), by a grid of three-phase plans:
   in each phase (0-400, 400-800, 800-1200 s) the dynamic-pressure law with set point r gives the CL command and
   w the drag level; the shared protection loop is active. Morphing: r x w per phase; fixed geometry: r only.
2. Convex hulls H_morph, H_fixed of the feasible end points. Target region R = H_morph shrunk by
   targets.region_shrink towards its centroid.
3. T1-T6 by rule, at the ranges D_k = D_lo + f_k (D_hi - D_lo), f = targets.fixed_fractions, [D_lo, D_hi] the
   range interval of H_fixed:
     T1-T3  speed = middle of the H_fixed chord at D_k (reachable without morphing)
     T4-T6  speed = middle of the band between the lower H_morph and lower H_fixed chords at D_k
            (reachable with morphing only)
4. Training targets (seeds.train) and random test targets (seeds.random_targets): uniform in R, outside the
   exclusion box around T1-T6, rng = default_rng([seed, 11]).

    python -m target_rl.region --workers 14            # writes runs/<library>/region.json
"""
import argparse
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from . import config, flight as F
from .scenario import Scenario, Target, scenario
from .util import save_json

NOMINAL = Scenario('L0', 0, 0., 0., 1., 1.)        # no offset, no disturbance


def plan_command(plan, phases, morph):
    def command(env):
        k = 0 if env.t < phases[0] else (1 if env.t < phases[1] else 2)
        w = env.w
        p = w['prot']
        if 'cl' in plan:            # v3: absolute CL target per phase (fed back on gamma), reaches the high-CL flights
            cl = plan['cl'][k]-env.kg*env.x[2]
        else:
            cl = F.law_cl(w['model'], env.x, env.q(), plan['r'][k], w['qmax'], p['kp'], p['kg'])
        return cl, (plan['w'][k] if morph else 0.)
    return command


def fly_plan(sc, plan, morph=True, target=None):
    cfg = config.load()
    env = F.TargetFlight(sc, target or Target('none', 0., 0.), morph)
    env.advance(env.w['T'], plan_command(plan, cfg['reach_grid']['phases_s'], morph))
    return env


SPEED_STEP = 25.


def _job(args):
    plan, morph = args
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    cfg = config.load()
    if cfg['mission'].get('end') == 'speed':
        # v3: fly (plant, nominal scenario) down to the lowest speed of interest; every crossing of a speed level v before a
        # limit violation is a reachable end point (range at v, v) of the task "end when the speed reaches v"
        v_lo = float(cfg['targets']['v2']['grid']['V'][0])-SPEED_STEP
        env = F.TargetFlight(NOMINAL, Target('none', 0., v_lo), morph, trace=True)
        env.advance(env.w['T'], plan_command(plan, cfg['reach_grid']['phases_s'], morph))
        tr = np.asarray(env.trace)
        v, r = np.r_[env.w['x0'][1], tr[:, 2]], np.r_[0., tr[:, 4]]
        if env.failed:                                   # drop the step that violated
            v, r = v[:-1], r[:-1]
        levels = np.arange(v_lo, v.max(), SPEED_STEP)
        pts = []
        for lv in levels:
            k = np.nonzero((v[:-1] > lv) & (v[1:] <= lv))[0]
            if len(k):
                i = k[0]
                u = (v[i]-lv)/(v[i]-v[i+1])
                pts.append([float(r[i]+u*(r[i+1]-r[i])), float(lv)])
        return dict(plan=plan, morph=morph, feasible=bool(pts), points=pts, reason=env.reason)
    env = fly_plan(NOMINAL, plan, morph)
    return dict(plan=plan, morph=morph, feasible=not env.failed, range_km=env.range_km(), V_end=env.x[1],
                reason=env.reason)


def grid(morph):
    g = config.load()['reach_grid']
    plans = [dict(r=list(r)) for r in itertools.product(g['r'], repeat=3)]
    plans += [dict(cl=list(c)) for c in itertools.product(g.get('cl', []), repeat=3)]     # v3
    if not morph:
        return [dict(p, w=[0., 0., 0.]) for p in plans]
    return [dict(p, w=list(w)) for p in plans for w in itertools.product(g['w'], repeat=3)]


def _poly_centroid(P):
    x, y = P[:, 0], P[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    c = x*y1-x1*y
    A = c.sum()/2
    return np.array([((x+x1)*c).sum()/(6*A), ((y+y1)*c).sum()/(6*A)])


def hull(points):
    P = np.asarray(points, float)
    h = ConvexHull(P)
    return P[h.vertices]


def shrink(V, f):
    c = _poly_centroid(V)
    return c+(V-c)*(1-f)


def inside(V, p):
    """Point in convex polygon (counter-clockwise vertices from ConvexHull)."""
    x, y = p
    n = len(V)
    for i in range(n):
        (x1, y1), (x2, y2) = V[i], V[(i+1) % n]
        if (x2-x1)*(y-y1)-(y2-y1)*(x-x1) < -1e-9:
            return False
    return True


def chord(V, D):
    """(V_low, V_high) of the convex polygon at range D, or None."""
    ys = []
    n = len(V)
    for i in range(n):
        (x1, y1), (x2, y2) = V[i], V[(i+1) % n]
        if (x1-D)*(x2-D) <= 0 and x1 != x2:
            ys.append(y1+(D-x1)*(y2-y1)/(x2-x1))
    return (min(ys), max(ys)) if ys else None


def build(rows, cfg=None):
    cfg = cfg or config.load()
    tg = cfg['targets']
    if rows and 'points' in rows[0]:          # v3: reachable (range at v, v) points of every plan, v in the target speed band
        vg = tg['v2']['grid']['V']
        band = lambda p: vg[0]-SPEED_STEP-1e-9 <= p[1] <= vg[1]+SPEED_STEP+1e-9
        pm = [tuple(p) for r in rows if r['morph'] for p in r['points'] if band(p)]
        pf = [tuple(p) for r in rows if not r['morph'] for p in r['points'] if band(p)]
    else:
        pm = [(r['range_km'], r['V_end']) for r in rows if r['morph'] and r['feasible']]
        pf = [(r['range_km'], r['V_end']) for r in rows if not r['morph'] and r['feasible']]
    Hm, Hf = hull(pm), hull(pf)
    R = shrink(Hm, tg['region_shrink'])
    lo, hi = Hf[:, 0].min(), Hf[:, 0].max()
    targets, notes = [], []
    for k, f in enumerate(tg['fixed_fractions']):
        D = float(lo+f*(hi-lo))
        cf, cm = chord(Hf, D), chord(Hm, D)
        targets.append(dict(name=f'T{k+1}', range_km=D, speed_mps=float(.5*(cf[0]+cf[1])), kind='fixed_reachable'))
        band = cf[0]-cm[0]
        targets.append(dict(name=f'T{k+4}', range_km=D, speed_mps=float(.5*(cf[0]+cm[0])), kind='morph_only',
                            band_mps=float(band)))
        if band < 4*cfg['hit']['speed_mps']:
            notes.append(f'T{k+4}: morph-only band at {D:.0f} km is only {band:.0f} m/s wide')
    targets.sort(key=lambda t: int(t['name'][1:]))
    for t in targets:
        if not inside(R, (t['range_km'], t['speed_mps'])):
            notes.append(f"{t['name']} lies outside the shrunk target region")
    return dict(morph_hull=Hm.tolist(), fixed_hull=Hf.tolist(), region=R.tolist(), targets=targets, notes=notes,
                area_ratio=float(ConvexHull(Hm).volume/ConvexHull(Hf).volume),
                feasible=dict(morph=len(pm), fixed=len(pf)), fixed_range=[float(lo), float(hi)])


_LOADED = {}


def load_region(path=None):
    p = Path(path or config.region_path())
    if str(p) not in _LOADED:
        if not p.exists():
            raise FileNotFoundError(f'{p} missing: run python -m target_rl.region first')
        _LOADED[str(p)] = json.loads(p.read_text())
    return _LOADED[str(p)]


def named_targets(region=None):
    region = region or load_region()
    return [Target(t['name'], t['range_km'], t['speed_mps']) for t in region['targets']]


def random_target(seed, name, region=None, cfg=None):
    cfg = cfg or config.load()
    region = region or load_region()
    R = np.asarray(region['region'])
    ex = cfg['targets']['exclusion']
    fixed = [(t['range_km'], t['speed_mps']) for t in region['targets']]
    rng = np.random.default_rng([int(seed), 11])
    (x0, y0), (x1, y1) = R.min(0), R.max(0)
    for _ in range(100000):
        p = (float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1)))
        if inside(R, p) and not any(abs(p[0]-a) <= ex[0] and abs(p[1]-b) <= ex[1] for a, b in fixed):
            return Target(name, round(p[0], 3), round(p[1], 3))
    raise RuntimeError('no target found in the region')


# ------------------------------------------------------------------------------------------- v2 target distribution
def _seg_dist(p, a, b):
    ab = b-a
    t = np.clip(np.dot(p-a, ab)/max(np.dot(ab, ab), 1e-12), 0., 1.)
    return float(np.linalg.norm(p-(a+t*ab)))


def edge_depth(p, region=None, cfg=None):
    """Distance of the target p = (range km, speed m/s) to the edge of the target region, in normalised units
    (range / targets.v2.scale[0] km, speed / targets.v2.scale[1] m/s)."""
    cfg = cfg or config.load()
    region = region or load_region()
    sc = np.asarray(cfg['targets']['v2']['scale'], float)
    q = np.asarray(p, float)/sc
    P = np.asarray(region['region'], float)/sc
    return min(_seg_dist(q, P[i], P[(i+1) % len(P)]) for i in range(len(P)))


def v2_target(seed, stratum, name, k=0, region=None, cfg=None):
    """v2 random target: inside the target region, at least targets.v2.margin from its edge, and in stratum 'F' (also reachable
    with fixed geometry in the nominal fixed-geometry region) or 'M' (only the morphing region contains it)."""
    cfg = cfg or config.load()
    region = region or load_region()
    R, FX = np.asarray(region['region']), np.asarray(region['fixed_hull'])
    margin = float(cfg['targets']['v2']['margin'])
    rng = np.random.default_rng([int(seed), 23, int(stratum == 'F'), int(k)])
    (x0, y0), (x1, y1) = R.min(0), R.max(0)
    for _ in range(200000):
        p = (float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1)))
        if inside(R, p) and inside(FX, p) == (stratum == 'F') and edge_depth(p, region, cfg) >= margin:
            return Target(name, round(p[0], 3), round(p[1], 3))
    raise RuntimeError('no v2 target found')


def domain_grid(region=None, cfg=None):
    """Grid points (range, speed) of the task-domain map inside the nominal morphing-reachable hull."""
    cfg = cfg or config.load()
    region = region or load_region()
    g = cfg['targets']['v2']['grid']
    MH = np.asarray(region['morph_hull'])
    return [(float(d), float(v)) for d in np.arange(g['D'][0], g['D'][1]+1e-9, g['D'][2])
            for v in np.arange(g['V'][0], g['V'][1]+1e-9, g['V'][2]) if inside(MH, (float(d), float(v)))]


def v2_cases(which, region=None, cfg=None):
    """v2 case sets (scenario, target).
    train       one random target per training scenario, strata F / M alternating (300 cases)
    validation  two random targets (one per stratum) per validation scenario (decision set)
    test        test scenarios x (targets.v2.test_per_stratum random targets per stratum + T1-T6); run once
    domain      domain-map scenarios x grid points"""
    cfg = cfg or config.load()
    region = region or load_region()
    lvl = cfg['disturbance']['main_level']
    if which in ('train', 'accept'):          # accept (v3b): held-out acceptance cases of the training, seeds.acceptance
        key, pre = ('train', 'train') if which == 'train' else ('acceptance', 'acc')
        return [(scenario(lvl, s), v2_target(s, 'F' if s % 2 == 0 else 'M', f'{pre}{s}', 0, region, cfg))
                for s in config.seed_range(cfg['seeds'][key])]
    if which == 'validation':
        return [(scenario(lvl, s), v2_target(s, st, f'{st}{s}', 0, region, cfg))
                for s in config.seed_range(cfg['seeds']['validation']) for st in ('F', 'M')]
    if which == 'test':
        n = int(cfg['targets']['v2']['test_per_stratum'])
        return [(scenario(lvl, s), t) for s in config.seed_range(cfg['seeds']['test'])
                for t in [v2_target(s, st, f'{st}{s}_{k}', k, region, cfg) for st in ('F', 'M') for k in range(n)]+named_targets(region)]
    if which == 'domain':
        grid = domain_grid(region, cfg)
        return [(scenario(lvl, s), Target(f'g{int(d)}_{int(v)}', d, v)) for s in config.seed_range(cfg['seeds']['domain']) for d, v in grid]
    raise ValueError(which)


def training_cases(region=None, cfg=None):
    cfg = cfg or config.load()
    region = region or load_region()
    if cfg.get('targets', {}).get('v2'):
        return v2_cases('train', region, cfg)
    lvl = cfg['disturbance']['main_level']
    return [(scenario(lvl, s), random_target(s, f'train{s}', region, cfg)) for s in config.seed_range(cfg['seeds']['train'])]


def test_targets(region=None, cfg=None):
    cfg = cfg or config.load()
    region = region or load_region()
    rnd = [random_target(s, f'R{k+1}', region, cfg) for k, s in enumerate(config.seed_range(cfg['seeds']['random_targets']))]
    return named_targets(region)+rnd


def test_cases(region=None, cfg=None, level=None, which='test'):
    """which = 'test': test scenarios x (T1-T6 + random targets); 'dev': pilot scenarios x T1-T6 (never tested)."""
    cfg = cfg or config.load()
    lvl = level or cfg['disturbance']['main_level']
    if which == 'dev':
        return [(scenario(lvl, s), t) for s in config.seed_range(cfg['seeds']['dev']) for t in named_targets(region)]
    tg = test_targets(region, cfg)
    return [(scenario(lvl, s), t) for s in config.seed_range(cfg['seeds']['test']) for t in tg]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--workers', type=int, default=os.cpu_count() or 1)
    p.add_argument('--out', default=None, help='default runs/<library>/region.json')
    a = p.parse_args(argv)
    jobs = [(g, False) for g in grid(False)]+[(g, True) for g in grid(True)]
    with mp.get_context('spawn').Pool(a.workers) as pool:
        rows = pool.map(_job, jobs, chunksize=32)
    out = build(rows)
    out['rows'] = [{k: v for k, v in r.items() if k != 'points'} for r in rows]   # v3: the points are not stored (size)
    out['library'] = config.library()
    save_json(a.out or config.region_path(), out)
    print(json.dumps({k: out[k] for k in ('targets', 'notes', 'area_ratio', 'feasible', 'fixed_range')}, indent=1))


if __name__ == '__main__':
    main()

"""Attribution report of experiments/improve: per-arm table, declared paired comparisons, bound gap.

    python scripts/analysis/improve_report.py runs/aoa/improve --spec experiments/improve/comparisons.json \
        --bounds experiments/improve/bounds --out runs/aoa/improve_report

Reads <dir>/<policy>/<state>/<arm>/summary.json written by aoa_gate_experiment.py. An arm is an execution
layer, i.e. objective[@safety variant] such as v16, consistent, consistent@minimal_first or
aligned@minimal_first. A policy is const0.025, seed<N> or a trained run name. The spec lists the
declared comparisons: [{"name", "a": [arm, policy], "b": [arm, policy], "kind": "execution"|"rl"|...}].
Each comparison reports the paired difference a − b over the states both flew: mean, 95 % paired
bootstrap CI, wins, worst, completion and integrated heat load. With --bounds it also reports the
per-state gap to the relaxed physical upper bound (src/certified_flight/bound.py) and the gap closure
(a − b)/(UB − b).
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from rollout_improvement.evaluation import paired_bootstrap_ci  # noqa: E402


def load(root):
    rows = {}
    for path in Path(root).glob('*/*/*/summary.json'):
        arm, state, policy = path.parts[-2], path.parts[-3], path.parts[-4]
        s = json.loads(path.read_text(encoding='utf-8'))
        m = s.get('metrics', {})
        rows[(arm, policy, state)] = dict(range_km=s['range_km'], completed=bool(s.get('completed')),
                                          heat_load=m.get('heat_load_MJ_m2'), ptp=m.get('intra_step_ptp_median_deg'),
                                          q_cruise=m.get('q_dyn_cruise_mean_kPa'), channels=s.get('channels', {}),
                                          lookahead_offsets=s.get('lookahead_offset_steps'))
    return rows


def bounds(path):
    out = {}
    for f in Path(path).glob('dh*_dv*.json'):
        if '_' in f.stem.split('_dv')[1]:
            continue                                  # geometry-restricted bounds are reported separately
        r = json.loads(f.read_text())
        out[f.stem] = r['relaxed']['range_km']
    return out


def describe(x):
    x = np.asarray(x, float)
    if not len(x):
        return dict(n=0)
    return dict(n=int(len(x)), mean=float(x.mean()), ci95=paired_bootstrap_ci(x), wins=int((x > 0).sum()),
                worst=float(x.min()), median=float(np.median(x)))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('root')
    p.add_argument('--spec', required=True)
    p.add_argument('--bounds')
    p.add_argument('--out', required=True)
    a = p.parse_args(argv)
    rows = load(a.root)
    ub = bounds(a.bounds) if a.bounds else {}
    spec = json.loads(Path(a.spec).read_text(encoding='utf-8'))
    arms = sorted({(k[0], k[1]) for k in rows})
    md = ['# Improvement attribution report', '', '## Arms', '',
          '| arm | policy | states | completed | mean range km | mean heat load MJ/m² | median intra-step ptp ° | '
          'mean cruise q kPa | non-surrogate decisions | mean gap to bound km |', '|---|---|---|---|---|---|---|---|---|---|']
    report = dict(arms={}, comparisons=[])
    for arm, pol in arms:
        r = {s: v for (x, y, s), v in rows.items() if (x, y) == (arm, pol)}
        gaps = [ub[s]-v['range_km'] for s, v in r.items() if s in ub]
        e = dict(states=len(r), completed=sum(v['completed'] for v in r.values()),
                 range_km=float(np.mean([v['range_km'] for v in r.values()])),
                 heat_load=float(np.mean([v['heat_load'] for v in r.values() if v['heat_load'] is not None] or [np.nan])),
                 ptp=float(np.median([v['ptp'] for v in r.values() if v['ptp'] is not None] or [np.nan])),
                 q_cruise=float(np.mean([v['q_cruise'] for v in r.values() if v['q_cruise'] is not None] or [np.nan])),
                 fallback=int(sum(n for v in r.values() for c, n in v['channels'].items() if c != 'FAST_APPROXIMATE')),
                 gap_to_bound_km=float(np.mean(gaps)) if gaps else None, gap_states=len(gaps))
        report['arms'][f'{arm}|{pol}'] = e
        md.append(f"| {arm} | {pol} | {e['states']} | {e['completed']} | {e['range_km']:.2f} | {e['heat_load']:.1f} | "
                  f"{e['ptp']:.3f} | {e['q_cruise']:.1f} | {e['fallback']} | "
                  + (f"{e['gap_to_bound_km']:.2f} ({e['gap_states']})" if gaps else '-') + ' |')
    md += ['', '## Declared comparisons (paired by initial state; a − b)', '',
           '| comparison | kind | states | mean km [95 % CI] | wins | worst km | completed a/b | Δ heat load MJ/m² | '
           'gap closure |', '|---|---|---|---|---|---|---|---|---|']
    for c in spec['comparisons']:
        A, B = tuple(c['a']), tuple(c['b'])
        states = sorted({s for (x, y, s) in rows if (x, y) == A} & {s for (x, y, s) in rows if (x, y) == B})
        d = describe([rows[A+(s,)]['range_km']-rows[B+(s,)]['range_km'] for s in states])
        dh = [rows[A+(s,)]['heat_load']-rows[B+(s,)]['heat_load'] for s in states
              if rows[A+(s,)]['heat_load'] is not None and rows[B+(s,)]['heat_load'] is not None]
        closure = [(rows[A+(s,)]['range_km']-rows[B+(s,)]['range_km'])/(ub[s]-rows[B+(s,)]['range_km'])
                   for s in states if s in ub and ub[s] > rows[B+(s,)]['range_km']]
        e = dict(c, states=len(states), difference_km=d,
                 completed=[sum(rows[A+(s,)]['completed'] for s in states), sum(rows[B+(s,)]['completed'] for s in states)],
                 heat_load_delta=float(np.mean(dh)) if dh else None,
                 gap_closure=float(np.mean(closure)) if closure else None, gap_closure_states=len(closure))
        if 'criterion' in c and d.get('n'):
            e['criterion_met'] = bool(d['ci95'][0] > 0) if c['criterion'] == 'ci_lower_gt_0' else None
        report['comparisons'].append(e)
        g = (f"{d['mean']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]" if d.get('n') else '-')
        md.append(f"| {c['name']} | {c.get('kind', '')} | {len(states)} | {g} | {d.get('wins', '-')} | "
                  f"{d.get('worst', float('nan')):+.2f} | {e['completed'][0]}/{e['completed'][1]} | "
                  + (f"{e['heat_load_delta']:+.1f}" if dh else '-') + ' | '
                  + (f"{e['gap_closure']:.2f} ({len(closure)})" if closure else '-') + ' |')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out/'report.json').write_text(json.dumps(report, indent=1)+'\n', encoding='utf-8')
    (out/'report.md').write_text('\n'.join(md)+'\n', encoding='utf-8')
    print('\n'.join(md))


if __name__ == '__main__':
    main()

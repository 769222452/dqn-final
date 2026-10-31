"""Tables and pre-declared verdicts for the AoA-sawtooth experiments.

Reads <out>/<stage>/<policy>/<state>/<objective>/summary.json (written by aoa_gate_experiment.py)
and writes <out>/report.md and <out>/report.json. Criteria (experiments/aoa_fix/README.md):

Gate (nominal state, per policy):
  G1 sawtooth removed: consistent 1/3 Hz power fraction <= 0.25 x V16, boundary-jump/drift opposition
     <= 0.65 (0.5 = no sawtooth) and median intra-step peak-to-peak <= 0.5 x V16
  G2 feasibility: consistent completes the 1200 s mission
  G3 range kept: |range(consistent) - range(V16)| <= 0.5 % of range(V16)
Held-out-3 (41 states, paired by state):
  H1 range kept: per policy, consistent - V16 range has mean >= -0.5 % of the V16 mean range
     and the 95 % CI upper bound >= 0
  H2 learned gain kept: per seed, gain over constant CL 0.025 under the consistent objective has
     CI lower bound > 0
  H3 sawtooth removed on every completed held-out flight (G1 thresholds per state)
Failed flights count with their truncated range, as in V16.
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from rollout_improvement.evaluation import paired_bootstrap_ci  # noqa: E402

KEYS = ('completed', 'range_km', 'reversal_rate', 'intra_step_ptp_median_deg', 'boundary_jump_median_deg',
        'sawtooth_opposition', 'power_3s_fraction', 'hf_rms_deg', 'CL_mean_abs_error_median',
        'lookahead_offset_steps', 'fallback_steps')


def load(out):
    rows = {}
    for path in Path(out).glob('*/*/*/*/summary.json'):
        objective, state, policy, stage = path.parts[-2], path.parts[-3], path.parts[-4], path.parts[-5]
        s = json.loads(path.read_text(encoding='utf-8'))
        m = s.get('metrics', {})
        row = {k: m[k] for k in KEYS if k in m}
        # The run summary (not the metrics block) is authoritative for outcome and range.
        row.update(completed=bool(s.get('completed', s.get('stop') == 'SUCCESS' and s.get('t_end_s') == 1200.)),
                   range_km=s['range_km'], stop=s['stop'], lookahead_offset_steps=s.get('lookahead_offset_steps'),
                   fallback_steps=s.get('fallback_steps'))
        rows[(stage, policy, state, objective)] = row
    return rows


def sawtooth_removed(c, v):
    try:
        opposition = c['sawtooth_opposition']
        # None: no step with both jump and drift above the noise floor, i.e. no sawtooth at all.
        return bool(c['power_3s_fraction'] <= .25*v['power_3s_fraction']
                    and (opposition is None or opposition <= .65)
                    and c['intra_step_ptp_median_deg'] <= .5*v['intra_step_ptp_median_deg'])
    except (KeyError, TypeError):
        return False


def describe(x):
    x = np.asarray(x, float)
    if not len(x):
        return dict(n=0)
    return dict(n=len(x), mean=float(x.mean()), ci95=paired_bootstrap_ci(x), wins=int((x > 0).sum()),
                worst=float(x.min()), median=float(np.median(x)))


def fmt(v, d=3):
    return '-' if v is None else (f'{v:.{d}f}' if isinstance(v, float) else str(v))


def main(out):
    out = Path(out)
    rows = load(out)
    report, md = dict(gate={}, heldout={}), ['# AoA-sawtooth experiment report', '']
    gate = {k: v for k, v in rows.items() if k[0] == 'gate'}
    if gate:
        md += ['## Gate (nominal state)', '', '| policy | objective | ' + ' | '.join(KEYS) + ' |',
               '|' + '---|'*(len(KEYS)+2)]
        for (stage, policy, state, objective), r in sorted(gate.items()):
            md.append(f'| {policy} | {objective} | ' + ' | '.join(fmt(r.get(k)) for k in KEYS) + ' |')
        md.append('')
        for policy in sorted({k[1] for k in gate}):
            v, c = gate.get(('gate', policy, 'dh0_dv0', 'v16')), gate.get(('gate', policy, 'dh0_dv0', 'consistent'))
            if not (v and c):
                continue
            verdict = dict(G1_sawtooth_removed=sawtooth_removed(c, v), G2_feasible=c['completed'],
                           G3_range_kept=abs(c['range_km']-v['range_km']) <= .005*v['range_km'],
                           range_delta_km=c['range_km']-v['range_km'],
                           reversal_rate_below_5pct=bool((c.get('reversal_rate') or 1) < .05))
            verdict['PASS'] = verdict['G1_sawtooth_removed'] and verdict['G2_feasible'] and verdict['G3_range_kept']
            report['gate'][policy] = verdict
            md.append(f"- **{policy}**: " + ', '.join(f'{k}={fmt(x)}' for k, x in verdict.items()))
        md.append('')
    held = {k: v for k, v in rows.items() if k[0] == 'heldout'}
    if held:
        md += ['## Held-out-3 (paired by initial state)', '']
        states = sorted({k[2] for k in held})
        policies = sorted({k[1] for k in held})
        for policy in policies:
            pairs = [(held[('heldout', policy, s, 'v16')], held[('heldout', policy, s, 'consistent')]) for s in states
                     if ('heldout', policy, s, 'v16') in held and ('heldout', policy, s, 'consistent') in held]
            if not pairs:
                continue
            diff = [c['range_km']-v['range_km'] for v, c in pairs]
            mean_v16 = float(np.mean([v['range_km'] for v, _ in pairs]))
            d = describe(diff)
            h1 = bool(d['n'] and d['mean'] >= -.005*mean_v16 and d['ci95'][1] >= 0)
            removed = [sawtooth_removed(c, v) for v, c in pairs if c['completed'] and v['completed']]
            entry = dict(states=len(pairs), range_consistent_minus_v16_km=d, H1_range_kept=h1,
                         completed_v16=sum(v['completed'] for v, _ in pairs),
                         completed_consistent=sum(c['completed'] for _, c in pairs),
                         H3_sawtooth_removed_share=float(np.mean(removed)) if removed else None,
                         median_power_3s=dict(v16=float(np.median([v['power_3s_fraction'] for v, _ in pairs])),
                                              consistent=float(np.median([c['power_3s_fraction'] for _, c in pairs]))))
            report['heldout'][policy] = entry
        for policy in [p for p in policies if p != 'const0.025']:
            gains = {}
            for objective in ('v16', 'consistent'):
                g = [held[('heldout', policy, s, objective)]['range_km']-held[('heldout', 'const0.025', s, objective)]['range_km']
                     for s in states if s != 'dh0_dv0' and ('heldout', policy, s, objective) in held
                     and ('heldout', 'const0.025', s, objective) in held]
                gains[objective] = describe(g)
            if policy in report['heldout']:
                c = gains['consistent']
                report['heldout'][policy].update(gain_over_const_v16=gains['v16'], gain_over_const_consistent=c,
                                                 H2_gain_kept=bool(c['n'] and c['ci95'][0] > 0))
        md += ['| policy | states | done v16/cons | consistent−V16 km (mean [CI]) | H1 | gain V16 obj | gain consistent | H2 | sawtooth removed | 1/3 Hz power v16→cons |',
               '|---|---|---|---|---|---|---|---|---|---|']
        for policy, e in report['heldout'].items():
            d = e['range_consistent_minus_v16_km']
            gv, gc = e.get('gain_over_const_v16', {}), e.get('gain_over_const_consistent', {})
            g = lambda x: f"{x['mean']:+.2f} [{x['ci95'][0]:+.2f}, {x['ci95'][1]:+.2f}]" if x.get('n') else '-'
            md.append(f"| {policy} | {e['states']} | {e['completed_v16']}/{e['completed_consistent']} | {g(d)} | {e['H1_range_kept']} | "
                      f"{g(gv)} | {g(gc)} | {e.get('H2_gain_kept', '-')} | {fmt(e['H3_sawtooth_removed_share'], 2)} | "
                      f"{e['median_power_3s']['v16']:.3f}→{e['median_power_3s']['consistent']:.3f} |")
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    (out/'report.md').write_text('\n'.join(md)+'\n', encoding='utf-8')
    print('\n'.join(md))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'runs/aoa')

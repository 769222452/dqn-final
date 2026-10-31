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

Stages: gate and fidelity fly the true OCP every decision; heldout flies the production surrogate
pipeline (V16 surrogate for v16, the refitted surrogate for consistent); heldout-ocp (optional) is the
held-out stage with the true OCP. Also reported:
  archive reproduction  heldout/v16 ranges vs the archived V16 held-out-3 evaluation
                        (experiments/aoa_fix/v16_heldout3_archive.json; identical code path, so ~0 km)
  fidelity              surrogate (heldout) minus true-OCP teacher (fidelity) on the same flights:
                        range and sawtooth metrics, per objective
  channels              decisions flown by the surrogate / physics baseline / true OCP fallback
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
                   fallback_steps=s.get('fallback_steps'), channels=s.get('channels', {}), inner=s.get('inner', 'ocp'))
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


# Amendment 2 (declared after the local nominal surrogate flights, before any held-out flight; README
# section 3): the opposition share is scale-free, so a residual of a few hundredths of a degree (below
# the refitted surrogate's 0.06 deg test RMSE) still reads as "sawtooth". H3a reports the amplitude
# reading next to the declared H3: opposition is only required when the median boundary jump exceeds
# RESIDUAL_FLOOR_DEG. H3 as declared is kept and reported.
RESIDUAL_FLOOR_DEG = .05


def sawtooth_removed_amplitude(c, v):
    try:
        if c['boundary_jump_median_deg'] > RESIDUAL_FLOOR_DEG:
            return sawtooth_removed(c, v)
        return bool(c['power_3s_fraction'] <= .25*v['power_3s_fraction']
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
            # Deviation (documented in experiments/aoa_fix/README.md): G3 was written two-sided, so a
            # range GAIN above 0.5 % fails it. The intended no-loss reading is reported separately and
            # is what decides whether the held-out stage runs; the declared verdict above is kept.
            verdict['G3_no_loss'] = c['range_km'] >= .995*v['range_km']
            verdict['PROCEED_TO_HELDOUT'] = verdict['G1_sawtooth_removed'] and verdict['G2_feasible'] and verdict['G3_no_loss']
            report['gate'][policy] = verdict
            md.append(f"- **{policy}**: " + ', '.join(f'{k}={fmt(x)}' for k, x in verdict.items()))
        md.append('')
    for stage in ('heldout', 'heldout-ocp'):
        md += heldout_section(rows, stage, report)
    md += retrained_section(rows, report)
    md += reproduction_section(rows, report)
    md += label_section(out, report)
    md += fidelity_section(rows, report)
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    (out/'report.md').write_text('\n'.join(md)+'\n', encoding='utf-8')
    print('\n'.join(md))


def heldout_section(rows, stage, report):
    md = []
    held = {('heldout',)+k[1:]: v for k, v in rows.items() if k[0] == stage}
    report[stage] = {}
    if held:
        inner = 'surrogate pipeline' if stage == 'heldout' else 'true OCP'
        md += [f'## Held-out-3 [{inner}] (paired by initial state)', '']
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
            done = [(v, c) for v, c in pairs if c['completed'] and v['completed']]
            removed = [sawtooth_removed(c, v) for v, c in done]
            removed_a = [sawtooth_removed_amplitude(c, v) for v, c in done]
            entry = dict(states=len(pairs), range_consistent_minus_v16_km=d, H1_range_kept=h1,
                         completed_v16=sum(v['completed'] for v, _ in pairs),
                         completed_consistent=sum(c['completed'] for _, c in pairs),
                         H3_sawtooth_removed_share=float(np.mean(removed)) if removed else None,
                         H3a_amplitude_removed_share=float(np.mean(removed_a)) if removed_a else None,
                         median_intra_ptp_deg=dict(v16=float(np.median([v['intra_step_ptp_median_deg'] for v, _ in pairs])),
                                                   consistent=float(np.median([c['intra_step_ptp_median_deg'] for _, c in pairs]))),
                         median_power_3s=dict(v16=float(np.median([v['power_3s_fraction'] for v, _ in pairs])),
                                              consistent=float(np.median([c['power_3s_fraction'] for _, c in pairs]))))
            report[stage][policy] = entry
        for policy in [p for p in policies if p != 'const0.025']:
            gains = {}
            for objective in ('v16', 'consistent'):
                g = [held[('heldout', policy, s, objective)]['range_km']-held[('heldout', 'const0.025', s, objective)]['range_km']
                     for s in states if s != 'dh0_dv0' and ('heldout', policy, s, objective) in held
                     and ('heldout', 'const0.025', s, objective) in held]
                gains[objective] = describe(g)
            if policy in report[stage]:
                c = gains['consistent']
                report[stage][policy].update(gain_over_const_v16=gains['v16'], gain_over_const_consistent=c,
                                                 H2_gain_kept=bool(c['n'] and c['ci95'][0] > 0))
        md += ['| policy | states | done v16/cons | consistent−V16 km (mean [CI]) | H1 | gain V16 obj | gain consistent | H2 | H3 (declared) | H3a (amplitude) | intra-ptp v16→cons deg | 1/3 Hz power v16→cons |',
               '|---|---|---|---|---|---|---|---|---|---|---|---|']
        for policy, e in report[stage].items():
            d = e['range_consistent_minus_v16_km']
            gv, gc = e.get('gain_over_const_v16', {}), e.get('gain_over_const_consistent', {})
            g = lambda x: f"{x['mean']:+.2f} [{x['ci95'][0]:+.2f}, {x['ci95'][1]:+.2f}]" if x.get('n') else '-'
            md.append(f"| {policy} | {e['states']} | {e['completed_v16']}/{e['completed_consistent']} | {g(d)} | {e['H1_range_kept']} | "
                      f"{g(gv)} | {g(gc)} | {e.get('H2_gain_kept', '-')} | {fmt(e['H3_sawtooth_removed_share'], 2)} | "
                      f"{fmt(e['H3a_amplitude_removed_share'], 2)} | "
                      f"{e['median_intra_ptp_deg']['v16']:.3f}→{e['median_intra_ptp_deg']['consistent']:.3f} | "
                      f"{e['median_power_3s']['v16']:.3f}→{e['median_power_3s']['consistent']:.3f} |")
        md.append('')
        ch = {}
        for k, v in held.items():
            for name, n in (v.get('channels') or {}).items():
                ch.setdefault(k[3], {}).setdefault(name, 0)
                ch[k[3]][name] += n
        report[stage+'_channels'] = ch
        md += ['Decisions per channel: '+'; '.join(f'{o}: '+', '.join(f'{n}={c}' for n, c in sorted(v.items()))
                                                   for o, v in sorted(ch.items())), '']
    return md


def retrained_section(rows, report):
    """H2r (optional retrain stage): retrained policy gain over constant CL 0.025, consistent objective."""
    md, out = [], {}
    for (stage, policy, state, objective), r in rows.items():
        if stage == 'retrained':
            out.setdefault(policy.removeprefix('retrained_'), {})[state] = r
    if not out:
        return md
    report['retrained'] = {}
    md += ['## Retrained policies (consistent objective, surrogate pipeline, held-out-3)', '',
           '| seed | states | done | gain over const (consistent) | H2r | retrained − V16 arm (same seed) |',
           '|---|---|---|---|---|---|']
    g = lambda x: f"{x['mean']:+.2f} [{x['ci95'][0]:+.2f}, {x['ci95'][1]:+.2f}]" if x.get('n') else '-'
    for seed, states in sorted(out.items()):
        gain = describe([r['range_km']-rows[('heldout', 'const0.025', s, 'consistent')]['range_km']
                         for s, r in states.items() if s != 'dh0_dv0' and ('heldout', 'const0.025', s, 'consistent') in rows])
        vs = describe([r['range_km']-rows[('heldout', seed, s, 'v16')]['range_km']
                       for s, r in states.items() if ('heldout', seed, s, 'v16') in rows])
        e = dict(states=len(states), completed=sum(r['completed'] for r in states.values()), gain_over_const=gain,
                 H2r_gain_kept=bool(gain.get('n') and gain['ci95'][0] > 0), minus_v16_arm_km=vs)
        report['retrained'][seed] = e
        md.append(f"| {seed} | {e['states']} | {e['completed']} | {g(gain)} | {e['H2r_gain_kept']} | {g(vs)} |")
    return md + ['']


def label_section(out, report):
    """Cross-machine determinism of the teacher labels: this run vs the committed local labels."""
    import gzip
    md = []
    local_dir = ROOT/'experiments/aoa_fix/surrogate_local'
    for name in ('labels', 'dagger_labels'):
        mine, ref = Path(out)/'surrogate'/f'{name}.jsonl', local_dir/f'{name}.jsonl.gz'
        if not (mine.exists() and ref.exists()):
            continue
        a = {r['id']: r for r in map(json.loads, mine.read_text(encoding='utf-8').splitlines())}
        b = {r['id']: r for r in map(json.loads, gzip.decompress(ref.read_bytes()).decode().splitlines())}
        common = sorted(set(a) & set(b))
        both = [k for k in common if a[k]['ok'] and b[k]['ok']]
        da = [float(np.max(np.abs(np.rad2deg(np.subtract(a[k]['post_alpha'], b[k]['post_alpha']))))) for k in both]
        dm = [abs(a[k]['pre_mach']-b[k]['pre_mach']) for k in both]
        e = dict(common=len(common), only_here=len(set(a)-set(b)), only_local=len(set(b)-set(a)),
                 ok_mismatch=sum(a[k]['ok'] != b[k]['ok'] for k in common),
                 max_alpha_diff_deg=max(da) if da else None, max_mach_diff=max(dm) if dm else None,
                 identical=sum(x == 0 for x in da))
        report.setdefault('label_reproduction', {})[name] = e
        md += [f"Labels `{name}` vs local: {e['common']} common ids, {e['identical']} identical, "
               f"max |d alpha| {fmt(e['max_alpha_diff_deg'], 6)} deg, max |d Mach| {fmt(e['max_mach_diff'], 8)}, "
               f"ok mismatches {e['ok_mismatch']}, only here/local {e['only_here']}/{e['only_local']}", '']
    return (['## Teacher-label reproduction (this machine vs committed local labels)', ''] + md) if md else []


def reproduction_section(rows, report):
    archive = json.loads((ROOT/'experiments/aoa_fix/v16_heldout3_archive.json').read_text(encoding='utf-8'))['ranges']
    diffs = []
    for (stage, policy, state, objective), r in rows.items():
        if stage == 'heldout' and objective == 'v16' and state in archive.get(policy, {}):
            diffs.append(dict(policy=policy, state=state, delta_km=r['range_km']-archive[policy][state]['range_km']))
    if not diffs:
        return []
    d = np.array([x['delta_km'] for x in diffs])
    report['archive_reproduction'] = dict(flights=len(diffs), max_abs_delta_km=float(abs(d).max()),
                                          identical_1e6_km=int((abs(d) < 1e-6).sum()),
                                          worst=sorted(diffs, key=lambda x: -abs(x['delta_km']))[:5])
    e = report['archive_reproduction']
    return ['## Archive reproduction (heldout / v16 vs archived V16 held-out-3)', '',
            f"{e['identical_1e6_km']}/{e['flights']} flights identical to 1e-6 km; max |delta| "
            f"{e['max_abs_delta_km']:.3g} km", '']


def fidelity_section(rows, report):
    md, out = [], {}
    keys = ('range_km', 'intra_step_ptp_median_deg', 'power_3s_fraction', 'CL_mean_abs_error_median')
    for (stage, policy, state, objective), teacher in sorted(rows.items()):
        student = rows.get(('heldout', policy, state, objective))
        if stage != 'fidelity' or student is None:
            continue
        out.setdefault(objective, []).append(dict(
            policy=policy, state=state, completed=[student['completed'], teacher['completed']],
            **{k: [student.get(k), teacher.get(k)] for k in keys}))
    if not out:
        return md
    report['fidelity'] = {}
    md += ['## Fidelity: surrogate pipeline (heldout) vs true-OCP teacher (fidelity), same flights', '',
           '| objective | flights | range student−teacher km (mean, worst) | intra-ptp student/teacher deg (median) | '
           '1/3 Hz power student/teacher (median) |', '|---|---|---|---|---|']
    for objective, items in out.items():
        dr = [a-b for a, b in (x['range_km'] for x in items)]
        summary = dict(flights=len(items), range_delta_km=describe(dr),
                       intra_ptp_deg=[float(np.median([x['intra_step_ptp_median_deg'][i] for x in items])) for i in (0, 1)],
                       power_3s=[float(np.median([x['power_3s_fraction'][i] for x in items])) for i in (0, 1)],
                       items=items)
        report['fidelity'][objective] = summary
        md.append(f"| {objective} | {len(items)} | {np.mean(dr):+.2f}, {max(dr, key=abs):+.2f} | "
                  f"{summary['intra_ptp_deg'][0]:.3f} / {summary['intra_ptp_deg'][1]:.3f} | "
                  f"{summary['power_3s'][0]:.3f} / {summary['power_3s'][1]:.3f} |")
    return md + ['']


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'runs/aoa')

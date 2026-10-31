"""Isolated pilot label generation, refit, and exact transactional flight.

Old teacher labels supply contexts only, never targets. A passed current audit is
not a future witness; the continuous guard checks both. No legacy certificate,
policy training target, surrogate output, or checkpoint is accepted here.
"""
from copy import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import yaml

from certified_flight import CL_ACTIONS, integrity
from certified_flight.aero import AeroDatabase
from certified_flight.build import ROOT, runtime_config
from certified_flight.mission import Mission
from certified_flight.plant import CompiledPlant
from certified_flight.surrogate import OracleRequired
from certified_flight.transition import Transition
from continuous_tracking.experiment import clean, write_json
from continuous_tracking.plant import ContinuousPlant, prefix
from . import PROTOCOL

DEFAULT_CONFIG = ROOT/'experiments/continuous_surrogate/config.json'


def digest(value):
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def source_identity(config):
    paths = [*sorted((ROOT/'src').rglob('*.py')), *sorted((ROOT/'scripts').glob('*.py')),
             *sorted((ROOT/'configs').glob('*')), ROOT/integrity.SUMS, Path(config)]
    files = {str(p.relative_to(ROOT)): integrity.sha256(p) for p in paths if p.is_file()}
    return dict(protocol=PROTOCOL, files=files, sha256=digest(files))


def build(config=DEFAULT_CONFIG):
    from .safety import ContinuousSafety
    from .reachability import ContinuousCLReachability
    from .teacher import create
    integrity.verify(ROOT)
    cfg = json.loads(Path(config).read_text(encoding='utf-8'))
    if cfg['protocol'] != PROTOCOL:
        raise ValueError('Control protocol mismatch')
    original = runtime_config(ROOT)
    physical = original['physical']
    mission = Mission(ROOT, physical['mission'])
    db = AeroDatabase(ROOT/physical['snapshot'])
    plant = ContinuousPlant(CompiledPlant(mission, db))
    options = dict(original['safety_options'])
    options['tracking_quality'] = cfg['tracking_quality']
    safety = ContinuousSafety(plant, options)
    reachability = ContinuousCLReachability(plant)
    trajectory = yaml.safe_load((ROOT/physical['trajectory']).read_text(encoding='utf-8'))
    teacher = create(mission, db, trajectory, physical['ocp'], plant, cfg['teacher'])
    return dict(config=cfg, original=original, mission=mission, db=db, plant=plant,
                safety=safety, teacher=teacher, reachability=reachability,
                identity=source_identity(config))


def baseline(system):
    """Both baseline heads share the new continuous, state-updated predictor."""
    def predict(c, head):
        nodes = system['safety'].candidate(c)
        if head == 'post':
            return np.asarray(nodes)[1:]
        if head != 'pre':
            raise ValueError(head)
        p = system['plant']
        r = p.integrate(c['x'], nodes, np.arange(7)*.5, c['source'], c['target'],
                        c['fraction'], c['rate'], .5)
        mid = (np.asarray(c['x'])+r['x'][-1])/2
        return np.array([mid[1]/p.atmosphere.speed_of_sound(mid[0])])
    return predict


def context_from_state(state, cl):
    z = Transition(**state['transition'])
    z.settle(state['t'])
    return dict(x=list(state['x']), alpha_prev=float(state['alpha']), t=float(state['t']),
                source=z.source, target=z.target, fraction=z.fraction(state['t']),
                rate=.125 if z.active(state['t']) else 0., CL_des=float(cl), CL_raw=float(cl),
                CD_des=None, control_contract=PROTOCOL)


def transition_from_context(c):
    return Transition(c['source'], c['target'], c['t']-8*c['fraction'])


def prepare_contexts(system, source):
    """Deterministic grouped pilot split, selected before solving any new label.

    This uses old training groups for development only. Previous review has seen
    these states. The held-out groups are not claimed to be a new blind test.
    """
    rows = [json.loads(line) for line in Path(source).read_text(encoding='utf-8').splitlines()]
    pool = [r for r in rows if r['head'] == 'pre' and r['role'] == 'train']
    groups = sorted({r['group'] for r in pool})
    rng = np.random.default_rng(system['config']['training']['seed'])
    rng.shuffle(groups)
    selected = []
    # One nominal initial-state case and 23 distinct historical trajectory groups.
    m, db = system['mission'], system['db']
    c0 = dict(x=m.data['initial_state']['x_h_V_gamma_s'], alpha_prev=m.data['initial_state']['alpha_rad'],
              t=0., source=db.index['G42'], target=db.index['G42'], fraction=1., rate=0.,
              CL_des=.025, CL_raw=.025, CD_des=None, control_contract=PROTOCOL)
    selected.append(dict(context=c0, group='pilot_nominal_initial', role='train'))
    cl_values = [.025, .015, .02, .035, .04, .045, .03, .025]
    for i, group in enumerate(groups[:23]):
        candidates = sorted([r for r in pool if r['group'] == group], key=lambda r:r['context']['t'])
        target_time = (i*53) % 1180
        original = min(candidates, key=lambda r:abs(r['context']['t']-target_time))
        c = dict(original['context'])
        c.update(CL_des=cl_values[i % len(cl_values)], CL_raw=cl_values[i % len(cl_values)],
                 CD_des=None, control_contract=PROTOCOL)
        selected.append(dict(context=c, group=group, role='train' if i < 15 else 'validation' if i < 19 else 'test'))
    if len(selected) != 24:
        raise ValueError('Insufficient distinct pilot context groups')
    return selected


def generate_labels(system, source, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    contexts = prepare_contexts(system, source)
    write_json(out/'design.json', dict(protocol=PROTOCOL, source_sha256=integrity.sha256(Path(source)),
        source=str(source), contexts=contexts, note='All groups are development evidence, not a blind generalization test.'))
    records, started = [], time.perf_counter()
    cpu_start = time.process_time()
    for i, item in enumerate(contexts):
        c = item['context']
        folder = out/f'case_{i:02d}'
        folder.mkdir()
        pre, plan = system['teacher'].solve_audited(c, 'pre', output_dir=folder/'pre')
        pre.update(role=item['role'], group=item['group'], case=i)
        records.append(pre)
        if pre['accepted']:
            choice = system['db'].generate(float(pre['y'][0]), c['CL_des'], current=c['target'])
            z = transition_from_context(c)
            if choice['geometry_index'] is not None:
                z.command(choice['geometry_index'], c['t'])
            post_c = dict(c, source=z.source, target=z.target, fraction=z.fraction(c['t']),
                          rate=.125 if z.active(c['t']) else 0.)
            post, _ = system['teacher'].solve_audited(post_c, 'post', output_dir=folder/'post')
            post.update(role=item['role'], group=item['group'], case=i, generator=choice)
            records.append(post)
        write_json(out/'labels.json', records)
        print(json.dumps(dict(stage='fresh_teacher_labels', contexts_done=i+1, contexts_total=len(contexts),
                              accepted=sum(r['accepted'] for r in records), attempted=len(records),
                              wall_s=time.perf_counter()-started)), flush=True)
    write_json(out/'summary.json', dict(wall_s=time.perf_counter()-started, cpu_s=time.process_time()-cpu_start,
        accepted=dict(Counter((r['head']+'_'+r['role']) for r in records if r['accepted'])),
        refused=[dict(case=r['case'], head=r['head'], reason=r.get('reason')) for r in records if not r['accepted']]))
    return records


def fit_model(system, records, out):
    from .model import ContinuousSurrogate
    for head in ('pre','post'):
        n = sum(r['accepted'] and r['head'] == head and r['role'] == 'train' for r in records)
        if n < system['config']['training']['minimum_training_labels_per_head']:
            raise RuntimeError(f'Insufficient audited training labels for {head}: {n}')
    model = ContinuousSurrogate.fit(records, system['db'], system['mission'], baseline(system),
                                    metadata=dict(source_identity=system['identity'], pilot=True,
                                                  final_model_qualification=False))
    model.save(out)
    validation = model.evaluate(records)
    physical_checks = []
    for r in records:
        if not r['accepted'] or r['head'] != 'post':
            continue
        predicted, diag = model._predict(r['context'], 'post', False)
        audit, _ = system['plant'].audit(r['context'], np.r_[r['context']['alpha_prev'], predicted],
                                        np.arange(7)*.5, r['context']['CL_des'])
        physical_checks.append(dict(case=r['case'], role=r['role'], supported=diag['supported'],
                                    current_audit=audit))
    validation['post_current_physical_checks'] = physical_checks
    write_json(Path(out).with_name('model_validation.json'), validation)
    return model


def bind_proof(state, nodes, identity, transition, cl, endpoint, witness_hash):
    return dict(protocol=PROTOCOL, state_hash=digest(state), nodes_hash=digest(np.asarray(nodes)),
                identity_hash=identity, transition_hash=digest(vars(transition)), CL_cmd=float(cl),
                times_s=(np.arange(7)*.5).tolist(), endpoint=list(np.asarray(endpoint)),
                witness_hash=witness_hash)


def commit_audited(system, state, proposed_transition, nodes, proof, identity, cl, witness):
    """Reintegrate exactly the certified candidate; mutate no caller state."""
    if proof != bind_proof(state, nodes, identity, proposed_transition, cl,
                           proof.get('endpoint', []), digest(witness)):
        raise ValueError('Stale or incompatible continuous certificate')
    c = context_from_state(dict(state, transition=vars(proposed_transition)), cl)
    audit, arrays = system['plant'].audit(c, nodes, np.arange(7)*.5, cl)
    if not audit['valid']:
        raise ValueError('PRECOMMIT_CONTINUOUS_AUDIT_REJECTED')
    old = np.asarray(state['x'])
    x = arrays['0.5']['x'][-1].copy()
    if not np.array_equal(x, proof['endpoint']):
        raise ValueError('Certified versus committed endpoint mismatch')
    t = state['t']+3.
    z = copy(proposed_transition)
    z.settle(t)
    outcome = system['mission'].outcome(t, x)
    reward = system['mission'].reward(float(x[3]-old[3]), outcome)['reward']
    return dict(state, x=x.tolist(), alpha=float(nodes[-1]), t=t, transition=vars(z).copy(),
                reward=state['reward']+reward), audit, arrays


def run_flight(system, model, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    cpu_start = time.process_time()
    m, db, plant, safety = (system[k] for k in ('mission','db','plant','safety'))
    cfg = system['config']
    cl = cfg['flight']['CL']
    if cl not in CL_ACTIONS:
        raise ValueError('Unchanged absolute CL action required')
    state = dict(x=list(m.data['initial_state']['x_h_V_gamma_s']), alpha=m.data['initial_state']['alpha_rad'],
                 t=0., transition=vars(Transition(db.index['G42'], db.index['G42'])), reward=0., switches=0)
    identity = digest(dict(source=system['identity']['sha256'], config=cfg,
                           weights_hash=system.get('model_hash'),
                           model=getattr(model, 'metadata', {})))
    rows, reason, counts = [], 'NOT_STARTED', Counter()
    def result():
        margins = {k:min(r['margins'][k] for r in rows) for k in rows[0]['margins']} if rows else {}
        return dict(protocol=PROTOCOL, completed_1200s=reason=='SUCCESS', stop_reason=reason,
            range_km=(state['x'][3]-m.data['initial_state']['x_h_V_gamma_s'][3])/1000,
            time_s=state['t'], steps=len(rows), switches=state['switches'], reward=state['reward'],
            wall_s=time.perf_counter()-start, cpu_s=time.process_time()-cpu_start, CL=cl,
            CL_rmse=float(np.sqrt(np.mean([r['CL_rmse']**2 for r in rows]))) if rows else None,
            max_boundary_jump_deg=max([abs(r['boundary_jump_deg']) for r in rows],default=0),
            max_alpha_rate_degps=max([r['max_alpha_rate_degps'] for r in rows],default=0),
            minimum_normalized_margins=margins,
            tracking_deviation_steps=sum(not r['quality']['passed'] for r in rows if r['quality']),
            safety_intervention_steps=sum(r['safety_intervention'] for r in rows),
            total_drag_work_J=sum(r['drag_work_J'] for r in rows),
            decision_wall_p50_s=float(np.median([r['wall_s'] for r in rows])) if rows else None,
            decision_wall_p95_s=float(np.percentile([r['wall_s'] for r in rows],95)) if rows else None,
            counts=dict(counts), final_state=state, rows=rows,
            safety_scope='continuous current3s and finite H30 witnesses; no legacy qualification inherited',
            training_performed=False, surrogate_refit_performed=True,
            pilot_not_production_qualification=True)
    try:
        for step in range(cfg['flight']['max_decisions']):
            tick = time.perf_counter()
            cpu_tick = time.process_time()
            folder = out/f'decision_{step:04d}'
            folder.mkdir()
            write_json(folder/'before.json', state)
            z = Transition(**state['transition'])
            z.settle(state['t'])
            reachable = system['reachability'].check(np.asarray(state['x']), state['alpha'], z,
                                                     state['t'], cl, enabled=True)
            cmd = reachable['CL_cmd']
            c = context_from_state(state, cmd)
            c['CL_raw'] = cl
            fallbacks = []
            pre_record = None
            try:
                mach, pd = model.predict(c, 'pre')
            except OracleRequired as exc:
                counts['pre_ood'] += 1
                mach = baseline(system)(c, 'pre')
                pd = dict(reason=exc.reason, supported=False, fallback='CONTINUOUS_PHYSICAL')
                fallbacks.append('PRE_PHYSICAL_ON_MODEL_REFUSAL')
            choice = db.generate(float(mach[0]), cmd, current=z.target)
            switched = False
            if choice['geometry_index'] is not None:
                switched = z.command(choice['geometry_index'], state['t'])
            post_c = context_from_state(dict(state, transition=vars(z)), cmd)
            post_c['CL_raw'] = cl
            try:
                future, postd = model.predict(post_c, 'post')
                nodes = np.r_[state['alpha'], future]
                post_c['proposal_origin'] = 'POST_SURROGATE'
                counts['post_surrogate_proposed'] += 1
            except OracleRequired as exc:
                counts['post_ood'] += 1
                nodes = safety.candidate(post_c)
                post_c['proposal_origin'] = 'CONTINUOUS_PHYSICAL_ON_MODEL_REFUSAL'
                postd = dict(reason=exc.reason, supported=False, fallback='CONTINUOUS_PHYSICAL')
                fallbacks.append('POST_PHYSICAL_ON_MODEL_REFUSAL')
            checked = safety.assess(post_c, nodes)
            write_json(folder/'safety.json', dict(context=post_c, diagnostics=checked.diagnostics,
                accepted=checked.accepted, status=checked.status, audit=checked.audit))
            if not checked.accepted:
                counts['true_ocp_fallback'] += 1
                record, plan = system['teacher'].solve_audited(post_c, 'post', output_dir=folder/'ocp_fallback')
                fallbacks.append('TRUE_CONTINUOUS_FINAL_ON_GUARD_REFUSAL')
                if record['accepted']:
                    nodes, _ = prefix(plan.alpha_nodes, plan.times, 3.)
                    # A safety recovery may intentionally deviate from CL.
                    # Preserve its independently audited continuation instead
                    # of replacing it with a tracking-only physical seed.
                    checked = safety.assess_ocp_plan(post_c, plan)
                    write_json(folder/'ocp_tail_safety.json', dict(context=post_c,
                        diagnostics=checked.diagnostics, accepted=checked.accepted,
                        status=checked.status, audit=checked.audit))
                if not record['accepted'] or not checked.accepted:
                    reason = 'UNKNOWN_NOT_CERTIFIED'
                    counts[reason] += 1
                    write_json(out/'stop_state.json', dict(reason=reason, state=state,
                        last_certification=str(folder/('ocp_tail_safety.json' if record['accepted'] else 'ocp_fallback/label.json')),
                        teacher_record=record,
                        safety_status=checked.status))
                    break
            nodes = np.asarray(checked.alpha_nodes)
            cert = bind_proof(state, nodes, identity, z, cmd, checked.arrays['0.5']['x'][-1],
                              digest(checked.diagnostics))
            cert['status'] = 'VERIFIED_PASS'
            write_json(folder/'certificate.json', dict(cert, proposed_transition=vars(z),
                future_witness=checked.diagnostics, reachability=reachable))
            binding = dict(cert)
            binding.pop('status')
            next_state, audit, arrays = commit_audited(system, state, z, nodes, binding, identity, cmd,
                                                      checked.diagnostics)
            if not np.array_equal(arrays['0.5']['x'][-1], checked.arrays['0.5']['x'][-1]):
                raise ValueError('Certified versus committed trajectory mismatch')
            dense = audit['grids']['0.1']
            margins = {k:min(dense['margins'][k],audit['grids']['0.5']['margins'][k]) for k in dense['margins']}
            ss = arrays['0.1']['samples']
            energy = (ss[10]*ss[1]).reshape(-1,5)[:,:4] @ (np.array([1,2,2,1])/6)
            row = dict(step=step, t0=state['t'], t1=next_state['t'], CL_raw=cl, CL_cmd=cmd,
                CL_mean=dense['CL_mean_first3'], CL_rmse=dense['CL_rmse_first3'],
                alpha_nodes_deg=np.rad2deg(nodes).tolist(), boundary_jump_deg=float(np.rad2deg(nodes[0]-state['alpha'])),
                max_alpha_rate_degps=audit['maximum_alpha_rate_degps'],
                predicted_flight_mach=float(mach[0]), generator=choice, switched=bool(switched),
                source=db.ids[z.source], target=db.ids[z.target], fraction=post_c['fraction'],
                margins=margins, pre=pd, post=postd, fallbacks=fallbacks,
                safety_status=checked.status, wall_s=time.perf_counter()-tick,
                cpu_s=time.process_time()-cpu_tick,
                quality=checked.diagnostics.get('quality'),
                safety_intervention=checked.diagnostics.get('safety_intervention', False),
                drag_work_J=float(np.sum(energy)*.1),
                dynamic_pressure_max_pa=float(ss[4].max()), heat_max_wm2=float(ss[5].max()),
                load_max_g=float(ss[6].max()),
                state_end=next_state['x'])
            np.savez_compressed(folder/'executed.npz', alpha_nodes=nodes, times=np.arange(7)*.5,
                **{f'{key}_{field}': arr[field] for key, arr in arrays.items() for field in ('x','samples')})
            next_state['switches'] += int(switched)
            state = next_state
            rows.append(row)
            counts['VERIFIED_PASS'] += 1
            counts[checked.status] += 1
            reason = m.outcome(state['t'], np.asarray(state['x']))
            write_json(out/'checkpoint.json', dict(protocol=PROTOCOL, identity=identity, state=state))
            write_json(out/'result.json', result())
            print(json.dumps(dict(stage='flight', t=state['t'], range_km=state['x'][3]/1000,
                CL_rmse=row['CL_rmse'], status=row['safety_status'], wall_s=row['wall_s'])), flush=True)
            if reason != 'INCOMPLETE':
                break
    except Exception as exc:
        reason = 'ERROR:'+type(exc).__name__+':'+str(exc)
        write_json(out/'stop_state.json', dict(reason=reason, state=state))
        raise
    finally:
        write_json(out/'result.json', result())
    return result()

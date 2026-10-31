"""Serial fresh teacher augmentation and local residual-model preparation.

The eight contexts are specified before solving. Historical development splits
remain untouched; all added labels belong to one explicit training group. This
module does not execute a flight or change the old controller artifacts.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time

import numpy as np

from certified_flight import integrity
from certified_flight.build import ROOT
from continuous_surrogate.experiment import (
    baseline, build as old_build, context_from_state, source_identity,
    transition_from_context, write_json,
)
from continuous_surrogate.teacher import PROTOCOL
from .local_model import LocalResidualSurrogate
from .tracking import TrackingPrioritySafety

CONTROLLER_PROTOCOL = 'continuous_recovery/1'
DEFAULT_CONFIG = ROOT/'experiments/continuous_recovery/config.json'
DEFAULT_OLD_LABELS = ROOT/'runs/continuous_surrogate_20261031_01/labels/labels.json'
DEFAULT_FLIGHT = ROOT/'runs/continuous_surrogate_20261031_02/flight'
DEFAULT_RECOVERY_STEP = ROOT/'runs/ripple_diagnosis_20261031/tracking_first/step_09.json'
ADDED_GROUP = 'nominal_closed_loop_20261031'
ADDED_TIMES = (120., 300., 588., 600., 750., 900., 1080.)


def build(config=DEFAULT_CONFIG):
    """Keep original physics/teacher; replace only the experimental P1 guard."""
    config = Path(config).resolve(strict=True)
    cfg = json.loads(config.read_text(encoding='utf-8'))
    if cfg.get('controller_protocol') != CONTROLLER_PROTOCOL:
        raise ValueError('Continuous recovery controller protocol mismatch')
    if cfg.get('model', {}).get('max_neighbor_distance') != .25:
        raise ValueError('This experiment fixes the original support radius at 0.25')
    if cfg.get('model', {}).get('local_baseline_prior_weight') != 1.:
        raise ValueError('This experiment fixes the zero-residual prior at 1.0')
    if cfg.get('recovery', {}).get('exit_confirmations') != 2:
        raise ValueError('This experiment requires two certified exit confirmations')
    if cfg.get('training', {}).get('additional_contexts') != 8:
        raise ValueError('The additional teacher design contains exactly eight contexts')
    if cfg.get('flight') != {'CL': .025, 'max_decisions': 400}:
        raise ValueError('This pilot is one fixed CL=0.025, 400-decision flight')
    started = time.perf_counter()
    system = old_build(config)
    old_guard = system['safety']
    priority_started = time.perf_counter()
    system['safety'] = TrackingPrioritySafety(old_guard)
    system['config_path'] = str(config)
    system['controller_protocol'] = CONTROLLER_PROTOCOL
    system['identity'] = source_identity(config)
    system['build_timings'] = dict(total_wall_s=time.perf_counter()-started,
                                  priority_guard_wall_s=time.perf_counter()-priority_started)
    return system


def _reference(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=integrity.sha256(path))


def prepare_contexts(system, flight=DEFAULT_FLIGHT, recovery_step=DEFAULT_RECOVERY_STEP):
    """Read seven saved actual states plus the explicitly diagnosed recovery state."""
    contexts = []
    cl = float(system['config']['flight']['CL'])
    for t in ADDED_TIMES:
        decision = int(t/3.)
        folder = Path(flight)/f'decision_{decision:04d}'
        path = folder/'before.json'
        state = json.loads(path.read_text(encoding='utf-8'))
        if float(state['t']) != t:
            raise ValueError(f'Saved state time mismatch: {path}')
        context = context_from_state(state, cl)
        previous_proof = Path(flight)/f'decision_{decision-1:04d}'/'certificate.json'
        contexts.append(dict(context=context, role='train', group=ADDED_GROUP,
                             kind='actual_nominal_closed_loop_state',
                             source=_reference(path), previous_execution_proof=_reference(previous_proof)))
    path = Path(recovery_step)
    saved = json.loads(path.read_text(encoding='utf-8'))
    context = deepcopy(saved['context'])
    if float(context['t']) != 615. or saved.get('accepted') is not True:
        raise ValueError('Expected the accepted diagnostic recovery state at t=615 s')
    if context['control_contract'] != PROTOCOL or float(context['CL_des']) != cl:
        raise ValueError('Diagnostic recovery context control contract mismatch')
    context.pop('proposal_origin', None)
    context.update(CL_raw=cl, CD_des=None)
    contexts.append(dict(context=context, role='train', group=ADDED_GROUP,
                         kind='diagnosed_tracking_priority_recovery_state', source=_reference(path)))
    if len(contexts) != 8:
        raise ValueError('Expected exactly eight additional context groups')
    return contexts


def generate_labels(system, old_labels, out, flight=DEFAULT_FLIGHT,
                    recovery_step=DEFAULT_RECOVERY_STEP):
    """Solve each new Pre, apply unchanged Generator, then solve each new Post."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    old_labels = Path(old_labels).resolve(strict=True)
    old = json.loads(old_labels.read_text(encoding='utf-8'))
    if len(old) != 48:
        raise ValueError('This experiment reuses exactly the 48 original continuous teacher records')
    if any(row.get('teacher_contract') != system['teacher'].experiment for row in old):
        raise ValueError('Reused labels have an incompatible teacher contract')
    if any(row.get('group') == ADDED_GROUP for row in old):
        raise ValueError('New closed-loop group already appears in the source labels')
    contexts = prepare_contexts(system, flight, recovery_step)
    write_json(out/'design.json', dict(protocol=PROTOCOL, controller_protocol=CONTROLLER_PROTOCOL,
        reused_labels=_reference(old_labels), added_contexts=contexts,
        expected_new_solves=16, order='Serial Pre, Generator, Post per context; no parallel solves',
        note='All evidence is development data; prior validation/test partitions are unchanged, not blind tests.'))
    write_json(out/'reused_labels.json', old)
    records = deepcopy(old)
    fresh = []
    first_case = max(int(row['case']) for row in old)+1
    started, cpu_started = time.perf_counter(), time.process_time()

    def persist():
        write_json(out/'labels.json', records)
        write_json(out/'fresh_labels.json', fresh)
        write_json(out/'summary.json', dict(protocol=PROTOCOL, controller_protocol=CONTROLLER_PROTOCOL,
            wall_s=time.perf_counter()-started, cpu_s=time.process_time()-cpu_started,
            reused_records=len(old), new_ocp_labels_generated=len(fresh),
            new_accepted=sum(row['accepted'] for row in fresh),
            accepted=dict(Counter(row['head']+'_'+row['role'] for row in records if row['accepted'])),
            refused=[dict(case=row['case'], head=row['head'], reason=row.get('reason'))
                     for row in fresh if not row['accepted']], development_only=True))

    persist()
    for i, item in enumerate(contexts):
        c = item['context']
        case = first_case+i
        folder = out/f'case_{case:02d}'
        folder.mkdir()
        pre, _ = system['teacher'].solve_audited(c, 'pre', output_dir=folder/'pre')
        pre.update(role='train', group=ADDED_GROUP, case=case, context_source=item['source'])
        records.append(pre)
        fresh.append(pre)
        persist()
        if pre['accepted']:
            choice = system['db'].generate(float(pre['y'][0]), c['CL_des'], current=c['target'])
            z = transition_from_context(c)
            if choice['geometry_index'] is not None:
                z.command(choice['geometry_index'], c['t'])
            post_c = dict(c, source=z.source, target=z.target, fraction=z.fraction(c['t']),
                          rate=.125 if z.active(c['t']) else 0.)
            post, _ = system['teacher'].solve_audited(post_c, 'post', output_dir=folder/'post')
            post.update(role='train', group=ADDED_GROUP, case=case, generator=choice,
                        context_source=item['source'])
            records.append(post)
            fresh.append(post)
            persist()
        print(json.dumps(dict(stage='fresh_recovery_teacher_labels', contexts_done=i+1,
            contexts_total=8, accepted=sum(row['accepted'] for row in fresh),
            attempted=len(fresh), t=c['t'], wall_s=time.perf_counter()-started)), flush=True)
    return records


def fit_model(system, records, out):
    """Fit the unchanged Pre algorithm and compact local Post from audited labels."""
    config = dict(system['config']['model'],
                  minimum_training_labels_per_head=system['config']['training']['minimum_training_labels_per_head'])
    model = LocalResidualSurrogate.fit(records, system['db'], system['mission'], baseline(system),
        config=config, metadata=dict(source_identity=system['identity'], pilot=True,
            controller_protocol=CONTROLLER_PROTOCOL, final_model_qualification=False,
            development_partitions_only=True))
    model.save(out)
    validation = model.evaluate(records)
    checks = []
    for row in records:
        if not row['accepted'] or row['head'] != 'post':
            continue
        predicted, diagnostics = model._predict(row['context'], 'post', False)
        audit, _ = system['plant'].audit(row['context'],
            np.r_[row['context']['alpha_prev'], predicted], np.arange(7)*.5, row['context']['CL_des'])
        checks.append(dict(case=row['case'], role=row['role'], group=row['group'],
                           supported=diagnostics['supported'], current_audit=audit))
    validation.update(post_current_physical_checks=checks, development_partitions_only=True,
                      controller_protocol=CONTROLLER_PROTOCOL)
    write_json(Path(out).with_name('model_validation.json'), validation)
    return model


def snapshot_sources(system, out):
    """Archive the source/config bytes identified before any teacher solves."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for relative, expected in system['identity']['files'].items():
        src = ROOT/relative
        dst = out/relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        if integrity.sha256(src) != expected:
            raise ValueError(f'Source changed before snapshot: {relative}')
        shutil.copyfile(src, dst)
        if integrity.sha256(dst) != expected:
            raise ValueError(f'Source snapshot hash mismatch: {relative}')
        manifest[relative] = expected
    write_json(out/'manifest.json', manifest)
    return manifest

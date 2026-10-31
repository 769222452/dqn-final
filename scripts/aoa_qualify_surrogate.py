"""Offline qualification of a refitted surrogate (same P1 evaluation as data/lookahead/AB_metrics.json).

    python scripts/aoa_qualify_surrogate.py --objective consistent \
        --model data/surrogate_consistent/controller.npz --labels runs/aoa/surrogate/labels.jsonl ...

On the validation and test labels (flights never used for fitting) the surrogate is compared with its
teacher: raw network (pre Mach on the recorded pre context, six alphas on the teacher's post context) and
corrected network (V16 post corrector with the objective's settings). Reported per split and method:
alpha error (deg), planned-Mach error, CL tracking error of the 3 s plan, local range and end-state
error vs the teacher plan, dense-audit failures, OOD refusals, per-context sawtooth indicators (intra-step
peak-to-peak and boundary jump) for teacher, raw and corrected plans, and the speed-up.

P1_physical_gate_passed (required by LookaheadGuard.qualify) is the V16 rule: zero independent dense
failures of corrected plans on validation and test (gate_config dense_violation_rate_max = 0) with at
least gate_config's minimum sample counts. The accuracy gates of gate_config are reported, not required
(the V16 surrogate itself did not pass them: controller.json scale_up_gate_passed=false).
Writes <qualification-dir>/AB_metrics.json and manifest.json (default data/lookahead_<name>/).
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from certified_flight.build import runtime_config  # noqa: E402
from certified_flight.corrector import PostCorrector, audit_controls  # noqa: E402
from certified_flight.diagnostics import interval_moments  # noqa: E402
from certified_flight.distill import Teacher, read_jsonl, sha256  # noqa: E402
from certified_flight.lookahead import QUALIFIED_KEYS  # noqa: E402
from certified_flight.objectives import OBJECTIVES, objective_safety  # noqa: E402
from certified_flight.serialization import clean  # noqa: E402
from certified_flight.surrogate import OracleRequired, SurrogateController  # noqa: E402


def stats(x):
    x = np.asarray([v for v in x if v is not None], float)
    if not len(x):
        return dict(count=0)
    return dict(count=int(x.size), mean=float(x.mean()), mae=float(np.mean(abs(x))), rmse=float(np.sqrt(np.mean(x**2))),
                abs_p95=float(np.percentile(abs(x), 95)), max_abs=float(np.max(abs(x))))


def plan_metrics(plant, context, alpha, teacher_end):
    audit = audit_controls(plant, context, alpha)
    out = dict(dense_valid=bool(audit['valid']),
               intra_ptp_deg=float(np.rad2deg(np.ptp(alpha))),
               boundary_jump_deg=float(np.rad2deg(abs(alpha[0]-context['alpha_prev']))))
    if audit['actual'] is not None:
        end = audit['actual']['x'][-1]
        out.update(CL_tracking_error=interval_moments(audit['actual'])['mean']-context['CL_des'],
                   end=end.tolist())
        if teacher_end is not None:
            out['end_error'] = (end-teacher_end).tolist()
    return out


def evaluate(teacher, model, corrector, rows):
    plant = teacher.plant
    methods = {m: dict(samples=0, accepted=0, independent_dense_failures=0, prediction_or_correction_failures=0,
                       alpha=[], mach=[], CL=[], end=[], ptp=[], jump=[], wall=[], refusals={})
               for m in ('teacher', 'raw_network', 'corrected_network')}
    for r in rows:
        pre = dict(r['pre_context'], control_contract=model.control_contract)
        post = dict(r['post_context'], control_contract=model.control_contract)
        u_teacher = np.asarray(r['post_alpha'])
        t_metrics = plan_metrics(plant, post, u_teacher, None)
        teacher_end = np.asarray(t_metrics['end']) if 'end' in t_metrics else None
        m = methods['teacher']
        m['samples'] += 1
        m['accepted'] += t_metrics['dense_valid']
        m['independent_dense_failures'] += not t_metrics['dense_valid']
        m['CL'].append(t_metrics.get('CL_tracking_error'))
        m['ptp'].append(t_metrics['intra_ptp_deg'])
        m['jump'].append(t_metrics['boundary_jump_deg'])
        m['wall'].append(r['wall_s'])
        tick = time.perf_counter()
        try:
            mach, _ = model.predict(pre, 'pre')
            alpha, _ = model.predict(post, 'post')
        except OracleRequired as exc:
            for name in ('raw_network', 'corrected_network'):
                methods[name]['samples'] += 1
                methods[name]['prediction_or_correction_failures'] += 1
                methods[name]['refusals'][exc.reason] = methods[name]['refusals'].get(exc.reason, 0)+1
            continue
        raw_wall = time.perf_counter()-tick
        candidates = dict(raw_network=(alpha, raw_wall))
        tick = time.perf_counter()
        corrected = corrector.correct(post, alpha)
        if corrected.accepted:
            candidates['corrected_network'] = (corrected.alpha, raw_wall+time.perf_counter()-tick)
        else:
            m = methods['corrected_network']
            m['samples'] += 1
            m['prediction_or_correction_failures'] += 1
            m['refusals'][corrected.status] = m['refusals'].get(corrected.status, 0)+1
        for name, (u, wall) in candidates.items():
            pm = plan_metrics(plant, post, np.asarray(u), teacher_end)
            m = methods[name]
            m['samples'] += 1
            m['accepted'] += pm['dense_valid']
            m['independent_dense_failures'] += not pm['dense_valid']
            m['alpha'] += np.rad2deg(np.asarray(u)-u_teacher).tolist()
            m['mach'].append(float(mach[0])-r['pre_mach'])
            m['CL'].append(pm.get('CL_tracking_error'))
            m['end'].append(pm.get('end_error'))
            m['ptp'].append(pm['intra_ptp_deg'])
            m['jump'].append(pm['boundary_jump_deg'])
            m['wall'].append(wall)
    out = {}
    for name, m in methods.items():
        ends = np.asarray([e for e in m['end'] if e is not None]).reshape(-1, 4)
        out[name] = dict({k: m[k] for k in ('samples', 'accepted', 'independent_dense_failures',
                                            'prediction_or_correction_failures', 'refusals')},
                         alpha_error_deg=stats(m['alpha']), planned_mach_error=stats(m['mach']),
                         CL_tracking_error=stats(m['CL']),
                         local_range_error_m=stats(ends[:, 3]) if len(ends) else dict(count=0),
                         state_end_error={k: stats(ends[:, i]) if len(ends) else dict(count=0)
                                          for i, k in enumerate(('h_m', 'V_mps', 'gamma_rad', 's_m'))},
                         intra_step_ptp_deg=stats(m['ptp']), boundary_jump_deg=stats(m['jump']),
                         wall_s=stats(m['wall']))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--objective', choices=sorted(OBJECTIVES), required=True)
    p.add_argument('--model', required=True, help='controller.npz (repository-relative)')
    p.add_argument('--labels', nargs='+', required=True)
    p.add_argument('--out', help='qualification directory (default data/lookahead_<name>)')
    a = p.parse_args(argv)
    started = time.perf_counter()
    model_path = Path(a.model)
    out = Path(a.out) if a.out else model_path.parent.with_name(
        'lookahead_'+model_path.parent.name.removeprefix('surrogate_'))
    teacher = Teacher(a.objective)
    model = SurrogateController.load(ROOT/model_path, teacher.db, teacher.mission, teacher.physical,
                                     {'mission_sha256': teacher.mission.config_path,
                                      'trajectory_sha256': teacher.mission.path(teacher.physical['trajectory'])})
    safety = objective_safety(runtime_config(ROOT)['safety_options'], a.objective)
    corrector = PostCorrector(teacher.plant, safety)
    rows = [r for f in a.labels for r in read_jsonl(f) if r.get('ok')]
    gate = model.metadata['gate_config']
    splits = {}
    for split in ('validation', 'test'):
        chosen = [r for r in rows if r['split'] == split]
        splits[split] = dict(expected_samples=len(chosen), actual_samples=len(chosen),
                             methods=evaluate(teacher, model, corrector, chosen))
    corrected = {s: v['methods']['corrected_network'] for s, v in splits.items()}
    raw = {s: v['methods']['raw_network'] for s, v in splits.items()}
    checks = dict(
        validation_samples=splits['validation']['actual_samples'] >= gate['min_validation_samples'],
        test_samples=splits['test']['actual_samples'] >= gate['min_test_samples'],
        validation_corrected_dense_failures_zero=corrected['validation']['independent_dense_failures'] == 0,
        test_corrected_dense_failures_zero=corrected['test']['independent_dense_failures'] == 0)
    teacher_wall = splits['test']['methods']['teacher']['wall_s'].get('mean')
    fast_wall = corrected['test']['wall_s'].get('mean')
    accuracy = dict(
        test_raw_planned_mach_mae=dict(value=raw['test']['planned_mach_error'].get('mae'),
                                       threshold=gate['planned_mach_mae_max'], kind='max'),
        test_raw_alpha_rmse_deg=dict(value=raw['test']['alpha_error_deg'].get('rmse'),
                                     threshold=gate['alpha_rmse_deg_max'], kind='max'),
        test_raw_alpha_abs_p95_deg=dict(value=raw['test']['alpha_error_deg'].get('abs_p95'),
                                        threshold=gate['alpha_abs_p95_deg_max'], kind='max'),
        speedup=dict(value=teacher_wall/fast_wall if teacher_wall and fast_wall else None,
                     threshold=gate['min_speedup'], kind='min'))
    for v in accuracy.values():
        v['passed'] = None if v['value'] is None else bool(
            v['value'] >= v['threshold'] if v['kind'] == 'min' else v['value'] <= v['threshold'])
    passed = all(checks.values())
    report = dict(status='COMPLETE', objective=a.objective, model=str(model_path),
                  model_sha256=model.metadata['model_sha256'], control_contract=model.control_contract,
                  splits=splits, checks=checks, P1_physical_gate_passed=passed,
                  P1_physical_gate_status='PASSED' if passed else 'FAILED', accuracy_gates_reported=accuracy,
                  evaluation_role='AOA_FIX_REFIT_QUALIFICATION', wall_s=time.perf_counter()-started)
    out.mkdir(parents=True, exist_ok=True)
    (out/'AB_metrics.json').write_text(json.dumps(clean(report), indent=1)+'\n', encoding='utf-8')
    sources = {str(model_path): model.metadata['model_sha256'],
               str(model_path.with_suffix('.json')): sha256(ROOT/model_path.with_suffix('.json'))}
    sources.update({str(f): sha256(f) for f in a.labels})
    for f in ('scripts/aoa_qualify_surrogate.py', 'src/certified_flight/distill.py',
              'src/certified_flight/corrector.py', 'src/certified_flight/ocp.py'):
        sources[f] = sha256(ROOT/f)
    manifest = dict(schema='aoa-fix-P1-evaluation/1', status='COMPLETE', source_sha256=sources,
                    config=dict(objective=a.objective, fast_safety={k: safety[k] for k in QUALIFIED_KEYS if k in safety}),
                    control_contract=model.control_contract, gates=dict(maximum_dense_violations=0, **gate),
                    expected_validation=splits['validation']['actual_samples'],
                    expected_test=splits['test']['actual_samples'], gate_status=report['P1_physical_gate_status'])
    (out/'manifest.json').write_text(json.dumps(clean(manifest), indent=1)+'\n', encoding='utf-8')
    brief = {s: {m: dict(dense_failures=v['independent_dense_failures'], refused=v['prediction_or_correction_failures'],
                         alpha_rmse_deg=v['alpha_error_deg'].get('rmse'), ptp_mean_deg=v['intra_step_ptp_deg'].get('mean'),
                         jump_mean_deg=v['boundary_jump_deg'].get('mean'))
                 for m, v in d['methods'].items()} for s, d in splits.items()}
    print(json.dumps(clean(dict(P1=report['P1_physical_gate_status'], checks=checks, splits=brief,
                                accuracy=accuracy, out=str(out)))))
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())

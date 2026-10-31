"""Bounded frozen-state and 12-second true dual-OCP checks; no RL training."""
import argparse
from copy import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'src'))

for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
from certified_flight.transition import Transition
from certified_flight.diagnostics import interval_moments
from continuous_tracking.experiment import build as original_build, write_json, solve_audited, Refused, quality
from continuous_tracking.plant import prefix
from controller import PROTOCOL, UniformTrackingOCP


def build(config=HERE/'config.json'):
    base, _, old_identity = original_build()
    cfg = json.loads(Path(config).read_text(encoding='utf-8'))
    ocp = UniformTrackingOCP(base.m, base.db, base.traj, base.cfg, base.plant, cfg)
    hashes = dict(old_identity['files'])
    for path in [*sorted(HERE.glob('*.py')), Path(config)]:
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return ocp, cfg, dict(protocol=PROTOCOL, files=hashes,
                         sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest())


def memory(rows, step):
    if step == 0:
        return dict(alpha_rate_prev=0., alpha_interval_prev_s=.5)
    prev = rows[step-1]
    dt = float(prev['times'][-1]-prev['times'][-2])
    return dict(alpha_rate_prev=float(np.deg2rad(prev['alpha_nodes_deg'][-1]-prev['alpha_nodes_deg'][-2])/dt),
                alpha_interval_prev_s=dt)


def old_plan(source, step):
    folder = source/f'decision_{step:04d}'
    record = json.loads((folder/'final.json').read_text(encoding='utf-8'))
    k = len(record['attempts'])-1
    with np.load(folder/f'final_{k}.npz') as z:
        return record['attempts'][k], z['times'].copy(), z['alpha_nodes'].copy()


def metrics(ocp, cfg, c, plan, audit, arrays):
    aa, tt = prefix(plan.alpha_nodes, plan.times, 3.)
    q = quality(audit, c['CL_des'], cfg['tracking_quality'])
    slope = np.diff(aa)/np.diff(tt)
    stats = audit['grids']['0.1']
    full = interval_moments(arrays['0.1'], horizon=plan.times[-1], target=c['CL_des'])
    return dict(CL_mean=stats['CL_mean_first3'], CL_rmse=stats['CL_rmse_first3'], quality=q,
        alpha_nodes_deg=np.rad2deg(aa).tolist(), times=tt.tolist(),
        alpha_peak_to_peak_deg=float(np.rad2deg(np.ptp(aa))),
        alpha_boundary_jump_deg=float(np.rad2deg(aa[0]-c['alpha_prev'])),
        rate_boundary_jump_degps=float(np.rad2deg(slope[0]-c['alpha_rate_prev'])),
        maximum_alpha_rate_degps=float(np.rad2deg(np.max(abs(slope)))),
        maximum_planned_CL_slack=plan.terms['maximum_CL_slack'],
        planned_CL_rmse=full['rmse'], full_plan_audit_valid=audit['valid'],
        solver_s=plan.solve_s, solver_status=plan.status, iterations=plan.iterations)


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source_raw = (args.source/'result.json').read_bytes()
    source = json.loads(source_raw)
    ocp, cfg, identity = build(args.config)
    write_json(args.out/'identity.json', identity)
    write_json(args.out/'config.json', cfg)
    write_json(args.out/'source.json', dict(path=str(args.source), sha256=hashlib.sha256(source_raw).hexdigest(),
        snapshot_time_s=source['time_s'], scope='Saved completed evidence only; no original files modified'))
    result = dict(protocol=PROTOCOL, frozen=[], short=None, training_performed=False, surrogate_used=False)
    for step in args.frozen_steps:
        folder = args.out/f'frozen_{step:04d}'; folder.mkdir()
        old, times, alpha = old_plan(args.source, step)
        c = dict(old['context'], **memory(source['rows'], step))
        write_json(folder/'context.json', c)
        tick = time.perf_counter()
        row = dict(step=step, t=c['t'], old=source['rows'][step], context=c)
        try:
            plan, aud, arrays, attempts = solve_audited(ocp, c, alpha, folder, 'final')
            row.update(status='AUDITED_PASS', new=metrics(ocp, cfg, c, plan, aud, arrays), attempts=len(attempts))
        except (Refused, RuntimeError, ValueError) as exc:
            row.update(status=str(exc))
        row['wall_s'] = time.perf_counter()-tick
        result['frozen'].append(row)
        write_json(args.out/'result.json', result)
        print(json.dumps({'frozen_t':c['t'], 'status':row['status'], 'new':row.get('new'), 'wall_s':row['wall_s']}), flush=True)

    before = json.loads((args.source/f'decision_{args.start_step:04d}'/'before.json').read_text())
    t, x, a = before['t'], np.array(before['x'], float), before['alpha']
    z = Transition(**before['transition'])
    memo = memory(source['rows'], args.start_step)
    previous = None if args.start_step == 0 else old_plan(args.source, args.start_step-1)[1:]
    rows, reason = [], 'SHORT_TEST_RUNNING'
    cl = source['rows'][args.start_step]['CL_cmd']
    if args.start_step+args.decisions > len(source['rows']):
        raise ValueError('Original comparison does not cover the short-test interval')
    if any(row['CL_cmd'] != cl for row in source['rows'][args.start_step:args.start_step+args.decisions]):
        raise ValueError('Short comparison requires the original constant CL')
    short_start = time.perf_counter()
    calls = retries = 0
    folder = args.out/'short'; folder.mkdir()

    def state():
        return dict(t=t, x=x.tolist(), alpha=a, transition=vars(z), **memo)

    try:
        for i in range(args.decisions):
            dest = folder/f'decision_{i:04d}'; dest.mkdir()
            write_json(dest/'before.json', state())
            tick = time.perf_counter()
            candidate = copy(z); candidate.settle(t)
            c = dict(x=x.tolist(), alpha_prev=a, t=t, source=candidate.source, target=candidate.target,
                     fraction=candidate.fraction(t), rate=.125 if candidate.active(t) else 0., CL_des=cl, **memo)
            grid = np.r_[0., np.cumsum(ocp.grid(1200-t))]
            guess = None if previous is None else np.interp(grid+3., previous[0], previous[1])
            pre, pre_audit, pre_arrays, attempts = solve_audited(ocp, c, guess, dest, 'pre')
            calls += len(attempts); retries += len(attempts)-1
            midpoint = (x+pre_arrays['0.5']['x'][5])/2
            mach = float(midpoint[1]/ocp.plant.atmosphere.speed_of_sound(midpoint[0]))
            choice = ocp.db.generate(mach, cl, current=candidate.target)
            if choice['geometry_index'] is None:
                raise Refused('GENERATOR_NO_FEASIBLE_GEOMETRY')
            changed = candidate.command(choice['geometry_index'], t)
            c.update(source=candidate.source, target=candidate.target, fraction=candidate.fraction(t),
                     rate=.125 if candidate.active(t) else 0.)
            final, aud, full, attempts = solve_audited(ocp, c, pre.alpha_nodes, dest, 'final')
            calls += len(attempts); retries += len(attempts)-1
            aa, tt = prefix(final.alpha_nodes, final.times, 3.)
            current_audit, current = ocp.plant.audit(c, aa, tt, cl)
            if not current_audit['valid']:
                raise Refused('CURRENT_INTERVAL_AUDIT_REJECTED')
            if not np.array_equal(current['0.5']['x'][-1], full['0.5']['x'][5]):
                raise Refused('AUDITED_PREFIX_ENDPOINT_MISMATCH')
            row = metrics(ocp, cfg, c, final, aud, full)
            row.update(t0=t, t1=t+3., pre_solve_s=pre.solve_s, final_solve_s=final.solve_s,
                       generator=choice, switch_started=changed, wall_s=time.perf_counter()-tick,
                       min_margins={k:min(current_audit['grids'][str(dt)]['margins'][k] for dt in (.1,.5))
                                    for k in current_audit['grids']['0.1']['margins']})
            write_json(dest/'execution_audit.json', dict(context=c, audit=current_audit, metrics=row))
            np.savez_compressed(dest/'executed.npz', alpha_nodes=aa, times=tt,
                **{f'{k}_{f}': v[f] for k,v in current.items() for f in ('x','samples')})
            memo = dict(alpha_rate_prev=float((aa[-1]-aa[-2])/(tt[-1]-tt[-2])),
                        alpha_interval_prev_s=float(tt[-1]-tt[-2]))
            t, x, a, z = t+3., current['0.5']['x'][-1].copy(), float(aa[-1]), candidate
            z.settle(t); previous = (final.times, final.alpha_nodes)
            rows.append(row)
            write_json(folder/'checkpoint.json', dict(protocol=PROTOCOL, state=state(), diagnostic_only=True))
            print(json.dumps({'short_t':t, 'CL_rmse':row['CL_rmse'], 'alpha_peak_to_peak_deg':row['alpha_peak_to_peak_deg'],
                              'rate_jump_degps':row['rate_boundary_jump_degps'], 'wall_s':row['wall_s']}), flush=True)
        reason = 'SHORT_TEST_COMPLETE'
    except (Refused, RuntimeError, ValueError) as exc:
        reason = str(exc)
        write_json(folder/'stop_state.json', dict(reason=reason, state=state()))
    finally:
        # Count saved attempts too, including a refused solve that never returned.
        groups = [json.loads(path.read_text())['attempts'] for path in folder.glob('decision_*/*.json')
                  if path.name in ('pre.json','final.json')]
        attempts = [entry for group in groups for entry in group]
        result['short'] = dict(start_t=before['t'], final_t=t, reason=reason, requested_decisions=args.decisions,
            completed_decisions=len(rows), wall_s=time.perf_counter()-short_start, state=state(), rows=rows,
            optimizer_calls=len(attempts), retries=sum(max(0,len(group)-1) for group in groups),
            old_rows=source['rows'][args.start_step:args.start_step+args.decisions],
            warning='12-second diagnostic only; not a completed 1200-second mission')
        result['wall_s'] = time.perf_counter()-started
        write_json(args.out/'result.json', result)
    print(json.dumps({'finished':reason, 'completed_decisions':len(rows), 'wall_s':result['wall_s']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--config', type=Path, default=HERE/'config.json')
    p.add_argument('--frozen-steps', type=int, nargs='+', default=[0,16,125,163])
    p.add_argument('--start-step', type=int, default=163)
    p.add_argument('--decisions', type=int, default=4, choices=range(1,9))
    run(p.parse_args())

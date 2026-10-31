"""Audited real-OCP experiment, with evidence and explicit tracking-quality flags."""
from copy import copy
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
from certified_flight.transition import Transition
from . import PROTOCOL
from .ocp import TrackingOCP
from .plant import ContinuousPlant, prefix

DEFAULT_CONFIG = ROOT/'experiments/continuous_tracking/config.json'


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def quality(audit, target, thresholds):
    g = audit['grids']['0.1']
    errors = dict(mean_abs_error=abs(g['CL_mean_first3']-target), rmse=g['CL_rmse_first3'],
                  peak_abs_error=g['CL_peak_error_first3'])
    failed = [k for k, v in errors.items() if v > thresholds[k+'_max']]
    return dict(passed=not failed, failed=failed, **errors,
                status='TRACKING_OK' if not failed else 'SAFE_WITH_TRACKING_DEVIATION',
                deviation_required_for_safety_proven=False)


def build(config=DEFAULT_CONFIG):
    integrity.verify(ROOT)
    config = Path(config).resolve(strict=True)
    experiment = json.loads(config.read_text(encoding='utf-8'))
    if experiment['protocol'] != PROTOCOL:
        raise ValueError('Experimental protocol mismatch; legacy checkpoints are not accepted')
    for k in ('mean_abs_error_max', 'rmse_max', 'peak_abs_error_max'):
        v = experiment['tracking_quality'][k]
        if not np.isfinite(v) or v <= 0:
            raise ValueError('Invalid tracking-quality threshold')
    physical = runtime_config(ROOT)['physical']
    mission = Mission(ROOT, physical['mission'])
    db = AeroDatabase(ROOT/physical['snapshot'])
    plant = ContinuousPlant(CompiledPlant(mission, db))
    trajectory = yaml.safe_load((ROOT/physical['trajectory']).read_text(encoding='utf-8'))
    ocp = TrackingOCP(mission, db, trajectory, physical['ocp'], plant, experiment)
    sources = [*sorted((ROOT/'src').rglob('*.py')), *sorted((ROOT/'scripts').glob('*.py')), *sorted((ROOT/'configs').glob('*')),
               ROOT/integrity.SUMS, config]
    identity = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): integrity.sha256(p)
                for p in sources if p.is_file()}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return ocp, experiment, dict(protocol=PROTOCOL, sha256=digest, files=identity)


class Refused(RuntimeError):
    def __init__(self, reason, phase=None, attempts=None):
        super().__init__(reason)
        self.phase, self.attempts = phase, attempts or []


def solve_audited(ocp, context, guess, folder, name):
    """A warm-start failure may retry the unchanged cold initializer once."""
    attempts = []
    for attempt in range(2 if guess is not None else 1):
        started = time.perf_counter()
        plan = ocp.solve_context(context, guess if attempt == 0 else None)
        row = dict(context=context, solver=plan.summary(),
                   build_and_solve_wall_s=time.perf_counter()-started,
                   initializer='warm_reintegrated' if attempt == 0 and guess is not None else 'cold_equal_lift')
        arrays = {}
        if plan.success:
            row['audit'], arrays = ocp.plant.audit(context, plan.alpha_nodes, plan.times, context['CL_des'])
        attempts.append(row)
        np.savez_compressed(folder/f'{name}_{attempt}.npz', alpha_nodes=plan.alpha_nodes,
            times=plan.times, optimized_states=plan.states,
            **{f'{key}_{field}': arr[field] for key, arr in arrays.items() for field in ('x', 'samples')})
        write_json(folder/f'{name}.json', dict(attempts=attempts))
        if plan.success and row['audit']['valid']:
            return plan, row['audit'], arrays, attempts
    reason = 'TRUE_OCP_SOLVE_FAILED' if not plan.success else 'TRUE_OCP_PLAN_AUDIT_REJECTED'
    raise Refused(name+':'+reason,name,attempts)


def run_constant(out, cl=.025, max_decisions=400, config=DEFAULT_CONFIG):
    if cl not in CL_ACTIONS:
        raise ValueError('CL must be one of the unchanged 19 absolute actions')
    if not 1 <= max_decisions <= 400:
        raise ValueError('max_decisions must be between 1 and 400')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    ocp, experiment, identity = build(config)
    write_json(out/'identity.json', identity)
    write_json(out/'config.json', experiment)
    mission, db, plant = ocp.m, ocp.db, ocp.plant
    x = np.array(mission.data['initial_state']['x_h_V_gamma_s'], float)
    alpha = float(mission.data['initial_state']['alpha_rad'])
    z = Transition(db.index['G42'], db.index['G42'])
    t, reward, switches, previous = 0., 0., 0, None
    rows, reason = [], 'MAX_DECISIONS'
    solve_counts = {'pre': 0, 'final': 0, 'cold_retries': 0, 'optimizer_calls': 0}

    def count_attempts(phase, attempts):
        solve_counts[phase] += len(attempts)
        solve_counts['cold_retries'] += max(0,len(attempts)-1)
        solve_counts['optimizer_calls'] += sum(1+int('secondary_status' in a['solver']['stages']) for a in attempts)

    def context(transition):
        return dict(x=x.tolist(), alpha_prev=alpha, t=t, source=transition.source, target=transition.target,
                    fraction=transition.fraction(t), rate=.125 if transition.active(t) else 0., CL_des=cl)

    def snapshot():
        return dict(t=t, x=x.tolist(), alpha=alpha, transition=vars(z).copy(), reward=reward, switches=switches)

    def totals():
        duration = len(rows)*3.
        rmse = float(np.sqrt(sum(r['CL_rmse']**2*3 for r in rows)/duration)) if rows else None
        margins = {k: min(r['margins'][k] for r in rows) for k in rows[0]['margins']} if rows else {}
        return dict(protocol=PROTOCOL, CL=cl, completed_1200s=t == 1200. and reason == 'SUCCESS',
            stop_reason=reason, time_s=t, range_km=float((x[3]-mission.data['initial_state']['x_h_V_gamma_s'][3])/1000),
            reward=reward, switches=switches, steps=len(rows), wall_s=time.perf_counter()-started,
            CL_rmse=rmse, tracking_quality_failed_steps=sum(not r['quality']['passed'] for r in rows),
            max_boundary_alpha_jump_deg=max([abs(r['boundary_jump_deg']) for r in rows], default=0.),
            max_alpha_rate_degps=max([r['max_alpha_rate_degps'] for r in rows], default=0.),
            min_normalized_margins=margins, solver_calls=solve_counts, final_state=snapshot(),
            rows=rows, training_performed=False, surrogate_used=False,
            certificate_scope='continuous-alpha real Pre/Final plans; full-horizon .1/.5 audits and current-interval .1/.5 audits; not legacy H30 qualification')

    write_json(out/'result.json', totals())
    try:
        for step in range(max_decisions):
            tick = time.perf_counter()
            folder = out/f'decision_{step:04d}'
            folder.mkdir()
            write_json(folder/'before.json', snapshot())
            candidate = copy(z)
            candidate.settle(t)
            cpre = context(candidate)
            grid = np.r_[0., np.cumsum(ocp.grid(1200-t))]
            guess = None
            if previous is not None and experiment['warm_start_from_previous_final']:
                guess = np.interp(grid+3., previous.times, previous.alpha_nodes)
            pre, prea, pre_arrays, attempts = solve_audited(ocp, cpre, guess, folder, 'pre')
            count_attempts('pre',attempts)
            midpoint = (x+pre_arrays['0.5']['x'][5])/2
            mach = float(midpoint[1]/plant.atmosphere.speed_of_sound(midpoint[0]))
            choice = db.generate(mach, cl, current=candidate.target)
            if choice['geometry_index'] is None:
                raise Refused('GENERATOR_NO_FEASIBLE_GEOMETRY')
            changed = candidate.command(choice['geometry_index'], t)
            cfinal = context(candidate)
            final, full_audit, full_arrays, attempts = solve_audited(ocp, cfinal, pre.alpha_nodes, folder, 'final')
            count_attempts('final',attempts)
            aa, tt = prefix(final.alpha_nodes, final.times, 3.)
            audit, arrays = plant.audit(cfinal, aa, tt, cl)
            if not audit['valid']:
                raise Refused('CURRENT_INTERVAL_AUDIT_REJECTED')
            # The executed plan must be the audited prefix, including alpha at t=0.
            if not np.array_equal(arrays['0.5']['x'][-1], full_arrays['0.5']['x'][5]):
                raise Refused('AUDITED_PREFIX_ENDPOINT_MISMATCH')
            q = quality(audit, cl, experiment['tracking_quality'])
            end = arrays['0.5']['x'][-1].copy()
            boundary = float(np.rad2deg(aa[0]-alpha))
            stats = audit['grids']['0.1']
            row = dict(step=step, t0=t, t1=t+3., CL_cmd=cl,
                CL_mean=stats['CL_mean_first3'], CL_rmse=stats['CL_rmse_first3'], quality=q,
                alpha_nodes_deg=np.rad2deg(aa).tolist(), times=tt.tolist(), boundary_jump_deg=boundary,
                max_alpha_rate_degps=audit['maximum_alpha_rate_degps'],
                margins={k:min(stats['margins'][k], audit['grids']['0.5']['margins'][k]) for k in stats['margins']},
                generator=choice, predicted_flight_mach=mach, switch_started=changed,
                wall_s=time.perf_counter()-tick,
                pre_solve_s=pre.solve_s, final_solve_s=final.solve_s)
            write_json(folder/'execution_audit.json', dict(context=cfinal, audit=audit, quality=q))
            np.savez_compressed(folder/'executed.npz', alpha_nodes=aa, times=tt,
                **{f'{key}_{field}': arr[field] for key, arr in arrays.items() for field in ('x', 'samples')})
            # Commit only after every independent audit passed.
            ds = float(end[3]-x[3])
            x, alpha, t, z = end, float(aa[-1]), t+3., candidate
            z.settle(t)
            switches += int(changed)
            reward += mission.reward(ds, mission.outcome(t,x))['reward']
            previous = final
            rows.append(row)
            reason = mission.outcome(t,x) if t >= 1200 else 'MAX_DECISIONS'
            write_json(out/'checkpoint.json', dict(protocol=PROTOCOL, identity=identity['sha256'], state=snapshot(),
                previous_plan=dict(times=final.times, alpha_nodes=final.alpha_nodes), next_decision=step+1,
                note='Diagnostic checkpoint; no legacy replay, policy, optimizer, or weights'))
            write_json(out/'result.json', totals())
            print(json.dumps(clean({k:row[k] for k in ('step','t1','CL_mean','CL_rmse','quality','wall_s')})), flush=True)
            if t >= 1200:
                break
    except (Refused, RuntimeError, ValueError) as exc:
        if isinstance(exc,Refused) and exc.phase in ('pre','final'):
            count_attempts(exc.phase,exc.attempts)
        reason = str(exc)
        write_json(out/'stop_state.json', dict(reason=reason, state=snapshot(), last_decision_folder=str(folder)))
        print('STOP: '+reason, flush=True)
    except KeyboardInterrupt:
        reason = 'EXTERNAL_INTERRUPT'
        write_json(out/'stop_state.json', dict(reason=reason, state=snapshot()))
        raise
    finally:
        result = totals()
        write_json(out/'result.json', result)
    return result

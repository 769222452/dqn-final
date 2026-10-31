"""Nominal-policy lookahead and cross-period recovery without cached PASS reuse."""
from copy import deepcopy
import time

import numpy as np

from certified_flight.surrogate import OracleRequired
from certified_flight.transition import Transition
from continuous_surrogate.experiment import baseline, context_from_state, digest, write_json
from continuous_surrogate.safety import SafetyResult
from . import PROTOCOL
from .plan import create_recovery_plan, certify_tail, validate


class RecoveryController:
    def __init__(self, system, model, identity):
        self.system, self.model, self.identity = system, model, identity
        self.guard, self.plant, self.db = (system[k] for k in ('safety', 'plant', 'db'))
        self.physical = baseline(system)
        self.memory = None
        self.epoch = 0
        self.exit_passes = 0
        self.exit_confirmations = int(system['config'].get('recovery', {}).get('exit_confirmations', 2))
        if self.exit_confirmations < 1:
            raise ValueError('Positive recovery exit confirmations required')

    def state_dict(self):
        return deepcopy(dict(protocol=PROTOCOL, identity=self.identity, memory=self.memory,
                             epoch=self.epoch, exit_passes=self.exit_passes))

    def load_state_dict(self, data):
        if data.get('protocol') != PROTOCOL or data.get('identity') != self.identity:
            raise ValueError('Incompatible recovery checkpoint')
        memory = data['memory']
        epoch, exit_passes = int(data['epoch']), int(data['exit_passes'])
        if epoch != data['epoch'] or exit_passes != data['exit_passes']:
            raise ValueError('Recovery counters must be integers')
        if epoch < 0 or not 0 <= exit_passes < self.exit_confirmations:
            raise ValueError('Invalid recovery checkpoint counters')
        if memory is not None:
            validate(memory)
            if memory['identity'] != self.identity or memory['epoch'] != epoch:
                raise ValueError('Recovery memory identity/epoch differs from checkpoint')
        elif exit_passes:
            raise ValueError('Recovery exit confirmations without a recovery plan')
        # Loading is transactional: a rejected checkpoint changes no state.
        self.memory = deepcopy(memory)
        self.epoch, self.exit_passes = epoch, exit_passes

    def pre_mach(self, c):
        try:
            prediction, diagnostics = self.model.predict(c, 'pre')
        except OracleRequired as exc:
            prediction = self.physical(c, 'pre')
            diagnostics = dict(reason=exc.reason, supported=False, fallback='CONTINUOUS_PHYSICAL')
        return float(np.asarray(prediction).reshape(-1)[0]), diagnostics

    def prepare(self, state, cl):
        z = Transition(**state['transition'])
        z.settle(state['t'])
        reachable = self.system['reachability'].check(np.asarray(state['x']), state['alpha'], z,
                                                       state['t'], cl, enabled=True)
        cmd = float(reachable['CL_cmd'])
        c = context_from_state(dict(state, transition=vars(z)), cmd)
        c['CL_raw'] = float(cl)
        mach, pre = self.pre_mach(c)
        choice = self.db.generate(mach, cmd, current=z.target)
        switched = False
        if choice['geometry_index'] is not None:
            switched = z.command(choice['geometry_index'], state['t'])
        c = context_from_state(dict(state, transition=vars(z)), cmd)
        c['CL_raw'] = float(cl)
        try:
            future, post = self.model.predict(c, 'post')
            nodes = np.r_[state['alpha'], future]
            c['proposal_origin'] = 'LOCAL_POST_SURROGATE'
        except OracleRequired as exc:
            nodes = self.guard.candidate(c)
            post = dict(reason=exc.reason, supported=False, fallback='CONTINUOUS_PHYSICAL')
            c['proposal_origin'] = 'CONTINUOUS_PHYSICAL_ON_MODEL_REFUSAL'
        details = dict(reachable=reachable, pre=pre, post=post, mach=mach, generator=choice,
                       switched=bool(switched), proposal_nodes=nodes.tolist(), context=c,
                       fallbacks=[], events=[], ocp_called=False)
        for head, diag in (('PRE', pre), ('POST', post)):
            if not diag.get('supported', True):
                details['fallbacks'].append(head+'_PHYSICAL_ON_MODEL_REFUSAL')
        return c, z, details

    def nominal_witness(self, c, nodes):
        started = time.perf_counter()
        try:
            return self._nominal_witness(c, nodes)
        except (AttributeError, KeyError, TypeError, ValueError, IndexError, RuntimeError,
                FloatingPointError, OverflowError) as exc:
            diagnostics = dict(witness_protocol='continuous-nominal-policy-h30/1',
                status='NOMINAL_NUMERICAL_OR_DOMAIN_FAILURE', error=repr(exc),
                wall_s=time.perf_counter()-started, proof_status='UNKNOWN_NOT_CERTIFIED',
                future_candidate_witness=False, universal_infeasibility_proven=False,
                certificate_reused=False, selected_source=None)
            return SafetyResult(False, diagnostics['status'], None, None, None, None, diagnostics)

    def _nominal_witness(self, c, nodes):
        """Simulate the same Pre/Generator/Post/P1, with the raw CL request held.

        The dense state is advanced independently across decision boundaries.
        A current-only certificate, or an ideal equal-lift future, cannot exit
        recovery. No recovery or real OCP is allowed inside this witness.
        """
        started = time.perf_counter()
        deadline = started+self.guard.budget
        checked = self.guard.correct(c, nodes)
        diagnostics = dict(witness_protocol='continuous-nominal-policy-h30/1',
                           current=checked.diagnostics, future_steps=[],
                           reference_prediction='hold_current_raw_CL',
                           future_controller='same Pre/Generator/localPost/trackingPriority',
                           independent_dense_state=True, certificate_reused=False)

        def finish(status, passed=False):
            passed = passed and time.perf_counter() <= deadline
            if time.perf_counter() > deadline:
                status = 'SEARCH_BUDGET_EXHAUSTED'
            diagnostics.update(status=status, wall_s=time.perf_counter()-started,
                proof_status='VERIFIED_PASS' if passed else 'UNKNOWN_NOT_CERTIFIED',
                future_candidate_witness=passed, universal_infeasibility_proven=False,
                selected_source='TRACKING_PRIORITY_NOMINAL' if passed else None,
                safety_intervention=False,
                quality=checked.diagnostics.get('quality'))
            return SafetyResult(passed, status, checked.alpha_nodes if passed else None,
                checked.times if passed else None, checked.audit if passed else None,
                checked.arrays if passed else None, diagnostics)

        def recovery_trigger(result, at_time, phase):
            # A safe P1 result may already contain a physical constraint-avoidance
            # manoeuvre. That is not ordinary CL tracking and must not qualify
            # as a nominal-policy continuation/exit merely because its current
            # three-second audit passes. Use the existing independent physical
            # audit of the unchanged equal-lift candidate, with no new margins
            # or tracking-error threshold, to identify this mode transition.
            baseline = next((attempt for attempt in result.diagnostics.get('attempts', [])
                             if attempt.get('source') == 'CONTINUOUS_EQUAL_LIFT'), None)
            if baseline is None or baseline.get('audit', {}).get('valid') is not False:
                return None
            trigger = dict(reason='CONTINUOUS_EQUAL_LIFT_PHYSICAL_AUDIT_REJECTED',
                           time_s=float(at_time), phase=phase,
                           baseline_audit=deepcopy(baseline['audit']),
                           corrected_audit=deepcopy(result.audit),
                           potentially_feasible_with_recovery=True,
                           physical_infeasibility_proven=False,
                           hard_constraints_unchanged=True,
                           meaning='Nominal tracking requires a physical avoidance intervention; request a persistent recovery plan')
            diagnostics['recovery_trigger'] = trigger
            diagnostics['potentially_feasible_with_recovery'] = True
            return trigger

        if not checked.accepted:
            return finish('NOMINAL_CURRENT_'+checked.status)
        if recovery_trigger(checked, c['t'], 'current') is not None:
            return finish('NOMINAL_CURRENT_REQUIRES_SAFETY_RECOVERY')
        z = Transition(c['source'], c['target'], c['t']-8*c['fraction'])
        now = float(c['t'])+3.
        x = checked.arrays['0.5']['x'][-1].copy()
        dense_x = checked.arrays['0.1']['x'][-1].copy()
        alpha = float(checked.alpha_nodes[-1])
        end = min(c['t']+3.+self.guard.horizon, self.guard.duration)
        while now < end-1e-9:
            if time.perf_counter() > deadline:
                return finish('SEARCH_BUDGET_EXHAUSTED')
            z.settle(now)
            future_state = dict(x=x.tolist(), alpha=alpha, t=now, transition=vars(z))
            future_c, future_z, detail = self.prepare(future_state, c.get('CL_raw', c['CL_des']))
            future = self.guard.correct(future_c, detail['proposal_nodes'])
            entry = dict(t=now, context=future_c, generator=detail['generator'],
                         predicted_flight_mach=detail['mach'], status=future.status,
                         correction=future.diagnostics, audit=future.audit)
            diagnostics['future_steps'].append(entry)
            if not future.accepted:
                return finish('NOMINAL_FUTURE_'+future.status)
            trigger = recovery_trigger(future, now, 'future')
            if trigger is not None:
                entry['recovery_trigger'] = trigger
                return finish('NOMINAL_FUTURE_REQUIRES_SAFETY_RECOVERY')
            dense_c = dict(future_c, x=dense_x.tolist())
            dense_audit, dense_arrays = self.guard.audit(dense_c, future.alpha_nodes, future.times)
            entry['independent_dense_audit'] = dense_audit
            entry['alpha_nodes_rad'] = future.alpha_nodes.tolist()
            if not dense_audit['valid']:
                return finish('NOMINAL_INDEPENDENT_DENSE_REJECTED')
            x = future.arrays['0.5']['x'][-1].copy()
            dense_x = dense_arrays['0.1']['x'][-1].copy()
            alpha = float(future.alpha_nodes[-1])
            z = future_z
            now += 3.
        diagnostics.update(witness_duration_s=now-c['t'], future_horizon_s=now-c['t']-3.,
                           final_coarse_state=x.tolist(), final_dense_state=dense_x.tolist())
        return finish('NOMINAL_H30_WITNESS_FOUND', True)

    def _fresh_recovery(self, c, folder, detail):
        """Solve and certify a replacement tail; never fall back to a nominal PASS.

        A pending exit confirmation still requires recovery execution. If the
        previous tail has expired, the replacement must independently pass, or
        the decision stays UNKNOWN even if this is the first nominal H30 PASS.
        """
        started = time.perf_counter()
        detail['ocp_called'] = True
        detail['fallbacks'].append('TRUE_CONTINUOUS_FINAL_RECOVERY')

        def unknown(status, reason=None):
            diagnostics = dict(status=status, reason=reason,
                proof_status='UNKNOWN_NOT_CERTIFIED', selected_source=None,
                future_candidate_witness=False, universal_infeasibility_proven=False,
                exit_confirmations=self.exit_passes,
                required_exit_confirmations=self.exit_confirmations,
                wall_s=time.perf_counter()-started)
            return SafetyResult(False, status, None, None, None, None, diagnostics)

        try:
            record, plan = self.system['teacher'].solve_audited(c, 'post', output_dir=folder/'ocp_recovery')
            detail['teacher_record'] = record
            detail['events'].append(dict(event='REAL_OCP_RECOVERY', accepted=record['accepted'],
                                         reason=record.get('reason')))
            if not record['accepted']:
                return unknown('REAL_OCP_RECOVERY_NOT_ACCEPTED', record.get('reason'))
            fresh_epoch = self.epoch+1
            fresh_memory = create_recovery_plan(plan, c, self.identity, fresh_epoch)
            selected = certify_tail(self.guard, c, fresh_memory, self.identity, self.pre_mach)
            write_json(folder/'fresh_recovery_witness.json', dict(accepted=selected.accepted,
                status=selected.status, audit=selected.audit, diagnostics=selected.diagnostics))
            if selected.accepted:
                self.memory, self.epoch = fresh_memory, fresh_epoch
            else:
                detail['events'].append(dict(event='FRESH_RECOVERY_REJECTED', reason=selected.status))
            return selected
        except (AttributeError, KeyError, TypeError, ValueError, IndexError, RuntimeError,
                FloatingPointError, OverflowError) as exc:
            detail['events'].append(dict(event='FRESH_RECOVERY_ERROR', reason=repr(exc)))
            return unknown('REAL_OCP_RECOVERY_NUMERICAL_OR_DOMAIN_FAILURE', repr(exc))

    def decide(self, state, cl, folder):
        from pathlib import Path
        folder = Path(folder)
        c, z, detail = self.prepare(state, cl)
        nominal = self.nominal_witness(c, detail['proposal_nodes'])
        write_json(folder/'nominal_witness.json', dict(context=c, accepted=nominal.accepted,
                   status=nominal.status, audit=nominal.audit, diagnostics=nominal.diagnostics))
        tail = None
        if self.memory is not None:
            tail = certify_tail(self.guard, c, self.memory, self.identity, self.pre_mach)
            write_json(folder/'recovery_reaudit.json', dict(accepted=tail.accepted, status=tail.status,
                       audit=tail.audit, diagnostics=tail.diagnostics))
            if not tail.accepted:
                detail['events'].append(dict(event='RECOVERY_REVALIDATION_FAILED', reason=tail.status))
        if nominal.accepted:
            if self.memory is None:
                selected = nominal
            else:
                self.exit_passes += 1
                if self.exit_passes >= self.exit_confirmations:
                    detail['events'].append(dict(event='RECOVERY_EXIT_NOMINAL_H30', confirmations=self.exit_passes))
                    self.memory, self.exit_passes = None, 0
                    selected = nominal
                else:
                    detail['events'].append(dict(event='RECOVERY_EXIT_CONFIRMATION_PENDING', confirmations=self.exit_passes))
                    if tail is not None and tail.accepted:
                        selected = tail
                    else:
                        detail['events'].append(dict(event='RECOVERY_REPLAN_PENDING_EXIT',
                            reason=None if tail is None else tail.status,
                            confirmations=self.exit_passes))
                        selected = self._fresh_recovery(c, folder, detail)
        else:
            self.exit_passes = 0
            if tail is not None and tail.accepted:
                selected = tail
            else:
                selected = self._fresh_recovery(c, folder, detail)
        detail['selected_source'] = selected.diagnostics.get('selected_source')
        detail['nominal_status'] = nominal.status
        detail['controller_memory_sha256'] = digest(self.state_dict())
        return selected, z, c['CL_des'], detail, self.state_dict()

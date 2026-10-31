"""Bound, immutable dual-grid evidence replaces only duplicate commit integration."""
from copy import copy, deepcopy
from dataclasses import FrozenInstanceError
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from certified_flight.transition import Transition
from continuous_recovery.execution import CertifiedExecution, seal_execution, commit_certified
from continuous_recovery.experiment import bind_proof, commit_audited
from continuous_surrogate.experiment import context_from_state, digest
from continuous_surrogate.safety import SafetyResult


@pytest.fixture
def case():
    state = dict(x=[40000., 2400., -.01, 1000.], alpha=.05, t=6.,
                 transition=vars(Transition(0, 1, 0., 8.)), reward=.125, switches=1)
    transition = Transition(**state['transition'])
    nodes, times, cl = .05+np.arange(7)*.0001, np.arange(7)*.5, .025
    calls = []

    def audit(c, alpha, grid, target):
        calls.append((c, np.asarray(alpha).copy()))
        arrays, grids = {}, {}
        for key in ('0.1', '0.5'):
            dt = float(key)
            n, tt = round(3./dt), np.arange(round(3./dt))*dt
            # Separate deterministic paths deliberately have different endpoints.
            velocity = np.asarray([target, -.01*dt, .00001, 2000.+target])
            x = np.asarray(c['x'])[None]+(tt+dt)[:, None]*velocity[None]
            samples = np.zeros((12, 5*n))
            for i, tick in enumerate(tt):
                sample_t = np.asarray([tick, tick+dt/2, tick+dt/2, tick+dt, tick+dt])
                samples[:4, 5*i:5*i+5] = np.asarray(c['x'])[:, None]+velocity[:, None]*sample_t[None]
            indices = np.clip(np.searchsorted(grid, tt+1e-9, side='right')-1, 0, len(alpha)-2)
            arrays[key] = dict(x=x, samples=samples, times=tt, dt=dt,
                alpha=np.interp(tt, grid, alpha), alpha_end=np.interp(tt+dt, grid, alpha),
                alpha_rate=(np.diff(alpha)/np.diff(grid))[indices],
                valid=True, min_margin=.5, margins=[.5]*11)
            grids[key] = dict(valid=True, end_state=x[-1].tolist(), min_margin=.5,
                margins={'dynamic_pressure': .5}, CL_mean_first3=target+.001,
                CL_rmse_first3=.001, CL_peak_error_first3=.002)
        return dict(valid=True, continuous_at_entry=True, control_valid=True,
                    maximum_alpha_rate_degps=float(np.rad2deg(np.max(abs(np.diff(alpha)/np.diff(grid))))),
                    grids=grids), arrays

    mission = SimpleNamespace(data={'duration_s': 1200., 'limits': {'load_max': 3.}},
        outcome=lambda t, x: 'SUCCESS' if t == 1200. else 'INCOMPLETE',
        reward=lambda ds, outcome: {'reward': ds/2.4e6+(.1 if outcome == 'SUCCESS' else 0.)})
    system = dict(mission=mission, plant=SimpleNamespace(audit=audit), db=object(),
                  config={'safety': {'horizon': 30.}}, original={'physics': 'same'},
                  identity={'sha256': 'source'}, model_hash='frozen-model',
                  controller_protocol='continuous_recovery/1')
    au, arrays = audit(context_from_state(state, cl), nodes, times, cl)
    au['CL_target'] = cl
    checked = SafetyResult(True, 'NOMINAL_CLOSED_LOOP_PASS', nodes, times, au, arrays,
                           {'proof_status': 'VERIFIED_PASS'})
    return dict(system=system, state=state, transition=transition, identity='flight', cl=cl,
                witness={'manager': {'epoch': 3}, 'H30': True}, checked=checked, calls=calls)


def seal(c):
    return seal_execution(c['system'], c['state'], c['transition'], c['identity'],
                          c['cl'], c['witness'], c['checked'])


def commit(c, evidence, proof=None, nodes=None):
    return commit_certified(c['system'], c['state'], c['transition'],
        c['checked'].alpha_nodes if nodes is None else nodes,
        evidence.proof if proof is None else proof,
        c['identity'], c['cl'], c['witness'], evidence)


def test_bound_commit_matches_old_dual_grid_reintegration_bitwise(case):
    c = case
    checked = c['checked']
    old = bind_proof(c['state'], checked.alpha_nodes, c['identity'], c['transition'], c['cl'],
                     checked.arrays['0.5']['x'][-1], c['witness'])
    expected = commit_audited(c['system'], c['state'], c['transition'], checked.alpha_nodes,
                              old, c['identity'], c['cl'], c['witness'])
    evidence = seal(c)
    calls = len(c['calls'])
    def forbidden(*args, **kwargs):
        raise AssertionError('Duplicate execution integration must not run')
    c['system']['plant'].audit = forbidden
    c['system']['plant'].integrate = forbidden
    actual = commit(c, evidence)
    assert len(c['calls']) == calls
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert digest(actual[2]) == digest(expected[2])
    for grid in ('0.1', '0.5'):
        for field in ('x', 'samples', 'times', 'alpha', 'alpha_end', 'alpha_rate'):
            assert np.array_equal(actual[2][grid][field], expected[2][grid][field])
            assert actual[2][grid][field].dtype == expected[2][grid][field].dtype
    assert actual[0]['transition']['source'] == actual[0]['transition']['target'] == 1
    assert c['transition'].source == 0  # Settlement does not mutate the proposal.
    assert evidence.consumed


def test_execution_evidence_is_single_use(case):
    evidence = seal(case)
    assert copy(evidence) is evidence
    assert deepcopy(evidence) is evidence
    with pytest.raises(TypeError, match='cannot be checkpointed'):
        pickle.dumps(evidence)
    commit(case, evidence)
    with pytest.raises(ValueError, match='consumed'):
        commit(case, evidence)


def test_failed_commit_does_not_consume_evidence(case):
    evidence = seal(case)
    original = case['system']['mission'].reward
    case['system']['mission'].reward = lambda *args: (_ for _ in ()).throw(RuntimeError('accounting'))
    with pytest.raises(RuntimeError, match='accounting'):
        commit(case, evidence)
    assert not evidence.consumed
    case['system']['mission'].reward = original
    commit(case, evidence)


@pytest.mark.parametrize('field', ['state', 'transition', 'identity', 'cl', 'witness', 'nodes',
                                 'proof', 'model', 'config', 'original', 'source', 'mission', 'plant'])
def test_mismatched_inputs_rejected_without_integrating_or_consuming(case, field):
    evidence = seal(case)
    proof, nodes = evidence.proof, case['checked'].alpha_nodes.copy()
    if field == 'state':
        case['state']['x'][0] = np.nextafter(case['state']['x'][0], np.inf)
    elif field == 'transition':
        case['transition'].started += .001
    elif field == 'identity':
        case['identity'] = 'other'
    elif field == 'cl':
        case['cl'] = np.nextafter(case['cl'], np.inf)
    elif field == 'witness':
        case['witness']['manager']['epoch'] += 1
    elif field == 'nodes':
        nodes[3] = np.nextafter(nodes[3], np.inf)
    elif field == 'proof':
        proof['endpoint'][3] += 1.
    elif field == 'model':
        case['system']['model_hash'] = 'different-model'
    elif field == 'config':
        case['system']['config']['safety']['horizon'] = 33.
    elif field == 'original':
        case['system']['original']['physics'] = 'different'
    elif field == 'source':
        case['system']['identity']['sha256'] = 'different-source'
    elif field == 'mission':
        case['system']['mission'].data['limits']['load_max'] = 4.
    elif field == 'plant':
        case['system']['plant'] = SimpleNamespace(audit=lambda *args: pytest.fail('must not run'))
    before = len(case['calls'])
    with pytest.raises(ValueError, match='incompatible'):
        commit(case, evidence, proof, nodes)
    assert len(case['calls']) == before and not evidence.consumed


def test_sealed_arrays_and_proof_cannot_be_mutated_through_inputs_or_outputs(case):
    evidence = seal(case)
    frozen_endpoint = evidence.proof['endpoint'].copy()
    nodes = case['checked'].alpha_nodes.copy()
    case['checked'].arrays['0.5']['x'][-1] += 1e6
    case['checked'].arrays['0.1']['samples'][:] = np.nan
    case['checked'].audit['grids']['0.1']['margins']['dynamic_pressure'] = -99.
    case['checked'].alpha_nodes[:] = -99.
    detached_proof = evidence.proof
    detached_proof['endpoint'][:] = [0.]*4
    assert evidence.proof['endpoint'] == frozen_endpoint
    with pytest.raises(FrozenInstanceError):
        evidence._payload = b'altered'
    state, audit, arrays = commit(case, evidence, nodes=nodes)
    assert state['x'] == frozen_endpoint
    assert np.isfinite(arrays['0.1']['samples']).all()
    assert audit['grids']['0.1']['margins']['dynamic_pressure'] == .5
    arrays['0.5']['x'][-1] += 100.
    assert evidence.proof['endpoint'] == frozen_endpoint


@pytest.mark.parametrize('defect', ['unknown', 'false_pass', 'missing_dense', 'dense_rejected',
    'invalid_grid', 'wrong_times', 'nan_nodes', 'nan_arrays', 'wrong_endpoint',
    'wrong_initial', 'wrong_control', 'wrong_shape', 'control_rejected', 'wrong_cl', 'overrun'])
def test_invalid_or_incomplete_evidence_cannot_be_sealed(case, defect):
    c = case['checked']
    if defect == 'unknown':
        c.accepted = False
    elif defect == 'false_pass':
        c.diagnostics['proof_status'] = 'UNKNOWN_NOT_CERTIFIED'
    elif defect == 'missing_dense':
        del c.arrays['0.1']
    elif defect == 'dense_rejected':
        c.audit['grids']['0.1']['valid'] = False
    elif defect == 'invalid_grid':
        c.arrays['0.1']['valid'] = False
    elif defect == 'wrong_times':
        c.times[-1] = 30.
    elif defect == 'nan_nodes':
        c.alpha_nodes[2] = np.nan
    elif defect == 'nan_arrays':
        c.arrays['0.1']['samples'][6, 6] = np.nan
    elif defect == 'wrong_endpoint':
        c.audit['grids']['0.5']['end_state'][3] += 1.
    elif defect == 'wrong_initial':
        c.arrays['0.1']['samples'][0, 0] += .001
    elif defect == 'wrong_control':
        c.arrays['0.5']['alpha'][2] += .001
    elif defect == 'wrong_shape':
        c.arrays['0.5']['x'] = c.arrays['0.5']['x'][:-1]
    elif defect == 'control_rejected':
        c.audit['control_valid'] = False
    elif defect == 'wrong_cl':
        c.audit['CL_target'] = .04
    else:
        case['state']['t'] = 1198.
    with pytest.raises(ValueError):
        seal(case)


def test_unsealed_evidence_cannot_execute(case):
    with pytest.raises(TypeError, match='seal_execution'):
        CertifiedExecution()
    with pytest.raises(ValueError, match='Missing'):
        commit_certified(case['system'], case['state'], case['transition'], case['checked'].alpha_nodes,
                         {}, case['identity'], case['cl'], case['witness'], None)


def test_exact_mission_end_preserves_terminal_reward(case):
    case['state']['t'] = 1197.
    evidence = seal(case)
    state, _, _ = commit(case, evidence)
    expected = .125+(state['x'][3]-1000.)/2.4e6+.1
    # This is the exact original reward addition order, not a tolerance check.
    assert state['reward'] == .125+((state['x'][3]-1000.)/2.4e6+.1)
    assert np.isclose(state['reward'], expected)
    assert state['t'] == 1200.

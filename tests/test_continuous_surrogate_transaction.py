"""Certified continuous controls must be exactly the controls committed."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from certified_flight.transition import Transition
from continuous_surrogate.experiment import bind_proof, commit_audited, digest


@pytest.fixture
def transaction():
    state = dict(x=[40000., 2400., -.01, 100.], alpha=.05, t=0.,
                 transition=vars(Transition(0, 0)), reward=.3, switches=0)
    nodes = np.linspace(.05, .056, 7)
    z = Transition(0, 1, started=0., duration=8.)
    endpoint = np.array([39980., 2399., -.009, 7299.])
    plant = SimpleNamespace(valid=True, endpoint=endpoint.copy(), audit_calls=0)

    def audit(context, candidate, times, cl):
        plant.audit_calls += 1
        assert np.array_equal(times, np.arange(7)*.5)
        assert candidate[0] == context['alpha_prev']
        arrays = {key: dict(x=np.asarray([plant.endpoint]), alpha=np.asarray(candidate[:-1]),
                            alpha_end=np.asarray(candidate[1:])) for key in ('0.1', '0.5')}
        return dict(valid=plant.valid), arrays

    plant.audit = audit
    mission = SimpleNamespace(outcome=lambda t, x: 'INCOMPLETE',
                              reward=lambda ds, outcome: dict(reward=ds/2.4e6))
    witness = dict(accepted_witness=True, horizon_s=30., control_sequence=[dict(alpha_nodes_rad=nodes.tolist())])
    identity = 'new-model-config-code-identity'
    proof = bind_proof(state, nodes, identity, z, .025, endpoint, digest(witness))
    return dict(system=dict(plant=plant, mission=mission), state=state, z=z, nodes=nodes,
                identity=identity, proof=proof, witness=witness, endpoint=endpoint)


def commit(case, **changes):
    case = dict(case, **changes)
    return commit_audited(case['system'], case['state'], case['z'], case['nodes'], case['proof'],
                          case['identity'], case.get('cl', .025), case['witness'])


def test_commit_uses_end_alpha_and_preserves_caller_state(transaction):
    case = transaction
    before = deepcopy(case['state'])
    before_transition = deepcopy(vars(case['z']))
    state, _, _ = commit(case)
    assert state['alpha'] == case['nodes'][-1]
    assert state['alpha'] != case['nodes'][-2]  # Interval-start alpha is not the endpoint.
    assert state['t'] == 3.
    assert np.array_equal(state['x'], case['endpoint'])
    assert state['reward'] == pytest.approx(.3+(7299.-100.)/2.4e6)
    assert case['state'] == before
    assert vars(case['z']) == before_transition
    assert case['system']['plant'].audit_calls == 1


@pytest.mark.parametrize('changed', ['state', 'transition', 'cl', 'identity', 'nodes', 'witness'])
def test_stale_or_mismatched_proof_cannot_commit(transaction, changed):
    case = transaction
    before = deepcopy(case['state'])
    edits = {}
    if changed == 'state':
        edits['state'] = dict(case['state'], t=3.)
    elif changed == 'transition':
        edits['z'] = Transition(0, 2, started=0., duration=8.)
    elif changed == 'cl':
        edits['cl'] = .03
    elif changed == 'identity':
        edits['identity'] = 'different-model-or-control-protocol'
    elif changed == 'nodes':
        edits['nodes'] = case['nodes'].copy()
        edits['nodes'][-1] += .001
    else:
        edits['witness'] = dict(case['witness'], accepted_witness=False)
    with pytest.raises(ValueError):
        commit(case, **edits)
    assert case['state'] == before
    assert case['system']['plant'].audit_calls == 0


def test_independent_audit_failure_does_not_commit(transaction):
    case = transaction
    before = deepcopy(case['state'])
    case['system']['plant'].valid = False
    with pytest.raises(ValueError, match='AUDIT_REJECTED'):
        commit(case)
    assert case['state'] == before


def test_reintegrated_endpoint_must_equal_certificate(transaction):
    case = transaction
    before = deepcopy(case['state'])
    case['system']['plant'].endpoint = case['endpoint']+np.array([0., 0., 0., .01])
    with pytest.raises(ValueError):
        commit(case)
    assert case['state'] == before


def test_transition_completion_keeps_exact_eight_second_clock(transaction):
    case = transaction
    case['state'] = dict(case['state'], t=6.)
    case['proof'] = bind_proof(case['state'], case['nodes'], case['identity'], case['z'], .025,
                                case['endpoint'], digest(case['witness']))
    state, _, _ = commit(case)
    assert state['t'] == 9.
    assert state['transition']['source'] == state['transition']['target'] == 1
    assert state['transition']['started'] == 0.
    assert state['transition']['duration'] == 8.

"""Unit tests of the learning components (no flight simulation)."""
import json
from pathlib import Path

import numpy as np
import pytest

from rollout_improvement import ensemble, features, policy, rollouts, training

DATA = Path(__file__).resolve().parent/'data'


class FakeEnv:
    def __init__(self, t=0., h=40000., V=2400., gamma=0., alpha=.05, prev=None):
        self.x = np.array([h, V, gamma, 0.])
        self.alpha, self.t = alpha, t
        self.previous_cl_raw = 0. if prev is None else prev
        self.history_valid = prev is not None


def test_features_scaling():
    f = features.features(FakeEnv(t=600., h=50000., V=3200., gamma=.01, alpha=.1, prev=.08))
    np.testing.assert_allclose(f, [1., 1., .1, 1., .5, 1.], rtol=1e-12)
    assert features.features(FakeEnv())[5] == 0.          # CL 0.04 before the first decision


def test_parse_inputs():
    assert features.parse_inputs('h,V,gamma,t') == [0, 1, 2, 4]
    assert features.parse_inputs('all') is None
    assert features.parse_inputs('prev,alpha,t,gamma,V,h') is None
    with pytest.raises(ValueError):
        features.parse_inputs('h,h')


def test_preference_ranks_closest_action_first():
    order = np.argsort(-policy.preference(.025))
    assert order[0] == 5 and set(order[1:3]) == {4, 6}
    assert order[1] == 4                                   # ties resolved towards the lower index


def test_chain_constant_and_time_override():
    f = features.features(FakeEnv(t=96.))
    assert policy.chain_action([], f) == policy.INITIAL_ACTION
    chain = [policy.constant_stage(.025)]
    assert policy.chain_action(chain, f) == 5
    entry = dict(t=96., features=f.tolist(), action=6, base_action=5, advantage=1e-5)
    chain.append(dict(kind='override', entries=[entry], match='t'))
    assert policy.chain_action(chain, f) == 6
    assert policy.chain_action(chain, features.features(FakeEnv(t=108.))) == 5


def test_state_gated_override_requires_matching_state():
    f = features.features(FakeEnv(t=96.))
    stage = dict(kind='override', tol=.05, entries=[dict(t=96., features=f.tolist(), action=9, base_action=8)])
    assert policy.chain_action([stage], f) == 9
    assert policy.chain_action([stage], features.features(FakeEnv(t=96., h=41000.))) == 8


def test_model_stage_respects_trust_window_and_margin():
    zero = [[[0.]*19 for _ in range(4)], [0.]*19]
    bias = [0.]*19
    bias[10] = 3.                                          # +3e-4 predicted for CL 0.05 (two classes up)
    member = [(zero[0], bias)]
    stage = dict(kind='model', members=[member]*5, margin=1., times=[96.], window=48., inputs=[0, 1, 2, 4])
    assert policy.chain_action([stage], features.features(FakeEnv(t=96.))) == 10
    assert policy.chain_action([stage], features.features(FakeEnv(t=200.))) == policy.INITIAL_ACTION
    assert policy.chain_action([dict(stage, margin=3.)], features.features(FakeEnv(t=96.))) == policy.INITIAL_ACTION


def test_policy_holds_action_and_replays_schedule():
    env = FakeEnv()
    pi = policy.ChainPolicy([], env, override=(12., 3), schedule={0: 8, 12: 8, 24: 1})
    for t, expected in ((0., 8), (3., 8), (12., 3), (21., 3), (24., 1)):
        env.t = t
        assert pi.decide() == expected
    assert pi.decisions == [(0., 8), (12., 3), (24., 1)]


def test_alternatives_alternate_radius():
    assert rollouts.alternatives(8, 0.) == [7, 9]
    assert rollouts.alternatives(8, 48.) == [7, 9]
    assert rollouts.alternatives(8, 96.) == [6, 10]
    assert rollouts.alternatives(8, 96., sweep=1) == [7, 9]
    assert rollouts.alternatives(0, 0.) == [1]
    radius = {t: abs(rollouts.alternatives(8, t)[0]-8) for t in training.state_times(0, 25, 48.)}
    assert [radius[t] for t in (0., 48., 96., 144., 192., 1152.)] == [1, 1, 2, 2, 1, 1]


def test_state_times_rotate_and_cover_last_decision():
    assert training.state_times(0, 25, 48.)[:3] == [0., 48., 96.]
    assert training.state_times(3, 25, 48.)[-1] == 1188.
    covered = {t for i in range(4) for t in training.state_times(i, 25, 48.)}
    assert covered == {12.*k for k in range(100)}


def test_margin_grid():
    assert training.margin_grid(14) == [.25, .5, .75, 1., 1.25, 1.5]
    assert training.margin_grid(1) == [.25, .5, 1.]


def test_label_matrix_clips_and_averages():
    labels = [dict(features=[0.]*6, action=3, advantage=1e-4), dict(features=[0.]*6, action=3, advantage=3e-4),
              dict(features=[1.]*6, action=4, advantage=-1.)]
    X, A, M, W = ensemble.label_matrix(labels, weights=[1., 1., 1.])
    assert M.sum() == 2 and A[0, 3] == pytest.approx(2.) and A[1, 4] == -ensemble.CLIP


def test_retained_labels_decay():
    archive = [dict(v=v) for v in range(10)]
    kept, w = ensemble.retained_labels(archive, 9, keep=3, decay=.5)
    assert [x['v'] for x in kept] == [7, 8, 9] and w == [.25, .5, 1.]


def test_stop_rule():
    clean = dict(accepted=False, positive_pairs=0, invalid=0)
    busy = dict(accepted=False, positive_pairs=2, invalid=0)
    assert training.stop_reason([clean]*7, 4) is None
    assert 'clean sweeps' in training.stop_reason([clean]*8, 4)
    assert 'no candidate accepted' in training.stop_reason([busy]*12, 4)
    assert training.stop_reason([busy]*11+[dict(busy, accepted=True)], 4) is None


def test_override_sets():
    entries = [dict(t=float(k)) for k in range(5)]
    assert [len(s) for s in training.override_sets(entries)] == [1, 2, 3, 4, 5]


def test_ensemble_fit_reproduces_reference():
    """The fit of the first iteration is deterministic (seeded bootstrap and initialisation).

    The reference was recorded with hidden layers (16, 16)."""
    ref = json.loads((DATA/'first_iteration.json').read_text(encoding='utf-8'))
    labels = [dict(x, v=0) for x in ref['labels']]
    kept, weights = ensemble.retained_labels(labels, 0)
    members, diag = ensemble.fit_ensemble(kept, ref['seed'], weights, ref['inputs'], hidden=(16, 16))
    assert diag['states'] == 25 and diag['labels'] == 75
    assert len(members) == len(ref['members']) and all(len(m) == len(e) == 3 for m, e in zip(members, ref['members']))
    for got, expected in zip(members, ref['members']):
        for (W, b), (We, be) in zip(got, expected):
            np.testing.assert_allclose(W, We, rtol=0, atol=1e-12)
            np.testing.assert_allclose(b, be, rtol=0, atol=1e-12)


def test_default_network_shape():
    ref = json.loads((DATA/'first_iteration.json').read_text(encoding='utf-8'))
    labels = [dict(x, v=0) for x in ref['labels']]
    kept, weights = ensemble.retained_labels(labels, 0)
    members, _ = ensemble.fit_ensemble(kept, ref['seed'], weights, ref['inputs'])
    assert [np.shape(W) for W, _ in members[0]] == [(4, 64), (64, 64), (64, 64), (64, 64), (64, 19)]
    assert ensemble.parse_hidden('64,64,64,64') == ensemble.HIDDEN == (64, 64, 64, 64)
    with pytest.raises(ValueError):
        ensemble.parse_hidden('16,0')


def test_prediction_check_on_reference_labels():
    ref = json.loads((DATA/'first_iteration.json').read_text(encoding='utf-8'))
    check = ensemble.prediction_check(ref['members'], ref['labels'], 1e-5, ref['inputs'])
    assert check['pairs'] == 50 and -1 <= check['spearman'] <= 1

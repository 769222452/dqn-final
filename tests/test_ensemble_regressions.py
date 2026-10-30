from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from rollout_improvement.ensemble import bootstrap_indices, lower_confidence_bound
from rollout_improvement.policy import chain_action, prepare_chain


def test_bootstrap_keeps_identical_model_inputs_out_of_oob():
    X = np.array([[1.,2.],[1.,2.],[3.,4.],[5.,6.],[5.,6.]])
    for seed in range(20):
        boot,oob = bootstrap_indices(X,np.random.default_rng(seed))
        assert not {tuple(x) for x in X[boot]} & {tuple(x) for x in X[oob]}


def test_unique_inputs_preserve_original_bootstrap_exactly():
    X = np.arange(28).reshape(7,4)
    old = np.random.default_rng(17)
    new = np.random.default_rng(17)
    for _ in range(5):
        expected = old.integers(0,len(X),len(X))
        boot,oob = bootstrap_indices(X,new)
        assert np.array_equal(boot,expected)
        assert np.array_equal(oob,np.setdiff1d(np.arange(len(X)),np.unique(expected)))


def test_prepared_policy_is_bit_identical_and_checkpoint_unchanged():
    ref = json.loads((Path(__file__).parent/'data/first_iteration.json').read_text(encoding='utf-8'))
    stage = dict(kind='model',members=ref['members'],inputs=ref['inputs'],margin=.5,times=[0.,1200.],window=1200.)
    chain = [stage]
    original = deepcopy(chain)
    ready = prepare_chain(chain)
    for label in ref['labels']:
        f = np.array(label['features'])
        expected = lower_confidence_bound(stage['members'],f,stage['inputs'])
        got = lower_confidence_bound(ready[0]['members'],f,stage['inputs'])
        assert all(np.array_equal(a,b) for a,b in zip(expected,got))
        assert chain_action(chain,f) == chain_action(ready,f)
    assert chain == original
    assert not ready[0]['members'][0][0][0].flags.writeable

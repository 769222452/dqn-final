"""Exact flight state: the fields a transaction copies and the record certificates compare."""
from copy import deepcopy

# Fields copied between an environment and its candidate copies.
STATE_FIELDS = ('x', 'alpha', 't', 'done', 'transition', 'total_reward',
                'switches', 'last_outcome', 'previous_cl_raw', 'previous_cl_cmd',
                'previous_cl_actual', 'history_valid', 'decision_index',
                'last_transition_completed')

HISTORY_FIELDS = ('previous_cl_raw', 'previous_cl_cmd', 'previous_cl_actual',
                  'history_valid', 'decision_index', 'last_transition_completed')


def capture_state(env):
    """JSON-serialisable double-precision state with full transition and history."""
    state = {key: deepcopy(getattr(env, key)) for key in
             ('alpha', 't', 'done', 'total_reward', 'switches', 'last_outcome', *HISTORY_FIELDS)}
    state.update(x=env.x.tolist(), transition=vars(env.transition).copy(),
                 control_contract=env.control_contract)
    return state

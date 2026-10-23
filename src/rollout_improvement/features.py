"""State features seen by the advantage networks."""
import numpy as np

FEATURE_NAMES = ('h', 'V', 'gamma', 'alpha', 't', 'prev')
#: Network inputs of the proposed method: flight state and time, without the policy's own action feedback
#: (angle of attack and previous lift-coefficient command).
STATE_INPUTS = 'h,V,gamma,t'


def features(env):
    """Scaled features of the current state.

    altitude, speed, flight-path angle, angle of attack, mission time, previous CL target (0.04 before the
    first decision).
    """
    x = env.x
    prev = env.previous_cl_raw if env.history_valid and env.previous_cl_raw is not None else .04
    return np.array([(x[0]-40000)/10000, (x[1]-2400)/800, x[2]/.1, env.alpha/.1, env.t/1200, (prev-.04)/.04], float)


def parse_inputs(text):
    """Feature subset from a comma-separated list, e.g. 'h,V,gamma,t' -> [0, 1, 2, 4].

    Returns None for all six features.
    """
    if text is None or text == 'all':
        return None
    idx = [FEATURE_NAMES.index(x.strip()) for x in text.split(',')]
    if len(set(idx)) != len(idx) or not idx:
        raise ValueError('duplicate or empty feature list')
    return sorted(idx) if sorted(idx) != list(range(len(FEATURE_NAMES))) else None


def input_names(inputs):
    return [FEATURE_NAMES[j] for j in (inputs if inputs is not None else range(len(FEATURE_NAMES)))]

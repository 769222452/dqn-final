"""Inner-loop OCP objectives of the angle-of-attack study (experiments/aoa_fix/README.md).

v16         V16 runtime: CL tracked as a 3 s mean ('mean_3s'), boundary jump from the executed alpha free.
consistent  CL tracked pointwise over the whole 60 s horizon with one stationary weight
            ('pointwise_horizon', weight frozen at 20) and the first command's jump from the executed
            alpha charged by the control-rate cost, in the OCP and in the post corrector.
aligned     consistent, with the integrated heat-load penalty removed (heat_weight 0). The inner loop then
            optimises the mission objective (range) only. The heat-flux limit stays a hard constraint and
            the heat load is reported (experiments/improve/README.md, Phase 1).

objective_runtime_config() turns configs/runtime.json into the runtime configuration of an objective,
optionally with a surrogate fitted to that objective's teacher labels and its qualification report.
"""
from copy import deepcopy
import json
from pathlib import Path

from .contracts import mechanics_digest

OBJECTIVES = dict(
    v16=dict(),
    consistent=dict(tracking_mode='pointwise_horizon', boundary_smoothing=True),
    aligned=dict(tracking_mode='pointwise_horizon', boundary_smoothing=True, heat_weight=0.),
)
POINTWISE = ('consistent', 'aligned')
FROZEN_HORIZON_WEIGHT = 20.
CONTRACT_SCHEME = 'rl298-consistent/2'


def objective_ocp(ocp, objective, horizon_weight=FROZEN_HORIZON_WEIGHT, curvature=0.):
    out = dict(ocp, **OBJECTIVES[objective])
    if objective in POINTWISE:
        out['horizon_tracking_weight'] = float(horizon_weight)
        out['second_difference_weight'] = float(curvature)
    return out


SAFETY_VARIANTS = dict(minimal_first=dict(lookahead_offset_order='minimal_first'))


def objective_safety(safety, objective, variant=None):
    out = deepcopy(safety)
    if objective in POINTWISE:
        out['boundary_smoothing'] = True
    if variant:
        out.update(SAFETY_VARIANTS[variant])
    return out


def objective_runtime_config(config, objective, surrogate=None, qualification=None,
                             horizon_weight=FROZEN_HORIZON_WEIGHT, curvature=0., safety_variant=None):
    """Runtime configuration of an objective.

    surrogate: repository-relative controller.npz fitted to this objective (default: the V16 surrogate,
    valid only for objective 'v16'); qualification: its AB_metrics.json (default: next to it in
    data/lookahead_<name>/ as written by scripts/aoa_qualify_surrogate.py).
    """
    out = deepcopy(config)
    out['physical']['ocp'] = objective_ocp(config['physical']['ocp'], objective, horizon_weight, curvature)
    out['safety_options'] = objective_safety(config['safety_options'], objective, safety_variant)
    if surrogate is not None:
        root = Path(__file__).resolve().parents[2]
        meta = json.loads((root/surrogate).with_suffix('.json').read_text(encoding='utf-8'))
        out['surrogate'] = str(surrogate)
        out['model_sha256'] = meta['model_sha256']
        out['safety_options']['lookahead_qualification_report'] = str(
            qualification or Path(surrogate).parent.with_name(
                'lookahead_'+Path(surrogate).parent.name.removeprefix('surrogate_'))/'AB_metrics.json')
    elif objective != 'v16':
        raise ValueError('The V16 surrogate was fitted to the v16 objective; pass the refitted surrogate')
    return out


def contract(physical):
    return CONTRACT_SCHEME+':'+mechanics_digest(physical)

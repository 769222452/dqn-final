"""Construction of the certified flight from configs/runtime.json."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import yaml

from . import integrity
from .aero import AeroDatabase
from .certification import CL_ACTIONS, Certifier
from .contracts import matching_contract
from .environment import CertifiedFlightEnvironment
from .mission import Mission
from .ocp import TwoStageOCP
from .plant import CompiledPlant
from .serialization import write_json
from .surrogate import SurrogateController

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = 'configs/runtime.json'
RUNTIME_KEYS = {'description', 'surrogate', 'model_sha256', 'ood_physics_fallback', 'exact_integration_memo',
                'dead_end_retry', 'safety_options', 'physical'}


def runtime_config(root=ROOT):
    """Load and validate configs/runtime.json."""
    config = json.loads((Path(root)/RUNTIME).read_text(encoding='utf-8'))
    if set(config) != RUNTIME_KEYS:
        raise ValueError('runtime.json keys must be exactly '+', '.join(sorted(RUNTIME_KEYS)))
    if not isinstance(config['ood_physics_fallback'], bool) or not isinstance(config['exact_integration_memo'], bool):
        raise ValueError('ood_physics_fallback and exact_integration_memo must be booleans')
    retry = config['dead_end_retry']
    if set(retry) != {'enabled', 'wall_s', 'max_evaluations'}:
        raise ValueError('Invalid dead_end_retry schema')
    if (not isinstance(retry['enabled'], bool) or not np.isfinite(retry['wall_s'])
            or retry['wall_s'] < 3 or not isinstance(retry['max_evaluations'], int)
            or retry['max_evaluations'] < 12):
        raise ValueError('Retry may not shrink the original search budget')
    physical = config['physical']
    if physical['cl_actions'] != CL_ACTIONS:
        raise ValueError('physical.cl_actions must equal CL_ACTIONS')
    if physical['transition']['duration_s'] != 8.:
        raise ValueError('The plant and lookahead assume an 8 s transition')
    return config


def build_flight(folder, label, initial_offset=None, config=None):
    """Build the certified flight environment and its certifier.

    folder: output folder (created); proofs go to folder/'proofs', the jsonl logs
    to folder, oracle events and artifacts to folder/'oracle'.
    label: stored in certifier.tag['label'].
    initial_offset: optional (dh_m, dV_mps) added to the initial altitude and speed.
    config: optional runtime configuration (default configs/runtime.json), e.g. from
    certified_flight.objectives.objective_runtime_config.
    Returns (env, certifier).
    """
    integrity.verify(ROOT)
    config = runtime_config(ROOT) if config is None else deepcopy(config)
    physical = config['physical']
    safety = deepcopy(config['safety_options'])
    mission = Mission(ROOT, physical['mission'])
    db = AeroDatabase(ROOT/physical['snapshot'])
    plant = CompiledPlant(mission, db)
    trajectory_path = mission.path(physical['trajectory'])
    controller = SurrogateController.load(ROOT/config['surrogate'], db, mission, physical,
                                          {'mission_sha256': mission.config_path,
                                           'trajectory_sha256': trajectory_path})
    traj = yaml.safe_load(trajectory_path.read_text(encoding='utf-8'))
    oracle = TwoStageOCP(mission, db, traj, physical['ocp'], plant)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    def event_sink(event):
        write_json(folder/'oracle'/f"a{event['attempt']:06d}"/'event.json', event)

    def artifact_sink(event, head, plan, arrays):
        target = folder/'oracle'/f"a{event['attempt']:06d}"/head
        target.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target/'solver_plan.npz', alpha=plan.alpha, times=plan.times, x=plan.x,
                            dual_x=plan.dual.get('x', []), dual_g=plan.dual.get('g', []))
        for dt, values in arrays.items():
            np.savez_compressed(target/f'audit_dt{dt}.npz', **{k: values[k] for k in ('x', 'samples', 'alpha', 'times')})

    env = CertifiedFlightEnvironment(plant, controller, physical, oracle, safety, event_sink=event_sink,
                                     artifact_sink=artifact_sink, ood_physics_fallback=config['ood_physics_fallback'])
    guard = env.lookahead_guard
    assert guard is not None and guard.budget == 30. and guard.horizon == 30. and guard.max_evaluations == 12
    assert env.cl_actions == CL_ACTIONS
    assert controller.metadata['model_sha256'] == config['model_sha256']
    if initial_offset is not None:
        dh, dv = initial_offset
        env.x[0] += float(dh)
        env.x[1] += float(dv)
    tag = dict(label=label, control_contract=matching_contract(controller.metadata['control_contract'], physical),
               model_sha256=config['model_sha256'], physical=physical, safety_options=safety,
               ood_physics_fallback=config['ood_physics_fallback'], dead_end_retry=config['dead_end_retry'],
               exact_integration_memo=config['exact_integration_memo'],
               search=dict(order='descending_preference', first_pass_then_all=True))
    if initial_offset is not None:
        tag['initial_offset_h_V'] = [float(dh), float(dv)]
    certifier = Certifier(env, folder/'proofs', tag, memo=config['exact_integration_memo'],
                          retry_options=config['dead_end_retry'])
    return env, certifier

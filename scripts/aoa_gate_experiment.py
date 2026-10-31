"""Gate experiment for the angle-of-attack sawtooth: true OCP as the inner controller.

No surrogate is in the command path, so the experiment isolates the effect of the inner-loop
OCP objective. Every 3 s decision: CL screen -> pre OCP -> generator -> post OCP -> V16 post
corrector -> V16 30 s look-ahead guard (future policies use the unchanged V16 surrogate)
-> 0.1/0.5 s audits. A refused CL request falls back to the nearest certifiable action, as
in V16 certification. Objectives:

  v16         tracking_mode mean_3s (V16 runtime.json), boundary jump free
  consistent  tracking_mode pointwise_horizon (CL penalised pointwise over the whole
              60 s horizon) + boundary-aware rate cost (OCP and corrector)

    python scripts/aoa_gate_experiment.py --objective consistent --cl 0.025 --out runs/aoa/c025
    python scripts/aoa_gate_experiment.py --objective v16 --chain experiments/aoa_fix/policies/final_seed101.json \
        --offset -738 -63.1 --out runs/aoa/v16_seed101_case

The consistent objective's CL weight is frozen at 20 (calibrated on the nominal 0-120 s
development flight to match V16's executed CL error; see experiments/aoa_fix/README.md).
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse  # noqa: E402
from copy import deepcopy  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import subprocess  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'/'analysis'))

from certified_flight import CL_ACTIONS, integrity  # noqa: E402
from certified_flight.aero import AeroDatabase  # noqa: E402
from certified_flight.build import runtime_config  # noqa: E402
from certified_flight.corrector import PostCorrector  # noqa: E402
from certified_flight.environment import FlightEnvironment  # noqa: E402
from certified_flight.lookahead import LookaheadGuard  # noqa: E402
from certified_flight.mission import Mission  # noqa: E402
from certified_flight.ocp import OracleController, TwoStageOCP  # noqa: E402
from certified_flight.plant import CompiledPlant  # noqa: E402
from certified_flight.serialization import clean, write_json  # noqa: E402
from certified_flight.surrogate import OracleRequired, SurrogateController  # noqa: E402
from rollout_improvement.policy import ChainPolicy, constant_stage  # noqa: E402
from aoa_metrics import flight_metrics  # noqa: E402

OBJECTIVES = dict(
    v16=dict(),
    consistent=dict(tracking_mode='pointwise_horizon', boundary_smoothing=True),
)


class OCPInnerLoop:
    """Controller interface backed by a fresh pre/post OCP pair per decision."""

    def __init__(self, ocp, control_contract):
        self.ocp, self.control_contract = ocp, control_contract
        self.metadata = {'control_contract': control_contract}
        self.oracle, self.last = None, {}

    def predict(self, context, head):
        if head == 'pre':
            self.oracle = OracleController(self.ocp, dict(solver_calls={'pre': 0, 'post': 0}, phases={}))
        out = self.oracle.predict(context, head)
        self.last[head] = self.oracle.record['phases'][head]['solver']
        return out


FROZEN_HORIZON_WEIGHT = 20.


def source_revision():
    try:
        rev = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = bool(subprocess.run(['git', '-C', str(ROOT), 'status', '--porcelain', '--', 'src', 'scripts',
                                     'configs', 'data'], capture_output=True, text=True).stdout.strip())
        return dict(commit=rev, uncommitted_changes=dirty)
    except (OSError, subprocess.CalledProcessError):
        return dict(commit=None, uncommitted_changes=None)


def build(objective, horizon_weight, curvature, offset=None):
    integrity.verify(ROOT)
    config = runtime_config(ROOT)
    physical = config['physical']
    safety = deepcopy(config['safety_options'])
    mission = Mission(ROOT, physical['mission'])
    db = AeroDatabase(ROOT/physical['snapshot'])
    plant = CompiledPlant(mission, db)
    trajectory_path = mission.path(physical['trajectory'])
    surrogate = SurrogateController.load(ROOT/config['surrogate'], db, mission, physical,
                                         {'mission_sha256': mission.config_path,
                                          'trajectory_sha256': trajectory_path})
    ocp_cfg = dict(physical['ocp'], **OBJECTIVES[objective])
    if objective == 'consistent':
        ocp_cfg['horizon_tracking_weight'] = horizon_weight
        ocp_cfg['second_difference_weight'] = curvature
        safety['boundary_smoothing'] = True
    ocp = TwoStageOCP(mission, db, yaml.safe_load(trajectory_path.read_text(encoding='utf-8')), ocp_cfg, plant)
    inner = OCPInnerLoop(ocp, surrogate.control_contract)
    corrector = PostCorrector(plant, safety)
    env = FlightEnvironment(plant, inner, physical, safety, post_corrector=corrector)
    # Same guard as V16 (its future candidate policies use the V16 surrogate).
    env.lookahead_guard = LookaheadGuard(plant, surrogate, physical, safety, corrector)
    env.lookahead_qualification = LookaheadGuard.qualify(plant, surrogate, safety)
    if offset is not None:
        # Same convention as certified_flight.build_flight(initial_offset=...).
        env.x[0] += float(offset[0])
        env.x[1] += float(offset[1])
    return env, inner, dict(objective=objective, ocp=ocp_cfg, safety=safety,
                            initial_offset_h_V=None if offset is None else [float(v) for v in offset])


def run(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    env, inner, setup = build(a.objective, a.horizon_weight, a.curvature, a.offset)
    if a.chain:
        chain = json.loads(Path(a.chain).read_text(encoding='utf-8-sig'))['chain']
        policy_name = str(a.chain)
    else:
        chain = [constant_stage(a.cl)] if a.cl != .04 else []
        policy_name = f'constant CL {a.cl:g}'
    policy = ChainPolicy(chain, env)
    write_json(out/'setup.json', dict(setup, policy=policy_name, max_decisions=a.max_decisions,
                                      source=source_revision(), python=sys.version,
                                      chain=chain if a.chain else None))
    log = (out/'steps.jsonl').open('w', encoding='utf-8')
    infos, stop, started = [], 'MAX_DECISIONS', time.perf_counter()
    for k in range(a.max_decisions):
        tick = time.perf_counter()
        preference = policy.preference()
        requested = int(np.argmax(preference))
        info, refusals = None, []
        for action in [int(x) for x in np.argsort(-preference)][:a.fallback_actions]:
            try:
                _, _, done, info = env.step(action)
                break
            except (OracleRequired, ValueError) as exc:
                refusals.append(dict(action=action, reason=getattr(exc, 'reason', repr(exc))[:200]))
        if info is None:
            stop = 'NEXT_STATE_NOT_CERTIFIED'
            break
        row = dict(step=k, t_start_s=info['t_start_s'], t_end_s=info['t_end_s'], requested_action=requested,
                   action=info['action'], CL_cmd=info['CL_cmd'], executed_CL_mean=info['executed_CL_mean'],
                   executed_CL_target_rmse=info['executed_CL_target_rmse'], range_m=info['range_m'],
                   post_context=dict(alpha_prev=info['post_context']['alpha_prev']),
                   plant=dict(alpha=info['plant']['alpha']), network_alpha_rad=info.get('network_alpha_rad'),
                   corrector_source=info.get('post_correction', {}).get('selected_source'),
                   lookahead_offset_deg=info.get('lookahead', {}).get('selected_offset_deg'),
                   lookahead_status=info.get('lookahead', {}).get('status'),
                   transition_started=info['transition_started'], refusals=refusals,
                   solver={h: dict(status=v['status'], iterations=v['iterations'], solve_s=v['solve_s'])
                           for h, v in inner.last.items()},
                   wall_s=time.perf_counter()-tick)
        log.write(json.dumps(clean(row))+'\n')
        log.flush()
        infos.append(row)
        if k % 10 == 0:
            print(f"t={row['t_end_s']:.0f} range={row['range_m']/1e3:.2f} km CL={row['CL_cmd']:.3f} "
                  f"alpha={np.round(np.rad2deg(row['plant']['alpha']), 2).tolist()} wall={row['wall_s']:.1f}s", flush=True)
        if done:
            stop = env.last_outcome
            break
    log.close()
    summary = dict(policy=policy_name, objective=a.objective, initial_offset_h_V=setup['initial_offset_h_V'],
                   stop=stop, completed=bool(stop == 'SUCCESS' and abs(float(env.t)-1200.) < 1e-6),
                   t_end_s=float(env.t),
                   range_km=float(env.x[3]/1000), wall_s=time.perf_counter()-started,
                   fallback_steps=sum(r['action'] != r['requested_action'] for r in infos),
                   corrector_changed_steps=sum(r['corrector_source'] not in (None, 'NETWORK') for r in infos),
                   lookahead_offset_steps=sum((r['lookahead_offset_deg'] or 0) != 0 for r in infos))
    if len(infos) >= 3:
        summary['metrics'] = flight_metrics(infos)
    write_json(out/'summary.json', summary)
    print(json.dumps(clean(summary)), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--objective', choices=sorted(OBJECTIVES), required=True)
    p.add_argument('--cl', type=float, default=.025, help='constant CL policy (ignored with --chain)')
    p.add_argument('--chain', help='state.json of a trained V16 policy, flown closed loop')
    p.add_argument('--horizon-weight', type=float, default=FROZEN_HORIZON_WEIGHT,
                   help='consistent objective: CL tracking weight (frozen default 20)')
    p.add_argument('--curvature', type=float, default=0., help='consistent objective: second-difference weight')
    p.add_argument('--max-decisions', type=int, default=400)
    p.add_argument('--offset', type=float, nargs=2, metavar=('DH_M', 'DV_MPS'),
                   help='initial altitude/speed offset, e.g. a held-out-3 state')
    p.add_argument('--fallback-actions', type=int, default=19, help='nearest actions tried when a request is refused')
    p.add_argument('--out', required=True)
    a = p.parse_args(argv)
    if a.cl not in CL_ACTIONS:
        p.error('--cl must be one of the 19 actions')
    run(a)


if __name__ == '__main__':
    main()

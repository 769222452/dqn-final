"""Print the AoA experiment job list, one job per line: inner objective surrogate policy dh dv out_dir.

Stages (inner loop in brackets; surrogate '-' = V16 surrogate / none):
  gate         [ocp]        nominal state; v16/consistent; constant CL 0.025 and final_seed101
  dagger       [surrogate]  development states (13); consistent with the round-0 surrogate; const + seed101
  heldout      [surrogate]  held-out-3 (nominal + 40); v16 (V16 surrogate) / consistent (refitted surrogate);
                            constant CL 0.025 and the 7 V16 policies = 656 flights
  fidelity     [ocp]        nominal + first 5 held-out-3 states; v16/consistent; const + seed101 = 24 flights
                            (surrogate-vs-teacher check of the held-out stage)
  heldout-ocp  [ocp]        the held-out stage with the true OCP every decision (optional, ~800 CPU-h)
  improve      [surrogate]  held-out-3 (nominal + 40); execution layers ARMS (default v16, v16@minimal_first, consistent,
                            consistent@minimal_first, aligned@minimal_first) x IMPROVE_POLICIES (default
                            const0.025 + the 7 V16 policies; trained runs as train_<name>_seed<N>)
  retrained    [surrogate]  held-out-3; consistent; the 7 policies retrained under the consistent objective
                            (runs/aoa/retrain/seed*/state.json, optional retrain stage) = 287 flights

Environment filters: POLICIES (space separated, e.g. "const0.025 seed101"), STATES (number of held-out
states after the nominal state), OBJECTIVES (default "v16 consistent"), SURROGATE (refitted model,
default data/surrogate_consistent/controller.npz), SURROGATE_R0 (default data/surrogate_consistent_r0/...).
Other objectives (e.g. OBJECTIVES="v16 consistent aligned") use data/surrogate_<objective>[_r0]/controller.npz;
REFIT_OBJECTIVE (default consistent) selects the objective of the dagger and retrained stages.
"""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from rollout_improvement.evaluation import INITIAL_OFFSETS, test_set_points  # noqa: E402

SEEDS = (17, 23, 41, 59, 83, 97, 101)
STAGES = dict(gate='ocp', dagger='surrogate', heldout='surrogate', fidelity='ocp')
STAGES['heldout-ocp'] = 'ocp'
STAGES['retrained'] = 'surrogate'
STAGES['improve'] = 'surrogate'


def policies(stage):
    if stage == 'improve':
        return os.environ.get('IMPROVE_POLICIES', ' '.join(['const0.025']+[f'seed{s}' for s in SEEDS])).split()
    if stage == 'retrained':
        names = [f'retrained_seed{s}' for s in SEEDS]
        chosen = os.environ.get('POLICIES')
        return [p for p in names if not chosen or p in chosen.split()]
    names = ['const0.025']+[f'seed{s}' for s in SEEDS] if stage in ('heldout', 'heldout-ocp', 'improve') \
        else ['const0.025', 'seed101']
    chosen = os.environ.get('POLICIES')
    return [p for p in names if not chosen or p in chosen.split()]


def offsets(stage):
    if stage == 'gate':
        return [(0., 0.)]
    if stage == 'dagger':
        return list(INITIAL_OFFSETS)
    points, _ = test_set_points('held-out-3')
    if stage == 'fidelity':
        return points[:6]
    n = int(os.environ.get('STATES', len(points)-1))
    return points[:1+n]


def jobs(stage, out):
    inner = STAGES[stage]
    refit = os.environ.get('REFIT_OBJECTIVE', 'consistent')
    if os.environ.get('SAFETY_VARIANT'):            # fly the dagger/retrained stages in the deployed configuration
        refit += '@'+os.environ['SAFETY_VARIANT']
    if stage == 'improve':
        objectives = os.environ.get('ARMS', 'v16 v16@minimal_first consistent consistent@minimal_first aligned@minimal_first').split()
    else:
        objectives = [refit] if stage in ('dagger', 'retrained') else os.environ.get('OBJECTIVES', 'v16 consistent').split()
    for policy in policies(stage):
        for dh, dv in offsets(stage):
            for objective in objectives:
                base = objective.partition('@')[0]     # '@<variant>' = same surrogate, other safety option
                default = f'data/surrogate_{base}' + ('_r0' if stage == 'dagger' else '') + '/controller.npz'
                if base == 'consistent':
                    default = os.environ.get('SURROGATE_R0' if stage == 'dagger' else 'SURROGATE', default)
                surrogate = default if inner == 'surrogate' and base != 'v16' else '-'
                folder = Path(out)/stage/policy/f'dh{dh:g}_dv{dv:g}'/objective
                yield inner, objective, surrogate, policy, dh, dv, folder


if __name__ == '__main__':
    stage, out = sys.argv[1], sys.argv[2]
    if stage not in STAGES:
        raise SystemExit('stage must be one of '+', '.join(STAGES))
    for inner, objective, surrogate, policy, dh, dv, folder in jobs(stage, out):
        print(inner, objective, surrogate, policy, repr(dh), repr(dv), folder)

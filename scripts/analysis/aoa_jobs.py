"""Print the AoA experiment job list, one job per line: inner objective surrogate policy dh dv out_dir.

Stages (inner loop in brackets; surrogate '-' = V16 surrogate / none):
  gate         [ocp]        nominal state; v16/consistent; constant CL 0.025 and final_seed101
  dagger       [surrogate]  development states (13); consistent with the round-0 surrogate; const + seed101
  heldout      [surrogate]  held-out-3 (nominal + 40); v16 (V16 surrogate) / consistent (refitted surrogate);
                            constant CL 0.025 and the 7 V16 policies = 656 flights
  fidelity     [ocp]        nominal + first 5 held-out-3 states; v16/consistent; const + seed101 = 24 flights
                            (surrogate-vs-teacher check of the held-out stage)
  heldout-ocp  [ocp]        the held-out stage with the true OCP every decision (optional, ~800 CPU-h)

Environment filters: POLICIES (space separated, e.g. "const0.025 seed101"), STATES (number of held-out
states after the nominal state), OBJECTIVES (default "v16 consistent"), SURROGATE (refitted model,
default data/surrogate_consistent/controller.npz), SURROGATE_R0 (default data/surrogate_consistent_r0/...).
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


def policies(stage):
    names = ['const0.025']+[f'seed{s}' for s in SEEDS] if stage in ('heldout', 'heldout-ocp') \
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
    objectives = ['consistent'] if stage == 'dagger' else os.environ.get('OBJECTIVES', 'v16 consistent').split()
    model = os.environ.get('SURROGATE_R0' if stage == 'dagger' else 'SURROGATE',
                           'data/surrogate_consistent_r0/controller.npz' if stage == 'dagger'
                           else 'data/surrogate_consistent/controller.npz')
    for policy in policies(stage):
        for dh, dv in offsets(stage):
            for objective in objectives:
                surrogate = model if inner == 'surrogate' and objective == 'consistent' else '-'
                folder = Path(out)/stage/policy/f'dh{dh:g}_dv{dv:g}'/objective
                yield inner, objective, surrogate, policy, dh, dv, folder


if __name__ == '__main__':
    stage, out = sys.argv[1], sys.argv[2]
    if stage not in STAGES:
        raise SystemExit('stage must be one of '+', '.join(STAGES))
    for inner, objective, surrogate, policy, dh, dv, folder in jobs(stage, out):
        print(inner, objective, surrogate, policy, repr(dh), repr(dv), folder)

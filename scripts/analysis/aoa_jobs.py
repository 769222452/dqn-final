"""Print the AoA experiment job list, one job per line: objective policy dh dv out_dir.

Stages:
  gate     nominal state; objectives v16/consistent; constant CL 0.025 and final_seed101
  heldout  held-out-3 (nominal + 40 states); objectives v16/consistent; constant CL 0.025 and the 7 V16 policies

Environment filters: POLICIES (space separated, e.g. "const0.025 seed101"), STATES (count of held-out
states to include after the nominal state, default all 40), OBJECTIVES (default "v16 consistent").
"""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from rollout_improvement.evaluation import test_set_points  # noqa: E402

SEEDS = (17, 23, 41, 59, 83, 97, 101)


def policies(stage):
    names = ['const0.025', 'seed101'] if stage == 'gate' else ['const0.025']+[f'seed{s}' for s in SEEDS]
    chosen = os.environ.get('POLICIES')
    return [p for p in names if not chosen or p in chosen.split()]


def jobs(stage, out):
    objectives = os.environ.get('OBJECTIVES', 'v16 consistent').split()
    if stage == 'gate':
        offsets = [(0., 0.)]
    else:
        offsets, _ = test_set_points('held-out-3')
        n = int(os.environ.get('STATES', len(offsets)-1))
        offsets = offsets[:1+n]
    for policy in policies(stage):
        for dh, dv in offsets:
            for objective in objectives:
                folder = Path(out)/stage/policy/f'dh{dh:g}_dv{dv:g}'/objective
                yield objective, policy, dh, dv, folder


if __name__ == '__main__':
    stage, out = sys.argv[1], sys.argv[2]
    if stage not in ('gate', 'heldout'):
        raise SystemExit('stage must be gate or heldout')
    for objective, policy, dh, dv, folder in jobs(stage, out):
        print(objective, policy, repr(dh), repr(dv), folder)

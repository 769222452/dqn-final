"""Fresh offline teacher/refit and one fixed-CL pilot flight; no RL training."""
import argparse
import json
from pathlib import Path
import sys
import time
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from certified_flight import integrity
from continuous_surrogate.experiment import (DEFAULT_CONFIG, baseline, build, fit_model,
    generate_labels, run_flight, write_json)
from continuous_surrogate.model import ContinuousSurrogate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--teacher-contexts', type=Path,
        default=Path('D:/v6-mission-morph-clean/research/teacher_929/all_labels.jsonl'))
    parser.add_argument('--stage', choices=['all','labels','flight'], default='all')
    parser.add_argument('--reuse-labels', type=Path,
        help='Refit from audited labels of this exact continuous teacher contract; never old ZOH labels')
    args = parser.parse_args()
    started = time.perf_counter()
    if args.stage != 'flight':
        args.out.mkdir(parents=True, exist_ok=False)
    elif not (args.out/'model.npz').is_file():
        raise ValueError('Flight stage requires the previously created model.npz')
    system = build(args.config)
    if args.stage != 'flight':
        write_json(args.out/'identity.json', system['identity'])
        write_json(args.out/'config.json', system['config'])
        if args.reuse_labels is None:
            records = generate_labels(system, args.teacher_contexts, args.out/'labels')
        else:
            records = json.loads(args.reuse_labels.read_text(encoding='utf-8'))
            if any(r.get('teacher_contract') != system['teacher'].experiment for r in records):
                raise ValueError('Reused labels were generated under a different teacher contract')
            folder = args.out/'labels'
            folder.mkdir()
            write_json(folder/'labels.json', records)
            write_json(folder/'source_reference.json', dict(path=str(args.reuse_labels.resolve()),
                sha256=integrity.sha256(args.reuse_labels), teacher_contract=system['teacher'].experiment))
            write_json(folder/'summary.json', dict(wall_s=0., cpu_s=0., new_ocp_labels_generated=0,
                reused_from=str(args.reuse_labels.resolve()),
                accepted=dict(Counter(r['head']+'_'+r['role'] for r in records if r['accepted'])),
                refused=[dict(case=r['case'],head=r['head'],reason=r.get('reason')) for r in records if not r['accepted']]))
        model = fit_model(system, records, args.out/'model.npz')
        write_json(args.out/'offline_preparation.json', dict(wall_s=time.perf_counter()-started,
            teacher_labels=len(records), accepted=sum(r['accepted'] for r in records),
            model_metadata=model.metadata))
    else:
        # Implementation identity may not drift between label fitting and execution.
        original = json.loads((args.out/'identity.json').read_text(encoding='utf-8'))
        if original != system['identity']:
            raise ValueError('Source/config identity changed after fitting; prepare a new pilot')
        model = ContinuousSurrogate.load(args.out/'model.npz', system['db'], system['mission'], baseline(system))
    system['model_hash'] = integrity.sha256(args.out/'model.npz')
    if args.stage != 'labels':
        result = run_flight(system, model, args.out/'flight')
        print(json.dumps({k:v for k,v in result.items() if k != 'rows'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

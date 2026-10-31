"""One fixed-CL continuous-recovery pilot; no RL update or online model refit."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from certified_flight import integrity
from continuous_surrogate.experiment import baseline, write_json
from continuous_recovery.experiment import run_flight
from continuous_recovery.local_model import LocalResidualSurrogate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True, help='New local-residual model.npz')
    parser.add_argument('--out', type=Path, required=True, help='New directory; overwrite is refused')
    parser.add_argument('--config', type=Path, default=ROOT/'experiments/continuous_recovery/config.json')
    parser.add_argument('--start-state', type=Path, help='Saved before.json or recovery checkpoint wrapper')
    parser.add_argument('--max-decisions', type=int, default=400)
    args = parser.parse_args()
    if not 1 <= args.max_decisions <= 400:
        parser.error('--max-decisions must be in [1, 400]')
    if not args.model.is_file():
        parser.error('--model does not exist')
    if args.start_state is not None and not args.start_state.is_file():
        parser.error('--start-state does not exist')
    if args.out.exists():
        parser.error('--out already exists; refusing overwrite')
    from continuous_recovery.preparation import build

    args.out.mkdir(parents=True, exist_ok=False)
    started, cpu_start = time.perf_counter(), time.process_time()
    try:
        system = build(args.config)
        model = LocalResidualSurrogate.load(args.model, system['db'], system['mission'], baseline(system))
        system['model_hash'] = integrity.sha256(args.model)
        write_json(args.out/'identity.json', system['identity'])
        write_json(args.out/'config.json', system['config'])
        write_json(args.out/'model_reference.json', dict(path=str(args.model.resolve()),
            sha256=system['model_hash'], metadata_sha256=integrity.sha256(args.model.with_suffix('.json')),
            metadata=model.metadata, model_refit_performed=False))
        write_json(args.out/'setup.json', dict(wall_s=time.perf_counter()-started,
            cpu_s=time.process_time()-cpu_start, max_decisions=args.max_decisions,
            start_state=None if args.start_state is None else str(args.start_state.resolve()),
            start_state_sha256=None if args.start_state is None else integrity.sha256(args.start_state)))
        result = run_flight(system, model, args.out/'flight', args.start_state, args.max_decisions)
        print(json.dumps({k:v for k, v in result.items() if k not in ('rows', 'decision_wall_s')},
                         ensure_ascii=False), flush=True)
    except Exception as exc:
        write_json(args.out/'run_error.json', dict(type=type(exc).__name__, reason=str(exc),
                   wall_s=time.perf_counter()-started))
        raise


if __name__ == '__main__':
    main()

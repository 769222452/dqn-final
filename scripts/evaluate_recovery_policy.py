"""Evaluate a frozen learned policy with fresh continuous-recovery certificates."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True, help='External frozen chain state.json')
    parser.add_argument('--model', type=Path, required=True, help='Local residual surrogate model.npz')
    parser.add_argument('--out', type=Path, required=True, help='New directory; overwrite is refused')
    parser.add_argument('--max-decisions', type=int, default=400)
    args = parser.parse_args()
    if not 1 <= args.max_decisions <= 400:
        parser.error('--max-decisions must be in [1, 400]')
    for key in ('policy', 'model'):
        if not getattr(args, key).is_file():
            parser.error('--'+key+' does not exist')
    if args.out.exists():
        parser.error('--out already exists; refusing overwrite')
    from certified_flight import integrity
    from continuous_surrogate.experiment import baseline, write_json
    from continuous_recovery.local_model import LocalResidualSurrogate
    from continuous_recovery.learned_policy import FrozenChainPolicy
    from continuous_recovery.policy_experiment import run_policy_flight, PROTOCOL
    from continuous_recovery.preparation import build, snapshot_sources

    args.out.mkdir(parents=True, exist_ok=False)
    started, cpu_started = time.perf_counter(), time.process_time()
    try:
        system = build()
        policy = FrozenChainPolicy(args.policy)
        model = LocalResidualSurrogate.load(args.model, system['db'], system['mission'], baseline(system))
        system['model_hash'] = integrity.sha256(args.model)
        snapshot_sources(system, args.out/'source_snapshot')
        write_json(args.out/'identity.json', system['identity'])
        write_json(args.out/'config.json', system['config'])
        write_json(args.out/'policy_reference.json', policy.metadata)
        write_json(args.out/'model_reference.json', dict(path=str(args.model.resolve()),
            sha256=system['model_hash'], metadata_sha256=integrity.sha256(args.model.with_suffix('.json')),
            metadata=model.metadata, model_refit_performed=False))
        write_json(args.out/'setup.json', dict(protocol=PROTOCOL, wall_s=time.perf_counter()-started,
            cpu_s=time.process_time()-cpu_started, max_decisions=args.max_decisions,
            policy_driven=True, fixed_config_flight_CL_used=False,
            policy_decision_s=12., execution_s=3., training_performed=False,
            evaluation='Frozen external policy migrated to a new continuous controller; no old certification reused'))
        result = run_policy_flight(system, model, policy, args.out/'flight', args.max_decisions)
        print(json.dumps({key:value for key, value in result.items() if key not in ('rows', 'decision_wall_s')},
                         ensure_ascii=False), flush=True)
    except Exception as exc:
        write_json(args.out/'run_error.json', dict(type=type(exc).__name__, reason=str(exc),
                   wall_s=time.perf_counter()-started))
        raise


if __name__ == '__main__':
    main()

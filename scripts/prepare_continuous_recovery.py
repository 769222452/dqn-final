"""Prepare audited real-OCP augmentation and local model; never execute a flight."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from continuous_recovery.preparation import (
    DEFAULT_CONFIG, DEFAULT_FLIGHT, DEFAULT_OLD_LABELS, DEFAULT_RECOVERY_STEP,
    build, fit_model, generate_labels, snapshot_sources,
)
from continuous_surrogate.experiment import source_identity, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--old-labels', type=Path, default=DEFAULT_OLD_LABELS)
    parser.add_argument('--source-flight', type=Path, default=DEFAULT_FLIGHT)
    parser.add_argument('--recovery-step', type=Path, default=DEFAULT_RECOVERY_STEP)
    args = parser.parse_args()
    args.config = args.config.resolve(strict=True)
    # All immutable inputs must exist before allocating the new output folder.
    for source in (args.config, args.old_labels, args.recovery_step):
        source.resolve(strict=True)
    args.source_flight.resolve(strict=True)
    args.out.mkdir(parents=True, exist_ok=False)
    started, cpu_started = time.perf_counter(), time.process_time()
    try:
        system = build(args.config)
        write_json(args.out/'identity.json', system['identity'])
        write_json(args.out/'config.json', system['config'])
        snapshot_sources(system, args.out/'source_snapshot')
        records = generate_labels(system, args.old_labels, args.out/'labels',
                                  args.source_flight, args.recovery_step)
        model = fit_model(system, records, args.out/'model.npz')
        if source_identity(args.config) != system['identity']:
            raise ValueError('Source/config identity changed during preparation; do not execute this model')
        result = dict(status='PREPARED_NOT_FLOWN', wall_s=time.perf_counter()-started,
            cpu_s=time.process_time()-cpu_started, build_timings=system['build_timings'],
            teacher_records=len(records), accepted=sum(row['accepted'] for row in records),
            model_metadata=model.metadata, training_performed=False,
            surrogate_refit_performed=True, full_model_qualification_passed=False)
        write_json(args.out/'offline_preparation.json', result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception as exc:
        write_json(args.out/'preparation_failure.json', dict(status='FAILED',
            wall_s=time.perf_counter()-started, reason=type(exc).__name__, message=str(exc),
            partial_labels_preserved=True))
        raise


if __name__ == '__main__':
    main()

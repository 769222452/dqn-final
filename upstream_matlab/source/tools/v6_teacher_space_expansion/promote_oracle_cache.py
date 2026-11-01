"""Reuse an existing oracle cache only after proving identical effective inputs.

The running evaluator hashes all P2 rows although mission_cd_for_geometry reads
only the shared mission grid. This helper materializes the evaluator's existing
key for an enriched table from a cache of its original table. It never changes
evaluation.py, P2 values, ranking, or a running process. Cache sources must match
the exact current evaluator, reference, domain, and original P2 content hash.
"""
from __future__ import annotations

import argparse
from exp_common import *
from evaluation import all_aero, mission_cd_for_geometry

KEY = ['geometry_id', 'M_flight', 'alpha_deg']
COLUMNS = ['M_flight', 'alpha_deg', 'CL', 'CD_pressure', 'model_valid']


def base_aero(inventory):
    pieces = [pd.read_csv(OUT / 'baseline/frozen49.csv', float_precision='round_trip')]
    selected = OLD / 'work/selected_results.csv'
    if selected.exists():
        pieces.append(pd.read_csv(selected, float_precision='round_trip'))
    for path in sorted((OUT / 'cases').glob('*/frozen49.csv')):
        pieces.append(pd.read_csv(path, float_precision='round_trip'))
    for path in sorted((OUT / 'aero').glob('*/*.csv')):
        if not path.name.startswith('verification_') and not path.name.endswith(('.queries.csv', '.tmp.csv')):
            pieces.append(pd.read_csv(path, float_precision='round_trip'))
    data = pd.concat(pieces, ignore_index=True)
    return data[data.geometry_id.isin(inventory.geometry_id)].drop_duplicates(KEY, keep='first')


def same_arrays(left, right):
    return all(np.array_equal(left[key], right[key], equal_nan=True) for key in ('CD', 'alpha'))


def promote(apply=False):
    inventory = pool()
    eligible = inventory[gate(inventory)]
    base = {gid: g for gid, g in base_aero(inventory).groupby('geometry_id')}
    full = {gid: g for gid, g in all_aero().groupby('geometry_id')}
    paths = [OUT / 'mission/reference.csv', OUT / 'mission/domain.json', TOOLS / 'evaluation.py']
    frozen_bytes = [p.read_bytes() for p in paths]
    suffix = b''.join(frozen_bytes)
    signature = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
    signature['promoter_sha256'] = sha(__file__)
    domain = load(paths[1])
    grid = pd.MultiIndex.from_product([domain['M_grid'], domain['alpha_grid']], names=['M_flight', 'alpha_deg'])
    required = set(grid.tolist())
    plans, sources, skipped = [], {}, []
    present = 0

    def cache_path(gid, g):
        content = g.sort_values(['M_flight', 'alpha_deg'])[COLUMNS].to_csv(index=False).encode()
        key = hashlib.sha256(content + suffix).hexdigest()[:16]
        return OUT / 'oracle_cache' / f'{gid}_{key}.npz'

    def effective(g):
        return g[COLUMNS].set_index(['M_flight', 'alpha_deg']).reindex(grid).to_csv().encode()

    for row in eligible.itertuples():
        gid = row.geometry_id
        if gid not in base or gid not in full or not required.issubset(set(zip(base[gid].M_flight, base[gid].alpha_deg))):
            skipped.append({'geometry_id': gid, 'reason': 'INCOMPLETE_BASE_GRID'})
            continue
        source, destination = cache_path(gid, base[gid]), cache_path(gid, full[gid])
        if source.exists():
            sources[gid] = source
        if destination.exists():
            present += 1
            continue
        if not source.exists():
            skipped.append({'geometry_id': gid, 'reason': 'EXACT_SOURCE_KEY_NOT_PRESENT'})
            continue
        original, enriched = effective(base[gid]), effective(full[gid])
        if original != enriched:
            skipped.append({'geometry_id': gid, 'reason': 'EFFECTIVE_INPUT_GRID_DIFFERS'})
            continue
        plans.append({'geometry_id': gid, 'source': source, 'destination': destination,
                      'effective_input_sha256': hashlib.sha256(original).hexdigest(),
                      'source_P2_rows': len(base[gid]), 'enriched_P2_rows': len(full[gid])})
    preview = {'utc': utc(), 'eligible_geometries': len(eligible), 'current_key_present': present,
               'equivalent_promotions_available': len(plans), 'skipped': skipped}
    print(json.dumps(clean(preview), ensure_ascii=True), flush=True)
    if not apply or not plans:
        return preview

    validation_path = OUT / 'validation_oracle_cache_promotion.json'
    validated = load(validation_path) if validation_path.exists() else {}
    if validated.get('status') != 'PASS' or validated.get('signature') != signature:
        # Include enriched old geometry, a new geometry, and a cached trajectory
        # with non-finite intervals when available. No Teacher/P2 invocation.
        selected = [plans[0]['geometry_id']]
        new_ids = set(eligible[eligible.phase.ne('baseline')].geometry_id)
        selected += sorted(new_ids & set(sources))[:1]
        for gid, source in sources.items():
            with np.load(source) as z:
                if not np.isfinite(z['CD']).all() and gid not in selected:
                    selected.append(gid)
                    break
        for plan in plans:
            if len(set(selected)) >= 3:
                break
            if plan['geometry_id'] not in selected:
                selected.append(plan['geometry_id'])
        ref = pd.read_csv(paths[0])
        checks = []
        for gid in dict.fromkeys(selected):
            drag, alpha = mission_cd_for_geometry(full[gid], ref)
            with np.load(sources[gid]) as z:
                exact = same_arrays(z, {'CD': drag, 'alpha': alpha})
            record = {'geometry_id': gid, 'arrays_bit_exact_including_NaN': exact,
                      'trajectory_intervals': len(drag), 'nonfinite_CD_intervals': int((~np.isfinite(drag)).sum()),
                      'base_P2_rows': len(base[gid]), 'enriched_P2_rows': len(full[gid]),
                      'source_cache_sha256': sha(sources[gid])}
            checks.append(record)
            print(json.dumps(record), flush=True)
            if not exact:
                dump(validation_path, {'utc': utc(), 'status': 'FAIL', 'signature': signature, 'checks': checks})
                raise RuntimeError('Existing oracle cache differs from the original function; no cache was promoted')
        validated = {'utc': utc(), 'status': 'PASS', 'signature': signature, 'checks': checks,
                     'scope': 'Exact-current-key source caches; full shared-grid identity checked separately for every promotion. Representative original-function recalculations include invalid intervals.',
                     'Teacher_or_P2_calls': 0}
        dump(validation_path, validated)

    if [p.read_bytes() for p in paths] != frozen_bytes:
        raise RuntimeError('Evaluator/reference/domain changed during validation; no cache was promoted')
    committed = []
    for plan in plans:
        source, destination = plan['source'], plan['destination']
        blob = source.read_bytes()
        if destination.exists():
            with np.load(source) as a, np.load(destination) as b:
                if not same_arrays(a, b):
                    raise RuntimeError(f'Existing destination disagrees for {plan["geometry_id"]}')
            continue
        temporary = destination.with_name(destination.name + f'.promotion-{os.getpid()}.tmp')
        with temporary.open('wb') as handle:
            handle.write(blob)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        committed.append({**plan, 'source_cache_sha256': hashlib.sha256(blob).hexdigest(),
                          'destination_sha256': sha(destination)})
    result = {**preview, 'signature': signature, 'validated_by': str(validation_path),
              'promoted_count': len(committed), 'promoted': committed,
              'scope': 'Derived cache reuse only; no physical values, solver processes, or scientific rankings changed.'}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    dump(OUT / 'operational/oracle_cache_promotions' / f'{stamp}.json', result)
    dump(OUT / 'operational/oracle_cache_promotions/latest.json', result)
    print(json.dumps({'utc': utc(), 'promoted_count': len(committed), 'Teacher_or_P2_calls': 0}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    promote(parser.parse_args().apply)

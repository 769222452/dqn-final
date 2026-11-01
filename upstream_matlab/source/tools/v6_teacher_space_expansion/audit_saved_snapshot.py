"""Independent saved-table checks; never generates geometry or calls P2."""
from __future__ import annotations

import argparse
import hashlib
import json
import math

from exp_common import OUT, ROOT, OLD, TOOLS, SREF, CLMIN, FROZEN_M, csv, dump, gate, load, np, pd, sha, true, utc


def read(path):
    return pd.read_csv(path, float_precision='round_trip')


def near(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12), (a, b)


def audit_core(proof, original, ids, folder, source):
    """Re-sort saved raw CL/CD directly, independently of ranking helpers."""
    records = proof['core_results']
    needed = {tuple(e['coordinates']) for r in records for e in r['neighbor_evidence']}
    pieces = []
    paths = [OUT/'baseline/frozen49.csv', OLD/'work/selected_results.csv']
    paths += sorted((OUT/'cases').glob('*/frozen49.csv'))
    paths += [p for p in sorted((OUT/'aero').glob('*/*.csv'))
              if not p.name.endswith(('.queries.csv', '.tmp.csv')) and p.parent.name in ids]
    for path in paths:
        if not path.exists():
            continue
        digest = sha(path)
        data = read(path)
        keep = data.geometry_id.isin(ids) & pd.Series(
            [point in needed for point in zip(data.M_flight, data.alpha_deg)], index=data.index)
        if keep.any():
            source(path)
            assert sha(path) == digest, f'Raw data changed while reading: {path}'
            pieces.append(data.loc[keep, ['geometry_id', 'M_flight', 'alpha_deg', 'CL', 'CD_pressure', 'model_valid']])
    assert pieces or not records
    raw = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    overlap_count, max_overlap_difference = 0, 0.
    if len(raw):
        keys = ['geometry_id', 'M_flight', 'alpha_deg']
        overlaps = raw[raw.duplicated(keys, keep=False)]
        for _, group in overlaps.groupby(keys):
            overlap_count += len(group)-1
            values = group[['CL', 'CD_pressure']].to_numpy(float)
            np.testing.assert_allclose(values, np.repeat(values[:1], len(values), axis=0), rtol=0, atol=1e-15, equal_nan=True)
            assert true(group.model_valid).nunique() == 1
            finite_difference = np.abs(values-values[:1])
            if np.isfinite(finite_difference).any():
                max_overlap_difference = max(max_overlap_difference, float(np.nanmax(finite_difference)))
        # Preserve the original evaluator's frozen-first precedence. Legacy CSV
        # serialization can differ at the final decimal; record that difference.
        raw = raw.drop_duplicates(keys, keep='first')
    checked, witnesses = 0, []
    for record in records:
        center = original.iloc[record['source_row']]
        for key in ['geometry_id', 'M_flight', 'alpha_deg']:
            assert center[key] == record[key]
        margins, retained = [], []
        for evidence in record['neighbor_evidence']:
            m, a = evidence['coordinates']
            point = raw[raw.M_flight.eq(m) & raw.alpha_deg.eq(a)]
            assert set(point.geometry_id) == ids
            eligible = true(point.model_valid) & np.isfinite(point.CL) & np.isfinite(point.CD_pressure) & point.CD_pressure.gt(0) & point.CL.ge(CLMIN)
            ordered = point[eligible].copy()
            ordered['LD_direct'] = ordered.CL / ordered.CD_pressure
            ordered = ordered.sort_values(['LD_direct', 'CD_pressure', 'geometry_id'], ascending=[False, True, True])
            assert len(ordered) >= 2
            first, second = ordered.iloc[0], ordered.iloc[1]
            margin = (first.LD_direct-second.LD_direct)/second.LD_direct
            same = first.geometry_id == record['geometry_id']
            assert first.geometry_id == evidence['winner']
            assert second.geometry_id == evidence['runner_up']
            assert same == evidence['rank_retained']
            near(margin, evidence['margin'])
            margins.append(margin)
            retained.append(same)
            checked += 1
        assert len(margins) == record['points_checked']
        assert all(retained) == record['stable_all_neighbors']
        near(min(margins), record['minimum_local_winner_margin'])
        if record['geometry_id'] in proof['stable_core_winners'] and center.delta_LD >= .005 and all(retained):
            witnesses.append({'geometry_id':center.geometry_id, 'M_flight':center.M_flight,
                              'alpha_deg':center.alpha_deg, 'original_margin':center.delta_LD,
                              'refined_min_margin':min(margins), 'points_checked':len(margins),
                              'source_row':record['source_row']})
    for gid in proof['stable_core_winners']:
        coords = [(list(FROZEN_M).index(w['M_flight']), list(np.arange(5.,10.01)).index(w['alpha_deg']))
                  for w in witnesses if w['geometry_id'] == gid]
        assert any(abs(a[0]-b[0])+abs(a[1]-b[1]) == 1 for a in coords for b in coords)
    if witnesses:
        csv(folder/'verified_core_regions.csv', witnesses)
    return {'all_saved_neighbor_rank_checks':checked, 'diagnostic_centers':len(records),
            'formal_original_witness_rows':len(witnesses), 'stable_core_winners':proof['stable_core_winners'],
            'overlapping_legacy_rows_reconciled':overlap_count,
            'maximum_legacy_overlap_coefficient_difference':max_overlap_difference,
            'legacy_overlap_absolute_tolerance':1e-15,
            'legacy_precedence':'Frozen baseline first, matching the unchanged original evaluator; no coefficients modified.',
            'scope':'Every saved core neighbor independently re-sorted from committed raw CL/CD across the exact eligible pool; original adjacent >=0.5% witnesses and refined rank retention checked separately.'}


def audit(tag, previous_tag=None):
    sources = {}
    source_paths = {}

    def source(path):
        key = path.relative_to(OUT).as_posix() if path.is_relative_to(OUT) else '@repository/'+path.relative_to(ROOT).as_posix()
        sources[key] = sha(path)
        source_paths[key] = path
        return path

    def metric(suffix):
        return source(OUT / f'metrics/{tag}_{suffix}')

    frozen = read(metric('frozen49_all_scores.csv'))
    ids = set(frozen.loc[gate(frozen), 'geometry_id'])
    pool_hash = hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()
    folder = OUT / 'verification' / pool_hash[:16]
    result = {'utc': utc(), 'tag': tag, 'pool_count': len(ids),
              'pool_ids_sha256': pool_hash, 'Teacher_or_P2_calls': 0}
    completed = folder / 'completed.json'
    if completed.exists():
        proof = load(source(completed))
        assert proof['pool_ids_sha256'] == pool_hash
        result['verification_scope']={'mode':proof.get('verification_mode','FULL_NOMINAL_DIAGNOSTICS'),
                                      'deferred_nominal_core_ids':proof.get('deferred_nominal_core_ids',[]),
                                      'deferred_nominal_equal_ids':proof.get('deferred_nominal_equal_ids',[]),
                                      'all_new_nominal_diagnostics_complete':bool(proof['all_new_nominal_core_checked'] and proof['all_new_nominal_equal_checked'])}
        result['core_verification'] = audit_core(proof, read(metric('CORE42_top1.csv')), ids, folder, source)
        original = read(metric('equal_lift_top1.csv'))
        domain = load(source(OUT / 'mission/domain.json'))
        rounds = []
        for plan_path in sorted(folder.glob('round_*_plan.json')):
            plan = load(source(plan_path))
            table_path = plan_path.with_name(plan_path.name.replace('_plan.json', '_dense_equal_lift.csv'))
            if table_path.exists():
                rounds.append((plan, read(source(table_path))))
        checked = 0
        witnesses = []
        for record in proof['equal_results']:
            row = original.iloc[record['source_row']]
            for key in ['geometry_id', 'M_flight', 'CL_target']:
                assert row[key] == record[key]
            matches = [(p, d) for p, d in rounds if str(record['source_row']) in p['equal']]
            assert matches, record
            plan, dense = matches[-1]
            evidence = record['neighbor_evidence']
            actual_margins = []
            actual_retained = []
            for e in evidence:
                m, cl = e['coordinates']
                point = dense[dense.M_flight.eq(m) & dense.CL_target.eq(cl)]
                assert set(point.geometry_id) == ids
                assert not point.geometry_id.duplicated().any()
                valid = point[point.status.eq('OK')]
                assert np.isfinite(valid[['CD_at_target', 'alpha_at_target']].to_numpy()).all()
                assert valid.CD_at_target.gt(0).all()
                ordered = valid.sort_values(['CD_at_target', 'geometry_id'])
                assert len(ordered) >= 2
                first, second = ordered.iloc[0], ordered.iloc[1]
                margin = (second.CD_at_target - first.CD_at_target) / second.CD_at_target
                retained = first.geometry_id == record['geometry_id']
                assert first.geometry_id == e['winner']
                assert second.geometry_id == e['runner_up']
                assert retained == e['rank_retained']
                near(margin, e['margin'])
                actual_margins.append(margin)
                actual_retained.append(retained)
                checked += 1
            assert len(evidence) == record['points_checked']
            assert all(actual_retained) == record['stable_all_neighbors']
            near(min(actual_margins), record['minimum_local_winner_margin'])
            if (record['geometry_id'] in proof['stable_equal_winners']
                    and row.delta_CD >= .005 and record['stable_all_neighbors']):
                witnesses.append({'geometry_id': row.geometry_id, 'M_flight': row.M_flight,
                                  'CL_target': row.CL_target, 'original_margin': row.delta_CD,
                                  'refined_min_margin': min(actual_margins),
                                  'points_checked': len(evidence), 'source_row': record['source_row']})
        for gid in proof['stable_equal_winners']:
            coords = [(domain['M_grid'].index(w['M_flight']), domain['CL_targets'].index(w['CL_target']))
                      for w in witnesses if w['geometry_id'] == gid]
            assert any(abs(a[0]-b[0]) + abs(a[1]-b[1]) == 1 for a in coords for b in coords)
        csv(folder / 'verified_equal_regions.csv', witnesses)
        result['equal_verification'] = {
            'rounds': proof['rounds'], 'all_saved_neighbor_rank_checks': checked,
            'formal_original_witness_rows': len(witnesses),
            'stable_equal_winners': proof['stable_equal_winners'],
            'stable_core_winners': proof['stable_core_winners'],
            'scope': 'Every saved equal-CL neighbor independently re-sorted across the full eligible pool. Original >=0.5% adjacent witnesses and refined rank retention checked separately; untested regions not certified.'}

    oracle = load(metric('oracle.json'))
    trace = read(metric('oracle_trace.csv'))
    fixed = read(metric('fixed_drag.csv'))
    reference_path = source(OUT / 'mission/reference.csv')
    # Match the original evaluator's actual CSV reader; also disclose its
    # last-bit differences from an alternate round-trip parse.
    reference = pd.read_csv(reference_path)
    reference_rt = read(reference_path)
    differences = {}
    for col in ['time_s', 'dt_s', 'Mach', 'alpha_deg', 'h_m', 'V_mps', 'CL_actual', 'q_pa']:
        assert np.array_equal(trace[col].to_numpy(), reference[col].to_numpy()), col
        delta = float(np.max(np.abs(reference[col]-reference_rt[col])))
        if delta:
            differences[col] = delta
    assert len(trace) == 2400
    assert trace.time_s.iloc[0] == 0 and trace.time_s.iloc[-1] + trace.dt_s.iloc[-1] == 1200
    near(trace.dt_s.sum(), 1200)
    assert trace.dt_s.gt(0).all()
    assert np.isfinite(trace[['oracle_CD', 'best_fixed_CD']].to_numpy()).all()
    assert trace[['oracle_CD', 'best_fixed_CD']].gt(0).all().all()
    assert set(fixed.geometry_id) == ids and not fixed.geometry_id.duplicated().any()
    assert fixed.evaluation_status.eq('COMPLETE_GRID').all()
    assert oracle['complete_candidate_pool'] and not oracle['incomplete_geometry_ids']
    full = fixed[fixed.full_1200s_feasible]
    assert len(full) == oracle['full_fixed_feasible_count']
    assert full.feasible_time_s.eq(1200).all()
    jf = float((trace.q_pa*SREF*trace.best_fixed_CD*trace.dt_s).sum())
    jo = float((trace.q_pa*SREF*trace.oracle_CD*trace.dt_s).sum())
    near(jf, full.J_fixed_Ns.min())
    near(jf, oracle['J_fixed_Ns'])
    near(jo, oracle['J_oracle_Ns'])
    near((jf-jo)/jf, oracle['eta_potential'])
    np.testing.assert_allclose((trace.q_pa*SREF*trace.oracle_CD*trace.dt_s).cumsum(),
                               trace.oracle_cumulative_Ns, rtol=1e-12, atol=1e-8)
    np.testing.assert_allclose((trace.q_pa*SREF*trace.best_fixed_CD*trace.dt_s).cumsum(),
                               trace.fixed_cumulative_Ns, rtol=1e-12, atol=1e-8)
    result['oracle'] = {'intervals': len(trace), 'full_time_s': float(trace.dt_s.sum()),
                        'negative_CL_intervals_retained': int(trace.CL_actual.lt(0).sum()),
                        'full_fixed_feasible_count': len(full), 'J_fixed_reintegrated_Ns': jf,
                        'J_oracle_reintegrated_Ns': jo, 'eta_reintegrated': (jf-jo)/jf,
                        'reference_matches_actual_evaluator_reader': True,
                        'alternate_parser_max_abs_differences': differences}
    if previous_tag:
        previous = load(source(OUT / f'metrics/{previous_tag}_oracle.json'))
        result['oracle']['improvement_vs_previous_J_oracle_Ns'] = previous['J_oracle_Ns']-jo

    if not tag.endswith('_verified') and (OUT/f'metrics/{tag}_verified_summary.json').exists():
        preserved = {}
        for suffix in ['frozen49_all_scores.csv', 'frozen49_top1.csv', 'CORE42_top1.csv',
                       'equal_lift.csv', 'equal_lift_top1.csv', 'mission_all_scores.csv',
                       'mission_top1.csv', 'oracle_trace.csv', 'oracle.json']:
            a = metric(suffix)
            b = source(OUT/f'metrics/{tag}_verified_{suffix}')
            assert sha(a) == sha(b), suffix
            preserved[suffix] = sha(a)
        result['common_grid_preserved_exactly_through_refinement'] = preserved
    for path, digest in sources.items():
        assert sha(source_paths[path]) == digest, f'Input changed during audit: {path}'
    result.update(status='PASS', source_sha256=sources,
                  evaluator_source_sha256=sha(TOOLS/'evaluation.py'),
                  formal_target_claimed=False)
    dump(OUT/f'validation_{tag}_saved_snapshot.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--previous-tag')
    args = parser.parse_args()
    result = audit(args.tag, args.previous_tag)
    print(json.dumps({k:v for k,v in result.items() if k != 'source_sha256'}, ensure_ascii=False, indent=2))

"""Audit saved research outputs without calling Teacher/P2 or changing campaign state.

Run after terminal output has been refreshed. A live snapshot is useful, but never
constitutes completion. Visual review and the eleven report answers remain human
review items even when the automated checks pass.
"""
from __future__ import annotations

import argparse
import re
from datetime import timedelta
import psutil
from exp_common import *


def live_campaign_processes():
    found = []
    for process in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        try:
            info = process.info
            command = ' '.join(info['cmdline'] or []).replace('\\', '/')
            if (info['name'] or '').lower() in ('python.exe', 'matlab.exe') and 'v6_teacher_space_expansion' in command and ('run.py' in command or 'exp_worker(' in command):
                found.append({'pid': info['pid'], 'created': info['create_time'], 'command': command})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


def audit_delivery():
    from analysis import rank_fixed, winners, distribution, rank_equal, adjacent_ids
    from evaluation import all_aero, mission_pairs

    cfg = load(OUT / 'config.json')
    status = load(OUT / 'status.json')
    stats = load(OUT / 'metrics/latest.json')
    tag = stats['tag']
    checks = {}
    evidence_hashes = {}

    def evidence(relative):
        path = OUT / relative
        if path.is_file():
            evidence_hashes[relative] = sha(path)
        return path

    def check(name, ok, details):
        checks[name] = {'passed': bool(ok), 'details': details}

    def read_csv(relative):
        return pd.read_csv(evidence(relative), float_precision='round_trip')

    live = live_campaign_processes()
    calls = len(list((OUT / 'cases').glob('*/started.json'))) + len(list((OUT / 'reproducibility').glob('*/started.json')))
    deadline = datetime.fromisoformat(cfg['deadline_override_utc']) if cfg.get('deadline_override_utc') else datetime.fromisoformat(cfg['created_utc']) + timedelta(hours=cfg['max_wall_time_hours'])
    budget_reached = datetime.now(timezone.utc) >= deadline or calls >= cfg['max_teacher_attempts']
    terminal = status['status'] in ('TARGET_REACHED_VERIFIED', 'TARGET_NOT_REACHED')
    permitted_outcome = terminal and (status['status'] == 'TARGET_REACHED_VERIFIED' or budget_reached)
    check('permitted_terminal_outcome_and_no_live_solver', permitted_outcome and not live,
          {'status': status['status'], 'budget_reached': budget_reached, 'deadline_utc': deadline.isoformat(), 'live_processes': live})
    try:
        verify_sources()
        source_ok, source_error = True, None
    except RuntimeError as exc:
        source_ok, source_error = False, str(exc)
    check('protected_original_sources_and_results', source_ok, source_error or 'All recorded protected hashes match.')
    evidence('baseline/source_hashes.json')
    evidence('baseline/additional_source_hashes.json')
    validation = load(evidence('validation.json'))
    check('original_453_and_frozen_top5_baseline', len(old_pool()) == 453 and validation['status'] == 'PASS' and validation['frozen_top5_rows_exact_identity'] == 245, validation['status'])
    reproduction = load(evidence('reproducibility/summary.json'))
    preflight = load(evidence('preflight/summary.json'))
    check('three_reproductions_and_twelve_real_preflight', reproduction['passed'] and reproduction['teacher_calls'] == 3 and preflight['passed'] and preflight['teacher_attempts'] == 12,
          {'reproduction': reproduction, 'preflight': preflight})
    ranking = cfg['ranking']
    expected_constants = {'CL_min': CLMIN, 'volume_min': .85, 'thickness_min': .5, 'Sref': SREF, 'Lref': 5., 'mode': 'P2', 'friction': 'none', 'base_pressure_ratio': None}
    check('unchanged_gates_and_physical_reference', all(ranking[k] == v for k, v in expected_constants.items()) and cfg['matlab_concurrency'] == 1, expected_constants)

    inventory = pool()
    new = inventory[inventory.phase.ne('baseline')]
    eligible = inventory[gate(inventory)]
    expected_ids = set(eligible.geometry_id)
    case_issues = []
    for started in sorted((OUT / 'cases').glob('*/started.json')):
        case = started.parent
        missing = [name for name in ('candidate.json', 'generation_status.json', 'audit.json', 'record.json') if not (case / name).is_file()]
        if missing:
            case_issues.append({'case': case.name, 'missing': missing})
            continue
        candidate = load(case / 'candidate.json')
        record = load(case / 'record.json')
        if len(candidate['inputs10']) != 10 or candidate['inputs10'] != [candidate[k] for k in INPUTS] or candidate['inputs10'] != record['inputs10']:
            case_issues.append({'case': case.name, 'error': 'Ten-input identity mismatch'})
        if gate(pd.DataFrame([record])).iloc[0]:
            for name in ('native_geometry.mat', 'frozen49.csv', 'metrics.json', 'intersections.json', 'sections.json'):
                if not (case / name).is_file():
                    case_issues.append({'case': case.name, 'missing': name})
    check('all_started_candidates_accounted_for_and_audited', not case_issues, {'calls': calls, 'new_records': len(new), 'issues': case_issues})
    check('report_inventory_matches_current_snapshot', stats['committed_teacher_candidates'] == len(new) and stats['new_unique_hard_gate_pass'] == int(gate(new).sum()) and status['teacher_calls_started'] == calls,
          {'current_teacher_calls': calls, 'current_new_unique_hard_gate_pass': int(gate(new).sum())})

    domain = load(evidence('mission/domain.json'))
    trajectory_issues = []
    for row in domain['trajectories']:
        if sha(ROOT / row['path']) != row['sha256'] or sha(row['success_evidence']) != row['success_sha256']:
            trajectory_issues.append(row['trajectory_id'])
    samples = read_csv('mission/samples.csv')
    duration = samples.groupby('trajectory_id').dt_s.sum()
    check('successful_1200s_trajectories_and_original_CL_targets', not trajectory_issues and np.allclose(duration, 1200.) and set(CLTARGETS).issubset(set(domain['CL_targets'])),
          {'changed_trajectory_sources': trajectory_issues, 'duration_s': duration.to_dict(), 'CL_targets': domain['CL_targets']})

    data = all_aero()
    raw_frozen = data[data.M_flight.isin(FROZEN_M) & data.alpha_deg.isin(FROZEN_A)]
    coverage = {gid: set(zip(g.M_flight, g.alpha_deg)) for gid, g in data.groupby('geometry_id')}
    missing_frozen = sorted(gid for gid in expected_ids if not set(frozen_pairs()).issubset(coverage.get(gid, set())))
    missing_mission = sorted(gid for gid in expected_ids if not set(mission_pairs()).issubset(coverage.get(gid, set())))
    check('complete_frozen49_and_shared_task_grid', not missing_frozen and not missing_mission,
          {'eligible_geometries': len(expected_ids), 'missing_frozen49': missing_frozen, 'missing_mission': missing_mission})
    recomputed = winners(rank_fixed(raw_frozen, inventory))
    published = read_csv(f'metrics/{tag}_frozen49_top1.csv')
    matched = recomputed.merge(published, on=['M_flight', 'alpha_deg'], suffixes=('_raw', '_saved'), validate='one_to_one')
    fixed_ok = len(matched) == 49 and matched.geometry_id_raw.eq(matched.geometry_id_saved).all() and matched.runner_up_raw.eq(matched.runner_up_saved).all()
    fixed_ok &= np.allclose(matched.L_over_D_raw, matched.L_over_D_saved, rtol=1e-13, atol=0)
    fixed_ok &= np.allclose(matched.delta_LD_raw, matched.delta_LD_saved, rtol=1e-12, atol=1e-15)
    check('saved_frozen_ranking_reproduces_raw_P2', fixed_ok, {'conditions_compared': len(matched)})
    core = recomputed[recomputed.alpha_deg.ge(5)].copy()
    observed = distribution(core)
    check('CORE42_metrics_reproduce_raw_ranking', observed == stats['CORE42'], observed)

    equal = read_csv(f'metrics/{tag}_equal_lift.csv')
    _, equal_top = rank_equal(equal)
    saved_equal = read_csv(f'metrics/{tag}_equal_lift_top1.csv')
    joined = equal_top.merge(saved_equal, on=['M_flight', 'CL_target'], suffixes=('_raw', '_saved'), validate='one_to_one')
    equal_ok = len(joined) == len(saved_equal) and joined.geometry_id_raw.eq(joined.geometry_id_saved).all() and np.allclose(joined.CD_at_target_raw, joined.CD_at_target_saved, rtol=1e-13, atol=0)
    check('equal_CL_ranking_and_feasibility_saved_separately', equal_ok and set(equal.geometry_id) == expected_ids and set(CLTARGETS).issubset(set(equal.CL_target)), {'winner_cells': len(saved_equal), 'candidate_geometries': equal.geometry_id.nunique()})

    verified = load(evidence('verification/latest.json')) if (OUT / 'verification/latest.json').exists() else {}
    current_hash = hashlib.sha256('\n'.join(sorted(expected_ids)).encode()).hexdigest()
    proof_current = verified.get('pool_ids_sha256') == current_hash
    core_verified = set(verified.get('stable_core_winners', [])) if proof_current else set()
    equal_verified = set(verified.get('stable_equal_winners', [])) if proof_current else set()
    proof_issues = []
    for name, table, records, axes, values, margin, ids in (
        ('core', core, verified.get('core_results', []), ('M_flight', 'alpha_deg'), [list(FROZEN_M), list(np.arange(5., 10.01))], 'delta_LD', core_verified),
        ('equal', equal_top, verified.get('equal_results', []), ('M_flight', 'CL_target'), [domain['M_grid'], domain['CL_targets']], 'delta_CD', equal_verified)):
        stable_rows = []
        for row in records:
            if row.get('stable_all_neighbors'):
                neighbors = row.get('neighbor_evidence', [])
                if not neighbors or len(neighbors) != row['points_checked'] or not all(n.get('rank_retained') and n.get('winner') == row['geometry_id'] for n in neighbors):
                    proof_issues.append({'kind': name, 'source_row': row['source_row']})
                stable_rows.append({key: row[key] for key in ('geometry_id', *axes)})
        reproduced_ids = set()
        if stable_rows:
            subset = table.merge(pd.DataFrame(stable_rows), on=['geometry_id', *axes])
            reproduced_ids = set(adjacent_ids(subset, axes=axes, margin=margin, axis_values=values))
        if ids != reproduced_ids:
            proof_issues.append({'kind': name, 'saved': sorted(ids), 'recomputed': sorted(reproduced_ids)})
    new_ids = set(new.geometry_id)
    all_nominal_checked=verified.get('all_new_nominal_core_checked',False) and verified.get('all_new_nominal_equal_checked',False)
    formal = proof_current and all_nominal_checked and not proof_issues and len(core_verified) >= 6 and observed['AB_share'] < .6 and len(core_verified & new_ids) >= 3 and len(equal_verified) >= 4 and len(equal_verified & new_ids) >= 2 and not missing_mission
    check('formal_claim_matches_current_pool_proof', bool(stats['formal_target_reached']) == bool(formal) and (status['status'] != 'TARGET_REACHED_VERIFIED' or formal),
          {'formal_target_reproduced': bool(formal), 'proof_current': proof_current,'all_new_nominal_diagnostics_complete':bool(all_nominal_checked), 'proof_issues': proof_issues, 'core_verified': sorted(core_verified), 'equal_verified': sorted(equal_verified)})

    oracle = stats.get('oracle', {})
    trace = read_csv(f'metrics/{tag}_oracle_trace.csv')
    fixed = read_csv(f'metrics/{tag}_fixed_drag.csv')
    reference = samples[samples.trajectory_id.eq(domain['reference_trajectory'])].reset_index(drop=True)
    time_ok = len(trace) == len(reference) and np.allclose(trace.time_s, reference.time_s) and np.allclose(trace.dt_s, reference.dt_s) and np.allclose(trace.CL_actual, reference.CL_actual)
    j_fixed = float((trace.q_pa * SREF * trace.best_fixed_CD * trace.dt_s).sum())
    j_oracle = float((trace.q_pa * SREF * trace.oracle_CD * trace.dt_s).sum())
    best_saved = float(fixed.loc[true(fixed.full_1200s_feasible), 'J_fixed_Ns'].min())
    integral_ok = np.allclose([j_fixed, j_oracle, j_fixed], [oracle.get('J_fixed_Ns', np.nan), oracle.get('J_oracle_Ns', np.nan), best_saved], rtol=1e-12, atol=1e-8)
    eta_ok = np.isclose((j_fixed-j_oracle)/j_fixed, oracle.get('eta_potential', np.nan), rtol=1e-10, atol=1e-13)
    check('full_1200s_oracle_integrals_and_unfavorable_states', time_ok and integral_ok and eta_ok and np.isclose(trace.dt_s.sum(), 1200.) and set(fixed.geometry_id) == expected_ids and oracle.get('complete_candidate_pool', False),
          {'J_fixed_Ns_recomputed': j_fixed, 'J_oracle_Ns_recomputed': j_oracle, 'negative_CL_intervals_retained': int(trace.CL_actual.lt(0).sum()), 'time_s': float(trace.dt_s.sum())})

    batch_issues = []
    for path in sorted((OUT / 'batches').glob('*/candidates.json')):
        for name in ('strategy.json', 'surrogate.npz', 'state_after_proposal.json'):
            if not (path.parent / name).is_file():
                batch_issues.append(str(path.parent / name))
        evidence(path.relative_to(OUT).as_posix())
    search = load(evidence('search_state.json'))
    check('adaptive_proposals_surrogate_and_RNG_checkpoint', not batch_issues and 'rng_state' in search, {'missing': batch_issues})

    figure_names = ['mission_domain.png', f'{tag}_Mach_alpha.png', f'{tag}_Mach_CL.png', f'{tag}_top1_shares.png', 'current_CL_CD_polars.png', 'search_history.png', f'{tag}_full_task_Mach_alpha.png', f'{tag}_oracle_impulse_gain.png', 'oracle_cumulative_drag.png']
    missing_figures = [name for name in figure_names if not (OUT / 'figures' / name).is_file()]
    check('required_figure_artifacts_present', not missing_figures, {'missing': missing_figures, 'visual_review_still_required': True})
    report = evidence('reports/FINAL_REPORT_ZH.md').read_text(encoding='utf-8')
    candidates = read_csv('gocart_candidates.csv')
    required_candidate_ids = set(core.geometry_id) | set(equal_top.geometry_id) | {'G42', *AB}
    check('Chinese_report_candidate_export_and_resume_command', status['status'] in report and set(candidates.geometry_id) == required_candidate_ids and set(INPUTS).issubset(candidates.columns) and candidates.CFD_status.eq('NOT_RUN').all() and r'tools\v6_teacher_space_expansion\run.py --resume' in report,
          {'candidate_count': len(candidates), 'eleven_answers_need_final_content_review': True})
    if status['status']=='TARGET_NOT_REACHED':
        commands=[line for line in report.splitlines() if r'tools\v6_teacher_space_expansion\run.py --resume' in line]
        usable_resume=False
        for command in commands:
            hours=re.search(r'--extend-hours\s+(\d+(?:\.\d+)?)',command)
            cap=re.search(r'--max-teacher-attempts\s+(\d+)',command)
            usable_resume |= bool(hours and cap and float(hours.group(1))>0 and int(cap.group(1))>calls)
        check('budget_end_recovery_command_grants_a_future_stage',usable_resume,
              {'commands':commands,'current_calls':calls,'scope':'Documentation only; the audit does not execute a budget extension.'})
    result = {'utc': utc(), 'scope': 'Saved-output audit; no Teacher/P2 invocation or changes to physical data, search state, or terminal status.',
              'status': status['status'], 'metrics_tag': tag, 'checks': checks,
              'automated_checks_passed': all(v['passed'] for v in checks.values()),
              'ready_for_final_manual_review': all(v['passed'] for v in checks.values()) and terminal and not live,
              'formal_target_reproduced': bool(formal), 'evidence_sha256': evidence_hashes,
              'manual_review_remaining': ['Inspect final plots for correct labels, complete legends, and current metric provenance.', 'Review all eleven Chinese report answers against the audit and scientific tables.', 'Run the committed-P2 identity auditor on the final stopped snapshot.'],
              'goal_completion_claimed': False}
    dump(OUT / 'delivery_audit.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('checks', 'evidence_sha256')}, ensure_ascii=True))
    print(json.dumps({'not_passed': [k for k, v in checks.items() if not v['passed']]}, ensure_ascii=True))
    return result


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    audit_delivery()

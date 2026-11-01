"""Freeze stage-one configuration, then execute five bounded static trials."""
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from morphing_waverider.generator.interface import GeneratorState, TargetAero
from morphing_waverider.teacher_v5.direct_target_matching import (
    TeacherDesignSpace, SearchConfig, DirectTeacherEvaluator, TeacherTargetMatchingGenerator,
    save, sha256, read_json, digest)
import numpy as np

OUT = ROOT / 'results/v5_teacher_target_generator_direct'
TOOLS = ROOT / 'tools/teacher_target_generator_direct'


def snapshot():
    result = {}
    for root in [Path(p) for p in ['D:/final-DQN', 'D:/final-DQN-v3', 'D:/final-DQN-physics-v4', 'D:/mat', str(ROOT)]]:
        for p in root.rglob('*'):
            if not p.is_file() or any(v in {'.git','__pycache__','.pytest_cache','.venv','node_modules'} for v in p.parts):
                continue
            if OUT in p.parents or TOOLS in p.parents:
                continue
            result[str(p)] = sha256(p)
    return result


def table(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8-sig')
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict,list,tuple)) else v for k,v in r.items()} for r in rows)


def persist(evaluator, trials, config_payload, identity, prep):
    save(OUT / 'generator_trials.json', {'trials': trials, 'target_construction': prep,
        'search_config_file_sha256': sha256(OUT / 'search_config.json'),
        'morph_path_validated': 'NOT_TESTED'})
    rows = []
    for t in trials:
        m = t['result']['metadata']
        rows.append({'trial': t['trial'], 'purpose': t['purpose'], 'success': t['result']['success'],
            **{k: m[k] for k in ['solver_status','target_match_status','geometry_id','geometry_hash','z_cmd',
                'CL_pred','CD_pred','target','target_error','geometry_validity','historical_pareto',
                'number_of_geometry_evaluations','number_of_geometry_validations','number_of_aero_evaluations',
                'valid_candidates','invalid_candidates','error_candidates','runtime','failure_reason','search_stop_reason']}})
    table(OUT / 'generator_trials.csv', rows)
    save(OUT / 'candidate_evaluations.json', {'events': evaluator.events,
        'note': 'Only rows eligible_for_score=true have real teacher aero coefficients. All caches exact; no interpolation.'})
    table(OUT / 'candidate_evaluations.csv', evaluator.events)
    save(OUT / 'geometry_cache_manifest.json', {'design_spec_hash': evaluator.space.design_spec_hash,
        'entries': list(evaluator.geometry_cache.values()), 'key_semantics': '(design_spec_hash, exact six z values)'})
    save(OUT / 'aero_cache_manifest.json', {'entries': evaluator.aero_history,
        'reuse_scope': 'Same trial only; key binds exact geometry hash, MAT bytes, M/alpha, source model/reference/options identity'})
    save(OUT / 'failure_log.json', {'candidate_failures': [r for r in evaluator.events if r.get('status') != 'VALID'],
        'trial_failures': [r for r in rows if not r['success']],
        'note': 'A bounded-search target miss is not proof of global mathematical infeasibility.'})
    save(OUT / 'search_budget.json', {'frozen_per_target': {'new_geometry':30, 'aero':36, 'wall_s':300, 'per_call_s':120},
        'target_construction': prep, 'trials': [{k: r[k] for k in ['trial','number_of_geometry_evaluations',
            'number_of_geometry_validations','number_of_aero_evaluations','runtime','search_stop_reason']} for r in rows],
        'matlab_startup_s': evaluator.worker.startup_s,
        'actual_total_geometry_calls': sum(c['request']['operation']=='geometry' for c in evaluator.worker.calls),
        'actual_total_aero_calls': sum(c['request']['operation']=='aero' for c in evaluator.worker.calls),
        'ocp_calls':0, 'dqn_training_calls':0, 'episode_calls':0, 'surrogate_training_calls':0,
        'full52_generation_or_audit_rerun':False})


def main():
    OUT.mkdir(parents=True, exist_ok=False)  # never overwrite an accepted run
    protection = snapshot()
    save(OUT / 'protected_at_trial_start.json', protection)
    old = read_json(ROOT / 'results/v5_invalid_geometry_root_cause/protected_after.json')
    interface = ROOT / 'src/morphing_waverider/generator/interface.py'
    save(OUT / 'authorized_source_change.json', {'path': str(interface),
        'before_sha256': old[str(interface)], 'after_sha256': sha256(interface),
        'change': 'Only add four GenerationStatus enum values for explicit miss/invalid/aero failure/timeout; no existing semantics removed.'})
    space, config = TeacherDesignSpace(), SearchConfig()
    W1 = {'M_star': 15., 'alpha_star_rad': float(np.deg2rad(4.)), 'frame': 'body'}
    payload = {'frozen_utc': datetime.now(timezone.utc).isoformat(), 'search': asdict(config),
        'algorithm': 'Seeded derivative-free bounded coordinate pattern search; exact target-error ranking; parameter midpoint proposal; <=2 random full-box proposals.',
        'trial_plan': ['G42 self-consistency','G34 self-consistency','G16 self-consistency',
                       'midpoint of actual G34/G16 CL/CD at W1','obviously out-of-observed-range CL=CD=1000'],
        'W1': W1, 'seed_library_size':26, 'new_candidate_geometry_gate':'unchanged tools/geometry_audit/scan.audit, opposite diagonal enabled; AMBIGUOUS maps to ERROR',
        'infeasibility_semantics':'Finite budget gives BEST_EFFORT_NOT_WITHIN_TOLERANCE, never global infeasibility proof.',
        'near_deadline_policy':'Do not launch new geometry if remaining < max(20s,1.5*(maximum observed geometry+validation cycle+aero)+2s); never extend deadline.',
        'cold_start_and_target_construction':'Recorded separately from target-search latency; all new geometry/gate/aero measured including IPC.',
        'morph_distance':'Squared parameter distance normalized by SOURCE bounds; exact target-error tie only, not energy.',
        'MORPH_PATH_VALIDATED':'NOT_TESTED'}
    save(OUT / 'search_config.json', payload)
    source_paths = [ROOT / 'src/morphing_waverider/teacher_v5/direct_target_matching.py',
                    ROOT / 'src/morphing_waverider/teacher_v5/target_matching.py', interface,
                    TOOLS / 'run_trials.py', TOOLS / 'teacher_worker.m', TOOLS / 'validate_native.py',
                    ROOT / 'tests/test_teacher_direct_target_matching.py']
    identity = {'created_before_first_teacher_call_utc': datetime.now(timezone.utc).isoformat(),
        'design_spec': space.design_spec, 'design_spec_hash': space.design_spec_hash,
        'reference': space.reference, 'search_config_file_sha256': sha256(OUT / 'search_config.json'),
        'search_config_semantic_hash': digest(asdict(config)),
        'implementation_hashes': {str(p):sha256(p) for p in source_paths},
        'legal_seed_ids':list(space.seeds), 'frozen_teacher_production_core_modified':False,
        'prior_discrete_trial_module_preserved':True,
        'model_scope':'Native frozen teacher geometry and pressure-only offdesign_aero_v4. No synthetic/surrogate objective.',
        'software_tests': {'command':'python -B -m pytest tests/test_teacher_direct_target_matching.py tests/test_teacher_target_matching.py -q -p no:cacheprovider',
            'observed_exit_code':0, 'observed_pass_count':40, 'provenance':'Successful shell execution before freezing this run; not a physical validation.'}}
    save(OUT / 'identity_manifest.json', identity)
    ev = DirectTeacherEvaluator(space, OUT, config)
    gen = TeacherTargetMatchingGenerator(space, ev)
    trials, prep = [], {}
    print('CONFIG_FROZEN ' + identity['search_config_file_sha256'], flush=True)
    try:
        ev.begin('target_construction')
        pairs = {}
        for name in ('G42','G34','G16'):
            row = ev.evaluate(space.seeds[name].parameters, W1['M_star'], W1['alpha_star_rad'])
            if row['status'] != 'VALID':
                raise RuntimeError('AERO_COUPLING_BLOCKED during true target construction')
            pairs[name] = np.array([row['CL_pred'], row['CD_pred']])
            print('TARGET_SOURCE ' + json.dumps({'geometry':name, 'CL':row['CL_pred'], 'CD':row['CD_pred'], 'kernel_s':row['aero_kernel_s']}), flush=True)
        prep = {'geometry_evaluations':ev.geometry_evaluations, 'aero_evaluations':ev.aero_evaluations,
            'wall_s':time.monotonic()-ev.start_time, 'pairs':{k:v.tolist() for k,v in pairs.items()},
            'scope':'Three fresh teacher aero calls on exact previously audited native geometry; not reused old coefficients.'}
        plans = [('G42 self-consistency', pairs['G42'], 'G34'), ('G34 self-consistency', pairs['G34'], 'G42'),
                 ('G16 self-consistency', pairs['G16'], 'G42'),
                 ('new continuous geometry for midpoint target', .5*(pairs['G34']+pairs['G16']), 'G42'),
                 ('out-of-observed-range target', np.array([1000.,1000.]), 'G42')]
        persist(ev, trials, payload, identity, prep)
        for i,(purpose,pair,current) in enumerate(plans,1):
            for p,h in identity['implementation_hashes'].items():
                assert sha256(Path(p)) == h, 'Implementation changed after freeze'
            assert sha256(OUT / 'search_config.json') == identity['search_config_file_sha256']
            state = GeneratorState(W1['M_star'], 0., 0., 0., W1['alpha_star_rad'], np.array(space.seeds[current].parameters),
                {'source':'static teacher inverse-design trial; not an OCP or dynamics state'})
            print(f'TRIAL_START {i} {purpose}', flush=True)
            result = gen.generate(state, TargetAero(*map(float,pair)))
            trials.append({'trial':i,'purpose':purpose,'result':result.as_dict()})
            persist(ev, trials, payload, identity, prep)
            m=result.metadata
            print('TRIAL_DONE ' + json.dumps({k:m[k] for k in ['solver_status','geometry_id','target_error',
                'historical_pareto','number_of_geometry_evaluations','number_of_aero_evaluations','runtime','search_stop_reason']}),flush=True)
            if m['search_stop_reason'] in ('AERO_COUPLING_BLOCKED','ERROR','TEACHER_CALL_TIMEOUT','GEOMETRY_VALIDATION_TIMEOUT'):
                print('BLOCKED_STOP no automatic continuation or retry',flush=True)
                break
    except Exception as exc:
        save(OUT / 'run_failure.json', {'type':type(exc).__name__,'message':str(exc)})
        raise
    finally:
        ev.worker.close()
        persist(ev, trials, payload, identity, prep)
    print('RUN_COMPLETED', flush=True)


if __name__ == '__main__':
    main()

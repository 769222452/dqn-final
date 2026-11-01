"""Summarize real Teacher outcomes without changing sampling or aerodynamic ranking."""
from __future__ import annotations
import argparse
import re
from exp_common import *


def aerodynamic_counts(data, required_ids, pairs):
    expected=len(required_ids)*len(pairs)
    result={'required_geometries':len(required_ids),'expected_points':expected}
    if data is None:
        return {**result,'status':'NOT_YET_SAVED','evaluated_points':None,'model_valid_points':None,
                'CL_floor_eligible_points':None,'model_valid_fraction':None,'complete':False}
    d=data[data.geometry_id.isin(required_ids)].copy()
    d=d.loc[[p in pairs for p in zip(d.M_flight,d.alpha_deg)]]
    if d.duplicated(['geometry_id','M_flight','alpha_deg']).any():
        raise RuntimeError('Duplicate conditions in batch aerodynamic diagnostic source')
    if d.empty:
        return {**result,'status':'COMPLETE' if expected==0 else 'INCOMPLETE','evaluated_points':0,
                'model_valid_points':0,'CL_floor_eligible_points':0,'model_valid_fraction':None,'complete':expected==0}
    valid=true(d.model_valid)&np.isfinite(d.CL)&np.isfinite(d.CD_pressure)&d.CD_pressure.gt(0)
    eligible=valid&d.CL.ge(CLMIN)
    return {**result,'status':'COMPLETE' if len(d)==expected else 'INCOMPLETE',
            'evaluated_points':len(d),'model_valid_points':int(valid.sum()),
            'CL_floor_eligible_points':int(eligible.sum()),
            'model_valid_fraction':float(valid.mean()) if len(d) else None,'complete':len(d)==expected}


def diagnose(batch_number):
    folder = OUT / 'batches' / f'batch_{batch_number:04d}'
    manifest = folder / 'candidates.json'
    candidates = load(manifest)
    rows, frozen_frames, sources = [], [], {str(manifest.relative_to(OUT)): sha(manifest)}
    for case in candidates:
        path = Path(case['case_dir'])
        generation = path / 'generation_status.json'
        record = path / 'record.json'
        if not generation.exists() or not record.exists():
            raise RuntimeError(f'Candidate not committed: {case["geometry_id"]}')
        g, r = load(generation), load(record)
        for source in (generation, record):
            sources[str(source.relative_to(OUT))] = sha(source)
        generated = g['status'] == 'GENERATED'
        valid = r['geometry_validity'] == 'VALID'
        interrupted = bool(g.get('interrupted', False))
        passed = bool(r['hard_gate_pass'])
        if interrupted:
            outcome = 'EXECUTION_INTERRUPTED'
        elif not generated:
            outcome = 'TEACHER_GENERATION_FAILED'
        elif not valid:
            outcome = 'NATIVE_AUDIT_INVALID'
        elif not passed:
            reasons = []
            if r['volume_ratio_G42'] < .85:
                reasons.append('VOLUME_BELOW_0P85')
            if r['thickness_ratio_G42'] < .50:
                reasons.append('THICKNESS_BELOW_0P50')
            if r.get('duplicate_of'):
                reasons.append('DUPLICATE_GEOMETRY')
            outcome = ';'.join(reasons) or 'OTHER_HARD_GATE_FAILURE'
        else:
            outcome = 'UNIQUE_HARD_GATE_PASS'
        error = g.get('error_message', '')
        match = re.search(r'\[([^\[\]]+)\]$', error)
        frozen=path/'frozen49.csv'; evaluated=valid_points=eligible_points=0
        if frozen.exists():
            frame=pd.read_csv(frozen,float_precision='round_trip'); frozen_frames.append(frame)
            sources[str(frozen.relative_to(OUT))]=sha(frozen)
            checked=aerodynamic_counts(frame,{case['geometry_id']},set(frozen_pairs()))
            evaluated=checked['evaluated_points']; valid_points=checked['model_valid_points']; eligible_points=checked['CL_floor_eligible_points']
            if valid_points!=r.get('aero_valid_points',0) or eligible_points!=r.get('eligible_points',0):
                raise RuntimeError(f'Saved aerodynamic counts disagree with actual P2 rows: {case["geometry_id"]}')
        rows.append({'geometry_id':case['geometry_id'], 'attempt_index':case['attempt_index'],
                     'strategy':case['strategy'], **{k:case[k] for k in INPUTS},
                     'design_Mach_bin':min(15, int(np.floor(case['Ma_design']))),
                     'generated':generated, 'native_VALID':valid, 'hard_gate_pass':passed,
                     'execution_interrupted':interrupted, 'outcome':outcome,
                     'native_error_identifier':match.group(1) if match else '',
                     'native_error_message':error, 'audit_reason':r.get('failure_reason', ''),
                     'volume_ratio_G42':r.get('volume_ratio_G42'),
                     'thickness_ratio_G42':r.get('thickness_ratio_G42'),
                     'frozen49_evaluated_points':evaluated,'frozen49_model_valid_points':valid_points,
                     'frozen49_CL_floor_eligible_points':eligible_points})
    d = pd.DataFrame(rows)
    csv(folder / 'outcome_diagnostics.csv', d)
    groups = {}
    for key in ('strategy', 'design_Mach_bin'):
        groups[key] = [{key:name, 'attempts':len(g), 'generated':int(g.generated.sum()),
                        'native_VALID':int(g.native_VALID.sum()), 'unique_hard_gate_pass':int(g.hard_gate_pass.sum()),
                        'outcomes':g.outcome.value_counts().to_dict()}
                       for name, g in d.groupby(key)]
    ids=set(d.loc[d.hard_gate_pass,'geometry_id'])
    frozen_data=pd.concat(frozen_frames,ignore_index=True) if frozen_frames else pd.DataFrame(columns=['geometry_id','M_flight','alpha_deg','model_valid','CL','CD_pressure'])
    domain_path=OUT/'mission/domain.json'; domain=load(domain_path); sources[str(domain_path.relative_to(OUT))]=sha(domain_path)
    mission_path=OUT/f'metrics/batch_{batch_number:04d}_mission_all_scores.csv'
    mission_data=pd.read_csv(mission_path,float_precision='round_trip') if mission_path.exists() else None
    if mission_path.exists(): sources[str(mission_path.relative_to(OUT))]=sha(mission_path)
    aerodynamic={'frozen49':aerodynamic_counts(frozen_data,ids,set(frozen_pairs())),
                 'mission':aerodynamic_counts(mission_data,ids,{(m,a) for m in domain['M_grid'] for a in domain['alpha_grid']}),
                 'definition':'P2 valid means model_valid, finite CL/CD_pressure, CD_pressure>0; CL floor eligibility additionally requires the unchanged CL minimum. Missing task-grid snapshots are not classified as physical failures.'}
    summary = {'utc':utc(), 'batch':batch_number, 'attempts':len(d),'last_attempt_index':int(d.attempt_index.max()),
               'outcomes':d.outcome.value_counts().to_dict(),
               'native_error_identifiers':d.loc[d.native_error_identifier.ne(''), 'native_error_identifier'].value_counts().to_dict(),
               'groups':groups, 'aerodynamics':aerodynamic, 'source_sha256':sources,
               'scope':'Descriptive outcomes of real saved calls. No new Teacher/P2 call, parameter change, or performance correction. Generator nonconvergence is a failed attempt, not proof that no feasible geometry exists in its neighborhood.'}
    dump(folder / 'outcome_diagnostics.json', summary)
    return {k:v for k,v in summary.items() if k != 'source_sha256'}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--batch', type=int, required=True)
    args = ap.parse_args()
    print(json.dumps(clean(diagnose(args.batch)), ensure_ascii=True, indent=2))

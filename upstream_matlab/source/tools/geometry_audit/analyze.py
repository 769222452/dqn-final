"""Aggregate the completed scan and select isolated valid candidates (no aero)."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.stats import spearmanr,rankdata
from prepare import ROOT,OUT,TOOLS,PARAMS,dump,sha,protected

def csvout(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    doc=json.loads((OUT/'pareto_geometry_audit.json').read_text(encoding='utf-8'));results=doc['results'];assert len(results)==52
    manifest=json.loads((OUT/'source_manifest.json').read_text(encoding='utf-8'));rules_hash=sha(OUT/'classification_rules.json');assert rules_hash==manifest['rules_sha256'];assert all(r['classification_rules_sha256']==rules_hash for r in results)
    classes=['VALID','INVALID','AMBIGUOUS','GENERATION_FAILED'];counts={c:sum(r['classification']==c for r in results) for c in classes}
    bounds=np.array([c['source_bounds'] for c in manifest['parameter_columns']]);span=bounds[:,1]-bounds[:,0];z=np.array([[r['parameters'][p] for p in PARAMS] for r in results]);norm=(z-bounds[:,0])/span
    valid=np.array([r['classification']=='VALID' for r in results]);invalid=np.array([r['classification']=='INVALID' for r in results]);dist=np.minimum(norm,1-norm)
    stored_ld=np.array([r['stored_L_D'] for r in results]);stored_vol=np.array([r['stored_Vol_eff'] for r in results]);ldrank=rankdata(-stored_ld,method='min');vrank=rankdata(-stored_vol,method='min')
    rows=[]
    for i,r in enumerate(results):
        row={'design_index':r['design_index'],'original_row_index':r['original_row_index'],**r['parameters'],'classification':r['classification'],'valid_binary':int(valid[i]),'section_minimum_vertical_gap':r.get('section_minimum_vertical_gap'),'surface_intersection_count':r.get('surface_intersection_count'),'stored_L_D':stored_ld[i],'stored_L_D_rank_best1':int(ldrank[i]),'stored_Vol_eff':stored_vol[i],'stored_Vol_eff_rank_best1':int(vrank[i]),'min_normalized_distance_to_parameter_bound':float(dist[i].min())}
        row.update({p+'_normalized_distance_to_bound':float(dist[i,k]) for k,p in enumerate(PARAMS)});rows.append(row)
    csvout('parameter_vs_validity.csv',rows)
    features={p:z[:,i] for i,p in enumerate(PARAMS)}
    features.update(delta_a=z[:,1]-z[:,0],delta_b=z[:,3]-z[:,2],d1_times_ax=z[:,4]*z[:,5],stored_L_D=stored_ld,stored_Vol_eff=stored_vol,min_distance_to_parameter_bound=dist.min(1))
    targets={'valid_binary':valid.astype(float),'intersection_pair_count':np.array([r.get('surface_intersection_count',np.nan) for r in results]),'minimum_section_vertical_gap':np.array([r.get('section_minimum_vertical_gap',np.nan) for r in results])}
    correlations=[]
    for name,x in features.items():
        for target,y in targets.items():
            mask=np.isfinite(x)&np.isfinite(y);status='DESCRIPTIVE_ONLY_NONINDEPENDENT_PARETO_SAMPLE'
            if len(np.unique(y[mask]))<2 or len(np.unique(x[mask]))<2:rho=None;p=None;status='NO_VARIATION_IN_FEATURE_OR_CLASS'
            else:s=spearmanr(x[mask],y[mask]);rho=float(s.statistic);p=float(s.pvalue)
            correlations.append({'feature':name,'target':target,'n':int(mask.sum()),'spearman_rho':rho,'unadjusted_p_descriptive_only':p,'interpretation':status})
    csvout('correlation_diagnostics.csv',correlations)
    regions=[]
    for c in classes:
        selected=[r for r in results if r['classification']==c]
        for p in PARAMS:
            v=[r['parameters'][p] for r in selected]
            regions.append({'classification':c,'parameter':p,'n':len(v),'min':min(v) if v else None,'max':max(v) if v else None,'median':float(np.median(v)) if v else None,'scope':'Observed points only; NOT a filled valid/invalid box or an approved interpolation domain'})
    csvout('observed_parameter_ranges.csv',regions)
    available=list(np.where(valid)[0]);chosen=[]
    if available:
        old=norm[44];nearest=min(available,key=lambda i:np.linalg.norm(norm[i]-old));chosen.append((nearest,'距离原选第45号最近的合法样本（按六维原优化边界归一化欧氏距离）'));available.remove(nearest)
    if available:
        center=np.median(norm[valid],axis=0);central=min(available,key=lambda i:np.linalg.norm(norm[i]-center));chosen.append((central,'合法样本参数中位数附近代表；与第一候选不同'));available.remove(central)
    if available:
        diverse=max(available,key=lambda i:min(np.linalg.norm(norm[i]-norm[j]) for j,_ in chosen));chosen.append((diverse,'与前两个候选最分离的合法样本，覆盖另一参数位置'))
    candidates=[]
    for i,reason in chosen:
        r=results[i];candidates.append({'design_index':r['design_index'],'original_row_index':r['original_row_index'],'parameters':r['parameters'],'inputs10':r['inputs10'],'geometry_hash':r['geometry_hash'],'native_mat':str(Path(r['case_dir'])/'native_geometry.mat'),'classification':r['classification'],'reason':reason,'normalized_distance_from_old45':float(np.linalg.norm(norm[i]-norm[44])),'stored_L_D':r['stored_L_D'],'stored_Vol_eff':r['stored_Vol_eff'],'aerodynamic_validation':'NOT_EXECUTED','path_to_other_candidates':'NOT_VALIDATED'})
    dump(OUT/'V5_NEXT_VALID_GEOMETRY_CANDIDATES.json',{'scope':'Only these exact saved geometries; no shape family, parameter box or intermediate morphing path approved. Does not change current V5 nominal profile.','candidates':candidates,'all_valid_design_indices':[r['design_index'] for r in results if r['classification']=='VALID']})
    oldmat=ROOT/'results/v5_teacher/calls/geometry_002_e96314d4/geometry.mat';newmat=OUT/'cases/design_45/native_geometry.mat';old=loadmat(oldmat);new=loadmat(newmat)
    keys=['wav_x','wav_y','wav_z','wav_up_x','wav_up_y','wav_up_z']
    oldmatch={'old_mat':str(oldmat),'old_mat_sha256':sha(oldmat),'new_mat':str(newmat),'new_mat_sha256':sha(newmat),'comparison':{k:{'array_equal':bool(np.array_equal(old[k],new[k])),'max_abs_difference':float(np.max(np.abs(old[k]-new[k])))} for k in keys},'hash_note':'MAT file byte hashes may differ because of creation headers; native coordinate identity is checked separately. Audit geometry hash includes input10 and has a different schema from old V5 family hash.'}
    spec=manifest['old_selected_design_spec'];h=hashlib.sha256(b'teacher-native-surface-v5.1')
    for prefix in ['wav_','wav_up_']:h.update(np.ascontiguousarray(np.stack([new[prefix+a] for a in 'xyz'],-1),dtype='<f8').tobytes())
    h.update(json.dumps(spec,sort_keys=True,separators=(',',':'),allow_nan=False).encode());h.update(b'[0.0]');oldmatch['recomputed_old_v5_geometry_hash_from_new_arrays']=h.hexdigest();dump(OUT/'old45_identity_verification.json',oldmatch)
    before=json.loads((OUT/'protected_files_before.json').read_text(encoding='utf-8'));after=protected();dump(OUT/'protected_files_after.json',after)
    changed=[p for p,h in before['files'].items() if after['files'].get(p)!=h];added=[p for p in after['files'] if p not in before['files']];protection={'protected_file_count_before':len(before['files']),'protected_file_count_after':len(after['files']),'changed_or_missing':changed,'new_outside_allowed_output_and_tool_dirs':added,'all_unchanged':not changed and not added,'excluded_transient_dirs':['.git','__pycache__','.pytest_cache','.venv','node_modules'],'roots':before['roots']};dump(OUT/'protected_file_verification.json',protection)
    successful=[r for r in results if r['generation_status']=='GENERATED'];intersection=[r for r in successful if r['surface_intersection_count']]
    summary={'counts':counts,'valid_designs':[r['design_index'] for r in results if r['classification']=='VALID'],'invalid_designs':[r['design_index'] for r in results if r['classification']=='INVALID'],'ambiguous_designs':[r['design_index'] for r in results if r['classification']=='AMBIGUOUS'],'single_valued_both_count':sum(r['single_valued_graph'] for r in successful),'lower_xy_fold_designs':[r['design_index'] for r in successful if not r['lower_diagnostics']['single_valued_xy_graph_certificate']],'upper_xy_graph_count':sum(r['upper_diagnostics']['single_valued_xy_graph_certificate'] for r in successful),'all_six_sections_single_valued_count':sum(r['defined_section_count']==6 for r in successful),'self_intersection_total':sum(r['self_intersection_count'] for r in successful),'cross_triangle_pair_count_range':[min(r['surface_intersection_count'] for r in intersection),max(r['surface_intersection_count'] for r in intersection)] if intersection else None,'intersection_x_over_L_extent':[min(r['intersection_region']['x_over_L_range'][0] for r in intersection),max(r['intersection_region']['x_over_L_range'][1] for r in intersection)] if intersection else None,'intersection_y_program_extent':[min(r['intersection_region']['min_program'][1] for r in intersection),max(r['intersection_region']['max_program'][1] for r in intersection)] if intersection else None,'all_body_transform_invariant':all(r['body_transform_invariant'] for r in successful),'all_diagonal_crossing_presence_agrees':all(r['opposite_diagonal_sensitivity']['crossing_presence_agrees'] for r in successful),'maximum_leading_edge_gap':max(r['boundary_checks']['leading_edge_max_gap'] for r in successful),'maximum_symmetry_y_error':max(r['boundary_checks']['symmetry_max_abs_y'] for r in successful),'all_open_base':all(r['boundary_checks']['base']=='OPEN_BASE' for r in successful),'normal_issue_total':sum(r['normal_issue_count'] for r in successful),'per_surface_intentional_tip_degenerate_faces':sorted(set(r['lower_diagnostics']['allowed_tip_degenerate_count'] for r in successful)),'case':'CASE_C_MAJORITY_INVALID_WITH_ISOLATED_VALID_SUPPORT' if counts['INVALID']>26 else ('CASE_A_MAJORITY_VALID' if counts['VALID']>26 else 'CASE_B_MIXED'),'current_v5_next_stage':'BLOCKED' if results[44]['classification']!='VALID' else 'READY','old45':results[44],'candidate_designs':[c['design_index'] for c in candidates],'correlations':correlations,'protected_files':protection,'audit_elapsed_s_total':sum(r.get('audit_elapsed_s',0) for r in results)}
    dump(OUT/'analysis_summary.json',summary)
    # Add diagnosis runtime to the separate audit ledger without changing any
    # frozen classifications, source file, or previous V5 budget ledger.
    budget=json.loads((OUT/'geometry_audit_budget.json').read_text(encoding='utf-8'));budget['audit_cpu_wall_s_sum']=summary['audit_elapsed_s_total'];budget['generation_status_counts']={s:sum(c['status']==s for c in budget['calls']) for s in ['GENERATED','GENERATION_FAILED']};dump(OUT/'geometry_audit_budget.json',budget)
    print(json.dumps({k:summary[k] for k in ['counts','valid_designs','single_valued_both_count','self_intersection_total','candidate_designs','current_v5_next_stage','cross_triangle_pair_count_range']},ensure_ascii=False))
if __name__=='__main__':main()

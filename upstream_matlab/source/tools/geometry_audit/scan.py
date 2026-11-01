"""Audit saved native geometry. Never calls MATLAB or aerodynamic models."""
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from prepare import ROOT,OUT,PARAMS,sha,dump
from mesh_checks import mesh_diagnostics,intersections,section_compare,triangulate,area_normals

def native_hash(lower,upper,inputs):
    h=hashlib.sha256(b'native-mesh-inputs-audit-v1\0')
    for x in [np.asarray(inputs),lower,upper]:
        x=np.asarray(x,dtype='<f8',order='C');h.update(json.dumps(list(x.shape)).encode());h.update(x.tobytes())
    return h.hexdigest()

def audit(job,opposite=True):
    start=time.perf_counter();case=Path(job['case_dir']);status=json.loads((case/'generation_status.json').read_text(encoding='utf-8-sig'))
    result=dict(job);result['generation_status']=status['status'];result['generation_elapsed_s']=status.get('elapsed_s');result['classification_rules_sha256']=sha(OUT/'classification_rules.json')
    if status['status']!='GENERATED':
        result.update(classification='GENERATION_FAILED',failure_reason=status.get('error_message','No native geometry'));return result
    m=loadmat(case/'native_geometry.mat');lower=np.stack([m['wav_'+a] for a in ['x','y','z']],-1);upper=np.stack([m['wav_up_'+a] for a in ['x','y','z']],-1)
    result['mat_sha256']=sha(case/'native_geometry.mat');result['geometry_hash']=native_hash(lower,upper,job['inputs10'])
    result['numeric_checks']={'nan_count':int(np.isnan(lower).sum()+np.isnan(upper).sum()),'inf_count':int(np.isinf(lower).sum()+np.isinf(upper).sum()),'complex_count':int(np.count_nonzero(np.imag(lower))+np.count_nonzero(np.imag(upper))),'lower_shape':list(lower.shape),'upper_shape':list(upper.shape)}
    if any(result['numeric_checks'][k] for k in ['nan_count','inf_count','complex_count']):
        result.update(classification='INVALID',failure_reason='Nonfinite/complex native coordinates');return result
    length=float(np.ptp(np.concatenate([lower.reshape(-1,3),upper.reshape(-1,3)]),axis=0).max());eps=max(128*np.finfo(float).eps*length,1e-9*length);area_eps=1e-14*length**2
    result.update(length_scale=length,length_tolerance=eps,area_tolerance=area_eps,units='m under existing V5 1 metre/program-unit assumption; physical scale unconfirmed')
    ld,lt,lf,li=mesh_diagnostics(lower,eps,area_eps);ud,ut,uf,ui=mesh_diagnostics(upper,eps,area_eps)
    result['lower_diagnostics']=ld;result['upper_diagnostics']=ud
    le_error=float(np.linalg.norm(lower[:,0]-upper[:,0],axis=1).max());leading=np.stack([lower[:-1,0],lower[1:,0]],1)
    cross=intersections(lt,ut,eps,area_eps,boundary=leading if le_error<=4*eps else np.empty((0,2,3)))
    sl=intersections(lt,lt,eps,area_eps,self_mode=True);su=intersections(ut,ut,eps,area_eps,self_mode=True)
    for records,ia,ib in [(cross['records'],li,ui),(sl['records'],li,li),(su['records'],ui,ui)]:
        for r in records:
            r['native_triangle_a']=int(ia[r['triangle_a']]);r['native_triangle_b']=int(ib[r['triangle_b']])
    dump(case/'intersections.json',{'coordinate_system':'program','count_definition':'Triangle-pair contacts not on actual allowed boundary; native half body; multiple pairs can describe the same curve. No unique-crossing count claimed.','upper_lower':cross,'lower_self':sl,'upper_self':su})
    result['intersection_diagnostics']={k:{a:b for a,b in v.items() if a!='records'} for k,v in [('upper_lower',cross),('lower_self',sl),('upper_self',su)]}
    if cross['records']:
        points=np.concatenate([r['points_program'] for r in cross['records']]);result['intersection_region']={'min_program':points.min(0).tolist(),'max_program':points.max(0).tolist(),'x_over_L_range':[float(points[:,0].min()/length),float(points[:,0].max()/length)],'y_over_halfspan_range':[float(points[:,1].min()/lower[:,:,1].max()),float(points[:,1].max()/lower[:,:,1].max())]}
    else:result['intersection_region']=None
    x0=float(min(lower[:,:,0].min(),upper[:,:,0].min()));x1=float(max(lower[:,:,0].max(),upper[:,:,0].max()))
    sections=[section_compare(lt,ut,x0+f*(x1-x0),eps) for f in [.1,.25,.5,.75,.9,1.]]
    ok=[s for s in sections if s['status']=='OK'];worst=min(ok,key=lambda s:s['min_thickness']) if ok else None
    if cross['records']:
        # Extra intersection section chosen using observed crossing x, does not
        # replace the six predeclared sections or change geometry/class rules.
        ix=float(np.median(points[:,0]));extra=section_compare(lt,ut,ix,eps);extra['purpose']='Median x of actual triangle intersection points';sections.append(extra)
    dump(case/'sections.json',{'method':'Native triangle / x-plane intersections; union y breakpoints on common domain; no extrapolation','predeclared_x_over_L':[.1,.25,.5,.75,.9,1.],'sections':sections})
    boundary={'leading_edge_max_gap':le_error,'symmetry_max_abs_y':float(max(np.abs(lower[-1,:,1]).max(),np.abs(upper[-1,:,1]).max())),'all_native_y_nonnegative':bool(min(lower[:,:,1].min(),upper[:,:,1].min())>=-eps),'tip_max_gap':float(np.linalg.norm(lower[0]-upper[0],axis=1).max()),'tail_x_max_deviation':float(max(np.abs(lower[:,-1,0]-x1).max(),np.abs(upper[:,-1,0]-x1).max())),'base':'OPEN_BASE','symmetry':'Each half-skin is mirrored at y=0. Upper/lower need not meet on this plane. No cap added.','half_to_full_self_sufficiency':'If y>=0 and y=0 confined to symmetry edge, reflected interiors are disjoint; native-half self tests suffice plus seam continuity.'}
    boundary['non_symmetry_interior_vertices_on_y0']=int(np.sum(np.abs(lower[1:-1,1:-1,1])<=eps)+np.sum(np.abs(upper[1:-1,1:-1,1])<=eps))
    result['boundary_checks']=boundary
    result.update(surface_intersection_count=cross['count'],self_intersection_count=sl['count']+su['count'],degenerate_face_count=ld['degenerate_face_count']+ud['degenerate_face_count'],allowed_tip_degenerate_count=ld['allowed_tip_degenerate_count']+ud['allowed_tip_degenerate_count'],normal_issue_count=ld['normal_issue_count']+ud['normal_issue_count'],min_section_thickness=worst['min_thickness'] if worst else None,negative_thickness_fraction=sum(s['negative_count'] for s in ok)/sum(s['breakpoint_count'] for s in ok) if ok else None,negative_thickness_count=sum(s['negative_count'] for s in ok),worst_section_location=worst['worst_location'] if worst else None,tail_min_thickness=ok[-1]['min_thickness'] if ok else None,first_sampled_negative_location=next((s['first_negative_location'] for s in ok if s['first_negative_location']),None),single_valued_graph=ld['single_valued_xy_graph_certificate'] and ud['single_valued_xy_graph_certificate'],body_transform_invariant=all(v['body_invariant'] for v in [cross,sl,su]))
    result['defined_section_count']=len(ok)
    result['minimum_defined_section_thickness']=result['min_section_thickness']
    result['defined_section_negative_fraction']=result['negative_thickness_fraction']
    result['tail_min_thickness']=sections[5].get('min_thickness')
    if len(ok)!=6:
        result['min_section_thickness']=None
        result['negative_thickness_fraction']=None
    result['section_minimum_vertical_gap']=min((s.get('min_thickness',s.get('minimum_branch_gap')) for s in sections[:6] if s.get('min_thickness',s.get('minimum_branch_gap')) is not None),default=None)
    result['vertical_gap_semantics']='Unique thickness only for single-valued sections; multivalued branch gaps are diagnostics, not a thickness field.'
    reasons=[];ambiguous=[]
    if result['surface_intersection_count']:reasons.append('Actual non-leading-edge upper/lower triangle intersection')
    if result['self_intersection_count']:reasons.append('Actual surface self-intersection')
    if ld['unapproved_degenerate_count']+ud['unapproved_degenerate_count']:reasons.append('Undocumented degenerate native faces')
    if not result['single_valued_graph']:ambiguous.append('Single-valued XY graph not certified')
    if result['normal_issue_count']:ambiguous.append('Adjacent native normals reverse')
    if le_error>4*eps or boundary['symmetry_max_abs_y']>4*eps or not boundary['all_native_y_nonnegative']:ambiguous.append('Boundary connectivity unresolved')
    if len(ok)!=6:ambiguous.append('Some required sections unresolved')
    if worst and worst['min_thickness']<-eps and not result['surface_intersection_count']:ambiguous.append('Negative thickness without triangle crossing confirmation')
    if not result['body_transform_invariant']:ambiguous.append('Rigid-transform implementation verification failed')
    result['classification']='INVALID' if reasons else ('AMBIGUOUS' if ambiguous else 'VALID');result['failure_reason']='; '.join(reasons+ambiguous) or 'All frozen native-mesh checks passed; OPEN_BASE declared'
    if opposite:
        alt_l,_=triangulate(lower,True);alt_u,_=triangulate(upper,True);al,_=area_normals(alt_l);au,_=area_normals(alt_u)
        alt=intersections(alt_l[al>area_eps],alt_u[au>area_eps],eps,area_eps,boundary=leading,body_verify=False)
        result['opposite_diagonal_sensitivity']={k:v for k,v in alt.items() if k!='records'}
        result['opposite_diagonal_sensitivity']['crossing_presence_agrees']=bool(cross['count'])==bool(alt['count'])
    result['audit_elapsed_s']=time.perf_counter()-start
    return result

def write_table(results):
    rows=[]
    for r in results:
        row={k:r.get(k) for k in ['design_index','original_row_index','generation_status','geometry_hash','generation_elapsed_s','min_section_thickness','negative_thickness_fraction','surface_intersection_count','self_intersection_count','degenerate_face_count','allowed_tip_degenerate_count','normal_issue_count','classification','failure_reason','stored_L_D','stored_Vol_eff','single_valued_graph','body_transform_invariant','audit_elapsed_s']}
        row.update(r['parameters']);row['inputs10']=json.dumps(r['inputs10']);rows.append(row)
    if rows:
        with (OUT/'pareto_geometry_audit.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    dump(OUT/'pareto_geometry_audit.json',{'schema':'native-pareto-geometry-audit-1','source_manifest_sha256':sha(OUT/'source_manifest.json'),'rules_sha256':sha(OUT/'classification_rules.json'),'count_definition':'Native half-skin triangle pairs, not number of unique intersection curves','results':results})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--start',type=int,default=1);ap.add_argument('--end',type=int,default=52);ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    jobs=json.loads((OUT/'jobs.json').read_text(encoding='utf-8'));results=[]
    for j in jobs:
        p=Path(j['case_dir'])/'audit.json'
        if not args.start<=j['design_index']<=args.end:
            if p.exists():results.append(json.loads(p.read_text(encoding='utf-8')))
            continue
        if args.resume and p.exists():r=json.loads(p.read_text(encoding='utf-8'))
        else:
            r=audit(j);dump(p,r)
        results.append(r);write_table(results)
        print(json.dumps({k:r.get(k) for k in ['design_index','classification','surface_intersection_count','self_intersection_count','min_section_thickness','audit_elapsed_s']},ensure_ascii=False),flush=True)
    write_table(sorted(results,key=lambda r:r['design_index']))
if __name__=='__main__':main()

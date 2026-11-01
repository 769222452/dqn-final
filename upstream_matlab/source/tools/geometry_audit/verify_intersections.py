"""Second numerical formulation: segment/triangle barycentric intersections.

Validates saved cross-surface curves independently of plane-interval overlap.
No geometry generation and no changes to audit classifications.
"""
import json
import numpy as np
from scipy.io import loadmat
from mesh_checks import triangulate,unique_points
from prepare import OUT,dump

def segment_triangle(p,q,tri,tol=1e-10):
    # Solve p+t*(q-p) = a+u*(b-a)+v*(c-a) directly. Independent 3x3 solve.
    matrix=np.column_stack([q-p,-(tri[1]-tri[0]),-(tri[2]-tri[0])])
    if abs(np.linalg.det(matrix))<1e-18:return None
    t,u,v=np.linalg.solve(matrix,tri[0]-p)
    if -tol<=t<=1+tol and u>=-tol and v>=-tol and u+v<=1+tol:return p+t*(q-p)
    return None

def segment_method(a,b):
    points=[]
    for first,second in [(a,b),(b,a)]:
        for i,j in [(0,1),(1,2),(2,0)]:
            p=segment_triangle(first[i],first[j],second)
            if p is not None:points.append(p)
    return unique_points(points,1e-9)

def main():
    results=[]
    for case in sorted((OUT/'cases').glob('design_*')):
        if not (case/'audit.json').exists():continue
        r=json.loads((case/'audit.json').read_text(encoding='utf-8'))
        if r['generation_status']!='GENERATED':continue
        m=loadmat(case/'native_geometry.mat');lt,_=triangulate(np.stack([m['wav_'+a] for a in 'xyz'],-1));ut,_=triangulate(np.stack([m['wav_up_'+a] for a in 'xyz'],-1))
        records=json.loads((case/'intersections.json').read_text(encoding='utf-8'))['upper_lower']['records'];confirmed=0;interior=0;pointonly=0;coplanar=0;unconfirmed=[];maximum_error=0.;lengths=[]
        for rec in records:
            a=lt[rec['native_triangle_a']];b=ut[rec['native_triangle_b']];expected=np.asarray(rec['points_program']);actual=segment_method(a,b)
            if rec['kind']=='coplanar':coplanar+=1;continue
            if not len(actual):unconfirmed.append(rec);continue
            error=float(np.linalg.norm(expected[:,None,:]-actual[None,:,:],axis=2).min(1).max());maximum_error=max(maximum_error,error)
            if error<5e-8:confirmed+=1
            else:unconfirmed.append({**rec,'second_method_error':error})
            midpoint=expected.mean(0);proper=True
            for t in [a,b]:
                uv=np.linalg.lstsq((t[1:]-t[0]).T,midpoint-t[0],rcond=None)[0];proper &= bool(min(uv[0],uv[1],1-uv.sum())>1e-9)
            if len(expected)>1:
                length=float(np.linalg.norm(expected[-1]-expected[0]));lengths.append(length);interior+=int(proper and length>5e-9)
            else:pointonly+=1
        results.append({'design_index':r['design_index'],'record_count':len(records),'confirmed_by_independent_edge_triangle_solve':confirmed,'proper_interior_crossing_pairs':interior,'point_only':pointonly,'coplanar':coplanar,'unconfirmed_count':len(unconfirmed),'maximum_endpoint_difference_m':maximum_error,'segment_length_sum_with_possible_duplicates_m':sum(lengths),'maximum_segment_length_m':max(lengths,default=0),'unconfirmed_records':unconfirmed})
    dump(OUT/'independent_intersection_verification.json',{'method':'Six segment/triangle 3x3 barycentric solves per recorded pair; independent from plane interval narrow phase','results':results,'all_confirmed':all(r['unconfirmed_count']==0 and r['coplanar']==0 for r in results)})
    print(json.dumps({'designs':len(results),'unconfirmed':sum(r['unconfirmed_count'] for r in results),'proper_interior_min':min(r['proper_interior_crossing_pairs'] for r in results),'maximum_error':max(r['maximum_endpoint_difference_m'] for r in results)}))
if __name__=='__main__':main()

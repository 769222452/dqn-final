"""Independent native-mesh audit. No project generator/aero imports.

Broad phase: native triangle AABB hierarchy. Narrow phase: plane/triangle
sections and collinear interval overlap, or coplanar polygon clipping.
All tolerances are fixed before scanning; no positive-thickness repair.
"""
import numpy as np
from dataclasses import dataclass

def triangulate(grid, opposite=False):
    m,n=grid.shape[:2]; ids=np.arange(m*n).reshape(m,n)
    a=ids[:-1,:-1].ravel(); b=ids[1:,:-1].ravel(); c=ids[1:,1:].ravel(); d=ids[:-1,1:].ravel()
    faces=np.stack([np.stack([a,b,d],1),np.stack([b,c,d],1)],1).reshape(-1,3) if opposite else np.stack([np.stack([a,b,c],1),np.stack([a,c,d],1)],1).reshape(-1,3)
    return grid.reshape(-1,3)[faces],faces

def area_normals(tris):
    raw=np.cross(tris[:,1]-tris[:,0],tris[:,2]-tris[:,0]); length=np.linalg.norm(raw,axis=1)
    return length/2,raw/np.maximum(length[:,None],np.finfo(float).tiny)

@dataclass
class Node:
    lo:np.ndarray
    hi:np.ndarray
    ids:object=None
    left:object=None
    right:object=None

def tree(lo,hi,ids):
    low=lo[ids].min(0); high=hi[ids].max(0)
    if len(ids)<=12: return Node(low,high,ids)
    centres=(lo[ids]+hi[ids])/2; ax=np.argmax(np.ptp(centres,axis=0)); order=np.argsort(centres[:,ax],kind='stable');half=len(ids)//2
    return Node(low,high,None,tree(lo,hi,ids[order[:half]]),tree(lo,hi,ids[order[half:]]))

def aabb_pairs(a,b,eps,self_mode=False):
    alo=a.min(1);ahi=a.max(1);blo=b.min(1);bhi=b.max(1)
    ta=tree(alo,ahi,np.arange(len(a)));tb=ta if self_mode else tree(blo,bhi,np.arange(len(b)))
    stack=[(ta,tb)];chunks=[]
    while stack:
        x,y=stack.pop()
        if np.any(x.hi+eps<y.lo) or np.any(y.hi+eps<x.lo):continue
        if x.ids is not None and y.ids is not None:
            ix=x.ids[:,None];iy=y.ids[None,:]
            mask=np.all(ahi[ix]+eps>=blo[iy],axis=2)&np.all(bhi[iy]+eps>=alo[ix],axis=2)
            if self_mode:mask &= iy>ix
            u,v=np.where(mask)
            if len(u):chunks.append(np.stack([x.ids[u],y.ids[v]],1))
        elif y.ids is not None or (x.ids is None and np.prod(x.hi-x.lo)>=np.prod(y.hi-y.lo)):
            stack.extend([(x.left,y),(x.right,y)])
        else:stack.extend([(x,y.left),(x,y.right)])
    if not chunks:return np.empty((0,2),int)
    pairs=np.concatenate(chunks)
    return np.unique(pairs,axis=0) if self_mode else pairs

def unique_points(points,eps):
    result=[]
    for p in points:
        if not any(np.linalg.norm(p-q)<=eps for q in result):result.append(p)
    return np.asarray(result).reshape(-1,3)

def plane_section(tri,dist,eps):
    points=[tri[i] for i in range(3) if abs(dist[i])<=eps]
    for i,j in [(0,1),(1,2),(2,0)]:
        if (dist[i]<-eps and dist[j]>eps) or (dist[j]<-eps and dist[i]>eps):
            points.append(tri[i]+dist[i]/(dist[i]-dist[j])*(tri[j]-tri[i]))
    return unique_points(points,eps)

def cross2(a,b):return a[0]*b[1]-a[1]*b[0]

def coplanar_clip(a,b,normal,eps):
    axes=[i for i in range(3) if i!=np.argmax(np.abs(normal))]
    orientation=np.sign(cross2(b[1,axes]-b[0,axes],b[2,axes]-b[0,axes])); poly=list(a.copy())
    for i,j in [(0,1),(1,2),(2,0)]:
        if not poly:break
        edge=b[j,axes]-b[i,axes];tol=eps*np.linalg.norm(edge)
        def dist(p):return orientation*cross2(edge,p[axes]-b[i,axes])
        result=[];prev=poly[-1];dp=dist(prev)
        for cur in poly:
            dc=dist(cur);pi=dp>=-tol;ci=dc>=-tol
            if pi!=ci and abs(dp-dc)>np.finfo(float).tiny:result.append(prev+dp/(dp-dc)*(cur-prev))
            if ci:result.append(cur)
            prev=cur;dp=dc
        poly=result
    return unique_points(poly,eps)

def triangle_intersection(a,b,eps,area_eps):
    aa,an=area_normals(a[None]);ba,bn=area_normals(b[None]);na=an[0];nb=bn[0]
    if aa[0]<=area_eps or ba[0]<=area_eps:return np.empty((0,3)),'degenerate'
    da=(b-a[0])@na;db=(a-b[0])@nb
    # Geometric tolerance is not a licence to move vertices onto a nearby
    # plane: that amplifies errors for almost parallel triangles. Plane signs
    # use a roundoff bound; eps only controls contact classification.
    pred_eps=128*np.finfo(float).eps*max(np.abs(a).max(),np.abs(b).max(),1.)
    if np.min(da)>pred_eps or np.max(da)<-pred_eps or np.min(db)>pred_eps or np.max(db)<-pred_eps:return np.empty((0,3)),'separate'
    direction=np.cross(na,nb);ln=np.linalg.norm(direction)
    if max(np.abs(da).max(),np.abs(db).max())<=pred_eps:
        pts=coplanar_clip(a,b,na,pred_eps)
        return pts,'coplanar'
    if ln<=128*np.finfo(float).eps:return np.empty((0,3)),'parallel'
    pa=plane_section(a,db,pred_eps);pb=plane_section(b,da,pred_eps)
    if not len(pa) or not len(pb):return np.empty((0,3)),'separate'
    direction/=ln; origin=pa[0];ua=(pa-origin)@direction;ub=(pb-origin)@direction
    low=max(ua.min(),ub.min());high=min(ua.max(),ub.max())
    if high<low-pred_eps:return np.empty((0,3)),'separate'
    if high-low<=eps:return (origin+(low+high)/2*direction)[None,:],'point'
    return np.stack([origin+low*direction,origin+high*direction]),'segment'

def distance_to_segments(points,segments):
    if not len(segments):return np.full(len(points),np.inf)
    a=segments[:,0];d=segments[:,1]-a;den=np.sum(d*d,1)
    t=np.sum((points[:,None,:]-a)*d,axis=2)/np.maximum(den,1e-300)
    proj=a+np.clip(t,0,1)[:,:,None]*d
    return np.linalg.norm(points[:,None,:]-proj,axis=2).min(1)

def contact_allowed(pts,a,b,eps,boundary=None):
    # Check full segment/polygon (vertices and edge midpoints), not just one end.
    probe=np.concatenate([pts,(pts+np.roll(pts,1,axis=0))/2])
    if boundary is not None:return bool(np.all(distance_to_segments(probe,boundary)<=4*eps))
    shared=unique_points([p for p in a if np.min(np.linalg.norm(b-p,axis=1))<=eps],eps)
    if not len(shared):return False
    if len(shared)==1:return bool(np.all(np.linalg.norm(probe-shared[0],axis=1)<=4*eps))
    if len(shared)==2:return bool(np.all(distance_to_segments(probe,shared[None,:,:])<=4*eps))
    return False # coincident full triangles are not legitimate adjacency

def intersections(a,b,eps,area_eps,self_mode=False,boundary=None,body_verify=True):
    pairs=aabb_pairs(a,b,eps,self_mode)
    broad=len(pairs)
    if not broad:return {'broad_phase_pairs':0,'narrow_phase_pairs':0,'count':0,'allowed_contacts':0,'body_invariant':True,'records':[]}
    _,na=area_normals(a);_,nb=area_normals(b)
    aa=a[pairs[:,0]];bb=b[pairs[:,1]]
    da=np.einsum('nkj,nj->nk',bb-aa[:,0,None,:],na[pairs[:,0]])
    db=np.einsum('nkj,nj->nk',aa-bb[:,0,None,:],nb[pairs[:,1]])
    keep=(da.min(1)<=eps)&(da.max(1)>=-eps)&(db.min(1)<=eps)&(db.max(1)>=-eps)
    pairs=pairs[keep];aa=aa[keep];bb=bb[keep]
    allowed=0;records=[];invariant=True;max_transform_error=0.
    # Shared-edge neighbours cannot cross beyond their edge unless coplanar
    # on the same side. This excludes only a mathematically proven contact.
    if self_mode and len(pairs):
        equal=np.linalg.norm(aa[:,:,None,:]-bb[:,None,:,:],axis=3)<=eps
        sharecount=np.sum(np.any(equal,axis=2),axis=1)
        normals_cross=np.linalg.norm(np.cross(na[pairs[:,0]],nb[pairs[:,1]]),axis=1)
        safe=(sharecount==2)&(normals_cross>1e-10)
        # If a pair shares exactly one point and either triangle's other two
        # vertices are strictly on one side of the other plane, the only
        # possible contact is that shared point. This is a proof, not a mask.
        pd_a=np.einsum('nkj,nj->nk',bb-aa[:,0,None,:],na[pairs[:,0]])
        pd_b=np.einsum('nkj,nj->nk',aa-bb[:,0,None,:],nb[pairs[:,1]])
        roundoff=128*np.finfo(float).eps*max(np.abs(a).max(),np.abs(b).max(),1.)
        one_side_a=(np.sum(pd_a>roundoff,axis=1)==2)|(np.sum(pd_a<-roundoff,axis=1)==2)
        one_side_b=(np.sum(pd_b>roundoff,axis=1)==2)|(np.sum(pd_b<-roundoff,axis=1)==2)
        safe |= (sharecount==1)&(one_side_a|one_side_b)
        for k in np.where((sharecount==2)&~safe)[0]:
            sa=np.where(np.any(equal[k],axis=1))[0];othera=np.where(~np.any(equal[k],axis=1))[0][0];otherb=np.where(~np.any(equal[k],axis=0))[0][0]
            edge=aa[k,sa[1]]-aa[k,sa[0]]
            qa=np.cross(edge,aa[k,othera]-aa[k,sa[0]]);qb=np.cross(edge,bb[k,otherb]-aa[k,sa[0]])
            if np.dot(qa,qb)<0:safe[k]=True
        allowed=int(safe.sum());pairs=pairs[~safe];aa=aa[~safe];bb=bb[~safe]
    for pair,x,y in zip(pairs,aa,bb):
        pts,kind=triangle_intersection(x,y,eps,area_eps)
        if not len(pts):continue
        is_allowed=contact_allowed(pts,x,y,eps,boundary)
        if body_verify:
            rot=np.array([1.,-1.,-1.]);pts_body,kind_body=triangle_intersection(x*rot,y*rot,eps,area_eps)
            allow_body=contact_allowed(pts_body,x*rot,y*rot,eps,None if boundary is None else boundary*rot) if len(pts_body) else False
            invariant &= bool(len(pts_body)>0 and is_allowed==allow_body)
            if len(pts_body):max_transform_error=max(max_transform_error,float(np.linalg.norm(pts[:,None,:]*rot-pts_body[None,:,:],axis=2).min(1).max()))
        if is_allowed:allowed+=1;continue
        records.append({'triangle_a':int(pair[0]),'triangle_b':int(pair[1]),'kind':kind,'points_program':pts.tolist()})
    return {'broad_phase_pairs':broad,'narrow_phase_pairs':len(pairs),'count':len(records),'allowed_contacts':allowed,'body_invariant':bool(invariant),'body_max_point_error':max_transform_error,'records':records}

def segment_cross2(a,b,c,d,eps):
    v=b-a;w=d-c;det=cross2(v,w)
    if abs(det)<=eps*max(np.linalg.norm(v),np.linalg.norm(w)):
        if abs(cross2(c-a,v))>eps*np.linalg.norm(v):return False
        den=np.dot(v,v)
        if den<=eps**2:return np.linalg.norm(a-c)<=eps
        t=sorted([np.dot(c-a,v)/den,np.dot(d-a,v)/den]);return min(t[1],1)>max(t[0],0)+eps
    t=cross2(c-a,w)/det;u=cross2(c-a,v)/det
    return -eps<=t<=1+eps and -eps<=u<=1+eps

def mesh_diagnostics(grid,eps,area_eps):
    tris,faces=triangulate(grid);area,normal=area_normals(tris)
    tip_collapsed=bool(np.max(np.linalg.norm(grid[0]-grid[0,0],axis=1))<=eps)
    deg=area<=area_eps
    allowed=deg & (np.sum(faces//grid.shape[1]==0,axis=1)>=2) & tip_collapsed
    # Discard intentional coincident tip vertices for the simple projected perimeter.
    boundary=np.concatenate([grid[:,0],grid[-1,1:],grid[-2::-1,-1],grid[0,-2:0:-1]])[:,:2]
    cleaned=[]
    for p in boundary:
        if not cleaned or np.linalg.norm(p-cleaned[-1])>eps:cleaned.append(p)
    boundary=np.asarray(cleaned)
    if np.linalg.norm(boundary[-1]-boundary[0])<=eps:boundary=boundary[:-1]
    boundary_cross=0
    for i in range(len(boundary)):
        for j in range(i+2,len(boundary)):
            if i==0 and j==len(boundary)-1:continue
            if segment_cross2(boundary[i],boundary[(i+1)%len(boundary)],boundary[j],boundary[(j+1)%len(boundary)],eps):boundary_cross+=1
    signed=np.cross(tris[:,1]-tris[:,0],tris[:,2]-tris[:,0])[:,2]/2
    positive=bool(np.all(signed[~deg]>area_eps));negative=bool(np.all(signed[~deg]<-area_eps))
    graph=bool((positive or negative) and boundary_cross==0 and np.all(deg==allowed))
    edge_map={};dots=[]
    for k,f in enumerate(faces):
        if deg[k]:continue
        for u,v in [(f[0],f[1]),(f[1],f[2]),(f[2],f[0])]:
            e=tuple(sorted((int(u),int(v))))
            if e in edge_map:dots.append(float(np.dot(normal[k],normal[edge_map[e]])))
            else:edge_map[e]=k
    edges=np.concatenate([np.linalg.norm(np.diff(grid,axis=0),axis=2).ravel(),np.linalg.norm(np.diff(grid,axis=1),axis=2).ravel()]);positive_edges=edges[edges>eps]
    _,cnt=np.unique(np.round(grid.reshape(-1,3)/eps).astype(np.int64),axis=0,return_counts=True)
    return {'shape':list(grid.shape[:2]),'ranges':{'min':grid.min((0,1)).tolist(),'max':grid.max((0,1)).tolist()},'duplicate_points_within_quantized_tolerance':int(np.sum(cnt-1)), 'duplicate_points_exact':int(grid.size//3-len(np.unique(grid.reshape(-1,3),axis=0))), 'degenerate_face_count':int(deg.sum()),'allowed_tip_degenerate_count':int(allowed.sum()),'unapproved_degenerate_count':int((deg&~allowed).sum()),'zero_length_edges':int(np.sum(edges<=eps)),'long_edge_diagnostic_count':int(np.sum(edges>20*np.median(positive_edges))),'max_edge_length':float(edges.max()),'median_positive_edge_length':float(np.median(positive_edges)),'normal_issue_count':int(np.sum(np.asarray(dots)<-1e-10)),'max_adjacent_normal_angle_deg':float(np.degrees(np.arccos(np.clip(min(dots) if dots else 1.,-1,1)))),'projected_negative_faces':int(np.sum(signed[~deg]<-area_eps)),'projected_boundary_intersections':boundary_cross,'single_valued_xy_graph_certificate':graph,'certificate_basis':'Native disk after collapsed-tip identification, strictly consistent XY triangle orientation and simple XY boundary => injective PL map.'},tris[~deg],faces[~deg],np.where(~deg)[0]

def section_curve(tris,x,eps):
    mask=(tris[:,:,0].min(1)<=x+eps)&(tris[:,:,0].max(1)>=x-eps)
    points=[];segments=[]
    for t in tris[mask]:
        p=plane_section(t,t[:,0]-x,eps)
        if len(p):points.extend(p)
        if len(p)>=2:
            order=np.argsort(p[:,1]);segments.append([p[order[0]],p[order[-1]]])
    if not points:return None
    points=np.asarray(points);points=points[np.argsort(points[:,1],kind='stable')];group=[];groups=[]
    for p in points:
        if group and abs(p[1]-group[-1][1])>eps:groups.append(np.array(group));group=[]
        group.append(p)
    if group:groups.append(np.array(group))
    multivalued=max(float(np.ptp(g[:,2])) for g in groups)>4*eps
    curve=np.array([g.mean(0) for g in groups])
    seg=np.asarray(segments)
    # Folded branches need not have coincident endpoint y values. Test the
    # complete union of segment breakpoints against every containing segment.
    for yy in curve[:,1]:
        valid=(seg[:,0,1]<=yy+eps)&(seg[:,1,1]>=yy-eps)
        s=seg[valid];dy=s[:,1,1]-s[:,0,1];nonvertical=np.abs(dy)>eps
        s=s[nonvertical];dy=dy[nonvertical]
        zz=s[:,0,2]+np.clip((yy-s[:,0,1])/dy,0,1)*(s[:,1,2]-s[:,0,2])
        if len(zz)>1 and np.ptp(zz)>4*eps:multivalued=True;break
    # Mean merges numerical duplicates only. Ambiguous multiple heights never
    # become a fabricated thickness curve.
    if multivalued:return {'multivalued':True,'raw_points':points.tolist(),'segments':seg}
    return {'multivalued':False,'curve':curve,'segments':np.asarray(segments)}

def section_compare(lower_tris,upper_tris,x,eps):
    lo=section_curve(lower_tris,x,eps);up=section_curve(upper_tris,x,eps)
    if lo is None or up is None:return {'x':float(x),'status':'AMBIGUOUS_SECTION','reason':'No common section'}
    raw={'lower_segments_program':lo['segments'].tolist(),'upper_segments_program':up['segments'].tolist()}
    if lo['multivalued'] or up['multivalued']:
        gaps=[]
        # Preserve each physical segment separately. A multivalued section has
        # no unique thickness; these are explicitly labelled branch gaps only.
        for ls in lo['segments']:
            for us in up['segments']:
                a=max(ls[:,1].min(),us[:,1].min());b=min(ls[:,1].max(),us[:,1].max())
                if b-a<=eps:continue
                yy=np.array([a,b]);lz=np.interp(yy,ls[:,1],ls[:,2]);uz=np.interp(yy,us[:,1],us[:,2]);gaps.append(np.stack([yy,lz-uz],1).tolist())
        flat=np.asarray(gaps).reshape(-1,2) if gaps else np.empty((0,2))
        worst=flat[np.argmin(flat[:,1])] if len(flat) else None
        return {'x':float(x),'status':'AMBIGUOUS_SECTION','reason':'Multivalued z(x,y): no unique thickness','lower_multivalued':lo['multivalued'],'upper_multivalued':up['multivalued'],**raw,'branch_vertical_gaps':gaps,'minimum_branch_gap':float(worst[1]) if worst is not None else None,'worst_branch_location':[float(x),float(worst[0]),float(worst[1])] if worst is not None else None}
    lc=lo['curve'];uc=up['curve'];a=max(lc[0,1],uc[0,1]);b=min(lc[-1,1],uc[-1,1])
    ys=np.sort(np.concatenate([lc[:,1],uc[:,1],[a,b]]));ys=ys[(ys>=a)&(ys<=b)];ys=ys[np.r_[True,np.diff(ys)>eps]]
    zl=np.interp(ys,lc[:,1],lc[:,2]);zu=np.interp(ys,uc[:,1],uc[:,2]);h=zl-zu;neg=h<-eps
    zeros=[]
    for i in range(len(ys)-1):
        if h[i]*h[i+1]<0:zeros.append(float(ys[i]-h[i]*(ys[i+1]-ys[i])/(h[i+1]-h[i])))
    negwidth=0.
    for i in range(len(ys)-1):
        if h[i]<-eps and h[i+1]<-eps:negwidth+=ys[i+1]-ys[i]
        elif h[i]<-eps or h[i+1]<-eps:
            z=ys[i]-h[i]*(ys[i+1]-ys[i])/(h[i+1]-h[i]);negwidth+=z-ys[i] if h[i]<-eps else ys[i+1]-z
    worst=int(np.argmin(h))
    return {'x':float(x),'status':'OK',**raw,'common_y_domain':[float(a),float(b)],'y':ys.tolist(),'lower_z_program':zl.tolist(),'upper_z_program':zu.tolist(),'thickness':h.tolist(),'min_thickness':float(h.min()),'negative_count':int(neg.sum()),'breakpoint_count':len(ys),'negative_fraction':float(neg.mean()),'negative_y_length_fraction':float(negwidth/(b-a)) if b>a else 0.,'zero_crossing_y':zeros,'first_negative_location':None if not neg.any() else [float(x),float(ys[neg][0]),float(h[neg][0])],'worst_location':[float(x),float(ys[worst]),float(h[worst])]}

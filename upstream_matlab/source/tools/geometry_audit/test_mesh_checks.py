import unittest
import numpy as np
from mesh_checks import triangle_intersection,contact_allowed,intersections,section_compare,triangulate,mesh_diagnostics,aabb_pairs

class GeometryChecksTest(unittest.TestCase):
    def setUp(self):
        self.a=np.array([[0.,0,0],[2,0,0],[0,2,0]]);self.eps=1e-9;self.area=1e-14
    def test_disjoint(self):
        pts,_=triangle_intersection(self.a,self.a+[0,0,1],self.eps,self.area);self.assertEqual(len(pts),0)
    def test_proper_transverse_intersection(self):
        b=np.array([[.5,-.2,-1],[.5,1.5,1],[.5,1.5,-1]])
        pts,kind=triangle_intersection(self.a,b,self.eps,self.area)
        self.assertEqual(kind,'segment');self.assertGreater(np.linalg.norm(pts[1]-pts[0]),.5)
        self.assertTrue(np.allclose(pts[:,0],.5));self.assertTrue(np.allclose(pts[:,2],0))
        self.assertFalse(contact_allowed(pts,self.a,b,self.eps))
    def test_coplanar_overlap(self):
        b=self.a+[.3,.2,0];pts,kind=triangle_intersection(self.a,b,self.eps,self.area)
        self.assertEqual(kind,'coplanar');self.assertGreaterEqual(len(pts),3);self.assertFalse(contact_allowed(pts,self.a,b,self.eps))
    def test_shared_edge_only(self):
        b=np.array([[0.,0,0],[2,0,0],[0,-1,0]])
        pts,_=triangle_intersection(self.a,b,self.eps,self.area)
        self.assertTrue(contact_allowed(pts,self.a,b,self.eps))
        result=intersections(np.array([self.a,b]),np.array([self.a,b]),self.eps,self.area,True)
        self.assertEqual(result['count'],0)
    def test_shared_vertex_does_not_hide_overlap(self):
        b=np.array([[0.,0,0],[1,.2,-1],[1,.2,1]])
        pts,_=triangle_intersection(self.a,b,self.eps,self.area);self.assertFalse(contact_allowed(pts,self.a,b,self.eps))
    def test_rigid_invariance(self):
        b=np.array([[.5,-.2,-1],[.5,1.5,1],[.5,1.5,-1]])
        r=intersections(self.a[None],b[None],self.eps,self.area)
        self.assertEqual(r['count'],1);self.assertTrue(r['body_invariant']);self.assertLess(r['body_max_point_error'],1e-10)
    def test_not_exclude_tail(self):
        b=np.array([[.5,-.2,-1],[.5,1.5,1],[.5,1.5,-1]])
        boundary=np.array([[[0,0,0],[2,0,0]]],float)
        r=intersections(self.a[None],b[None],self.eps,self.area,boundary=boundary);self.assertEqual(r['count'],1)
    def test_different_y_grid_same_coordinate_thickness(self):
        low=np.array([[[0,0,2],[1,0,2]],[[0,1,2],[1,1,2]]],float)
        up=np.array([[[0,.2,1],[1,.2,1]],[[0,.8,1],[1,.8,1]]],float)
        lt,_=triangulate(low);ut,_=triangulate(up);s=section_compare(lt,ut,.5,self.eps)
        self.assertEqual(s['status'],'OK');self.assertAlmostEqual(s['common_y_domain'][0],.2);self.assertAlmostEqual(s['common_y_domain'][1],.8)
        self.assertTrue(np.allclose(s['thickness'],1))
    def test_folded_self_surface(self):
        b=np.array([[.5,-.2,-1],[.5,1.5,1],[.5,1.5,-1]])
        t=np.array([self.a,b]);r=intersections(t,t,self.eps,self.area,True);self.assertEqual(r['count'],1)
    def test_collapsed_tip_is_documented_degeneracy(self):
        grid=np.array([[[1,1,0],[1,1,0],[1,1,0]],[[0,0,0],[.5,0,0],[1,0,0]]],float)
        d,*_=mesh_diagnostics(grid,self.eps,self.area)
        self.assertEqual(d['degenerate_face_count'],2);self.assertEqual(d['unapproved_degenerate_count'],0)
        self.assertTrue(d['single_valued_xy_graph_certificate'])
    def test_coplanar_adjacent_fold_is_not_excluded(self):
        b=np.array([[0.,0,0],[2,0,0],[.5,.5,0]])
        t=np.array([self.a,b]);r=intersections(t,t,self.eps,self.area,True);self.assertEqual(r['count'],1)
    def test_near_coplanar_shared_vertex_not_displaced(self):
        a=np.array([[0.,0,0],[1.,0,0],[0,1,1e-10]])
        b=np.array([[1.,0,0],[2,0,.004],[1,1,.00000002]])
        pts,_=triangle_intersection(a,b,5e-9,self.area)
        self.assertTrue(contact_allowed(pts,a,b,5e-9))
    def test_multivalued_section_cannot_be_sorted_into_fake_curve(self):
        lower=np.array([[[0.,0.,2.],[1,0,2],[0,2,2]],[[0,.4,3],[1,.4,3],[0,1.6,3]]])
        upper=np.array([[[0.,0.,1.],[1,0,1],[0,2,1]]])
        s=section_compare(lower,upper,.2,1e-9);self.assertEqual(s['status'],'AMBIGUOUS_SECTION')
    def test_random_noncoplanar_agrees_with_independent_segment_solve(self):
        from verify_intersections import segment_method
        rng=np.random.default_rng(19041)
        for _ in range(300):
            a=rng.normal(size=(3,3));b=rng.normal(size=(3,3))
            first,_=triangle_intersection(a,b,1e-9,1e-14);second=segment_method(a,b)
            self.assertEqual(bool(len(first)),bool(len(second)))
            if len(first):self.assertLess(np.linalg.norm(first[:,None,:]-second[None,:,:],axis=2).min(1).max(),1e-8)
    def test_aabb_does_not_drop_brute_force_hits(self):
        rng=np.random.default_rng(552);a=rng.normal(size=(22,3,3));b=rng.normal(size=(17,3,3));pairs=set(map(tuple,aabb_pairs(a,b,1e-9)))
        for i in range(len(a)):
            for j in range(len(b)):
                pts,_=triangle_intersection(a[i],b[j],1e-9,1e-14)
                if len(pts):self.assertIn((i,j),pairs)

if __name__=='__main__':unittest.main(verbosity=2)

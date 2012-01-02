#!/usr/bin/env python
"""
Test TuLiP code for working with polytopes.

SCL; 18 Dec 2011.
(partially based on tests by Petter Nilsson, summer 2011.)
"""

from tulip.polytope import *
import numpy as np


class projection_test:

    def setUp(self):
        # V1 = np.random.rand(8,4)
        # V2 = np.random.rand(9,4)
        # V3 = np.random.rand(10,5)
        V1 = np.array([[ 0.62172127,  0.67215066,  0.59790871,  0.64533192],
                       [ 0.35934319,  0.13000867,  0.7253058 ,  0.85523852],
                       [ 0.4365557 ,  0.25368766,  0.62158234,  0.55920116],
                       [ 0.40909606,  0.23706418,  0.2801729 ,  0.78390209],
                       [ 0.46798578,  0.29137474,  0.60722839,  0.933466  ],
                       [ 0.87885437,  0.71935124,  0.39073996,  0.66184515],
                       [ 0.11370657,  0.35345949,  0.69318399,  0.393163  ],
                       [ 0.24287919,  0.23532902,  0.57264901,  0.75579302]])
        V2 = np.array([[ 0.46634539,  0.49884031,  0.50573789,  0.50090436],
                       [ 0.10582403,  0.3830428 ,  0.46346761,  0.71697628],
                       [ 0.96552843,  0.61527801,  0.57179316,  0.64408848],
                       [ 0.42259817,  0.99549325,  0.98890162,  0.59631314],
                       [ 0.88061246,  0.07435611,  0.96615703,  0.85504157],
                       [ 0.31295164,  0.24028111,  0.35224679,  0.7266734 ],
                       [ 0.69387742,  0.27138377,  0.43745559,  0.14473992],
                       [ 0.50441162,  0.54725909,  0.54523661,  0.75656769],
                       [ 0.76962887,  0.82288897,  0.30209961,  0.98262237]])
        V3 = np.array([[ 0.40533147,  0.78349094,  0.99726276,  0.2958655 ,  0.91497285],
                       [ 0.67319939,  0.88506473,  0.50311203,  0.44429169,  0.9703398 ],
                       [ 0.40377078,  0.54573565,  0.63045403,  0.48461906,  0.74936346],
                       [ 0.17415397,  0.46370796,  0.72081135,  0.03414818,  0.66476217],
                       [ 0.15794558,  0.8639695 ,  0.4785077 ,  0.05686428,  0.43697716],
                       [ 0.12672406,  0.86770421,  0.12899261,  0.0364164 ,  0.81417427],
                       [ 0.18667199,  0.65907174,  0.45780287,  0.8467233 ,  0.46633635],
                       [ 0.35775067,  0.99905858,  0.22090214,  0.58431796,  0.19371926],
                       [ 0.70008459,  0.04277322,  0.29156081,  0.47845242,  0.38754299],
                       [ 0.2036763 ,  0.33939134,  0.45042596,  0.46070396,  0.48405637]])
        Vproj1 = V1[:,[0,1]]
        Vproj2 = V2[:,[0,1,2]]
        Vproj3 = V3[:,[0,1,2]]
       
        self.P1 = qhull(V1) 
        self.P2 = qhull(V2)
        self.P3 = qhull(V3)
       
        self.Pred1 = qhull(Vproj1)
        self.Pred2 = qhull(Vproj2)
        self.Pred3 = qhull(Vproj3)

    def tearDown(self):
        self.P1 = None
        self.P2 = None
        self.P3 = None
        self.Pred1 = None
        self.Pred2 = None
        self.Pred3 = None

    def test_poly1(self):
        Pred = projection(self.P1,[1,2])
        test1 = mldivide(Pred,self.Pred1)
        test2 = mldivide(self.Pred1,Pred)
        assert (not is_fulldim(test1)) and (not is_fulldim(test2))
    
    def test_poly2(self):
        # Should use fourier-motzkin
        Pred = projection(self.P2,[1,2,3])
        test1 = mldivide(Pred,self.Pred2)
        test2 = mldivide(self.Pred2,Pred)
        assert (not is_fulldim(test1)) and (not is_fulldim(test2))

    def test_poly3(self):
        Pred = projection(self.P3,[1,2,3])
        test1 = mldivide(Pred,self.Pred3)
        test2 = mldivide(self.Pred3,Pred)
        assert (not is_fulldim(test1)) and (not is_fulldim(test2))
        
class union_test:

    def setUp(self):
        # Create some polytopes
        # 2d polytopes
        
        # Polytope with extreme points in (0,0), (0,1), (1,0) and (1,1)
        A = np.array([[ 0. ,-1.],   
                       [-1.  ,0.],  
                       [ 0. , 1.], 
                       [ 1. , 0.]])
        
        b = np.array([0.,0.,1.,1.])
        self.P1 = Polytope(A,b)
        
        # Polytope with extreme points in (1,1), (1,0) and (2,0.5)
        A = np.array([[ 0.447213595499958,  0.894427190999916],
        [-1. , 0.],  
        [0.447213595499958,  -0.894427190999916]])    
        
        b = np.array([1.341640786499874,-1.,0.447213595499958])
        self.P2 = Polytope(A,b)
        
        # Polytope with extreme points in (1,1), (0,1) and (1,2)
        A = np.array([[ 1. ,0.],
                [-np.sqrt(2)/2 , np.sqrt(2)/2] ,
                [ 0. , -1.]])
        
        b = np.array([1.,np.sqrt(2)/2.,-1.])
        self.P3 = Polytope(A,b)
        
        # 3d polytopes
        
        # Polytope with extreme points in (0 0 0), (1 0 0), (0 1 0), (0 0 1), 
        # (0 1 1), (1 0 1) and (1 1 0)
        A = np.array([[0,-1.,0],
                    [0,0,1.],
                    [-1.,0,0],
                    [0,1. ,0],
                    [0.577350269189626,0.577350269189626,0.577350269189626],
                    [ 1.,0,0],
                    [0,0  ,-1.]])
        b = np.array([0,1.,0,1.,1.154700538379252,1.,0])
        self.R1 = Polytope(A,b)
        
        A = np.array([[0,0,-1.],
                    [-0.408248290463863,-0.816496580927726,-0.408248290463863],
                    [-0.408248290463863,-0.408248290463863,-0.816496580927726],
                    [-1.,0,0],
                    [-0.816496580927726,-0.408248290463863,-0.408248290463863],
                    [0,-1.,0],
                    [0,0,1.],
                    [0,1.,0],
                    [1.,0,0]])
        b = np.array([0,-0.408248290463863,-0.408248290463863,0,-0.408248290463863,0,1.,1.,1.])
        self.R2 = Polytope(A,b)

    def tearDown(self):
        self.R1 = None
        self.R2 = None
        self.P1 = None
        self.P2 = None
        self.P3 = None
        
                
    def test_convexunion(self):
        U = union(self.P1,self.P2,True)
        R = Region([self.P1,self.P2],[])
        assert len(U) == 1
        p1 = U.list_poly[0]
        diff1 = mldivide(p1,R)
        diff2 = mldivide(R,p1)
        assert not is_fulldim(diff1)
        assert not is_fulldim(diff2)
        
    def test_nonconvexunion(self):
        U = union(self.P2,self.P3,True)
        assert len(U) == 2
    
    def test_tripleunion(self):
        # Union of three polytopes, where P1 U P2 and P1 U P3 are convex
        U1 = union(self.P1,union(self.P2,self.P3,True),True)
        U2 = union(self.P1,union(self.P3,self.P2,True),True)
        assert len(U1) == 2
        assert len(U2) == 2
        isect = intersect(U1,U2)
        diff1 = mldivide(U1,isect)
        diff2 = mldivide(U2,isect)
        assert not is_fulldim(diff1)
        assert not is_fulldim(diff2)
    
    def test_wrongdim(self):
        b = False
        try:
            U = union(self.P1,self.R1,True)
        except:
            b = True
        assert b == True
        
    def test_3dunion(self):
        U = union(self.R1,self.R2,True)
        ver = Polytope(np.array([[0.,0.,-1.],[0.,-1.,0.],[-1.,0.,0.],[0.,0.,1.],[0.,1.,0.],[1.,0.,0.]]),np.array([0.,0.,0.,1.,1.,1.]))
        diff1 = mldivide(U,ver)
        diff2 = mldivide(ver,U)
        assert dimension(diff1) == 0
        assert dimension(diff2) == 0

class cheby_test:
    def setUp(self):
        A = np.array([[ 0. ,-1.],   
                       [-1.  ,0.],  
                       [ 0. , 1.], 
                       [ 1. , 0.]])
        
        b = np.array([0.,0.,1.,1.])
        self.P1 = Polytope(A,b)
        
        self.abs_tol = 1e-7
        
    def tearDown(self):
        self.P1 = None
        self.abs_tol = None
    
    def test_cheby(self):
        rc,xc = cheby_ball(self.P1)
        assert np.abs(rc-0.5) < self.abs_tol
        assert np.sum(np.abs(xc.flatten() - 0.5)) < self.abs_tol


def normalized_polytope_projection_test():
    """Normalization of fixed and random polytopes given to projection methods.
    """
    abs_tol = 1e-10
    A = np.array([[2, 0],
                  [-2, 0],
                  [0, 2],
                  [0, -2]], dtype=np.float64)
    b = np.array([6, 6, 4, 4], dtype=np.float64)
    P = Polytope(A=A, b=b, normalize=False)
    P_normalized = Polytope(A=A, b=b, normalize=True)
    ppoly = projection(P, [1, 2], solver="fm")
    ppoly_n = projection(P_normalized, [1, 2], solver="fm")
    cheby_ball(ppoly)
    cheby_ball(ppoly_n)
    assert abs(ppoly.chebR-ppoly_n.chebR) < abs_tol
    # also check if initial polytopes match
    cheby_ball(P)
    cheby_ball(P_normalized)
    assert abs(P.chebR-P_normalized.chebR) < abs_tol

    for max_dim in [5,]:# 8]:
        for i in range(3):
            orig_dim = np.random.randint(max_dim-1)+2
            red_dim = np.random.randint(orig_dim-1)+1
            V = np.random.random((max_dim*2, orig_dim))
            P_normalized = qhull(V)
            P = P_normalized.copy()
            scalar = 50*np.random.random()
            P.A *= scalar
            P.b *= scalar
            proj_dims = [k+1 for k in np.sort(np.random.permutation(orig_dim)[:red_dim])]
            ppoly = projection(P, proj_dims, solver="iterhull")
            ppoly_n = projection(P_normalized, proj_dims, solver="iterhull")
            cheby_ball(ppoly)
            cheby_ball(ppoly_n)
            assert abs(ppoly.chebR-ppoly_n.chebR) < abs_tol
            # also check if initial polytopes match
            cheby_ball(P)
            cheby_ball(P_normalized)
            assert abs(P.chebR-P_normalized.chebR) < abs_tol

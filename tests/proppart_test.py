#!/usr/bin/env python
"""
Test TuLiP code for working with (proposition preserving) partitions.

SCL; 23 July 2012.
(based on test code previously at the bottom of tulip/prop2part.py.)
"""

from tulip.prop2part import prop2part2
import tulip.polytope as pc
import numpy as np


def prop2part2_test():
    domain_poly_A = np.array(np.vstack([np.eye(2),-np.eye(2)]))
    domain_poly_b = np.array([2., 2, 0, 0]).T
    state_space = pc.Polytope(domain_poly_A, domain_poly_b)

    cont_props = []
    A = []
    b = []
    A.append(np.array([[1., 0.],
                       [-1., 0.],
                       [0., 1.],
                       [0., -1.]]))
    b.append(np.array([[1., 0., 1., 0.]]).T)
    cont_props.append(pc.Polytope(A[0], b[0]))
    
    A.append(np.array([[1., 0.],
                       [-1., 0.],
                       [0., 1.],
                       [0., -1.]]))
    b.append(np.array([[1., 0., 2., -1.]]).T)
    cont_props.append(pc.Polytope(A[1], b[1]))
    
    A.append(np.array([[1., 0.],
                       [-1., 0.],
                       [0., 1.],
                       [0., -1.]]))
    b.append(np.array([[2., -1., 1., 0.]]).T)
    cont_props.append(pc.Polytope(A[2], b[2]))

    A.append(np.array([[1., 0.],
                       [-1., 0.],
                       [0., 1.],
                       [0., -1.]]))
    b.append(np.array([[2., -1., 2., -1.]]).T)
    cont_props.append(pc.Polytope(A[3], b[3]))
    
    cont_props_dict = dict([("C"+str(i), pc.Polytope(A[i], b[i])) for i in range(4)])
    
    mypartition = prop2part2(state_space, cont_props_dict)
    
    # A4 = np.array([[1., 0.],
    #                [-1., 0.],
    #                [0., 1.],
    #                [0., -1.]])
    # b4 = np.array([[0.5, 0., 0.5, 0.]]).T
    # poly1 = pc.Polytope(A4,b4)
    # r1 = pc.mldivide(mypartition.list_region[3],poly1)
    
    ref_adjacency = np.array([[1,1,1,0],[1,1,0,1],[1,0,1,1],[0,1,1,1]])
    assert np.all(mypartition.adj == ref_adjacency)

    assert len(mypartition.list_region) == 4
    for reg in mypartition.list_region:
        assert len(reg.list_prop) == 4
        assert len(reg.list_poly) == 1
        i = [i for i in range(len(reg.list_prop)) if reg.list_prop[i] == 1]
        assert len(i) == 1
        i = i[0]
        assert cont_props_dict.has_key(mypartition.list_prop_symbol[i])
        ref_V = pc.extreme(cont_props_dict[mypartition.list_prop_symbol[i]])
        ref_V = set([(v[0],v[1]) for v in ref_V.tolist()])
        actual_V = pc.extreme(reg.list_poly[0])
        actual_V = set([(v[0],v[1]) for v in actual_V.tolist()])
        assert ref_V == actual_V

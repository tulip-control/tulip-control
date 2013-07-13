#!/usr/bin/env python

from tulip.abstract import prop2part
import tulip.polytope as pc
import numpy as np

def prop2part_test():
    state_space = pc.Polytope.from_box(np.array([[0., 2.],[0., 2.]]))
    
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
    
    
    mypartition = prop2part.prop2part(state_space, cont_props_dict)
    ref_adjacency = np.array([[1,1,1,0],[1,1,0,1],[1,0,1,1],[0,1,1,1]])
    assert np.all(mypartition.adj.todense() == ref_adjacency)

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


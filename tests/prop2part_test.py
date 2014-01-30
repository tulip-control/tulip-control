#!/usr/bin/env python
"""
Tests for abstract.prop2partition
"""

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
    b.append(np.array([[.5, 0., .5, 0.]]).T)
    cont_props.append(pc.Polytope(A[0], b[0]))
    
    A.append(np.array([[1., 0.],
                       [-1., 0.],
                       [0., 1.],
                       [0., -1.]]))
    b.append(np.array([[2., -1.5, 2., -1.5]]).T)
    cont_props.append(pc.Polytope(A[1], b[1]))
    
    cont_props_dict = dict([("C"+str(i), pc.Polytope(A[i], b[i])) for i in range(2)])
    
    
    mypartition = prop2part(state_space, cont_props_dict)
    ref_adjacency = np.array([[1,0,1],[0,1,1],[1,1,1]])
    assert np.all(mypartition.adj.todense() == ref_adjacency)

    assert len(mypartition.regions) == 3
    
    for reg in mypartition.regions[0:2]:
        assert len(reg.props) == 2
        assert len(reg.list_poly) == 1
        i = [i for i in range(len(reg.props)) if reg.props[i] == 1]
        assert len(i) == 1
        i = i[0]
        assert cont_props_dict.has_key(mypartition.prop_symbols[i])
        ref_V = pc.extreme(cont_props_dict[mypartition.prop_symbols[i]])
        ref_V = set([(v[0],v[1]) for v in ref_V.tolist()])
        actual_V = pc.extreme(reg.list_poly[0])
        actual_V = set([(v[0],v[1]) for v in actual_V.tolist()])
        assert ref_V == actual_V
        
    assert len(mypartition.regions[2].props) == 2
    assert sum(mypartition.regions[2].props) == 0
    assert len(mypartition.regions[2].list_poly) == 3
    dum = state_space.copy()
    for reg in mypartition.regions[0:2]:
        dum = dum.diff(reg)
    assert pc.is_empty(dum.diff(mypartition.regions[2]) )
    assert pc.is_empty(mypartition.regions[2].diff(dum) )


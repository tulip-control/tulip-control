#!/usr/bin/env python

"""
Unit Tests for transys module.
"""
from collections import Iterable

from nose.tools import assert_raises

from tulip.transys.mathset import MathSet, SubSet, PowerSet
from tulip.transys.mathset import compare_lists
from tulip import transys as trs

def mathset_test():
    s = MathSet([1,2,[1,2] ] )
    q = MathSet()
    
    q.add(1)
    q |= [1,2]
    
    q |= ['a', 3, {3}]
    
    c = s | s
    
    d = s | q
    
    assert(c is not s)
    
    assert(q._set == {'a',1,2,3} )
    assert(q._list == [{3} ] )
    
    assert(c._set == {1,2} )
    assert(c._list == [[1,2] ] )
    
    assert(d._set == {1,2,3,'a'} )
    assert(d._list == [[1,2], {3} ] )
    
    assert(isinstance(s, Iterable) )
    assert(1 in s)
    assert(2 in s)
    assert([1, 2] in s)
    assert(5 not in q)
    
    assert(len(s) == 3)
    
    s.remove([1,2] )
    assert([1,2] not in s)
    assert(s._set == {1,2} )
    assert(s._list == [])
    assert(len(s) == 2)
    
    for item in s:
        print(item)
    
    """Mutable"""
    a = MathSet()
    b = MathSet([{'a':1} ] )
    
    a |= b
    
    assert(a._set == set() )
    assert(a._list == [{'a':1} ] )

def subset_test():
    a = SubSet([1,2,3,4, {1:2} ] )
    print(a)
    
    a.add(1)
    a.add_from([1,2] )
    a |= [3,4]
    
    assert(a._set == {1,2,3,4} )
    
    a |= [{1:2} ]
    assert(a._list == [{1:2} ] )
    
    a.superset = [1,2,3,4, {1:2}, '6']
    
    b = SubSet([1,'2'] )
    b.add('2')
    assert(b._set == {'2'} )
    assert(not bool(b._list) )
    assert(b._superset == [1,'2'] )
    
    superset = [1, 2]
    s = SubSet(superset)
    s |= [1, 2]
    print(s)
    assert(s._set == {1,2} )
    assert(not bool(s._list) )
    
    #s.add(3)
    
    return a

def powerset_test():
    s = [[1, 2], '3', {'a':1}, 1]
    
    p = PowerSet(s)
    
    s2 = ['3', {'a':1}, [1,2], 1]
    q = PowerSet()
    q.math_set = MathSet(s2)
    
    assert(p.math_set == MathSet(s) )
    assert(q.math_set == MathSet(s) )
    assert(isinstance(q.math_set, MathSet) )
    assert(p == q)
    
    q.math_set.add(6)
    
    # CAUTION: comparing p() and q() might yield False, due to ordering
    f = p +q
    assert(isinstance(f, PowerSet) )
    assert(f.math_set._set == {1, '3', 6} )
    assert(compare_lists(f.math_set._list, [[1, 2], {'a':1} ] ) )
    
    return s

def labeled_digraph_test():
    p = PowerSet({1, 2})
    node_labeling = [('month', ['Jan', 'Feb']),
                     ('day', ['Mon', 'Tue']),
                     ('comb', p, p.math_set)]
    edge_labeling = node_labeling
    g = trs.labeled_graphs.LabeledDiGraph(node_labeling, edge_labeling)
    
    g.states.add_from({1, 2})
    g.transitions.add_labeled(1, 2, {'month':'Jan', 'day':'Mon'})
    
    assert_raises(Exception, g.transitions.add_labeled,
                  1, 2, {'month':'Jan', 'day':'abc'})

def rabin_test():
    dra = trs.DRA()
    print(dra)
    
    dra.states.add_from(range(10) )
    
    dra.states.accepting.add([1,2], [3,4] )
    assert(isinstance(dra.states.accepting._pairs, list) )
    assert(dra.states.accepting._pairs[0][0]._set == {1,2} )
    assert(dra.states.accepting._pairs[0][0]._list == [] )
    assert(dra.states.accepting._pairs[0][1]._set == {3,4} )
    assert(dra.states.accepting._pairs[0][1]._list == [] )
    print(dra.states.accepting)
    
    dra.states.accepting.remove([1,2], [3,4] )
    
    assert(isinstance(dra.states.accepting._pairs, list) )
    assert(not dra.states.accepting._pairs)
    
    dra.states.accepting.add([1], [2, 4] )
    dra.states.accepting.add_states(0, [2], [] )
    
    dra.states.accepting.add([2, 1], [] )
    
    assert(isinstance(dra.states.accepting._pairs, list) )
    assert(dra.states.accepting._pairs[0][0]._set == {1,2} )
    assert(dra.states.accepting._pairs[0][0]._list == [] )
    assert(dra.states.accepting._pairs[0][1]._set == {2,4} )
    assert(dra.states.accepting._pairs[0][1]._list == [] )
    
    assert(dra.states.accepting._pairs[1][0]._set == {2,1} )
    assert(dra.states.accepting._pairs[1][0]._list == [] )
    assert(dra.states.accepting._pairs[1][1]._set == set() )
    assert(dra.states.accepting._pairs[1][1]._list == [] )

if __name__ == '__main__':
    mathset_test()
    subset_test()
    powerset_test()
    rabin_test()

#!/usr/bin/env python

"""
Unit Tests for transys module.
"""

from collections import Iterable

from tulip.transys.mathset import MathSet, SubSet, PowerSet, compare_lists

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
    """MathSet"""
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
    
    """PowerSet"""
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

if __name__ == '__main__':
    subset_test()
    powerset_test()

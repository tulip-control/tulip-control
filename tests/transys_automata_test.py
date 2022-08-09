#!/usr/bin/env python
"""
Tests for transys.automata (part of transys subpackage)
"""
from tulip import transys as trs


def rabin_test():
    dra = trs.DRA()
    print(dra)
    dra.states.add_from(range(10) )
    dra.accepting.add([1,2], [3,4] )
    assert(isinstance(dra.accepting._pairs, list) )
    assert(dra.accepting._pairs[0][0]._set == {1,2} )
    assert(dra.accepting._pairs[0][0]._list == list())
    assert(dra.accepting._pairs[0][1]._set == {3,4} )
    assert(dra.accepting._pairs[0][1]._list == list())
    print(dra.accepting)
    dra.accepting.remove([1,2], [3,4] )
    assert(isinstance(dra.accepting._pairs, list) )
    assert(not dra.accepting._pairs)
    dra.accepting.add([1], [2, 4] )
    dra.accepting.add_states(0, [2], list())
    dra.accepting.add([2, 1], list())
    assert(isinstance(dra.accepting._pairs, list) )
    assert(dra.accepting._pairs[0][0]._set == {1,2} )
    assert(dra.accepting._pairs[0][0]._list == list())
    assert(dra.accepting._pairs[0][1]._set == {2,4} )
    assert(dra.accepting._pairs[0][1]._list == list())
    assert(dra.accepting._pairs[1][0]._set == {2,1} )
    assert(dra.accepting._pairs[1][0]._list == list())
    assert(dra.accepting._pairs[1][1]._set == set() )
    assert(dra.accepting._pairs[1][1]._list == list())

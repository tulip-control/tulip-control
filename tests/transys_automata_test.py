#!/usr/bin/env python
"""
Tests for transys.automata (part of transys subpackage)
"""

from tulip import transys as trs


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

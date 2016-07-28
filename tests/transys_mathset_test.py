#!/usr/bin/env python
"""
Tests for transys.mathset (part of transys subpackage)
"""
from nose.tools import raises
from collections import Iterable

from tulip.transys.mathset import MathSet, SubSet, PowerSet, TypedDict
from tulip.transys.mathset import compare_lists, unique, contains_multiple
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

class MathSet_operations_test:
    def setUp(self):
        self.x = MathSet(['a', 1, [1, 2], {'a', 'b', '8'} ] )
        self.y = MathSet(['b', -2, [3.5, 2.25], {'/', '_'} ] )
        self.small2_listnum = MathSet([[1,2], 0.2])
        self.small1_set = MathSet([{-1, 1}])

    def tearDown(self):
        self.x = None
        self.y = None

    def test_mul(self):
        assert self.small2_listnum * self.small1_set == \
            MathSet([([1,2], {-1, 1}), (0.2, {-1, 1})])
        assert self.small2_listnum * self.small2_listnum == \
            MathSet([([1,2], [1,2]), ([1,2], 0.2), (0.2, [1,2]), (0.2, 0.2)])

    def test_sub(self):
        assert self.small2_listnum - self.small2_listnum== MathSet()
        assert self.small1_set - self.small2_listnum == self.small1_set
        assert self.x - self.small2_listnum == \
            MathSet(['a', 1, {'a', 'b', '8'} ] )

    def test_isub(self):
        q = MathSet(self.small2_listnum)
        q -= self.small2_listnum
        assert q == MathSet()

        q = MathSet(self.small2_listnum)
        q -= MathSet()
        assert q == q

    def test_add_unhashable(self):
        self.small1_set.add([1,2])
        assert self.small1_set == MathSet([{-1, 1}, [1, 2]])

    def test_pop(self):
        # These tests could be improved by seeding random number
        # generation to ensure determinism.
        assert self.small1_set.pop() == {-1, 1}
        assert self.small1_set == MathSet()

        original = MathSet(self.small2_listnum)
        q = self.small2_listnum.pop()
        assert len(self.small2_listnum) + 1 == len(original)
        self.small2_listnum.add(q)
        assert self.small2_listnum == original

    def test_intersection(self):
        assert self.x.intersection(self.small2_listnum) == MathSet([[1, 2]])
        assert self.x.intersection(MathSet()) == MathSet()

    def test_intersects(self):
        assert self.x.intersects(self.small2_listnum)
        assert not self.small2_listnum.intersects(self.small1_set)


def unique_check(iterable, expected):
    print unique(iterable)
    assert unique(iterable) == expected

def unique_test():
    for (iterable, expected) in [(range(3), set([0, 1, 2])),
                                 ([], set([])),
                                 ([1, 1, -1], set([1, -1])),
                                 ([[1, 2], 3, 3], [[1, 2], 3]),
                                 ("Dessert!!", set("Desert!"))]:
        yield unique_check, iterable, expected

def contains_multiple_test():
    assert contains_multiple([1, 1])
    assert not contains_multiple(("cc",))
    assert contains_multiple("cc")


def test_tuple():
    s = MathSet((1,2))
    assert(s._set == {1,2})
    assert(s._list == [])

def subset_test():
    a = SubSet([1,2,3,4, {1:2} ] )
    print(a)

    a.add(1)
    a.add_from([1,2] )
    a |= [3,4]

    assert(a._set == {1,2,3,4} )

    a |= [{1:2} ]
    assert(a._list == [{1:2} ] )

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

class PowerSet_operations_test:
    def setUp(self):
        self.p = PowerSet({1, 2, 3})
        self.q_unhashable = PowerSet(MathSet([[1,2], ["a", "b"]]))
        self.singleton = PowerSet(set([1]))
        self.empty = PowerSet()

    def tearDown(self):
        self.p = None
        self.q_unhashable = None
        self.singleton = None

    def test_len(self):
        assert len(self.p) == 2**3
        assert len(self.q_unhashable) == 2**2
        assert len(self.singleton) == 2
        assert len(self.empty) == 1

    def test_call(self):
        p = [set(x) for x in self.p]
        assert len(p) == 2**3
        assert (set() in p)
        assert (set([1]) in p) and (set([2]) in p) and (set([3]) in p)
        assert (set([1,2]) in p) and (set([2,3]) in p) and (set([1,3]) in p)
        assert set([1,2,3]) in p
        assert set(self.singleton) == set([(), (1,)])
        assert set(self.empty) == set([()])

class TypedDict_test():
    def setUp(self):
        d = TypedDict()
        d.set_types({'animal':{'dog', 'cat'} })
        self.d = d

    def test_add_typed_key_value(self):
        d = self.d

        d['animal'] = 'dog'
        assert(d['animal'] == 'dog')

    @raises(ValueError)
    def test_add_typed_key_illegal_value(self):
        d = self.d

        d['animal'] = 'elephant'

    def test_add_untyped_key_value(self):
        d = self.d

        d['human'] = 'Bob'
        assert(d['human'] == 'Bob')

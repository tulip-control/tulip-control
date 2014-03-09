#!/usr/bin/env python
"""
Tests for transys.labeled_graphs (part of transys subpackage)
"""

from nose.tools import raises
from networkx import MultiDiGraph
from tulip.transys import labeled_graphs


def str2singleton_test():
    assert labeled_graphs.str2singleton("p") == {"p"}
    assert labeled_graphs.str2singleton({"Cal"}) == {"Cal"}

def prepend_with_check(states, prepend_str, expected):
    assert labeled_graphs.prepend_with(states, prepend_str) == expected

def prepend_with_test():
    for (states, prepend_str, expected) in [([0,1], "s", ['s0', 's1']),
                                            ([], "s", []),
                                            ([0], "Cal", ["Cal0"])]:
        yield prepend_with_check, states, prepend_str, expected


class States_test():
    def setUp(self):
        self.S_mutable = labeled_graphs.States(MultiDiGraph(), mutable=True)
        self.S_immutable = labeled_graphs.States(MultiDiGraph(), mutable=False)

    def tearDown(self):
        self.S_mutable = None
        self.S_immutable = None

    def test_contains(self):
        # This also serves as a test for len
        self.S_immutable.add(1)
        self.S_immutable.add(2)
        assert len(self.S_immutable) == 2
        assert (2 in self.S_immutable) and (1 in self.S_immutable)
        assert 3 not in self.S_immutable

        self.S_mutable.add([-1,0,1])
        assert len(self.S_mutable) == 1
        assert [-1,0,1] in self.S_mutable
        assert None not in self.S_mutable

    def test_add(self):
        self.S_immutable.add(1)
        assert set([s for s in self.S_immutable]) == set([1])
        self.S_immutable.add(2)
        assert set([s for s in self.S_immutable]) == set([1, 2])

        self.S_mutable.add([1, 2])
        self.S_mutable.add("Cal")
        S_mutable_list = [s for s in self.S_mutable]
        assert (len(S_mutable_list) == 2) and ("Cal" in S_mutable_list) and \
            ([1,2] in S_mutable_list)

    def test_add_from(self):
        self.S_immutable.add_from(range(3))
        assert len(self.S_immutable) == 3
        assert set([s for s in self.S_immutable]) == set(range(3))
        self.S_mutable.add_from([[1.0, "Cal"], -1])
        assert (len(self.S_mutable) == 2) and \
            ([1.0, "Cal"] in self.S_mutable) and (-1 in self.S_mutable)

    def test_remove(self):
        # This also tests remove_from
        self.S_immutable.add_from(range(3))
        self.S_immutable.remove(1)
        assert set([s for s in self.S_immutable]) == set([0, 2])
        self.S_immutable.remove_from([0, 1])
        assert len(self.S_immutable) == 1 and 2 in self.S_immutable

        self.S_mutable.add_from([[1, 2], -1])
        self.S_mutable.remove([1, 2])
        assert set([s for s in self.S_mutable]) == set([-1])


class Transitions_test:
    def setUp(self):
        G = labeled_graphs.LabeledStateDiGraph()
        self.T = labeled_graphs.Transitions(G)
        G.transitions = self.T
        self.T.graph.states.add_from([1, 2, 3])

    def tearDown(self):
        self.T = None

    def test_len(self):
        assert len(self.T) == 0
        self.T.add(1, 2)
        assert len(self.T) == 1
        self.T.add(2, 3)
        assert len(self.T) == 2
        self.T.graph.states.add_from([10])
        assert len(self.T) == 2

    @raises(Exception)
    def test_missing_states(self):
        self.T.add(10, 11, check_states=True)

    def test_add_from(self):
        # This also tests Transitions.add(..., check_states=False)
        self.T.add(1, 4, check_states=False)
        assert len(self.T) == 1 and set([t for t in self.T()]) == set([(1, 4)])
        self.T.add(1, 4, check_states=True)
        assert len(self.T) == 1  # Edge already exists, so not added

        self.T.add_from([5, 2], [4, 3], check_states=False)
        assert len(self.T) == 5
        assert set([t for t in self.T()]) == set([(5, 4), (5, 3), (1, 4),
                                                  (2, 4), (2, 3)])

    def test_remove(self):
        # This also tests remove_from
        self.T.add_from([5, 2], [4, 3], check_states=False)
        assert len(self.T) == 4
        self.T.remove(5, 3)
        assert len(self.T) == 3
        assert set([t for t in self.T()]) == set([(5, 4), (2, 4), (2, 3)])
        self.T.remove_from((2,), (4, 3))
        assert set([t for t in self.T()]) == set([(5, 4)])

    def test_between(self):
        self.T.add_from([5, 2], [4, 3], check_states=False)
        assert set(self.T.between([5, 2], [3,])) == set([(5, 3), (2, 3)])

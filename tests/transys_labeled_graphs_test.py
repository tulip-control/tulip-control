#!/usr/bin/env python
"""Tests for transys.labeled_graphs (part of transys subpackage)"""
from nose.tools import raises, assert_raises
from tulip.transys import labeled_graphs
from tulip.transys.mathset import PowerSet, MathSet
from tulip.transys.transys import FTS


def str2singleton_test():
    assert labeled_graphs.str2singleton("p") == {"p"}
    assert labeled_graphs.str2singleton({"Cal"}) == {"Cal"}


def prepend_with_check(states, prepend_str, expected):
    assert labeled_graphs.prepend_with(states, prepend_str) == expected


def prepend_with_test():
    for (states, prepend_str, expected) in [([0,1], "s", ['s0', 's1']),
                                            ([], "s", []),
                                            ([0], "Cal", ["Cal0"]),
                                            ([0, 1], None, [0, 1])]:
        yield prepend_with_check, states, prepend_str, expected


class States_test():
    def setUp(self):
        self.S = labeled_graphs.States(labeled_graphs.LabeledDiGraph())

    def tearDown(self):
        self.S = None

    def test_contains(self):
        # This also serves as a test for len
        self.S.add(1)
        self.S.add(2)
        assert len(self.S) == 2
        assert (2 in self.S) and (1 in self.S)
        assert 3 not in self.S

    def test_ior(self):
        self.S.add(1)
        other_S = labeled_graphs.States(labeled_graphs.LabeledDiGraph())
        other_S.add(0)
        assert len(self.S) == 1
        assert set([s for s in self.S]) == {1}
        self.S |= other_S
        assert len(self.S) == 2
        assert set([s for s in self.S]) == {1, 0}

    def test_add(self):
        self.S.add(1)
        assert set([s for s in self.S]) == set([1])
        self.S.add(2)
        assert set([s for s in self.S]) == set([1, 2])
        self.S.add("Cal")
        assert set([s for s in self.S]) == set([1, 2, "Cal"])

    def test_add_from(self):
        self.S.add_from(range(3))
        assert len(self.S) == 3
        assert set([s for s in self.S]) == set(range(3))
        self.S.add_from(["Cal", "tech"])
        assert len(self.S) == 5
        assert set([s for s in self.S]) == set(range(3)+["Cal", "tech"])

    def test_remove(self):
        # This also tests remove_from
        self.S.add_from(range(4))
        self.S.remove(1)
        assert set([s for s in self.S]) == set([0, 2, 3])
        self.S.remove_from([0, 3])
        assert len(self.S) == 1 and 2 in self.S

    def test_call(self):
        self.S.add_from([-1, "Cal"])
        S_imm_dat = self.S(data=True)
        assert (len(S_imm_dat) == 2) and ((-1, dict()) in S_imm_dat) and \
            (("Cal", dict()) in S_imm_dat)

    def test_postpre(self):
        self.S.add_from(range(5))
        self.S.graph.add_edges_from([(0, 1), (0, 2), (1, 3), (3, 4)])
        assert self.S.post(0) == {1, 2}
        assert self.S.post([0, 1]) == {1, 2, 3}
        assert self.S.pre(4) == {3}
        assert self.S.pre([1, 2, 4]) == {0, 3}

    def test_is_terminal(self):
        self.S.add_from([0, 1])
        self.S.graph.add_edge(0, 1)
        assert not self.S.is_terminal(0)
        assert self.S.is_terminal(1)


class Transitions_test:
    def setUp(self):
        G = labeled_graphs.LabeledDiGraph()
        self.T = labeled_graphs.Transitions(G)
        G.transitions = self.T
        self.T.graph.states.add_from([1, 2, 3, 4, 5])

    def tearDown(self):
        self.T = None

    def test_len(self):
        assert len(self.T) == 0
        self.T.add(1, 2)
        assert len(self.T) == 1
        self.T.add(2, 3)
        assert len(self.T) == 2

        # Transitions should be unaffected by new states
        self.T.graph.states.add_from([10])
        assert len(self.T) == 2

    @raises(Exception)
    def test_missing_states(self):
        self.T.add(10, 11, check=True)

    def test_add_from(self):
        self.T.add(1, 4)
        assert len(self.T) == 1 and set([t for t in self.T()]) == set([(1, 4)])
        assert_raises(Exception, self.T.add, 1, 4)
        assert len(self.T) == 1  # Edge already exists, so not added

        self.T.add_from([(5, 2), (4, 3)])
        assert len(self.T) == 3
        assert set([t for t in self.T()]) == set([(1, 4), (5, 2), (4, 3)])

    def test_add_comb(self):
        self.T.add_comb([1, 2], [3, 4])
        assert len(self.T) == 4 and set([t for t in self.T()]) == set([(1, 3),
                                                                       (2, 3),
                                                                       (1, 4),
                                                                       (2, 4)])

    def test_remove(self):
        # This also tests remove_from
        self.T.add_from([(1, 2), (1, 3), (4, 3), (3, 2)], check=False)
        assert len(self.T) == 4
        self.T.remove(1, 2)
        assert len(self.T) == 3
        assert set([t for t in self.T()]) == set([(1, 3), (4, 3), (3, 2)])
        self.T.remove_from([(1, 2), (4, 3), (3, 2)])
        assert set([t for t in self.T()]) == set([(1, 3)])


class States_labeling_test:
    def setUp(self):
        node_label_def = [{
            'name': 'ap',
            'values': PowerSet({'p', 'q', 'r', 'x', 'a', 'b'})}]
        G = labeled_graphs.LabeledDiGraph(node_label_def)
        self.S_ap = labeled_graphs.States(G)
        G.states = self.S_ap

    def tearDown(self):
        self.S_ap = None

    @raises(Exception)
    def test_add_untyped_keys(self):
        self.S_ap.add(1, foo=MathSet(['p']), check=True)

    def test_add(self):
        self.S_ap.add(1, ap={'p'} )
        assert len(self.S_ap) == 1
        self.S_ap.add(2, ap={'p'} )
        assert len(self.S_ap) == 2
        self.S_ap.add(3, ap={'q', 'r'} )
        assert len(self.S_ap) == 3

        assert self.S_ap[1] == {'ap': {'p'} }
        assert self.S_ap[3] == {'ap': {'q', 'r'} }

        nodes = {u for u,l in self.S_ap.find(
            with_attr_dict={'ap': {'p'} })}
        assert nodes == set([1, 2])

    def test_add_from(self):
        self.S_ap.add_from([(0, {'ap':{'p'}}), (1, {'ap':{'q'} })])
        assert len(self.S_ap) == 2
        current_labels = [l["ap"] for (s,l) in self.S_ap(data=True)]

        assert len(current_labels) == 2 and \
            MathSet(current_labels) == MathSet([{'p'}, {'q'}])

        self.S_ap.add_from([(10, {'ap':{'r'} }),
                                      (11, {'ap':{'x'} })],
                                      check=False)
        assert len(self.S_ap) == 4
        self.S_ap.add_from([(10, {'ap':{'a'} }),
                                      (11, {'ap':{'b'} })])
        assert len(self.S_ap) == 4
        current_labels = [l["ap"] for (s,l) in self.S_ap(data=True)]
        assert len(current_labels) == 4 and \
            MathSet(current_labels)== MathSet([{'p'}, {'q'}, {'a'}, {'b'}])
        assert MathSet([l["ap"] for (s,l) in self.S_ap(data=True)]) == MathSet(current_labels)

    def test_find(self):
        state_list = ["state"+str(i) for i in range(4)]
        state_list = zip(state_list,
                         [{"ap": L} for L in [{'p'}, {'q'}, {'p'}, {'q'}]])
        self.S_ap.add_from(state_list, check=False)
        result = self.S_ap.find("state1")
        assert len(result) == 1 and result[0] == ("state1", {"ap": set(['q'])})

        result = self.S_ap.find(["state1", "state0"])
        assert len(result) == 2 and \
           ("state1", {"ap": set(['q'])}) in result and \
           ("state0", {"ap": set(['p'])}) in result

        result = self.S_ap.find(with_attr_dict={"ap": {'p'}})
        print result
        assert len(result) == 2 and \
           set([s for (s, l) in result]) == set(["state0", "state2"])

        same_result = self.S_ap.find(ap={'p'})
        assert(same_result == result)


class LabeledDiGraph_test():
    def setUp(self):
        p = PowerSet({1, 2})
        node_labeling = [
            {
                'name': 'month',
                'values': ['Jan', 'Feb']
            },
            {
                'name': 'day',
                'values': ['Mon', 'Tue']
            },
            {
                'name': 'comb',
                'values': p,
                'setter': p.math_set
            }
        ]
        edge_labeling = node_labeling
        G = labeled_graphs.LabeledDiGraph(node_labeling, edge_labeling)

        G.states.add_from({1, 2})
        G.transitions.add(1, 2, month='Jan', day='Mon')

        assert_raises(Exception, G.transitions.add,
                      1, 2, {'month': 'Jan', 'day': 'abc'})

        # note how untyped keys can be set directly via assignment,
        # whereas check=False is needed for G.add_node
        G.node[1]['mont'] = 'Feb'
        assert(G.node[1] == {'mont':'Feb'})

        G[1][2][0]['day'] = 'Tue'
        assert(G[1][2][0] == {'month':'Jan', 'day':'Tue'})

        self.G = G

    @raises(AttributeError)
    def test_add_edge_only_typed(self):
        """check that untyped attribute keys are caught"""
        self.G.add_edge(1, 2, mo='Jan')

    def test_add_edge_untyped(self):
        """the untyped attribute key 'mo' should be allowed,
        because check=False
        """
        self.G.add_edge(1, 2, mo='Jan', check=False)
        assert(self.G[1][2][1] == {'mo':'Jan'})

    @raises(ValueError)
    def test_add_edge_illegal_value(self):
        self.G.add_edge(1, 2, month='haha')

    @raises(ValueError)
    def test_node_subscript_assign_illegal_value(self):
        self.G.node[1]['month'] = 'abc'

    @raises(ValueError)
    def test_edge_subscript_assign_illegal_value(self):
        self.G[1][2][0]['day'] = 'abc'


def open_fts_multiple_env_actions_test():
    env_modes = MathSet({'up', 'down'})
    env_choice = MathSet({'left', 'right'})

    env_actions = [
        {
            'name': 'env_modes',
            'values': env_modes,
            'setter': True},
        {
            'name': 'env_choices',
            'values': env_choice}]
    ts = FTS(env_actions)
    assert(ts.env_modes is env_modes)
    assert(not hasattr(ts, 'env_choices') )
    assert(ts.sys_actions == MathSet() )



def test_remove_deadends():
    g = labeled_graphs.LabeledDiGraph()

    # cycle
    n = 5
    g.add_nodes_from(range(n))
    for i in range(n):
        j = (i + 1) % n
        g.add_edge(i, j)

    g.remove_deadends()
    assert(len(g) == n)

    # line + cycle
    g.add_nodes_from(range(n, 2*n))
    for i in xrange(n, 2*n-1):
        g.add_edge(i, i+1)
    assert(len(g) == 2*n)

    g.remove_deadends()
    assert(len(g) == n)

    # line + self-loop
    g.remove_edge(4, 0)
    g.add_edge(0, 0)

    g.remove_deadends()
    assert(len(g) == 1)

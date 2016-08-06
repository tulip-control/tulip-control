#!/usr/bin/env python
"""
Tests for the interface with JTLV.
"""

import networkx as nx
import nose.tools as nt
from tulip.spec import GRSpec
import tulip.interfaces.jtlv as jtlv
import networkx as nx


class basic_test(object):
    def setUp(self):
        self.check_realizable = jtlv.check_realizable
        self.synthesize = jtlv.synthesize
        self.f_un = GRSpec(env_vars="x", sys_vars="y",
                           env_init="x", env_prog="x",
                           sys_init="y", sys_safety=["y -> X(!y)", "!y -> X(y)"],
                           sys_prog="y && x",
                           moore=False, plus_one=False, qinit='\A \E')
        self.f = GRSpec(env_vars="x", sys_vars="y",
                        env_init="x", env_prog="x",
                        sys_init="y",
                        sys_prog=["y & x", "!y"],
                        moore=False, plus_one=False, )
        self.dcounter = GRSpec(sys_vars={"y": (0,5)}, sys_init=["y=0"],
                               sys_prog=["y=0", "y=5"],
                               moore=False, plus_one=False, qinit='\A \E')

    def tearDown(self):
        self.f_un = None
        self.f = None
        self.dcounter = None

    def test_check_realizable(self):
        assert not self.check_realizable(self.f_un)
        self.f_un.sys_safety = []
        assert self.check_realizable(self.f_un)
        assert self.check_realizable(self.dcounter)

    def test_synthesize(self):
        g = self.synthesize(self.f_un)
        assert not isinstance(g, nx.DiGraph)

        g = self.synthesize(self.f)
        # There is more than one possible strategy realizing this
        # specification.  Checking only for one here makes this more like
        # a regression test (fragile).  However, it is more meaningful
        # than simply checking that synthesize() returns something
        # non-None (i.e., realizability, which is tested elsewhere).
        assert g is not None

        # assert len(g.env_vars) == 1 and g.env_vars.has_key('x')
        # assert len(g.sys_vars) == 1 and g.sys_vars.has_key('y')
        print(g.nodes())
        assert len(g) == 5
        assert set(g.edges()) == set([(0, 1), (0, 2), (1, 3), (1, 4),
                                              (2, 3), (2, 4), (3, 0), (3, 3),
                                              (4, 0), (4, 3)])
        label_reference = {0: (1,1), # value is bitvector of x,y
                           1: (0,0),
                           2: (1,0),
                           3: (0,0),
                           4: (1,0)}
        for u, d in g.nodes_iter(data=True):
            state = d['state']
            assert(len(state) == 2)
            assert(label_reference[u] == (state['x'], state['y']))


def hash_question_mark_test():
    specs = GRSpec(env_vars={'w': ['low', 'medium', 'high']},
                   sys_vars={'a': (0, 2)},
                   # env
                   env_init=['w="low"'],
                   env_safety=['(a=1) -> ((w="low") || (w="medium"))'],
                   env_prog=['(w="high")'],
                   sys_init=['a=2'],
                   sys_safety=['a=2'],
                   sys_prog=['a=2'],
                   moore=False, plus_one=False, qinit='\A \E')
    with nt.assert_raises(ValueError):
        jtlv.synthesize(specs)

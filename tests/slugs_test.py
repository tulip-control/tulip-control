#!/usr/bin/env python
"""Tests for the interface with slugs."""
from __future__ import print_function

import logging

import networkx as nx
from tulip.interfaces import slugs
from tulip.spec import GRSpec


logger = logging.getLogger(__name__)


def bitfields_to_ints_test():
    t = {'a': (0, 30)}

    # test int values
    bits = {'a@0.0.30': 0, 'a@1': 1, 'a@2': 1,
            'a@3': 0, 'a@4': 1, 'a@5': 0}
    n = slugs._bitfields_to_ints(bits, t)
    logger.debug(n)
    assert n == {'a': 22}

    # test str values
    bits = {'a@0.0.30': '0', 'a@1': '1', 'a@2': '1',
            'a@3': '0', 'a@4': '1', 'a@5': '0'}
    n = slugs._bitfields_to_ints(bits, t)
    logger.debug(n)
    assert n == {'a': 22}

    # range
    for n in range(30):
        bits = list(bin(n).lstrip('0b').zfill(6))
        bits.reverse()  # little-endian
        d = {'a@{i}'.format(i=i): v for i, v in enumerate(bits)}
        d['a@0.0.30'] = d.pop('a@0')
        t = {'a': (0, 30)}
        print(d)
        m = slugs._bitfields_to_ints(d, t)
        logger.debug((n, m))
        assert m == {'a': n}


class basic_test(object):
    def setUp(self):
        self.check_realizable = lambda x: slugs.synthesize(x) is not None
        self.synthesize = slugs.synthesize
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

        # assert len(g.env_vars) == 1 and 'x' in g.env_vars
        # assert len(g.sys_vars) == 1 and 'y' in g.sys_vars
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
        for u, d in g.nodes(data=True):
            state = d['state']
            assert(len(state) == 2)
            assert(label_reference[u] == (state['x'], state['y']))

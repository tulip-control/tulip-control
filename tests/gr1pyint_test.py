#!/usr/bin/env python
"""
Tests for the interface with gr1py.
"""
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('tulip.spec.lexyacc').setLevel(logging.WARNING)
from nose.tools import assert_raises
import os
from tulip.spec import GRSpec, translate
from tulip.interfaces import gr1py


class basic_test(object):
    def setUp(self):
        self.f_un = GRSpec(
            env_vars="x",
            sys_vars="y",
            env_init="x",
            env_prog="x",
            sys_init="y",
            sys_safety=["y -> X(!y)", "!y -> X(y)"],
            sys_prog="y && x",
            moore=False,
            plus_one=False,
            qinit='\A \E')
        self.dcounter = GRSpec(
            sys_vars={"y": (0, 5)},
            sys_init=["y=0"],
            sys_prog=["y=0", "y=5"],
            moore=False,
            plus_one=False,
            qinit='\A \E')

    def tearDown(self):
        self.f_un = None
        self.dcounter = None

    def test_check_realizable(self):
        # f_un
        assert not gr1py.check_realizable(self.f_un)
        self.f_un.sys_safety = []
        assert gr1py.check_realizable(self.f_un)
        self.f_un.qinit = '\A \A'
        self.f_un.env_init = ['x & y']
        self.f_un.sys_init = list()
        with assert_raises(AssertionError):
            assert gr1py.check_realizable(self.f_un)
        # counter
        assert gr1py.check_realizable(self.dcounter)
        self.dcounter.qinit = '\A \A'
        self.dcounter.sys_init = list()
        with assert_raises(AssertionError):
            assert gr1py.check_realizable(self.dcounter)

    def test_synthesize(self):
        self.f_un.sys_safety = []  # Make it realizable
        g = gr1py.synthesize(self.f_un)
        assert g is not None
        assert len(g.env_vars) == 1 and 'x' in g.env_vars
        assert len(g.sys_vars) == 1 and 'y' in g.sys_vars

        g = gr1py.synthesize(self.dcounter)
        assert g is not None
        assert len(g.env_vars) == 0
        assert len(g.sys_vars) == 1 and 'y' in g.sys_vars
        assert len(g) == 2, [g.nodes(data=True), g.edges(data=True)]

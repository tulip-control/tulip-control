#!/usr/bin/env python
"""
Tests for the interface with JTLV.
"""

from tulip.spec import GRSpec
from tulip.interfaces.jtlv import check_realizable, synthesize
from tulip.transys import MealyMachine


class basic_test:
    def setUp(self):
        self.f_un = GRSpec(env_vars="x", sys_vars="y",
                           env_init="x", env_prog="x",
                           sys_init="y", sys_safety=["y -> X(!y)", "!y -> X(y)"],
                           sys_prog="y && x")
        self.f = GRSpec(env_vars="x", sys_vars="y",
                        env_init="x", env_prog="x",
                        sys_init="y",
                        sys_prog=["y & x", "!y"])
        self.dcounter = GRSpec(sys_vars={"y": (0,5)}, sys_init=["y=0"],
                               sys_prog=["y=0", "y=5"])

    def tearDown(self):
        self.f_un = None
        self.f = None
        self.dcounter = None

    def test_check_realizable(self):
        assert not check_realizable(self.f_un)
        self.f_un.sys_safety = []
        assert check_realizable(self.f_un)
        assert check_realizable(self.dcounter)

    def test_synthesize(self):
        mach = synthesize(self.f_un)
        assert not isinstance(mach, MealyMachine)

        mach = synthesize(self.f)
        # There is more than one possible strategy realizing this
        # specification.  Checking only for one here makes this more like
        # a regression test (fragile).  However, it is more meaningful
        # than simply checking that synthesize() returns something
        # non-None (i.e., realizability, which is tested elsewhere).
        assert mach is not None
        assert len(mach.inputs) == 1 and mach.inputs.has_key("x")
        assert len(mach.outputs) == 1 and mach.outputs.has_key("y")
        assert len(mach.states()) == 6
        assert set(mach.transitions()) == set([(0, 1), (0, 2), (1, 3), (1, 4),
                                              (2, 3), (2, 4), (3, 0), (3, 3),
                                              (4, 0), (4, 3), ("Sinit", 0)])
        label_reference = {(0, 1) : (0,0),  # value is bitvector of x,y
                           (0, 2) : (1,0),
                           (1, 3) : (0,0),
                           (1, 4) : (1,0),
                           (2, 3) : (0,0),
                           (2, 4) : (1,0),
                           (3, 0) : (1,1),
                           (3, 3) : (0,0),
                           (4, 0) : (1,1),
                           (4, 3) : (0,0),
                           ("Sinit", 0) : (1,1)}
        for (from_state, to_state, slabel) in mach.transitions(data=True):
            assert label_reference[(from_state, to_state)] == (slabel["x"],
                                                               slabel["y"])

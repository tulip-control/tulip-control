#!/usr/bin/env python
"""
Test the interface with JTLV.

SCL; 5 Sep 2013.
Vasu Raman; 4 Sept 2013
"""

import os

from tulip.spec import GRSpec
from tulip.jtlvint import *


def test_dumpjtlv():
    spec = GRSpec(env_vars="x", sys_vars="y",
                  env_init="x", env_prog="x",
                  sys_init="y", sys_safety=["y -> !next(y)", "!y -> next(y)"],
                  sys_prog="y & x")
    specLTL = spec.to_jtlv()
    assumption = specLTL[0]
    guarantee = specLTL[1]
    assert check_gr1(assumption, guarantee, spec.env_vars.keys(), spec.sys_vars.keys())

def test_check_realizable():
    spec = GRSpec(env_vars="x", sys_vars="y",
                  env_init="x", env_prog="x",
                  sys_init="y", sys_safety=["y -> !next(y)", "!y -> next(y)"],
                  sys_prog="y & x")
    assert not check_realizable(spec)
    spec.sys_safety = []
    assert check_realizable(spec)

def test_synthesize():
    spec = GRSpec(env_vars="x", sys_vars="y",
                  env_init="x", env_prog="x",
                  sys_init="y",
                  sys_prog=["y & x", "!y"])
    aut = synthesize(spec)
    # There is more than one possible strategy realizing this
    # specification.  Checking only for one here makes this more like
    # a regression test (fragile).  However, it is more meaningful
    # than simply checking that synthesize() returns something
    # non-None (i.e., realizability, which is tested elsewhere).
    assert aut is not None
    assert len(aut.inputs) == 1 and aut.inputs.has_key("x")
    assert len(aut.outputs) == 1 and aut.outputs.has_key("y")
    assert set(aut.states()) == set(range(5))
    assert set(aut.transitions()) == set([(0, 1), (0, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 0), (3, 3), (4, 0), (4, 3)])
    label_reference = {(0, 1) : (0,0),  # value is bitvector of x,y
                       (0, 2) : (1,0),
                       (1, 3) : (0,0),
                       (1, 4) : (1,0),
                       (2, 3) : (0,0),
                       (2, 4) : (1,0),
                       (3, 0) : (1,1),
                       (3, 3) : (0,0),
                       (4, 0) : (1,1),
                       (4, 3) : (0,0)}
    for (from_state, to_state, slabel) in aut.transitions(labeled=True):
        assert label_reference[(from_state, to_state)] == (slabel["x"], slabel["y"])

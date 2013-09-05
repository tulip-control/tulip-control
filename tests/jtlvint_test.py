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
    assert aut is not None

#!/usr/bin/env python
"""
Based on test code moved here from bottom of tulip/jtlvint.py

SCL; 31 Dec 2011.
Vasu Raman; 4 Sept 2013
"""

import os

from tulip.spec import GRSpec
from tulip.jtlvint import *

class jtlvint_test:
    
    def test_dumpjtlv(self):
        spec = GRSpec(env_vars="x", sys_vars="y",
                      env_init="x", env_prog="x",
                      sys_init="y", sys_safety=["y -> !y'", "!y -> y'"],
                      sys_prog="y & x")
        specLTL = spec.to_jtlv()
        assumption = specLTL[0]
        guarantee = specLTL[1]
        assert check_gr1(assumption, guarantee, spec.env_vars.keys(), spec.sys_vars.keys())
        
    def test_check_realizable(self):
        spec = GRSpec(env_vars="x", sys_vars="y",
                      env_init="x", env_prog="x",
                      sys_init="y", sys_safety=["y -> !y'", "!y -> y'"],
                      sys_prog="y & x")
        assert not check_realizable(spec)
        spec.sys_safety = []
        assert check_realizable(spec)
        
    def test_synthesize(self):
        spec = GRSpec(env_vars="x", sys_vars="y",
                      env_init="x", env_prog="x",
                      sys_init="y",
                      sys_prog=["y & x", "!y"])
        aut = synthesize(spec)
        assert aut is not None
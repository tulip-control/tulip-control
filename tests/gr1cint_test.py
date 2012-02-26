#!/usr/bin/env python
"""
Test the interface with gr1c.

SCL; 25 Feb 2012.
"""

import os

from tulip.spec import GRSpec
from tulip.gr1cint import *


REFERENCE_SPECFILE = """
ENV: x;
SYS: y;

ENVINIT: x;
ENVTRANS:;# [](x -> !x') & [](!x -> x');
ENVGOAL: []<>x;

# Blank lines are optional and can placed between sections or parts of
# formulas.

SYSINIT: y;

SYSTRANS: # Notice the safety formula spans two lines.
[](y -> !y')
& [](!y -> y');

SYSGOAL: []<>y&x;
"""


class gr1cint_test:

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_check_syntax(self):
        assert check_syntax(REFERENCE_SPECFILE)
        assert not check_syntax("foo")

    def test_dumpgr1c(self):
        spec = GRSpec(env_vars="x", sys_vars="y",
                      env_init="x", env_prog="x",
                      sys_init="y", sys_safety=["y -> !y'", "!y -> y'"],
                      sys_prog="y & x")
        assert check_syntax(spec.dumpgr1c())

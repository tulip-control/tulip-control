#!/usr/bin/env python
"""
Tests for the export mechanisms of tulip.dumpsmach.
"""

import ast

from tulip import spec, synth, dumpsmach


class basic_test:
    def setUp(self):
        self.triv = spec.GRSpec(env_vars="x", sys_vars="y",
                                env_init="x", env_prog="x",
                                sys_init="y", sys_prog="y && x")
        self.triv_M = synth.synthesize("gr1c", self.triv)

        self.dcounter = spec.GRSpec(sys_vars={"y": (0,5)}, sys_init=["y=0"],
                                    sys_prog=["y=0", "y=5"])
        self.dcounter_M = synth.synthesize("gr1c", self.dcounter)

    def tearDown(self):
        self.dcounter = None
        self.dcounter_M = None

    def test_python_case(self):
        compile(dumpsmach.python_case(self.triv_M),
                filename="<string>", mode="exec")
        compile(dumpsmach.python_case(self.dcounter_M),
                filename="<string>", mode="exec")

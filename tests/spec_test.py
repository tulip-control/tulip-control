#!/usr/bin/env python
"""
Tests for the tulip.spec subpackage.
"""

import copy
import nose.tools as nt

from tulip.spec import LTL, GRSpec, mutex
from tulip.spec.parser import parse


def GR1specs_equal(s1, s2):
    """Return True if s1 and s2 are *roughly* syntactically equal.

    This function seems to be of little or no use outside this test
    module because of its fragility.
    """
    if s1 is None or s2 is None:
        raise TypeError
    for s in [s1, s2]:
        if hasattr(s.env_init, "sort"):
            s.env_init.sort()
        if hasattr(s.env_safety, "sort"):
            s.env_safety.sort()
        if hasattr(s.env_prog, "sort"):
            s.env_prog.sort()
        if hasattr(s.sys_init, "sort"):
            s.sys_init.sort()
        if hasattr(s.sys_safety, "sort"):
            s.sys_safety.sort()
        if hasattr(s.sys_prog, "sort"):
            s.sys_prog.sort()
    if s1.env_vars != s2.env_vars or s1.sys_vars != s2.sys_vars:
        return False
    if s1.env_init != s2.env_init or s1.env_safety != s2.env_safety or s1.env_prog != s2.env_prog:
        return False
    if s1.sys_init != s2.sys_init or s1.sys_safety != s2.sys_safety or s1.sys_prog != s2.sys_prog:
        return False
    return True


class LTL_test:
    def setUp(self):
        self.f = LTL("[](p -> <>q)", input_variables={"p":"boolean"},
                     output_variables={"q":"boolean"})

    def tearDown(self):
        self.f = None

    def test_loads_dumps_id(self):
        # Dump/load identity test: Dumping the result from loading a
        # dump should be identical to the original dump.
        assert self.f.dumps() == LTL.loads(self.f.dumps()).dumps()


class GRSpec_test:
    def setUp(self):
        self.f = GRSpec(env_vars={"x"}, sys_vars={"y"},
                        env_init=["x"], sys_safety=["y"],
                        env_prog=["!x", "x"], sys_prog=["y&&!x"])
        self.triv = GRSpec(env_vars=["x"], sys_vars=["y"],
                           env_init=["x && !x"])
        self.empty = GRSpec()

    def tearDown(self):
        self.f = None

    def test_sym_to_prop(self):
        original_env_vars = copy.copy(self.f.env_vars)
        original_sys_vars = copy.copy(self.f.sys_vars)
        self.f.sym_to_prop({"x":"bar", "y":"uber||cat"})
        assert self.f.env_vars == original_env_vars and self.f.sys_vars == original_sys_vars
        assert self.f.env_prog == ["!(bar)", "(bar)"] and self.f.sys_prog == ["(uber||cat)&&!(bar)"]

    def test_or(self):
        g = GRSpec(env_vars={"z"}, env_prog=["!z"])
        h = self.f | g
        assert len(h.env_vars) == 2 and h.env_vars.has_key("z")
        assert len(h.env_prog) == len(self.f.env_prog)+1 and "!z" in h.env_prog

        # Domain mismatch on system variable y
        g.sys_vars = {"y": (0,5)}
        nt.assert_raises(ValueError, self.f.__or__, g)

        # Domain mismatch on environment variable x
        g.sys_vars = dict()
        g.env_vars["x"] = (0,3)
        nt.assert_raises(ValueError, self.f.__or__, g)

    def test_to_canon(self):
        # Fragile!
        assert self.f.to_canon() == "((x) && []<>(!x) && []<>(x)) -> ([](y) && []<>(y&&!x))"
        # N.B., for self.triv, to_canon() returns a formula missing
        # the assumption part not because it detected that the
        # assumption is false, but rather the guarantee is empty (and
        # thus interpreted as being "True").
        assert self.triv.to_canon() == "True"
        assert self.empty.to_canon() == "True"

    def test_init(self):
        assert len(self.f.env_vars) == 1 and len(self.f.sys_vars) == 1
        assert self.f.env_vars["x"] == "boolean" and self.f.sys_vars["y"] == "boolean"


def parse_parse_check(formula, expected_length):
    # If expected_length is None, then the formula is malformed, and
    # thus we expect parsing to fail.
    if expected_length is not None:
        assert len(parse(formula)) == expected_length
    else:
        nt.assert_raises(Exception, parse, formula)

def parse_parse_test():
    for (formula, expected_len) in [("G p", 2),
                                    ("p G", None)]:
        yield parse_parse_check, formula, expected_len


def form_mutex_check(varnames, expected_formulae):
    # More like a regression test given fragility of formula strings.
    assert mutex(varnames) == expected_formulae

def form_mutex_test():
    for (varnames, expected_form) in [(["a", "b", "c"], {"a -> ! (b || c)",
                                                         "b -> ! (c)"}),
                                      (set(), set()),
                                      ({"cat"}, set())]:
        yield form_mutex_check, varnames, expected_form

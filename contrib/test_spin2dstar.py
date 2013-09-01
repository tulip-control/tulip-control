#!/usr/bin/env python
"""
Regression tests for spin2dstar.py

SCL; 1 Sep 2013
"""

import sys
import nose

from spin2dstar import spin_to_dstar


S2D_BATTERY = {
    "[]<> a": "G F a",
    "foo": "foo",
    "p0 && p1": "& p0 p1",
    "([] <> a) -> ([] <> b)": "i G F a G F b",
    "(p U q) || (r U z)": "| U p q U r z"
}


def check_spin_to_dstar(informula, expected_outformula):
    assert expected_outformula == spin_to_dstar(informula)

def test_spin_to_dstar():
    for (k,v) in S2D_BATTERY.items():
        yield check_spin_to_dstar, k, v


if __name__ == "__main__":
    sys.argv.append("--verbose")
    nose.run()

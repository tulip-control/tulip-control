#!/usr/bin/env python
"""
SCL; 28 June 2012.
"""

import numpy as np
import tulip.gridworld as gw
from tulip.gr1cint import check_realizable


REFERENCE_GWFILE = """
# A very small example, realizable by itself.
6 10

*  G*
  ***  ***
         *
I  *  *  *
  ****** *
*
"""

TRIVIAL_GWFILE = """
2 2
*
"""

class GridWorld_test:
    def setUp(self):
        self.prefix = "testworld"
        self.X = gw.GridWorld(REFERENCE_GWFILE, prefix=self.prefix)

    def tearDown(self):
        self.X = None
        
    def test_size(self):
        assert self.X.size() == (6, 10)

    def test_getitem(self):
        assert self.X[0,0] == self.prefix+"_"+str(0)+"_"+str(0)
        assert self.X[-1,0] == self.prefix+"_"+str(5)+"_"+str(0)
        assert self.X[-1,-2] == self.prefix+"_"+str(5)+"_"+str(8)

    def test_equality(self):
        assert self.X == gw.GridWorld(REFERENCE_GWFILE)
        Y = gw.GridWorld()
        assert self.X != Y
        Y = gw.GridWorld(TRIVIAL_GWFILE)
        assert self.X != Y

    def test_dumploadloop(self):
        assert self.X == gw.GridWorld(self.X.dumps())

    def test_spec_realizable(self):
        assert check_realizable(self.X.spec(), verbose=1)


def extract_coord_test():
    assert gw.extract_coord("test_3_0") == ("test", 3, 0)
    assert gw.extract_coord("obstacle_5_4_11") == ("obstacle_5", 4, 11)
    assert gw.extract_coord("test3_0") is None

def prefix_filt_test():
    assert gw.prefix_filt({"Y_0_0": 0, "Y_0_1": 1, "X_0_1_0": 1}, "Y") == {"Y_0_0": 0, "Y_0_1": 1}

"""
Tests for the tulip.gridworld
"""

import numpy as np
import tulip.gridworld as gw
from tulip.synth import is_realizable


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

    def test_state(self):
        assert self.X.state((2,3)) == {'testworld_3_9': 0, 'testworld_1_8': 0, 'testworld_1_9': 0, 'testworld_1_4': 0, 'testworld_1_5': 0, 'testworld_1_6': 0, 'testworld_1_7': 0, 'testworld_1_0': 0, 'testworld_1_1': 0, 'testworld_1_2': 0, 'testworld_1_3': 0, 'testworld_0_5': 0, 'testworld_0_4': 0, 'testworld_0_7': 0, 'testworld_0_6': 0, 'testworld_0_1': 0, 'testworld_0_0': 0, 'testworld_0_3': 0, 'testworld_0_2': 0, 'testworld_5_7': 0, 'testworld_0_9': 0, 'testworld_0_8': 0, 'testworld_3_2': 0, 'testworld_3_3': 0, 'testworld_2_9': 0, 'testworld_2_8': 0, 'testworld_3_6': 0, 'testworld_3_7': 0, 'testworld_3_4': 0, 'testworld_3_5': 0, 'testworld_2_3': 1, 'testworld_2_2': 0, 'testworld_2_1': 0, 'testworld_2_0': 0, 'testworld_2_7': 0, 'testworld_2_6': 0, 'testworld_2_5': 0, 'testworld_2_4': 0, 'testworld_4_1': 0, 'testworld_4_0': 0, 'testworld_4_3': 0, 'testworld_4_2': 0, 'testworld_4_5': 0, 'testworld_4_4': 0, 'testworld_4_7': 0, 'testworld_4_6': 0, 'testworld_4_9': 0, 'testworld_4_8': 0, 'testworld_5_8': 0, 'testworld_5_2': 0, 'testworld_5_9': 0, 'testworld_3_0': 0, 'testworld_3_1': 0, 'testworld_5_3': 0, 'testworld_5_5': 0, 'testworld_5_0': 0, 'testworld_5_4': 0, 'testworld_5_1': 0, 'testworld_5_6': 0, 'testworld_3_8': 0}
        assert self.X.state((-1,0)) == {'testworld_3_9': 0, 'testworld_1_8': 0, 'testworld_1_9': 0, 'testworld_1_4': 0, 'testworld_1_5': 0, 'testworld_1_6': 0, 'testworld_1_7': 0, 'testworld_1_0': 0, 'testworld_1_1': 0, 'testworld_1_2': 0, 'testworld_1_3': 0, 'testworld_0_5': 0, 'testworld_0_4': 0, 'testworld_0_7': 0, 'testworld_0_6': 0, 'testworld_0_1': 0, 'testworld_0_0': 0, 'testworld_0_3': 0, 'testworld_0_2': 0, 'testworld_5_7': 0, 'testworld_0_9': 0, 'testworld_0_8': 0, 'testworld_3_2': 0, 'testworld_3_3': 0, 'testworld_2_9': 0, 'testworld_2_8': 0, 'testworld_3_6': 0, 'testworld_3_7': 0, 'testworld_3_4': 0, 'testworld_3_5': 0, 'testworld_2_3': 0, 'testworld_2_2': 0, 'testworld_2_1': 0, 'testworld_2_0': 0, 'testworld_2_7': 0, 'testworld_2_6': 0, 'testworld_2_5': 0, 'testworld_2_4': 0, 'testworld_4_1': 0, 'testworld_4_0': 0, 'testworld_4_3': 0, 'testworld_4_2': 0, 'testworld_4_5': 0, 'testworld_4_4': 0, 'testworld_4_7': 0, 'testworld_4_6': 0, 'testworld_4_9': 0, 'testworld_4_8': 0, 'testworld_5_8': 0, 'testworld_5_2': 0, 'testworld_5_9': 0, 'testworld_3_0': 0, 'testworld_3_1': 0, 'testworld_5_3': 0, 'testworld_5_5': 0, 'testworld_5_0': 1, 'testworld_5_4': 0, 'testworld_5_1': 0, 'testworld_5_6': 0, 'testworld_3_8': 0}
        assert self.X.state((-1,-1)) == {'testworld_3_9': 0, 'testworld_1_8': 0, 'testworld_1_9': 0, 'testworld_1_4': 0, 'testworld_1_5': 0, 'testworld_1_6': 0, 'testworld_1_7': 0, 'testworld_1_0': 0, 'testworld_1_1': 0, 'testworld_1_2': 0, 'testworld_1_3': 0, 'testworld_0_5': 0, 'testworld_0_4': 0, 'testworld_0_7': 0, 'testworld_0_6': 0, 'testworld_0_1': 0, 'testworld_0_0': 0, 'testworld_0_3': 0, 'testworld_0_2': 0, 'testworld_5_7': 0, 'testworld_0_9': 0, 'testworld_0_8': 0, 'testworld_3_2': 0, 'testworld_3_3': 0, 'testworld_2_9': 0, 'testworld_2_8': 0, 'testworld_3_6': 0, 'testworld_3_7': 0, 'testworld_3_4': 0, 'testworld_3_5': 0, 'testworld_2_3': 0, 'testworld_2_2': 0, 'testworld_2_1': 0, 'testworld_2_0': 0, 'testworld_2_7': 0, 'testworld_2_6': 0, 'testworld_2_5': 0, 'testworld_2_4': 0, 'testworld_4_1': 0, 'testworld_4_0': 0, 'testworld_4_3': 0, 'testworld_4_2': 0, 'testworld_4_5': 0, 'testworld_4_4': 0, 'testworld_4_7': 0, 'testworld_4_6': 0, 'testworld_4_9': 0, 'testworld_4_8': 0, 'testworld_5_8': 0, 'testworld_5_2': 0, 'testworld_5_9': 1, 'testworld_3_0': 0, 'testworld_3_1': 0, 'testworld_5_3': 0, 'testworld_5_5': 0, 'testworld_5_0': 0, 'testworld_5_4': 0, 'testworld_5_1': 0, 'testworld_5_6': 0, 'testworld_3_8': 0}

    def test_equality(self):
        assert self.X == gw.GridWorld(REFERENCE_GWFILE)
        Y = gw.GridWorld()
        assert self.X != Y
        Y = gw.GridWorld(TRIVIAL_GWFILE)
        assert self.X != Y

    def test_dumploadloop(self):
        assert self.X == gw.GridWorld(self.X.dumps())

    def test_spec_realizable(self):
        assert is_realizable('gr1c', self.X.spec())

    def test_isEmpty(self):
        assert not self.X.isEmpty((0, 0))
        assert self.X.isEmpty((0, 1))
        assert not self.X.isEmpty((-1, 0))
        assert self.X.isEmpty((0, -1))

    def test_dumpsubworld(self):
        # No offset
        X_local = self.X.dumpsubworld((2,4), prefix="X")
        assert X_local.size() == (2, 4)
        assert X_local[0,0] == "X_0_0"
        assert not X_local.isEmpty((0,0))
        assert X_local.isEmpty((0,1))

        # Offset
        X_local = self.X.dumpsubworld((2,4), offset=(1,0), prefix="Xoff")
        assert X_local.size() == (2, 4)
        assert X_local[0,0] == "Xoff_0_0"
        assert X_local.isEmpty((0,0))
        assert X_local.isEmpty((0,1))
        assert not X_local.isEmpty((0,3))


def extract_coord_test():
    assert gw.extract_coord("test_3_0") == ("test", 3, 0)
    assert gw.extract_coord("obstacle_5_4_11") == ("obstacle_5", 4, 11)
    assert gw.extract_coord("test3_0") is None

def prefix_filt_test():
    assert gw.prefix_filt({"Y_0_0": 0, "Y_0_1": 1, "X_0_1_0": 1}, "Y") == {"Y_0_0": 0, "Y_0_1": 1}

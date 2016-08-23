"""Tests for the tulip.gridworld."""
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

UNREACHABLE_GOAL_GWFILE = """
4 4
**G
  **

I* *
"""

TRIVIAL_GWFILE = """
2 2
*
"""


# Module-level fixture setup
def setUp():
    np.random.seed(0)  # Make pseudorandom number sequence repeatable


class GridWorld_test(object):
    def setUp(self):
        self.prefix = "testworld"
        self.X = gw.GridWorld(REFERENCE_GWFILE, prefix=self.prefix)
        self.Y_testpaths = gw.GridWorld(UNREACHABLE_GOAL_GWFILE,
                                        prefix=self.prefix)

    def tearDown(self):
        self.X = None
        self.Y_testpaths = None

    def test_reachability(self):
        # Reachability is assumed to be bidirectional
        assert not self.Y_testpaths.is_reachable((3,0), (0,2))
        assert not self.Y_testpaths.is_reachable((0,2), (3,0))
        assert self.Y_testpaths.is_reachable((1,1), (2,3))
        assert self.Y_testpaths.is_reachable((2,3), (1,1))

    def test_size(self):
        assert self.X.size() == (6, 10)

    def test_copy(self):
        Z = self.X.copy()
        assert Z is not self.X
        assert Z.W is not self.X.W
        assert Z == self.X

    def test_getitem(self):
        assert self.X.__getitem__(
            (0,0), nonbool=False) == self.prefix+"_"+str(0)+"_"+str(0)
        assert self.X.__getitem__(
            (-1,0), nonbool=False) == self.prefix+"_"+str(5)+"_"+str(0)
        assert self.X.__getitem__(
            (-1,-2), nonbool=False) == self.prefix+"_"+str(5)+"_"+str(8)

    def test_state(self):
        assert self.X.state((2,3), nonbool=False) == {
            'testworld_3_9': 0, 'testworld_1_8': 0, 'testworld_1_9': 0,
            'testworld_1_4': 0, 'testworld_1_5': 0, 'testworld_1_6': 0,
            'testworld_1_7': 0, 'testworld_1_0': 0, 'testworld_1_1': 0,
            'testworld_1_2': 0, 'testworld_1_3': 0, 'testworld_0_5': 0,
            'testworld_0_4': 0, 'testworld_0_7': 0, 'testworld_0_6': 0,
            'testworld_0_1': 0, 'testworld_0_0': 0, 'testworld_0_3': 0,
            'testworld_0_2': 0, 'testworld_5_7': 0, 'testworld_0_9': 0,
            'testworld_0_8': 0, 'testworld_3_2': 0, 'testworld_3_3': 0,
            'testworld_2_9': 0, 'testworld_2_8': 0, 'testworld_3_6': 0,
            'testworld_3_7': 0, 'testworld_3_4': 0, 'testworld_3_5': 0,
            'testworld_2_3': 1, 'testworld_2_2': 0, 'testworld_2_1': 0,
            'testworld_2_0': 0, 'testworld_2_7': 0, 'testworld_2_6': 0,
            'testworld_2_5': 0, 'testworld_2_4': 0, 'testworld_4_1': 0,
            'testworld_4_0': 0, 'testworld_4_3': 0, 'testworld_4_2': 0,
            'testworld_4_5': 0, 'testworld_4_4': 0, 'testworld_4_7': 0,
            'testworld_4_6': 0, 'testworld_4_9': 0, 'testworld_4_8': 0,
            'testworld_5_8': 0, 'testworld_5_2': 0, 'testworld_5_9': 0,
            'testworld_3_0': 0, 'testworld_3_1': 0, 'testworld_5_3': 0,
            'testworld_5_5': 0, 'testworld_5_0': 0, 'testworld_5_4': 0,
            'testworld_5_1': 0, 'testworld_5_6': 0, 'testworld_3_8': 0}
        assert self.X.state((-1,0), nonbool=False) == {
            'testworld_3_9': 0, 'testworld_1_8': 0, 'testworld_1_9': 0,
            'testworld_1_4': 0, 'testworld_1_5': 0, 'testworld_1_6': 0,
            'testworld_1_7': 0, 'testworld_1_0': 0, 'testworld_1_1': 0,
            'testworld_1_2': 0, 'testworld_1_3': 0, 'testworld_0_5': 0,
            'testworld_0_4': 0, 'testworld_0_7': 0, 'testworld_0_6': 0,
            'testworld_0_1': 0, 'testworld_0_0': 0, 'testworld_0_3': 0,
            'testworld_0_2': 0, 'testworld_5_7': 0, 'testworld_0_9': 0,
            'testworld_0_8': 0, 'testworld_3_2': 0, 'testworld_3_3': 0,
            'testworld_2_9': 0, 'testworld_2_8': 0, 'testworld_3_6': 0,
            'testworld_3_7': 0, 'testworld_3_4': 0, 'testworld_3_5': 0,
            'testworld_2_3': 0, 'testworld_2_2': 0, 'testworld_2_1': 0,
            'testworld_2_0': 0, 'testworld_2_7': 0, 'testworld_2_6': 0,
            'testworld_2_5': 0, 'testworld_2_4': 0, 'testworld_4_1': 0,
            'testworld_4_0': 0, 'testworld_4_3': 0, 'testworld_4_2': 0,
            'testworld_4_5': 0, 'testworld_4_4': 0, 'testworld_4_7': 0,
            'testworld_4_6': 0, 'testworld_4_9': 0, 'testworld_4_8': 0,
            'testworld_5_8': 0, 'testworld_5_2': 0, 'testworld_5_9': 0,
            'testworld_3_0': 0, 'testworld_3_1': 0, 'testworld_5_3': 0,
            'testworld_5_5': 0, 'testworld_5_0': 1, 'testworld_5_4': 0,
            'testworld_5_1': 0, 'testworld_5_6': 0, 'testworld_3_8': 0}
        assert self.X.state((-1,-1), nonbool=False) == {
            'testworld_3_9': 0, 'testworld_1_8': 0, 'testworld_1_9': 0,
            'testworld_1_4': 0, 'testworld_1_5': 0, 'testworld_1_6': 0,
            'testworld_1_7': 0, 'testworld_1_0': 0, 'testworld_1_1': 0,
            'testworld_1_2': 0, 'testworld_1_3': 0, 'testworld_0_5': 0,
            'testworld_0_4': 0, 'testworld_0_7': 0, 'testworld_0_6': 0,
            'testworld_0_1': 0, 'testworld_0_0': 0, 'testworld_0_3': 0,
            'testworld_0_2': 0, 'testworld_5_7': 0, 'testworld_0_9': 0,
            'testworld_0_8': 0, 'testworld_3_2': 0, 'testworld_3_3': 0,
            'testworld_2_9': 0, 'testworld_2_8': 0, 'testworld_3_6': 0,
            'testworld_3_7': 0, 'testworld_3_4': 0, 'testworld_3_5': 0,
            'testworld_2_3': 0, 'testworld_2_2': 0, 'testworld_2_1': 0,
            'testworld_2_0': 0, 'testworld_2_7': 0, 'testworld_2_6': 0,
            'testworld_2_5': 0, 'testworld_2_4': 0, 'testworld_4_1': 0,
            'testworld_4_0': 0, 'testworld_4_3': 0, 'testworld_4_2': 0,
            'testworld_4_5': 0, 'testworld_4_4': 0, 'testworld_4_7': 0,
            'testworld_4_6': 0, 'testworld_4_9': 0, 'testworld_4_8': 0,
            'testworld_5_8': 0, 'testworld_5_2': 0, 'testworld_5_9': 1,
            'testworld_3_0': 0, 'testworld_3_1': 0, 'testworld_5_3': 0,
            'testworld_5_5': 0, 'testworld_5_0': 0, 'testworld_5_4': 0,
            'testworld_5_1': 0, 'testworld_5_6': 0, 'testworld_3_8': 0}

    def test_equality(self):
        assert self.X == gw.GridWorld(REFERENCE_GWFILE)
        Y = gw.GridWorld()
        assert self.X != Y
        Y = gw.GridWorld(TRIVIAL_GWFILE)
        assert self.X != Y

    def test_dumploadloop(self):
        assert self.X == gw.GridWorld(self.X.dumps())

    def test_spec_realizable_bool(self):
        spec = self.X.spec(nonbool=False)
        spec.moore = False
        spec.plus_one = False
        spec.qinit = r'\A \E'
        assert is_realizable('omega', spec)

    def test_spec_realizable(self):
        spec = self.X.spec()
        spec.moore = False
        spec.plus_one = False
        spec.qinit = r'\A \E'
        assert is_realizable('omega', spec)

    def check_is_empty(self, coord, expected):
        assert self.X.is_empty(coord) == expected

    def test_is_empty(self):
        for coord, expected in [((0, 0), False), ((0, 1), True),
                                ((-1, 0), False), ((0, -1), True)]:
            yield self.check_is_empty, coord, expected

    def check_is_empty_extend(self, coord, expected):
        assert self.X.is_empty(coord, extend=True) == expected

    def test_is_empty_extend(self):
        for coord, expected in [((0, 0), False), ((0, 1), True),
                                ((-1, 0), False), ((0, -1), False)]:
            yield self.check_is_empty_extend, coord, expected

    def test_dump_subworld(self):
        # No offset
        X_local = self.X.dump_subworld((2,4), prefix="X")
        assert X_local.size() == (2, 4)
        assert X_local.__getitem__((0,0), nonbool=False) == "X_0_0"
        assert not X_local.is_empty((0,0))
        assert X_local.is_empty((0,1))

        # Offset
        X_local = self.X.dump_subworld((2,4), offset=(1,0), prefix="Xoff")
        assert X_local.size() == (2, 4)
        assert X_local.__getitem__((0,0), nonbool=False) == "Xoff_0_0"
        assert X_local.is_empty((0,0))
        assert X_local.is_empty((0,1))
        assert not X_local.is_empty((0,3))

    def test_dump_subworld_extend(self):
        # No offset
        Xsize = self.X.size()
        X_local = self.X.dump_subworld((Xsize[0]+1, Xsize[1]), prefix="X",
                                      extend=True)
        X_local.goal_list = self.X.goal_list[:]
        X_local.init_list = self.X.init_list[:]
        assert X_local.size() == (7, 10)
        assert X_local.__getitem__((0,0), nonbool=False) == "X_0_0"
        assert not X_local.is_empty((0,0))
        assert X_local.is_empty((0,1))
        # Equal except for the last row, which should be all occupied in X_local
        X_local_s = X_local.dumps().splitlines()
        assert np.all(X_local_s[1:-1] == self.X.dumps().splitlines()[1:])
        assert not X_local.is_empty((6,1))
        assert X_local_s[-1] == "*"*10

        # Offset
        X_local = self.X.dump_subworld((3,4), offset=(-1,0), prefix="Xoff",
                                      extend=True)
        assert X_local.size() == (3, 4)
        assert X_local.__getitem__((0,0), nonbool=False) == "Xoff_0_0"
        assert not X_local.is_empty((0,0))
        assert not X_local.is_empty((0,1))
        assert not X_local.is_empty((0,3))
        assert X_local.is_empty((1,1))


class RandomWorld_test(object):
    def setUp(self):
        self.wall_densities = [.2, .4, .6]
        self.sizes = [(4,5), (4,5), (10,20)]
        self.rworlds = [
            gw.random_world(
                self.sizes[r], wall_density=self.wall_densities[r], prefix="Y")
            for r in range(len(self.sizes))]
        self.rworlds_ensuredfeasible = [
            gw.random_world(
                self.sizes[r], self.wall_densities[r], num_init=2,
                num_goals=2, ensure_feasible=True)
            for r in range(len(self.sizes))]

    def tearDown(self):
        self.rworlds = []

    def test_feasibility(self):
        for r in range(len(self.rworlds_ensuredfeasible)):
            print "test \"ensured feasible\" world index", r
            print self.rworlds_ensuredfeasible[r]
            assert self.rworlds_ensuredfeasible[r].is_reachable(
                self.rworlds_ensuredfeasible[r].init_list[0],
                self.rworlds_ensuredfeasible[r].init_list[1])
            assert self.rworlds_ensuredfeasible[r].is_reachable(
                self.rworlds_ensuredfeasible[r].init_list[1],
                self.rworlds_ensuredfeasible[r].goal_list[0])
            assert self.rworlds_ensuredfeasible[r].is_reachable(
                self.rworlds_ensuredfeasible[r].goal_list[0],
                self.rworlds_ensuredfeasible[r].goal_list[1])
            assert self.rworlds_ensuredfeasible[r].is_reachable(
                self.rworlds_ensuredfeasible[r].goal_list[1],
                self.rworlds_ensuredfeasible[r].init_list[0])

    def test_size(self):
        for r in range(len(self.rworlds)):
            print "test world index", r
            print self.rworlds[r]
            assert self.sizes[r] == self.rworlds[r].size()

    def test_density(self):
        for r in range(len(self.rworlds)):
            print "test world index", r
            print self.rworlds[r]
            (num_rows, num_cols) = self.rworlds[r].size()
            num_occupied = 0
            for i in range(num_rows):
                for j in range(num_cols):
                    if not self.rworlds[r].is_empty((i,j)):
                        num_occupied += 1
            assert (
                float(num_occupied) / (num_rows*num_cols) ==
                self.wall_densities[r])

RandomWorld_test.slow = True


def extract_coord_check(label, expected_coord):
    assert gw.extract_coord(label) == expected_coord


def extract_coord_test():
    for (label, expected_coord) in [
            ("test_3_0", ("test", 3, 0)),
            ("obstacle_5_4_11", ("obstacle_5", 4, 11)),
            ("test3_0", None)]:
        yield extract_coord_check, label, expected_coord


def eq_gridworld_check(G, H, eq):
    if eq:
        G == H
    else:
        not (G == H)


def eq_gridworld_test():
    empty = gw.GridWorld()
    trivial_nonempty = gw.GridWorld(TRIVIAL_GWFILE)
    trivial_diff = gw.GridWorld(TRIVIAL_GWFILE)
    if trivial_diff.is_empty((0, 0)):
        trivial_diff.mark_occupied((0, 0))
    else:
        trivial_diff.mark_empty((0, 0))
    trivial_nonempty_2goals = gw.GridWorld(TRIVIAL_GWFILE)
    trivial_nonempty_2goals.goal_list = [(0, 0), (1, 1)]
    trivial_nonempty_2init = gw.GridWorld(TRIVIAL_GWFILE)
    trivial_nonempty_2init.init_list = [(0, 0), (1, 1)]
    for (G, H, is_equal) in [
            (gw.GridWorld(), gw.GridWorld(), True),
            (empty, trivial_nonempty, False),
            (trivial_nonempty_2goals, trivial_nonempty, False),
            (trivial_nonempty_2init, trivial_nonempty, False),
            (trivial_nonempty, trivial_diff, False),
            (gw.unoccupied((3, 5)), gw.unoccupied((1, 1)), False)]:
        yield eq_gridworld_check, G, H, is_equal


def narrow_passage_test():
    G = gw.narrow_passage((5, 10), num_init=1, num_goals=1)
    assert G.is_reachable(G.init_list[0], G.goal_list[0])


def scale_gridworld_test():
    G = gw.unoccupied((1, 2))
    assert G.size() == (1, 2)
    assert G.scale().size() == G.size()
    assert G.scale(xf=1, yf=1).size() == G.size()
    assert G.scale(xf=2).size() == (1, 4)
    assert G.scale(yf=2).size() == (2, 2)
    assert G.scale(xf=3, yf=4).size() == (4, 6)


def add_trolls_test():
    G = gw.unoccupied((3, 5))
    G.init_list = [(0, 0)]
    G.goal_list = [(0, 4)]
    spc = gw.add_trolls(G, [((2, 2), 1)], get_moves_lists=False)
    spc.moore = False
    spc.plus_one = False
    spc.qinit = r'\A \E'
    assert is_realizable('omega', spc)

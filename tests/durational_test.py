# This test includes unit tests for the tulip.transys.DurationalKripkeTree module.
#
# * kripke_container_test(): a unit test for constructing a DurationalKripkeTree object,
#   ensuring that all the getter functions return the correct values

from tulip.transys import DurationalKripkeTree
from tulip.trajectory import DiscreteTimeFiniteTrajectory as Trajectory


def kripke_container_test():
    s0 = (0, 0)
    K = DurationalKripkeTree(s0)

    s1 = (1, 1)
    K.add_state(s1, False)

    (best_goal, best_cost) = K.get_best_goal()
    assert best_goal is None

    tstep = 0.5
    traj01 = Trajectory(tstep, [s0, (0.5, 0.5), s1])
    cost01 = 1.0
    K.connect(s0, s1, traj01, cost01)

    (best_goal, best_cost) = K.get_best_goal()
    assert best_goal is None

    s2 = (2, 2)
    K.add_state(s2, True)

    (best_goal, best_cost) = K.get_best_goal()
    assert best_goal is None

    traj02 = Trajectory(tstep, [(0.2 * i, 0.2 * i) for i in range(11)])
    cost02 = 3.0
    K.connect(s0, s2, traj02, cost02)

    (best_goal, best_cost) = K.get_best_goal()
    assert best_goal == s2
    assert best_cost == cost02

    traj12 = Trajectory(tstep, [s1, (1.5, 1.5), s2])
    cost12 = 1.0
    K.connect(s1, s2, traj12, cost12)

    (best_goal, best_cost) = K.get_best_goal()
    assert best_goal == s2
    assert best_cost == cost01 + cost12

    assert K.get_initial() == s0
    assert set(K.get_states()) == {s0, s1, s2}
    assert set(K.get_goals()) == {s2}

    (cost, trajectory) = K.get_transition(s0, s1)
    assert cost == cost01
    assert trajectory == traj01

    (cost, trajectory) = K.get_transition(s0, s2)
    assert cost == cost02
    assert trajectory == traj02

    (cost, trajectory) = K.get_transition(s1, s2)
    assert cost == cost12
    assert trajectory == traj12

    assert K.get_transition(s0, s0) is None

    assert K.get_cost_to_come(s0) == 0
    assert K.get_cost_to_come(s1) == cost01
    assert K.get_cost_to_come(s2) == cost01 + cost12

    assert K.get_best_parent(s0) is None
    assert K.get_best_parent(s1) == s0
    assert K.get_best_parent(s2) == s1

    assert K.get_trajectories_to(s0) == []
    assert K.get_trajectories_to(s1) == [traj01]
    assert K.get_trajectories_to(s2) == [traj12, traj01]

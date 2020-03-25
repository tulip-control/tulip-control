import copy
from nose.tools import raises
from tulip.trajectory import DiscreteTimeFiniteTrajectory as Trajectory


@raises(ValueError)
def invalid_trajectory_test():
    trajectory = Trajectory(0.1, [])


def trajectory_test():
    def labeling_function(state):
        label = set()
        if state < 1:
            label.add("s")
        if state < 5:
            label.add("m")
        return label

    num_states = 20
    t_eps = 1e-3
    time_step = 0.1
    states = [i * 0.5 for i in range(num_states)]
    trajectory = Trajectory(time_step, states)
    print("trajectory: {}".format(trajectory))
    assert len(trajectory) == num_states
    assert trajectory[0.1 * t_eps] == states[0]
    for t in [time_step + 0.1 * time_step * i for i in range(11)]:
        s = states[1] + ((states[2] - states[1]) * (t - time_step) / time_step)
        assert abs(trajectory[t] - s) < 1e-6

    assert trajectory.get_states() == states
    assert trajectory.get_final_time() == time_step * (num_states - 1)
    assert trajectory.get_final_state() == states[-1]
    assert trajectory.get_final_state() == trajectory[trajectory.get_final_time()]

    ftw = trajectory.get_finite_timed_word(labeling_function)
    expected_ftw = [({"s", "m"}, 0.2), ({"m"}, 0.8), (set(), 0.9)]
    _check_ftw_equivalence(ftw, expected_ftw)


def concat_test():
    tstep = 0.1
    n1 = 10
    n2 = 15
    states1 = [i for i in range(n1)]
    states2 = [i for i in range(n2)]
    traj1 = Trajectory(tstep, copy.copy(states1))
    traj2 = Trajectory(tstep, copy.copy(states2))

    traj = traj1 + traj2
    assert traj1.get_states() == states1
    assert traj2.get_states() == states2
    assert traj.get_states() == states1 + states2[1:]
    assert traj1.get_final_time() == tstep * (n1 - 1)
    assert traj2.get_final_time() == tstep * (n2 - 1)
    assert traj.get_final_time() == traj1.get_final_time() + traj2.get_final_time()

    traj1 += traj2
    assert traj1.get_states() == traj.get_states()
    assert traj1.get_final_time() == traj.get_final_time()
    assert traj2.get_states() == states2
    assert traj2.get_final_time() == tstep * (n2 - 1)


def _check_ftw_equivalence(ftw1, ftw2):
    assert len(ftw1) == len(ftw2)
    for i in range(len(ftw1)):
        assert ftw1[i][0] == ftw2[i][0]
        assert abs(ftw1[i][1] - ftw2[i][1]) < 1e-6

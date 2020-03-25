import numpy
from tulip.trajectory import DiscreteTimeFiniteTrajectory as Trajectory
from tulip.spec.prioritized_safety import PrioritizedSpecification
from tulip.transys.cost import VectorCost
from tulip.mvp import solve_incremental_sifltlgx as solve_mvp
from tulip.mvp import IncrementalPrimitives


def _diff(s0, s1):
    return tuple(s1[i] - s0[i] for i in range(len(s0)))


def _dist(s0, s1):
    return numpy.linalg.norm(_diff(s0, s1))


def _inc(s0, s1, num_steps):
    diff = _diff(s0, s1)
    return tuple(float(diff[i] / num_steps) for i in range(len(s0)))


def _steer(s0, s1):
    step_size = 0.1
    num_steps = int(round(_dist(s0, s1) / step_size))
    inc = _inc(s0, s1, num_steps)
    states = [
        tuple(s0[i] + j * inc[i] for i in range(len(s0))) for j in range(num_steps)
    ]
    return Trajectory(step_size, states)


def _sample(n):
    return ((n + 1) % 10, (n + 1) / 10)


def _near(state, others):
    bound = 3
    return [other for other in others if _dist(state, other) < bound]


def _labeling(state):
    ret = set()
    if state[1] < 0.5:
        ret.add("S")
    if state[0] < 2 or state[1] > 1.0:
        ret.add("C")
    if state[0] > 4:
        ret.add("G")
    return ret


def _rule_s(l1, l2):
    return "S" in l1


def _rule_c(l1, l2):
    return "C" in l1


def _check_state_equal(s1, s2):
    assert _dist(s1, s2) < 1e-6


def incremental_siflflgx_test():
    primitives = IncrementalPrimitives(
        sampling=_sample, steering=_steer, near=_near, labeling=_labeling,
    )

    spec1 = PrioritizedSpecification()
    spec1.add_rule(_rule_s, priority=1, level=0, is_siFLTLGX=True)
    spec1.add_rule(_rule_c, priority=1, level=1, is_siFLTLGX=True)
    (best_cost, best_goal, best_trajectories, K) = solve_mvp(
        initial=(0, 0), goal_label="G", spec=spec1, primitives=primitives, num_it=100,
    )
    _check_state_equal(best_goal, (5, 0.5))
    assert best_cost.almost_equal(VectorCost([0, 2.7, 4.5]))
    n = 5
    expected_state_sequence = [(i, 0.1 * i) for i in range(n)]
    assert len(best_trajectories) == n
    for i in range(n):
        _check_state_equal(best_trajectories[n - i - 1][0], expected_state_sequence[i])

    spec2 = PrioritizedSpecification()
    spec2.add_rule(_rule_c, priority=1, level=0, is_siFLTLGX=True)
    spec2.add_rule(_rule_s, priority=1, level=1, is_siFLTLGX=True)
    (best_cost, best_goal, best_trajectories, K) = solve_mvp(
        initial=(0, 0), goal_label="G", spec=spec2, primitives=primitives, num_it=100,
    )
    _check_state_equal(best_goal, (5, 1.5))
    assert best_cost.almost_equal(VectorCost([0, 3.5, 5.0]))
    expected_state_sequence = [(0, 0), (1.0, 0.1), (2.0, 1.2), (3.0, 1.3), (4.0, 1.4)]
    assert len(best_trajectories) == len(expected_state_sequence)
    for i in range(len(expected_state_sequence)):
        _check_state_equal(best_trajectories[n - i - 1][0], expected_state_sequence[i])

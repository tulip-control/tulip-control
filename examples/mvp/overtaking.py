# The overtaking example shown in
# T. Wongpiromsarn, K. Slutsky, E. Frazzoli, and U. Topcu
# Minimum-Violation Planning for Autonomous Systems: Theoretical and Practical Considerations, 2021

import os
import time
import math
import random
from matplotlib import pyplot as plt
from dubins import shortest_path
from tulip.transys import DurationalKripkeTree, DurationalKripkeGraph
from tulip.mvp import update_incremental_sifltlgx as update_mvp
from tulip.mvp import IncrementalPrimitives
from tulip.trajectory import DiscreteTimeFiniteTrajectory as Trajectory
from tulip.spec.prioritized_safety import PrioritizedSpecification
from vehicle_state import VehicleState
from straight_road import Road, Obstacle
from plot import plot_all


# Obstacle
obstacle_x0 = 20.0
obstacle_y0 = 0.5
obstacle_length = 5.0
obstacle_width = 1.4
obstacle = Obstacle(obstacle_x0, obstacle_y0, obstacle_width, obstacle_length)

# Road
lane_width = VehicleState.half_width * 2 + obstacle_width + Obstacle.lateral_clearance
road_length = 2 * (obstacle_x0 + obstacle_length)
road = Road(lane_width, road_length)

# Domain
Sx = (VehicleState.rear_length, road.length - VehicleState.front_length)
Sy = (-3, road.width + 3)
Stheta = (-math.pi / 4.0, math.pi / 4.0)
near_constant = (
    8 * (Sx[1] - Sx[0]) * (Sy[1] - Sy[0]) * (Stheta[1] - Stheta[0]) / math.pi
)
near_constant = 2 * (near_constant ** (1 / 3))

# Initial state and goal
initial = VehicleState((VehicleState.rear_length, road.lane_width / 2, 0))
goal_x = road.length - VehicleState.front_length - 10.0

# Labels
label_road = "R"
label_lane = "L"
label_clear = "C"
label_safe = "S"
label_goal = "G"


# Primitive functions
turning_radius = 1.0
step_size = 0.1
speed = 1
tstep = step_size / speed


def steer_dubins(q0, q1):
    path = shortest_path(q0.configuration, q1.configuration, turning_radius)
    configurations, _ = path.sample_many(step_size)
    states = [VehicleState(state) for state in configurations]
    return Trajectory(tstep, states)


def uniform_sampling(n):
    x = random.uniform(Sx[0], Sx[1])
    y = random.uniform(Sy[0], Sy[1])
    theta = random.uniform(Stheta[0], Stheta[1])
    return VehicleState((x, y, theta))


def func_near(state, others):
    n = len(others)
    distance_bound = near_constant * ((math.log(n) / n) ** (1 / 3))
    ret = []
    for other in others:
        if state.dist(other) < distance_bound:
            ret.append(other)
    return ret


def func_labeling(state):
    labels = set()

    is_in_lane = False
    if road.contain(state.footprint):
        labels.add(label_road)
        if road.is_in_lane(state.footprint):
            is_in_lane = True
            labels.add(label_lane)

    if not obstacle.is_in_collision(state.footprint):
        labels.add(label_safe)
        if not obstacle.is_in_clearance_zone(state.footprint):
            labels.add(label_clear)

    if state.x >= goal_x and is_in_lane:
        labels.add(label_goal)
    return labels


primitives = IncrementalPrimitives(
    sampling=uniform_sampling,
    steering=steer_dubins,
    near=func_near,
    labeling=func_labeling,
)


# Specification
spec = PrioritizedSpecification(True)
spec.add_rule(lambda l1, l2: label_safe in l1, priority=1, level=0)
spec.add_rule(lambda l1, l2: label_road in l1, priority=1, level=1)
spec.add_rule(lambda l1, l2: label_clear in l1, priority=1, level=2)
spec.add_rule(lambda l1, l2: label_lane in l1, priority=1, level=2)

# Kripke structure
K = DurationalKripkeGraph(initial)

# Plot
fig, ax = plt.subplots()
plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")

# Update solution
num_it = 20
total_it = 40
sampling_step_size = int(1 / step_size)
comp_time_list = []
best_cost_list = []

for i in range(total_it):
    start_time = time.clock()
    update_mvp(
        K=K,
        goal_label=label_goal,
        spec=spec,
        primitives=primitives,
        num_it=num_it,
        sampling_step_size=sampling_step_size,
    )
    comp_time = time.clock() - start_time

    (best_trajectories, best_cost) = K.get_trajectories_to(K.get_goals())

    print("Iteration: {}, Time: {:.3f}, Cost: {}".format(i, comp_time, best_cost))
    comp_time_list.append(comp_time)
    best_cost_list.append(best_cost)

    # Plot
    if ((i + 1) % 10) == 0:
        ax.set_facecolor((0.0, 0.3, 0.3))

        plot_all(
            ax,
            [road, obstacle],
            [trajectory for (c, trajectory) in K.W.values()],
            best_trajectories,
            K.S,
        )

        ax.set_xlim((0, road.length))
        ax.set_ylim((-5, road.width + 5))

        filename = "overtaking" + str(i + 1) + ".png"
        plt.savefig(os.path.join(plot_path, filename))
        ax.clear()

print("Total time: {}".format(sum(comp_time_list)))

iteration_list = [i + 1 for i in range(len(comp_time_list))]
fig, ax = plt.subplots()
ax.plot(iteration_list, comp_time_list)
ax.set_xlabel("Iteration")
ax.set_ylabel("Computation time (seconds)")


valid_iterations = [
    i + 1
    for i in range(len(best_cost_list))
    if best_cost_list[i] != DurationalKripkeTree.INF_COST
]
fig, ax = plt.subplots()
if len(valid_iterations) > 0:
    for level in range(spec.get_num_levels()):
        ax.plot(
            valid_iterations,
            [best_cost_list[i - 1][level] for i in valid_iterations],
            label="i = {}".format(level+1),
        )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("The level of unsafety")
        ax.legend(loc="best")

plt.show()

# 1000 iterations: [0, 0, 2.2199999999999998]

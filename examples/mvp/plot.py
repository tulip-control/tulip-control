# Plotting functionalities for overtaking.py example

import math


def _update_bound(bound, new):
    bound[0] = min(bound[0], new[0])
    bound[1] = max(bound[1], new[1])


def _concatenate_trajectories(trajectories):
    if len(trajectories) == 0:
        return None

    trajectory = trajectories[0]
    for i in range(1, len(trajectories)):
        trajectory = trajectory + trajectories[i]

    return trajectory


def plot_trajectory(ax, trajectory, linestyle, linewidth, zorder, draw_footprint=False):
    state_sequence = trajectory.get_states()
    x_list = [state.x for state in state_sequence]
    y_list = [state.y for state in state_sequence]
    ax.plot(
        x_list, y_list, linestyle, linewidth=linewidth, zorder=zorder,
    )
    xbound = [min(x_list), max(x_list)]
    ybound = [min(y_list), max(y_list)]
    if draw_footprint:
        for state in state_sequence:
            (this_xbound, this_ybound) = state.plot(ax)
            _update_bound(xbound, this_xbound)
            _update_bound(ybound, this_ybound)

    return (xbound, ybound)


def plot_all(ax, objects, trajectories_all, best_trajectories, states):
    xbound = [math.inf, -math.inf]
    ybound = [math.inf, -math.inf]

    for obj in objects:
        (this_xbound, this_ybound) = obj.plot(ax)
        _update_bound(xbound, this_xbound)
        _update_bound(ybound, this_ybound)

    for trajectory in trajectories_all:
        (this_xbound, this_ybound) = plot_trajectory(
            ax, trajectory, "c-", linewidth=1, zorder=0
        )
        _update_bound(xbound, this_xbound)
        _update_bound(ybound, this_ybound)

    trajectory = _concatenate_trajectories(best_trajectories)
    if trajectory is not None:
        (this_xbound, this_ybound) = plot_trajectory(
            ax, trajectory, "k-", linewidth=5, zorder=4
        )
        _update_bound(xbound, this_xbound)
        _update_bound(ybound, this_ybound)

        trajectory[0].plot(ax, zorder=4)

    for state in states:
        ax.plot([state.x], [state.y], "bo", markersize=5)

    return (xbound, ybound)

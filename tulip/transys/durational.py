# Copyright (c) 2020 by California Institute of Technology
# and Iowa State University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder(s) nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CALTECH
# OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.

"""Module for defining durational structure, e.g., durational Kripke Structure"""

import math
from tulip.transys.graph_algorithms import (
    dijkstra_single_source_multiple_targets_general as dijkstra,
)


class DurationalKripkeGraph(object):
    """A class for defining a container for Durational Kripke structure

    As opposed to tulip.transys.KripkeStructure,
    this class better supports incremental construction
    and storage of continuous trajectory.

    Examples
    ========
    >>> K = DurationalKripkeGraph(initial)

    * initial is the initial state

    >>> K.add_state(s, is_goal)

    Add a state s and mark whether it is a goal.

    >>> K.connect(s1, s2, trajectory, cost)

    Add a transition from s1 to s2 with the given cost and trajectory.

    >>> K.get_trajectories_to(target_set)

    Return a tuple (trajectory, cost) to reach a state in target_set from the initial state.

    The returned trajectory is represented by a list
    [t_0, ..., t_n] where t_i is the trajectory of the transition
    (s_i, s_{i+1}), s_0 is the initial state, s_{n+1} is in target_set,
    and s_i is a parent of s_{i+1}.

    >>> K.get_initial()

    Return the initial state

    >>> K.get_states()

    Return the list of states

    >>> K.get_goals()

    Return the list of states that are considered to be goal states.

    >>> K.get_transition(s1, s2)

    Return a tuple (cost, trajectory) of transition from s1 to s2.
    If the transition does not exist, None is returned.
    """

    INF_COST = math.inf

    def __init__(self, initial):
        # Initial state
        self.initial = initial

        # The set of states
        self.S = [initial]

        # Transition cost
        # W is a dictionary whose key is a pair (s1,s2), representing the
        # transition from state s1 to state s2 and whose value is a tuple
        # (cost, trajectory)
        self.W = {}

        # The set of goal states
        self.Sgoals = []

        # The children of each state.
        self.children = {}

        # The parents of each state
        self._init_connection_of(initial)

    def add_state(self, s, is_goal):
        """Add a state s

        @param s: a state to be added
        @param is_goal: a Boolean that indicates whether s is a goal
        """
        self.S.append(s)
        if is_goal:
            self.Sgoals.append(s)
        self._init_connection_of(s)

    def connect(self, s1, s2, trajectory, cost):
        """Add a transition from s1 to s2

        @param s1, s2: states in self.S
        @param trajectory: a trajectory that connects s1 and s2
        @param cost: the cost of the transition from s1 to s2
        """
        if cost == self.INF_COST:
            return False

        self.children[s1].append(s2)
        self.W[(s1, s2)] = (cost, trajectory)
        return True

    def remove_transition(self, s1, s2):
        try:
            self.children[s1].remove(s2)
            del self.W[(s1, s2)]
            return True
        except (ValueError, KeyError):
            return False

    def get_trajectories_to(self, target_set):
        """Return a tuple (trajectory, cost) from the initial state to a set of states

        The returned trajectory is represented by a list
        [t_0, ..., t_n] where t_i is the trajectory of the transition
        (s_i, s_{i+1}), s_0 is the initial state, s_{n+1} is in target_set,
        and s_i is a parent of s_{i+1}.

        @param target_set: a list of states in self.S
        """

        def get_connection_from(s):
            return [(u, self.W[(s, u)][0]) for u in self.children[s]]

        (cost, state_path) = dijkstra(self.initial, target_set, get_connection_from)

        trajectories = []

        if len(state_path) == 0:
            return (trajectories, self.INF_COST)

        state_iter = iter(state_path)
        curr_state = next(state_iter)

        while True:
            try:
                next_state = next(state_iter)
                trajectories.append(self.W[(curr_state, next_state)][1])
                curr_state = next_state
            except StopIteration:
                break
        return (trajectories, cost)

    def get_initial(self):
        """Return the initial state"""
        return self.initial

    def get_states(self):
        """Return the set of states"""
        return self.S

    def get_goals(self):
        """Return the set of goal states"""
        return self.Sgoals

    def get_transition(self, s1, s2):
        """Return a tuple (cost, trajectory) for transition from s1 to s2"""
        return self.W.get((s1, s2), None)

    def _init_connection_of(self, s):
        """Initialize components of the graph related to state s"""
        # The parents of s
        self.children[s] = []


class DurationalKripkeTree(DurationalKripkeGraph):
    """A class for defining a container for Durational Kripke structure that is a tree

    The key difference from DurationalKripkeGraph is that each vertex only has at most one parent.
    It inherits all the function of DurationalKripkeGraph.
    The following examples demonstrate their differences and additional functions.


    Examples
    ========
    >>> K = DurationalKripkeTree(initial)

    * initial is the initial state

    >>> K.connect(s1, s2, trajectory, cost)

    Add a transition from s1 to s2 with the given cost and trajectory.
    It also updates the cost to come and best parent of s2.

    >>> K.get_trajectories_to(target_set)

    Return a tuple (trajectory, cost) to reach a state in target_set from the initial state
    by inductively getting the best parent the vertices.

    >>> K.get_best_goal()

    Return the goal with minimum cost to come.

    >>> K.get_cost_to_come(s)

    Return cost to come to a state s in K.get_states().

    >>> K.get_best_parent(s)

    Return the best parent of a state s in K.get_states().
    """

    def __init__(self, initial):

        # The best parent of each state
        self.parent = {}

        # The best cost to come at each state
        self.J = {}

        super(DurationalKripkeTree, self).__init__(initial)

        self.J[initial] = 0

    def connect(self, s1, s2, trajectory, cost):
        """Update the best parent and cost to come of all the affected vertices
        based on the transition from s1 to s2

        @param s1, s2: states in self.S
        @param trajectory: a trajectory that connects s1 and s2
        @param cost: the cost of the transition from s1 to s2
        """
        if cost == self.INF_COST:
            return False

        # Update best parent of s2 if the cost decreases
        new_cost_to_come = self.J[s1] + cost

        if new_cost_to_come >= self.J[s2]:
            return False

        # Update parent and cost to come of all the affected vertices
        parent = self.parent[s2]
        if parent is not None:
            self.remove_transition(parent, s2)

        super(DurationalKripkeTree, self).connect(s1, s2, trajectory, cost)
        self.parent[s2] = s1
        self.J[s2] = new_cost_to_come

        self._update_cost_to_come(self.children[s2])

        return True

    def get_trajectories_to(self, target_set):
        """Return a tuple (trajectory, cost) from the initial state to a set of states
        by following (backward) the best parent

        The returned trajectory is represented by a list
         [t_0, ..., t_n] where t_i is the trajectory of the transition
         (s_i, s_{i+1}), s_0 is the initial state, s_{n+1} = s,
         and s_i is the best parent of s_{i+1}.

        @param target_set: a list of states in self.S
        """
        (best_state, best_cost) = self._get_best_state(target_set)
        trajectories = []

        if best_state is None:
            return (trajectories, best_cost)

        s = best_state

        while s != self.initial:
            parent = self.parent[s]
            if parent is None:
                return (trajectories, best_cost)
            trajectories.insert(0, self.W[parent, s][1])
            s = parent
        return (trajectories, best_cost)

    def get_best_goal(self):
        """Return the goal with the minimum cost to come"""
        return self._get_best_state(self.Sgoals)

    def get_cost_to_come(self, s):
        """Return cost to come to s"""
        return self.J.get(s, None)

    def get_best_parent(self, s):
        """Return the best parent of s"""
        return self.parent.get(s, None)

    def _init_connection_of(self, s):
        super(DurationalKripkeTree, self)._init_connection_of(s)

        # For a tree, _connection stores the unique parent of each state s
        self.parent[s] = None

        # The best cost to come at s
        self.J[s] = self.INF_COST

    def _get_best_state(self, state_set):
        """Get the state in state_set with minimum cost to come"""
        best_state = None
        best_cost = self.INF_COST
        for state in state_set:
            this_cost = self.J[state]
            if this_cost != self.INF_COST and this_cost < best_cost:
                best_state = state
                best_cost = this_cost
        return (best_state, best_cost)

    def _update_cost_to_come(self, states):
        while len(states) > 0:
            s = states.pop(0)

            parent = self.parent[s]
            cost_to_come = self.J[parent] + self.W[(parent, s)][0]

            if cost_to_come < self.J[s]:
                self.J[s] = cost_to_come
                states.extend(self.children[s])

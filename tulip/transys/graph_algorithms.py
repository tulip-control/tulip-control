# Copyright (c) 2020 by California Institute of Technology
# and University of Texas at Austin
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

"""Graph algorithms"""

import copy
from heapq import heappush, heappop


def dijkstra_single_source_multiple_targets(
        graph, source, target_set, cost_key="cost"):
    """Return the shortest path on graph from a source state to a set of target states

    @param graph: the graph of type tuplip.transys.LabeledDiGraph
    @param source: a state in graph identified as the source state
    @param target_set: a set of list of states identified as the target states
    @param cost_key: the transition attribute that indicates the cost of the transition
        The type of cost c can be anything that contains __add__ function
        such that c + 0 is defined.

    @return (cost, path) where path is a list representing the optimal path
        and cost is the sum of the edge costs on this path.
    """

    dist = {source: 0}
    visited = set()
    Q = [(0, source, [])]

    while Q:
        (cost, u, path) = heappop(Q)
        if u in visited:
            continue
        visited.add(u)
        path_to_u = copy.copy(path)
        path_to_u.append(u)
        if u in target_set:
            return (cost, path_to_u)

        for transition in graph.transitions.find(u):
            v = transition[1]
            if v in visited:
                continue
            current_cost = dist.get(v, None)
            new_cost = transition[2][cost_key] + cost
            if not current_cost or new_cost < current_cost:
                dist[v] = new_cost
                heappush(Q, (new_cost, v, path_to_u))

    return (float("inf"), ())


def dijkstra_multiple_sources_multiple_targets(
        graph, source_set, target_set, cost_key="cost"):
    """Return the shortest path on graph from a set of source states
    to a set of target states.

    @param graph: the graph of type tuplip.transys.LabeledDiGraph
    @param source_set: a set or list of states identified as the source states
    @param target_set: a set or list of states identified as the target states
    @param cost_key: the transition attribute that indicates the cost of the transition
        The type of cost c can be anything that contains __add__ function
        such that c + 0 is defined.

    @return (cost, path) where path is a list representing the optimal path
        and cost is the sum of the edge costs on this path.
    """

    best_cost = float("inf")
    best_path = []
    for source in source_set:
        (cost, path) = dijkstra_single_source_multiple_targets(
            graph, source, target_set, cost_key
        )
        if cost < best_cost:
            best_cost = cost
            best_path = path
    return (best_cost, best_path)

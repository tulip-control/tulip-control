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
#
"""Minimum violation planning module"""
import itertools as _itr
import typing as _ty

import tulip.transys as _trs
import tulip.transys.cost as _cost
import tulip.transys.graph_algorithms as _gralgo
import tulip.spec.prioritized_safety as _prio
import tulip._utils as _utl


KS = _trs.KripkeStructure
WFA = _trs.WeightedFiniteStateAutomaton
PrioSpec = _prio.PrioritizedSpecification
ProductState = _utl.n_tuple(2)


def _get_rule_violation_cost(
        from_prod_state:
            ProductState,
        to_prod_state:
            ProductState,
        spec:
            PrioSpec,
        to_ap:
            set
        ) -> list[float]:
    """Return the rule violation cost on the transition from
    from_prod_state to to_prod_state.

    This follows Definition 8 (weighted finite automaton)
    and Definition 9 (product automaton) in
    J. Tumova, G.C Hall, S. Karaman, E. Frazzoli and D. Rus.
    Least-violating Control Strategy Synthesis with Safety Rules, HSCC 2013.

    @param from_prod_state, to_prod_state:
        tuple (ks_state, aut_state)
        where ks_state is the state of the Kripke struture
        and aut_state is a tuple,
        representing the state of the finite automaton.
        aut_state[i] corresponds to spec[i].
    @param spec:
        the prioritized safety specification
    @param to_ap:
        the atomic proposition of to_prod_state
    """
    from_spec_state = from_prod_state[1]
    to_spec_state = to_prod_state[1]
    cost = [0] * spec.get_num_levels()
    for idx, rule in enumerate(spec):
        rule_transitions = rule.automaton().transitions.find(
            from_spec_state[idx], to_spec_state[idx])
        rule_transition_labels = [
            set(transition[2]["letter"])
            for transition in rule_transitions]
        if to_ap not in rule_transition_labels:
            cost[rule.level()] += rule.priority()
    return cost


def _add_transition(
        from_prod_states:
            ProductState,
        to_prod_states:
            ProductState,
        ks:
            KS,
        spec:
            PrioSpec,
        trans_ks_cost,
        fa:
            WFA):
    """Add a transition from from_prod_state to to_prod_state to fa.

    This follows Definition 8 (weighted finite automaton)
    and Definition 9 (product automaton) in
    J. Tumova, G.C Hall, S. Karaman, E. Frazzoli and D. Rus.
    Least-violating Control Strategy Synthesis with Safety Rules, HSCC 2013.

    @param from_prod_state, to_prod_state:
        tuple (ks_state, aut_state)
        where ks_state is the state of the Kripke struture
        and aut_state is a tuple,
        representing the state of the finite automaton.
        aut_state[i] corresponds to spec[i].
    @param spec:
        the prioritized safety specification
    @param trans_ks_cost:
        the cost of transition from from_prod_state[0] to to_prod_state[0]
        in ks
    """
    for from_prod_state in from_prod_states:
        for to_prod_state in to_prod_states:
            _add_transition_edge(
                from_prod_state, to_prod_state,
                ks, spec, trans_ks_cost, fa)


def _add_transition_edge(
        from_prod_state,
        to_prod_state,
        ks:
            KS,
        spec:
            PrioSpec,
        trans_ks_cost,
        fa:
            WFA
        ) -> None:
    """Add to `fa` transition between states."""
    to_ap = ks.states[to_prod_state[0]]["ap"]
    cost = _get_rule_violation_cost(
        from_prod_state, to_prod_state, spec, to_ap)
    cost.append(trans_ks_cost)
    fa.transitions.add(
        from_prod_state,
        to_prod_state,
        {"cost": _cost.VectorCost(cost)})


def _construct_weighted_product_automaton(
        ks:
            KS,
        spec:
            PrioSpec
        ) -> tuple[WFA, str]:
    """Compute the weighted product automaton ks times spec

    This follows Definition 9 (product automaton) in
    J. Tumova, G.C Hall, S. Karaman, E. Frazzoli and D. Rus.
    Least-violating Control Strategy Synthesis with Safety Rules, HSCC 2013.

    @param spec:
        the prioritized safety specification
    """
    null_state = "null"
    while null_state in ks.states:
        null_state += "0"
    fa = WFA()
    fa.states.add_from(_itr.product(ks.states, spec.get_states()))
    fa.states.add_from(_itr.product([null_state], spec.get_states()))
    fa.states.initial.add_from(
        _itr.product([null_state], spec.get_initial_states()))
    fa.states.accepting.add_from(
        _itr.product(ks.states, spec.get_accepting_states()))
    fa.atomic_propositions.add_from(ks.atomic_propositions)
    for transition in ks.transitions.find():
        from_ks_state = transition[0]
        to_ks_state = transition[1]
        trans_ks_cost = transition[2].get("cost", 0)
        _add_transition(
            _itr.product([from_ks_state], spec.get_states()),
            _itr.product([to_ks_state], spec.get_states()),
            ks,
            spec,
            trans_ks_cost,
            fa)
    _add_transition(
        fa.states.initial,
        _itr.product(ks.states.initial, spec.get_states()),
        ks,
        spec,
        0,
        fa)
    return (fa, null_state)


def solve(
        ks:
            KS,
        goal_label,
        spec:
            PrioSpec
        ) -> tuple:
    """Solve the minimum violation planning problem

    This follows from
    J. Tumova, G.C Hall, S. Karaman, E. Frazzoli and D. Rus.
    Least-violating Control Strategy Synthesis with Safety Rules, HSCC 2013.

    @param goal_label:
        a label in ks.atomic_propositions that indicates the goal
    @param spec:
        the prioritized safety specification
    @return:
        (best_cost, best_path, weighted_product_automaton) where
        * best_cost is the optimal cost of reaching the goal
        * best_path is the optimal path to the goal
        * weighted_product_automaton is the weighted product
          automaton ks times spec
    """
    if not isinstance(ks, KS):
        raise TypeError(ks)
    if not isinstance(spec, PrioSpec):
        raise TypeError(spec)
    if ks.atomic_propositions != spec.atomic_propositions:
        raise ValueError(ks, spec)
    wpa, null_state = _construct_weighted_product_automaton(ks, spec)
    goal_states = [
        state
        for state in ks.states
        if goal_label in ks.states[state]["ap"]]
    accepting_goal_states = _trs.SubSet(wpa.states.accepting)
    accepting_goal_states.add_from(
        _itr.product(goal_states, spec.get_states()))
    cost, product_path = _gralgo.dijkstra_multiple_sources_multiple_targets(
        wpa,
        wpa.states.initial,
        accepting_goal_states,
        cost_key="cost")
    state_path = [
        state[0]
        for state in product_path
        if state[0] != null_state]
    return (cost, state_path, product_path, wpa)

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
"""Compositions of transys"""

import copy
from itertools import product
from functools import reduce
from operator import or_
from tulip.transys import KripkeStructure as KS
from tulip.transys import WeightedKripkeStructure as WKS
from tulip.transys import MarkovChain as MC
from tulip.transys import MarkovDecisionProcess as MDP


def sum_values(*values):
    """Return the sum of values, considering only elements that are not None.
    An item v,w in values can be anything that contains __add__ function
    such that v+0 and v+w is defined.
    """
    # Cannot simply return sum([v for v in values if v is not None])
    # because it does 0 + v which will not work for v of type, e.g., VectorCost
    current = 0
    for v in values:
        if v is not None:
            current = v + current
    return current


def mult_values(*values):
    """Return the product of values, considering only elements that are not None.
    An item v,w in values can be anything that contains __mul__ function
    such that v*1 and v*w is defined.
    """
    current = 1
    for v in values:
        if v is not None:
            current = v * current
    return current


def neglect_none(*values):
    """Return a tuple of values, considering only elements that are not None.
    If the tuple only has one element, just return that element.
    """
    ret = tuple([v for v in values if v is not None])
    if len(ret) == 1:
        return ret[0]
    return ret


def synchronous_parallel(models, transition_attr_operations={}):
    """Construct a model that represents
    the synchronous paralel composition of the given models
    (i.e., tensor product in graph theory
    https://en.wikipedia.org/wiki/Tensor_product_of_graphs)

    It follows definition 2.42 (synchronous product) in
    U{[BK08] <https://tulip-control.sourceforge.io/doc/bibliography.html#bk08>},
    with the only exception that Act does not have the be the same
    for all the models in models.

    @type models: `list` of objects of types `KripeStructure`, `WeightedKripkeStructure`,
        `MarkovChain` or `MarkovDecisionProcess`
    @type transition_attr_operations: `dict` whose key is the transition attribute key
        and value is the operation to be performed for this transition attribute.
        For an attribute whose operation is not specified,
        a tuple of attribute values from all models will be used.

    @return: the synchronous parallel composition of all the objects in models
    @rtype: one of the following types:
        * L{transys.KripkeStructure}
        * L{transys.WeightedKripkeStructure}
        * L{transys.MarkovChain}
        * L{transys.MarkovDecisionProcess}
    """

    # Let models = [K_1, ..., K_n].
    # Let
    # * prod_states = [S_1, ..., S_n] where S_i is the set of states of K_i
    # * prod_initials = [I_1, ..., I_n] where I_i is the set of initial
    #   states of K_i
    prod_states = []
    prod_initials = []

    # Construct prod_states and prod_initials and
    # construct the composed model ts with all the atomic propositions.
    composed_type = _get_composed_model_type(models)
    if composed_type is None:
        raise TypeError("Can only compose [WKS, KS] or [MDP, MC, KS]")
    ts = composed_type()
    for model in models:
        prod_states.append(set(model.states))
        prod_initials.append(model.states.initial)
        ts.atomic_propositions.add_from(model.atomic_propositions)

    # Compute the state of ts: S = S_1 \times ... \times S_n.
    # Also, compute the label at each state (s_1, ..., s_n).
    # By definition L(s_1, ..., s_n) = \bigcup_i L_i(s_i)
    # where L_i is the labeling function of K_i.
    for state in product(*prod_states):
        ts.states.add(state)
        ts.states[state]["ap"] = reduce(
            or_, [models[i].states[state[i]]["ap"] for i in range(len(models))]
        )

    # Compute the initial state of ts: I = I_1 \times ... \times I_n
    for state in product(*prod_initials):
        ts.states.initial.add(state)

    # Compute the set of actions
    if type(ts) == MDP:
        prod_actions = [list(m.actions) for m in models if type(m) == MDP]
        if len(prod_actions) == 1:
            ts.actions.add_from(prod_actions[0])
        else:
            ts.actions.add_from(list(product(*prod_actions)))

    if WKS.cost_label not in transition_attr_operations:
        transition_attr_operations[WKS.cost_label] = sum_values
    if MC.probability_label not in transition_attr_operations:
        transition_attr_operations[MC.probability_label] = mult_values
    if MDP.action_label not in transition_attr_operations:
        transition_attr_operations[MDP.action_label] = neglect_none

    # Compute the transition of ts according to the rule
    # ((s_1, ..., s_n), (a_1, ..., a_n), (s_1', ..., s_n'))
    # in the transition relation of ts
    # iff (s_i, a_i, s_i') is in the transition relation of K_i for all i
    for from_state in ts.states:
        transitions = [
            models[coord].transitions.find(from_state[coord])
            for coord in range(len(models))
        ]
        for transition in product(*transitions):
            to_state = tuple(t[1] for t in transition)
            attr = _get_transition_attr(transition, transition_attr_operations)
            ts.transitions.add(
                from_state, to_state, attr,
            )

    return ts


def apply_policy(model, policy):
    """Apply the policy on the MarkovDecisionProcess and return the induced MarkovChain

    @type model: `MarkovDecisionProcess`
    @type policy: An object such that for any state in model.states, policy[state]
        is an action in model.actions

    @return: the induced MarkovChain
    """
    result_model_type = _get_apply_policy_model_type(model)
    result = result_model_type()
    result.states.add_from(model.states)
    result.states.initial.add_from(model.states.initial)
    result.atomic_propositions.add_from(model.atomic_propositions)

    for state in model.states:
        result.states[state]["ap"] = copy.deepcopy(model.states[state]["ap"])
        action = policy[state]
        for transition in model.transitions.find(state):
            if transition[2][MDP.action_label] != action:
                continue
            transition_attr = copy.deepcopy(transition[2])
            del transition_attr[MDP.action_label]
            result.transitions.add(transition[0], transition[1], transition_attr)

    return result




def _get_transition_attr(trans_prod, transition_attr_operations):
    """Return the attribute of a transition constructed by taking the product
    of transitions in trans_prod.

    @type trans_prod: `list` of `Transitions` objects
    @type transition_attr_operations: `dict` whose key is the transition attribute key
        and value is the operation to be performed for this transition attribute.
        For an attribute whose operation is not specified,
        a tuple of attribute values from all models will be used.
    """
    trans_attr = {}
    for idx, trans in enumerate(trans_prod):
        for trans_attr_key, trans_attr_value in trans[2].items():
            if trans_attr_key not in trans_attr:
                trans_attr[trans_attr_key] = [None for i in range(len(trans_prod))]
            trans_attr[trans_attr_key][idx] = trans_attr_value

    for key, val in trans_attr.items():
        operation = transition_attr_operations.get(key, None)
        if operation is None:
            trans_attr[key] = tuple(val)
        else:
            trans_attr[key] = operation(*val)
    return trans_attr


def _get_composed_model_type(models):
    """Return the class of model obtained from taking a composition of those given by models

    @type models: `list` of objects of type KripkeStructure, WeightedKripkeStructure,
        MarkovChain or MarkovDecisionProcess
    """
    if all(type(m) in [MDP, MC, KS] for m in models):
        if any(type(m) == MDP for m in models):
            return MDP
        if any(type(m) == MC for m in models):
            return MC
        return KS

    if all(type(m) in [WKS, KS] for m in models):
        if any(type(m) == WKS for m in models):
            return WKS
        return KS

    return None


def _get_apply_policy_model_type(model):
    """Return the class of model obtained from applying a policy on the given model

    @type model: KripkeStructure, WeightedKripkeStructure or MarkovDecisionProcess
    """
    if type(model) == MDP:
        return MC
    raise TypeError("Cannot apply policy for model of type {}".format(type(model)))

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
"""Compositions of transys."""
import copy
import functools as _ft
import itertools as _itr
import operator as _op
import typing as _ty

import tulip.transys as _trs


KS = _trs.KripkeStructure
WKS = _trs.WeightedKripkeStructure
MC = _trs.MarkovChain
MDP = _trs.MarkovDecisionProcess


def sum_values(*values) -> float:
    """Return the sum of values that are not `None`.

    An item `v, w` in `values` can be anything with
    an `__add__` method,
    so that `v + 0` and `v + w` be defined.
    """
    return _ft.reduce(_op.add,
        filter(_is_not_none, values))


def mult_values(*values) -> float:
    """Return product of values that are not `None`.

    An item `v, w` in `values` can be anything with
    a `__mul__` nethod
    so that `v * 1` and `v * w` be defined.
    """
    return _ft.reduce(_op.mul,
        filter(_is_not_none, values))


def neglect_none(*values) -> tuple:
    """Return a tuple of values that are not `None`.

    If the tuple only has one element,
    then return that element (unpack).
    """
    ret = tuple(filter(_is_not_none, values))
    if len(ret) == 1:
        return ret[0]
    return ret


def _is_not_none(x):
    return x is not None


Model = (
    KS |
    WKS |
    MC |
    MDP)


def synchronous_parallel(
        models:
            list[Model],
        transition_attr_operations:
            dict |
            None=None
        ) -> Model:
    """Construct synchronous parallel composition.

    Construct a model that represents the synchronous parallel composition
    of the given models (i.e., tensor product in graph theory
    <https://en.wikipedia.org/wiki/Tensor_product_of_graphs>)

    It follows definition 2.42 (synchronous product) in
    [BK08](
        https://tulip-control.sourceforge.io/doc/bibliography.html#bk08),
    with the only exception that Act does not have the be the same
    for all the models in models.

    @param transition_attr_operations:
        `dict` whose key is the
        transition attribute key and
        value is the operation to be performed for this transition attribute.
        For an attribute whose operation is not specified,
        a tuple of attribute values from all models will be used.
    @return:
        the synchronous parallel composition of
        all the objects in models
    """
    if transition_attr_operations is None:
        transition_attr_operations = dict()
    # Let models = [K_1, ..., K_n].
    # Let
    # * prod_states = [S_1, ..., S_n]
    #   where S_i is the set of states of K_i
    # * prod_initials = [I_1, ..., I_n]
    #   where I_i is the set of initial
    #   states of K_i
    prod_states = list()
    prod_initials = list()
    #
    # Construct prod_states and
    # prod_initials and
    # construct the composed model ts with
    # all the atomic propositions.
    composed_type = _get_composed_model_type(models)
    if composed_type is None:
        raise TypeError(
            'Can only compose [WKS, KS] or '
            '[MDP, MC, KS]')
    ts = composed_type()
    for model in models:
        prod_states.append(set(model.states))
        prod_initials.append(model.states.initial)
        ts.atomic_propositions.add_from(
            model.atomic_propositions)
    #
    # Compute the state of ts:
    # S = S_1 \times ... \times S_n.
    # Also, compute the label at
    # each state (s_1, ..., s_n).
    # By definition
    # L(s_1, ..., s_n) = \bigcup_i L_i(s_i)
    # where L_i is the labeling function of K_i.
    for state in _itr.product(*prod_states):
        ts.states.add(state)
        ts.states[state]["ap"] = _ft.reduce(
            _op.or_, [
                models[i].states[state[i]]["ap"]
                for i in range(len(models))])
    #
    # Compute the initial state of ts:
    # I = I_1 \times ... \times I_n
    for state in _itr.product(*prod_initials):
        ts.states.initial.add(state)
    #
    # Compute the set of actions
    if type(ts) == MDP:
        prod_actions = [
            list(m.actions)
            for m in models
            if type(m) == MDP]
        if len(prod_actions) == 1:
            ts.actions.add_from(prod_actions[0])
        else:
            ts.actions.add_from(_itr.product(*prod_actions))
    if WKS.cost_label not in transition_attr_operations:
        transition_attr_operations[WKS.cost_label] = sum_values
    if MC.probability_label not in transition_attr_operations:
        transition_attr_operations[MC.probability_label] = mult_values
    if MDP.action_label not in transition_attr_operations:
        transition_attr_operations[MDP.action_label] = neglect_none
    #
    # Compute the transition of ts according to the rule
    # ((s_1, ..., s_n), (a_1, ..., a_n), (s_1', ..., s_n'))
    # in the transition relation of ts
    # iff (s_i, a_i, s_i') is in
    # the transition relation of K_i for all i
    for from_state in ts.states:
        transitions = [
            models[coord].transitions.find(from_state[coord])
            for coord in range(len(models))]
        for transition in _itr.product(*transitions):
            to_state = tuple(t[1] for t in transition)
            attr = _get_transition_attr(
                transition, transition_attr_operations)
            ts.transitions.add(
                from_state, to_state, attr)
    return ts


def apply_policy(
        model:
            MDP,
        policy
        ) -> MC:
    """Apply `policy` on `model` and return the induced Markov chain.

    Apply the policy `policy` on the Markov decision process `model`
    and return the induced Markov chain.

    @type policy:
        An object such that for any state in `model.states`,
        `policy[state]` is an action in `model.actions`
    @return:
        the induced Markov chain
    """
    result_model_type = _get_apply_policy_model_type(model)
    result = result_model_type()
    result.states.add_from(model.states)
    result.states.initial.add_from(model.states.initial)
    result.atomic_propositions.add_from(model.atomic_propositions)
    for state in model.states:
        result.states[state]["ap"] = copy.deepcopy(
            model.states[state]["ap"])
        action = policy[state]
        for transition in model.transitions.find(state):
            if transition[2][MDP.action_label] != action:
                continue
            transition_attr = copy.deepcopy(transition[2])
            del transition_attr[MDP.action_label]
            result.transitions.add(
                transition[0],
                transition[1],
                transition_attr)
    return result


def _get_transition_attr(
        trans_prod:
            list,
        transition_attr_operations:
            dict
        ) -> dict:
    """Return product of transitions in `trans_prod`.

    Return the attribute of
    a transition constructed by
    taking the product of
    transitions in `trans_prod`.

    @param trans_prod:
        `list` of `Transitions` objects
    @param transition_attr_operations:
        `dict` whose key is the transition
        attribute key and value is the
        operation to be performed for
        this transition attribute.

        For an attribute whose operation
        is not specified, a tuple of
        attribute values from all models
        will be used.
    """
    trans_attr = dict()
    for idx, trans in enumerate(trans_prod):
        for attr_key, attr_value in trans[2].items():
            if attr_key not in trans_attr:
                trans_attr[attr_key
                    ] = [None] * len(trans_prod)
            trans_attr[attr_key][idx] = attr_value
    for key, value in trans_attr.items():
        operation = transition_attr_operations.get(key)
        if operation is None:
            trans_attr[key] = tuple(value)
        else:
            trans_attr[key] = operation(*value)
    return trans_attr


Model = (
    KS |
    WKS |
    MC |
    MDP)


def _get_composed_model_type(
        models:
            list[Model]
        ) -> (
            type[Model] |
            None):
    """Return class representing composition.

    Return the class of model obtained from
    composing the items in `models`.
    """
    def mk(types):
        def isinstance_(instance):
            return isinstance(instance, types)
        return isinstance_
    type_is_ok = mk(MDP | MC | KS)
    is_mdp = mk(MDP)
    is_mc = mk(MC)
    is_wks = mk(WKS)
    all_types_ok = all(map(
        type_is_ok, models))
    any_is_mdp = any(map(
        is_mdp, models))
    any_is_mc = any(map(
        is_mc, models))
    any_is_wks = any(map(
        is_wks, models))
    if not all_types_ok:
        return None
    if any_is_mdp:
        return MDP
    if any_is_mc:
        return MC
    if any_is_wks:
        return WKS
    return KS


def _get_apply_policy_model_type(
        model:
            KS |
            WKS |
            MDP
        ) -> type[MC]:
    """Return class representing policy application.

    Return the class of model that results when
    a policy is applied to `model`.
    """
    if isinstance(model, MDP):
        return MC
    raise TypeError(
        f'Cannot apply policy for model of type {type(model)}')

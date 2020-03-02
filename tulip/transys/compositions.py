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

from tulip.transys import KripkeStructure as KS
from tulip.transys import WeightedKripkeStructure as WKS
from itertools import product
from functools import reduce
from operator import or_


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
                trans_attr[trans_attr_key] = [
                    None for i in range(len(trans_prod))]
            trans_attr[trans_attr_key][idx] = trans_attr_value

    for key, val in trans_attr.items():
        operation = transition_attr_operations.get(key, None)
        if operation is None:
            trans_attr[key] = tuple(val)
        else:
            trans_attr[key] = operation(*val)
    return trans_attr


def sum_values(*values):
    """Return the sum of values, considering only elements that are not None.
    An item v in values can be anything that contains __add__ function
    such that v + 0 is defined.
    """
    # Cannot simply return sum([v for v in values if v is not None])
    # because it does 0 + v which will not work for v of type, e.g., VectorCost
    current = 0
    for v in values:
        if v is not None:
            current = v + current
    return current


def ks_synchronous_parallel(
    ks_models, transition_attr_operations={
        "cost": sum_values}):
    """Construct a KripkeStructure object that represents
    the synchronous paralel composition of KripeStructure objects
    (i.e., tensor product in graph theory
    https://en.wikipedia.org/wiki/Tensor_product_of_graphs)

    It follows definition 2.42 (synchronous product) in Baier-Katoen,
    with the only exception that Act does not have the be the same
    for all the models in ks_models.

    @type ks_models: `list` of `KripeStructure` objects
    @type transition_attr_operations: `dict` whose key is the transition attribute key
        and value is the operation to be performed for this transition attribute.
        For an attribute whose operation is not specified,
        a tuple of attribute values from all models will be used.

    @return: the synchronous parallel composition of all the KripeStructure
        objects in ks_models
    @rtype: L{transys.KripkeStructure}
    """

    # Let ks_models = [K_1, ..., K_n] where K_i is of type KripkeStructure.
    # Let
    # * prod_states = [S_1, ..., S_n] where S_i is the set of states of K_i
    # * prod_initials = [I_1, ..., I_n] where I_i is the set of initial
    #   states of K_i
    prod_states = []
    prod_initials = []

    # Construct prod_states and prod_initials and
    # construct the composed model ts with all the atomic propositions.
    ts = WKS() if any(isinstance(m, WKS) for m in ks_models) else KS()
    for model in ks_models:
        assert isinstance(model, KS)
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
            or_, [ks_models[i].states[state[i]]["ap"] for i in range(len(ks_models))]
        )

    # Compute the initial state of ts: I = I_1 \times ... \times I_n
    for state in product(*prod_initials):
        ts.states.initial.add(state)

    # Compute the transition of ts according to the rule
    # ((s_1, ..., s_n), (a_1, ..., a_n), (s_1', ..., s_n'))
    # in the transition relation of ts
    # iff (s_i, a_i, s_i') is in the transition relation of K_i for all i
    for from_state in ts.states:
        transitions = [
            ks_models[coord].transitions.find(from_state[coord])
            for coord in range(len(ks_models))
        ]
        for transition in product(*transitions):
            to_state = tuple(t[1] for t in transition)
            ts.transitions.add(
                from_state,
                to_state,
                _get_transition_attr(transition, transition_attr_operations),
            )

    return ts

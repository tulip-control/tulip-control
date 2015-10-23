# Copyright (c) 2013-2015 by California Institute of Technology
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
# 3. Neither the name of the California Institute of Technology nor
#    the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
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
"""Algorithms on Kripke structures and Automata"""
from __future__ import absolute_import
import logging
import copy
from tulip.transys.transys import FiniteTransitionSystem
from tulip.transys.automata import BuchiAutomaton
import networkx as nx
from tulip.interfaces import ltl2ba as ltl2baint
# possible future:
# from tulip.transys.transys import TransitionSystem
# from tulip.transys.automata import Automaton
from tulip.transys.labeled_graphs import LabeledDiGraph


_hl = 40 * '-'
logger = logging.getLogger(__name__)


# build parser once only
parser = ltl2baint.Parser()


def ltl2ba(formula):
    """Convert LTL formula to Buchi Automaton using ltl2ba.

    @type formula: `str(formula)` must be admissible ltl2ba input

    @return: Buchi automaton whose edges are annotated
        with Boolean formulas as `str`
    @rtype: [`Automaton`]
    """
    ltl2ba_out = ltl2baint.call_ltl2ba(str(formula))
    symbols, g, initial, accepting = parser.parse(ltl2ba_out)
    ba = Automaton('Buchi', alphabet=symbols)
    ba.add_nodes_from(g)
    ba.add_edges_from(g.edges_iter(data=True))
    ba.initial_nodes = initial
    ba.accepting_sets = accepting
    logger.info('Resulting automaton:\n\n{ba}\n'.format(ba=ba))
    return ba


def _multiply_mutable_states(self, other, prod_graph, prod_sys):
    def prod_ids2states(prod_state_id, self, other):
        (idx1, idx2) = prod_state_id
        state1 = self.states._int2mutant(idx1)
        state2 = other.states._int2mutant(idx2)
        prod_state = (state1, state2)
        return prod_state

    def label_union(nx_label):
        (v1, v2) = nx_label

        if v1 is None or v2 is None:
            raise Exception(
                'At least one factor has unlabeled state, '
                "or the state sublabel types don't match.")
        try:
            return v1 | v2
        except:
            pass
        try:
            return v2 + v2
        except:
            raise TypeError(
                'The state sublabel types should support ' +
                'either | or + for labeled system products.')

    def state_label_union(attr_dict):
        prod_attr_dict = dict()
        for k, v in attr_dict.iteritems():
            prod_attr_dict[k] = label_union(v)
        return prod_attr_dict

    # union of state labels from the networkx tuples
    for prod_state_id, attr_dict in prod_graph.nodes_iter(data=True):
        prod_attr_dict = state_label_union(attr_dict)
        prod_state = prod_ids2states(prod_state_id, self, other)

        prod_sys.states.add(prod_state)
        prod_sys.states.add(prod_state, **prod_attr_dict)
    print(prod_sys.states)

    # prod of initial states
    inits1 = self.states.initial
    inits2 = other.states.initial

    prod_init = []
    for (init1, init2) in zip(inits1, inits2):
        new_init = (init1, init2)
        prod_init.append(new_init)
    prod_sys.states.initial |= prod_init

    # # multiply mutable states (only the reachable added)
    # if self.states.mutants or other.states.mutants:
    #     for idx, prod_state_id in enumerate(prod_graph.nodes_iter() ):
    #         prod_state = prod_ids2states(prod_state_id, self, other)
    #         prod_sys.states.mutants[idx] = prod_state
    #
    #     prod_sys.states.min_free_id = idx +1
    # # no else needed: otherwise self already not mutant

    # action labeling is taken care by nx,
    # since transition taken at a time
    for from_state_id, to_state_id, edge_dict in prod_graph.edges_iter(data=True):

        from_state = prod_ids2states(from_state_id, self, other)
        to_state = prod_ids2states(to_state_id, self, other)

        prod_sys.transitions.add(
            from_state, to_state, **edge_dict)
    return prod_sys


# binary operators (for magic binary operators: see above)
def tensor_product(self, other, prod_sys=None):
    """Return strong product with given graph.

    Reference
    =========
    http://en.wikipedia.org/wiki/Strong_product_of_graphs
    nx.algorithms.operators.product.strong_product
    """
    prod_graph = nx.product.tensor_product(self, other)
    # not populating ?
    if prod_sys is None:
        if self.states.mutants or other.states.mutants:
            mutable = True
        else:
            mutable = False
        prod_sys = LabeledDiGraph(mutable=mutable)
    prod_sys = self._multiply_mutable_states(
        other, prod_graph, prod_sys)
    return prod_sys


def cartesian_product(self, other, prod_sys=None):
    """Return Cartesian product with given graph.

    If u,v are nodes in C{self} and z,w nodes in C{other},
    then ((u,v), (z,w) ) is an edge in the Cartesian product of
    self with other if and only if:
        - (u == v) and (z,w) is an edge of C{other}
        OR
        - (u,v) is an edge in C{self} and (z == w)

    In system-theoretic terms, the Cartesian product
    is the interleaving where at each step,
    only one system/process/player makes a move/executes.

    So it is a type of parallel system.

    This is an important distinction with the C{strong_product},
    because that includes "diagonal" transitions, i.e., two
    processes executing truly concurrently.

    Note that a Cartesian interleaving is different from a
    strong interleaving, because the latter can skip states
    and transition directly along the diagonal.

    For a model of computation, strong interleaving
    would accurately model the existence of multiple cores,
    not just multiple processes executing on a single core.

    References
    ==========
      - U{http://en.wikipedia.org/wiki/Cartesian_product_of_graphs}
      - networkx.algorithms.operators.product.cartesian_product
    """
    prod_graph = nx.product.cartesian_product(self, other)
    # not populating ?
    if prod_sys is None:
        if self.states.mutants or other.states.mutants:
            mutable = True
        else:
            mutable = False
        prod_sys = LabeledDiGraph(mutable=mutable)
    prod_sys = self._multiply_mutable_states(
        other, prod_graph, prod_sys)
    return prod_sys


def ts_sync_prod(ts1, ts2):
    """Synchronous (tensor) product with other FTS.

    @type ts1, ts2: L{FiniteTransitionSystem}
    """
    prod_ts = FiniteTransitionSystem()
    # union of AP sets
    prod_ts.atomic_propositions |= \
        ts1.atomic_propositions | ts2.atomic_propositions
    # use more label sets, instead of this explicit approach
    #
    # for synchronous product: Cartesian product of action sets
    # prod_ts.actions |= ts1.actions * ts2.actions
    prod_ts = super(FiniteTransitionSystem, self).tensor_product(
        ts, prod_sys=prod_ts)
    return prod_ts


def sync_prod(ts, ba):
    """Synchronous product between (BA, TS), or (BA1, BA2).

    The result is always a L{BuchiAutomaton}:

        - If C{ts_or_ba} is a L{FiniteTransitionSystem} TS,
            then return the synchronous product BA * TS.

            The accepting states of BA * TS are those which
            project on accepting states of BA.

        - If C{ts_or_ba} is a L{BuchiAutomaton} BA2,
            then return the synchronous product BA * BA2.

            The accepting states of BA * BA2 are those which
            project on accepting states of both BA and BA2.

            This definition of accepting set extends
            Def.4.8, p.156 U{[BK08]
            <http://tulip-control.sourceforge.net/doc/bibliography.html#bk08>}
            to NBA.

    Synchronous product TS * BA or TS1 * TS2.

    Returns a Finite Transition System, because TS is
    the first term in the product.

    Changing term order, i.e., BA * TS, returns the
    synchronous product as a BA.

    Caution
    =======
    This method includes semantics for true\in\Sigma (p.916, U{[BK08]
    <http://tulip-control.sourceforge.net/doc/bibliography.html#bk08>}),
    so there is a slight overlap with logic grammar.  In other
    words, not completely isolated from logics.

    See Also
    ========
    L{_ts_ba_sync_prod}

    @param ts_or_ba: other with which to take synchronous product
    @type ts_or_ba: L{FiniteTransitionSystem} or L{BuchiAutomaton}

    @return: self * ts_or_ba
    @rtype: L{BuchiAutomaton}

    See Also
    ========
    __mul__, async_prod, BuchiAutomaton.sync_prod, tensor_product
    Def. 2.42, pp. 75--76 U{[BK08]
    <http://tulip-control.sourceforge.net/doc/bibliography.html#bk08>}
    Def. 4.62, p.200 U{[BK08]
    <http://tulip-control.sourceforge.net/doc/bibliography.html#bk08>}

    @param ts_or_ba: system with which to take synchronous product
    @type ts_or_ba: L{FiniteTransitionSystem} or L{BuchiAutomaton}

    @return: synchronous product C{self} x C{ts_or_ba}
    @rtype: L{FiniteTransitionSystem}
    """
    if not isinstance(ba, BuchiAutomaton):
        raise Exception
    if not isinstance(ts, FiniteTransitionSystem):
        raise Exception


def add(self, other):
    """Merge two Finite Transition Systems.

    States, Initial States, Actions, Atomic Propositions and
    State labels and Transitions of the second Transition System
    are merged into the first and take precedence, overwriting
    existing labeling.

    Example
    =======
    This can be useful to construct systems quickly by creating
    standard "pieces" using the functions: line_labeled_with,
    cycle_labeled_with

    >>> n = 4
    >>> L = n*['p'] # state labeling
    >>> ts1 = line_labeled_with(L, n-1)
    >>> ts1.plot()
    >>>
    >>> L = n*['p']
    >>> ts2 = cycle_labeled_with(L)
    >>> ts2.states.add('s3', ap={'!p'})
    >>> ts2.plot()
    >>>
    >>> ts3 = ts1 +ts2
    >>> ts3.transitions.add('s'+str(n-1), 's'+str(n) )
    >>> ts3.default_layout = 'circo'
    >>> ts3.plot()

    See Also
    ========
    L{line_labeled_with}, L{cycle_labeled_with}

    @param other: system to merge with
    @type other: C{FiniteTransitionSystem}

    @return: merge of C{self} with C{other}, union of states,
        initial states, atomic propositions, actions, edges and
        labelings, those of C{other} taking precedence over C{self}.
    @rtype: L{FiniteTransitionSystem}
    """
    if not isinstance(other, FiniteTransitionSystem):
        msg = (
            'other class must be FiniteTransitionSystem.\n'
            'Got instead:\n\t' + str(other) +
            '\nof type:\n\t' + str(type(other)))
        raise TypeError(msg)
    self.atomic_propositions |= other.atomic_propositions
    self.actions |= other.actions
    # add extra states & their labels
    for state, label in other.states.find():
        if state not in self.states:
            self.states.add(state)
        if label:
            self.states[state]['ap'] = label['ap']
    self.states.initial |= set(other.states.initial)
    # copy extra transitions (be careful w/ labeling)
    for (from_state, to_state, label_dict) in other.transitions.find():
        # labeled edge ?
        if not label_dict:
            self.transitions.add(from_state, to_state)
        else:
            sublabel_value = label_dict['actions']
            self.transitions.add(
                from_state, to_state, actions=sublabel_value)
    return copy.copy(self)


def async_prod(self, ts):
    """Asynchronous product TS1 x TS2 between FT Systems.

    See Also
    ========
    __or__, sync_prod, cartesian_product
    Def. 2.18, p.38 U{[BK08]
    <http://tulip-control.sourceforge.net/doc/bibliography.html#bk08>}
    """
    if not isinstance(ts, FiniteTransitionSystem):
        raise TypeError('ts must be a FiniteTransitionSystem.')
    if self.states.mutants or ts.states.mutants:
        mutable = True
    else:
        mutable = False
    # union of AP sets
    prod_ts = FiniteTransitionSystem(mutable=mutable)
    prod_ts.atomic_propositions |= \
        self.atomic_propositions | ts.atomic_propositions
    # for parallel product: union of action sets
    prod_ts.actions |= self.actions | ts.actions
    prod_ts = super(FiniteTransitionSystem, self).cartesian_product(
        ts, prod_sys=prod_ts)
    return prod_ts

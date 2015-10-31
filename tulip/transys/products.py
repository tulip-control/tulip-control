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
"""Products between automata and transition systems"""
from __future__ import absolute_import
import logging
import warnings
from tulip.transys import transys
from tulip.transys import automata


logger = logging.getLogger(__name__)
_hl = 40 * '-'


class OnTheFlyProductAutomaton(automata.BuchiAutomaton):
    """Dynamically extends itself by adding successors.

    Note that performs on-the-fly BA * TS.
    The given TS can be explicit or on-the-fly,
    depending on what you pass to C{__init__}.

    The state space serves as the set of "visited" states.

    Note that this is equivalent to adding successors
    to both the queue and the set of visited states
    at the end of each iteration during a search,
    instead of adding each successor to the visited states
    when it is poped from the queue.
    """
    def __init__(self, ba, ts):
        self.ba = ba
        self.ts = ts
        super(OnTheFlyProductAutomaton, self).__init__()
        self.atomic_propositions |= ts.atomic_propositions
        self._add_initial()

    def _add_initial(self):
        ts = self.ts
        ba = self.ba
        s0s = set(ts.states.initial)
        q0s = set(ba.states.initial)

        logger.debug('\n' + _hl + '\n' +
                     ' Product BA construction:' +
                     '\n' + _hl + '\n')

        if not s0s:
            msg = (
                'Transition System has no initial states !\n'
                '=> Empty product system.\n'
                'Did you forget to define initial states ?'
            )
            warnings.warn(msg)

        for s0 in s0s:
            logger.debug('initial state:\t' + str(s0))

            for q0 in q0s:
                enabled_ba_trans = find_ba_succ(q0, s0, ts, ba)

                # q0 blocked ?
                if not enabled_ba_trans:
                    continue

                # which q next ?     (note: curq0 = q0)
                for (curq0, q, sublabels) in enabled_ba_trans:
                    new_sq0 = (s0, q)

                    self.states.add(new_sq0)
                    self.states.initial.add(new_sq0)

                    # accepting state ?
                    if q in ba.states.accepting:
                        self.states.accepting.add(new_sq0)

    def add_successors(self, s, q):
        """Add the successors of (s, q) to the state space.

        @param s: TS state

        @param q: BA state

        @return: those successors that are new states
        """
        sq = (s, q)
        ts = self.ts
        ba = self.ba

        logger.debug('Creating successors from'
                     ' product state:\t' + str(sq))

        # get next states
        next_ss = ts.states.post(s)
        next_sqs = set()
        for next_s in next_ss:
            enabled_ba_trans = find_ba_succ(q, next_s, ts, ba)

            if not enabled_ba_trans:
                continue

            (new_sqs, new_accepting) = find_prod_succ(
                sq, next_s, enabled_ba_trans,
                self, ba, ts
            )

            next_sqs.update(new_sqs)
            self.states.accepting |= new_accepting

        # new_sqs = {x for x in next_sqs if x not in self}

        logger.debug('next product states: ' + str(next_sqs))
        logger.debug('new unvisited product states: ' + str(new_sqs))

        return new_sqs

    def add_all_states(self):
        """Iterate L{add_successors} until all states are added.

        In other words until the state space
        reaches a fixed point.
        """
        Q = set(self.states)
        while Q:
            Qnew = set()
            for sq in Q:
                (s, q) = sq
                new = self.add_successors(s, q)
                Qnew.update(new)
            Q = Qnew


def ts_ba_sync_prod(transition_system, buchi_automaton):
    """Construct transition system for the synchronous product TS * BA.

    Def. 4.62, p.200 U{[BK08]
    <http://tulip-control.sourceforge.net/doc/bibliography.html#bk08>}

    Erratum
    =======
    note the erratum: P_{pers}(A) is ^_{q\in F} !q, verified from:
    http://www-i2.informatik.rwth-aachen.de/~katoen/errata.pdf

    See Also
    ========
    L{ba_ts_sync_prod}, L{sync_prod}

    @return: C{(product_ts, persistent_states)}, where:
        - C{product_ts} is the synchronous product TS * BA
        - C{persistent_states} are those in TS * BA which
            project on accepting states of BA.
    @rtype:
        - C{product_TS} is a L{transys.FiniteTransitionSystem}
        - C{persistent_states} is the set of states which project
            on accepting states of the Buchi Automaton BA.
    """
    # if not hasattr(transition_system, FiniteTransitionSystem):
    #    msg = 'transition_system not transys.FiniteTransitionSystem.\n'
    #    msg += 'Actual type passed: ' +str(type(transition_system) )
    #    raise TypeError(msg)

    if not hasattr(buchi_automaton, 'alphabet'):
        msg = 'transition_system not transys.BuchiAutomaton.\n'
        msg += 'Actual type passed: ' + str(type(buchi_automaton))
        raise TypeError(msg)

    if not buchi_automaton.atomic_proposition_based:
        msg = """Buchi automaton not stored as Atomic Proposition-based.
                synchronous product with Finite Transition System
                is not well-defined."""
        raise Exception(msg)

    fts = transition_system
    ba = buchi_automaton

    prodts_name = fts.name + '*' + ba.name
    prodts = transys.FiniteTransitionSystem()
    prodts.name = prodts_name

    prodts.atomic_propositions.add_from(ba.states())
    prodts.sys_actions.add_from(fts.actions)

    # construct initial states of product automaton
    s0s = set(fts.states.initial)
    q0s = set(ba.states.initial)

    accepting_states_preimage = set()

    logger.debug('\n' + _hl + '\n' +
                 ' Product TS construction:' +
                 '\n' + _hl + '\n')

    if not s0s:
        msg = (
            'Transition System has no initial states !\n'
            '=> Empty product system.\n'
            'Did you forget to define initial states ?')
        warnings.warn(msg)

    for s0 in s0s:
        logger.debug('initial state:\t' + str(s0))

        for q0 in q0s:
            enabled_ba_trans = find_ba_succ(q0, s0, fts, ba)

            # q0 blocked ?
            if not enabled_ba_trans:
                continue

            # which q next ?     (note: curq0 = q0)
            for (curq0, q, sublabels) in enabled_ba_trans:
                new_sq0 = (s0, q)
                prodts.states.add(new_sq0)
                prodts.states.initial.add(new_sq0)
                prodts.states[new_sq0]['ap'] = {q}

                # accepting state ?
                if q in ba.states.accepting:
                    accepting_states_preimage.add(new_sq0)

    # start visiting reachable in DFS or BFS way
    # (doesn't matter if we are going to store the result)
    queue = set(prodts.states.initial)
    visited = set()
    while queue:
        sq = queue.pop()
        visited.add(sq)
        (s, q) = sq

        logger.debug('Current product state:\t' + str(sq))

        # get next states
        next_ss = fts.states.post(s)
        next_sqs = set()
        for next_s in next_ss:
            enabled_ba_trans = find_ba_succ(q, next_s, fts, ba)

            if not enabled_ba_trans:
                continue

            (new_sqs, new_accepting) = find_prod_succ(
                sq, next_s, enabled_ba_trans,
                prodts, ba, fts)

            next_sqs.update(new_sqs)
            accepting_states_preimage.update(new_accepting)

        logger.debug('next product states: ' + str(next_sqs))
        # discard visited & push them to queue
        new_sqs = {x for x in next_sqs if x not in visited}
        logger.debug('new unvisited product states: ' + str(new_sqs))
        queue.update(new_sqs)

    return (prodts, accepting_states_preimage)


def find_ba_succ(prev_q, next_s, fts, ba):
    q = prev_q

    logger.debug('Next state:\t' + str(next_s))
    try:
        ap = fts.node[next_s]['ap']
    except:
        raise Exception(
            'No AP label for FTS state: ' + str(next_s) +
            '\n Did you forget labeing it ?')

    Sigma_dict = {'letter': ap}
    logger.debug("Next state's label:\t" + str(ap))

    enabled_ba_trans = ba.transitions.find(
        [q], with_attr_dict=Sigma_dict)
    enabled_ba_trans += ba.transitions.find(
        [q], letter={True})
    logger.debug('Enabled BA transitions:\n\t' +
                 str(enabled_ba_trans))

    if not enabled_ba_trans:
        logger.debug('No enabled BA transitions at: ' + str(q))

    logger.debug('---\n')

    return enabled_ba_trans


def find_prod_succ(prev_sq, next_s, enabled_ba_trans, product, ba, fts):
    (s, q) = prev_sq

    new_accepting = set()
    next_sqs = set()
    for (curq, next_q, sublabels) in enabled_ba_trans:
        assert(curq == q)

        new_sq = (next_s, next_q)

        if new_sq not in product:
            next_sqs.add(new_sq)
            product.states.add(new_sq)

            logger.debug('Adding state:\t' + str(new_sq))

        if hasattr(product, 'actions'):
            product.states[new_sq]['ap'] = {next_q}

        # accepting state ?
        if next_q in ba.states.accepting:
            new_accepting.add(new_sq)
            logger.debug(str(new_sq) +
                         ' contains an accepting state.')

        logger.debug('Adding transitions:\t' +
                     str(prev_sq) + '--->' + str(new_sq))

        # is fts transition labeled with an action ?
        enabled_ts_trans = fts.transitions.find(
            [s], to_states=[next_s],
            with_attr_dict=None)
        for (from_s, to_s, sublabel_values) in enabled_ts_trans:
            assert(from_s == s)
            assert(to_s == next_s)

            logger.debug('Sublabel value:\n\t' +
                         str(sublabel_values))

            # labeled transition ?
            if hasattr(product, 'alphabet'):
                product.transitions.add(
                    prev_sq, new_sq,
                    letter=fts.states[to_s]['ap'])
            elif hasattr(product, 'actions'):
                if not sublabel_values:
                    product.transitions.add(prev_sq, new_sq)
                else:
                    product.transitions.add(
                        prev_sq, new_sq,
                        actions=sublabel_values['actions'])
    return (next_sqs, new_accepting)


def ba_ts_sync_prod(buchi_automaton, transition_system):
    """Construct Buchi Automaton equal to synchronous product TS x NBA.

    See Also
    ========
    L{ts_ba_sync_prod}, L{sync_prod}

    @return: C{prod_ba}, the product L{BuchiAutomaton}.
    """
    logger.debug('\n' + _hl + '\n'
                 'Product: BA * TS' +
                 '\n' + _hl + '\n')

    (prod_ts, persistent) = ts_ba_sync_prod(
        transition_system, buchi_automaton)

    prod_name = buchi_automaton.name + '*' + transition_system.name

    prod_ba = automata.BuchiAutomaton()
    prod_ba.name = prod_name

    # copy S, S0, from prod_TS-> prod_BA
    prod_ba.states.add_from(prod_ts.states())
    prod_ba.states.initial |= set(prod_ts.states.initial)

    # accepting states = persistent set
    prod_ba.states.accepting |= persistent

    # copy edges, translating transitions,
    # i.e., changing transition labels
    if not buchi_automaton.atomic_proposition_based:
        msg = (
            'Buchi Automaton must be Atomic Proposition-based,'
            ' otherwise the synchronous product is not well-defined.')
        raise Exception(msg)

    # direct access, not the inefficient
    #   prod_ba.alphabet.add_from(buchi_automaton.alphabet() ),
    # which would generate a combinatorially large alphabet
    prod_ba.alphabet.math_set |= buchi_automaton.alphabet.math_set

    for (from_state, to_state) in prod_ts.transitions():
        # prject prod_TS state to TS state
        ts_to_state = to_state[0]
        msg = (
            'prod_TS: to_state =\n\t' + str(to_state) + '\n'
            'TS: ts_to_state =\n\t' + str(ts_to_state))
        logger.debug(msg)

        state_label_pairs = transition_system.states.find(ts_to_state)
        (ts_to_state_, transition_label_dict) = state_label_pairs[0]
        transition_label_value = transition_label_dict['ap']
        prod_ba.transitions.add(
            from_state, to_state, letter=transition_label_value)
    return prod_ba

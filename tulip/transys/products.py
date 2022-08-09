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
import collections.abc as _abc
import logging
import warnings

import tulip.transys.automata as automata
import tulip.transys.labeled_graphs as _graphs
import tulip.transys.transys as transys
import tulip._utils as _utl


__all__ = [
    'OnTheFlyProductAutomaton']


_logger = logging.getLogger(__name__)
_hl = 40 * '-'
FTS = transys.FTS


class OnTheFlyProductAutomaton(automata.BuchiAutomaton):
    """Dynamically extends itself by adding successors.

    Note that performs on-the-fly BA * TS.
    The given TS can be explicit or on-the-fly,
    depending on what you pass to `__init__`.

    The state space serves as the set of "visited" states.

    Note that this is equivalent to adding successors
    to both the queue and the set of visited states
    at the end of each iteration during a search,
    instead of adding each successor to the visited states
    when it is poped from the queue.
    """

    def __init__(
            self,
            ba:
                automata.BuchiAutomaton,
            ts:
                FTS):
        self.ba = ba
        self.ts = ts
        super().__init__()
        self.atomic_propositions.update(
            ts.atomic_propositions)
        self._add_initial()

    def _add_initial(self):
        ts = self.ts
        ba = self.ba
        s0s = set(ts.states.initial)
        q0s = set(ba.states.initial)
        _logger.debug(
            f'\n{_hl}\n'
            ' Product BA construction:\n'
            f'{_hl}\n')
        if not s0s:
            warnings.warn(
                'Transition System has no initial states !\n'
                '=> Empty product system.\n'
                'Did you forget to define initial states ?')
        for s0 in s0s:
            _logger.debug(f'initial state:\t{s0}')
            for q0 in q0s:
                enabled_ba_trans = find_ba_succ(q0, s0, ts, ba)
                # q0 blocked ?
                if not enabled_ba_trans:
                    continue
                # which q next ?     (note: curq0 = q0)
                for curq0, q, sublabels in enabled_ba_trans:
                    new_sq0 = (s0, q)
                    self.states.add(new_sq0)
                    self.states.initial.add(new_sq0)
                    # accepting state ?
                    if q in ba.accepting:
                        self.accepting.add(new_sq0)

    def add_successors(
            self,
            s,
            q
            ) -> set:
        """Add the successors of (s, q) to the state space.

        @param s:
            TS state
        @param q:
            BA state
        @return:
            those successors that are new states
        """
        sq = (s, q)
        ts = self.ts
        ba = self.ba
        _logger.debug(
            'Creating successors from'
            f' product state:\t{sq}')
        # get next states
        next_ss = ts.states.post([s])
        next_sqs = set()
        for next_s in next_ss:
            enabled_ba_trans = find_ba_succ(q, next_s, ts, ba)
            if not enabled_ba_trans:
                continue
            new_sqs, new_accepting = find_prod_succ(
                sq, next_s, enabled_ba_trans,
                self, ba, ts)
            next_sqs.update(new_sqs)
            self.accepting.update(
                new_accepting)
        # new_sqs = {x for x in next_sqs if x not in self}
        _logger.debug(f'next product states: {next_sqs}')
        _logger.debug(f'new unvisited product states: {new_sqs}')
        return new_sqs

    def add_all_states(self):
        """Iterate `add_successors` until all states are added.

        In other words until the state space
        reaches a fixed point.
        """
        Q = set(self.states)
        while Q:
            Qnew = set()
            for sq in Q:
                s, q = sq
                new = self.add_successors(s, q)
                Qnew.update(new)
            Q = Qnew


def ts_ba_sync_prod(
        transition_system:
            FTS,
        buchi_automaton:
            automata.BuchiAutomaton
        ) -> tuple[
            transys.FiniteTransitionSystem,
            set]:
    r"""Construct transition system for the synchronous product TS * BA.

    Def. 4.62, p.200 [BK08](
        https://tulip-control.sourceforge.io/doc/bibliography.html#bk08)

    Erratum
    =======
    note the erratum: P_{pers}(A) is ^_{q\in F} !q, verified from:
    <http://www-i2.informatik.rwth-aachen.de/~katoen/errata.pdf>

    See Also
    ========
    `ba_ts_sync_prod`, `sync_prod`

    @return:
        `(product_ts, persistent_states)`, where:
        - `product_ts` is the synchronous product TS * BA
        - `persistent_states` are those in TS * BA which
            project on accepting states of BA.
    """
    # if not hasattr(transition_system, FiniteTransitionSystem):
    #    msg = 'transition_system not transys.FiniteTransitionSystem.\n'
    #    msg += f'Actual type passed: {type(transition_system)}')
    #    raise TypeError(msg)
    if not hasattr(buchi_automaton, 'alphabet'):
        raise TypeError(
            'transition_system not transys.BuchiAutomaton.\n'
            f'Actual type passed: {type(buchi_automaton)}')
    if not buchi_automaton.atomic_proposition_based:
        raise ValueError(
            'Buchi automaton not stored as '
            'Atomic Proposition-based. '
            'Synchronous product with '
            'Finite Transition System is '
            'not well-defined.')
    fts = transition_system
    ba = buchi_automaton
    prodts_name = f'{fts.name}*{ba.name}'
    prodts = transys.FiniteTransitionSystem()
    prodts.name = prodts_name
    prodts.atomic_propositions.add_from(ba.states())
    prodts.sys_actions.add_from(fts.actions)
    # construct initial states of product automaton
    s0s = set(fts.states.initial)
    q0s = set(ba.states.initial)
    accepting_states_preimage = set()
    _logger.debug(
        f'\n{_hl}\n'
        ' Product TS construction:'
        f'\n{_hl}\n')
    if not s0s:
        warnings.warn(
            'Transition System has no initial states !\n'
            '=> Empty product system.\n'
            'Did you forget to define initial states ?')
    for s0 in s0s:
        _logger.debug(f'initial state:\t{s0}')
        for q0 in q0s:
            enabled_ba_trans = find_ba_succ(q0, s0, fts, ba)
            # q0 blocked ?
            if not enabled_ba_trans:
                continue
            # which q next ?     (note: curq0 = q0)
            for curq0, q, sublabels in enabled_ba_trans:
                new_sq0 = (s0, q)
                prodts.states.add(new_sq0)
                prodts.states.initial.add(new_sq0)
                prodts.states[new_sq0]['ap'] = {q}
                # accepting state ?
                if q in ba.accepting:
                    accepting_states_preimage.add(new_sq0)
    # start visiting reachable in DFS or BFS way
    # (doesn't matter if we are going to store the result)
    queue = set(prodts.states.initial)
    visited = set()
    while queue:
        sq = queue.pop()
        visited.add(sq)
        s, q = sq
        _logger.debug(f'Current product state:\t{sq}')
        # get next states
        next_ss = fts.states.post([s])
        next_sqs = set()
        for next_s in next_ss:
            enabled_ba_trans = find_ba_succ(q, next_s, fts, ba)
            if not enabled_ba_trans:
                continue
            new_sqs, new_accepting = find_prod_succ(
                sq, next_s, enabled_ba_trans,
                prodts, ba, fts)
            next_sqs.update(new_sqs)
            accepting_states_preimage.update(new_accepting)
        _logger.debug(
            f'next product states: {next_sqs}')
        # discard visited & push them to queue
        new_sqs = {x for x in next_sqs if x not in visited}
        _logger.debug(
            f'new unvisited product states: {new_sqs}')
        queue.update(new_sqs)
    return (
        prodts,
        accepting_states_preimage)


def find_ba_succ(
        prev_q,
        next_s,
        fts:
            FTS,
        ba:
            automata.BuchiAutomaton
        ) -> list[tuple]:
    q = prev_q
    _logger.debug(f'Next state:\t{next_s}')
    try:
        ap = fts.nodes[next_s]['ap']
    except:
        raise Exception(
            f'No AP label for FTS state: {next_s}'
            '\n Did you forget labeing it ?')
    Sigma_dict = {'letter': ap}
    _logger.debug(f"Next state's label:\t{ap}")
    enabled_ba_trans = ba.transitions.find(
        [q], with_attr_dict=Sigma_dict)
    enabled_ba_trans += ba.transitions.find(
        [q], letter={True})
    _logger.debug(
        'Enabled BA transitions:\n\t'
        f'{enabled_ba_trans}')
    if not enabled_ba_trans:
        _logger.debug(
            f'No enabled BA transitions at: {q}')
    _logger.debug('---\n')
    return enabled_ba_trans


def find_prod_succ(
        prev_sq:
            _utl.n_tuple(2),
        next_s,
        enabled_ba_trans:
            _abc.Iterable[
                _utl.n_tuple(3)],
        product:
            _graphs.LabeledDiGraph,
        ba:
            automata.BuchiAutomaton,
        fts:
            FTS
        ) -> tuple[
            set,
            set]:
    s, q = prev_sq
    new_accepting = set()
    next_sqs = set()
    for curq, next_q, sublabels in enabled_ba_trans:
        if curq != q:
            raise AssertionError(
                curq, q)
        new_sq = (next_s, next_q)
        if new_sq not in product:
            next_sqs.add(new_sq)
            product.states.add(new_sq)
            _logger.debug(
                f'Adding state:\t{new_sq}')
        if hasattr(product, 'actions'):
            product.states[new_sq]['ap'] = {next_q}
        # accepting state ?
        if next_q in ba.accepting:
            new_accepting.add(new_sq)
            _logger.debug(
                f'{new_sq} contains an accepting state.')
        _logger.debug(
            'Adding transitions:\t'
            f'{prev_sq} ---> {new_sq}')
        # is fts transition labeled with an action ?
        enabled_ts_trans = fts.transitions.find(
            [s], to_states=[next_s],
            with_attr_dict=None)
        for from_s, to_s, sublabel_values in enabled_ts_trans:
            if from_s != s:
                raise AssertionError(
                    from_s, s)
            if to_s != next_s:
                raise AssertionError(
                    to_s, next_s)
            _logger.debug('Sublabel value:\n\t' +
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


def ba_ts_sync_prod(
        buchi_automaton:
            automata.BuchiAutomaton,
        transition_system:
            FTS
        ) -> automata.BuchiAutomaton:
    """Construct Buchi Automaton equal to synchronous product TS x NBA.

    See Also
    ========
    `ts_ba_sync_prod`, `sync_prod`

    @return:
        `prod_ba`, the product `BuchiAutomaton`.
    """
    _logger.debug(
        f'\n{_hl}\n'
        'Product: BA * TS'
        f'\n{_hl}\n')
    prod_ts, persistent = ts_ba_sync_prod(
        transition_system, buchi_automaton)

    prod_name = (
        f'{buchi_automaton.name}*'
        f'{transition_system.name}')
    prod_ba = automata.BuchiAutomaton()
    prod_ba.name = prod_name
    # copy S, S0, from prod_TS-> prod_BA
    prod_ba.states.add_from(prod_ts.states())
    prod_ba.states.initial.update(
        prod_ts.states.initial)
    # accepting states = persistent set
    prod_ba.accepting.update(persistent)
    # copy edges, translating transitions,
    # i.e., changing transition labels
    if not buchi_automaton.atomic_proposition_based:
        raise ValueError(
            'Buchi Automaton must be Atomic Proposition-based,'
            ' otherwise the synchronous product is not well-defined.')
    # direct access, not the inefficient
    #   prod_ba.alphabet.add_from(buchi_automaton.alphabet() ),
    # which would generate a combinatorially large alphabet
    prod_ba.alphabet.math_set.update(
        buchi_automaton.alphabet.math_set)
    for from_state, to_state in prod_ts.transitions():
        # prject prod_TS state to TS state
        ts_to_state = to_state[0]
        _logger.debug(
            f'prod_TS: to_state =\n\t{to_state}\n'
            f'TS: ts_to_state =\n\t{ts_to_state}')
        state_label_pairs = transition_system.states.find(ts_to_state)
        ts_to_state_, transition_label_dict = state_label_pairs[0]
        transition_label_value = transition_label_dict['ap']
        prod_ba.transitions.add(
            from_state, to_state,
            letter=transition_label_value)
    return prod_ba

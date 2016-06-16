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
"""Automata Module"""
from __future__ import absolute_import
import logging
import copy
from collections import Iterable
from pprint import pformat
from tulip.transys.labeled_graphs import (
    LabeledDiGraph, str2singleton, prepend_with)
from tulip.transys.mathset import SubSet, PowerSet
from tulip.transys.transys import GameGraph


logger = logging.getLogger(__name__)
_hl = 40 * '-'


class FiniteStateAutomaton(LabeledDiGraph):
    """Set of sequences described with a graph and a condition.

    It has:
        - states
        - states.initial
        - states.accepting (types have names, and classes)
        - alphabet = set of symbols that label edges.


    Note
    ====
    If all paths in the graph belong to the set you
    want to describe, then just use L{FiniteTransitionSystem}.

    To describe an input-output function (which is a set too),
    it is more convenient to use L{FiniteStateMachine}.


    See Also
    ========
    L{BA}, L{RabinAutomaton}.
    """

    def __init__(
            self, deterministic=False,
            accepting_states_type=None,
            atomic_proposition_based=True,
            symbolic=False,
    ):
        """Initialize FiniteStateAutomaton.

        Additional keyword arguments are passed to L{LabeledDiGraph.__init__}.

        @param atomic_proposition_based: If False, then the alphabet
            is represented by a set.  If True, then the alphabet is
            represented by a powerset 2^AP.
        """
        self.atomic_proposition_based = atomic_proposition_based
        self.symbolic = symbolic

        # edge labeling
        if symbolic:
            alphabet = None  # no checks
        else:
            if atomic_proposition_based:
                alphabet = PowerSet([])
                self.atomic_propositions = alphabet.math_set
            else:
                alphabet = set()
        self.alphabet = alphabet

        edge_label_types = [
            {'name': 'letter',
             'values': alphabet,
             'setter': True}]
        super(FiniteStateAutomaton, self).__init__(
            edge_label_types=edge_label_types)
        # accepting states
        if accepting_states_type is None:
            self._accepting = SubSet(self.states)
            self._accepting_type = SubSet
        else:
            self._accepting = accepting_states_type(self)
            self._accepting_type = accepting_states_type
        self.states.accepting = self._accepting
        # used before label value
        self._transition_dot_label_format = {'letter': '',
                                             'type?label': '',
                                             'separator': r'\\n'}
        self._transition_dot_mask = dict()

        self.dot_node_shape = {'normal': 'circle',
                               'accepting': 'doublecircle'}
        self.default_export_fname = 'fsa'
        self.automaton_type = 'Finite State Automaton'

    @property
    def accepting(self):
        return self._accepting

    def __str__(self):
        s = (
            _hl + '\n' + self.automaton_type + ': ' +
            self.name + '\n' + _hl + '\n' +
            'States:\n' +
            pformat(self.states(data=False), indent=3) + 2 * '\n' +
            'Initial States:\n' +
            pformat(self.states.initial, indent=3) + 2 * '\n' +
            'Accepting States:\n' +
            pformat(self.states.accepting, indent=3) + 2 * '\n')
        if self.atomic_proposition_based:
            s += 'Input Alphabet Letters (\in 2^AP):\n\t'
        else:
            if hasattr(self, 'alphabet'):
                s += ('Input Alphabet Letters:\n\t' +
                      str(self.alphabet) + 2 * '\n')
        s += (
            'Transitions & labeling w/ Input Letters:\n' +
            pformat(self.transitions(data=True), indent=3) +
            '\n' + _hl + '\n')
        return s

    def remove_node(self, node):
        """Remove state (also referred to as "node").

        More than a wrapper since the state is also removed from the
        accepting set if present.
        """
        # intercept to remove also from accepting states
        self.accepting.remove(node)
        super(FiniteStateAutomaton, self).remove_node(node)


class FiniteWordAutomaton(FiniteStateAutomaton):
    """Finite-word finite-state automaton.

    By default non-deterministic (NFA).
    To enforce determinism (DFA):

    >>> a = FiniteWordAutomaton(deterministic=True)
    """
    def __init__(self, deterministic=False,
                 atomic_proposition_based=True):
        super(FiniteWordAutomaton, self).__init__(
            deterministic=deterministic,
            atomic_proposition_based=atomic_proposition_based)
        self.automaton_type = 'Finite-Word Automaton'


def dfa2nfa(dfa):
    """Copy DFA to an NFA, so remove determinism restriction."""
    nfa = copy.deepcopy(dfa)
    nfa.transitions._deterministic = False
    nfa.automaton_type = 'Non-Deterministic Finite Automaton'
    return nfa


class OmegaAutomaton(FiniteStateAutomaton):
    def __init__(self, *args, **kwargs):
        super(OmegaAutomaton, self).__init__(*args, **kwargs)


class BuchiAutomaton(OmegaAutomaton):
    def __init__(
            self, deterministic=False,
            atomic_proposition_based=True,
            symbolic=False
    ):
        super(BuchiAutomaton, self).__init__(
            deterministic=deterministic,
            atomic_proposition_based=atomic_proposition_based,
            symbolic=symbolic)
        self.automaton_type = 'Buchi Automaton'


class BA(BuchiAutomaton):
    """Alias to L{BuchiAutomaton}."""

    def __init__(self, **args):
        super(BA, self).__init__(**args)


def tuple2ba(S, S0, Sa, Sigma_or_AP, trans, name='ba', prepend_str=None,
             atomic_proposition_based=True):
    """Create a Buchi Automaton from a tuple of fields.

    defines Buchi Automaton by a tuple (S, S0, Sa, \\Sigma, trans)
    (maybe replacing \\Sigma by AP since it is an AP-based BA ?)

    See Also
    ========
    L{tuple2fts}

    @param S: set of states
    @param S0: set of initial states, must be \\subset S
    @param Sa: set of accepting states
    @param Sigma_or_AP: Sigma = alphabet
    @param trans: transition relation, represented by list of triples::
            [(from_state, to_state, guard), ...]
    where guard \\in \\Sigma.

    @param name: used for file export
    @type name: str

    @rtype: L{BuchiAutomaton}
    """
    # args
    if not isinstance(S, Iterable):
        raise TypeError('States S must be iterable, even for single state.')
    if not isinstance(S0, Iterable) or isinstance(S0, str):
        S0 = [S0]
    if not isinstance(Sa, Iterable) or isinstance(Sa, str):
        Sa = [Sa]
    # comprehensive names
    states = S
    initial_states = S0
    accepting_states = Sa
    alphabet_or_ap = Sigma_or_AP
    transitions = trans
    # prepending states with given str
    if prepend_str:
        logger.debug('Given string:\n\t' + str(prepend_str) + '\n' +
                     'will be prepended to all states.')
    states = prepend_with(states, prepend_str)
    initial_states = prepend_with(initial_states, prepend_str)
    accepting_states = prepend_with(accepting_states, prepend_str)

    ba = BuchiAutomaton(atomic_proposition_based=atomic_proposition_based)
    ba.name = name

    ba.states.add_from(states)
    ba.states.initial |= initial_states
    ba.states.accepting |= accepting_states

    if atomic_proposition_based:
        ba.alphabet.math_set |= alphabet_or_ap
    else:
        ba.alphabet.add(alphabet_or_ap)
    for transition in transitions:
        (from_state, to_state, guard) = transition
        [from_state, to_state] = prepend_with([from_state, to_state],
                                              prepend_str)
        # convention
        if atomic_proposition_based:
            if guard is None:
                guard = set()
            guard = str2singleton(guard)
        ba.transitions.add(from_state, to_state, letter=guard)
    return ba


class RabinPairs(object):
    """Acceptance pairs for Rabin automaton.

    Each pair defines an acceptance condition.
    A pair (L, U) comprises of:
        - a set L of "good" states
        - a set U of "bad" states
    L,U must each be a subset of States.

    A run: (q0, q1, ...) is accepted if for at least one Rabin Pair,
    it in intersects L an inf number of times, but U only finitely.

    Internally a list of 2-tuples of SubSet objects is maintained::
        [(L1, U1), (L2, U2), ...]
    where: Li, Ui, are SubSet objects, with superset
    the Rabin automaton's States.

    Caution
    =======
    Here and in ltl2dstar documentation L denotes a "good" set.
    U{[BK08] <http://tulip-control.sourceforge.net/doc/bibliography.html#bk08>}
    denote the a "bad" set with L.  To avoid ambiguity, attributes:
    .good, .bad were used here.

    Example
    =======
    >>> dra = RabinAutomaton()
    >>> dra.states.add_from([1, 2, 3] )
    >>> dra.states.accepting.add([1], [2] )
    >>> dra.states.accepting

    >>> dra.states.accepting.good(1)

    >>> dra.states.accepting.bad(1)

    See Also
    ========
      - L{RabinAutomaton}
      - Def. 10.53, p.801, U{[BK08]
        <http://tulip-control.sourceforge.net/doc/bibliography.html#bk08>}
      - U{ltl2dstar<http://ltl2dstar.de/>} documentation
    """

    def __init__(self, automaton_states):
        self._states = automaton_states
        self._pairs = []

    def __str__(self):
        s = 'L = Good states, U = Bad states\n' + 30 * '-' + '\n'
        for index, (good, bad) in enumerate(self._pairs):
            s += (
                'Pair: ' + str(index) + ', L = ' + str(good) +
                ', U = ' + str(bad) + '\n')
        return s

    def __getitem__(self, index):
        return self._pairs[index]

    def __iter__(self):
        return iter(self._pairs)

    def __call__(self):
        """Get list of 2-tuples (L, U) of good-bad sets of states."""
        return list(self._pairs)

    def add(self, good_states, bad_states):
        """Add new acceptance pair (L, U).

        See Also
        ========
        remove, add_states, good, bad

        @param good_states: set L of good states for this pair
        @type good_states: container of valid states

        @param bad_states: set U of bad states for this pair
        @type bad_states: container of valid states
        """
        good_set = SubSet(self._states)
        good_set |= good_states
        bad_set = SubSet(self._states)
        bad_set |= bad_states
        self._pairs.append((good_set, bad_set))

    def remove(self, good_states, bad_states):
        """Delete pair (L, U) of good-bad sets of states.

        Note
        ====
        Removing a pair which is not last changes
        the indices of all other pairs, because internally
        a list is used.

        The sets L,U themselves (good-bad) are required
        for the deletion, instead of an index, to prevent
        acceidental deletion of an unintended pair.

        Get the intended pair using __getitem__ first
        (or in any other way) and them call remove.
        If the pair is corrent, then the removal will
        be successful.

        See Also
        ========
        add

        @param good_states: set of good states of this pair
        @type good_states: iterable container
        """
        good_set = SubSet(self._states)
        good_set |= good_states
        bad_set = SubSet(self._states)
        bad_set |= bad_states
        self._pairs.remove((good_set, bad_set))

    def add_states(self, pair_index, good_states, bad_states):
        try:
            self._pairs[pair_index][0].add_from(good_states)
            self._pairs[pair_index][1].add_from(bad_states)
        except IndexError:
            raise Exception("A pair with pair_index doesn't exist.\n" +
                            'Create a new one by callign .add.')

    def good(self, index):
        """Return set L of "good" states for this pair.

        @param index: number of Rabin acceptance pair
        @type index: int <= current total number of pairs
        """
        return self._pairs[index][0]

    def bad(self, index):
        """Return set U of "bad" states for this pair.

        @param index: number of Rabin acceptance pair
        @type index: int <= current total number of pairs
        """
        return self._pairs[index][1]

    def has_superset(self, superset):
        """Return true if the given argument is the superset."""
        return superset is self._states


class RabinAutomaton(OmegaAutomaton):
    """Rabin automaton.

    See Also
    ========
    L{DRA}, L{BuchiAutomaton}
    """

    def __init__(self, deterministic=False,
                 atomic_proposition_based=False):
        super(RabinAutomaton, self).__init__(
            deterministic=deterministic,
            accepting_states_type=RabinPairs,
            atomic_proposition_based=atomic_proposition_based)
        self.automaton_type = 'Rabin Automaton'


class DRA(RabinAutomaton):
    """Deterministic Rabin Automaton.

    See Also
    ========
    L{RabinAutomaton}
    """

    def __init__(self, atomic_proposition_based=True):
        super(DRA, self).__init__(
            deterministic=True,
            atomic_proposition_based=atomic_proposition_based)
        self.automaton_type = 'Deterministic Rabin Automaton'


class ParityGame(GameGraph):
    """GameGraph equipped with coloring.

    Define as C{k} the highest color that
    occurs infinitely many times.

    If C{k} is even, then Player 0 wins.
    Otherwise Player 1 wins (C{k} is odd).
    So the winner is Player (k mod 2).

    To define the number of colors C{c}:

    >>> p = ParityGame(c=4)

    Note that the colors are: 0, 1, ..., c-1

    See also
    ========
    L{transys.GameGraph}
    """

    def __init__(self, c=2):
        node_label_types = [{
            'name': 'color',
            'values': range(c),
            'default': 0}]
        super(ParityGame, self).__init__(node_label_types, [])

    def __str__(self):
        s = (
            'Parity Game\n'
            '-----------\n'
            'n: node, p: player, c: color\n\n')
        for node, attr in self.states(data=True):
            s += 'nd = {node}, p = {player}, c = {color}\n'.format(
                npde=node, player=attr['player'], color=attr['color'])
        s += '\n{t}'.format(t=self.transitions)
        return s

    @property
    def max_color(self):
        max_c = -1
        # node = None
        for x in self:
            if self.node[x]['color'] > max_c:
                max_c = self.node[x]['color']
                # node = x
        return max_c

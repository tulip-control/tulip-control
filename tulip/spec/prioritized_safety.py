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

from itertools import product
from tulip.transys.automata import FiniteStateAutomaton as FA


class FAWithPriority(object):
    """A class for defining a rule, represented by a nondeterministic finite automaton,
    with priority"""

    def __init__(self, fa, priority, level):
        assert isinstance(fa, FA)
        assert isinstance(priority, int)
        assert isinstance(level, int)

        self._fa = fa
        self._priority = priority
        self._level = level

    def priority(self):
        """Get the priority of this rule"""
        return self._priority

    def automaton(self):
        """Get the automaton of this rule"""
        return self._fa

    def level(self):
        """Get the level of this rule"""
        return self._level


class PrioritizedSpecification(object):
    """A class for defining a prioritized safety specification"""

    def __init__(self):
        self._Psi = []
        self.atomic_propositions = []

    def __getitem__(self, key):
        assert key >= 0
        level = 0
        while level < len(self._Psi) and key >= len(self._Psi[level]):
            key -= len(self._Psi[level])
            level += 1
        if level < len(self._Psi) and key < len(self._Psi[level]):
            return self._Psi[level][key]
        raise IndexError("index out of range")

    def __iter__(self):
        self._iter_level = 0
        self._iter_index = 0
        return self

    def __next__(self):
        while self._iter_level < len(self._Psi) and self._iter_index >= len(
            self._Psi[self._iter_level]
        ):
            self._iter_index = 0
            self._iter_level += 1

        if self._iter_level >= len(self._Psi) or self._iter_index >= len(
            self._Psi[self._iter_level]
        ):
            raise StopIteration
        result = self._Psi[self._iter_level][self._iter_index]
        self._iter_index += 1
        return result

    def next(self):
        return self.__next__()

    def __len__(self):
        return sum([len(psi) for psi in self._Psi])

    def add_rule(self, fa, priority, level):
        """Add rule with automaton fa, priority and level to the specification

        @param fa: FiniteStateAutomaton representing the correctness of the rule
        @param priority: float or int representing the priority of the rule
        @param level: int representing the level of the rule in the hierarchy
        """
        assert isinstance(fa, FA)
        assert isinstance(priority, float) or isinstance(priority, int)
        assert isinstance(level, int)
        assert priority > 0
        assert level >= 0

        # Check the consistency of atomic propositions
        if len(self._Psi) == 0:
            self.atomic_propositions = fa.atomic_propositions
        else:
            assert self.atomic_propositions == fa.atomic_propositions

        # Add the rule
        rule = FAWithPriority(fa, priority, level)

        for l in range(len(self._Psi), level + 1):
            self._Psi.append([])

        self._Psi[level].append(rule)

    def get_rules_at(self, level):
        """Return the list of rules at the given level

        @rtype a list of FAWithPriority
        """
        if level >= len(self._Psi):
            return []
        return self._Psi[level]

    def get_rules(self):
        """Return the list of all the rules

        @rtype a list of FAWithPriority
        """
        return [phi for psi in self._Psi for phi in psi]

    def get_states(self):
        """Get the product of the states in all the finite automata
        """
        return product(*[phi.automaton().states for phi in self])

    def get_initial_states(self):
        """Get the product of the initial states of all the finite automata
        """
        return product(*[phi.automaton().states.initial for phi in self])

    def get_accepting_states(self):
        """Get the product of the accepting states of all the finite automata
        """
        return product(*[phi.automaton().states.accepting for phi in self])

    def get_num_levels(self):
        """Get the number of levels
        """
        return len(self._Psi)

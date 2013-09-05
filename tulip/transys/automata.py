# Copyright (c) 2013 by California Institute of Technology
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
"""
Transition System Module
"""
from collections import Iterable, OrderedDict
from pprint import pformat
import warnings

from labeled_graphs import LabeledStateDiGraph
from labeled_graphs import vprint, prepend_with, str2singleton
from mathset import MathSet, SubSet, PowerSet, is_subset, dprint
import transys

hl = 60 *'-'

class FiniteStateAutomaton(LabeledStateDiGraph):
    """Generic automaton.
    
    1) states
    2) initial states
    3) accepting states
    
    4) input alphabet = set of input letters
    5) transition labels
    
    4) acceptor mode (i.e., you can ask is_accepted ?, but nothing more)
    5) for generator mode, use a synthesis algorithm
       To avoid misconceptions, NO SIMULATION MODE provided.
    
    Synthesis interpretation
    ------------------------
    A synthesis algorithm is applying model checking (is accepted ?)
    to each possible input word,
    normally represented by a deterministic finite transition system,
    but during acceptance checking implicitly by graph searching
    (assuming the model is everything an only the automaton is the constraint)
    
    Dually, a model checking algorithm is iteratively attempting synthesis
    for each possible input word.
    However, since synthesis is fundamentally operating by trying out words
    and seeing whether they get accepted, it follows that
    an automaton is operable ONLY in acceptor mode.
    
    Generator construction
    ----------------------
    For a "generator", we would need to convert the automaton to a
    (nontrivial) transition system producing maximal initial paths,
    as discussed below.
    
    The above algorithms return a single accepted input word, if found.
    That word is represented as a (deterministic) Finite Transition System.
    If we want to represent more than one accepted word (e.g. the whole
    language), we would need to find all possible such FTS and
    construct their "union".
    
    Open Systems
    ------------
    Finally, note that a Finite State Machine or transducer is an OPEN SYSTEM.
    As such, it does not represent an input word of an automaton.
    It can be used for game synthesis, where inputs and outputs make sense.
    
    Alternatively, only after closing a system can it be used
    (in the sense of having a program graph which can be unfolded).
    
    input
    -----
    
    returns
    -------
    
    alphabet
    --------
    Add single letter to alphabet.
        
    If C{atomic_proposition_based=False},
    then the alphabet is represented by a set.
    
    If C{atomic_proposition_based=True},
    the the alphabet is represented by a powerset 2^AP
    and you manage the set of Atomic Propositions AP within the powerset.
    
    see also
    --------    
    LabeledStateDiGraph._dot_str
        
    """
    def __init__(
        self, accepting_states=[], input_alphabet_or_atomic_propositions=[],
        atomic_proposition_based=True, mutable=False, **args
    ):
        # edge labeling
        if atomic_proposition_based:
            self.atomic_proposition_based = True
            alphabet = PowerSet(input_alphabet_or_atomic_propositions)
        else:
            self.atomic_proposition_based = False
            alphabet = set(input_alphabet_or_atomic_propositions)
        
        self._transition_label_def = OrderedDict([
            ['in_alphabet', alphabet]
        ])
        self.alphabet = self._transition_label_def['in_alphabet']
        
        LabeledStateDiGraph.__init__(
            self, mutable=mutable,
            removed_state_callback=self._removed_state_callback, **args
        )
        
        if mutable:
            self.accepting_states = list()
        else:
            self.accepting_states = set()
        self.add_accepting_states_from(accepting_states)
        
        # used before label value
        self._transition_dot_label_format = {'in_alphabet':'',
                                                'type?label':'',
                                                'separator':'\\n'}
        
        self.dot_node_shape = {'normal':'circle', 'accepting':'doublecircle'}
        self.default_export_fname = 'fsa'
        
    def __repr__(self):
        s = 'States:\n'
        s += pformat(self.states(data=False), indent=3) +2*'\n'
        s += 'Initial States:\n'
        s += pformat(self.states.initial, indent=3) +2*'\n'
        s += 'Input Alphabet Letters:\n\t'
        s += str(self.alphabet) +2*'\n'
        s += 'Transitions & labeling w/ Input Letters:\n'
        s += pformat(self.transitions(labeled=True), indent=3) +2*'\n'
        s += 'Accepting states differently defined for specific types'
        s += 'of automata, e.g. BA vs DRA.' +'\n' +hl +'\n'
        
        return s
    
    def __str__(self):
        return self.__repr__()
    
    def _removed_state_callback(self, rm_state):
        self.remove_accepting_state(rm_state)
    
    def _is_accepting(self, state):
        return is_subset([state], self.accepting_states)

    #TODO rm accepting states, pull them out as separate object,
    # because they differ from flavor to flavor of automaton
    # accepting states
    def add_accepting_state(self, new_accepting_state):
        if not new_accepting_state in self.states():
            msg = 'Given accepting state:\n\t' +str(new_accepting_state)
            msg += '\n\\notin States:\n\t' +str(self.states() )
            raise Exception(msg)
        
        # already accepting ?
        if self._is_accepting(new_accepting_state):
            warnings.warn('Attempting to add existing accepting state.\n')
            return
        
        # mutable states ?
        if self.states.mutants == None:
            self.accepting_states.add(new_accepting_state)
        else:
            self.accepting_states.append(new_accepting_state)

    def add_accepting_states_from(self, new_accepting_states):
        if not is_subset(new_accepting_states, self.states() ):
            raise Exception('Given Accepting States \\notsubset States.')
        
        # mutable states ?
        if self.states.mutants == None:
            self.accepting_states |= set(new_accepting_states)
        else:
            for new_accepting_state in new_accepting_states:
                self.add_accepting_state(new_accepting_state)
    
    def number_of_accepting_states(self):
        return len(self.accepting_states)
    
    def remove_accepting_state(self, rm_accepting_state):
        self.accepting_states.remove(rm_accepting_state)
    
    def remove_accepting_states_from(self, rm_accepting_states):
        if self.states.mutants == None:
            self.accepting_states = \
                self.accepting_states.difference(rm_accepting_states)
        else:
            for rm_accepting_state in rm_accepting_states:
                self.remove_accepting_state(rm_accepting_state)

    # checks
    def is_deterministic(self):
        """overloaded method.
        """
        raise NotImplementedError
        
    def is_blocking(self):
        """overloaded method.
        """
        raise NotImplementedError
    
    def is_accepted(self, input_word):
        """Check if input word is accepted.
        """
        sim = self.simulate(input_word)
        
        inf_states = set(sim.run.get_suffix() )
        
        if bool(inf_states & self.accepting_states):
            accept = True
        else:
            accept = False
        
        return accept
        
    def simulate(self, initial_state, input_word):
        """Returns an Omega Automaton Simulation, with prefix, suffix.
        """
        
        # should be implemented properly with nested depth-first search,
        # becaus of possible branching due to non-determinism
        
        for letter in input_word:
            dprint(letter)
            
            # blocked
        
        #return FSASim()

    # operations on two automata
    def add_subautomaton(self):
        raise NotImplementedError

class StarAutomaton(FiniteStateAutomaton):
    """Finite-word finite-state automaton.
    """

class DeterninisticFiniteAutomaton(StarAutomaton):
    """Deterministic finite-word finite-state Automaton.
    """

    # check each initial state added
    # check each transition added
    
class DFA(DeterninisticFiniteAutomaton):
    """Alias for deterministic finite-word finite-state automaton.
    """

class NonDeterministicFiniteAutomaton(StarAutomaton):
    """"Non-deterministic finite-word finite-state automaton.
    """
    
    # note:
    #   is_deterministic still makes sense
    
class NFA(NonDeterministicFiniteAutomaton):
    """Alias for non-deterministic finite-word finite-state automaton.
    """

def nfa2dfa():
    """Determinize NFA."""
    raise NotImplementedError
    
def dfa2nfa():
    """Relax state addition constraint of determinism."""
    raise NotImplementedError

class OmegaAutomaton(FiniteStateAutomaton):
    def __init__(self, **args):
        FiniteStateAutomaton.__init__(self, **args)

class BuchiAutomaton(OmegaAutomaton):
    def __init__(self, **args):
        OmegaAutomaton.__init__(self, **args)
    
    def __repr__(self):
        s = hl +'\nBuchi Automaton: ' +self.name +'\n' +hl +'\n'
        s += 'States:\n'
        s += pformat(self.states(data=False), indent=3) +2*'\n'
        s += 'Initial States:\n'
        s += pformat(self.states.initial, indent=3) +2*'\n'
        s += 'Accepting States:\n'
        s += pformat(self.accepting_states, indent=3) +2*'\n'
        
        if self.atomic_proposition_based:
            s += 'Input Alphabet Letters (\in 2^AP):\n\t'
        else:
            s += 'Input Alphabet Letters:\n\t'
        s += str(self.alphabet) +2*'\n'
        s += 'Transitions & labeling w/ Input Letters:\n'
        s += pformat(self.transitions(labeled=True), indent=3) +'\n' +hl +'\n'
        
        return s
    
    def __str__(self):
        return self.__repr__()
    
    def __add__(self, other):
        """Union of two automata, with equal states identified."""
        raise NotImplementedError
    
    def __mul__(self, ts_or_ba):
        return self.sync_prod(ts_or_ba)
    
    def __or__(self, ba):
        return self.async_prod(ba)
        
    def _ba_ba_sync_prod(self, ba2):
        #ba1 = self
        
        raise NotImplementedError
        #TODO BA x BA sync prod algorithm

    def sync_prod(self, ts_or_ba):
        """Synchronous product between (BA, TS), or (BA1, BA2).
        
        The result is always a Buchi Automaton.
        
        If C{ts_or_ba} is a Finite Transition System, then the result is the
        Buchi Automaton equal to the synchronous product of this Buchi Automaton
        with the given Transition System. Note that the accepting states of the
        product system are the preimage under projection of the set of accepting
        states of this Buchi Automaton.
        
        If C{ts_or_ba} is a Buchi Automaton, then the result is the Buchi Automaton
        equal to the synchronous product between this Buchi Automaton and the
        given Buchi Automaton. The set of accepting states of the resulting
        Buchi Automaton is equal to the intersection of the preimages under
        projection of the sets of accepting states of the individual Buchi Automata.
        
        This definition of accepting set extends Def.4.8, p.156 [Baier] to NBA.
        
        caution
        -------
        This method includes semantics for true\in\Sigma (p.916, [Baier]),
        so there is a slight overlap with logic grammar.
        In other words, this module is not completely isolated from logics.
        
        see also
        --------        
        ts_ba_sync_prod.
        """
        
        if isinstance(ts_or_ba, BuchiAutomaton):
            return self._ba_ba_sync_prod(ts_or_ba)
        elif isinstance(ts_or_ba, transys.FiniteTransitionSystem):
            ts = ts_or_ba
            return _ba_ts_sync_prod(self, ts)
        else:
            raise Exception('ts_or_ba should be an FTS or a BA.\n'+
                            'Got type: ' +str(ts_or_ba) )
    
    def acceptance_condition(self, prefix, suffix):
        """Check if given infinite word over alphabet \Sigma is accepted."""
    
    def determinize(self):
        raise NotImplementedError
    
    def complement(self):
        raise NotImplementedError

class BA(BuchiAutomaton):
    """Alias to BuchiAutomaton.
    """
    def __init__(self, **args):
        BuchiAutomaton.__init__(self, **args)

def tuple2ba(S, S0, Sa, Sigma_or_AP, trans, name='ba', prepend_str=None,
             atomic_proposition_based=True, verbose=False):
    """Create a Buchi Automaton from a tuple of fields.
    
    see also
    ========
    L{tuple2fts}

    @type ba_tuple: tuple
    @param ba_tuple: defines Buchi Automaton by a tuple (Q, Q_0, Q_F,
        \\Sigma, trans) (maybe replacing \\Sigma by AP since it is an
        AP-based BA ?)  where:

            - Q = set of states
            - Q_0 = set of initial states, must be \\subset S
            - Q_a = set of accepting states
            - \\Sigma = alphabet
            - trans = transition relation, represented by list of triples:
              [(from_state, to_state, guard), ...]
              where guard \\in \\Sigma.

    @param name: used for file export
    @type name: str
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
        vprint('Given string:\n\t' +str(prepend_str) +'\n' +
               'will be prepended to all states.', verbose)
    states = prepend_with(states, prepend_str)
    initial_states = prepend_with(initial_states, prepend_str)
    accepting_states = prepend_with(accepting_states, prepend_str)
    
    ba = BA(name=name, atomic_proposition_based=atomic_proposition_based)
    
    ba.states.add_from(states)
    ba.states.initial |= initial_states
    ba.states.add_accepting_from(accepting_states)
    
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
            guard = str2singleton(guard)
        ba.transitions.add_labeled(from_state, to_state, guard)
    
    return ba



def _ba_ts_sync_prod(buchi_automaton, transition_system):
    """Construct Buchi Automaton equal to synchronous product TS x NBA.
    
    returns
    -------
    C{prod_ba}, the product Buchi Automaton.
    
    see also
    --------
    _ts_ba_sync_prod, BuchiAutomaton.sync_prod
    """
    (prod_ts, persistent) = _ts_ba_sync_prod(
        transition_system, buchi_automaton
    )
    
    prod_name = buchi_automaton.name +'*' +transition_system.name
    
    if prod_ts.states.mutants:
        mutable = True
    else:
        mutable = False
    
    prod_ba = BuchiAutomaton(name=prod_name, mutable=mutable)
    
    # copy S, S0, from prod_TS-> prod_BA
    prod_ba.states.add_from(prod_ts.states() )
    prod_ba.states.initial |= prod_ts.states.initial()
    print('initial:\n\t' +str(prod_ts.states.initial) )
    # accepting states = persistent set
    prod_ba.states.add_accepting_from(persistent)
    
    # copy edges, translating transitions, i.e., chaning transition labels
    if buchi_automaton.atomic_proposition_based:
        # direct access, not the inefficient
        #   prod_ba.alphabet.add_from(buchi_automaton.alphabet() ),
        # which would generate a combinatorially large alphabet
        prod_ba.alphabet.math_set |= buchi_automaton.alphabet.math_set
    else:
        msg ="""
            Buchi Automaton must be Atomic Proposition-based,
            otherwise the synchronous product is not well-defined.
            """
        raise Exception(msg)
    
    for (from_state_id, to_state_id) in prod_ts.transitions():
        # prject prod_TS state to TS state
        
        from_state = prod_ts.states._int2mutant(from_state_id)
        to_state = prod_ts.states._int2mutant(to_state_id)
        
        ts_to_state = to_state[0]
        msg = 'prod_TS: to_state =\n\t' +str(to_state) +'\n'
        msg += 'TS: ts_to_state =\n\t' +str(ts_to_state)
        dprint(msg)
        
        state_label_pairs = transition_system.states.find(ts_to_state)
        (ts_to_state_, transition_label_dict) = state_label_pairs[0]
        transition_label_value = transition_label_dict['ap']
        prod_ba.transitions.add_labeled(
            from_state, to_state, transition_label_value
        )
    
    return prod_ba

def _ts_ba_sync_prod(transition_system, buchi_automaton):
    """Construct transition system equal to synchronous product TS x NBA.
    
    returns
    -------
    C{(prod_ts, persist) }, where C{prod_ts} is a transition system representing
    the synchronous product between the transition system TS and the
    non-deterministic Buchi Automaton NBA. C{persist} is the subset of states of
    C{prod_ts} which is the preimage under projection of the set of accepting
    states of the Buchi Automaton BA.
    
    Def. 4.62, p.200 [Baier]
    
    erratum
    -------
    note the erratum: P_{pers}(A) is ^_{q\in F} !q, verified from:
        http://www-i2.informatik.rwth-aachen.de/~katoen/errata.pdf
    
    see also
    --------
    _ba_ts_sync_prod, FiniteTransitionSystem.sync_prod
    """
    def convert_ts2ba_label(state_label_dict):
        """Replace 'ap' key with 'in_alphabet'.
        
        @param state_label_dict: FTS state label, its value \\in 2^AP
        @type state_label_dict: dict {'ap' : state_label_value}
        
        @return: BA edge label, its value \\in 2^AP
            (same value with state_label_dict)
        @rtype: dict {'in_alphabet' : edge_label_value}
            Note: edge_label_value is the BA edge "guard"
        """
        dprint('Ls0:\t' +str(state_label_dict) )
        
        (s0_, label_dict) = state_label_dict[0]
        Sigma_dict = {'in_alphabet': label_dict['ap'] }
        
        dprint('State label of: ' +str(s0) +', is: ' +str(Sigma_dict) )
        
        return Sigma_dict
    
    if not isinstance(transition_system, transys.FiniteTransitionSystem):
        msg = 'transition_system not transys.FiniteTransitionSystem.\n'
        msg += 'Actual type passed: ' +str(type(transition_system) )
        raise TypeError(msg)
    
    if not isinstance(buchi_automaton, BuchiAutomaton):
        msg = 'transition_system not transys.BuchiAutomaton.\n'
        msg += 'Actual type passed: ' +str(type(buchi_automaton) )
        raise TypeError(msg)
    
    if not buchi_automaton.atomic_proposition_based:
        msg = """Buchi automaton not stored as Atomic Proposition-based.
                synchronous product with Finite Transition System
                is not well-defined."""
        raise Exception(msg)
    
    fts = transition_system
    ba = buchi_automaton
    
    prodts_name = fts.name +'*' +ba.name
    
    if fts.states.mutants or ba.states.mutants:
        mutable = True
    else:
        mutable = False
    
    # using set() destroys order
    prodts = transys.FiniteTransitionSystem(
        name=prodts_name, states=set(), mutable=mutable
    )
    prodts.atomic_propositions.add_from(ba.states() )
    prodts.actions.add_from(fts.actions)

    # construct initial states of product automaton
    s0s = fts.states.initial()
    q0s = ba.states.initial()
    
    accepting_states_preimage = MathSet()
    
    dprint(hl +'\n' +' Product TS construction:\n' +hl +'\n')
    for s0 in s0s:
        dprint('Checking initial state:\t' +str(s0) )
        
        Ls0 = fts.states.find(s0)
        
        # desired input letter for BA
        Sigma_dict = convert_ts2ba_label(Ls0)
        
        for q0 in q0s:
            enabled_ba_trans = ba.transitions.find(
                [q0], desired_label=Sigma_dict
            )
            
            # q0 blocked ?
            if not enabled_ba_trans:
                dprint('blocked q0 = ' +str(q0) )
                continue
            
            # which q next ?     (note: curq0 = q0)
            dprint('enabled_ba_trans = ' +str(enabled_ba_trans) )
            for (curq0, q, sublabels) in enabled_ba_trans:
                new_sq0 = (s0, q)                
                prodts.states.add(new_sq0)
                prodts.states.initial.add(new_sq0)
                prodts.states.label(new_sq0, {q} )
                
                # accepting state ?
                if ba.states.is_accepting(q):
                    accepting_states_preimage.add(new_sq0)
    
    dprint(prodts)
    
    # start visiting reachable in DFS or BFS way
    # (doesn't matter if we are going to store the result)    
    queue = MathSet(prodts.states.initial() )
    visited = MathSet()
    while queue:
        sq = queue.pop()
        visited.add(sq)
        (s, q) = sq
        
        dprint('Current product state:\n\t' +str(sq) )
        
        # get next states
        next_ss = fts.states.post_single(s)
        next_sqs = MathSet()
        for next_s in next_ss:
            dprint('Next state:\n\t' +str(next_s) )
            
            Ls = fts.states.find(next_s)
            if not Ls:
                raise Exception(
                    'No AP label for FTS state: ' +str(next_s) +
                     '\n Did you forget labeing it ?'
                )
            
            Sigma_dict = convert_ts2ba_label(Ls)
            dprint("Next state's label:\n\t" +str(Sigma_dict) )
            
            enabled_ba_trans = ba.transitions.find(
                [q], desired_label=Sigma_dict
            )
            dprint('Enabled BA transitions:\n\t' +
                   str(enabled_ba_trans) )
            
            if not enabled_ba_trans:
                continue
            
            for (q, next_q, sublabels) in enabled_ba_trans:
                new_sq = (next_s, next_q)
                next_sqs.add(new_sq)
                dprint('Adding state:\n\t' +str(new_sq) )
                
                prodts.states.add(new_sq)
                
                if ba.states.is_accepting(next_q):
                    accepting_states_preimage.add(new_sq)
                    dprint(str(new_sq) +' contains an accepting state.')
                
                prodts.states.label(new_sq, {next_q} )
                
                dprint('Adding transitions:\n\t' +str(sq) +
                       '--->' +str(new_sq) )
                
                # is fts transition labeled with an action ?
                ts_enabled_trans = fts.transitions.find(
                    [s], to_states=[next_s],
                    desired_label='any', as_dict=False
                )
                for (from_s, to_s, sublabel_values) in ts_enabled_trans:
                    assert(from_s == s)
                    assert(to_s == next_s)
                    dprint('Sublabel value:\n\t' +str(sublabel_values) )
                    
                    # labeled transition ?
                    if not sublabel_values:
                        prodts.transitions.add(sq, new_sq)
                    else:
                        #TODO open FTS
                        prodts.transitions.add_labeled(
                            sq, new_sq, sublabel_values[0]
                        )
        
        # discard visited & push them to queue
        new_sqs = MathSet()
        for next_sq in next_sqs:
            if next_sq not in visited:
                new_sqs.add(next_sq)
                queue.add(next_sq)
    
    return (prodts, accepting_states_preimage)

class RabinPairs(object):
    """Acceptance pairs for Rabin automaton.
    
    Each pair defines an acceptance condition.
    A pair (L, U) comprises of:
        - a set L of "good" states
        - a set U of "bad" states
    L,U must each be a subset of States.
    
    A run: (q0, q1, ...) is accepted if for at least one Rabin Pair,
    it in intersects L an inf number of times, but U only finitely.
    
    Internally a list of 2-tuples of SubSet objects is maintained:
        [(L1, U1), (L2, U2), ...]
    where: Li, Ui, are SubSet objects, with superset
    the Rabin automaton's States.
    
    caution
    -------
    Here and in ltl2dstar documentation L denotes a "good" set.
    [Baier 2008] denote the a "bad" set with L.
    To avoid ambiguity, attributes: .good, .bad were used here.
    
    example
    -------
    >>> dra = RabinAutomaton()
    >>> dra.states.add_from([1, 2, 3] )
    >>> dra.states.accepting.add([1], [2] )
    >>> dra.states.accepting
    
    >>> dra.states.accepting.good(1)
    
    >>> dra.states.accepting.bad(1)
    
    see also
    --------
    RabinAutomaton
    Def. 10.53, p.801, [Baier 2008]
    ltl2dstar documentation: 
    """
    def __init__(self, automaton_states):
        self._states = automaton_states
        self._pairs = []
    
    def __repr__(self):
        s = 'L = Good states, U = Bad states\n' +30*'-' +'\n'
        for index, (good, bad) in enumerate(self._pairs):
            s += 'Pair: ' +str(index) +', L = ' +str(good)
            s += ', U = ' +str(bad) +'\n'
        return s
    
    def __str__(self):
        return self.__repr__()
    
    def __getitem__(self, index):
        return self._pairs[index]
    
    def __iter__(self):
        return iter(self._pairs)
    
    def __call__(self):
        """Get list of 2-tuples (L, U) of good-bad sets of states.
        """
        return list(self._pairs)
    
    def add(self, good_states, bad_states):
        """Add new acceptance pair (L, U).
        
        see also
        --------
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
        
        self._pairs.append((good_set, bad_set) )
    
    def remove(self, good_states, bad_states):
        """Delete pair (L, U) of good-bad sets of states.
        
        note
        ----
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
        
        see also
        --------
        add
        
        @param good_states: set of good states of this pair
        @type good_states: 
        """
        good_set = SubSet(self._states)
        good_set |= good_states
        
        bad_set = SubSet(self._states)
        bad_set |= bad_states
        
        self._pairs.remove((good_set, bad_set) )
    
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

class RabinAutomaton(OmegaAutomaton):
    """Rabin finite-state omega-automaton.
    """    
    def __init__(self, **args):
        OmegaAutomaton.__init__(self, **args)
        
        self.states.accepting = RabinPairs(self.states)
    
    def __repr__(self):
        s = hl +'\nRabin Automaton: ' +self.name +'\n' +hl +'\n'
        s += 'States:\n'
        s += pformat(self.states(data=False), indent=3) +2*'\n'
        s += 'Initial States:\n'
        s += pformat(self.states.initial, indent=3) +2*'\n'
        s += 'Accepting States:\n'
        s += pformat(self.accepting_states, indent=3) +2*'\n'
        
        if self.atomic_proposition_based:
            s += 'Input Alphabet Letters (\in 2^AP):\n\t'
        else:
            s += 'Input Alphabet Letters:\n\t'
        s += str(self.alphabet) +2*'\n'
        s += 'Transitions & labeling w/ Input Letters:\n'
        s += pformat(self.transitions(labeled=True), indent=3)
        s += '\n' +hl +'\n'
        
        return s
    
    def __str__(self):
        return self.__repr__()
    
    def acceptance_condition(self, prefix, suffix):
        raise NotImplementedError

class DRA(RabinAutomaton):
    """Alias of RabinAutomaton
    """
    # TODO enforce Determinism for this alias
    def __init__(self, **args):
        RabinAutomaton.__init__(self, **args)

class StreettAutomaton(OmegaAutomaton):
    """Omega-automaton with Streett acceptance condition.
    """
    def acceptance_condition(self, prefix, suffix):
        raise NotImplementedError

class MullerAutomaton(OmegaAutomaton):
    """Omega-automaton with Muller acceptance condition.
    """
    def acceptance_condition(self, prefix, suffix):
        raise NotImplementedError

def ba2dra():
    """Buchi to Deterministic Rabin Automaton converter.
    """
    raise NotImplementedError

def ba2ltl():
    """Buchi Automaton to Linear Temporal Logic formula convertr.
    """
    raise NotImplementedError

class ParityAutomaton(OmegaAutomaton):
    def gr1c_str():
        raise NotImplementedError

class ParityGameGraph():
    """Parity Games.
    """

class WeightedAutomaton():
    """Weighted Automaton.
    """

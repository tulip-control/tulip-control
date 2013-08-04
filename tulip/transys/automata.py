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

class FiniteStateAutomaton(LabeledStateDiGraph):
    """Generic automaton.
    
    1) states
    2) initial states
    3) final states
    
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
        self, final_states=[], input_alphabet_or_atomic_propositions=[],
        atomic_proposition_based=True, mutable=False, **args
    ):
        LabeledStateDiGraph.__init__(
            self, mutable=mutable,
            removed_state_callback=self._removed_state_callback, **args
        )
        
        if mutable:
            self.final_states = list()
        else:
            self.final_states = set()
        self.add_final_states_from(final_states)
        
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
        
        # used before label value
        self._transition_dot_label_format = {'in_alphabet':'',
                                                'type?label':'',
                                                'separator':'\\n'}
        
        self.dot_node_shape = {'normal':'circle', 'final':'doublecircle'}
        self.default_export_fname = 'fsa'
        
    def __str__(self):
        s = str(self.states)
        s += '\nState Labels:\n' +pformat(self.states(data=True) ) +'\n'
        s += str(self.transitions) +'\n'
        s += 'Alphabet:\n' +str(self.alphabet) +'\n'
        s += 'Final States:\n\t' +str(self.final_states)
        
        return s
    
    def _removed_state_callback(self, rm_state):
        self.remove_final_state(rm_state)
    
    def _is_final(self, state):
        return is_subset([state], self.final_states)

    # final states
    def add_final_state(self, new_final_state):
        if not new_final_state in self.states():
            raise Exception('Given final state:\n\t' +str(new_final_state) +
                            '\n\\notin States:\n\t' +str(self.states() ) )
        
        # already final ?
        if self._is_final(new_final_state):
            warnings.warn('Attempting to add existing final state.\n')
            return
        
        # mutable states ?
        if self.states.mutants == None:
            self.final_states.add(new_final_state)
        else:
            self.final_states.append(new_final_state)

    def add_final_states_from(self, new_final_states):
        if not is_subset(new_final_states, self.states() ):
            raise Exception('Given Final States \\notsubset States.')
        
        # mutable states ?
        if self.states.mutants == None:
            self.final_states |= set(new_final_states)
        else:
            for new_final_state in new_final_states:
                self.add_final_state(new_final_state)
    
    def number_of_final_states(self):
        return len(self.final_states)
    
    def remove_final_state(self, rm_final_state):
        self.final_states.remove(rm_final_state)
    
    def remove_final_states_from(self, rm_final_states):
        if self.states.mutants == None:
            self.final_states = self.final_states.difference(rm_final_states)
        else:
            for rm_final_state in rm_final_states:
                self.remove_final_state(rm_final_state)

    # checks
    def is_deterministic(self):
        """overloaded method."""
        raise NotImplementedError
        
    def is_blocking(self):
        """overloaded method."""
        raise NotImplementedError
    
    def is_accepted(self, input_word):
        """Check if input word is accepted."""
        sim = self.simulate(input_word)
        
        inf_states = set(sim.run.get_suffix() )
        
        if bool(inf_states & self.final_states):
            accept = True
        else:
            accept = False
        
        return accept
        
    def simulate(self, initial_state, input_word):
        """Returns an Omega Automaton Simulation, with prefix, suffix."""
        
        # should be implemented properly with nested depth-first search,
        # becaus of possible branching due to non-determinism
        
        for letter in input_word:
            dprint(letter)
            
            # blocked
        
        return FSASim()

    # operations on two automata
    def add_subautomaton(self):
        raise NotImplementedError

class StarAutomaton(FiniteStateAutomaton):
    """Finite-word finite-state automaton."""

class DeterninisticFiniteAutomaton(StarAutomaton):
    """Deterministic finite-word finite-state Automaton."""

    # check each initial state added
    # check each transition added
    
class DFA(DeterninisticFiniteAutomaton):
    """Alias for deterministic finite-word finite-state automaton."""

class NonDeterministicFiniteAutomaton(StarAutomaton):
    """"Non-deterministic finite-word finite-state automaton."""
    
    # note:
    #   is_deterministic still makes sense
    
class NFA(NonDeterministicFiniteAutomaton):
    """Alias for non-deterministic finite-word finite-state automaton."""

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
    
    def __str__(self):
        s = hl +'\nB' +u'\xfc' +'chi Automaton\n' +hl
        s += str(self.states) +'\n'
        s += 'Input Alphabet Letters (\in 2^AP):\n\t' +str(self.alphabet)
        s += '\n' +str(self.transitions) +'\n' +hl +'\n'
        
        return s
    
    def __repr__(self):
        return self.__str__()
    
    def __add__(self, other):
        """Union of two automata, with equal states identified."""
        raise NotImplementedError
    
    def __mul__(self, ts_or_ba):
        return self.sync_prod(ts_or_ba)
    
    def __or__(self, ba):
        return self.async_prod(ba)
        
    def _ba_ba_sync_prod(self, ba2):
        ba1 = self
        
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
        elif isinstance(ts_or_ba, FiniteTransitionSystem):
            ts = ts_or_ba
            return _ba_ts_sync_prod(self, ts)
        else:
            raise Exception('ts_or_ba should be an FTS or a BA.\n'+
                            'Got type: ' +str(ts_or_ba) )
    
    def async_prod(self, other):
        """Should it be defined in a superclass ?"""
        raise NotImplementedError
    
    def acceptance_condition(self, prefix, suffix):
        """Check if given infinite word over alphabet \Sigma is accepted."""
    
    def determinize(self):
        raise NotImplementedError
    
    def complement(self):
        raise NotImplementedError

class BA(BuchiAutomaton):
    def __init__(self, **args):
        BuchiAutomaton.__init__(self, **args)

def tuple2ba(S, S0, Sa, Sigma_or_AP, trans, name='ba', prepend_str=None,
             atomic_proposition_based=True, verbose=False):
    """Create a Buchi Automaton from a tuple of fields.
    
    note
    ====
    "final states" in the context of \\omega-automata is a misnomer,
    because the system never reaches a "final" state, as in non-transitioning.

    So "accepting states" allows for an evolving behavior,
    and is a better description.

    "final states" is appropriate for NFAs.
    
    see also
    ========
    L{tuple2fts}

    @type ba_tuple: tuple
    @param ba_tuple: defines Buchi Automaton by a tuple (Q, Q_0, Q_F,
        \\Sigma, trans) (maybe replacing \\Sigma by AP since it is an
        AP-based BA ?)  where:

            - Q = set of states
            - Q_0 = set of initial states, must be \\subset S
            - Q_F = set of final states
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
    ba.states.add_initial_from(initial_states)
    ba.states.add_final_from(accepting_states)
    
    if atomic_proposition_based:
        ba.alphabet.add_set_elements(alphabet_or_ap)
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
    (prod_ts, persistent) = _ts_ba_sync_prod(transition_system, buchi_automaton)
    
    prod_name = buchi_automaton.name +'*' +transition_system.name
    prod_ba = BuchiAutomaton(name=prod_name)
    
    # copy S, S0, from prod_TS-> prod_BA
    prod_ba.states.add_from(prod_ts.states() )
    prod_ba.states.add_initial_from(prod_ts.states.initial)
    
    # final states = persistent set
    prod_ba.states.add_final_from(persistent)
    
    # copy edges, translating transitions, i.e., chaning transition labels
    if buchi_automaton.atomic_proposition_based:
        # direct access, not the inefficient
        #   prod_ba.alphabet.add_from(buchi_automaton.alphabet() ),
        # which would generate a combinatorially large alphabet
        prod_ba.alphabet.add_set_elements(buchi_automaton.alphabet.math_set)
    else:
        msg ="""
            Buchi Automaton must be Atomic Proposition-based,
            otherwise the synchronous product is not well-defined.
            """
        raise Exception(msg)
    
    for (from_state, to_state) in prod_ts.edges_iter():
        # prject prod_TS state to TS state        
        ts_to_state = to_state[0]
        msg = 'prod_TS: to_state =\n\t' +str(to_state) +'\n'
        msg += 'TS: ts_to_state =\n\t' +str(ts_to_state)
        dprint(msg)
        
        transition_label = transition_system.atomic_propositions.of(ts_to_state)
        prod_ba.transitions.add_labeled(from_state, to_state, transition_label)   
    
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
    if not isinstance(transition_system, FiniteTransitionSystem):
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
    # using set() destroys order
    prodts = FiniteTransitionSystem(name=prodts_name, states=set() )
    prodts.atomic_propositions.add_from(ba.states() )
    prodts.actions.add_from(fts.actions)

    # construct initial states of product automaton
    s0s = fts.states.initial.copy()
    q0s = ba.states.initial.copy()
    
    final_states_preimage = set()    
    
    for s0 in s0s:
        dprint('----\nChecking initial state:\n\t' +str(s0) )        
        
        Ls0 = fts.atomic_propositions.of(s0)
        Ls0_dict = {'in_alphabet': Ls0}
        
        for q0 in q0s:
            enabled_ba_trans = ba.transitions.find({q0}, desired_label=Ls0_dict)
            
            # q0 blocked ?
            if len(enabled_ba_trans) == 0:
                dprint('blocked q0 = ' +str(q0) )
                continue
            
            # which q next ?     (note: curq0 = q0)
            for (curq0, q, sublabels) in enabled_ba_trans:
                new_sq0 = (s0, q)                
                prodts.states.add(new_sq0)
                prodts.states.add_initial(new_sq0)
                prodts.atomic_propositions.label_state(new_sq0, {q} )
                
                # final state ?
                if ba.states.is_final(q):
                    final_states_preimage.add(new_sq0)
    
    dprint(prodts)    
    
    # start visiting reachable in DFS or BFS way
    # (doesn't matter if we are going to store the result)    
    queue = prodts.states.initial.copy()
    visited = set()
    while queue:
        sq = queue.pop()
        visited.add(sq)
        (s, q) = sq
        
        dprint('Current product state:\n\t' +str(sq) )
        
        # get next states
        next_ss = fts.states.post_single(s)
        next_sqs = set()
        for next_s in next_ss:
            dprint('Next state:\n\t' +str(next_s) )
            
            Ls = fts.atomic_propositions.of(next_s)
            if Ls is None:
                raise Exception('No AP label for FTS state: ' +str(next_s) +
                                '\n Did you forget labeing it ?')
            Ls_dict = {'in_alphabet': Ls}

            dprint("Next state's label:\n\t" +str(Ls_dict) )
            
            enabled_ba_trans = ba.transitions.find({q}, desired_label=Ls_dict)
            dprint('Enabled BA transitions:\n\t' +str(enabled_ba_trans) )
            
            if len(enabled_ba_trans) == 0:
                continue
            
            for (q, next_q, sublabels) in enabled_ba_trans:
                new_sq = (next_s, next_q)
                next_sqs.add(new_sq)
                dprint('Adding state:\n\t' +str(new_sq) )
                
                prodts.states.add(new_sq)
                
                if ba.states.is_final(next_q):
                    final_states_preimage.add(new_sq)
                    dprint(str(new_sq) +' contains a final state.')
                
                prodts.atomic_propositions.label_state(new_sq, {next_q} )
                
                dprint('Adding transitions:\n\t' +str(sq) +'--->' +str(new_sq) )
                # is fts transition labeled with an action ?
                ts_enabled_trans = fts.transitions.find(
                    {s}, to_states={next_s}, desired_label='any', as_dict=False
                )
                for (from_s, to_s, sublabel_values) in ts_enabled_trans:
                    #attr_dict = fts.get_edge_data(from_s, to_s, key=edge_key)
                    assert(from_s == s)
                    assert(to_s == next_s)
                    dprint('Sublabel value:\n\t' +str(sublabel_values) )
                    
                    # labeled transition ?
                    if len(sublabel_values) == 0:
                        prodts.transitions.add(sq, new_sq)
                    else:
                        #TODO open FTS
                        prodts.transitions.add_labeled(sq, new_sq,
                                                       sublabel_values[0] )
        
        # discard visited & push them to queue
        new_sqs = set()
        for next_sq in next_sqs:
            if next_sq not in visited:
                new_sqs.add(next_sq)
                queue.add(next_sq)
    
    return (prodts, final_states_preimage)

class RabinAutomaton(OmegaAutomaton):
    """Remember to override the final set management."""
    def acceptance_condition(self, prefix, suffix):
        raise NotImplementedError

class StreettAutomaton(OmegaAutomaton):
    def acceptance_condition(self, prefix, suffix):
        raise NotImplementedError

class MullerAutomaton(OmegaAutomaton):
    """Probably not very useful as a data structure for practical purposes."""
    
    def acceptance_condition(self, prefix, suffix):
        raise NotImplementedError

def ba2dra():
    """Buchi to Deterministic Rabin Automaton converter."""

def ba2ltl():
    """Buchi Automaton to Linear Temporal Logic formula convertr."""

class ParityAutomaton(OmegaAutomaton):
    
    def dump_gr1c():
        raise NotImplementedError

class ParityGameGraph():
    """Parity Games."""

class WeightedAutomaton():
    """."""

# Copyright (c) 2013-2014 by California Institute of Technology
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
Products between automata and transition systems
"""
import logging
logger = logging.getLogger(__name__)

import warnings
from collections import Iterable
from pprint import pformat

from .labeled_graphs import LabeledDiGraph
from .labeled_graphs import prepend_with, str2singleton
from .mathset import MathSet, SubSet, PowerSet

from . import transys
from . import automata

_hl = 40 *'-'

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
    L{automata._ba_ts_sync_prod}, L{transys.FiniteTransitionSystem.sync_prod}
    
    @return: C{(product_ts, persistent_states)}, where:
        - C{product_ts} is the synchronous product TS * BA
        - C{persistent_states} are those in TS * BA which
            project on accepting states of BA.
    @rtype:
        - C{product_TS} is a L{transys.FiniteTransitionSystem}
        - C{persistent_states} is the set of states which project
            on accepting states of the Buchi Automaton BA.
    """
    if not isinstance(transition_system, FiniteTransitionSystem):
        msg = 'transition_system not transys.FiniteTransitionSystem.\n'
        msg += 'Actual type passed: ' +str(type(transition_system) )
        raise TypeError(msg)
    
    if not hasattr(buchi_automaton, 'alphabet'):
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
    prodts = transys.FiniteTransitionSystem(name=prodts_name)
    prodts.states.add_from(set() )
    prodts.atomic_propositions.add_from(ba.states() )
    prodts.actions.add_from(fts.actions)

    # construct initial states of product automaton
    s0s = set(fts.states.initial)
    q0s = set(ba.states.initial)
    
    accepting_states_preimage = MathSet()
    
    logger.debug(_hl +'\n' +' Product TS construction:\n' +_hl +'\n')
    
    if not s0s:
        msg = 'Transition System has no initial states !\n'
        msg += '=> Empty product system.\n'
        msg += 'Did you forget to define initial states ?'
        warnings.warn(msg)
    
    for s0 in s0s:
        logger.debug('Checking initial state:\t' +str(s0) )
        
        # desired input letter for BA
        ap = fts.node[s0]['ap']
        Sigma_dict = {'letter':ap}
        
        for q0 in q0s:
            enabled_ba_trans = ba.transitions.find(
                [q0], with_attr_dict=Sigma_dict
            )
            enabled_ba_trans += ba.transitions.find(
                [q0], letter={True}
            )
            
            # q0 blocked ?
            if not enabled_ba_trans:
                logger.debug('blocked q0 = ' +str(q0) )
                continue
            
            # which q next ?     (note: curq0 = q0)
            logger.debug('enabled_ba_trans = ' +str(enabled_ba_trans) )
            for (curq0, q, sublabels) in enabled_ba_trans:
                new_sq0 = (s0, q)
                prodts.states.add(new_sq0)
                prodts.states.initial.add(new_sq0)
                prodts.states[new_sq0]['ap'] = {q}
                
                # accepting state ?
                if q in ba.states.accepting:
                    accepting_states_preimage.add(new_sq0)
    
    logger.debug(prodts)
    
    # start visiting reachable in DFS or BFS way
    # (doesn't matter if we are going to store the result)    
    queue = MathSet(prodts.states.initial)
    visited = MathSet()
    while queue:
        sq = queue.pop()
        visited.add(sq)
        (s, q) = sq
        
        logger.debug('Current product state:\n\t' +str(sq) )
        
        # get next states
        next_ss = fts.states.post(s)
        next_sqs = MathSet()
        for next_s in next_ss:
            logger.debug('Next state:\n\t' +str(next_s) )
            
            try:
                ap = fts.node[next_s]['ap']
            except:
                raise Exception(
                    'No AP label for FTS state: ' +str(next_s) +
                     '\n Did you forget labeing it ?'
                )
            
            Sigma_dict = {'letter':ap}
            logger.debug("Next state's label:\n\t" +str(Sigma_dict) )
            
            enabled_ba_trans = ba.transitions.find(
                [q], with_attr_dict=Sigma_dict
            )
            enabled_ba_trans += ba.transitions.find(
                [q], letter={True}
            )
            logger.debug('Enabled BA transitions:\n\t' +
                          str(enabled_ba_trans) )
            
            if not enabled_ba_trans:
                logger.debug('No enabled transitions')
                continue
            
            for (q, next_q, sublabels) in enabled_ba_trans:
                new_sq = (next_s, next_q)
                next_sqs.add(new_sq)
                logger.debug('Adding state:\n\t' + str(new_sq) )
                
                prodts.states.add(new_sq)
                
                if next_q in ba.states.accepting:
                    accepting_states_preimage.add(new_sq)
                    logger.debug(str(new_sq) +
                                  ' contains an accepting state.')
                
                prodts.states[new_sq]['ap'] = {next_q}
                
                logger.debug('Adding transitions:\n\t' +
                              str(sq) + '--->' + str(new_sq) )
                
                # is fts transition labeled with an action ?
                ts_enabled_trans = fts.transitions.find(
                    [s], to_states=[next_s],
                    with_attr_dict=None
                )
                for (from_s, to_s, sublabel_values) in ts_enabled_trans:
                    assert(from_s == s)
                    assert(to_s == next_s)
                    logger.debug('Sublabel value:\n\t' +
                                  str(sublabel_values) )
                    
                    # labeled transition ?
                    if not sublabel_values:
                        prodts.transitions.add(sq, new_sq)
                    else:
                        #TODO open FTS
                        prodts.transitions.add(
                            sq, new_sq, actions=sublabel_values['actions']
                        )
        
        # discard visited & push them to queue
        new_sqs = MathSet()
        for next_sq in next_sqs:
            if next_sq not in visited:
                new_sqs.add(next_sq)
                queue.add(next_sq)
    
    return (prodts, accepting_states_preimage)

def ba_ts_sync_prod(buchi_automaton, transition_system):
    """Construct Buchi Automaton equal to synchronous product TS x NBA.
    
    See Also
    ========
    L{transys._ts_ba_sync_prod}, L{BuchiAutomaton.sync_prod}

    @return: C{prod_ba}, the product L{BuchiAutomaton}.
    """
    (prod_ts, persistent) = ts_ba_sync_prod(
        transition_system, buchi_automaton
    )
    
    prod_name = buchi_automaton.name +'*' +transition_system.name
    
    prod_ba = automata.BuchiAutomaton(name=prod_name)
    
    # copy S, S0, from prod_TS-> prod_BA
    prod_ba.states.add_from(prod_ts.states() )
    prod_ba.states.initial |= set(prod_ts.states.initial)
    
    # accepting states = persistent set
    prod_ba.states.accepting |= persistent
    
    # copy edges, translating transitions,
    # i.e., changing transition labels
    if not buchi_automaton.atomic_proposition_based:
        msg ='Buchi Automaton must be Atomic Proposition-based,' +\
             ' otherwise the synchronous product is not well-defined.'
        raise Exception(msg)
    
    # direct access, not the inefficient
    #   prod_ba.alphabet.add_from(buchi_automaton.alphabet() ),
    # which would generate a combinatorially large alphabet
    prod_ba.alphabet.math_set |= buchi_automaton.alphabet.math_set
    
    for (from_state, to_state) in prod_ts.transitions():
        # prject prod_TS state to TS state
        ts_to_state = to_state[0]
        msg = 'prod_TS: to_state =\n\t' +str(to_state) +'\n'
        msg += 'TS: ts_to_state =\n\t' +str(ts_to_state)
        logger.debug(msg)
        
        state_label_pairs = transition_system.states.find(ts_to_state)
        (ts_to_state_, transition_label_dict) = state_label_pairs[0]
        transition_label_value = transition_label_dict['ap']
        prod_ba.transitions.add(
            from_state, to_state, letter=transition_label_value
        )
    
    return prod_ba

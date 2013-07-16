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
Transition System module usage examples
"""

import networkx as nx
import tulip.transys as ts
from pprint import pprint
import warnings

def pprint_states(sys):
    system_class = sys.__class__.__name__
    system_name = sys.name
    print('The states of the ' +system_class +
            '\n named [' +system_name +'] are:'
        )
    pprint(sys.states() )

save_fig = True

def sims_demo():
    """Storing simulations."""
    #=============
    # Sequences
    #=============
    # fts def:
    #   s1 -[a]-> s2
    #   s2 -[b]-> s3
    #   s3 -[a]-> s3
    prefix = ['s1', 'a', 's2', 'b', 's3']
    suffix = ['s3', 'a', 's2', 'b', 's3']
    execution = ts.InfiniteSequence(prefix, suffix)
    #print(execution)
    
    # trace def
    prefix = [{'p1', 'p2'}, {}, {'p1'} ]
    suffix = [{'p1'}, {}, {'p1'} ]
    trace = ts.InfiniteSequence(prefix, suffix)
    
    #print(trace.get_prefix() )
    #print(trace.get_suffix() )
    
    #=============
    # FTS Sim
    #=============
    fts_sim = ts.FiniteTransitionSystemSimulation(execution, trace)
    
    print('Execution of FTS: s0, a1, s1, ..., aN, sN')
    print(fts_sim.execution)
    
    print('Path of states: s1, s2, ..., sN')
    print(fts_sim.path)
    
    print('Trace of stat labels: L(s0), L(s1), ..., L(sN)')
    print(fts_sim.trace)
    
    print('Action Trace: a1, a2, ..., aN')
    print(fts_sim.action_trace)
    
    print(fts_sim)
    
    #=============
    # Automaton Sim
    #=============
    prefix = [{'p1'}, {'p3'}, {'p2'}]
    suffix = [{'p2'}, {'p1'}, {'p2'}]
    input_word = ts.InfiniteWord(prefix, suffix)
    
    prefix = ['s1', 's2', 's3']
    suffix = ['s3', 's2', 's3']
    run = ts.InfiniteSequence(prefix, suffix)
    
    aut_sim = ts.FiniteStateAutomatonSimulation(input_word, run)
    print(aut_sim)

def fts_minimal_example():
    """Small example, for more see the maximal example."""
    
    fts = ts.FTS()
    fts.states.add_from(['s0', 's1'] )
    fts.states.add_initial('s0')
    
    fts.atomic_propositions.add_from({'green', 'not green'})
    fts.atomic_propositions.label_state('s0', {'not green'})
    fts.atomic_propositions.label_state('s1', {'green'})
    
    fts.transitions.add('s0', 's1')
    fts.transitions.add('s1', 's0')
    
    if save_fig:
        fts.save_pdf('small_fts.png')
    
    return fts

def fts_maximal_example():
    """Finite-Transition System demo."""
    
    msg = """
    =================================================
        Closed FTS   -    Example 2.2, p.21 [Baier]
    =================================================
    """
    print(msg)
    fts = ts.FiniteTransitionSystem(name='Beverage vending machine')
    
    # add state info
    fts.states.add('pay')
    fts.states.remove('pay')
    
    fts.states.add_from({'pay', 'soda', 'select', 'beer'} )
    fts.states.remove_from({'pay', 'soda'} )
    fts.states.add_from({'pay', 'soda'} )
    
    fts.states.set_current('pay')
    
    fts.states.add_initial('pay') # should already be a state
    fts.states.add_initial_from({'soda', 'select'} )
    fts.states.remove_initial('soda')
    fts.states.remove_initial_from({'soda', 'select'} )
    
    fts.states.check() # sanity
    
    # no transitions yet...
    pre = fts.states.pre({'pay'} )
    post = fts.states.post({'pay'} )
    print("Pre('pay') = " +str(pre) )
    print("Post('pay') = " +str(post) )
    
    try:
        fts.states.add_initial('not pay')
    except:
        warnings.warn('You cannot add an initial state \\notin states.')
    
    # get state info
    print('States:\n\t' +str(fts.states() ) )
    print('Number of states:\n\t' +str(fts.states.number() ) )
    print('Initial states:\n\t' +str(fts.states.initial) )
    print('Number of initial states:\n\t' +str(fts.states.number_of_initial() ) )
    print('Current state:\n\t' +str(fts.states.current) )
    
    print("Is 'pay' a state ?\n\t" +str('pay' in fts.states) )
    print("Is 'not pay' a state ?\n\t" +str('not pay' in fts.states() ) )
    print("Is 'bla' a state ?\n\t" +str('bla' in fts) )
    print('')
    
    # add transition info (unlabeled)
    fts.transitions.add('pay', 'select') # notice: no labels
    fts.transitions.add_from({'select'}, {'soda', 'beer'} )
    fts.transitions.add_from({'soda', 'beer'}, {'pay'} )
    fts.transitions.add('pay', 'soda')
    fts.transitions.remove('pay', 'soda')
    
    pre = fts.states.pre({'pay'} )
    post = fts.states.post({'pay'} )
    print("Pre('pay') =\n\t" +str(pre) )
    print("Post('pay') =\n\t" +str(post) +'\n')
    
    # another way
    fts['pay'] # post of 'pay', note the edge ids returned, due to MultiDiGraph
    fts['select']
    fts['select']['soda']
    
    try:
        pre = fts.states.pre('pay')
    except:
        print('pre cannot distinguish single states from sets of states, '+
               'because a state can be anything, e.g., something iterable.\n')
    # use instead
    pre = fts.states.pre_single('pay')
    
    # all 1-hop post sets
    print('\n-----\nPost sets are:\n')
    for state in fts.states():
        post = fts.states.post({state} )
        # same as:
        # post = fts.states.post_signle(state)
        
        print('\tof state: ' +str(state) +', the states: ' +str(post) )
    print('------\n')
    
    # same thing
    post_all = fts.states.post(fts.states() )
    print('Same thing as above:\n\t' +str(post_all) +'\n--------\n')
    
    try:
        fts.transitions.add('pay', 'not yet a state')
    except:
        print('First add  from_, to_ states, then you can add transition.\n')
    fts.transitions.add('pay', 'not yet a state', check_states=False)
    fts.states.remove('not yet a state') # undo
    
    try:
        fts.transitions.add_from({'not a state', 'also not a state'}, {'pay'} )
    except:
        print('Same state check as above.\n')
    fts.transitions.add_from({'not a state', 'also not a state'}, {'pay'},
                             check_states=False)
    fts.states.remove_from({'not a state', 'also not a state'} )
    
    #avoid adding characters 'p', 'a', 'y' at states
    fts.transitions.add_from({'pay'}, {'select'}, check_states=False)
    print("States now include 'p', 'a', 'y':\n\t" +str(fts.states() ) )
    fts.states.remove_from({'p', 'a', 'y'} )
    print("Fixed:\n\t" +str(fts.states() ) +'\n')
    
    # get transition info (unlabeled)
    print('Transitions:\n\t' +str(fts.transitions() ) )
    print('Number of transitions:\n\t' +str(fts.transitions.number() ) +'\n')
    
    print(fts) # pretty
    
    # ----------------------------------------
    # CAUTION: labeling now considered
    # ----------------------------------------
    
    # MultiDiGraph labeling issue vs multiple same edges resolved as follows:
    #   transition addition is strictly monitored
    #   each transition is identified uniquely with its set of labels (key=values)
    #   see Transitions.add_labeled()
    fts.actions.add('insert_coin')
    fts.actions.add_from({'get_soda', 'get_beer', ''} )
    
    try:
        fts.transitions.add_labeled('pay', 'select', 'insert_coin')
    except:
        print('Checks not pre-existing same labeled or unlabeled')
    
    # checking pre-existing CANNOT be overriden,
    # to preserve function semantics
    # check=False used only to add missing states or labels
    
    # first remove unlabeled, then add new labeled
    fts.transitions.remove('pay', 'select')
    print(fts)
    fts.transitions.add_labeled('pay', 'select', 'insert_coin')
    print(fts)
    fts.transitions.remove_labeled('pay', 'select', 'insert_coin')
    
    try:
        fts.transitions.add_labeled('pay', 'new state', 'insert_coin')
    except:
        print('trying to add labeled with new state fails.\n')
    
    try:
        fts.transitions.add_labeled('pay', 'select', 'new action')
    except:
        print('trying to add transition with new label also fails.\n')
    
    # to override and add new state and/or new labels
    print(fts)
    fts.transitions.add_labeled('pay', 'new state', 'new action', check=False)
    print(fts)
    fts.states.remove('new state')
    fts.actions.remove('new action')
    print(fts)
    
    fts.transitions.add_labeled('pay', 'select', 'insert_coin')
    fts.transitions.remove_from({'select'}, {'soda', 'beer'} )
    fts.transitions.add_labeled_from({'select'}, {'soda', 'beer'}, '')
    fts.transitions.label('soda', 'pay', 'get_soda')
    fts.transitions.label('beer', 'pay', 'get_be oops mistake', check_label=False)
    fts.transitions.relabel('beer', 'pay', 'get_be oops mistake', 'get_beer')
    
    fts.actions.remove('get_be oops mistake') # checks that it is not used by edges
    
    try:
        fts.transitions.add_labeled('c12', 'c13', 'insert_coin')
    except:
        print('First add states, then you can add labeled transition between them.')
    fts.transitions.add_labeled('c12', 'c13', 'insert_coin', check=False)
    fts.states.remove_from({'c12', 'c13'} ) # undo
    
    print('Type of (set of) actions: ' +str(fts.actions.name) )
    print('Number of actions: ' +str(fts.actions.number() ) )
    print('Actions: ' +str(fts.actions() ) )
    print('Labeled transitions: ' +str(fts.transitions(data=True) ) )
    
    # fast way to get all edges with value of actions
    nx.get_edge_attributes(fts, 'actions')
    
    # Atomic Propositions (AP)
    fts.atomic_propositions.add('paid')
    fts.atomic_propositions.add_from({'', 'drink'} )
    
    fts.atomic_propositions.label_state('pay', {''})
    fts.atomic_propositions.label_states({'soda', 'beer'}, {'paid', 'drink'} )
    fts.atomic_propositions.label_state('select', {'paid'} )
    
    # no checking of state
    fts.atomic_propositions.add_labeled_state('new state', {'paid'} )
    fts.states.remove('new state')
    
    # same thing, now 'hihi' also added
    fts.atomic_propositions.label_state('new state', {'hihi'}, check=False)
    fts.states.remove('new state')
    fts.atomic_propositions.remove('hihi')
    
    # export
    print('========\n CAUTION: Saving DOT, PDF files\n=========\n')
    path = './test_fts'
    dot_fname = path +'.dot'
    pdf_fname = path +'.pdf'
    
    if save_fig:
        fts.save_pdf(pdf_fname)
        #fts.save_dot(dot_fname)
    # svg support easy to add, so that latex native support is achieved

def ofts_maximal_example():
    """Open FTS demo."""
    msg = '==================\nOpen FTS\n=================='
    print(msg)
    
    ofts = ts.OpenFiniteTransitionSystem()
    
    ofts.states.add_from(['s1', 's2', 's3'] )
    ofts.states.add_initial('s1')
    
    ofts.transitions.add('s1', 's2') # unlabeled
    
    ofts.sys_actions.add('try')
    ofts.sys_actions.add_from({'start', 'stop'} )
    ofts.env_actions.add_from({'block', 'wait'} )
    
    print(ofts.sys_actions)
    print(ofts.env_actions)
    
    ofts.transitions.label('s1', 's2', ['try', 'block'] )
    ofts.transitions.add_labeled('s2', 's3', ['start', 'wait'] )
    ofts.transitions.add_labeled('s3', 's2', ['stop', 'block'] )
    
    print('The Open TS now looks like:')
    print(ofts.transitions() )
    
    ofts.atomic_propositions.add_from({'home', 'lot', 'p1'} )
    
    print(ofts)
    
    path = './test_ofts'
    dot_fname = path +'.dot'
    pdf_fname = path +'.pdf'
    
    if save_fig:
        ofts.save_pdf(pdf_fname)
        #ofts.save_dot(dot_fname)

def ba_minimal_example():
    """Small example.
    
    ![]<>green  = <>[]!green
    
    ref
    ---
    Example 4.64, p.202 [Baier]
    
    note
    ----
    q2 state is a bit redundant, just let the automaton die.
    """
    
    msg = '==================\nBuchi Automaton (small example):    '
    msg += 'Example 4.64, p.202 [Baier]'
    msg += '\n=================='
    print(msg)
    
    ba = ts.BuchiAutomaton(atomic_proposition_based=True)
    ba.states.add_from({'q0', 'q1', 'q2'})
    ba.states.add_initial('q0')
    ba.states.add_final('q1')
    
    # TODO auto negation alphabet closure ?
    
    true = {True}
    green = {'green'}
    not_green = {'not green'}
    ba.alphabet.add_from([true, green, not_green] )
    
    ba.transitions.add_labeled('q0', 'q0', true)
    ba.transitions.add_labeled('q0', 'q1', not_green)
    ba.transitions.add_labeled('q1', 'q1', not_green)
    ba.transitions.add_labeled('q1', 'q2', green)
    ba.transitions.add_labeled('q2', 'q2', true)
    
    if save_fig:
        ba.save_pdf('small_ba.png')
    
    return ba
        
def ba_maximal_example():
    """Buchi Automaton demo."""
    
    print('==================\nBuchi Automaton\n==================')
    ba = ts.BuchiAutomaton(atomic_proposition_based=True)
    
    ba.states.add('q0')
    ba.states.add_from({'q1', 'q2', 'q3'}, destroy_order=True)
    
    ba.states.add_initial('q0')
    
    ba.alphabet.add({'paid'} )
    ba.alphabet.add_from([{'drink', 'paid'}, {''}, {'drink'} ] )
    
    print('Number of letters: ' +str(ba.alphabet.number() ) +'\n')
    print('Alphabet: ' +str(ba.alphabet() ) +'\n')
    
    try:
        ba.transitions.add_labeled('q1', 'q10', {'paid'} )
    except:
        print('q10 not a state.')
    
    ba.transitions.add_labeled('q0', 'q1', frozenset([''] ) )
    ba.transitions.add_labeled('q0', 'q1', frozenset(['paid'] ) )
    ba.transitions.add_labeled('q1', 'q2', frozenset(['paid', 'drink'] ) )
    ba.transitions.add_labeled('q3', 'q0', frozenset([''] ) )
    ba.transitions.add_labeled('q1', 'q3', frozenset(['drink'] ) )
    
    # final states
    ba.add_final_state('q1')
    ba.add_final_states_from({'q2', 'q3'} )
    ba.remove_final_state('q2')
    ba.remove_final_states_from({'q2', 'q3'} )
    
    print('Number of final states:\n\t' +str(ba.number_of_final_states() ) +'\n')
    print('Final states:\n\t' +str(ba.final_states) +'\n')
    
    print(ba)
    
    path = './test_ba'
    dot_fname = path +'.dot'
    pdf_fname = path +'.pdf'
    
    if save_fig:
        ba.save_pdf(pdf_fname)
        #ba.save_dot(dot_fname)
    
    return ba
    
if __name__ == '__main__':
    sims_demo()
    fts_maximal_example()
    ofts_maximal_example()
    ba_maximal_example()    
    
    fts = fts_minimal_example()
    ba = ba_minimal_example()
    prod_fts, final_states_preimage = fts *ba
    
    prod_fts.save_pdf('prod.png')

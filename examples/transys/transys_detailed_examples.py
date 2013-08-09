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
Transition System module developer examples
"""

import networkx as nx
from collections import OrderedDict
import tulip.transys as trs
import warnings

hl = 60*'='
save_fig = False

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
    execution = trs.InfiniteSequence(prefix, suffix)
    #print(execution)
    
    # trace def
    prefix = [{'p1', 'p2'}, {}, {'p1'} ]
    suffix = [{'p1'}, {}, {'p1'} ]
    trace = trs.InfiniteSequence(prefix, suffix)
    
    #print(trace.get_prefix() )
    #print(trace.get_suffix() )
    
    #=============
    # FTS Sim
    #=============
    fts_sim = trs.FiniteTransitionSystemSimulation(execution, trace)
    
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
    input_word = trs.InfiniteWord(prefix, suffix)
    
    prefix = ['s1', 's2', 's3']
    suffix = ['s3', 's2', 's3']
    run = trs.InfiniteSequence(prefix, suffix)
    
    aut_sim = trs.FiniteStateAutomatonSimulation(input_word, run)
    print(aut_sim)

def fts_maximal_example():
    """Finite-Transition System demo."""
    
    print(hl +'\nClosed FTS   -    Example 2.2, p.21 [Baier]\n' +hl)
    fts = trs.FiniteTransitionSystem(name='Beverage vending machine')
    
    # add state info
    fts.states.add('pay')
    fts.states.remove('pay')
    
    fts.states.add_from({'pay', 'soda', 'select', 'beer'} )
    fts.states.remove_from({'pay', 'soda'} )
    fts.states.add_from({'pay', 'soda'} )
    
    fts.states.set_current(['pay'])
    
    fts.states.initial.add('pay') # should already be a state
    fts.states.initial |= {'soda', 'select'}
    fts.states.initial.remove('soda')
    
    fts.states.add('water')
    fts.states.initial.add('water')
    fts.states.initial -= {'water', 'select'}
    
    fts.states.check() # sanity
    
    # no transitions yet...
    pre = fts.states.pre({'pay'} )
    post = fts.states.post({'pay'} )
    print("Pre('pay') = " +str(pre) )
    print("Post('pay') = " +str(post) )
    
    try:
        fts.states.initial.add('not pay')
    except:
        warnings.warn('You cannot add an initial state \\notin states.')
    
    # get state info
    print('States:\n\t' +str(fts.states() ) )
    print('Number of states:\n\t' +str(len(fts.states) ) )
    print('Initial states:\n\t' +str(fts.states.initial) )
    print('Number of initial states:\n\t' +str(len(fts.states.initial) ) )
    print('Current state:\n\t' +str(fts.states.current) )
    
    print("Is 'pay' a state ?\n\t" +str('pay' in fts.states) )
    print("Is 'not pay' a state ?\n\t" +str('not pay' in fts.states() ) )
    print("Is 'bla' a state ?\n\t" +str('bla' in fts) )
    print('')
    
    fts.plot()
    
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
    print('\n' +10*'-' +'\nPost sets are:\n')
    for state in fts.states():
        post = fts.states.post({state} )
        # same as:
        # post = fts.states.post_signle(state)
        
        print('\tof state: ' +str(state) +', the states: ' +str(post) )
    print(10*'-' +'\n')
    
    # same thing
    post_all = fts.states.post(fts.states() )
    print('Same thing as above:\n\t' +str(post_all) +'\n' +10*'-' +'\n')
    
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
    print('Number of transitions:\n\t' +str(len(fts.transitions) ) +'\n')
    
    fts.plot()
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
    fts.plot()
    fts.transitions.add_labeled('pay', 'select', 'insert_coin')
    fts.plot()
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
    fts.plot()
    fts.transitions.add_labeled('pay', 'new state', 'new action', check=False)
    fts.plot()
    fts.states.remove('new state')
    fts.actions.remove('new action')
    fts.plot()
    
    fts.transitions.add_labeled('pay', 'select', 'insert_coin')
    fts.transitions.remove_from({'select'}, {'soda', 'beer'} )
    fts.transitions.add_labeled_from({'select'}, {'soda', 'beer'}, '')
    fts.transitions.label('soda', 'pay', 'get_soda')
    fts.transitions.label('beer', 'pay', 'get_be oops mistake', check_label=False)
    fts.transitions.relabel('beer', 'pay', 'get_be oops mistake', 'get_beer')
    fts.plot()
    
    fts.actions.remove('get_be oops mistake') # checks that it is not used by edges
    fts.plot()
    
    try:
        fts.transitions.add_labeled('c12', 'c13', 'insert_coin')
    except:
        print('First add states, then you can add labeled transition between them.')
    fts.transitions.add_labeled('c12', 'c13', 'insert_coin', check=False)
    fts.states.remove_from({'c12', 'c13'} ) # undo
    
    print('Types of actions: ' +str(fts._transition_label_def.keys() ) )
    print('Number of actions: ' +str(len(fts.actions) ) )
    print('Actions: ' +str(fts.actions ) )
    print('Labeled transitions: ' +str(fts.transitions() ) )
    fts.plot()
    
    # fast way to get all edges with value of actions
    nx.get_edge_attributes(fts, 'actions')
    
    # Atomic Propositions (AP)
    fts.atomic_propositions.add('paid')
    fts.atomic_propositions.add_from({'', 'drink'} )
    
    fts.states.label('pay', {''})
    fts.states.labels({'soda', 'beer'}, {'paid', 'drink'} )
    fts.states.label('select', {'paid'} )
    
    # no checking of state
    fts.states.label('new state', {'paid'}, check=False)
    fts.states.remove('new state')
    
    # same thing, now 'hihi' also added
    fts.states.label('new state', {'hihi'}, check=False)
    fts.states.remove('new state')
    fts.atomic_propositions.remove('hihi')
    fts.plot()
    
    # export
    print(hl +'\n CAUTION: Saving DOT, PDF files\n' +hl +'\n')
    path = './test_fts'
    dot_fname = path +'.dot'
    pdf_fname = path +'.pdf'
    
    if not fts.plot() and save_fig:
        fts.save(pdf_fname)
        #fts.save(dot_fname, 'dot')
        
def ba_maximal_example():
    """Buchi Automaton demo."""
    
    print(hl +'\nBuchi Automaton\n' +hl)
    ba = trs.BuchiAutomaton(atomic_proposition_based=True)
    
    ba.states.add('q0')
    ba.states.add_from({'q1', 'q2', 'q3'}, destroy_order=True)
    
    ba.states.initial.add('q0')
    ba.plot()
    
    ba.alphabet.math_set.add(['paid'] )
    ba.alphabet.math_set |= ['drink', 'paid', '']
    
    print('Number of letters: ' +str(len(ba.alphabet) ) +'\n')
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
    ba.plot()
    
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
    
    if not ba.plot() and save_fig:
        ba.save(pdf_fname)
        #ba.save(dot_fname, 'dot')
    
    return ba

def mealy_machine_example():
    import numpy as np   
    
    class check_diaphragm():
        """camera f-number."""
        def is_valid_value(x):
            if x <= 0.7 or x > 256:
                raise TypeError('This f-# is outside allowable range.')
        
        def is_valid_guard(self, guard):
            # when properly implemented, do appropriate syntactic check
            if isinstance(guard, float):
                return True
            
            return False
        
        def contains(guard_set, input_port_value):
            """This method "knows" that we are using x to denote the input
            within guards."""
            guard_var_def = {'x': input_port_value}
            guard_value = eval(guard_set, guard_var_def)
            
            if not isinstance(guard_value, bool):
                raise TypeError('Guard value is non-boolean.\n'
                                'A guard is a predicate, '
                                'so it can take only boolean values.')
    
    class check_camera():
        """is it looking upwards ?"""
        def is_valid_value(self, x):
            if x.shape != (3,):
                raise Exception('Not a 3d vector!')
        
        def is_valid_guard(self, guard):
            # when properly implemented, do appropriate syntactic check
            if isinstance(guard, np.ndarray) and guard.shape == (3,):
                return True
            
            return False
        
        def contains(self, guard_set, input_port_value):
            v1 = guard_set # guard_halfspace_normal_vector
            v2 = input_port_value # camera_direction_vector
            
            if np.inner(v1, v2) > 0.8:
                return True
            
            return False
    
    m = trs.MealyMachine()
    
    m.states.add('s0')
    m.states.add_from(['s1', 's2'])
    m.states.initial.add('s0')
    
    # note: guards are conjunctions, any disjunction is represented by 2 edges
    
    # input defs
    inputs = OrderedDict([
        ['speed', {'zero', 'low', 'high', 'crazy'} ],
        ['seats', trs.PowerSet(range(5) ) ],
        ['aperture', check_diaphragm() ],
        ['camera', check_camera() ]
    ])
    m.add_inputs(inputs)
    
    # outputs def
    outputs = OrderedDict([
        ['photo', {'capture', 'wait'} ]
    ])
    m.add_outputs(outputs)
    
    # state variables def
    state_vars = outputs = OrderedDict([
        ['light', {'on', 'off'} ]
    ])
    m.add_state_vars(state_vars)
    
    # guard defined using input sub-label ordering
    guard = ('low', (0, 1), 0.3, np.array([0,0,1]), 'capture')
    m.transitions.add_labeled('s0', 's1', guard)
    
    # guard defined using input sub-label names
    guard = {'camera':np.array([1,1,1]),
             'speed':'high',
             'aperture':0.3,
             'seats':(2, 3),
             'photo':'wait'}
    m.transitions.add_labeled('s1', 's2', guard)
    
    m.plot()
    
    return m

def scipy_sparse_labeled_adj():
    from scipy.sparse import lil_matrix
    from numpy.random import rand
    
    A = lil_matrix((10, 10))
    A[0, :3] = rand(3)
    
    print(A)
    
    ofts = trs.OpenFTS()
    
    # note: to avoid first defining actions, pass arg check_labels=False
    ofts.sys_actions.add('move')
    ofts.env_actions.add('rain')
    labels = ('move', 'rain')
    ofts.transitions.add_labeled_adj(A, labels)
    #ofts.transitions.add_labeled_adj(A, labels, check_labels=False)
    
    print(30*'-' +'\n'+
          'Successfully added adjacency matrix transitions on guard: '
          +str(labels) +', to open FTS.\n' +30*'-' +'result:\n')
    ofts.plot()
    
    return ofts

def label_per_state():
    """Add states with (possibly) different AP labels each."""
    
    fts = trs.FTS()
    fts.states.add_from(['s0', 's1'] )
    fts.atomic_propositions.add_from(['p', '!p'] )
    fts.states.labels(['s0', 's1'], [{'p'}, {'!p'}] )
    fts.plot()
    
    fts = trs.FTS()
    fts.states.labels(['s0', 's1'], [{'p'}, {'!p'}], check=False)
    fts.plot()
    
    fts = trs.FTS()
    fts.states.labels([1, 2], [{'p'}, {'!p'}], check=False)
    fts.plot()
    
    fts = trs.FTS()
    fts.states.labels(range(2), [{'p'}, {'!p'}], check=False)
    fts.plot()
    
    fts = trs.FTS()
    fts.states.labels('create', [{'p'}, {'!p'}] )
    fts.plot()

if __name__ == '__main__':
    #sims_demo()
    fts_maximal_example()
    #ofts_maximal_example()
    ba_maximal_example()
    
    ofts = scipy_sparse_labeled_adj()
    label_per_state()
    m = mealy_machine_example()
    
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
Transition System module usage small examples
"""

import tulip.transys as ts

save_fig = False

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
    
    if not fts.plot() and save_fig:
        fts.save(path='small_fts.png')
    
    return fts

def ofts_minimal_example():
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
    pdf_fname = path +'.pdf'
    
    if not ofts.plot() and save_fig:
        ofts.save(path=pdf_fname)

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
    
    if not ba.plot() and save_fig:
        ba.save(path='small_ba.png')
    
    return ba

if __name__ == '__main__':
    print('Intended to be run within IPython.\n'
          +'If no plots appear, change save_fig = True, '
          +'to save them to files instead.')
    
    fts = fts_minimal_example()
    ofts_minimal_example()
    ba = ba_minimal_example()
    
    (prod_fts, final_states_preimage) = fts *ba
    prod_ba = ba *fts
    
    if not prod_fts.plot() and save_fig:
        prod_fts.save(path='prod.png')
    
    if not prod_ba.plot() and save_fig:
        prod_ba.save(path='prod.png')

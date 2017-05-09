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
from __future__ import print_function

import tulip.transys as trs

hl = 60*'='
save_fig = False

def fts_minimal_example():
    """Small example, for more see the maximal example."""

    fts = trs.FTS()
    fts.states.add_from(['s0', 's1'] )
    fts.states.initial.add('s0')

    fts.atomic_propositions.add_from({'green', 'not_green'})
    fts.states.add('s0', ap={'not_green'})
    fts.states.add('s1', ap={'green'})

    fts.transitions.add('s0', 's1')
    fts.transitions.add('s1', 's0')

    if not fts.plot() and save_fig:
        fts.save('small_fts.png')

    return fts

def ofts_minimal_example():
    """Open FTS demo."""
    msg = hl +'\nOpen FTS\n' +hl
    print(msg)

    ofts = trs.FiniteTransitionSystem()

    ofts.states.add_from(['s1', 's2', 's3'] )
    ofts.states.initial.add('s1')

    ofts.atomic_propositions |= ['p']
    ofts.states.add('s1', ap={'p'})
    ofts.states.add('s2', ap=set() )
    ofts.states.add('s3', ap={'p'})

    ofts.transitions.add('s1', 's2') # unlabeled

    ofts.sys_actions.add('try')
    ofts.sys_actions.add_from({'start', 'stop'} )
    ofts.env_actions.add_from({'block', 'wait'} )

    print(ofts.sys_actions)
    print(ofts.env_actions)

    # remove unlabeled edge s1->s2
    ofts.transitions.remove('s1', 's2')

    ofts.transitions.add(
        's1', 's2',
        sys_actions='try', env_actions='block'
    )
    ofts.transitions.add(
        's2', 's3',
        sys_actions='start', env_actions='wait'
    )
    ofts.transitions.add(
        's3', 's2',
        sys_actions='stop', env_actions='block'
    )

    print('The Open TS now looks like:')
    print(ofts.transitions() )

    ofts.atomic_propositions |= {'home', 'lot', 'p1'}

    print(ofts)

    path = './test_ofts'
    pdf_fname = path +'.pdf'

    if not ofts.plot() and save_fig:
        ofts.save(pdf_fname)

    return ofts

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

    msg = hl +'\nBuchi Automaton (small example):    '
    msg += 'Example 4.64, p.202 [Baier]\n' +hl
    print(msg)

    ba = trs.BuchiAutomaton(atomic_proposition_based=True)
    ba.states.add_from({'q0', 'q1', 'q2'})
    ba.states.initial.add('q0')
    ba.states.accepting.add('q1')

    ba.alphabet.math_set |= [True, 'green', 'not_green']

    ba.transitions.add('q0', 'q0', letter={True})
    ba.transitions.add('q0', 'q1', letter={'not_green'})
    ba.transitions.add('q1', 'q1', letter={'not_green'})
    ba.transitions.add('q1', 'q2', letter={'green'})
    ba.transitions.add('q2', 'q2', letter={True})

    if not ba.plot() and save_fig:
        ba.save('small_ba.png')

    return ba

def merge_example():
    """Merge two small FT Systems.
    """
    n = 4
    L = n*['p']
    ts1 = trs.line_labeled_with(L, n-1)

    ts1.transitions.remove('s3', 's4')
    ts1.transitions.remove('s5', 's6')

    ts1.sys_actions |= ['step', 'jump']
    ts1.transitions.add('s3', 's4', sys_actions='step')
    ts1.transitions.add('s5', 's6', sys_actions='jump')

    ts1.plot()

    L = n*['p']
    ts2 = trs.cycle_labeled_with(L)
    ts2.states.add('s3', ap=set())

    ts2.transitions.remove('s0', 's1')
    ts2.transitions.remove('s1', 's2')

    ts2.sys_actions |= ['up', 'down']
    ts2.transitions.add('s0', 's1', sys_actions='up')
    ts2.transitions.add('s1', 's2', sys_actions='down')

    ts2.plot()

    return ts2

if __name__ == '__main__':
    print('Intended to be run within IPython.\n'
          +'If no plots appear, change save_fig = True, '
          +'to save them to files instead.')

    fts = fts_minimal_example()
    ofts = ofts_minimal_example()
    ba = ba_minimal_example()

    merger_ts = merge_example()

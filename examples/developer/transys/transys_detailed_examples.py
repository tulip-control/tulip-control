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
"""Transition System module developer examples."""
import networkx as nx
from numpy.random import rand
from scipy.sparse import lil_matrix
import tulip.transys as trs
import warnings


hl = 60*'='
save_fig = False


def fts_maximal_example():
    """Finite-Transition System demo."""
    print(f'{hl}\nClosed FTS   -    Example 2.2, p.21 [Baier]\n{hl}')
    fts = trs.FiniteTransitionSystem()
    fts.name = 'Beverage vending machine'
    # add state info
    fts.states.add('pay')
    fts.states.remove('pay')
    fts.states.add_from({'pay', 'soda', 'select', 'beer'})
    fts.states.remove_from({'pay', 'soda'})
    fts.states.add_from({'pay', 'soda'})
    fts.states.current = ['pay']
    fts.states.initial.add('pay')
        # should already be a state
    fts.states.initial |= {'soda', 'select'}
    fts.states.initial.remove('soda')
    fts.states.add('water')
    fts.states.initial.add('water')
    fts.states.initial -= {'water', 'select'}
    # no transitions yet...
    pre = fts.states.pre({'pay'})
    post = fts.states.post({'pay'})
    print(f"Pre('pay') = {pre}")
    print(f"Post('pay') = {post}")
    try:
        fts.states.initial.add('not pay')
    except:
        warnings.warn(
            'You cannot add an initial state \\notin states.')
    # get state info
    print(f'States:\n\t{fts.states()}')
    print(f'Number of states:\n\t{len(fts.states)}')
    print(f'Initial states:\n\t{fts.states.initial}')
    print(f'Number of initial states:\n\t{len(fts.states.initial)}')
    print(f'Current state:\n\t{fts.states.current}')
    yes = 'pay' in fts.states
    print(f"Is 'pay' a state ?\n\t{yes}")
    yes = 'not pay' in fts.states()
    print(f"Is 'not pay' a state ?\n\t{yes}")
    yes = 'bla' in fts
    print(f"Is 'bla' a state ?\n\t{yes}")
    print('')
    fts.plot()
    # add transition info (unlabeled)
    fts.transitions.add('pay', 'select') # notice: no labels
    fts.transitions.add_from(
        [('select', x) for x in {'soda', 'beer'}])
    fts.transitions.add_from(
        [('soda', x) for x in {'beer', 'pay'}])
    fts.transitions.add('pay', 'soda')
    fts.transitions.remove('pay', 'soda')
    pre = fts.states.pre({'pay'} )
    post = fts.states.post({'pay'} )
    print(f"Pre('pay') =\n\t{pre}")
    print(f"Post('pay') =\n\t{post}\n")
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
    pre = fts.states.pre('pay')
    # all 1-hop post sets
    print('\n' + 10 * '-' + '\nPost sets are:\n')
    for state in fts.states():
        post = fts.states.post({state} )
        # same as:
        # post = fts.states.post_signle(state)
        print(f'\tof state: {state}, the states: {post}')
    print(10*'-' +'\n')
    # same thing
    post_all = fts.states.post(fts.states() )
    print(f'Same thing as above:\n\t{post_all}\n' +10*'-' +'\n')
    try:
        fts.transitions.add('pay', 'not yet a state')
    except:
        print('First add from_, to_ states, then you can add transition.\n')
    try:
        fts.transitions.add_from({'not a state', 'also not a state'}, {'pay'} )
    except:
        print('Same state check as above.\n')
    # get transition info (unlabeled)
    print(f'Transitions:\n\t{fts.transitions()}')
    print(f'Number of transitions:\n\t{len(fts.transitions)}\n')
    fts.plot()
    print(fts) # pretty
    # ----------------------------------------
    # CAUTION: labeling now considered
    # ----------------------------------------
    #
    # MultiDiGraph labeling issue vs multiple same edges resolved as follows:
    #   transition addition is strictly monitored
    #   each transition is identified uniquely with its set of labels (key=values)
    #   see Transitions.add_labeled()
    fts.sys_actions.add('insert_coin')
    fts.sys_actions.add_from({'get_soda', 'get_beer', ''} )
    try:
        fts.transitions.add_labeled('pay', 'select', 'insert_coin')
    except:
        print('Checks not pre-existing same labeled or unlabeled')
    # checking pre-existing CANNOT be overriden,
    # to preserve function semantics
    # check=False used only to add missing states or labels
    #
    # first remove unlabeled, then add new labeled
    fts.transitions.remove('pay', 'select')
    fts.plot()
    fts.transitions.add('pay', 'select', sys_actions='insert_coin')
    fts.plot()
    fts.transitions.remove('pay', 'select', sys_actions='insert_coin')
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
    fts.states.add_from({'pay', 'new_state'})
    fts.sys_actions.add('new_action')
    fts.transitions.add('pay', 'new_state', sys_actions='new_action')
    fts.plot()
    fts.states.remove('new_state')
    fts.sys_actions.remove('new_action')
    fts.plot()
    fts.transitions.add('pay', 'select', sys_actions='insert_coin')
    fts.transitions.remove_from(
        [('select', x) for x in {'soda', 'beer'}])
    fts.transitions.add_from(
        [('select', x) for x in {'soda', 'beer'}])
    fts.transitions.remove('soda', 'pay')
    fts.transitions.add('soda', 'pay', sys_actions='get_soda')
    fts.plot()
    print(f'Types of actions: {list(fts._transition_label_def.keys())}')
    print(f'Number of actions: {len(fts.sys_actions)}')
    print(f'Actions: {fts.sys_actions}')
    print(f'Labeled transitions: {fts.transitions()}')
    fts.plot()
    # fast way to get all edges with value of actions
    nx.get_edge_attributes(fts, 'sys_actions')
    # Atomic Propositions (AP)
    fts.atomic_propositions.add('paid')
    fts.atomic_propositions.add_from({'', 'drink'} )
    fts.states.add('pay')
    nodes = {'soda', 'beer'}
    labels = {'paid', 'drink'}
    for node, label in zip(nodes, labels):
        fts.add_node(node, ap={label})
    fts.states.add('select', ap={'paid'})
    fts.plot()
    # export
    print(f'{hl}\n CAUTION: Saving DOT, PDF files\n{hl}\n')
    path = './test_fts'
    dot_fname = f'{path}.dot'
    pdf_fname = f'{path}.pdf'
    if not fts.plot() and save_fig:
        fts.save(pdf_fname)
        #fts.save(dot_fname, 'dot')


def ba_maximal_example():
    """Buchi Automaton demo."""
    print(hl +'\nBuchi Automaton\n' +hl)
    ba = trs.BuchiAutomaton(atomic_proposition_based=True)
    ba.states.add('q0')
    ba.states.add_from({'q1', 'q2', 'q3'})
    ba.states.initial.add('q0')
    ba.plot()
    # alternatives to add props
    ba.alphabet.math_set.add(['paid'] )
    ba.atomic_propositions |= {'drink', 'paid'}
    print(f'Number of letters: {len(ba.alphabet)}\n')
    print(f'Alphabet: {ba.alphabet}\n')
    try:
        ba.transitions.add_labeled('q1', 'q10', {'paid'} )
    except:
        print('q10 not a state.')
    ba.transitions.add('q0', 'q1', letter=set())
    ba.transitions.add('q0', 'q1', letter={'paid'})
    ba.transitions.add('q1', 'q2', letter={'paid', 'drink'})
    ba.transitions.add('q3', 'q0', letter=set())
    ba.transitions.add('q1', 'q3', letter={'drink'})
    ba.plot()
    # accepting states
    ba.accepting.add('q1')
    ba.accepting |= {'q2', 'q3'}
    ba.accepting.remove('q2')
    ba.accepting.remove('q3')
    print('Number of accepting states:\n\t'
          f'{len(ba.accepting)}\n')
    print(f'Accepting states:\n\t{ba.accepting}\n')
    print(ba)
    path = './test_ba'
    dot_fname = f'{path}.dot'
    pdf_fname = f'{path}.pdf'
    if not ba.plot() and save_fig:
        ba.save(pdf_fname)
        #ba.save(dot_fname, 'dot')
    return ba


def scipy_sparse_labeled_adj():
    n = 10
    A = lil_matrix((n, n))
    A[0, :3] = rand(3)
    adj2states = list(range(n))
    print(A)
    ofts = trs.FTS()
    ofts.states.add_from(set(range(10)))
    ofts.sys_actions.add('move')
    ofts.env_actions.add('rain')
    ofts.transitions.add_adj(
        A, adj2states,
        sys_actions='move',
        env_actions='rain')
    ofts.plot()
    """same thing as above, using A as a submatrix instead
    """
    A = lil_matrix((3, 3))
    A[0, :3] = rand(3)
    adj2states = [0, 1, 2]
    print(A)
    ofts = trs.FTS()
    ofts.states.add_from(set(range(10)))
    ofts.sys_actions.add('move')
    ofts.env_actions.add('rain')
    ofts.transitions.add_adj(
        A, adj2states,
        sys_actions='move',
        env_actions='rain')
    ofts.plot()
    return ofts


def label_per_state():
    """Add states with (possibly) different AP labels each."""
    fts = trs.FTS()
    fts.states.add_from(['s0', 's1'] )
    fts.atomic_propositions.add('p')
    fts.states.add('s0', ap={'p'})
    fts.states.add('s1', ap=set())
    fts.plot()


if __name__ == '__main__':
    fts_maximal_example()
    #ofts_maximal_example()
    ba_maximal_example()
    ofts = scipy_sparse_labeled_adj()
    label_per_state()

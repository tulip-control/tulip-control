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
"""Convert state graphs to Promela."""
import collections.abc as _abc
import time

import tulip.transys as _trs


def fts2promela(
        graph:
            '_trs.FiniteTransitionSystem',
        procname:
            str |
            None=None
        ) -> str:
    """Convert (possibly labeled) state graph to Promela str.

    Creates a process which can be simulated as an independent
    thread in the SPIN model checker.
    If atomic propositions label states,
    then they are exported as bit variables.

    The state graph is exported to Promela using goto statements.
    Goto statements avoid introducing additional states.
    Using structured constructs (if, do) can create
    extra states in certain cases.

    The structured program theorem ensures that avoiding goto
    statements is always possible.
    But this yields an equivalent algorithm only,
    not an equivalent unfolding of the program graph.

    Therefore verified properties can be affected,
    especially if intermediate states are introduced in
    one of multiple processes.

    @param procname:
        Promela process name (after proctype)
        (default: system name)
    """
    def state_ap2promela(
            state,
            graph,
            ap_alphabet:
                _abc.Iterable
            ) -> str:
        ap_label = get_label_of(state, graph)
        s = ''
        for prop in ap_alphabet:
            if prop is True:
                continue
            if prop in ap_label:
                s += f'\t\t {prop} = 1;\n'
            else:
                s += f'\t\t {prop} = 0;\n'
        s += f'\t\t printf("State: {state}\\n");\n'
        s += '\t\n'
        return s

    def trans2promela(
            transitions:
                _abc.Iterable,
            graph,
            ap_alphabet:
                _abc.Iterable
            ) -> str:
        s = '\t if\n'
        for from_state, to_state, sublabels_dict in transitions:
            s += (
                '\t :: atomic{\n'
                f'\t\t printf("{sublabels_dict}\\n");\n'
                f'{state_ap2promela(to_state, graph, ap_alphabet)}'
                f'\t\t goto {to_state}\n'
                '\t }\n')
        s += '\t fi;\n\n'
        return s
    def get_label_of(state, graph):
        state_label_pairs = graph.states.find([state])
        state_, ap_label = state_label_pairs[0]
        print(f'state:\t{state}')
        print(f'ap label:\t{ap_label}')
        return ap_label['ap']
    if procname is None:
        procname = graph.name
    s = '/*\n * Promela file generated with TuLiP\n'
    tm = time.strftime('%x %X %z')
    s += f' * Date: {tm}\n */\n\n'
    for ap in graph.atomic_propositions:
        # convention "!" means negation
        if ap not in {None, True}:
            s += f'bool {ap};\n'
    s += f'\nactive proctype {procname}(){{\n'
    s += '\t if\n'
    for initial_state in graph.states.initial:
        s += f'\t :: goto {initial_state}\n'
    s += '\t fi;\n'
    ap_alphabet = graph.atomic_propositions
    for state in graph.states():
        out_transitions = graph.transitions.find(
            {state},
            as_dict=True)
        s += str(state).replace(' ', '_') + ':'
        s += trans2promela(
            out_transitions, graph,
            ap_alphabet)
    s += '}\n'
    return s


# def mealy2promela():
#     """Convert Mealy machine to Promela str."""

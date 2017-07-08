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
Convert state graphs to promela
"""
from __future__ import print_function

from time import strftime

def fts2promela(graph, procname=None):
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

    @param graph: networkx

    @param procname: Promela process name (after proctype)
    @type procname: str (default: system name)
    """
    def state_ap2promela(state, graph, ap_alphabet):
        ap_label = get_label_of(state, graph)
        s = ''
        for prop in ap_alphabet:
            if prop is True:
                continue

            if prop in ap_label:
                s += '\t\t ' +str(prop) +' = 1;\n'
            else:
                s += '\t\t ' +str(prop) +' = 0;\n'

        s += '\t\t printf("State: ' +str(state) +'\\n");\n'
        s += '\t\n'
        return s

    def trans2promela(transitions, graph, ap_alphabet):
        s = '\t if\n'
        for (from_state, to_state, sublabels_dict) in transitions:
            s += '\t :: atomic{\n'
            s += '\t\t printf("' +str(sublabels_dict) +'\\n");\n'
            s += state_ap2promela(to_state, graph, ap_alphabet)
            s += '\t\t goto ' +str(to_state) +'\n'
            s += '\t }\n'
        s += '\t fi;\n\n'
        return s

    def get_label_of(state, graph):
        state_label_pairs = graph.states.find([state] )
        (state_, ap_label) = state_label_pairs[0]
        print('state:\t' +str(state) )
        print('ap label:\t' +str(ap_label) )
        return ap_label['ap']

    if procname is None:
        procname = graph.name

    s = '/*\n * Promela file generated with TuLiP\n'
    s += ' * Date: '+str(strftime('%x %X %z') ) +'\n */\n\n'
    for ap in graph.atomic_propositions:
        # convention "!" means negation
        if ap not in {None, True}:
            s += 'bool ' +str(ap) +';\n'

    s += '\nactive proctype ' +procname +'(){\n'

    s += '\t if\n'
    for initial_state in graph.states.initial:
        s += '\t :: goto ' +str(initial_state) +'\n'
    s += '\t fi;\n'

    ap_alphabet = graph.atomic_propositions
    for state in graph.states():
        out_transitions = graph.transitions.find(
            {state}, as_dict=True
        )

        s += str(state).replace(' ', '_') +':'
        s += trans2promela(out_transitions, graph,
                           ap_alphabet)

    s += '}\n'
    return s

#def mealy2promela():
#    """Convert Mealy machine to Promela str.
#    """

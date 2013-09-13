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
Convert graph to promela
"""
hl = 60 *'-'
    
def promela_str(self, procname=None):
    """Convert an automaton to Promela source code.
    
    Creats a process which can be simulated as an independent
    thread in the SPIN model checker.
    
    see also
    --------
    save, plot
    
    @param fname: file name
    @type fname: str
    
    @param procname: Promela process name, i.e., proctype procname())
    @type procname: str (default: system's name)
    
    @param add_missing_extension: add file extension 'pml', if missing
    @type add_missing_extension: bool
    """
    def state2promela(state, ap_label, ap_alphabet):
        s1 = str(state) +':\n'
        
        s2 = '\t :: atomic{\n'
        s2 += '\t\t\t printf("State: ' +str(state) +'\\n");\n'
        
        # convention ! means negation
        
        missing_props = filter(lambda x: x[0] == '!', ap_label)
        present_props = ap_label.difference(missing_props)
        
        assign_props = lambda x: str(x) + ' = 1;'
        s2 += '\t\t\t '
        if present_props:
            s2 += '\n\t\t\t '.join(map(assign_props, present_props) )
        
        # rm "!"
        assign_props = lambda x: str(x[1:] ) + ' = 0;'
        if missing_props:
            s2 += '\n\t\t\t '.join(map(assign_props, missing_props) )
        
        s2 += '\n'
        return (s1, s2)
    
    def outgoing_trans2promela(transitions, s2):
        s = '\t if\n'
        for (from_state, to_state, sublabels_dict) in transitions:
            s += s2
            s += '\t\t\t printf("' +str(sublabels_dict) +'\\n");\n'
            s += '\t\t\t goto ' +str(to_state) +'\n\t\t}\n'
        s += '\t fi;\n\n'
        return s
    
    if procname is None:
        procname = self.name
    
    s = ''
    for ap in self.atomic_propositions:
        # convention "!" means negation
        if ap is not None:
            s += 'bool ' +str(ap) +';\n'
    
    s += '\nactive proctype ' +procname +'(){\n'
    
    s += '\t if\n'
    for initial_state in self.states.initial:
        s += '\t :: goto ' +str(initial_state) +'\n'
    s += '\t fi;\n'
    
    for state in self.states():
        ap_alphabet = self.atomic_propositions
        lst = self.states.find([state] )
        (state_, ap_label) = lst[0]
        (s1, s2) = state2promela(state, ap_label['ap'], ap_alphabet)
        
        s += s1
        
        outgoing_transitions = self.transitions.find(
            {state}, as_dict=True
        )
        s += outgoing_trans2promela(outgoing_transitions, s2)
    
    s += '}\n'
    return s
    
# Copyright (c) 2012, 2013 by California Institute of Technology
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
Interface to library of synthesis tools, e.g., JTLV, gr1c
"""

import os

from tulip import transys
from tulip.spec import GRSpec
from tulip import jtlvint
from tulip import gr1cint


def sys_to_spec(sys):
    if not isinstance(sys, transys.FiniteTransitionSystem):
        raise TypeError("synth.sys_to_spec only supports FiniteTransitionSystem objects")
    # Assume everything is controlled; support for an environment
    # definition is forthcoming.
    sys_vars = list(sys.aps)
    sys_vars.extend([s for s in sys.states])
    trans = []

    # Initial state, including enforcement of mutual exclusion
    init = ["("+") | (".join(["("+str(current_state)+")"+" & "+ " & ".join(["!("+str(u)+")" for u in sys.states if u != current_state]) for current_state in sys.states.initial])+")"]
    for state in sys.states.initial:
        init.append(" & ".join(["("+str(ap)+")" for ap in sys.aps if ap in sys.states.label_of(state)["ap"]]))
        if len(init[-1]) > 0:
            init[-1] += " & "
        init[-1] +=  " & ".join(["!("+str(ap)+")" for ap in sys.aps if ap not in sys.states.label_of(state)["ap"]])
        init[-1] = "("+str(state)+") -> ("+init[-1]+")"

    # Transitions
    for from_state in sys.states:
        trans.append("("+str(from_state)+") -> ("+" | ".join(["("+str(v)+"')" for (u,v,l) in sys.transitions.find(from_states=[from_state])])+")")

    # Mutual exclusion of states
    trans.append("(("+") | (".join(["("+str(current_state)+"')"+" & "+ " & ".join(["!("+str(u)+"')" for u in sys.states if u != current_state]) for current_state in sys.states])+"))")

    # Require atomic propositions to follow states according to label
    for state in sys.states:
        if sys.states.label_of(state).has_key("ap"):
            trans.append(" & ".join([str(ap)+"'" for ap in sys.aps if ap in sys.states.label_of(state)["ap"]]))
        else:
            trans.append("")
        if len(trans[-1]) > 0:
            trans[-1] += " & "
        if not sys.states.label_of(state).has_key("ap"):
            trans[-1] +=  " & ".join(["!"+str(ap)+"'" for ap in sys.aps])
        else:
            trans[-1] +=  " & ".join(["!"+str(ap)+"'" for ap in sys.aps if ap not in sys.states.label_of(state)["ap"]])
        trans[-1] = "(("+str(state)+"') -> ("+trans[-1]+"))"

    return GRSpec(sys_vars=sys_vars, sys_init=init, sys_safety=trans)


def synthesize(option, specs, sys=None):
    """Function to call the appropriate synthesis tool on the spec.

    Beware!  This function provides a generic interface to a variety
    of routines.  Being under active development, the types of
    arguments supported and types of objects returned may change
    without notice.

    @param option: Magic string that declares what tool to invoke,
        what method to use, etc.  Currently recognized forms:

          - C{"gr1c"}: use gr1c for GR(1) synthesis via L{gr1cint}.
          - C{"jtlv"}: use JTLV for GR(1) synthesis via L{jtlvint}.
    @type specs: L{spec.GRSpec}
    @param sys: NOT IMPLEMENTED YET.

    @return: Return automaton implementing the strategy, or None if
        error.
    """
    if sys is not None:
        sform = sys_to_spec(sys)
        specs = specs | sform

    if option == 'gr1c':
        ctrl = gr1cint.synthesize(specs)
    elif option == 'jtlv':
        ctrl = jtlvint.synthesize(specs)
    else:
        raise Exception('Undefined synthesis option. '+\
                        'Current options are \"jtlv\" and \"gr1c\"')
    return ctrl

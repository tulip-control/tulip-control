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
from copy import deepcopy

from tulip import transys
from tulip.spec import GRSpec
from tulip import jtlvint
from tulip import gr1cint

def _disj(set0):
    return " || ".join([
        "(" +str(x) +")"
        for x in set0
    ])

def _conj_intersection(set0, set1, parenth=True):
    if parenth:
        return " && ".join([
            "("+str(x)+")"
            for x in set0
            if x in set1
        ])
    else:
        return " && ".join([
            str(x)
            for x in set0
            if x in set1
        ])

def _conj_neg(set0, parenth=True):
    if parenth:
        return " && ".join([
            "!("+str(x)+")"
            for x in set0
        ])
    else:
        return " && ".join([
            "!"+str(x)
            for x in set0
        ])

def _conj_neg_diff(set0, set1, parenth=True):
    if parenth:
        return " && ".join([
            "!("+str(x)+")"
            for x in set0
            if x not in set1
        ])
    else:
        return " && ".join([
            "!"+str(x)
            for x in set0
            if x not in set1
        ])

def sys_to_spec(sys):
    """Convert finite transition system to GR(1) specification.
    
    Forthcoming: support for OpenFiniteTransitionSystem.
    
    @type sys: transys.FiniteTransitionSystem
    
    @rtype: GRSpec
    """
    if not isinstance(sys, (transys.FiniteTransitionSystem,
                            transys.OpenFiniteTransitionSystem)):
        raise TypeError("synth.sys_to_spec does not support " + str(type(sys)))
    
    # Assume everything is controlled; support for an environment
    # definition is forthcoming.
    sys_vars = list(sys.aps)
    sys_vars.extend([s for s in sys.states])
    trans = []

    # Initial state, including enforcement of mutual exclusion
    if (len(sys.states.initial) > 0):
        init = [_disj([
            "("+str(x)+")" +" && " +_conj_neg_diff(sys.states, [x])
            for x in sys.states.initial
        ])]
    else:
        init = ""

    for state in sys.states.initial:
        label = sys.states.label_of(state)
        
        ap_init = _conj_intersection(sys.aps, label["ap"])
        init.append(ap_init)
        
        if len(init[-1]) > 0:
            init[-1] += " && "
        
        init[-1] += _conj_neg_diff(sys.aps, label["ap"])
        init[-1] = "("+str(state)+") -> ("+init[-1]+")"

    # Transitions
    for from_state in sys.states:
        post = sys.states.post(from_state)
        
        # no successor states ?
        if not post:
            continue
        
        post_states = _disj(post)
        trans += ["(" +str(from_state) +") -> X(" +post_states +")"]

    # Mutual exclusion of states
    trans.append(
        "X("+_disj([
            "("+str(x)+")"+" && " +_conj_neg_diff(sys.states, [x])
            for x in sys.states
        ])+")"
    )

    # Require atomic propositions to follow states according to label
    for state in sys.states:
        label = sys.states.label_of(state)
        
        if label.has_key("ap"):
            tmp = _conj_intersection(sys.aps, label["ap"], parenth=False)
        else:
            tmp = ""
        
        if len(tmp) > 0:
            tmp += " && "
        
        if label.has_key("ap"):
            tmp += _conj_neg_diff(sys.aps, label["ap"], parenth=False)
        else:
            tmp += _conj_neg(sys.aps, parenth=False)
        
        trans += ["X(("+ str(state) +") -> ("+ tmp +"))"]

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
    @param sys: A transition system that should be expressed with the
        specification (spec).

    @return: If spec is realizable,
        then return a Mealy machine implementing the strategy.
        Otherwise return list of counterexamples.
    @rtype: transys.Mealy or list

    """
    if sys is not None:
        sys = deepcopy(sys)
        
        sform = sys_to_spec(sys)
        specs = specs | sform

    if option == 'gr1c':
        ctrl = gr1cint.synthesize(specs)
    elif option == 'jtlv':
        ctrl = jtlvint.synthesize(specs)
    else:
        raise Exception('Undefined synthesis option. '+\
                        'Current options are "jtlv" and "gr1c"')
    return ctrl

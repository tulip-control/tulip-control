#!/usr/bin/env python
"""Discrete synthesis from a dummy abstraction with uncontrolled switching.

This is an example to demonstrate how the output of the TuLiP discretization
for a system with uncontrollable switching (i.e., modes are controlled by the
environment) might look like.

We will assume, we have the 6 cell robot example.

     +---+---+---+
     | 3 | 4 | 5 |
     +---+---+---+
     | 0 | 1 | 2 |
     +---+---+---+
"""
# NO, 26 Jul 2013.
from __future__ import print_function

import numpy as np
from scipy import sparse as sp
from tulip import spec
from tulip import synth
from tulip import transys


###########################################
# Environment switched system with 2 modes:
###########################################

sys_swe = transys.FTS()

# We assume robots ability to transition between cells depends on the surface
# characteristics which could be slippery or normal. This is controlled by the
# environment.

sys_swe.env_actions.add_from({'slippery','normal'})
# Environment actions are mutually exclusive.

# Discretization builds a transition matrix (invisible to the end user)

n = 6
states = ['s'+str(i) for i in range(n) ]

sys_swe.atomic_propositions.add_from(['home','lot'])
state_labels = [{'home'}, set(), set(), set(), set(), {'lot'}]

# Add states and decorate TS with state labels (aka atomic propositions)
for state, label in zip(states, state_labels):
    sys_swe.states.add(state, ap=label)

# Within each mode the transitions can be deterministically chosen, environment
# chooses the mode (the surface can be slippery or normal).
transmat1 = sp.lil_matrix(np.array(
                [[1,1,0,1,0,0],
                 [1,1,1,0,1,0],
                 [0,1,1,0,1,1],
                 [1,0,0,1,1,0],
                 [0,1,0,1,1,1],
                 [0,0,1,0,1,1]]
            ))

sys_swe.transitions.add_adj(transmat1, states, env_actions='normal')

# In slippery mode, the robot can't stay still and makes larger jumps.
transmat2 = sp.lil_matrix(np.array(
                [[0,0,1,1,0,0],
                 [1,0,1,0,1,0],
                 [1,0,0,0,1,1],
                 [1,0,0,0,0,1],
                 [0,1,0,1,0,1],
                 [0,0,1,1,0,0]]
            ))

sys_swe.transitions.add_adj(transmat2, states, env_actions='slippery')

# This is what is visible to the outside world (and will go into synthesis method)
print(sys_swe)

#
# Environment variables and specification
#
# The environment can issue a park signal that the robot just respond
# to by moving to the lower left corner of the grid.  We assume that
# the park signal is turned off infinitely often.
#
env_vars = {'park'}
env_init = set()                # empty set
env_prog = {'!park'}
env_safe = set()                # empty set

#
# System specification
#
# The system specification is that the robot should repeatedly revisit
# the upper right corner of the grid while at the same time responding
# to the park signal by visiting the lower left corner.  The LTL
# specification is given by
#
#     []<> home && [](park -> <>lot)
#
# Since this specification is not in GR(1) form, we introduce the
# variable X0reach that is initialized to True and the specification
# [](park -> <>lot) becomes
#
#     [](X (X0reach) <-> lot || (X0reach && !park))
#

# Augment the environmental description to make it GR(1)
#! TODO: create a function to convert this type of spec automatically

# Define the specification
#! NOTE: maybe "synthesize" should infer the atomic proposition from the
# transition system? Or, we can declare the mode variable, and the values
# of the mode variable are read from the transition system.
sys_vars = {'X0reach'}
sys_init = {'X0reach'}
sys_prog = {'home'}               # []<>home
sys_safe = {'(X (X0reach) <-> lot) || (X0reach && !park)'}
sys_prog |= {'X0reach'}

# Create the specification
specs = spec.GRSpec(env_vars, sys_vars, env_init, sys_init,
                    env_safe, sys_safe, env_prog, sys_prog)
# controller decides based on current values `env_vars, sys_vars`
# and next values `env_vars'`. A controller with this
# information flow is known as Mealy.
specs.moore = False
specs.qinit = '\A \E'

# Controller synthesis
#
# At this point we can synthesize the controller using one of the available
# methods.
#
ctrl = synth.synthesize(specs, sys=sys_swe, ignore_sys_init=True)
assert ctrl is not None, 'unrealizable'

# @plot_print@
if not ctrl.save('environment_switching.png'):
    print(ctrl)
# @plot_print_end@

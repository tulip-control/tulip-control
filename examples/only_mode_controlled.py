#!/usr/bin/env python
"""Discrete synthesis from a dummy abstraction of switched dynamics.

This is an example to demonstrate how the output of abstracting a switched
system, where the only control over the dynamics is through mode switching,
might look like.

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

###############################
# Switched system with 4 modes:
###############################

# In this scenario we have limited actions "left, right, up, down" with
# uncertain (nondeterministics) outcomes (e.g., due to bad actuators or
# bad low-level feedback controllers)

# Only control over the dynamics is through mode switching
# Transitions should be interpreted as nondeterministic

# Create a finite transition system
env_sws = transys.FTS()
env_sws.owner = 'env'

env_sws.sys_actions.add_from({'right','up','left','down'})

# str states
n = 6
states = transys.prepend_with(range(n), 's')
env_sws.states.add_from(set(states) )
env_sws.states.initial.add('s0')

env_sws.atomic_propositions.add_from(['home','lot'])
state_labels = [{'home'}, set(), set(), set(), set(), {'lot'}]

# Add states and decorate TS with state labels (aka atomic propositions)
for state, label in zip(states, state_labels):
    env_sws.states.add(state, ap=label)

# mode1 transitions
transmat1 = np.array([[0,1,0,0,1,0],
                      [0,0,1,0,0,1],
                      [0,0,1,0,0,0],
                      [0,1,0,0,1,0],
                      [0,0,1,0,0,1],
                      [0,0,0,0,0,1]])
env_sws.transitions.add_adj(
    sp.lil_matrix(transmat1), states, sys_actions='right'
)

# mode2 transitions
transmat2 = np.array([[0,0,0,1,0,0],
                      [0,0,0,0,1,1],
                      [0,0,0,0,0,1],
                      [0,0,0,1,0,0],
                      [0,0,0,0,1,0],
                      [0,0,0,0,0,1]])
env_sws.transitions.add_adj(
    sp.lil_matrix(transmat2), states, sys_actions='up'
)

# mode3 transitions
transmat3 = np.array([[1,0,0,0,0,0],
                      [1,0,0,1,0,0],
                      [0,1,0,0,1,0],
                      [0,0,0,1,0,0],
                      [1,0,0,1,0,0],
                      [0,1,0,0,1,0]])
env_sws.transitions.add_adj(
    sp.lil_matrix(transmat3), states, sys_actions='left'
)

# mode4 transitions
transmat4 = np.array([[1,0,0,0,0,0],
                      [0,1,0,0,0,0],
                      [0,0,1,0,0,0],
                      [1,0,0,0,0,0],
                      [0,1,1,0,0,0],
                      [0,0,1,0,0,0]])
env_sws.transitions.add_adj(
    sp.lil_matrix(transmat4), states, sys_actions='down'
)

# This is what is visible to the outside world (and will go into synthesis method)
print(env_sws)

#
# Environment variables and specification
#
# The environment can issue a park signal that the robot just respond
# to by moving to the lower left corner of the grid.  We assume that
# the park signal is turned off infinitely often.
#
env_vars = {'park'}
env_init = {'park'}
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
sys_init = {'X0reach','sys_actions = "right"'}
sys_prog = {'home'}               # []<>home
sys_safe = {'(X (X0reach) <-> lot) || (X0reach && !park)'}
sys_prog |= {'X0reach'}

# Create the specification
specs = spec.GRSpec(env_vars, sys_vars, env_init, sys_init,
                    env_safe, sys_safe, env_prog, sys_prog)
specs.qinit = '\E \A'

# Controller synthesis
#
# At this point we can synthesize the controller using one of the available
# methods.
#
ctrl = synth.synthesize(specs, env=env_sws)
assert ctrl is not None, 'unrealizable'

# Generate a graphical representation of the controller for viewing
if not ctrl.save('only_mode_controlled.png'):
    print(ctrl)

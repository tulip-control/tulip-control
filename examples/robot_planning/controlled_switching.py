# WARNING: This example may not yet be working.  Please check again in
#          the upcoming release.
#
# This is an example to demonstrate how the output of abstracting a switched
# system, where the dynamics are controlled through switching and
# if multiple transitions are possible from a state in some mode,
# then the system controls which one is taken.

# NO, 26 Jul 2013.

# We will assume, we have the 6 cell robot example.

#
#     +---+---+---+
#     | 3 | 4 | 5 |
#     +---+---+---+
#     | 0 | 1 | 2 |
#     +---+---+---+
#

from tulip import spec, synth, transys
import numpy as np
from scipy import sparse as sp


###############################
# Switched system with 4 modes:
###############################

# In this scenario we have limited actions "left, right, up, down" with 
# certain (nondeterministic) outcomes

# Create a finite transition system
sys_sws = transys.FTS()

sys_sws.sys_actions.add_from({'right','up','left','down'})

# str states
n = 6
states = transys.prepend_with(range(n), 's')
sys_sws.states.add_from(set(states) )
sys_sws.states.initial.add_from({'s0', 's3'})

sys_sws.atomic_propositions.add_from(['home','lot'])
state_labels = [{'home'}, set(), set(), set(), set(), {'lot'}]

# Add states and decorate TS with state labels (aka atomic propositions)
for state, label in zip(states, state_labels):
    sys_sws.states.add(state, ap=label)

# mode1 transitions
transmat1 = np.array([[0,1,0,0,1,0],
                      [0,0,1,0,0,1],
                      [0,0,1,0,0,0],
                      [0,1,0,0,1,0],
                      [0,0,1,0,0,1],
                      [0,0,0,0,0,1]])
sys_sws.transitions.add_adj(
    sp.lil_matrix(transmat1), states, sys_actions='right'
)
                      
# mode2 transitions
transmat2 = np.array([[0,0,0,1,0,0],
                      [0,0,0,0,1,1],
                      [0,0,0,0,0,1],
                      [0,0,0,1,0,0],
                      [0,0,0,0,1,0],
                      [0,0,0,0,0,1]])
sys_sws.transitions.add_adj(
    sp.lil_matrix(transmat2), states, sys_actions='up'
)
                      
# mode3 transitions
transmat3 = np.array([[1,0,0,0,0,0],
                      [1,0,0,1,0,0],
                      [0,1,0,0,1,0],
                      [0,0,0,1,0,0],
                      [1,0,0,1,0,0],
                      [0,1,0,0,1,0]])
sys_sws.transitions.add_adj(
    sp.lil_matrix(transmat3), states, sys_actions='left'
)
                      
# mode4 transitions
transmat4 = np.array([[1,0,0,0,0,0],
                      [0,1,0,0,0,0],
                      [0,0,1,0,0,0],
                      [1,0,0,0,0,0],
                      [0,1,1,0,0,0],
                      [0,0,1,0,0,0]])
sys_sws.transitions.add_adj(
    sp.lil_matrix(transmat4), states, sys_actions='down'
)

# This is what is visible to the outside world (and will go into synthesis method)
print(sys_sws)

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
sys_init = {'X0reach', 'sys_actions = "right"'}
sys_prog = {'home'}               # []<>home
sys_safe = {'X (X0reach) <-> lot || (X0reach && !park)'}
sys_prog |= {'X0reach'}

# Create the specification
specs = spec.GRSpec(env_vars, sys_vars, env_init, sys_init,
                    env_safe, sys_safe, env_prog, sys_prog)
                    
# Controller synthesis
#
# At this point we can synthesize the controller using one of the available
# methods.  Here we make use of JTLV.
#
ctrl = synth.synthesize('gr1c', specs, sys=sys_sws)

# Generate a graphical representation of the controller for viewing
if not ctrl.save('controlled_switching.png'):
    print(ctrl)

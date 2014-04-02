# WARNING: This example is not yet working.  Please check again in the
#          upcoming release.
#
# This is an example to demonstrate how the output of a discretization algorithm
# that abstracts a switched system, where the mode of the system depends on a 
# combination of environment and system controlled variables, might look like. 
# We assume within each mode the control authority is rich enough to establish 
# deterministic reachability relations through the use of low-level continuous 
# inputs.

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


###########################################
# Hybrid system with 2 env, 2 system modes:
###########################################

sys_hyb = transys.OpenFTS()

# We assume robots ability to transition between cells depends both on
# discrete controlled modes (e.g., gears) and environment modes (e.g., surface
# conditions).

sys_hyb.sys_actions.add_from({'gear0','gear1'})
sys_hyb.env_actions.add_from({'slippery','normal'})

# str states
n = 6
states = ['s'+str(i) for i in xrange(n) ]

sys_hyb.atomic_propositions.add_from(['home','lot'])
state_labels = [{'home'}, set(), set(), set(), set(), {'lot'}]

# Add states and decorate TS with state labels (aka atomic propositions)
for state, label in zip(states, state_labels):
    sys_hyb.states.add(state, ap=label)

# First environment chooses a mode, than the system chooses a mode and within
# each mode there exists a low level controller to take any available transition
# deterministically.

# gear0 basically stops the robot no matter what the enviornment does so
# We take the transitions to be identity.
trans1 = np.eye(6)

sys_hyb.transitions.add_adj(
    sp.lil_matrix(trans1), states,
    sys_actions='gear0', env_actions='normal'
)
sys_hyb.transitions.add_adj(
    sp.lil_matrix(trans1), states,
    sys_actions='gear0', env_actions='slippery'
)

# gear1 dynamics are similar to the environment switching example.
transmat1 = sp.lil_matrix(np.array(
                [[1,1,0,1,0,0],
                 [1,1,1,0,1,0],
                 [0,1,1,0,1,1],
                 [1,0,0,1,1,0],
                 [0,1,0,1,1,1],
                 [0,0,1,0,1,1]]
            ))

sys_hyb.transitions.add_adj(
    transmat1, states, sys_actions='gear1', env_actions='normal'
)

transmat2 = sp.lil_matrix(np.array(
                [[0,0,1,1,0,0],
                 [1,0,1,0,1,0],
                 [1,0,0,0,1,1],
                 [1,0,0,0,0,1],
                 [0,1,0,1,0,1],
                 [0,0,1,1,0,0]]
            ))

sys_hyb.transitions.add_adj(
    transmat2, states, sys_actions='gear1', env_actions='slippery'
)

# This is what is visible to the outside world (and will go into synthesis method)
print(sys_hyb)

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
#     [](next(X0reach) <-> lot || (X0reach && !park))
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
sys_safe = {'next(X0reach) <-> lot || (X0reach && !park)'}
sys_prog |= {'X0reach'}

# Possible additional specs
# It is unsafe to "break" (switch to gear0) when road is slippery
sys_safe |= {'(gear1 && slippery) -> next(gear1)'}

# to use int actions with gr1c:
# sys_safe |= {'((act = gear1) && (eact = slippery)) -> next(act = gear1)'}

# Create the specification
specs = spec.GRSpec(env_vars, sys_vars, env_init, sys_init,
                    env_safe, sys_safe, env_prog, sys_prog)

# Controller synthesis
#
# At this point we can synthesize the controller using one of the available
# methods.  Here we make use of JTLV.
#
ctrl = synth.synthesize('jtlv', specs, sys=sys_hyb, ignore_sys_init=True)

if not ctrl.save('hybrid.png'):
    print(ctrl)

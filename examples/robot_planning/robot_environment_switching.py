# This is an example to demonstrate how the output of the TuLiP discretization 
# for a system with uncontrolled switching might look like.
# We will assume, we have the 6 cell robot example.

#
#     +---+---+---+
#     | 3 | 4 | 5 |
#     +---+---+---+
#     | 0 | 1 | 2 |
#     +---+---+---+
#

from tulip import *
from tulip import spec
import numpy as np
from scipy import sparse as sp


sys_swe = transys.oFTS()

sys_swe.sys_actions.add('')

# We assume robots ability to transition between cells depends on the surface
# characteristics which could be slippery or normal. This is controlled by the
# environment.

sys_swe.env_actions.add_from({'slippery','normal'})
# environment actions are mutually exclusive

# Discretization builds a transition matrix (invisible to the end user)

# within each mode the transitions can be deterministically chosen, environment
# chooses the mode (the surface can be slippery or normal)
transmat1 = np.array([[1,1,0,1,0,0],
                     [1,1,1,0,1,0],
                     [0,1,1,0,1,1],
                     [1,0,0,1,1,0],
                     [0,1,0,1,1,1],
                     [0,0,1,0,1,1]])
                     
sys_swe.transitions.add_labeled_adj(sp.lil_matrix(transmat1),('','normal'))

# in slippery mode can't stay still and makes larger jumps
transmat2 = np.array([[0,0,1,1,0,0],
                     [1,0,1,0,1,0],
                     [1,0,0,0,1,1],
                     [1,0,0,0,0,1],
                     [0,1,0,1,0,1],
                     [0,0,1,1,0,0]])

sys_swe.transitions.add_labeled_adj(sp.lil_matrix(transmat2),('','slippery'))

# Decorate TS with state labels (aka atomic propositions)
sys_swe.atomic_propositions.add_from(['home','lot'])
sys_swe.atomic_propositions.label_per_state(range(6),[{'home'},set(),set(),set(),set(),{'lot'}])

# This is what is visible to the outside world (and will go into synthesis method)
print sys_swe

#
# Environment variables and specification
#
# The environment can issue a park signal that the robot just respond
# to by moving to the lower left corner of the grid.  We assume that
# the park signal is turned off infinitely often.
#
env_vars = {'park'}
env_init = set()                # empty set
env_prog = '!park'
env_safe = set()                # empty set

# We might want to add additional assumptions
# env_prog |= '!slippery'       
#! NOTE: what slippery means can be infered from TS, r do we need to declare
# it as environment variable explicitly?


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
#     [](next(X0reach) == X0 || (X0reach && !park))
#

# Augment the environmental description to make it GR(1)
#! TODO: create a function to convert this type of spec automatically
env_vars |= {'X0reach'}
env_init |= {'X0reach'}

# Define the specification
#! NOTE: maybe "synthesize" should infer the atomic proposition from the 
# transition system?
sys_vars = set()                # part of TS
sys_init = set()                # empty set
sys_prog = 'home'               # []<>X5
sys_safe = {'next(X0reach) == lot || (X0reach && !park)'}

# Create the specification
specs = spec.GRSpec(env_vars, sys_vars, env_init, sys_init,
                    env_safe, sys_safe, env_prog, sys_prog)
                    
# Controller synthesis
#
# At this point we can synthesize the controller using one of the available
# methods.  Here we make use of JTLV.
#
ctrl = synthesize('jtlv', specs, sys_swe)

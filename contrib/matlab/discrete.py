#!/usr/bin/env python
# robot_discrete.py - example using transition system dynamics
#
# RMM, 20 Jul 2013
"""
This example illustrates the use of TuLiP to synthesize a reactive
controller for system whose dynamics are described by a discrete
transition system.

Addendum - this example also illustrates how to generate a Mealy Machine for
Matlab/Simulink/Stateflow. See the very last line
"""


# Import the packages that we need
from tulip import transys, spec, synth
import aut_to_stateflow # import file that contains to_stateflow

#
# System dynamics
#
# The system is modeled as a discrete transition system in which the
# robot can be located anyplace no a 2x3 grid of cells.  Transitions
# between adjacent cells are allowed, which we model as a transition
# system in this example (it would also be possible to do this via a
# formula)
#
# We label the states using the following picture
#
#     +----+----+----+
#     | X3 | X4 | X5 |
#     +----+----+----+
#     | X0 | X1 | X2 |
#     +----+----+----+
#

# Create a finite transition system
sys = transys.FTS()          

# Define the states of the system
sys.states.add_from(['X0', 'X1', 'X2', 'X3', 'X4', 'X5'])
sys.states.initial.add('X0')    # start in state X0

# Define the allowable transitions
#! TODO (IF): can arguments be a singleton instead of a list?
#! TODO (IF): can we use lists instead of sets?
#!   * use optional flag to allow list as label
sys.transitions.add_from({'X0'}, {'X1', 'X3'})
sys.transitions.add_from({'X1'}, {'X0', 'X4', 'X2'})
sys.transitions.add_from({'X2'}, {'X1', 'X5'})
sys.transitions.add_from({'X3'}, {'X0', 'X4'})
sys.transitions.add_from({'X4'}, {'X3', 'X1', 'X5'})
sys.transitions.add_from({'X5'}, {'X4', 'X2'})

# Add atomic propositions to the states
sys.atomic_propositions.add_from({'home', 'lot'})
sys.states.label('X0', 'home')
sys.states.label('X5', 'lot')


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

# Augment the system description to make it GR(1)
#! TODO: create a function to convert this type of spec automatically
sys_vars = {'X0reach'}          # infer the rest from TS 
sys_init = {'X0reach'}          
sys_prog = {'home'}             # []<>home
sys_safe = {'(X (X0reach) <-> lot) || (X0reach && !park)'}
sys_prog |= {'X0reach'} 

# Create the specification
specs = spec.GRSpec(env_vars, sys_vars, env_init, sys_init,
                    env_safe, sys_safe, env_prog, sys_prog)

#
# Controller synthesis
#
# At this point we can synthesize the controller using one of the available
# methods.  Here we make use of gr1c.
#
ctrl = synth.synthesize('gr1c', specs, sys=sys)

# Generate a MATLAB script that generates a Mealy Machine
aut_to_stateflow.to_stateflow(ctrl, 'robot_discrete.m')

"""
This example illustrates how to export a Mealy Machine to
Matlab/Simulink/Stateflow (see the very last line).

For detailed comments on the example itself,
please see examples/robot_planning/discrete.py
"""
from tulip import transys, spec, synth

# import file that contains to_stateflow
import sys
sys.path.append('../')
import tomatlab

# Create a finite transition system
sys = transys.FTS()

# Define the states of the system
sys.states.add_from(['X0', 'X1', 'X2', 'X3', 'X4', 'X5'])
sys.states.initial.add('X0')    # start in state X0

# Define the allowable transitions
sys.transitions.add_comb({'X0'}, {'X1', 'X3'})
sys.transitions.add_comb({'X1'}, {'X0', 'X4', 'X2'})
sys.transitions.add_comb({'X2'}, {'X1', 'X5'})
sys.transitions.add_comb({'X3'}, {'X0', 'X4'})
sys.transitions.add_comb({'X4'}, {'X3', 'X1', 'X5'})
sys.transitions.add_comb({'X5'}, {'X4', 'X2'})

# Add atomic propositions to the states
sys.atomic_propositions.add_from({'home', 'lot'})
sys.states.add('X0', ap={'home'})
sys.states.add('X5', ap={'lot'})

# Environment variables and specification
env_vars = {'park'}
env_init = set()
env_prog = '!park'
env_safe = set()

# System specification
sys_vars = {'X0reach'}
sys_init = {'X0reach'}
sys_prog = {'home'}
sys_safe = {'(X (X0reach) <-> lot) || (X0reach && !park)'}
sys_prog |= {'X0reach'}

# Create the specification
specs = spec.GRSpec(env_vars, sys_vars, env_init, sys_init,
                    env_safe, sys_safe, env_prog, sys_prog)

# Controller synthesis
ctrl = synth.synthesize('gr1c', specs, sys=sys)

# Generate a MATLAB script that generates a Mealy Machine
tomatlab.export('robot_discrete.mat', ctrl)

#!/usr/bin/env python

""" The example presented at the MURI review, illustrating the use of jtlvint 
and automaton modules 

Nok Wongpiromsarn (nok@cds.caltech.edu)
August 3, 2010
"""
import sys, os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), '..'))

from prop2part import Region, PropPreservingPartition
from jtlvint import *
from automaton import *
from grsim import grsim

# Specify where the smv file, spc file and aut file will go
testfile = 'robot_discrete_simple'
path = os.path.abspath(os.path.dirname(sys.argv[0]))
smvfile = os.path.join(path, 'specs', testfile+'.smv')
spcfile = os.path.join(path, 'specs', testfile+'.spc')
autfile = os.path.join(path, 'specs', testfile+'.aut')

# Specify the environment variables
env_vars = {'park' : 'boolean'}

# Specify the discrete system variable
# Introduce a boolean variable X0reach to handle the spec [](park -> <>X0)
# X0reach starts with TRUE. 
# [](next(X0reach) = (X0 | X0reach) & !park)
sys_disc_vars = {'X0reach' : 'boolean'}

# Specify the transition system representing the continuous dynamics
disc_dynamics = PropPreservingPartition(list_region=[], list_prop_symbol=[])

# These following propositions specify in which cell the robot is, i.e., 
# Xi means that the robot is in cell Ci
disc_dynamics.list_prop_symbol = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5'] 
disc_dynamics.num_prop = len(disc_dynamics.list_prop_symbol)

# Regions. Note that the first argument of Region(poly, prop) should be a list of 
# polytopes. But since we are not dealing with the actual controller, we will 
# just fill it with a string (think of it as a name of the region).
# The second argument of Region(poly, prop) is a list that specifies which 
# propositions in cont_props above is satisfied. As specified below, regioni 
# satisfies proposition Xi.
region0 = Region('C0', [1, 0, 0, 0, 0, 0])
region1 = Region('C1', [0, 1, 0, 0, 0, 0])
region2 = Region('C2', [0, 0, 1, 0, 0, 0])
region3 = Region('C3', [0, 0, 0, 1, 0, 0])
region4 = Region('C4', [0, 0, 0, 0, 1, 0])
region5 = Region('C5', [0, 0, 0, 0, 0, 1])
disc_dynamics.list_region = [region0, region1, region2, region3, region4, region5]
disc_dynamics.num_regions = len(disc_dynamics.list_region)

# The transition relation between regions. disc_dynamics.trans[i][j] = 1 if starting from 
# region j, the robot can move to region i while only staying in the union of region i 
# and region j.
disc_dynamics.trans =   [[1, 1, 0, 1, 0, 0], \
                         [1, 1, 1, 0, 1, 0], \
                         [0, 1, 1, 0, 0, 1], \
                         [1, 0, 0, 1, 1, 0], \
                         [0, 1, 0, 1, 1, 1], \
                         [0, 0, 1, 0, 1, 1]]

# Spec
assumption = 'X0reach & []<>(!park)'
guarantee = '[]<>X5 & []<>(X0reach)'
guarantee += ' & [](next(X0reach) = ((X0 | X0reach) & !park))'

# Generate input to JTLV
prob = generateJTLVInput(env_vars, sys_disc_vars, [assumption, guarantee], \
                                   {}, disc_dynamics, smvfile, spcfile, verbose=2)

# Check realizability
realizability = checkRealizability(smv_file=smvfile, spc_file=spcfile, \
                                       aut_file=autfile, verbose=3)

# Compute an automaton
computeStrategy(smv_file=smvfile, spc_file=spcfile, aut_file=autfile, \
                                    priority_kind=3, verbose=3)
aut = Automaton(autfile, [], 3)

# Simulate
num_it = 30
init_state = {}
init_state['X0reach'] = True
env_states = []
for i in xrange(0,num_it):
    if (i%3 == 0):
        env_states.append({'park':True})
    else:
        env_states.append({'park':False})

states = grsim(aut, init_state, env_states, num_it)

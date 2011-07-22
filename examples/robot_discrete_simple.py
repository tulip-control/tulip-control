#!/usr/bin/env python
"""
The example presented at the MURI review, illustrating the use of
jtlvint and automaton modules

Nok Wongpiromsarn (nok@cds.caltech.edu)
August 3, 2010

minor refactoring by SCL <slivingston@caltech.edu>
1 May 2011.
"""

import sys, os
from subprocess import call

from tulip import *
from tulip import polytope_computations as pc


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
disc_dynamics = prop2part.PropPreservingPartition(list_region=[], list_prop_symbol=[])

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
region0 = pc.Region('C0', [1, 0, 0, 0, 0, 0])
region1 = pc.Region('C1', [0, 1, 0, 0, 0, 0])
region2 = pc.Region('C2', [0, 0, 1, 0, 0, 0])
region3 = pc.Region('C3', [0, 0, 0, 1, 0, 0])
region4 = pc.Region('C4', [0, 0, 0, 0, 1, 0])
region5 = pc.Region('C5', [0, 0, 0, 0, 0, 1])
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
prob = jtlvint.generateJTLVInput(env_vars, sys_disc_vars, [assumption, guarantee], \
                                   {}, disc_dynamics, smvfile, spcfile, verbose=2)

# Check realizability
realizability = jtlvint.checkRealizability(smv_file=smvfile, spc_file=spcfile, \
                                       aut_file=autfile, verbose=3)

# Compute an automaton
jtlvint.computeStrategy(smv_file=smvfile, spc_file=spcfile, aut_file=autfile, \
                                    priority_kind=3, verbose=3)
aut = automaton.Automaton(autfile, [], 3)

# Visualize automaton with DOT file

# This example uses environment vs. system turn distinction.  To
# disable it, just use (the default),
# if not aut.writeDotFile("rdsimple.dot"):
if not aut.writeDotFile("rdsimple.dot",
                        distinguishTurns={"env": prob.getEnvVars().keys(),
                                          "sys": prob.getSysVars().keys()},
                        turnOrder=("env", "sys")):
    print "Error occurred while generating DOT file."
else:
    try:
        call("dot rdsimple.dot -Tpng -o rdsimple.png".split())
    except:
        print "Failed to create image from DOT file. To do so, try\n\ndot rdsimple.dot -Tpng -o rdsimple.png\n"

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

states = grsim.grsim(aut, init_state, env_states, num_it)
grsim.writeStatesToFile(states, 'robot_sim.txt')

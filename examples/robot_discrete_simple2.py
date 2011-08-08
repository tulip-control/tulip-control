#!/usr/bin/env python
"""
The example presented at the MURI review, illustrating the use of jtlvint 
and automaton modules 

minor refactoring by SCL <slivingston@caltech.edu>
1 May 2011.
"""

import sys, os

from tulip import *
import tulip.polytope as pc


# Specify where the smv file, spc file and aut file will go
testfile = 'robot_discrete_simple2'
path = os.path.abspath(os.path.dirname(sys.argv[0]))
smvfile = os.path.join(path, 'specs', testfile+'.smv')
spcfile = os.path.join(path, 'specs', testfile+'.spc')
autfile = os.path.join(path, 'specs', testfile+'.aut')

# Specify the environment variables
env_vars = {'park' : 'boolean', 'cellID' : [0,1,2,3,4,5]}
# Another way to specify this is: env_vars = {'park' : 'boolean', 'cellID' : '{0,...,5}'}

# Specify the discrete system variable
sys_disc_vars = {'gear' : '{-1...1}'}

# Propositions on environment and discrete system variables
disc_props = {'Park' : 'park',
              'X0d' : 'cellID=0',
              'X1d' : 'cellID=1',
              'X2d' : 'cellID=2',
              'X3d' : 'cellID=3',
              'X4d' : 'gear = 0',
              'X5d' : 'gear = 1'}

# Specify the transition system representing the continuous dynamics
disc_dynamics = prop2part.PropPreservingPartition(list_region=[], list_prop_symbol=[])

# Propositions for the continuous system variables
disc_dynamics.list_prop_symbol = ['X0', 'X1', 'X2', 'X3', 'X4'] 
disc_dynamics.num_prop = len(disc_dynamics.list_prop_symbol)

# Regions. Note that the first argument of Region(poly, prop) should be a list of 
# polytopes. But since we are not dealing with the actual controller, we will just 
# fill it with a string (think of it as a name of the region).
# The second argument of Region(poly, prop) is a list that specifies which propositions
# in cont_props above is satisfied.
region0 = pc.Region('C0', [1, 0, 0, 0, 0])
region1 = pc.Region('C1', [0, 1, 0, 0, 0])
region2 = pc.Region('C2', [0, 0, 1, 0, 0])
region3 = pc.Region('C3', [0, 0, 0, 1, 0])
region4 = pc.Region('C4', [0, 0, 0, 0, 1])
region5 = pc.Region('C5', [1, 1, 1, 1, 1])
disc_dynamics.list_region = [region0, region1, region2, region3, region4, region5]
disc_dynamics.num_regions = len(disc_dynamics.list_region)

# The transition relation between regions. disc_dynamics.trans[i][j] = 1 if starting from 
# region j, the robot can move to region i while only staying in the union of region i 
# and region j.
disc_dynamics.trans =   [[1, 1, 0, 1, 0, 0],
                         [1, 1, 1, 0, 1, 0],
                         [0, 1, 1, 0, 0, 1],
                         [1, 0, 0, 1, 1, 0],
                         [0, 1, 0, 1, 1, 1],
                         [0, 0, 1, 0, 1, 1]]

# Spec
assumption = '[]<>(!park) & []<>(!X0d) & []<>(Park -> !X0d)'
guarantee = '[]<>(X0d -> X0) & []<>X1 & []<>(Park -> X4)'
spec = [assumption, guarantee]

# Generate input to JTLV
newvarname = jtlvint.generateJTLVInput(env_vars, sys_disc_vars, spec, disc_props,
                                       disc_dynamics, smvfile, spcfile, verbose=2)

# Check realizability
realizability = jtlvint.checkRealizability(smv_file=smvfile, spc_file=spcfile,
                                           aut_file=autfile, verbose=3)

# Compute an automaton
jtlvint.computeStrategy(smv_file=smvfile, spc_file=spcfile, aut_file=autfile,
                        priority_kind=3, verbose=3)
aut = automaton.Automaton(autfile, [], 3)

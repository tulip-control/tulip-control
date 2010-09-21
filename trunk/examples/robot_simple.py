#!/usr/bin/env python

""" The example is an extension of robot_discrete_simple.py by including continuous dynamics. 

Nok Wongpiromsarn (nok@cds.caltech.edu)
September 2, 2010
"""
import sys, os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), '..'))

from prop2part import Polytope, Region, PropPreservingPartition, prop2part2
from discretizeM import CtsSysDyn, discretizeM
from jtlvint import *
from automaton import *
from numpy import array
from grsim import grsim

# Specify where the smv file, spc file and aut file will go
testfile = 'robot_simple'
path = os.path.abspath(os.path.dirname(sys.argv[0]))
smvfile = os.path.join(path, 'specs', testfile+'.smv')
spcfile = os.path.join(path, 'specs', testfile+'.spc')
autfile = os.path.join(path, 'specs', testfile+'.aut')

# Environment variables
env_vars = {'park' : 'boolean'}

# Discrete system variable
# Introduce a boolean variable X0reach to handle the spec [](park -> <>X0)
# X0reach starts with TRUE. 
# [](next(X0reach) = (X0 | X0reach) & !park)
sys_disc_vars = {'X0reach' : 'boolean'}

# Continuous state space
cont_state_space = Polytope(array([[1., 0.],[-1., 0.], [0., 1.], [0., -1.]]), \
                                   array([[3.],[0.],[2.],[0.]]))

# Continuous proposition
cont_props = {}
for i in xrange(0, 3):
    for j in xrange(0,2):
        prop_sym = 'X' + str(3*j + i)
        cont_props[prop_sym] = Polytope(array([[1., 0.],[-1., 0.], [0., 1.], [0., -1.]]), \
                                   array([[float(i+1)],[float(-i)],[float(j+1)],[float(-j)]]))

# Continuous dynamics: \dot{x} = u_x, \dot{y} = u_y
A = array([[1.1052, 0.],[ 0., 1.1052]])
B = array([[1.1052, 0.],[ 0., 1.1052]])
U = Polytope(array([[1., 0.],[-1., 0.], [0., 1.], [0., -1.]]), array([[1.],[1.],[1.],[1.]]))
sys_dyn = CtsSysDyn(A,B,[],U,[])

# Compute the proposition preserving partition of the continuous state space
cont_partition = prop2part2(cont_state_space, cont_props)

# Discretize the continuous state space
disc_dynamics = discretizeM(cont_partition, sys_dyn, verbose=2)

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

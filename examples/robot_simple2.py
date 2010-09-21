#!/usr/bin/env python

""" The example presented at the MURI review, illustrating the use of rhtlp module. 

Nok Wongpiromsarn (nok@cds.caltech.edu)
September 2, 2010
"""
import sys, os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), '..'))

from prop2part import Polytope, Region, PropPreservingPartition
from discretizeM import CtsSysDyn, discretizeM
from spec import GRSpec
from rhtlp import SynthesisProb
from numpy import array
from grsim import grsim

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

# Spec
spec = GRSpec()
spec.env_prog = '!park'
spec.sys_init = 'X0reach'
spec.sys_safety = 'next(X0reach) = ((X0 | X0reach) & !park)'
spec.sys_prog = ['X5', 'X0reach']

# Construct the SynthesisProb object
prob = SynthesisProb(env_vars = env_vars, sys_disc_vars = sys_disc_vars, \
                         disc_props = {}, cont_state_space = cont_state_space, \
                         cont_props = cont_props, sys_dyn = sys_dyn, spec=spec, verbose=2)

# Check realizability
realizability = prob.checkRealizability(verbose=2)

# Compute an automaton
aut = prob.synthesizePlannerAut(verbose=2)

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

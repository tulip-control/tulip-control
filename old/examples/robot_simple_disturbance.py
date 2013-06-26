#!/usr/bin/env python
"""
The example is an extension of robot_discrete_simple.py by including
disturbanceand input computation using the "closed loop" algorithm.

Petter Nilsson (pettni@kth.se)
August 14, 2011
"""

import sys, os
import numpy as np
from subprocess import call
import matplotlib
import matplotlib.pyplot as plt

from tulip import *
import tulip.polytope as pc
from tulip.polytope.plot import plot_partition

# Problem parameters
input_bound = 0.4
uncertainty = 0.05
N = 5

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
cont_state_space = pc.Polytope(np.array([[1., 0.],[-1., 0.], [0., 1.], [0., -1.]]),
                               np.array([[3.],[0.],[2.],[0.]]))

# Continuous proposition
cont_props = {}
for i in xrange(0, 3):
    for j in xrange(0, 2):
        prop_sym = 'X' + str(3*j + i)
        cont_props[prop_sym] = pc.Polytope(
                            np.array([[1., 0.],[-1., 0.], [0., 1.], [0., -1.]]),
                np.array([[float(i+1)],[float(-i)],[float(j+1)],[float(-j)]]))

# Continuous dynamics
A = np.array([[1.1052, 0.],[ 0., 1.1052]])
B = np.array([[1.1052, 0.],[ 0., 1.1052]])
E = np.array([[1,0],[0,1]])
U = pc.Polytope(np.array([[1., 0.],[-1., 0.], [0., 1.], [0., -1.]]),  \
                input_bound*np.array([[1.],[1.],[1.],[1.]]))
W = pc.Polytope(np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]]), \
                uncertainty*np.array([1., 1., 1., 1.]))
sys_dyn = discretize.CtsSysDyn(A,B,E,[],U,W)


# Compute the proposition preserving partition of the continuous state space
cont_partition = prop2part.prop2part2(cont_state_space, cont_props)


# Discretize the continuous state space
disc_dynamics = discretize.discretize(cont_partition, sys_dyn, closed_loop=True, \
                N=N, min_cell_volume=0.1, verbose=0)

#plot_partition(disc_dynamics, plot_transitions=True)

# Spec
assumption = 'X0reach & []<>(!park)'
guarantee = '[]<>X5 & []<>(X0reach)'
guarantee += ' & [](next(X0reach) = ((X0 | X0reach) & !park))'

# Generate input to JTLV
prob = jtlvint.generateJTLVInput(env_vars, sys_disc_vars, [assumption, guarantee],
                                 {}, disc_dynamics, smvfile, spcfile, verbose=0)

# Check realizability
# realizability = jtlvint.checkRealizability(smv_file=smvfile, spc_file=spcfile,
#                                           aut_file=autfile, verbose=0)

# Compute an automaton
jtlvint.computeStrategy(smv_file=smvfile, spc_file=spcfile, aut_file=autfile,
                        priority_kind=3, verbose=0)
aut = automaton.Automaton(autfile, [], 3)

# Remove dead-end states from automaton
aut.trimDeadStates()

# Simulate
num_it = 10
init_state = {'X0reach': True}

states = grsim.grsim([aut], env_states=[init_state], num_it=num_it,
                     deterministic_env=False)

# Store discrete trajectory in np array
cellid_arr = []
for (autID, state) in states:
    cellid_arr.append(state.state['cellID'])
cellid_arr = np.array(cellid_arr)

# First continuous state is middle point of first cell
r, x = pc.cheby_ball(disc_dynamics.list_region[cellid_arr[0]])
x = x.flatten()
x_arr = x
u_arr = np.zeros([N*num_it, B.shape[1]])
d_arr = np.zeros([N*num_it, E.shape[1]])
for i in range(1, len(cellid_arr)):
    # For each step, calculate N input signals
    for j in range(N):
        u_seq = discretize.get_input(x, sys_dyn, disc_dynamics, \
                cellid_arr[i-1], cellid_arr[i], N-j, mid_weight=3, Q=np.eye(2*(N-j)), \
                test_result=True)
        u0 = u_seq[0,:] # Only the first input should be used
        u_arr[(i-1)*N + j,:] = u0   # Store input
        
        d = uncertainty * 2 * (np.random.rand(2) - 0.5 )   # Simulate disturbance
        d_arr[(i-1)*N + j,:] = d    # Store disturbance
        
        x = np.dot(sys_dyn.A, x).flatten() + np.dot(sys_dyn.B, u0).flatten() + np.dot(sys_dyn.E, d).flatten()
        x_arr = np.vstack([x_arr, x])   # Store state

# Print trajectory information
for i in range(x_arr.shape[0]-1):
    print "From: " + str(cellid_arr[np.floor(i/N)]) + " to " + str(cellid_arr[np.floor(i/N) + 1]) \
            + " u: " + str(u_arr[i,:]) + " x: " + str(x_arr[i,:]) + " d: " + str(d_arr[i,:])
print "Final state x: " + str(x_arr[-1,:])
        
# Plot state trajectory
ax = plot_partition(disc_dynamics, show=False)
arr_size = 0.05
for i in range(1,x_arr.shape[0]):
    x = x_arr[i-1,0]
    y = x_arr[i-1,1]
    dx = x_arr[i,0] - x
    dy = x_arr[i,1] - y
    arr = matplotlib.patches.Arrow(float(x),float(y),float(dx),float(dy),width=arr_size)
    ax.add_patch(arr)
spec_ind = range(0, x_arr.shape[0], N)

ax.plot(x_arr[spec_ind,0], x_arr[spec_ind,1], 'oy')
ax.plot(x_arr[0,0], x_arr[0,1], 'og')
ax.plot(x_arr[-1,0], x_arr[-1,1], 'or')

plt.show()

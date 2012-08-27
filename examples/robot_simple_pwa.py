#!/usr/bin/env python
"""
The example is an extension of the robot_simple_disturbance.py
code by Petter Nilsson and Nok Wongpiromsarn. It demonstrates 
the use of TuLiP for systems with piecewise affine dynamics.

Necmiye Ozay, August 26, 2012
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
N = 5

# Specify where the smv file, spc file and aut file will go
testfile = 'robot_simple_pwa'
path = os.path.abspath(os.path.dirname(sys.argv[0]))
smvfile = os.path.join(path, 'specs', testfile+'.smv')
spcfile = os.path.join(path, 'specs', testfile+'.spc')
autfile = os.path.join(path, 'specs', testfile+'.aut')

# Environment variables
env_vars = {'park' : 'boolean'}

# No discrete system variable
sys_disc_vars = {}

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

# Assume, for instance, our robot is traveling on a nonhomogenous surface (xy plane), 
# resulting in different dynamics at different parts of the plane. 
# Since the continuous state space in this example is just xy position, different
# dynamics in different parts of the surface can be modeled as a piecewise 
# affine system as follows:

input_bound = 0.4
uncertainty = 0.05

A0 = np.array([[1.1052, 0.],[ 0., 1.1052]])
B0 = np.array([[1.1052, 0.],[ 0., 1.1052]])
E0 = np.array([[1,0],[0,1]])
U0 = pc.Polytope(np.array([[1., 0.],[-1., 0.], [0., 1.], [0., -1.]]),  \
                input_bound*np.array([[1.],[1.],[1.],[1.]]))
W0 = pc.Polytope(np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]]), \
                uncertainty*np.array([1., 1., 1., 1.]))
dom0 = pc.Polytope(np.array([[1., 0.],[-1., 0.], [0., 1.], [0., -1.]]),\
                  np.array([[3],[0.],[2.],[-0.5]]))
sys_dyn0 = discretize.PwaSubsysDyn(A0,B0,E0,[],U0,W0,dom0)

A1 = np.array([[0.9948, 0.],[ 0., 1.1052]])
B1 = np.array([[-1.1052, 0.],[ 0., 1.1052]])
E1 = np.array([[1,0],[0,1]])
U1 = pc.Polytope(np.array([[1., 0.],[-1., 0.], [0., 1.], [0., -1.]]),  \
                input_bound*np.array([[1.],[1.],[1.],[1.]]))
W1 = pc.Polytope(np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]]), \
                uncertainty*np.array([1., 1., 1., 1.]))
dom1 = pc.Polytope(np.array([[1., 0.],[-1., 0.], [0., 1.], [0., -1.]]),\
                  np.array([[3],[0.],[0.5],[0.]]))
sys_dyn1 = discretize.PwaSubsysDyn(A1,B1,E1,[],U1,W1,dom1)

# Build piecewise affine system from its subsystems
sys_dyn = discretize.PwaSysDyn([sys_dyn0,sys_dyn1], cont_state_space)

# Compute the proposition preserving partition of the continuous state space
cont_partition = prop2part.prop2part2(cont_state_space, cont_props)


# Discretize the continuous state space
disc_dynamics = discretize.discretize(cont_partition, sys_dyn, closed_loop=True, \
                N=N, min_cell_volume=0.1, verbose=0)

#plot_partition(disc_dynamics, plot_transitions=True)

# Spec
assumption = '[]<>(!park)'
guarantee = '[]<>X5'
guarantee += ' & [](park -> <>X0)'

# Generate input to JTLV
prob = jtlvint.generateJTLVInput(env_vars, sys_disc_vars, [assumption, guarantee],
                                 {}, disc_dynamics, smvfile, spcfile, verbose=2)

# Check realizability
realizability = jtlvint.checkRealizability(smv_file=smvfile, spc_file=spcfile,
                                           aut_file=autfile, verbose=3)

# Compute an automaton
jtlvint.computeStrategy(smv_file=smvfile, spc_file=spcfile, aut_file=autfile,
                        priority_kind=3, verbose=3)
aut = automaton.Automaton(autfile, [], 3)

# Remove dead-end states from automaton
aut.trimDeadStates()



# Simulate
num_it = 10
init_state = {}

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
u_arr = np.zeros([N*num_it, B0.shape[1]])
d_arr = np.zeros([N*num_it, E0.shape[1]])
for i in range(1, len(cellid_arr)):
    ss = sys_dyn.list_subsys[disc_dynamics.list_subsys[cellid_arr[i-1]]]
    # For each step, calculate N input signals
    for j in range(N):
        u_seq = discretize.get_input(x, ss, disc_dynamics, \
                cellid_arr[i-1], cellid_arr[i], N-j, mid_weight=3, Q=np.eye(2*(N-j)), \
                test_result=True)
        u0 = u_seq[0,:] # Only the first input should be used
        u_arr[(i-1)*N + j,:] = u0   # Store input
        
        d = uncertainty * 2 * (np.random.rand(2) - 0.5 )   # Simulate disturbance
        d_arr[(i-1)*N + j,:] = d    # Store disturbance
        
        x = np.dot(ss.A, x).flatten() + np.dot(ss.B, u0).flatten() + np.dot(ss.E, d).flatten()
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
#!/usr/bin/env python
"""
The example is an extension of robot_discrete_simple.py by including
disturbance and input computation using the "closed loop" algorithm.
This is an almost verbatim copy of the robot_simple_disturbance.py
code by Petter Nilsson and Nok Wongpiromsarn.  It demonstrates use of
the gr1c synthesis tool, rather than the historic default of JTLV.

Toggle the truth value of load_from_XML to indicate whether to
generate a new tulipcon XML file, or read from one.

SCL; 11 Mar 2012.
"""
import sys, os
import numpy as np
from subprocess import call
import matplotlib
import matplotlib.pyplot as plt

from tulip import discretize, prop2part, grsim, conxml
from tulip import gr1cint
from tulip.spec import GRSpec
import tulip.polytope as pc
from tulip.polytope.plot import plot_partition

# Problem parameters
input_bound = 0.4
uncertainty = 0.05
N = 5

# Specify where the smv file, spc file and aut file will go
testfile = 'rsimple_example'
path = os.path.abspath(os.path.dirname(sys.argv[0]))
smvfile = os.path.join(path, 'specs', testfile+'.smv')
spcfile = os.path.join(path, 'specs', testfile+'.spc')
autfile = os.path.join(path, 'specs', testfile+'.aut')

load_from_XML = False
if not load_from_XML:

    # Environment variables
    env_vars = {'park' : 'boolean'}

    # Discrete system variable
    # Introduce a boolean variable X0reach to handle the spec [](park -> <>X0)
    # X0reach starts with TRUE. 
    # [](next(X0reach) = (X0 | X0reach) & !park)
    sys_disc_vars = {'X0reach' : 'boolean'}

    # Continuous state space
    cont_state_space = pc.Polytope(np.array([[1., 0.],
                                             [-1., 0.],
                                             [0., 1.],
                                             [0., -1.]]),
                                   np.array([[3.],[0.],[2.],[0.]]))

    # Continuous proposition
    cont_props = {}
    for i in xrange(0, 3):
        for j in xrange(0, 2):
            prop_sym = 'X' + str(3*j + i)
            cont_props[prop_sym] = pc.Polytope(np.array([[1., 0.],
                                                         [-1., 0.],
                                                         [0., 1.],
                                                         [0., -1.]]),
                                               np.array([[float(i+1)],
                                                         [float(-i)],
                                                         [float(j+1)],
                                                         [float(-j)]]))

    # Continuous dynamics
    A = np.array([[1.1052, 0.],[ 0., 1.1052]])
    B = np.array([[1.1052, 0.],[ 0., 1.1052]])
    E = np.array([[1,0],[0,1]])
    U = pc.Polytope(np.array([[1., 0.],[-1., 0.], [0., 1.], [0., -1.]]),
                    input_bound*np.array([[1.],[1.],[1.],[1.]]))
    W = pc.Polytope(np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]]),
                    uncertainty*np.array([1., 1., 1., 1.]))
    sys_dyn = discretize.CtsSysDyn(A,B,E,[],U,W)

    # Compute the proposition preserving partition of the continuous state space
    cont_partition = prop2part.prop2part2(cont_state_space, cont_props)

    # Discretize the continuous state space
    disc_dynamics = discretize.discretize(cont_partition, sys_dyn, closed_loop=True,
                                          N=N, min_cell_volume=0.1, verbose=0)

    # Spec
    spec = GRSpec(env_vars=env_vars.keys(),
                  sys_vars=sys_disc_vars.keys(),
                  sys_init=["X0reach"],
                  env_prog=["!park"],
                  sys_safety=["(X0reach' & ((X0' | X0reach) & !park')) | (!X0reach' & ((!X0' & !X0reach) | park'))"],
                  sys_prog=['X5', 'X0reach'])

    # Import discretization (abstraction) of continuous state space
    spec.importDiscDynamics(disc_dynamics)

    # Check realizability
    realizability = gr1cint.check_realizable(spec, verbose=1)

    # Compute an automaton
    aut = gr1cint.synthesize(spec, verbose=1)
    aut.writeDotFile("rdsimple_gr1c_example.dot", hideZeros=True)

    # Remove dead-end states from automaton
    #aut.trimDeadStates()

    #conxml.writeXMLfile("rsimple_example.xml", prob, spec, sys_dyn, aut, pretty=True)

else:
    # Read from tulipcon XML file
    (prob, sys_dyn, aut) = conxml.readXMLfile("rsimple_example.xml")
    disc_dynamics = prob.getDiscretizedDynamics()

# Simulate
num_it = 10
init_state = {'X0reach': True}

graph_vis = raw_input("Do you want to open in Gephi? (y/n)") == 'y'
destfile = 'rsdisturbance_example.gexf'
states = grsim.grsim([aut], env_states=[init_state], num_it=num_it,
                     deterministic_env=False, graph_vis=graph_vis,
                     destfile=destfile)

# Dump state sequence.
print "\n".join([str(state.state) for (autID, state) in states]) + "\n"

# Store discrete trajectory in np array
cellid_arr = []
for (autID, state) in states:
    occupied_cells = [int(k[len("cellID_"):]) for (k,v) in state.state.items() if v==1 and k.startswith("cellID")]
    if len(occupied_cells) > 1:
        print "ERROR: more than one cell occupied by continuous state."
        exit(-1)
    cellid_arr.append(occupied_cells[0])
cellid_arr = np.array(cellid_arr)

# First continuous state is middle point of first cell
r, x = pc.cheby_ball(disc_dynamics.list_region[cellid_arr[0]])
x = x.flatten()
x_arr = x
u_arr = np.zeros([N*num_it, sys_dyn.B.shape[1]])
d_arr = np.zeros([N*num_it, sys_dyn.E.shape[1]])
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

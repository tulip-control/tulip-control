#!/usr/bin/env python
"""
Usage: smallsurvey.py [H W]

will generate and solve a problem on a grid of size H by W, with
default of 2 by 3.  The agent is initialized in the upper-left corner
cell and a goal is created in the upper-right.  Further, if a flag is
set (unpredictably), then the lower-left cell should eventually be
visited.  This example draws heavily from that in robot_simple_disturbance.py


SCL; 28 June 2012.
"""

import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import tulip.polytope as pc
from tulip.polytope.plot import plot_partition
import tulip.gridworld as gw
from tulip import gr1cint, grsim, discretize
from tulip.spec import GRSpec


if __name__ == "__main__":
    if len(sys.argv) > 3 or "-h" in sys.argv:
        print "Usage: smallsurvey.py [H W]"
        exit(1)
    if len(sys.argv) < 3:
        (num_rows, num_cols) = (2, 3)
    else:
        (num_rows, num_cols) = (int(sys.argv[1]), int(sys.argv[2]))

    # Generate an entirely unoccupied (no obstacles) gridworld,
    #
    Y = gw.unoccupied((num_rows, num_cols))
    initial_partition = Y.dumpPPartition(side_lengths=(1., 1.),
                                         offset=(0., 0.))

    # Problem parameters
    input_bound = 0.4
    uncertainty = 0.05
    N = 5

    env_vars = {'park' : 'boolean'}  # Environment variables

    # Introduce a boolean variable Yreach to handle the spec
    # [](park -> <>Y(-1,0))
    # where Y(-1,0) denotes the lower-left grid cell.
    sys_disc_vars = {'Yreach' : 'boolean'}  # Discrete system variables

    # Continuous dynamics
    A = np.array([[1.1052, 0.],
                  [ 0., 1.1052]])
    B = np.array([[1.1052, 0.],
                  [ 0., 1.1052]])
    E = np.array([[1,0],
                  [0,1]])
    U = pc.Polytope(np.array([[1., 0.],[-1., 0.], [0., 1.], [0., -1.]]),
                    input_bound*np.array([[1.],[1.],[1.],[1.]]))
    W = pc.Polytope(np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]]),
                    uncertainty*np.array([1., 1., 1., 1.]))
    sys_dyn = discretize.CtsSysDyn(A,B,E,[],U,W)
    disc_dynamics = discretize.discretize(initial_partition, sys_dyn,
                                          closed_loop=True, N=N,
                                          min_cell_volume=0.1,
                                          verbose=2)

    # Build specification in terms of countable gridworld
    spec = GRSpec(env_vars=env_vars.keys(),
                  sys_vars=sys_disc_vars.keys(),
                  sys_init=["Yreach & "+Y[0,0]],
                  env_prog=["!park"],
                  sys_safety=["(Yreach' & (("+Y[-1,0]+" | Yreach) & !park)) | (!Yreach' & !(("+Y[-1,0]+" | Yreach) & !park))"],
                  sys_prog=[Y[0,-1], 'Yreach'])
    spec.importGridWorld(Y)

    # ...and then import discretization of continuous state space
    spec.importDiscDynamics(disc_dynamics)

    # Check realizability and compute an automaton
    realizability = gr1cint.check_realizable(spec, verbose=1)
    aut = gr1cint.synthesize(spec, verbose=1)

    
    # Simulate
    num_it = 20
    init_state = {'Yreach': True}

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
        arr = mpl.patches.Arrow(float(x),float(y),float(dx),float(dy),width=arr_size)
        ax.add_patch(arr)
    spec_ind = range(0, x_arr.shape[0], N)

    ax.plot(x_arr[spec_ind,0], x_arr[spec_ind,1], 'oy')
    ax.plot(x_arr[0,0], x_arr[0,1], 'og')
    ax.plot(x_arr[-1,0], x_arr[-1,1], 'or')

    plt.show()

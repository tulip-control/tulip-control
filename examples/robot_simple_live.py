#!/usr/bin/env python
"""
The example is an extension of robot_discrete_simple.py by including
disturbance and input computation using the "closed loop" algorithm.
This is an almost verbatim copy of the robot_simple_disturbance.py
code by Petter Nilsson and Nok Wongpiromsarn.  It demonstrates
elementary use of a live gr1c session.

SCL; 27 Mar 2012.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from tulip import discretize, prop2part, conxml
from tulip import gr1cint
from tulip.spec import GRSpec
import tulip.polytope as pc
from tulip.polytope.plot import plot_partition


spec_filename = "rsimple_live_example.spc"


def get_cellID(state):
    occupied_cells = [int(k[len("cellID_"):]) for (k,v) in state.items() if v==1 and k.startswith("cellID")]
    if len(occupied_cells) > 1:
        print "ERROR: more than one cell occupied by continuous state."
        return None
    return occupied_cells[0]


# Problem parameters
input_bound = 0.4
uncertainty = 0.05
N = 5
num_it = 20  # Number of iterations for simulation

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
realizable = gr1cint.check_realizable(spec, verbose=1)
if not realizable:
    print "Problem specification cannot be realized."
    exit(-1)

# Dump spec to a file
with open(spec_filename, "w") as f:
    f.write(spec.dumpgr1c())


############################################################
# Simulate

# Open interactive gr1c session
gs = gr1cint.GR1CSession(spec_filename,
                         sys_vars=spec.sys_vars, env_vars=spec.env_vars)
if gs.p is None:
    print "Error: failed to start gr1c process"
    exit(-1)

goal_mode = 0  # Track which goal the system is currently pursuing

state = {"X0reach": 1, "cellID_0": 1, "park": 0}
state.update([(v, 0) for v in spec.env_vars if v not in state.keys()])
state.update([(v, 0) for v in spec.sys_vars if v not in state.keys()])

# First continuous state is middle point of first cell
r, x = pc.cheby_ball(disc_dynamics.list_region[get_cellID(state)])
x = x.flatten()
x_arr = x
u_arr = np.zeros([N*num_it, sys_dyn.B.shape[1]])
d_arr = np.zeros([N*num_it, sys_dyn.E.shape[1]])
for i in range(num_it-1):
    # Switch to next goal mode if current satisfied
    if gs.getindex(state, goal_mode) == 0:
        if goal_mode < len(spec.sys_prog)-1:
            goal_mode += 1
        else:
            goal_mode = 0
        print "Switched to goal mode "+str(goal_mode)

    env_moves = gs.env_next(state)
    env_move = env_moves[np.random.randint(0, len(env_moves))]

    sys_moves = gs.sys_nextfeas(state, env_move, goal_mode)
    next_state = sys_moves[0]
    next_state.update(env_move)

    # For each step, calculate N input signals
    print "From cell " + str(get_cellID(state)) + " to " + str(get_cellID(next_state))+":"
    print "\tx0: " + str(x)
    for j in range(N):
        u_seq = discretize.get_input(x, sys_dyn, disc_dynamics,
                                     get_cellID(state), get_cellID(next_state),
                                     N-j, mid_weight=3, Q=np.eye(2*(N-j)),
                                     test_result=True)
        u0 = u_seq[0,:] # Only the first input should be used
        u_arr[(i-1)*N + j,:] = u0   # Store input
        
        d = uncertainty * 2 * (np.random.rand(2) - 0.5 )   # Simulate disturbance
        d_arr[(i-1)*N + j,:] = d    # Store disturbance
        
        x = np.dot(sys_dyn.A, x).flatten() + np.dot(sys_dyn.B, u0).flatten() + np.dot(sys_dyn.E, d).flatten()
        x_arr = np.vstack([x_arr, x])   # Store state

        print "\tu"+str(j)+": " + str(u0) + " d"+str(j)+": " + str(d) + " x"+str(j+1)+": " + str(x)

    state = next_state  # Prepare for next strategy-level simulation step


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

"""
Eric Wolff
"""

import numpy as np
import cProfile
import random
import scipy.linalg as linalg

from tulip import *
import tulip.polytope as pc
from tulip.polytope.plot import plot_partition

## Linear dynamics for TuLiP
# x_{t+1} = Ax_t + Bu_t + Ew_t
# u \in U and w \in W
A = np.array([[1, 0.],[ 0., 1]])
B = np.array([[1, 0.],[ 0., 1]])
E = np.array([[1,0],[0,1]])
u_bound = 0.6           #control bound for TuLiP
w_bound_tulip = 0.05    #noise bound for TuLiP
U = pc.Polytope(np.array([[1., 0.],[-1., 0.], [0., 1.], [0., -1.]]),  \
                u_bound*np.array([[1.],[1.],[1.],[1.]]))
W = pc.Polytope(np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]]), \
                w_bound_tulip*np.array([1., 1., 1., 1.]))

# Miscellaneous
numSteps = 5                #number of control steps between regions
plot_traj = False           #show plots of the system trajectories?

# Continuous state space
cont_state_space = pc.Polytope(np.array([[1., 0.],[-1., 0.], [0., 1.], [0., -1.]]),
                           np.array([[3.],[0.],[2.],[0.]]))

# Continuous proposition
cont_props = {}
for i in xrange(0, 3):
    for j in xrange(0, 2):
        prop_sym = 'x' + str(3*j + i)
        cont_props[prop_sym] = pc.Polytope(
                            np.array([[1., 0.],[-1., 0.], [0., 1.], [0., -1.]]),
                np.array([[float(i+1)],[float(-i)],[float(j+1)],[float(-j)]]))

sys_dyn = discretize.CtsSysDyn(A,B,E,[],U,W)

# Compute the proposition preserving partition of the continuous state space--dynamics not involved
cont_partition = prop2part.prop2part2(cont_state_space, cont_props)

# Discretize the continuous state space--based on system dynamics
disc_dynamics = discretize.discretize(cont_partition, sys_dyn, closed_loop=True, \
            use_mpt=False, N=numSteps, min_cell_volume=0.1, verbose=0)

print 'Num Regions:',disc_dynamics.num_regions
print 'Adjacency matrix:\n',disc_dynamics.adj.transpose()
print 'Transition matrix:\n',disc_dynamics.trans.transpose()
assert (disc_dynamics.trans.transpose() == disc_dynamics.trans).all()
assert (disc_dynamics.trans == disc_dynamics.adj).all()

#plot_partition(disc_dynamics, show=True)


# Determine the minimum, expected, and maximum costs between two regions.  
# Use the unconstrained LQR approximation as well as the fully constrained problem.
numSamples = 5
N = numSteps
ssys = sys_dyn
Q = 1.0*np.eye(2)
R = 1.0*np.eye(2)
#r = np.zeros([2,1])
print "Q=\n",Q
print "R=\n",R

#Q_block = linalg.block_diag(*[Q]*N)
#print "Q_block=\n",Q_block
#R_block = linalg.block_diag(*[R]*N)
#r_block = np.zeros([2*N,1])

for i in xrange(disc_dynamics.num_regions):
    for j in xrange(disc_dynamics.num_regions):
        if disc_dynamics.trans[i,j]:# and i == j:
            print ""

            H0 = disc_dynamics.list_region[i].list_poly[0]
            H1 = disc_dynamics.list_region[j].list_poly[0]
            
            rc, xc = pc.cheby_ball(H1)
            xc = xc.flatten()
            
            lqrMinCost = discretize.lqrMinCost(ssys, H0, xc, N, R, Q)
            lqrExpCost = discretize.lqrExpectedCost(numSamples, ssys, H0, xc, N, R, Q)
            lqrMaxCost = discretize.lqrMaxCost(ssys, H0, xc, N, R, Q)

            # Modified functions
            cstMaxCost = discretize.cstMaxCost(ssys, H0, H1, xc, N, R, Q)
            #cstMinCost = discretize.cstMinCost(ssys, H0, H1, xc, N, R, Q)
            cstExpCost = discretize.cstExpectedCost(numSamples, ssys, H0, H1, xc, N, R, Q)

            print i,j
            print "lqrMinCost=",lqrMinCost
            print "lqrExpCost=",lqrExpCost
            print "lqrMaxCost=",lqrMaxCost
            #print "cstMinCost=",cstMinCost
            print "cstExpCost=",cstExpCost
            print "cstMaxCost=",cstMaxCost


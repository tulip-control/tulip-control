#!/usr/bin/env python
import os,sys
import numpy as np
import time
import random

from tulip import *
import tulip.polytope as pc
from tulip.polytope.plot import plot_partition


def create_cell(sys_dyn, disc_dynamics, numSteps, w_bound_true, numSamples, plot_traj):
    # Create transition probability vector for a single cell
    numRegions = 9
    assert disc_dynamics.num_regions == numRegions  

    # Position in disc_dynamics.list_region
    mid = 1; north = 2; east = 0; south = 4; west = 6
    
    radius = 0.5
    center = np.array([1.5,1.5])

    relDir = np.zeros(6)    #(up,right,down,left,middle,error)  relative directions
    for n in xrange(numSamples):
        # Sample new point in region
        x = np.zeros(2)
        while True:
            x[0:2] = np.random.multivariate_normal(center,(1.0*radius)**2*np.eye(2))
            if pc.is_inside(disc_dynamics.list_region[mid], x[0:2]):
                break
        
        # Generate the disturbance (Gaussian with finite support)
        cov = 0.1*(w_bound_true**2*np.eye(2))
        w_arr = get_disturb(np.zeros(2),cov,w_bound_true,numSteps)
        
        # Propagate dynamics forward with disturbance and control using true system dynamics
        tulipNext = north
        u_arr = discretize.get_input(x[0:2], sys_dyn, disc_dynamics, mid, tulipNext, \
                                     numSteps, mid_weight=0, test_result=True)
        x_arr = sim_dynamics(x, u_arr, w_arr, numSteps, sys_dyn)
        
        # Plot the trajectory     
        if plot_traj == True: plot_xarr(x_arr,disc_dynamics)
        
        # Determine what region(s) the system transitioned to
        next,multiVisit = next_region(x_arr,disc_dynamics,numRegions,mid)
        
        # Convert next_region to form: [N,E,S,W,self=mid,error]
        if   next == north: relDir[0] += 1
        elif next == east:  relDir[1] += 1
        elif next == south: relDir[2] += 1
        elif next == west:  relDir[3] += 1
        elif next == mid:   relDir[4] += 1
        else:               relDir[5] += 1
    
    relDir = relDir / float(numSamples)
    
    up = relDir[0]; right = relDir[1]; down = relDir[2]; 
    left = relDir[3]; mid = relDir[4]; error =relDir[5]   #relative directions
    assert abs(up+left+right+down+mid+error - 1.0) < 1e-12
    cell = {}
    cell['north'] = np.array([up,    right, down,  left,  mid, error])
    cell['east']  = np.array([left,  up,    right, down,  mid, error])
    cell['south'] = np.array([down,  left,  up,    right, mid, error])
    cell['west']  = np.array([right, down,  left,  up,    mid, error])
    cell['stop']  = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]) #only does a self-transition (since it doesn't move)
    
    return cell


def get_disturb(mean, cov, bound, numSteps):
    # Generate the disturbance (Gaussian with finite support)
    w_arr = np.zeros([numSteps,len(mean)])
    for i in xrange(numSteps):
        while True:
            w =  np.random.multivariate_normal(mean,cov)
            w_bound = pc.Polytope.from_box(np.array(bound*[[-1., 1.],[-1., 1.]]))
            if pc.is_inside(w_bound, w):
                break
        w_arr[i] = w
    return w_arr


def sim_dynamics(x0, u_arr, w_arr, numSteps, sys_dyn):
    # Propagate dynamics forward with disturbance and control using true system dynamics
    A_true = sys_dyn.A
    B_true = sys_dyn.B
    E_true = sys_dyn.E
    
    x = x0
    x_arr = np.zeros((numSteps,len(x)))
    for i in xrange(numSteps):
        u = u_arr[i]
        w = w_arr[i]
        x = np.dot(A_true, x).flatten() + np.dot(B_true, u).flatten() + np.dot(E_true, w).flatten()
        x_arr[i] = x
    x_arr = np.vstack([x0,x_arr])
    return x_arr


def next_region(x_arr,disc_dynamics,numRegions,initReg):
    # Determine what region(s) the system transitioned to

    visited_regions = np.zeros(len(x_arr))
    multiVisit = False
    
    for i in xrange(len(x_arr)):
        # Record every region visited along system trajectory
        for j in xrange(numRegions):
            if pc.is_inside(disc_dynamics.list_region[j],x_arr[i][0:2]):
                visited_regions[i] = j
        
    # Remove initial trajectory prefix (i.e. current/initial region) 
    while len(visited_regions) > 0:
        if visited_regions[0] == initReg:
            visited_regions = visited_regions[1:]
        else:
            break
    if len(visited_regions) > 0:    # If visited more than initial region
        next_region = visited_regions[0]
        if len(np.nonzero(visited_regions != next_region)[0]) > 0:  #visited multiple regions
            multiVisit = True;
    else:         # Never left initial region
        next_region = initReg
    next = int(next_region)
    return next,multiVisit



def plot_xarr(x_arr,disc_dynamics):
    # Plot the trajectory     
    ax = plot_partition(disc_dynamics, show=False)
    arr_size = 0.05
    for i in xrange(1,len(x_arr)):
        x = x_arr[i-1,0]
        y = x_arr[i-1,1]
        dx = x_arr[i,0] - x
        dy = x_arr[i,1] - y
        arr = matplotlib.patches.Arrow(float(x),float(y),float(dx),float(dy),width=arr_size)
        ax.add_patch(arr)
    ax.plot(x_arr[0,0], x_arr[0,1], 'og')
    ax.plot(x_arr[-1,0], x_arr[-1,1], 'or')
    plt.show()
    plt.draw()


def create_MDP(cell, xmax, ymax):
    
    MDP = {}
    catch = (-1,-1)
    MDP[catch] = {'catch': {catch:1.0}}
    
    # Map between numeric identifier and (x,y) position
    id2pos = {}
    pos2id = {}
    id = 0
    id2pos[id] = catch
    pos2id[catch] = id
    
    for y in xrange(ymax):
        for x in xrange(xmax):
            id += 1
            id2pos[id] = (x,y)
            pos2id[(x,y)] = id
    
    # Determine transition probabilities between all cells
    for y in xrange(ymax):
        for x in xrange(xmax):
            pos = (x,y)
            MDP[pos] = {}
            
            neswNbhrs = [None]*4    #(north,east,south,west)
            if y+1 < ymax:  neswNbhrs[0] = (x,y+1)  #north
            if x+1 < xmax:  neswNbhrs[1] = (x+1,y)  #east
            if y-1 >= 0:    neswNbhrs[2] = (x,y-1)  #south
            if x-1 >= 0:    neswNbhrs[3] = (x-1,y)  #west

            for act in cell:
                MDP[pos][act] = {}
                f = cell[act]
                assert len(f) == 6
                for next in np.nonzero(f)[0]:
                    if next < 4:        #(N,E,S,W)
                        if neswNbhrs[next] == None:   #invalid neighbor (out of bounds)-->add to catch state
                            if MDP[pos][act].has_key(catch):
                                MDP[pos][act][catch] += f[next]
                            else:
                                MDP[pos][act][catch] = f[next]
                        else:
                            MDP[pos][act][neswNbhrs[next]] = f[next]
                    elif next == 4:
                        MDP[pos][act][pos] = f[next]    #same state
                    elif next == 5:
                        if MDP[pos][act].has_key(catch):
                            MDP[pos][act][catch] += f[next]
                        else:
                            MDP[pos][act][catch] = f[next]    
    
    # Normalize all probability distributions by adding remainder to error section
    for i in MDP.keys():
        for a in MDP[i].keys():
            Z = sum(MDP[i][a].values())
            if Z == 0:
                del MDP[i][a]
            elif abs(Z-1.0)> 1e-9:
                raise "Error.  Probabilities don't sum to 1.0. abs(Z-1.0)=",abs(Z-1.0)
    
    return MDP, id2pos, pos2id



# Continuous state space
#@contdyn@
cont_state_space = pc.Polytope.from_box(np.array([[0., 2.],[0., 3.]]))

# Continuous dynamics: \dot{x} = u_x, \dot{y} = u_y
# x_{t+1} = Ax_t + Bu_t + Ew_t
# u \in U and w \in W
A = np.array([[1, 0.],[ 0., 1]])
B = np.array([[1, 0.],[ 0., 1]])
u_bound = 0.3          #control bound for TuLiP
U = pc.Polytope.from_box(np.array(u_bound*[[-1., 1.],[-1., 1.]]))

sys_dyn = hybrid.LtiSysDyn(A,B,[],[],U,[], cont_state_space)
#@contdyn_end@

# Continuous proposition
cont_props = {}
cont_props['X1'] = pc.Polytope.from_box(np.array([[0., 1.],[0., 1.]]))
cont_props['X2'] = pc.Polytope.from_box(np.array([[1., 2.],[2., 3.]]))

# Compute the proposition preserving partition of the continuous state space
cont_partition = prop2part.prop2part(cont_state_space, cont_props)
disc_dynamics = discretize.discretize(cont_partition, sys_dyn, closed_loop=True, \
                N=8, min_cell_volume=0.1, verbose=0)

print 'Num Regions:',disc_dynamics.num_regions
print 'Adjacency matrix:\n',disc_dynamics.adj.transpose()
print 'Transition matrix:\n',disc_dynamics.trans.transpose()
assert (disc_dynamics.trans.transpose() == disc_dynamics.trans).all()
assert (disc_dynamics.trans == disc_dynamics.adj).all()
###############################################################################


# Miscellaneous
w_bound_true = 0.225     #actual noise levels
numSteps = 4              #number of control steps between regions
plot_traj = False       #show plots of the system trajectories?
numSamples = 100


# Use dynamic model and sampling to create a transition system (Option 1), or else input a user
# specified transition matrix (Option 2).

#################################################################################
### OPTION 1
cell = create_cell(sys_dyn, disc_dynamics, numSteps, w_bound_true, numSamples, plot_traj)

## OPTION 2--Custom designed cell
#cell = {}
#actions = ['north','east','south','west','stop']   #actions to take  
##(N,E,S,W,self,error)
#up = 0.89; left = 0.005; right = 0.005; down = 0.0; mid = 0.095; error = 0.005   #relative directions
#assert abs(up+left+right+down+mid+error - 1.0) < 1e-12
#cell['north'] = np.array([up,    right, down,  left,  mid, error])
#cell['east']  = np.array([left,  up,    right, down,  mid, error])
#cell['south'] = np.array([down,  left,  up,    right, mid, error])
#cell['west']  = np.array([right, down,  left,  up,    mid, error])
#
#cell['stop']  = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])   #only does a self-transition (since it doesn't move)
#################################################################################


# Print transition probabilities for each region and action
print "\nPrinting estimated transition probabilities..."
print "[N, E, S, W, mid, error]"
print "From Region mid:"
for a in cell.keys():
   print "    action",a,":", cell[a]


# Create an MDP from layout + cell data
t0 = time.time()
MDP,id2pos,pos2id = create_MDP(cell, xmax, ymax)
t1 = time.time()
print '\nTime to create MDP: ',t1-t0
print 'States in system MDP: ',len(MDP)

## Print transition probabilities for each region and action
#print "\nPrinting estimated transition probabilities..."
#for r in sorted(MDP.keys()):
#    print "\nFrom Region",r,":"
#    for a in sorted(MDP[r].keys()):
#       print "    action",a,":", MDP[r][a]
    
# Print transition probabilities cell
print "\nPrinting estimated transition probabilities..."
print "From Region mid:"
print "                  [north,  east,  south,  west,   mid,   catch]"
for a in cell.keys():
   print "    action",a,":", cell[a]
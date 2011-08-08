#
# Copyright (c) 2011 by California Institute of Technology
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 
# 3. Neither the name of the California Institute of Technology nor
#    the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CALTECH
# OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.

""" 
Algorithms related to discretization containing both MATLAB interface and Python
discretization. Input calculation is available in python only but should work also
for a state space partition discretized in MATLAB.

Classes:
    - CtsSysDyn
    
Primary functions:
    - discretize
    - get_input
    
Helper functions:
    - solveFeasable
    - discretizeM
    - discretizeToMatlab
    - discretizeFromMatlab
    - getInputHelper
    - createLM
    - get_max_extreme

Created by Ufuk Topcu, 8/30/10
Modified by Nok Wongpiromsarn, 9/3/10

:Version: 0.1.0

minor refactoring by SCL <slivingston@caltech.edu>
3 May 2011.
Modified by Petter Nilsson, summer 2011
"""

import sys, os, time, subprocess
from copy import deepcopy
import numpy as np
from scipy import io as sio
from cvxopt import matrix,solvers

import polytope as pc
from prop2part import PropPreservingPartition
from errorprint import printWarning, printError, printInfo

matfile_dir = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), \
                           'tmpmat')
to_matfile = os.path.join(matfile_dir, 'dataToMatlab.mat')
from_matfile = os.path.join(matfile_dir, 'dataFromMatlab.mat')
donefile = os.path.join(matfile_dir, 'done.txt')

class CtsSysDyn:
    """CtsSysDyn class for specifying the continuous dynamics:

        s[t+1] = A*s[t] + B*u[t] + E*d[t] + K
        u[t] \in Uset - polytope object
        d[t] \in Wset - polytope object

    A CtsSysDyn object contains the fields A, B, E, K, Uset and Wset as defined above.
    
    **Constructor**:
    
    **CtsSysDyn** ([ `A` = [][, `B` = [][, `E` = [][, `K` = [][, `Uset` = [][, `Wset` = []]]]]]])
    """

    def __init__(self, A=[], B=[], E=[], K=[], Uset=None, Wset=None):
        if Uset == None:
            print "Warning: Uset not given in CtsSysDyn()"
        self.A = A
        self.B = B
        self.E = E
        self.K = K
        self.Uset = Uset
        self.Wset = Wset

def discretize(part, ssys, N=10, min_cell_volume=0.1, closed_loop=False,  \
               use_mpt=True, conservative=False, max_num_poly=5, \
               use_all_horizon=False, remove_trans=False, abs_tol=1e-7, verbose=0):

    """Refine the partition and establish transitions based on reachability
    analysis.
    
    Input:
    
    - `part`: a PropPreservingPartition object
    - `ssys`: a CtsSysDyn object
    - `N`: horizon length
    - `min_cell_volume`: the minimum volume of cells in the resulting partition 
    - `closed_loop`: boolean indicating whether the `closed loop` algorithm
                     should be used. default False
    - `use_mpt`: if True, use MPT-based abstraction algorithm
    - `conservative`: if true, force sequence in reachability analysis to stay
                      inside starting cell. If false, safety is ensured by 
                      keeping the sequence inside the original proposition
                      preserving cell which needs to be convex
    - `max_num_poly`: maximum number of polytopes in a region to use in 
                      reachability analysis
    - `use_all_horizon`: in closed loop algorithm: if we should look for reach-
                         ability also in less than N steps
    - `remove_trans`: if True, remove found transitions between non-neighbors
    - `abs_tol`: maximum volume for an "empty" polytope
    - `verbose`: level of verbosity
    
    Output:
    
    - A PropPreservingPartition object with transitions
    """
    
    orig_list = []
    
    min_cell_volume = (min_cell_volume/np.finfo(np.double).eps ) * np.finfo(np.double).eps
    
    # We want to save original polytopes, require them to be convex 
    for poly in part.list_region:
        if len(poly) == 0:
            orig_list.append(poly.copy())
        elif len(poly) == 1:
            orig_list.append(poly.list_poly[0].copy())
        else:
            if not conservative:
                raise Exception("solveFeasible: original list contains non-convex \
                            polytope regions")
            else:
                orig_list.append(poly.copy())
    
    orig = range(len(orig_list))
    
    if use_mpt:
        return discretizeM(part, ssys, N = N, auto=True, minCellVolume = min_cell_volume, \
                    maxNumIterations = 5, useClosedLoopAlg = closed_loop, \
                    useAllHorizonLength=use_all_horizon, useLargeSset = False, \
                    timeout = -1, maxNumPoly = max_num_poly, verbose = 0)
    
    # Cheby radius of disturbance set
    if len(ssys.E) > 0:
        rd,xd = pc.cheby_ball(ssys.Wset)
    else:
        rd = 0.
    
    # Initialize matrix for pairs to check
    IJ = part.adj.copy()
    if verbose > 1:
        print "\n Starting IJ: \n" + str(IJ)
    
    # Initialize output
    transitions = np.zeros([part.num_regions,part.num_regions], dtype = int)
    sol = deepcopy(part.list_region)
    adj = part.adj.copy()

    while np.sum(IJ) > 0:
        ind = np.nonzero(IJ)
        i = ind[0][0]
        j = ind[1][0]
        
        si = sol[i]
        sj = sol[j]
        
        if verbose > 1:        
            print "\n Working with states " + str(i) + " and " + str(j) \
                  + " with lengths " + str(len(si)) + " and " + str(len(sj))
        
        if conservative:
            S0 = solveFeasable(si,sj,ssys,N, closed_loop=closed_loop, 
                        min_vol=min_cell_volume, max_num_poly=max_num_poly,\
                        use_all_horizon=use_all_horizon)
        else:
            S0 = solveFeasable(si,sj,ssys,N, closed_loop=closed_loop,\
                    min_vol=min_cell_volume, trans_set=orig_list[orig[i]], \
                    use_all_horizon=use_all_horizon, max_num_poly=max_num_poly)
        
        if verbose > 1:
            print "Computed reachable set S0 with volume " + str(pc.volume(S0))
        
        isect = pc.intersect(si, S0)
        risect, xi = pc.cheby_ball(isect)
        vol1 = pc.volume(isect)

        diff = pc.mldivide(si, S0)
        rdiff, xd = pc.cheby_ball(isect)
        vol2 = pc.volume(diff)
        
        # We don't want our partitions to be smaller than the disturbance set
        # Could be a problem since cheby radius is calculated for smallest
        # convex polytope, so if we have a region we might throw away a good
        # cell.
        if (vol1 > min_cell_volume) & (vol2 > min_cell_volume) & (rdiff > rd) & \
            (risect > rd):
        
            # Make sure new areas are Regions and add proposition lists
            if len(isect) == 0:
                isect = pc.Region([isect], si.list_prop)
            else:
                isect.list_prop = si.list_prop
        
            if len(diff) == 0:
                diff = pc.Region([diff], si.list_prop)
            else:
                diff.list_prop = si.list_prop
        
            # Add new states
            sol[i] = isect
            difflist = pc.separate(diff)    # Want only connected states
            num_new = len(difflist)
            for reg in difflist:
                sol.append(reg)
            size = len(sol)
            
            # Add transitions
            transitions = np.hstack([transitions, np.zeros([size-num_new, num_new], dtype=int) ])
            transitions = np.vstack([transitions, np.zeros([num_new, size], dtype=int) ])
            
            transitions[:, i] = np.zeros(size)
            for kk in range(num_new):
            
                transitions[size-1-kk,:] = transitions[i,:] # All sets reachable from start are reachable from both part's
                transitions[size-1-kk,i] = 0                # except possibly the new part
                transitions[size-1-kk,j] = 0            

            transitions[i,j] = 1        # sol[j] is reachable from intersection of sol[i] and S0..
    
            # Take care of adjacency
            old_adj = np.nonzero(adj[i,:])[0]
            adj[i,:] = np.zeros([size-num_new])
            adj[:,i] = np.zeros([size-num_new])
            
            adj = np.hstack([adj, np.zeros([size-num_new, num_new], dtype=int) ])
            adj = np.vstack([adj, np.zeros([num_new, size], dtype=int) ])
            
            for kk in range(num_new):
                adj[i,size-1-kk] = 1
                adj[size-1-kk,i] = 1
                adj[size-1-kk,size-1-kk] = 1
                orig = np.hstack([orig, orig[i]])
            adj[i,i] = 1
                        
            if verbose > 1:
                output = "\n Adding states " + output(i) + " and "
                for kk in range(num_new):
                    output += output(size-1-kk) + " and "
                print output + "\n"
                        
            for k in np.setdiff1d(old_adj,[i,size-1]):
                # Every "old" neighbor must be the neighbor of at least one of the new
                if pc.is_adjacent(sol[i],sol[k]):
                    adj[i,k] = 1
                    adj[k,i] = 1
                elif remove_trans:
                    # Actively remove transitions between non-neighbors
                    transitions[i,k] = 0
                    transitions[k,i] = 0
                for kk in range(num_new):
                    if pc.is_adjacent(sol[size-1-kk],sol[k]):
                        adj[size-1-kk,k] = 1
                        adj[k, size-1-kk] = 1
                    elif remove_trans:
                        # Actively remove transitions between non-neighbors
                        transitions[size-1-kk,k] = 0
                        transitions[k,size-1-kk] = 0
            
            # Just add adjacent cells for checking, unless transition already found            
            IJ = np.hstack([IJ, np.zeros([size-num_new, num_new], dtype=int) ])
            IJ = np.vstack([IJ, np.zeros([num_new, size], dtype=int) ])
            horiz1 = adj[i,:] - transitions[i,:] > 0
            verti1 = adj[:,i] - transitions[:,i] > 0
            IJ[i,:] = horiz1.astype(int)
            IJ[:,i] = verti1.astype(int)
            for kk in range(num_new):      
                horiz2 = adj[size-1-kk,:] - transitions[size-1-kk,:] > 0
                verti2 = adj[:,size-1-kk] - transitions[:,size-1-kk] > 0
                IJ[size-1-kk,:] = horiz2.astype(int)
                IJ[:,size-1-kk] = verti2.astype(int)      
            
            if verbose > 1:
                print "\n Updated adj: \n" + str(adj)
                print "\n Updated trans: \n" + str(transitions)
                print "\n Updated IJ: \n" + str(IJ)
            
        elif vol2 < abs_tol:
            if verbose > 1:
                print "Transition found"
            transitions[i,j] = 1
            IJ[i,j] = 0
        
        else:
            if verbose > 1:
                print "No transition found, diff vol: " + str(vol2) + \
                      ", intersect vol: " + str(vol1)
            transitions[i,j] = 0
            IJ[i,j] = 0
                  
    new_part = PropPreservingPartition(domain=part.domain, num_prop=part.num_prop, list_region=sol, num_regions=len(sol), adj=np.array([]), trans=transitions, list_prop_symbol=part.list_prop_symbol, orig_list_region=orig_list, orig=orig)                           
    return new_part

def discretize_overlap(part, ssys, N=10, min_cell_volume=0.1, closed_loop=False,\
               conservative=False, max_num_poly=5, use_mpt=False, \
               use_all_horizon=False, abs_tol=1e-7, verbose=0):

    """Refine the partition and establish transitions based on reachability
    analysis.
    
    Input:
    
    - `part`: a PropPreservingPartition object
    - `ssys`: a CtsSysDyn object
    - `N`: horizon length
    - `min_cell_volume`: the minimum volume of cells in the resulting partition 
    - `closed_loop`: boolean indicating whether the `closed loop` algorithm
                     should be used. default False
    - `conservative`: if true, force sequence in reachability analysis to stay
                      inside starting cell. If false, safety is ensured by 
                      keeping the sequence inside the original proposition
                      preserving cell
    - `max_num_poly`: maximum number of polytopes in a region to use in 
                      reachability analysis
    - `use_all_horizon`: in closed loop algorithm: if we should look for reach-
                         ability also in less than N steps
    - `abs_tol`: maximum volume for an "empty" polytope
    
    Output:
    
    - A PropPreservingPartition object with transitions
    """
    min_cell_volume = (min_cell_volume/np.finfo(np.double).eps ) * np.finfo(np.double).eps
    
    orig_list = []
    
    for poly in part.list_region:
        if len(poly) == 0:
            orig_list.append(poly.copy())
        elif len(poly) == 1:
            orig_list.append(poly.list_poly[0].copy())
        else:
            raise Exception("solveFeasible: original list contains non-convex \
                            polytope regions")
    
    orig = range(len(orig_list))
    
    # Cheby radius of disturbance set
    if len(ssys.E) > 0:
        rd,xd = pc.cheby_ball(ssys.Wset)
    else:
        rd = 0.
    
    # Initialize matrix for pairs to check
    IJ = part.adj.copy()
    if verbose > 1:
        print "\n Starting IJ: \n" + str(IJ)

    # Initialize output
    transitions = np.zeros([part.num_regions,part.num_regions], dtype = int)
    sol = deepcopy(part.list_region)
    adj = part.adj.copy()
    
    # List of how many "new" regions that have been created for each region
    # and a list of original number of neighbors
    num_new_reg = np.zeros(len(orig_list))
    num_orig_neigh = np.sum(adj, axis=1).flatten() - 1

    while np.sum(IJ) > 0:
        ind = np.nonzero(IJ)
        i = ind[0][0]
        j = ind[1][0]
                
        IJ[i,j] = 0
        num_new_reg[i] += 1
        print num_new_reg
        si = sol[i]
        sj = sol[j]
        
        if verbose > 1:        
            print "\n Working with states " + str(i) + " and " + str(j)
        
        if conservative:
            S0 = solveFeasable(si,sj,ssys,N, closed_loop=closed_loop, 
                        min_vol=min_cell_volume, max_num_poly=max_num_poly,\
                        use_all_horizon=use_all_horizon)
        else:
            S0 = solveFeasable(si,sj,ssys,N, closed_loop=closed_loop,\
                    min_vol=min_cell_volume, trans_set=orig_list[orig[i]], \
                    use_all_horizon=use_all_horizon, max_num_poly=max_num_poly)
        
        if verbose > 1:
            print "Computed reachable set S0 with volume " + str(pc.volume(S0))
        
        isect = pc.intersect(si, S0)
        risect, xi = pc.cheby_ball(isect)
        vol1 = pc.volume(isect)

        diff = pc.mldivide(si, S0)
        rdiff, xd = pc.cheby_ball(diff)
        
        # We don't want our partitions to be smaller than the disturbance set
        # Could be a problem since cheby radius is calculated for smallest
        # convex polytope, so if we have a region we might throw away a good
        # cell.
        
        if rdiff < abs_tol:
            if verbose > 1:
                print "Transition found"
            transitions[i,j] = 1
        
        elif (vol1 > min_cell_volume) & (risect > rd) & \
                (num_new_reg[i] <= num_orig_neigh[i]+1):
        
            # Make sure new cell is Region and add proposition lists
            if len(isect) == 0:
                isect = pc.Region([isect], si.list_prop)
            else:
                isect.list_prop = si.list_prop
        
            # Add new state
            sol.append(isect)
            size = len(sol)
            
            # Add transitions
            transitions = np.hstack([transitions, np.zeros([size - 1, 1], dtype=int) ])
            transitions = np.vstack([transitions, np.zeros([1, size], dtype=int) ])
            
            transitions[size-1,:] = transitions[i,:] # All sets reachable from orig cell are reachable from both cells
            transitions[size-1,j] = 1   # j is reachable from new cell            
            
            # Take care of adjacency
            old_adj = np.nonzero(adj[i,:])[0]
            
            adj = np.hstack([adj, np.zeros([size - 1, 1], dtype=int) ])
            adj = np.vstack([adj, np.zeros([1, size], dtype=int) ])
            adj[i,size-1] = 1
            adj[size-1,i] = 1
            adj[size-1,size-1] = 1
                                    
            for k in np.setdiff1d(old_adj,[i,size-1]):
                if pc.is_adjacent(sol[size-1],sol[k],overlap=True):
                    adj[size-1,k] = 1
                    adj[k, size-1] = 1
                else:
                    # Actively remove (valid) transitions between non-neighbors
                    transitions[size-1,k] = 0
                    transitions[k,size-1] = 0
                    
            # Assign original proposition cell to new state and update counts
            orig = np.hstack([orig, orig[i]])
            print num_new_reg
            num_new_reg = np.hstack([num_new_reg, 0])
            num_orig_neigh = np.hstack([num_orig_neigh, np.sum(adj[size-1,:])-1])
            
            if verbose > 1:
                print "\n Adding state" + str(size-1) + "\n"
            
            # Just add adjacent cells for checking, unless transition already found            
            IJ = np.hstack([IJ, np.zeros([size - 1, 1], dtype=int) ])
            IJ = np.vstack([IJ, np.zeros([1, size], dtype=int) ])
            horiz2 = adj[size-1,:] - transitions[size-1,:] > 0
            verti2 = adj[:,size-1] - transitions[:,size-1] > 0
            IJ[size-1,:] = horiz2.astype(int)
            IJ[:,size-1] = verti2.astype(int)      
            
            #if verbose > 1:
                #print "\n Updated adj: \n" + str(adj)
                #print "\n Updated trans: \n" + str(transitions)
                #print "\n Updated IJ: \n" + str(IJ)
                    
        else:
            if verbose > 1:
                print "No transition found, intersect vol: " + str(vol1)
            transitions[i,j] = 0
                  
    new_part = PropPreservingPartition(domain=part.domain, num_prop=part.num_prop,
                                       list_region=sol, num_regions=len(sol), adj=np.array([]), 
                                       trans=transitions, list_prop_symbol=part.list_prop_symbol,
                                       orig_list_region=orig_list, orig=orig)                           
    return new_part

def get_input(x0, ssys, part, start, end, N, R, Q, conservative=False, \
                test_result=False):
 
    """Calculate an input signal sequence taking the plant from state `start` 
    to state `end` in the partition part, such that x'Rx + u'Qu is minimal.
        
    Input:
    - `x0`: initial continuous state
    - `ssys`: CtsSysDyn object specifying system dynamics
    - `part`: PropPreservingPartition object specifying the state space partition
    - `start`: int specifying the number of the initial state in `part`
    - `end`: int specifying the number of the end state in `part`
    - `N`: the horizon length
    - `R`: state cost matrix for x = [x(1)' x(2)' .. x(N)']', 
           size (N*xdim x N*xdim)
    - `Q`: input cost matrix for u = [u(0)' u(1)' .. u(N-1)']', 
           size (N*udim x N*udim)
    - `conservative`: if True, force plant to stay inside initial state during
                      execution. if False, plant is forced to stay inside
                      the original proposition preserving cell
    - `test_result`: performs a simulation (without disturbance) to make sure
                     that the calculated input sequence is safe
    
    Output:
    - A (N x m) numpy array where row k contains u(k) for k = 0,1 ... N-1.
    
    
    Note1: The same horizon length as in reachability analysis should be 
           used in order to guarantee feasibility.
    
    Note2: If the closed loop algorithm has been used to compute reachability 
    the input needs to be recalculated for each time step (with decreasing 
    horizon length). In this case only u(0) should be used as a control signal 
    and u(1) ... u(N-1) thrown away.
    
    Note3: The "conservative" calculation makes sure that the plants remains 
    inside the convex hull of the starting region during execution, i.e. 
    x(1), x(2) ...  x(N-1) are in conv_hull(starting region). 
    If the original proposition preserving partition is not convex, 
    safety can not be guaranteed."""
    
    if (len(R) == 0) and (len(Q) == 0):
        Q = np.eye(N*ssys.B.shape[1])
        R = np.zeros([N*x0.size, N*x0.size])
    
    if (R.shape[0] != R.shape[1]) or (R.shape[0] != N*x0.size):
        raise Exception("get_input: R must be square and have side\
                        N * dim(state space)")
    
    if (Q.shape[0] != Q.shape[1]) or (Q.shape[0] != N*ssys.B.shape[1]):
        raise Exception("get_input: Q must be square and have side \
                        N * dim(input space)")
    
    if part.trans[start,end] != 1:
        raise Exception("get_input: no transition from state " + str(start) + \
            " to state " + str(end))
    
    if (not conservative) & part.orig == None:
        print "List of original proposition preserving partitions not given, \
              reverting to conservative mode"
        conservative = True
        
    P_start = part.list_region[start]
    P2 = ssys.Uset
    P_end = part.list_region[end]
    
    if not pc.is_inside(P_start, x0):
        raise Exception("getInput: x0 not inside given initial state")

    n = ssys.A.shape[1]
    m = ssys.B.shape[1]
    
    if conservative:
        # Take convex hull or P_start as constraint
        if len(P_start) > 0:
            if len(P_start.list_poly) > 1:
                # Take convex hull
                vert = pc.extreme(P_start.list_poly[0])
                for i in range(1, len(P_start.list_poly)):
                    vert = np.hstack([vert, extreme(P_start.list_poly[i])])
                P1 = pc.qhull(vert)
            else:
                P1 = P_start.list_poly[0]
        else:
            P1 = P_start
    else:
        # Take original proposition preserving cell as constraint
        P1 = part.orig_list_region[part.orig[start]]
    
    if len(P_end) > 0:
        low_cost = np.inf
        low_u = np.zeros([N,m])
        for i in range(len(P_end.list_poly)):
            P3 = P_end.list_poly[i]
            u, cost = getInputHelper(x0, ssys, P1, P2, P3, N, R, Q)
            if cost < low_cost:
                low_u = u
    else:
        P3 = P_end
        low_u, cost = getInputHelper(x0, ssys, P1, P2, P3, N, R, Q)
    
    if test_result:
        good = is_seq_inside(x0, low_u, ssys, P1, P3)
        if not good:
            print "Calculated sequence not good"
    return low_u

def solveFeasable(P1, P2, ssys, N, min_vol=0.1, max_cell=10, \
                    closed_loop=False, use_all_horizon=False, trans_set=None, \
                    max_num_poly=5):

    '''Computes the subset x0 of `P1' from which `P2' is reachable
    in horizon `N', with respect to system dynamics `ssys'. The closed
    loop algorithm solves for one step at a time, which keeps the dimension
    of the polytopes down.
    
    Input:
    `P1`: A Polytope or Region object
    `P2`: A Polytope or Region object
    `ssys`: A CtsSysDyn object
    `N`: The horizon length
    `closed_loop`: If true, take 1 step at the time to keep dimension down
    `trans_set`: If specified, force transitions to be in this set. If empty,
                 P1 is used
                
    Output:
    `x0`: A Polytope or Region object defining the set in P1 from which
          P2 is reachable
    '''
    part1 = P1.copy() # Initial set
    part2 = P2.copy() # Terminal set
    
    if closed_loop:
        temp_part = part2
        for i in range(N,0,-1): 
            x0 = solveFeasable(part1, temp_part, ssys, 1, closed_loop=False,
                                trans_set=trans_set)    
            if use_all_horizon:
                temp_part = pc.union(x0, temp_part, check_convex=True)
            else:
                temp_part = x0
                if not pc.is_fulldim(temp_part):
                    return pc.Polytope()
        return temp_part 
        
    if len(part1) > max_num_poly:
        # Just use the max_num_poly largest volumes for reachability
        vol_list = np.zeros(len(part1))
        for i in range(len(part1)):
            vol_list[i] = pc.volume(part1.list_poly[i])
        ind = np.argsort(-vol_list)
        temp = []
        for i in ind[range(max_num_poly)]:
            temp.append(part1.list_poly[i])
        part1 = pc.Region(temp, [])

    if len(part2) > max_num_poly:
        # Just use the max_num_poly largest volumes for reachability
        vol_list = np.zeros(len(part2))
        for i in range(len(part2)):
            vol_list[i] = pc.volume(part2.list_poly[i])
        ind = np.argsort(-vol_list)
        temp = []
        for i in ind[range(max_num_poly)]:
            temp.append(part2.list_poly[i])
        part2 = pc.Region(temp, [])
    
    if len(part1) > 0:
        # Recursive union of sets
        poly = pc.Polytope()
        
        for i in range(len(part1)):
            s0 = solveFeasable(part1.list_poly[i], part2, ssys, N,\
                               trans_set=trans_set)
            if pc.volume(s0) > min_vol/10:
                poly = pc.union(poly, s0, check_convex=True)
        return poly
    
    if len(part2) > 0:
        # Recursive intersection of sets 
        poly = solveFeasable(part1, part2.list_poly[0], ssys, N, \
                             trans_set=trans_set)
        if len(part2) == 1:
            return poly
        else:
            for i in range(1, len(part2)):
                s0 = solveFeasable(part1, part2.list_poly[i], ssys, N, \
                                   trans_set=trans_set)
                poly = pc.intersect(poly, s0)
            return poly
    
    L1 = part1.A
    M1 = part1.b.reshape(part1.b.size,1)  
    L2 = ssys.Uset.A
    M2 = ssys.Uset.b.reshape(ssys.Uset.b.size,1).copy()
    L3 = part2.A
    M3 = part2.b.reshape(part2.b.size,1)
            
    if trans_set == None:
        trans_set = part1

    L,M = createLM(ssys,N,L1,M1,trans_set.A,trans_set.b,L2,M2,L3,M3)
    
    # Ready to make polytope
    poly1 = pc.reduce(pc.Polytope(L,M))
    # Project poly1 onto lower dim
    n = np.shape(ssys.A)[1]
    poly1 = pc.projection(poly1, range(1,n+1))
    return pc.reduce(poly1)

def discretizeM(part, ssys, N = 10, auto=True, minCellVolume = 0.1, \
                    maxNumIterations = 5, useClosedLoopAlg = True, \
                    useAllHorizonLength = True, useLargeSset = False, \
                    timeout = -1, maxNumPoly = 5, verbose = 2):

    """Discretize the continuous state space using MATLAB implementation.
    
    Input:
    
    - `part`: a PropPreservingPartition object
    - `ssys`: a CtsSysDyn object
    - `N`: horizon length
    - `auto`: a boolean that indicates whether to automatically run the MATLAB  
      implementation of discretize.
    - `minCellVolume`: the minimum volume of cells in the resulting partition
    - `maxNumIterations`: the maximum number of iterations
    - `useClosedLoopAlg`: a boolean that indicates whether to use the closed loop algorithm.
      For the difference between the closed loop and the open loop algorithm, 
      see Borrelli, F. Constrained Optimal Control of Linear and Hybrid Systems, 
      volume 290 of Lecture Notes in Control and Information Sciences. Springer. 2003.
    - `useAllHorizonLength`: a boolean that indicates whether all the horizon length up
      to probStruct.N can be used. This option is relevant only when the closed 
      loop algorithm is used.
    - `useLargeSset`: a boolean that indicates whether when solving the reachability
      problem between subcells of the original partition, the cell of the
      original partition should be used for the safe set.
    - `timeout`: timeout (in seconds) for polytope union operation. 
      If negative, the timeout won't be used. Note that using timeout requires MATLAB
      parallel computing toolbox.
    - `maxNumPoly`: the maximum number of polytopes in a region used in computing reachability.
    - `verbose`: level of verbosity
    """
        
    if (os.path.isfile(globals()["to_matfile"])):
        os.remove(globals()["to_matfile"])
    if (os.path.isfile(globals()["from_matfile"])):
        os.remove(globals()["from_matfile"])
    if (os.path.isfile(globals()["donefile"])):
        os.remove(globals()["donefile"])
    
    starttime = time.time()
    discretizeToMatlab(part, ssys, N, minCellVolume, \
                           maxNumIterations, useClosedLoopAlg, \
                           useAllHorizonLength, useLargeSset, \
                           timeout, maxNumPoly, verbose)

    if (auto):
        try:
            mpath = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'matlab')
            mcommand = "addpath('" + mpath + "'); p = '" + matfile_dir + "';"
            mcommand += "try, runDiscretizeMatlab; catch, disp(lasterr); quit; end;"
            mcommand += "quit;"
            cmd = subprocess.call( \
                ["matlab", "-nojvm", "-nosplash", "-r", mcommand])
            auto = True
        except:
            printError("Cannot run matlab. Please make sure that MATLAB is in your PATH.")
            auto = False

        if (not os.path.isfile(globals()["donefile"]) or \
                os.path.getmtime(globals()["donefile"]) <= \
                os.path.getmtime(globals()["to_matfile"])):
            printError("Discretization failed!")
            auto = False

    if (not auto):
        printInfo("\nPlease run 'runDiscretizeMatlab' in the 'matlab' folder.\n")
        print("Waiting for MATLAB output...")

        while (not os.path.isfile(globals()["donefile"]) or \
                   os.path.getmtime(globals()["donefile"]) <= \
                   os.path.getmtime(globals()["to_matfile"])):
            if (verbose > 0):
                print("Waiting for MATLAB output...")
            time.sleep(10)

    dyn = discretizeFromMatlab(part)
    return dyn

def discretizeToMatlab(part, ssys, N = 10, minCellVolume=0.1, \
                           maxNumIterations=5, useClosedLoopAlg=True, \
                           useAllHorizonLength=True, useLargeSset=False, \
                           timeout=-1, maxNumPoly=5, verbose=0):

    """Generate an input file for MATLAB implementation of discretize.
    
    Input:
    
    - `part`: a PropPreservingPartition object
    - `ssys`: a CtsSysDyn object
    - `N`: horizon length
    - `minCellVolume`: the minimum volume of cells in the resulting partition
    - `maxNumIterations`: the maximum number of iterations
    - `useClosedLoopAlg`: a boolean that indicates whether to use the closed loop algorithm.
      For the difference between the closed loop and the open loop algorithm, 
      see Borrelli, F. Constrained Optimal Control of Linear and Hybrid Systems, 
      volume 290 of Lecture Notes in Control and Information Sciences. Springer. 2003.
    - `useAllHorizonLength`: a boolean that indicates whether all the horizon length up
      to probStruct.N can be used. This option is relevant only when the closed 
      loop algorithm is used.
    - `useLargeSset`: a boolean that indicates whether when solving the reachability
      problem between subcells of the original partition, the cell of the
      original partition should be used for the safe set.
    - `timeout`: timeout (in seconds) for polytope union operation. 
      If negative, the timeout won't be used. Note that using timeout requires MATLAB
      parallel computing toolbox.
    - `maxNumPoly`: the maximum number of polytopes in a region used in computing reachability.
    - `verbose`: level of verbosity
    """

    data = {}
    adj = deepcopy(part.adj)
    for i in xrange(0, len(adj)):
        adj[i][i] = 1
    data['adj'] = adj
    data['minCellVolume'] = minCellVolume
    data['maxNumIterations'] = maxNumIterations
    data['useClosedLoopAlg'] = useClosedLoopAlg
    data['useAllHorizonLength'] = useAllHorizonLength
    data['useLargeSset'] = useLargeSset
    data['timeout'] = timeout
    data['maxNumPoly'] = maxNumPoly
    data['verbose'] = verbose
    data['A'] = ssys.A
    data['B'] = ssys.B
    data['E'] = ssys.E
    data['UsetA'] = ssys.Uset.A
    data['Usetb'] = ssys.Uset.b
#    data['Uset'] = ssys.Uset
    if (isinstance(ssys.Wset, pc.Polytope)):
        data['WsetA'] = ssys.Wset.A
        data['Wsetb'] = ssys.Wset.b
    else:
        data['WsetA'] = []
        data['Wsetb'] = []
#    data['Wset'] = ssys.Wset
    data['N'] = N
    numpolyvec = []
    den = []
    for i1 in range(0,len(part.list_region)):
        numpolyvec.append(len(part.list_region[i1].list_poly))
        for i2 in range(0,len(part.list_region[i1].list_poly)):
            pp = part.list_region[i1].list_poly[i2]
            data['Reg'+str(i1+1)+'Poly'+str(i2+1)+'Ab'] = np.concatenate((pp.A,np.array([pp.b]).T),1)
    data['numpolyvec'] = numpolyvec
    matfile = globals()["to_matfile"]
    if (not os.path.exists(os.path.abspath(os.path.dirname(matfile)))):
        os.mkdir(os.path.abspath(os.path.dirname(matfile)))
    sio.savemat(matfile, data)
    print('MATLAB input saved to ' + matfile)

def discretizeFromMatlab(origPart):
    """Load the data from MATLAB discretize implementation.

    Input:

    - origPart: a PropPreservingPartition object
    """
    matfile = globals()["from_matfile"]
    if (os.path.getmtime(matfile) <= os.path.getmtime(globals()["to_matfile"])):
        printWarning("The MATLAB output file is older than the MATLAB input file.")
        cont = raw_input('Continue [c]?: ')
        if (cont.lower() != 'c'):
            return False

    print('Loading MATLAB output from ' + matfile)
    data = sio.loadmat(matfile)
    trans = data['trans']
    a1 = data['numNewCells']
    numNewCells = np.zeros((a1.shape[0],1))
    numNewCells[0:,0] = a1[:,0]
    newCellVol = data['newCellVol']
    num_cells = data['num_cells'][0][0]
    a2 = data['numpoly']
    numpoly = np.zeros(a2.shape)
    numpoly[0:,0:] = a2[0:,0:]
	
    regs = []
    for i1 in range(0,num_cells):
        for i2 in range(0,numNewCells[i1]):
            polys = []
            props = []
            for i3 in range(0,int(numpoly[i1,i2])):
                Ab = data['Cell'+str(i1)+'Reg'+str(i2)+'Poly'+str(i3)+'Ab']
                A = deepcopy(Ab[:,0:-1])
                b = np.zeros((A.shape[0],1))
                b[0:,0] = deepcopy(Ab[:,-1])
                polys.append(pc.Polytope(A,b))

            props = origPart.list_region[i1].list_prop
            regs.append(pc.Region(polys,props))	
				
    domain = deepcopy(origPart.domain)
    num_prop = deepcopy(origPart.num_prop)
    num_regions = len(regs)
    list_prop_symbol = deepcopy(origPart.list_prop_symbol)
    newPartition = PropPreservingPartition(domain, num_prop, regs, num_regions, [], \
                                               trans, list_prop_symbol)
    return newPartition

def getInputHelper(x0, ssys, P1, P2, P3, N, R, Q):

    if R.size == 0:
        R = np.zeros([x0.size*N, x0.size*N])
        
    if Q.size == 0:
        Q = np.zeros([P2.shape[1]*N, P2.shape[1]*N])

    n = ssys.A.shape[1]
    m = ssys.B.shape[1]
    
    L,M = createLM(ssys, N, P1.A, P1.b, P1.A, P1.b, P2.A, P2.b, P3.A, P3.b)
                    
    # Remove first constraint on x(0)
    L = L[range(P1.A.shape[0], L.shape[0]),:]
    M = M[range(P1.A.shape[0], M.shape[0]),:]
    
    # Separate L matrix
    Lx = L[:,range(n)]
    Lu = L[:,range(n,L.shape[1])] 
    
    M = M - np.dot(Lx, x0).reshape(Lx.shape[0],1)
    
    # Constraints
    G = matrix(Lu)
    h = matrix(M)
    
    # Build matrices Ct and An
    # Ct = [ B 0 ... 0; AB B 0 ... 0 ; A^2B AB B 0..0; ... ; A^(N-1)B ... B]
    # Kt = [ K; AK + K; A^2K+AK+K; ... ; A^(N-1)K + ... + K ]
    # An = [ A; A^2; A^3; ... ; A^N-1]
    b0 = ssys.B.shape[0]
    b1 = ssys.B.shape[1]
    aa = ssys.A.shape[0]
    Amatr = ssys.A.copy()
    ABmatr = ssys.B.copy()
    Ct = np.zeros([b0*N, b1*N])
    An = np.zeros([aa*N, aa])
    for i in range(N):
        for j in range(N-i):
            Ct[np.ix_(range((i+j)*b0, (i+j+1)*b0), range(j*b1,(j+1)*b1))] = i #ABmatr
        An[np.ix_(range(i*aa, (i+1)*aa), range(aa))] = i #Amatr
        
        Amatr = np.dot(ssys.A, Amatr)
        ABmatr = np.dot(ssys.A, ABmatr)
    An_temp = np.vstack([np.eye(ssys.A.shape[0]), An[range(An.shape[0] - ssys.A.shape[0]),:] ])
    Kt = np.dot(An_temp,ssys.K)
    for i in range(1,N):
        Kt[range(i*ssys.A.shape[0], (i+1)*ssys.A.shape[0]),:] += Kt[range((i-1)*ssys.A.shape[0], i*ssys.A.shape[0]),:] 

    # Cost function wrt u
    # min u'(Q + Ct'*R*Ct)u + 2 (x0'*An' + Kt)*R*Ct*u
    P = matrix(Q + np.dot(Ct.T, np.dot(R, Ct)))
    q = matrix(np.dot( np.dot(x0.reshape(1,x0.size), An.T) + Kt.T , np.dot(R, Ct) )).T
        
    sol = solvers.qp(P,q,G,h)
    
    if sol['status'] != "optimal":
        raise Exception("get_input: QP returned status " + str(sol['status']))
    
    u = np.array(sol['x']).flatten()
    cost = sol['primal objective']
    
    return u.reshape(N, m), cost

def createLM(ssys,N,L0,M0,L1,M1,L2,M2,L3,M3):
    """Compute the components of the polytope L [x(0)' u(0)' ... u(N-1)']' <= M
    which stacks the following constraints
    
    - L0 x(0) <= M0
    - L1 x(k) <= M1 for k= 1 ... N-1
    - L2 u(k) <= M2 for all k
    - L3 x(N) <= M3
    - x(t+1) = A x(t) + B u(t) + E d(t)
    
    The returned polytope describes the intersection of the polytopes for all
    possible disturbances
    
    Input:
    - `ssys`: CtsSysDyn dynamics
    - `N`: horizon length
    - `L1,M3,L2,M2,L3,M3`: polytopes for the constraints
    """
    
    A = ssys.A
    B = ssys.B
    E = ssys.E
    D = ssys.Wset
    K = ssys.K
    
    n = A.shape[1]  # State space dimension
    m = B.shape[1]  # Input space dimension
    if len(E) > 0:
        if pc.is_fulldim(D):
            p = E.shape[1]  # Disturbance space dimension
        else:
            p = 0
    else:
        p = 0

    L0n = np.shape(L0)[0]    
    L1n = np.shape(L1)[0]
    L2n = np.shape(L2)[0]
    L3n = np.shape(L3)[0]

    Lupper = np.zeros([L0n + L1n*(N-1),n+N*m])
    Lmiddle = np.zeros([L2n*N,n+N*m])
    Llower = np.zeros([L3n,n+N*m])
    
    if p > 0:
        # We have disturbance
        # This case is not yet tested!!
        Gupper = np.zeros([L0n + L1n*(N-1), p*N])
        Gmiddle = np.zeros([L2n*N, p*N])
        Glower = np.zeros([L3n, p*N])
        
    Lupper[np.ix_(range(L0n), range(n))] = L0
    
    if len(K) > 0:
        K_it = np.zeros(K.shape)
        Karr = np.zeros([L0n + L1n*(N-1) + L2n*N + L3n,1])  
    Amatr = np.eye(n)
    for i in range(N):

        ######### FOR L #########
        L1A = np.dot(L1,Amatr)
        L1AB = np.dot(L1A,B)
        L3AB = np.dot(np.dot(L3,Amatr),B)
        if i > 0:
            Lupper[np.ix_(range(L0n + (i-1)*L1n,L0n + i*L1n),range(n))] = L1A
        for pos in range(N-i-1):
            Lupper[np.ix_(range(L0n + (i+pos)*L1n,L0n + (i+pos+1)*L1n), \
                          range(n+(pos)*m,n+(pos+1)*m))] = L1AB
   
        Lmiddle[np.ix_( range(i*L2n,(i+1)*L2n),range(n + i*m,n + (i+1)*m)) ] = L2
        Llower[np.ix_(range(L3n), range(n+N*m - (i+1)*m, n+N*m - i*m))] = L3AB

        ######### FOR G #########
        if p > 0:
            L1AE = np.dot(L1A,E)
            L3AE = np.dot(np.dot(L3,Amatr),E)

            for pos in range(N-NN-1):
                Gupper[ np.ix_(range(L0n + (i+pos)*L1n, L0n + (1+i+pos)*L1n),\
                               range( pos * p, (pos + 1) * p) ) ] = L1AE

            Glower[np.ix_(range(L3n), range(N*p- (i+1)*p, N*p - i*p ))] = L3AE
        
        ######### FOR K ############
        if len(K) > 0:
            K_it = K_it + np.dot(Amatr,K)
            if i < N-1:
                Karr[range(L0n + i*L1n, L0n + (i+1)*L1n), :] = np.dot(L1,K_it)
          
	    #### Iterate
        Amatr = np.dot(Amatr, A)
    
    # "First" entry of Llower
    Llower[np.ix_(range(L3n), range(n))] = np.dot(L3,Amatr)

    # Put together matrices L, M
    L = np.vstack([Lupper,np.vstack([Lmiddle,Llower]) ])
    
    Mfirst = M0.reshape(M0.size,1)
    Mupper = np.tile(M1.reshape(M1.size,1), (N-1,1))
    Mmiddle = np.tile(M2.reshape(M2.size,1), (N,1))
    Mlower = M3.reshape(M3.size,1)
    M = np.vstack([np.vstack([Mfirst,Mupper]),np.vstack([Mmiddle,Mlower]) ])
    
    if len(K) > 0:
        Karr[range(L0n + L1n*(N-1) + L2n*N, L0n + L1n*(N-1) + L2n*N + L3n), :] \
           = np.dot(L3, K_it)
        M = M - Karr
    
    if p > 0:  
        G = np.vstack([np.vstack([Gupper,Gmiddle]),Glower])
        d_hat = get_max_extreme(G,D,N)
        M = M - d_hat.reshape(d_hat.size,1)
        
    return L,M

def get_max_extreme(G,D,N):
    '''Calculate the array d_hat such that d_hat = max(G*DN_extreme),
    where DN_extreme are the vertices of the set D^N. 
    
    This is used to describe the polytope L*x <= M - G*d_hat. Calculating d_hat
    is equivalen to taking the intersection of the polytopes L*x <= M - G*d_i 
    for every possible d_i in the set of extreme points to D^N.
    
    Input:
    - `G`: The matrix to maximize with respect to
    - `D`: Polytope describing the disturbance set
    - `N`: Horizon length
    
    Output:
    - `d_hat`: Array describing the maximum possible effect from disturbance'''
    
    D_extreme = pc.extreme(D)
    nv = D_extreme.shape[0]
    dim = D_extreme.shape[1]
        
    DN_extreme = np.zeros([dim*N, nv**N])
    
    for i in range(nv**N):
        # Last N digits are indices we want!
        ind = np.base_repr(i, base=nv, padding=N)
        for j in range(N):
            DN_extreme[range(j*dim,(j+1)*dim),i] = D_extreme[int(ind[-j-1]),:]
        
    d_hat = np.amax(np.dot(G,DN_extreme), axis=1)     
    
    return d_hat.reshape(d_hat.size,1)

def is_seq_inside(x0, u_seq, ssys, P0, P1):
    """Checks if the plant remains inside P0 for time t = 1, ... N-1 and 
    that the plant reaches P1 for time t = N.
    Used to test a computed input sequence. No disturbance is taken into
    account.
    
    Input:
    - `x0`: initial point for execution
    - `u_seq`: (N x m) array where row k is input for t = k
    - `ssys`: CtsSysDyn dynamics
    - `P0`: Polytope where we want x(k) to remain for k = 1, ... N-1
    
    Output:
    - `True` if x(k) \in P0 for k = 1, .. N-1 and x(N) \in P1. False otherwise  
    """
    
    N = u_seq.shape[0]
    x = x0
    
    A = ssys.A
    B = ssys.B
    if len(ssys.K) == 0:
        K = np.zeros(x.shape)
    else:
        K = ssys.K
    
    inside = True
    for i in range(N-1):
        u = u_seq[i,:]
    
        if not pc.is_inside(ssys.Uset,u):
            raise Exception("is_seq_inside: got u outside boundary set")
    
        x = np.dot(A,x) + np.dot(B,u) + K
        
        if not pc.is_inside(P0, x):
            inside = False
    un_1 = u_seq[N-1,:]
    xn = np.dot(A,x) + np.dot(B,un_1) + K
    if not pc.is_inside(P1, xn):
        inside = False
            
    return inside

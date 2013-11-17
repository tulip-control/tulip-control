# Copyright (c) 2011, 2012, 2013 by California Institute of Technology
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
#
""" 
Algorithms related to discretization of continuous dynamics.

Class:
    - AbstractSysDyn
    
Primary functions:
    - discretize
    - get_input
    
Helper functions:
    - solve_feasible
    - getInputHelper
    - createLM
    - get_max_extreme
"""
from copy import deepcopy
import numpy as np
from scipy import sparse as sp
from cvxopt import matrix,solvers
from tulip import polytope as pc
from tulip import transys as trs
from tulip.hybrid import LtiSysDyn, PwaSysDyn
from prop2part import PropPreservingPartition, pwa_partition

class AbstractSysDyn:
    """AbstractSysDyn class for discrete abstractions of continuous
    dynamics.
    
    An AbstractSysDyn object contains the fields:
    - ppp: a proposition preserving partition object each region of which
           corresponds to a discrete state of the abstraction
    - ofts: a finite transition system, abstracting the continuous system,
            that can be fed into discrete synthesis algorithms
    - orig_list_region: original proposition preserving regions
    - orig: list assigning an original proposition preserving region to each
            new region 
            
    Note1: There could be some redundancy in ppp and ofts in that they are
    both decorated with propositions. This might be useful to keep each of 
    them as functional units on their own (possible to change later). 
    """
    def __init__(self, ppp=None, ofts=None,
                 orig_list_region=None, orig=None):
        self.ppp = ppp
        self.ofts = ofts
        self.orig_list_region = orig_list_region
        self.orig = orig

def _block_diag2(A,B):
    """Like block_diag() in scipy.linalg, but restricted to 2 inputs.

    Old versions of the linear algebra package in SciPy (i.e.,
    scipy.linalg) do not have a block_diag() function.  Providing
    _block_diag2() here until most folks are using sufficiently
    up-to-date SciPy installations improves portability.
    """
    if len(A.shape) == 1:  # Cast 1d array into matrix
        A = np.array([A])
    if len(B.shape) == 1:
        B = np.array([B])
    C = np.zeros((A.shape[0]+B.shape[0], A.shape[1]+B.shape[1]))
    C[:A.shape[0], :A.shape[1]] = A.copy()
    C[A.shape[0]:, A.shape[1]:] = B.copy()
    return C

def discretize(
    part, ssys, N=10, min_cell_volume=0.1,
    closed_loop=True, conservative=True,
    max_num_poly=5, use_all_horizon=False,
    trans_length=1, remove_trans=False, 
    abs_tol=1e-7, verbose=0, plotting=None
):
    """Refine the partition and establish transitions
    based on reachability analysis.
    
    @param part: a PropPreservingPartition object
    @param ssys: a LtiSysDyn or PwaSysDyn object
    @param N: horizon length
    @param min_cell_volume: the minimum volume of cells in the resulting
        partition.
    @param closed_loop: boolean indicating whether the `closed loop`
        algorithm should be used. default True.
    @param conservative: if true, force sequence in reachability analysis
        to stay inside starting cell. If false, safety
        is ensured by keeping the sequence inside the
        original proposition preserving cell which needs
        to be convex. In order to use the value false,
        ensure to have a convex initial partition or use
        prop2partconvex to postprocess the proposition
        preserving partition before calling discretize.
    @param max_num_poly: maximum number of polytopes in a region to use in 
        reachability analysis.
    @param use_all_horizon: in closed loop algorithm: if we should look
        for reach- ability also in less than N steps.
    @param trans_length: the number of polytopes allowed to cross in a
        transition.  a value of 1 checks transitions
        only between neighbors, a value of 2 checks
        neighbors of neighbors and so on.
    @param remove_trans: if True, remove found transitions between
        non-neighbors.
    @param abs_tol: maximum volume for an "empty" polytope
    @param verbose: level of verbosity
    
    @rtype: AbstractSysDyn
    
    see also
    --------
    prop2part.pwa_partition
    """
    min_cell_volume = (min_cell_volume /np.finfo(np.double).eps
        *np.finfo(np.double).eps)
    
    if isinstance(ssys, PwaSysDyn):
        part = pwa_partition(ssys, part)
    
    # Save original polytopes, require them to be convex 
    if conservative:
        orig_list = None
        orig = 0
    else:
        orig_list = []
        for poly in part.list_region:
            if len(poly) == 0:
                orig_list.append(poly.copy())
            elif len(poly) == 1:
                orig_list.append(poly.list_poly[0].copy())
            else:
                raise Exception("solveFeasible: "
                    "original list contains non-convex"
                    "polytope regions")
        orig = range(len(orig_list))
    
    # Cheby radius of disturbance set
    # (defined within the loop for pwa systems)
    if isinstance(ssys, LtiSysDyn):
        if len(ssys.E) > 0:
            rd,xd = pc.cheby_ball(ssys.Wset)
        else:
            rd = 0.
    
    # Initialize matrix for pairs to check
    IJ = part.adj.copy()
    IJ = IJ.todense()
    IJ = np.array(IJ)
    
    IJ = reachable_within(trans_length, IJ,
                          np.array(part.adj.todense()) )
    
    # Initialize output
    transitions = np.zeros(
        [part.num_regions, part.num_regions],
        dtype = int
    )
    sol = deepcopy(part.list_region)
    adj = part.adj.copy()
    adj = adj.todense()
    adj = np.array(adj)
    subsys_list = deepcopy(part.list_subsys)
    ss = ssys
    
    # init graphics
    if plotting is not None:
        # here to avoid loading matplotlib unless requested
        try:
            from tulip.polytope.plot import plot_partition
            import matplotlib.pyplot as plt
        except:
            plot_partition = None
            print("polytope.plot_partition failed to import.\n"
                "No plotting by discretize during partitioning.")
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
    # Do the abstraction
    while np.sum(IJ) > 0:
        ind = np.nonzero(IJ)
        i = ind[1][0]
        j = ind[0][0]
        IJ[j, i] = 0
        si = sol[i]
        sj = sol[j]
        
        if isinstance(ssys, PwaSysDyn):
            ss = ssys.list_subsys[subsys_list[i]]
            if len(ss.E) > 0:
                rd,xd = pc.cheby_ball(ss.Wset)
            else:
                rd = 0.
        
        if verbose > 1:        
            print("\n Working with states " +str(i) +" and " +str(j) +
                " with lengths " +str(len(si)) +" and " +str(len(sj) ) )
            
            if isinstance(ssys, PwaSysDyn):
                print("where subsystem " +
                    str(subsys_list[i]) +" is active.")
        
        if conservative:
            # Don't use trans_set
            trans_set = None
        else:
            # Use original cell as trans_set
            trans_set = orig_list[orig[i]]
        
        S0 = solve_feasible(
            si, sj, ss, N, closed_loop,
            use_all_horizon, trans_set, max_num_poly
        )
        
        if verbose > 1:
            print("Computed reachable set S0 with volume " +
                str(pc.volume(S0)) )
        
        # isect = si \cap S0
        isect = pc.intersect(si, S0)
        vol1 = pc.volume(isect)
        risect, xi = pc.cheby_ball(isect)
        
        # diff = si \setminus S0
        diff = pc.mldivide(si, S0)
        vol2 = pc.volume(diff)
        rdiff, xd = pc.cheby_ball(diff)
        
        # We don't want our partitions to be smaller than the disturbance set
        # Could be a problem since cheby radius is calculated for smallest
        # convex polytope, so if we have a region we might throw away a good
        # cell.
        if (vol1 > min_cell_volume) and (risect > rd) and \
           (vol2 > min_cell_volume) and (rdiff > rd):
        
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
            difflist = pc.separate(diff)    # Separate the difference
            num_new = len(difflist)
            for reg in difflist:
                sol.append(reg)
                if isinstance(ssys, PwaSysDyn):
                    subsys_list.append(subsys_list[i])
            size = len(sol)
            
            # Update transition matrix
            transitions = adj_update(transitions, size, num_new)
            
            transitions[i, :] = np.zeros(size)
            for kk in xrange(num_new):
                r = size -1 -kk
                
                #transitions[:, r] = transitions[:, i]
                # All sets reachable from start are reachable from both part's
                # except possibly the new part
                transitions[i, r] = 0
                transitions[j, r] = 0            
            
            if i != j:
                # sol[j] is reachable from intersection of sol[i] and S0..
                transitions[j, i] = 1
            
            # Update adjacency matrix
            old_adj = np.nonzero(adj[i, :])[0]
            adj[i, :] = np.zeros([size -num_new])
            adj[:, i] = np.zeros([size -num_new])
            
            adj = adj_update(adj, size, num_new)
            
            for kk in xrange(num_new):
                r = size -1 -kk
                
                adj[i, r] = 1
                adj[r, i] = 1
                adj[r, r] = 1
                
                if not conservative:
                    orig = np.hstack([orig, orig[i]])
            adj[i, i] = 1
                        
            if verbose > 1:
                output = "\n Adding states " + str(i) + " and "
                for kk in xrange(num_new):
                    output += str(size-1-kk) + " and "
                print(output + "\n")
                        
            for k in np.setdiff1d(old_adj, [i,size-1]):
                # Every "old" neighbor must be the neighbor
                # of at least one of the new
                if pc.is_adjacent(sol[i], sol[k]):
                    adj[i, k] = 1
                    adj[k, i] = 1
                elif remove_trans and (trans_length == 1):
                    # Actively remove transitions between non-neighbors
                    transitions[i, k] = 0
                    transitions[k, i] = 0
                
                for kk in xrange(num_new):
                    r = size -1 -kk
                    
                    if pc.is_adjacent(sol[r], sol[k]):
                        adj[r, k] = 1
                        adj[k, r] = 1
                    elif remove_trans and (trans_length == 1):
                        # Actively remove transitions between non-neighbors
                        transitions[r, k] = 0
                        transitions[k, r] = 0
            
            # Update IJ matrix
            IJ = adj_update(IJ, size, num_new)
            adj_k = reachable_within(trans_length, adj, adj)
            sym_adj_change(IJ, adj_k, transitions, i)
            
            for kk in xrange(num_new):
                sym_adj_change(IJ, adj_k, transitions, size -1 -kk)
            
            if verbose > 1:
                print("\n Updated adj: \n" +str(adj) )
                print("\n Updated trans: \n" +str(transitions) )
                print("\n Updated IJ: \n" +str(IJ) )
        elif vol2 < abs_tol:
            if verbose > 1:
                print("Transition found")
            transitions[j,i] = 1
        else:
            if verbose > 1:
                print("No transition found, diff vol: " +str(vol2) +
                      ", intersect vol: " +str(vol1) )
            transitions[j,i] = 0
        
        # no plotting ?
        if plotting is None:
            continue
        if plot_partition is None:
            continue
        
        tmp_part = PropPreservingPartition(
            domain=part.domain, num_prop=part.num_prop,
            list_region=sol, num_regions=len(sol), adj=sp.lil_matrix(adj),
            list_prop_symbol=part.list_prop_symbol, list_subsys=subsys_list
        )
        
        ax.clear()
        plt.ion()
        plot_partition(tmp_part, transitions, ax=ax, color_seed=23)
        plt.pause(1)

    new_part = PropPreservingPartition(
        domain=part.domain, num_prop=part.num_prop,
        list_region=sol, num_regions=len(sol), adj=sp.lil_matrix(adj),
        list_prop_symbol=part.list_prop_symbol, list_subsys=subsys_list
    )
    
    # Generate transition system and add transitions       
    ofts = trs.OpenFTS()
    
    adj = sp.lil_matrix(transitions)
    n = adj.shape[0]
    ofts_states = range(n)
    ofts_states = trs.prepend_with(ofts_states, 's')
    
    # add set to destroy ordering
    ofts.states.add_from(set(ofts_states) )
    
    ofts.transitions.add_adj(adj, ofts_states)
    
    # Decorate TS with state labels
    prop_symbols = part.list_prop_symbol
    ofts.atomic_propositions.add_from(prop_symbols)
    prop_list = []
    for region in sol:
        state_prop = set([
            prop for (prop, x) in
            zip(prop_symbols, region.list_prop) if x == 1
        ])
        
        prop_list.append(state_prop)
    
    ofts.states.labels(ofts_states, prop_list)
    
    assert(len(prop_list) == n)
    
    return AbstractSysDyn(
        ppp=new_part,
        ofts=ofts,
        orig_list_region=orig_list,
        orig=orig
    )

def adj_update(adj, size, num_new):
    adj = np.hstack([
        adj,
        np.zeros(
            [size-num_new, num_new],
            dtype=int
        )
    ])
    adj = np.vstack([
        adj,
        np.zeros(
            [num_new, size],
            dtype=int)
    ])
    return adj

def reachable_within(trans_length, adj_k, adj):
    if trans_length <= 1:
        return adj_k
    
    k = 1
    while k < trans_length:
        adj_k = np.dot(adj_k, adj)
        k += 1
    adj_k = (adj_k > 0).astype(int)
    
    return adj_k

def sym_adj_change(IJ, adj_k, transitions, i):
    horizontal = adj_k[i, :] -transitions[i, :] > 0
    vertical = adj_k[:, i] -transitions[:, i] > 0
    
    IJ[i, :] = horizontal.astype(int)
    IJ[:, i] = vertical.astype(int)

# DEFUNCT until further notice
# def discretize_overlap(part, ssys, N=10, min_cell_volume=0.1, closed_loop=False,\
#                conservative=False, max_num_poly=5, \
#                use_all_horizon=False, abs_tol=1e-7, verbose=0):
# 
#     """Refine the partition and establish transitions based on reachability
#     analysis.
#     
#     Input:
#     
#     - `part`: a PropPreservingPartition object
#     - `ssys`: a LtiSysDyn object
#     - `N`: horizon length
#     - `min_cell_volume`: the minimum volume of cells in the resulting
#                          partition.
#     - `closed_loop`: boolean indicating whether the `closed loop`
#                      algorithm should be used. default False.
#     - `conservative`: if true, force sequence in reachability analysis
#                       to stay inside starting cell. If false, safety
#                       is ensured by keeping the sequence inside the
#                       original proposition preserving cell.
#     - `max_num_poly`: maximum number of polytopes in a region to use
#                       in reachability analysis.
#     - `use_all_horizon`: in closed loop algorithm: if we should look
#                          for reach- ability also in less than N steps.
#     - `abs_tol`: maximum volume for an "empty" polytope
#     
#     Output:
#     
#     - A PropPreservingPartition object with transitions
#     """
#     min_cell_volume = (min_cell_volume/np.finfo(np.double).eps ) * np.finfo(np.double).eps
#     
#     orig_list = []
#     
#     for poly in part.list_region:
#         if len(poly) == 0:
#             orig_list.append(poly.copy())
#         elif len(poly) == 1:
#             orig_list.append(poly.list_poly[0].copy())
#         else:
#             raise Exception("solveFeasible: original list contains non-convex \
#                             polytope regions")
#     
#     orig = range(len(orig_list))
#     
#     # Cheby radius of disturbance set
#     if len(ssys.E) > 0:
#         rd,xd = pc.cheby_ball(ssys.Wset)
#     else:
#         rd = 0.
#     
#     # Initialize matrix for pairs to check
#     IJ = part.adj.copy()
#     IJ = IJ.todense()
#     IJ = np.array(IJ)
#     if verbose > 1:
#         print("\n Starting IJ: \n" + str(IJ) )
# 
#     # Initialize output
#     transitions = np.zeros([part.num_regions,part.num_regions], dtype = int)
#     sol = deepcopy(part.list_region)
#     adj = part.adj.copy()
#     adj = adj.todense()
#     adj = np.array(adj)
#     
#     # List of how many "new" regions that have been created for each region
#     # and a list of original number of neighbors
#     num_new_reg = np.zeros(len(orig_list))
#     num_orig_neigh = np.sum(adj, axis=1).flatten() - 1
# 
#     while np.sum(IJ) > 0:
#         ind = np.nonzero(IJ)
#         i = ind[0][0]
#         j = ind[1][0]
#                 
#         IJ[i,j] = 0
#         num_new_reg[i] += 1
#         print(num_new_reg)
#         si = sol[i]
#         sj = sol[j]
#         
#         if verbose > 1:        
#             print("\n Working with states " + str(i) + " and " + str(j) )
#         
#         if conservative:
#             S0 = solve_feasible(si,sj,ssys,N, closed_loop=closed_loop, 
#                         min_vol=min_cell_volume, max_num_poly=max_num_poly,\
#                         use_all_horizon=use_all_horizon)
#         else:
#             S0 = solve_feasible(si,sj,ssys,N, closed_loop=closed_loop,\
#                     min_vol=min_cell_volume, trans_set=orig_list[orig[i]], \
#                     use_all_horizon=use_all_horizon, max_num_poly=max_num_poly)
#         
#         if verbose > 1:
#             print("Computed reachable set S0 with volume " + str(pc.volume(S0)) )
#         
#         isect = pc.intersect(si, S0)
#         risect, xi = pc.cheby_ball(isect)
#         vol1 = pc.volume(isect)
# 
#         diff = pc.mldivide(si, S0)
#         rdiff, xd = pc.cheby_ball(diff)
#         
#         # We don't want our partitions to be smaller than the disturbance set
#         # Could be a problem since cheby radius is calculated for smallest
#         # convex polytope, so if we have a region we might throw away a good
#         # cell.
#         
#         if rdiff < abs_tol:
#             if verbose > 1:
#                 print("Transition found")
#             transitions[i,j] = 1
#         
#         elif (vol1 > min_cell_volume) & (risect > rd) & \
#                 (num_new_reg[i] <= num_orig_neigh[i]+1):
#         
#             # Make sure new cell is Region and add proposition lists
#             if len(isect) == 0:
#                 isect = pc.Region([isect], si.list_prop)
#             else:
#                 isect.list_prop = si.list_prop
#         
#             # Add new state
#             sol.append(isect)
#             size = len(sol)
#             
#             # Add transitions
#             transitions = np.hstack([transitions, np.zeros([size - 1, 1], dtype=int) ])
#             transitions = np.vstack([transitions, np.zeros([1, size], dtype=int) ])
#             
#             transitions[size-1,:] = transitions[i,:] # All sets reachable from orig cell are reachable from both cells
#             transitions[size-1,j] = 1   # j is reachable from new cell            
#             
#             # Take care of adjacency
#             old_adj = np.nonzero(adj[i,:])[0]
#             
#             adj = np.hstack([adj, np.zeros([size - 1, 1], dtype=int) ])
#             adj = np.vstack([adj, np.zeros([1, size], dtype=int) ])
#             adj[i,size-1] = 1
#             adj[size-1,i] = 1
#             adj[size-1,size-1] = 1
#                                     
#             for k in np.setdiff1d(old_adj,[i,size-1]):
#                 if pc.is_adjacent(sol[size-1],sol[k],overlap=True):
#                     adj[size-1,k] = 1
#                     adj[k, size-1] = 1
#                 else:
#                     # Actively remove (valid) transitions between non-neighbors
#                     transitions[size-1,k] = 0
#                     transitions[k,size-1] = 0
#                     
#             # Assign original proposition cell to new state and update counts
#             if not conservative:
#                 orig = np.hstack([orig, orig[i]])
#             print(num_new_reg)
#             num_new_reg = np.hstack([num_new_reg, 0])
#             num_orig_neigh = np.hstack([num_orig_neigh, np.sum(adj[size-1,:])-1])
#             
#             if verbose > 1:
#                 print("\n Adding state " + str(size-1) + "\n")
#             
#             # Just add adjacent cells for checking, unless transition already found            
#             IJ = np.hstack([IJ, np.zeros([size - 1, 1], dtype=int) ])
#             IJ = np.vstack([IJ, np.zeros([1, size], dtype=int) ])
#             horiz2 = adj[size-1,:] - transitions[size-1,:] > 0
#             verti2 = adj[:,size-1] - transitions[:,size-1] > 0
#             IJ[size-1,:] = horiz2.astype(int)
#             IJ[:,size-1] = verti2.astype(int)      
#             
#             #if verbose > 1:
#                 #print("\n Updated adj: \n" + str(adj) )
#                 #print("\n Updated trans: \n" + str(transitions) )
#                 #print("\n Updated IJ: \n" + str(IJ) )
#                     
#         else:
#             if verbose > 1:
#                 print("No transition found, intersect vol: " + str(vol1) )
#             transitions[i,j] = 0
#                   
#     new_part = PropPreservingPartition(domain=part.domain, num_prop=part.num_prop,
#                                        list_region=sol, num_regions=len(sol), adj=np.array([]), 
#                                        trans=transitions, list_prop_symbol=part.list_prop_symbol,
#                                        orig_list_region=orig_list, orig=orig)                           
#     return new_part

def get_input(
    x0, ssys, part, start, end, N, R=[], r=[], Q=[],
    mid_weight=0., conservative=True,
    closed_loop=True, test_result=False
):
    """Calculate an input signal sequence taking the plant from state `start` 
    to state `end` in the partition part, such that 
    f(x,u) = x'Rx + r'x + u'Qu + mid_weight*|xc-x(0)|_2 is minimal. xc is
    the chebyshev center of the final cell.
    If no cost parameters are given, Q = I and mid_weight=3 are used. 
        
    Input:

    - `x0`: initial continuous state
    - `ssys`: LtiSysDyn object specifying system dynamics
    - `part`: PropPreservingPartition object specifying the state
              space partition.
    - `start`: int specifying the number of the initial state in `part`
    - `end`: int specifying the number of the end state in `part`
    - `N`: the horizon length
    - `R`: state cost matrix for x = [x(1)' x(2)' .. x(N)']', 
           size (N*xdim x N*xdim). If empty, zero matrix is used.
    - `r`: cost vector for x = [x(1)' x(2)' .. x(N)']', size (N*xdim x 1)
    - `Q`: input cost matrix for u = [u(0)' u(1)' .. u(N-1)']', 
           size (N*udim x N*udim). If empty, identity matrix is used.
    - `mid_weight`: cost weight for |x(N)-xc|_2
    - `conservative`: if True, force plant to stay inside initial
                      state during execution. if False, plant is
                      forced to stay inside the original proposition
                      preserving cell.
    - `closed_loop`: should be True if closed loop discretization has
                     been used.
    - `test_result`: performs a simulation (without disturbance) to
                     make sure that the calculated input sequence is
                     safe.
    
    Output:
    - A (N x m) numpy array where row k contains u(k) for k = 0,1 ... N-1.
    
    Note1: The same horizon length as in reachability analysis should
    be used in order to guarantee feasibility.
    
    Note2: If the closed loop algorithm has been used to compute
    reachability the input needs to be recalculated for each time step
    (with decreasing horizon length). In this case only u(0) should be
    used as a control signal and u(1) ... u(N-1) thrown away.
    
    Note3: The "conservative" calculation makes sure that the plants
    remains inside the convex hull of the starting region during
    execution, i.e.  x(1), x(2) ...  x(N-1) are in conv_hull(starting
    region).  If the original proposition preserving partition is not
    convex, safety can not be guaranteed.
    """
    if (len(R) == 0) and (len(Q) == 0) and \
    (len(r) == 0) and (mid_weight == 0):
        # Default behavior
        Q = np.eye(N*ssys.B.shape[1])
        R = np.zeros([N*x0.size, N*x0.size])
        r = np.zeros([N*x0.size,1])
        mid_weight = 3
    if len(R) == 0:
        R = np.zeros([N*x0.size, N*x0.size])
    if len(Q) == 0:
        Q = np.zeros([N*ssys.B.shape[1], N*ssys.B.shape[1]])    
    if len(r) == 0:
        r = np.zeros([N*x0.size,1])
    
    if (R.shape[0] != R.shape[1]) or (R.shape[0] != N*x0.size):
        raise Exception("get_input: "
            "R must be square and have side N * dim(state space)")
    
    if (Q.shape[0] != Q.shape[1]) or (Q.shape[0] != N*ssys.B.shape[1]):
        raise Exception("get_input: "
            "Q must be square and have side N * dim(input space)")
    if part.trans != None:
        if part.trans[end,start] != 1:
            raise Exception("get_input: "
                "no transition from state " + str(start) +
                " to state " + str(end)
            )
    else:
        print("get_input: "
            "Warning, no transition matrix found, assuming feasible")
    
    if (not conservative) & (part.orig == None):
        print("List of original proposition preserving "
            "partitions not given, reverting to conservative mode")
        conservative = True
       
    P_start = part.list_region[start]
    P_end = part.list_region[end]
    
    n = ssys.A.shape[1]
    m = ssys.B.shape[1]
    
    if conservative:
        # Take convex hull or P_start as constraint
        if len(P_start) > 0:
            if len(P_start.list_poly) > 1:
                # Take convex hull
                vert = pc.extreme(P_start.list_poly[0])
                for i in range(1, len(P_start.list_poly)):
                    vert = np.hstack([
                        vert,
                        pc.extreme(P_start.list_poly[i])
                    ])
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
            if mid_weight > 0:
                rc, xc = pc.cheby_ball(P3)
                R[
                    np.ix_(
                        range(n*(N-1), n*N),
                        range(n*(N-1), n*N)
                    )
                ] += mid_weight*np.eye(n)
                
                r[range((N-1)*n, N*n), :] += -mid_weight*xc
            try:
                u, cost = getInputHelper(
                    x0, ssys, P1, P3, N, R, r, Q,
                    closed_loop=closed_loop
                )
                r[range((N-1)*n, N*n), :] += mid_weight*xc
            except:
                r[range((N-1)*n, N*n), :] += mid_weight*xc
                continue
            if cost < low_cost:
                low_u = u
                low_cost = cost
        if low_cost == np.inf:
            raise Exception("get_input: Did not find any trajectory")
    else:
        P3 = P_end
        if mid_weight > 0:
            rc, xc = pc.cheby_ball(P3)
            R[
                np.ix_(
                    range(n*(N-1), n*N),
                    range(n*(N-1), n*N)
                )
            ] += mid_weight*np.eye(n)
            r[range((N-1)*n, N*n), :] += -mid_weight*xc
        low_u, cost = getInputHelper(
            x0, ssys, P1, P3, N, R, r, Q,
            closed_loop=closed_loop
        )
        
    if test_result:
        good = is_seq_inside(x0, low_u, ssys, P1, P3)
        if not good:
            print("Calculated sequence not good")
    return low_u
    
def solve_feasible(
    P1, P2, ssys, N, closed_loop=True,
    use_all_horizon=False, trans_set=None, max_num_poly=5
):
    """Computes the subset x0 of C{P1} from which C{P2} is reachable
    in horizon C{N}, with respect to system dynamics C{ssys}. The closed
    loop algorithm solves for one step at a time, which keeps the dimension
    of the polytopes down.
    
    @type P1: Polytope or Region
    @type P2: Polytope or Region
    @type ssys: LtiSysDyn
    @param N: The horizon length
    @param closed_loop: If true, take 1 step at a time.
        This keeps down polytope dimension and
        handles disturbances better.
        Default: True
    @type closed_loop: bool
    @param use_all_horizon: Used for closed loop algorithm.
        If true, then allow reachability also in less than N steps.
    @param trans_set: If specified,
        then force transitions to be in this set.
        If empty, P1 is used.
    
    @return: C{x0}, defines the set in P1 from which P2 is reachable
    @rtype: Polytope or Region
    """
    if closed_loop:
        return solve_feasible_closed_loop(
            P1, P2, ssys, N,
            use_all_horizon=use_all_horizon,
            trans_set=trans_set
        )
    else:
        return solve_feasible_open_loop(
            P1, P2, ssys, N,
            trans_set=trans_set,
            max_num_poly=max_num_poly
        )

def solve_feasible_closed_loop(
    P1, P2, ssys, N,
    use_all_horizon=False, trans_set=None
):
    part1 = P1.copy() # Initial set
    part2 = P2.copy() # Terminal set
    
    temp_part = part2
    
    if trans_set != None:
        # if ttsnd_set is defined,
        # then the intermediate steps are
        # allowed to be in trans_set
        ttt = trans_set
    else:
        ttt = part1
    
    for i in xrange(N,1,-1): 
        x0 = solve_feasible_open_loop(
            ttt, temp_part, ssys, N=1,
            trans_set=trans_set
        )
        
        if use_all_horizon:
            temp_part = pc.union(x0, temp_part,
                                 check_convex=True)
        else:
            temp_part = x0
            if not pc.is_fulldim(temp_part):
                return pc.Polytope()
    
    x0 = solve_feasible_open_loop(
        part1, temp_part, ssys, N=1,
        trans_set=trans_set
    )
    
    if use_all_horizon:
        temp_part = pc.union(x0, temp_part,
                             check_convex=True)
    else:
        temp_part = x0
    
    return temp_part

def solve_feasible_open_loop(
    P1, P2, ssys, N,
    trans_set=None, max_num_poly=5
):
    part1 = P1.copy() # Initial set
    part2 = P2.copy() # Terminal set
    
    if len(part1) > max_num_poly:
        # use the max_num_poly largest volumes for reachability
        part1 = volumes_for_reachability(part1, max_num_poly)

    if len(part2) > max_num_poly:
        # use the max_num_poly largest volumes for reachability
        part2 = volumes_for_reachability(part2, max_num_poly)
    
    if len(part1) > 0:
        # Recursive union of sets
        poly = pc.Polytope()
        for i in xrange(0, len(part1) ):
            s0 = solve_feasible_open_loop(
                part1.list_poly[i], part2,
                ssys, N, trans_set
            )
            poly = pc.union(poly, s0, check_convex=True)
        return poly
    
    if len(part2) > 0:
        # Recursive union of sets
        poly = pc.Polytope()
        for i in xrange(0, len(part2) ):
            s0 = solve_feasible_open_loop(
                part1, part2.list_poly[i],
                ssys, N, trans_set
            )
            poly = pc.union(poly, s0, check_convex=True)
        return poly
            
    if trans_set == None:
        trans_set = part1

    # stack polytope constraints
    L, M = createLM(ssys, N, part1, trans_set, part2) 
    
    # Ready to make polytope
    poly1 = pc.reduce(pc.Polytope(L, M) )
    
    # Project poly1 onto lower dim
    n = np.shape(ssys.A)[1]
    poly1 = pc.projection(poly1, range(1, n+1) )
    
    return pc.reduce(poly1)

def volumes_for_reachability(part, max_num_poly):
    vol_list = np.zeros(len(part) )
    for i in xrange(len(part) ):
        vol_list[i] = pc.volume(part.list_poly[i] )
    
    ind = np.argsort(-vol_list)
    temp = []
    for i in ind[range(max_num_poly) ]:
        temp.append(part.list_poly[i] )
    
    part = pc.Region(temp, [])
    return part

def getInputHelper(
    x0, ssys, P1, P3, N, R, r, Q,
    closed_loop=True
):
    """Calculates the sequence u_seq such that
    - x(t+1) = A x(t) + B u(t) + K
    - x(k) \in P1 for k = 0,...N
    - x(N) \in P3
    - [u(k); x(k)] \in PU
    
    and minimizes x'Rx + 2*r'x + u'Qu
    """
    n = ssys.A.shape[1]
    m = ssys.B.shape[1]
    
    list_P = []
    if closed_loop:
        temp_part = P3
        list_P.append(P3)
        for i in xrange(N-1,0,-1): 
            temp_part = solve_feasible(
                P1, temp_part, ssys, N=1,
                closed_loop=False, trans_set=P1
            )
            list_P.insert(0, temp_part)
        list_P.insert(0,P1)
        L,M = createLM(ssys, N, list_P, disturbance_ind=[1])
    else:
        list_P.append(P1)
        for i in xrange(N-1,0,-1):
            list_P.append(P1)
        list_P.append(P3)
        L,M = createLM(ssys, N, list_P)
    
    # Remove first constraint on x(0)
    L = L[range(list_P[0].A.shape[0], L.shape[0]),:]
    M = M[range(list_P[0].A.shape[0], M.shape[0]),:]
    
    # Separate L matrix
    Lx = L[:,range(n)]
    Lu = L[:,range(n,L.shape[1])] 
    
    M = M - np.dot(Lx, x0).reshape(Lx.shape[0],1)
        
    # Constraints
    G = matrix(Lu)
    h = matrix(M)

    B_diag = ssys.B
    for i in xrange(N-1):
        B_diag = _block_diag2(B_diag,ssys.B)
    K_hat = np.tile(ssys.K, (N,1))

    A_it = ssys.A.copy()
    A_row = np.zeros([n, n*N])
    A_K = np.zeros([n*N, n*N])
    A_N = np.zeros([n*N, n])

    for i in xrange(N):
        A_row = np.dot(ssys.A, A_row)
        A_row[np.ix_(
            range(n),
            range(i*n, (i+1)*n)
        )] = np.eye(n)

        A_N[np.ix_(
            range(i*n, (i+1)*n),
            range(n)
        )] = A_it
        
        A_K[np.ix_(
            range(i*n,(i+1)*n),
            range(A_K.shape[1])
        )] = A_row
        
        A_it = np.dot(ssys.A, A_it)
        
    Ct = np.dot(A_K, B_diag)
    P = matrix(Q + np.dot(Ct.T, np.dot(R, Ct)))
    q = matrix(
        np.dot(
            np.dot(x0.reshape(1,x0.size), A_N.T) +
            np.dot(A_K, K_hat).T , np.dot(R, Ct)
        ) +
        np.dot(r.T, Ct )
    ).T 
    
    sol = solvers.qp(P,q,G,h)
    
    if sol['status'] != "optimal":
        raise Exception("getInputHelper: "
            "QP solver finished with status " +
            str(sol['status'])
        )
    u = np.array(sol['x']).flatten()
    cost = sol['primal objective']
    
    return u.reshape(N, m), cost

def createLM(ssys, N, list_P, Pk=None, PN=None, disturbance_ind=None):
    """Compute the components of the polytope:
        L [x(0)' u(0)' ... u(N-1)']' <= M
    which stacks the following constraints
    
    - x(t+1) = A x(t) + B u(t) + E d(t)
    - [u(k); x(k)] \in ssys.Uset for all k
    
    If list_P is a Polytope:

    - x(0) \in list_P if list_P
    - x(k) \in Pk for k= 1,2, .. N-1
    - x(N) \in PN
    
    If list_P is a list of polytopes

    - x(k) \in list_P[k] for k= 0, 1 ... N
    
    The returned polytope describes the intersection of the polytopes
    for all possible

    @param ssys: system dynamics
    @type ssys: LtiSysDyn
    
    @param N: horizon length
    
    @type list_P: list of Polytopes or Polytope
    @type Pk: Polytope
    @type PN: Polytope
    
    @param disturbance_ind: list indicating which k's
        that disturbance should be taken into account.
        Default is [1,2, ... N]
    """
    if isinstance(list_P, pc.Polytope):
        list_P = [list_P] +(N-1) *[Pk] +[PN]
        
    if disturbance_ind is None:
        disturbance_ind = range(1,N+1)
    
    A = ssys.A
    B = ssys.B
    E = ssys.E
    D = ssys.Wset
    K = ssys.K
    PU = ssys.Uset

    n = A.shape[1]  # State space dimension
    m = B.shape[1]  # Input space dimension
    p = E.shape[1]  # Disturbance space dimension
    
    # non-zero disturbance matrix E ?
    if not np.all(E==0):
        if not pc.is_fulldim(D):
            E = np.zeros(K.shape)
    
    list_len = np.array([P.A.shape[0] for P in list_P])
    sumlen = np.sum(list_len)

    LUn = np.shape(PU.A)[0]
    
    Lk = np.zeros([sumlen, n+N*m])
    LU = np.zeros([LUn*N,n+N*m])
    
    Mk = np.zeros([sumlen,1])
    MU = np.tile(PU.b.reshape(PU.b.size,1), (N,1))
  
    Gk = np.zeros([sumlen, p*N])
    GU = np.zeros([LUn*N, p*N])
    
    K_hat = np.tile(K, (N,1))
    B_diag = B
    E_diag = E
    for i in xrange(N-1):
        B_diag = _block_diag2(B_diag,B)
        E_diag = _block_diag2(E_diag,E)
    A_n = np.eye(n)
    A_k = np.zeros([n, n*N])
    
    sum_vert = 0
    for i in xrange(N+1):
        Li = list_P[i]
        
        ######### FOR L #########
        AB_line = np.hstack([A_n, np.dot(A_k, B_diag)])
        Lk[
            np.ix_(
                range(sum_vert, sum_vert + Li.A.shape[0]),
                range(0,Lk.shape[1])
            )
        ] = np.dot(Li.A, AB_line)
        
        if i < N:
            if PU.A.shape[1] == m:
                LU[
                    np.ix_(
                        range(i*LUn, (i+1)*LUn),
                        range(n + m*i, n + m*(i+1))
                    )
                ] = PU.A
            elif PU.A.shape[1] == m+n:
                uk_line = np.zeros([m, n + m*N])
                uk_line[
                    np.ix_(range(m), range(n+m*i, n+m*(i+1)))
                ] = np.eye(m)
                A_mult = np.vstack([uk_line, AB_line])
                b_mult = np.zeros([m+n, 1])
                b_mult[range(m, m+n), :] = np.dot(A_k, K_hat)
                LU[
                    np.ix_(
                        range(i*LUn, (i+1)*LUn),
                        range(n+m*N)
                    )
                ] = np.dot(PU.A, A_mult)
                MU[range(i*LUn, (i+1)*LUn), :] -= np.dot(PU.A, b_mult)
        
        ######### FOR M #########
        Mk[range(sum_vert, sum_vert + Li.A.shape[0]), :] = \
            Li.b.reshape(Li.b.size,1) - \
            np.dot(np.dot(Li.A,A_k), K_hat)
        
        ######### FOR G #########
        if i in disturbance_ind:
            Gk[
                np.ix_(
                    range(sum_vert, sum_vert + Li.A.shape[0]),
                    range(Gk.shape[1])
                )
            ] = np.dot(np.dot(Li.A,A_k), E_diag)
            
            if (PU.A.shape[1] == m+n) & (i < N):
                A_k_E_diag = np.dot(A_k, E_diag)
                d_mult = np.vstack([np.zeros([m, p*N]), A_k_E_diag])
                GU[np.ix_(range(LUn*i, LUn*(i+1)), range(p*N))] = \
                   np.dot(PU.A, d_mult)
        
        ####### Iterate #########
        if i < N:
            sum_vert += Li.A.shape[0]
            A_n = np.dot(A, A_n)
            A_k = np.dot(A, A_k)
            A_k[np.ix_(range(n), range(i*n, (i+1)*n))] = np.eye(n)
                
    # Get disturbance sets
    if not np.all(Gk==0):  
        G = np.vstack([Gk, GU])
        D_hat = get_max_extreme(G, D, N)
    else:
        D_hat = np.zeros([sumlen + LUn*N, 1])

    # Put together matrices L, M
    L = np.vstack([Lk, LU])
    M = np.vstack([Mk, MU]) - D_hat
    return L,M

def get_max_extreme(G,D,N):
    """Calculate the array d_hat such that d_hat = max(G*DN_extreme),
    where DN_extreme are the vertices of the set D^N. 
    
    This is used to describe the polytope L*x <= M - G*d_hat. Calculating d_hat
    is equivalen to taking the intersection of the polytopes L*x <= M - G*d_i 
    for every possible d_i in the set of extreme points to D^N.
    
    Input:

    - `G`: The matrix to maximize with respect to
    - `D`: Polytope describing the disturbance set
    - `N`: Horizon length
    
    Output:
    - `d_hat`: Array describing the maximum possible effect from disturbance
    """
    D_extreme = pc.extreme(D)
    nv = D_extreme.shape[0]
    dim = D_extreme.shape[1]
    DN_extreme = np.zeros([dim*N, nv**N])
    
    for i in xrange(nv**N):
        # Last N digits are indices we want!
        ind = np.base_repr(i, base=nv, padding=N)
        for j in xrange(N):
            DN_extreme[range(j*dim,(j+1)*dim),i] = D_extreme[int(ind[-j-1]),:]

    d_hat = np.amax(np.dot(G,DN_extreme), axis=1)     
    return d_hat.reshape(d_hat.size,1)

def is_seq_inside(x0, u_seq, ssys, P0, P1):
    """Checks if the plant remains inside P0 for time t = 1, ... N-1
    and  that the plant reaches P1 for time t = N.
    Used to test a computed input sequence.
    No disturbance is taken into account.
    
    @param x0: initial point for execution
    @param u_seq: (N x m) array where row k is input for t = k
    
    @param ssys: dynamics
    @type ssys: LtiSysDyn
    
    @param P0: Polytope where we want x(k) to remain for k = 1, ... N-1
    
    @return: C{True} if x(k) \in P0 for k = 1, .. N-1 and x(N) \in P1.
        C{False} otherwise  
    """
    N = u_seq.shape[0]
    x = x0.reshape(x0.size,1)
    
    A = ssys.A
    B = ssys.B
    if len(ssys.K) == 0:
        K = np.zeros(x.shape)
    else:
        K = ssys.K
    
    inside = True
    for i in xrange(N-1):
        u = u_seq[i,:].reshape(u_seq[i,:].size,1)
        x = np.dot(A,x) + np.dot(B,u) + K       
        if not pc.is_inside(P0, x):
            inside = False
    un_1 = u_seq[N-1,:].reshape(u_seq[N-1,:].size,1)
    xn = np.dot(A,x) + np.dot(B,un_1) + K
    if not pc.is_inside(P1, xn):
        inside = False
            
    return inside
    
def get_cellID(x0, part):
    """Return an integer specifying in which discrete state
    the continuous state x0 belongs to.
        
    Input:
    - `x0`: initial continuous state
    - `part`: PropPreservingPartition object specifying
        the state space partition
    
    Output:
    - cellID: int specifying the discrete state in
        `part` x0 belongs to, -1 if x0 does 
        not belong to any discrete state.
    
    Note1: If there are overlapping partitions
    (i.e., x0 can belong to more than one discrete state),
    this just returns the first ID
    """
    cellID = -1
    for i in xrange(part.num_regions):
        if pc.is_inside(part.list_region[i], x0):
             cellID = i
             break
    return cellID

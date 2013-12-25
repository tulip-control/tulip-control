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
    
Helper functions:
    - solve_feasible
    - createLM
    - get_max_extreme

see also
--------
find_controller
"""
from copy import deepcopy
from collections import Iterable
from warnings import warn

import numpy as np
from scipy import sparse as sp

from tulip import polytope as pc
from tulip import transys as trs
from tulip.hybrid import LtiSysDyn, PwaSysDyn
from prop2partition import PropPreservingPartition, pwa_partition, part2convex

class AbstractSysDyn(object):
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
    - disc_params: a dictionary of parameters used in discretization that 
            should be passed to the controller refinement to ensure consistency 
            
    Note1: There could be some redundancy in ppp and ofts in that they are
    both decorated with propositions. This might be useful to keep each of 
    them as functional units on their own (possible to change later). 
    """
    def __init__(self, ppp=None, ofts=None,
                 orig_list_region=None, orig=None, disc_params={}):
        self.ppp = ppp
        self.ofts = ofts
        self.orig_list_region = orig_list_region
        self.orig = orig
        self.disc_params = disc_params

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
    closed_loop=True, conservative=False,
    max_num_poly=5, use_all_horizon=False,
    trans_length=1, remove_trans=False, 
    abs_tol=1e-7, verbose=0
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
        is ensured by keeping the sequence inside a convexified
        version of the original proposition preserving cell.
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
    prop2partition.pwa_partition
    prop2partition.part2convex
    """
    min_cell_volume = (min_cell_volume /np.finfo(np.double).eps
        *np.finfo(np.double).eps)
    
    ispwa = isinstance(ssys, PwaSysDyn)
    islti = isinstance(ssys, LtiSysDyn)
    
    if ispwa:
        part = pwa_partition(ssys, part)
    
    # Save original polytopes, require them to be convex 
    if conservative:
        orig_list = None
        orig = 0
    else:
        part = part2convex(part) # convexify
        remove_trans = False # already allowed in nonconservative
        orig_list = []
        for poly in part.list_region:
            if len(poly) == 0:
                orig_list.append(poly.copy())
            elif len(poly) == 1:
                orig_list.append(poly.list_poly[0].copy())
            else:
                raise Exception("discretize: "
                    "problem in convexification")
        orig = range(len(orig_list))
    
    # Cheby radius of disturbance set
    # (defined within the loop for pwa systems)
    if islti:
        if len(ssys.E) > 0:
            rd,xd = pc.cheby_ball(ssys.Wset)
        else:
            rd = 0.
    
    # Initialize matrix for pairs to check
    IJ = part.adj.copy()
    IJ = IJ.todense()
    IJ = np.array(IJ)
    if verbose > 1:
        print("\n Starting IJ: \n" + str(IJ) )
    
    # next line omitted in discretize_overlap
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
    
    # next 2 lines omitted in discretize_overlap
    subsys_list = deepcopy(part.list_subsys)
    ss = ssys
    
    # List of how many "new" regions
    # have been created for each region
    # and a list of original number of neighbors
    #num_new_reg = np.zeros(len(orig_list))
    #num_orig_neigh = np.sum(adj, axis=1).flatten() - 1
    
    # Do the abstraction
    while np.sum(IJ) > 0:
        ind = np.nonzero(IJ)
        # i,j swapped in discretize_overlap
        i = ind[1][0]
        j = ind[0][0]
        IJ[j, i] = 0
        si = sol[i]
        sj = sol[j]
        
        #num_new_reg[i] += 1
        #print(num_new_reg)
        
        if ispwa:
            ss = ssys.list_subsys[subsys_list[i]]
            if len(ss.E) > 0:
                rd,xd = pc.cheby_ball(ss.Wset)
            else:
                rd = 0.
        
        if verbose > 1:
            msg = '\n Working with states:\n\t'
            msg += str(i) +' (#polytopes = ' +str(len(si) ) +'), and:\n\t'
            msg += str(j) +' (#polytopes = ' +str(len(sj) ) +')'
            
            if ispwa:
                msg += 'with active subsystem:\n\t'
                msg += str(subsys_list[i])
            
            print(msg)
        
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
                if ispwa:
                    subsys_list.append(subsys_list[i])
            size = len(sol)
            
            # Update transition matrix
            transitions = np.pad(transitions, (0,num_new), 'constant')
            
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
            
            adj = np.pad(adj, (0,num_new), 'constant')
            
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
            IJ = np.pad(IJ, (0,num_new), 'constant')
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
    
    param = {'N':N, 'closed_loop':closed_loop,
        'conservative':conservative,
        'use_all_horizon':use_all_horizon}
    
    assert(len(prop_list) == n)
    
    return AbstractSysDyn(
        ppp=new_part,
        ofts=ofts,
        orig_list_region=orig_list,
        orig=orig,
        disc_params=param
    )

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
def discretize_overlap(closed_loop=False, conservative=False):
    """default False."""
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
#             transitions = np.hstack([transitions, np.zeros([size - 1, 1],
#                                     dtype=int) ])
#             transitions = np.vstack([transitions, np.zeros([1, size],
#                                     dtype=int) ])
#             
#             # All sets reachable from orig cell are reachable from both cells
#             transitions[size-1,:] = transitions[i,:]
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
#             # Just add adjacent cells for checking,
#             # unless transition already found            
#             IJ = np.hstack([IJ, np.zeros([size - 1, 1], dtype=int) ])
#             IJ = np.vstack([IJ, np.zeros([1, size], dtype=int) ])
#             horiz2 = adj[size-1,:] - transitions[size-1,:] > 0
#             verti2 = adj[:,size-1] - transitions[:,size-1] > 0
#             IJ[size-1,:] = horiz2.astype(int)
#             IJ[:,size-1] = verti2.astype(int)
#         else:
#             if verbose > 1:
#                 print("No transition found, intersect vol: " + str(vol1) )
#             transitions[i,j] = 0
#                   
#     new_part = PropPreservingPartition(
#                    domain=part.domain, num_prop=part.num_prop,
#                    list_region=sol, num_regions=len(sol), adj=np.array([]), 
#                    trans=transitions, list_prop_symbol=part.list_prop_symbol,
#                    orig_list_region=orig_list, orig=orig)                           
#     return new_part
    
def solve_feasible(
    P1, P2, ssys, N, closed_loop=True,
    use_all_horizon=False, trans_set=None, max_num_poly=5
):
    """Compute S0 \subset P1 from which P2 is reachable in horizon N.
    
    The system dynamics are C{ssys}.
    The closed-loop algorithm solves for one step at a time,
    which keeps the dimension of the polytopes down.
    
    @type P1: Polytope or Region
    @type P2: Polytope or Region
    @type ssys: LtiSysDyn
    @param N: The horizon length
    @param closed_loop: If true, take 1 step at a time.
        This keeps down polytope dimension and
        handles disturbances better.
    @type closed_loop: bool
    
    @param use_all_horizon: Used for closed loop algorithm.
        If true, then allow reachability also in less than N steps.
    @type use_all_horizon: bool
    
    @param trans_set: If specified,
        then force transitions to be in this set.
        Otherwise, P1 is used.
    
    @return: the subset S0 of P1 from which P2 is reachable
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

def createLM(ssys, N, list_P, Pk=None, PN=None, disturbance_ind=None):
    """Compute the components of the polytope:
    
        L [x(0)' u(0)' ... u(N-1)']' <= M
    
    which stacks the following constraints:
    
    - x(t+1) = A x(t) + B u(t) + E d(t)
    - [u(k); x(k)] \in ssys.Uset for all k
    
    If list_P is a Polytope:

    - x(0) \in list_P if list_P
    - x(k) \in Pk for k= 1,2, .. N-1
    - x(N) \in PN
    
    If list_P is a list of polytopes:

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
    if not isinstance(list_P, Iterable):
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
        
        if not isinstance(Li, pc.Polytope):
            warn('createLM: Li of type: ' +str(type(Li) ) )
        
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
    """Calculate the array d_hat such that:
    
        d_hat = max(G*DN_extreme),
    
    where DN_extreme are the vertices of the set D^N.
    
    This is used to describe the polytope:
    
        L*x <= M - G*d_hat.
    
    Calculating d_hat is equivalen to taking the intersection
    of the polytopes:
    
        L*x <= M - G*d_i
    
    for every possible d_i in the set of extreme points to D^N.
    
    @param G: The matrix to maximize with respect to
    @param D: Polytope describing the disturbance set
    @param N: Horizon length
    
    @return: d_hat: Array describing the maximum possible
        effect from the disturbance
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

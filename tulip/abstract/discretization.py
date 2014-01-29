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
import logging
logger = logging.getLogger(__name__)

from copy import deepcopy

import numpy as np
from scipy import sparse as sp

from tulip import polytope as pc
from tulip import transys as trs
from tulip.hybrid import LtiSysDyn, PwaSysDyn
from .prop2partition import PropPreservingPartition, pwa_partition, part2convex
from .feasible import solve_feasible

class AbstractSysDyn(object):
    """AbstractSysDyn class for discrete abstractions of continuous
    dynamics.
    
    An AbstractSysDyn object contains the fields:
    - ppp: a proposition preserving partition object each region of which
           corresponds to a discrete state of the abstraction
    - ofts: a finite transition system, abstracting the continuous system,
            that can be fed into discrete synthesis algorithms
    - original_regions: original proposition preserving regions
    - orig: list assigning an original proposition preserving region to each
            new region
    - disc_params: a dictionary of parameters used in discretization that 
            should be passed to the controller refinement to ensure consistency 
            
    Note1: There could be some redundancy in ppp and ofts in that they are
    both decorated with propositions. This might be useful to keep each of 
    them as functional units on their own (possible to change later). 
    """
    def __init__(self, ppp=None, ofts=None,
                 original_regions=None, orig=None, disc_params={}):
        self.ppp = ppp
        self.ofts = ofts
        self.original_regions = original_regions
        self.orig = orig
        self.disc_params = disc_params
    
    def __str__(self):
        s = str(self.ppp)
        s += str(self.ofts)
        
        s += 30 * '-' + '\n'
        s += 'Original Regions List:\n\n'
        for i, region in enumerate(self.original_regions):
            s += 'Region: ' + str(i) + '\n'
            s += str(region) + '\n'
        
        s += 'Map New to Original Regions:\n\n'
        for i, original_region in enumerate(self.orig):
            s += str(i) + ' -> ' + str(original_region) + '\n'
        
        s += 'Discretization Options:\n\t'
        s += str(self.disc_params) +'\n'
        
        return s

def discretize(
    part, ssys, N=10, min_cell_volume=0.1,
    closed_loop=True, conservative=False,
    max_num_poly=5, use_all_horizon=False,
    trans_length=1, remove_trans=False, 
    abs_tol=1e-7,
    plotit=False, save_img=False, cont_props=None,
    plot_every=1
):
    """Refine the partition and establish transitions
    based on reachability analysis.
    
    see also
    --------
    prop2partition.pwa_partition
    prop2partition.part2convex
    
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
    
    @param plotit: plot partitioning as it evolves
    @type plotit: boolean,
        default = False
    
    @param save_img: save snapshots of partitioning to PDF files,
        requires plotit=True
    @type save_img: boolean,
        default = False
    
    @param cont_props: continuous propositions to plot
    @type cont_props: list of Polytopes
    
    @rtype: AbstractSysDyn
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
            rd = ssys.Wset.chebR
        else:
            rd = 0.
    
    # Initialize matrix for pairs to check
    IJ = part.adj.copy()
    IJ = IJ.todense()
    IJ = np.array(IJ)
    logger.info("\n Starting IJ: \n" + str(IJ) )
    
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
    
    # init graphics
    if plotit:
        # here to avoid loading matplotlib unless requested
        try:
            from plot import plot_partition, plot_transition_arrow
        except Exception, e:
            logger.error(e)
            plot_partition = None
        
        try:
            import matplotlib.pyplot as plt
            plt.ion()
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.axis('scaled')
            ax2.axis('scaled')
            file_extension = 'png'
        except Exception, e:
            logger.error(e)
            plot_partition = None
        
    iter_count = 0
    
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
                rd, xd = pc.cheby_ball(ss.Wset)
            else:
                rd = 0.
        
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
        
        msg = '\n Working with states:\n\t'
        msg += str(i) +' (#polytopes = ' +str(len(si) ) +'), and:\n\t'
        msg += str(j) +' (#polytopes = ' +str(len(sj) ) +')'
            
        if ispwa:
            msg += 'with active subsystem:\n\t'
            msg += str(subsys_list[i])
            
        msg += 'Computed reachable set S0 with volume: '
        msg += str(S0.volume) + '\n'
        
        logger.info(msg)
        
        # isect = si \cap S0
        isect = si.intersect(S0)
        vol1 = isect.volume
        risect, xi = pc.cheby_ball(isect)
        
        # diff = si \setminus S0
        diff = si.diff(S0)
        vol2 = diff.volume
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
                        
            if logger.getEffectiveLevel() >= logging.INFO:
                msg = '\n Adding states ' + str(i) + ' and '
                for kk in xrange(num_new):
                    msg += str(size-1-kk) + ' and '
                msg += '\n'
                        
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
            
            msg += '\n\n Updated adj: \n' + str(adj)
            msg += '\n\n Updated trans: \n' + str(transitions)
            msg += '\n\n Updated IJ: \n' + str(IJ)
        elif vol2 < abs_tol:
            msg += 'Transition found'
            transitions[j,i] = 1
        else:
            msg += 'No transition found, diff vol: ' + str(vol2)
            msg += ', intersect vol: ' + str(vol1)
            transitions[j,i] = 0
        
        logger.info(msg)
        
        iter_count += 1
        
        # no plotting ?
        if not plotit:
            continue
        if plot_partition is None:
            continue
        if iter_count % plot_every != 0:
            continue
        
        tmp_part = PropPreservingPartition(
            domain=part.domain, num_prop=part.num_prop,
            list_region=sol, num_regions=len(sol), adj=sp.lil_matrix(adj),
            list_prop_symbol=part.list_prop_symbol, list_subsys=subsys_list
        )
        
        # plot pair under reachability check
        ax2.clear()
        si.plot(ax=ax2, color='green')
        sj.plot(ax2, color='red', hatch='o', alpha=0.5)
        plot_transition_arrow(si, sj, ax2)
        
        S0.plot(ax2, color='none', hatch='/', alpha=0.3)
        fig.canvas.draw()
        
        # plot partition
        ax1.clear()
        plot_partition(tmp_part, transitions, ax=ax1, color_seed=23)
        
        # plot dynamics
        ssys.plot(ax1, show_domain=False)
        
        # plot hatched continuous propositions
        if cont_props is not None:
	        for (prop, poly) in cont_props.iteritems():
	            poly.plot(ax1, color='none', hatch='/')
	            poly.text(prop, ax1, color='yellow')
        
        fig.canvas.draw()
        
        # scale view based on domain,
        # not only the current polytopes si, sj
        l,u = pc.bounding_box(part.domain)
        ax2.set_xlim(l[0,0], u[0,0])
        ax2.set_ylim(l[1,0], u[1,0])
        
        if save_img:
            fname = 'movie' +str(iter_count).zfill(3)
            fname += '.' + file_extension
            fig.savefig(fname, dpi=250)
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
    
    param = {'N':N, 'closed_loop':closed_loop,
        'conservative':conservative,
        'use_all_horizon':use_all_horizon}
    
    assert(len(prop_list) == n)
    
    return AbstractSysDyn(
        ppp=new_part,
        ofts=ofts,
        original_regions=orig_list,
        orig=orig,
        disc_params=param
    )

def reachable_within(trans_length, adj_k, adj):
    """Find cells reachable within trans_length hops.
    """
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
#             logger.info("Transition found")
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
#             logger.info("\n Adding state " + str(size-1) + "\n")
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
#             logger.info("No transition found, intersect vol: " + str(vol1) )
#             transitions[i,j] = 0
#                   
#     new_part = PropPreservingPartition(
#                    domain=part.domain, num_prop=part.num_prop,
#                    list_region=sol, num_regions=len(sol), adj=np.array([]), 
#                    trans=transitions, list_prop_symbol=part.list_prop_symbol,
#                    original_regions=orig_list, orig=orig)                           
#     return new_part

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

import warnings
import pprint
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
    
    - ppp: Partition into Regions.
        Each Region corresponds to
        a discrete state of the abstraction
        
        type: PropPreservingPartition
    
    - ts: Finite transition system abstracting the continuous system.
        Each state corresponds to a Region in ppp.
        It can be fed into discrete synthesis algorithms.
        
        type: OpenFTS
    
    - ppp2ts: map Regions to states of the transition system
        Each index denotes the Region with same index in:
            
            ppp.regions
            
        type: list of states
            (usually each state is a str)
    
    - original_regions: Regions of original
        proposition preserving partition
        Used for non-conservative planning.
        
        type: list of Region
    
    - ppp2orig: map of new Regions to original Regions:
            
            - i-th new Region in C{ppp.regions}
            - ppp2orig[i]-th original Region in C{original_regions}
            
        type: list of indices
    
    - ppp2pwa: map Regions to PwaSubSys.list_subsys
        Each partition corresponds to some mode.
        (for switched systems)
        
        In each mode a PwaSubSys is active.
        This PwaSubSys comprises of subsystems,
        which are listed in PwaSubSys.list_subsys.
        
        The list C{ppp2pwa} means:
        
            - i-th Region in C{regions}
            - ppp2pwa[i]-th system in PwaSubSys.list_subsys
                is active in the i-th Region
        
        type: list
    
    - disc_params: parameters used in discretization that 
        should be passed to the controller refinement
        to ensure consistency
        
        type: dict
    
    If any of the above is not given,
    then it is initialized to None.
            
    Note1: There could be some redundancy in ppp and ofts in that they are
    both decorated with propositions. This might be useful to keep each of 
    them as functional units on their own (possible to change later). 
    """
    def __init__(self, ppp=None, ts=None, ppp2ts=None,
                 original_regions=None, ppp2orig=None,
                 ppp2pwa=None, disc_params=None):
        if disc_params is None:
            disc_params = dict()
        
        # check consistency
        group0 = [ppp, ts, ppp2ts]
        names0 = 'ppp, ts, ppp2ts'
        all_dict0 = _all_dict(group0, names0)
        
        # disc_aprams excluded, because it is always a dict
        group1 = [original_regions, ppp2orig, ppp2pwa]
        names1 = 'original_regions, ppp2orig, ppp2pwa'
        all_dict1 = _all_dict(group1, names1)
        
        # prohibited combination
        if all_dict0 and not all_dict1:
            msg = 'if ' + names0 + ' are dict,\n'
            msg += 'so must ' + names1 + ' be.'
            raise Exception(msg)
        
        self.ppp = ppp
        self.ts = ts
        self.original_regions = original_regions
        self.ppp2orig = ppp2orig
        self.ppp2pwa = ppp2pwa
        self.disc_params = disc_params
    
    def __str__(self):
        s = str(self.ppp)
        s += str(self.ts)
        
        s += 30 * '-' + '\n'
        if isinstance(self.original_regions, dict):
            s += 'Original Regions:\n\n'
            for mode, orig_reg in self.original_regions.iteritems():
                s += 'mode: ' + str(mode)
                s += ', has: ' + str(len(orig_reg)) + ' Regions\n'
        else:
            s += 'Original Regions: ' + str(len(self.original_regions))
        
        s += 'Map of New to Original Regions:\n'
        if isinstance(self.ppp2orig, dict):
            for mode, ppp2orig in self.ppp2orig.iteritems():
                s += '\t mode: ' + str(mode) + '\n'
                s += self._ppp2orig_str(ppp2orig) + '\n'
        else:
            s += self._ppp2orig_str(self.ppp2orig) + '\n'
        
        s += 'Discretization Options:\n\t'
        s += pprint.pformat(self.disc_params) +'\n'
        
        return s
    
    def _ppp2orig_str(self, ppp2orig):
        s = ''
        for i, original_region in enumerate(ppp2orig):
            s += '\t\t' + str(i) + ' -> ' + str(original_region) + '\n'
        return s
    
    def _debug_str_(self):
        s = str(self.ppp)
        s += str(self.ts)
        
        s += 'Original Regions List:\n\n'
        for i, region in enumerate(self.original_regions):
            s += 'Region: ' + str(i) + '\n'
            s += str(region) + '\n'
    
    def plot(self):
        if self.ppp is None or self.ts is None:
            return
        
        if isinstance(self.ppp, dict):
            for mode, ppp in self.ppp.iteritems():
                ax = ppp.plot()
                ax.set_title('Partition for mode: ' + str(mode))
        else:
            self.ppp.plot(trans=self.ts)
        
        if isinstance(self.ts, dict):
            for ts in self.ts:
                ts.plot()
        else:
            self.ts.plot()

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
        (part, ppp2pwa) = pwa_partition(ssys, part)
    
    # Save original polytopes, require them to be convex 
    if conservative:
        orig_list = None
        orig = 0
    else:
        (part, new2old) = part2convex(part) # convexify
        
        # map new regions to pwa subsystems
        if ispwa:
            ppp2pwa = [ppp2pwa[i] for i in new2old]
        
        remove_trans = False # already allowed in nonconservative
        orig_list = []
        for poly in part.regions:
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
    num_regions = len(part.regions)
    transitions = np.zeros(
        [num_regions, num_regions],
        dtype = int
    )
    sol = deepcopy(part.regions)
    adj = part.adj.copy()
    adj = adj.todense()
    adj = np.array(adj)
    
    # next 2 lines omitted in discretize_overlap
    subsys_list = list(ppp2pwa)
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
                isect = pc.Region([isect], si.props)
            else:
                isect.props = si.props.copy()
        
            if len(diff) == 0:
                diff = pc.Region([diff], si.props)
            else:
                diff.props = si.props.copy()
        
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
            domain=part.domain,
            regions=sol, adj=sp.lil_matrix(adj),
            prop_regions=part.prop_regions
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
        domain=part.domain,
        regions=sol, adj=sp.lil_matrix(adj),
        prop_regions=part.prop_regions
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
    atomic_propositions = set(part.prop_regions)
    ofts.atomic_propositions.add_from(atomic_propositions)
    prop_list = []
    for region in sol:
        state_prop = region.props.copy()
        
        prop_list.append(state_prop)
    
    ofts.states.labels(ofts_states, prop_list)
    
    param = {
        'N':N,
        'trans_length':trans_length,
        'closed_loop':closed_loop,
        'conservative':conservative,
        'use_all_horizon':use_all_horizon,
        'min_cell_volume':min_cell_volume,
        'max_num_poly':max_num_poly
    }
    
    assert(len(prop_list) == n)
    
    return AbstractSysDyn(
        ppp=new_part,
        ts=ofts,
        ppp2ts=ofts_states,
        original_regions=orig_list,
        ppp2orig=orig,
        ppp2pwa=subsys_list,
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
#                 isect = pc.Region([isect], si.props)
#             else:
#                 isect.props = si.props.copy()
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
#                    domain=part.domain,
#                    regions=sol, adj=np.array([]),
#                    trans=transitions, prop_regions=part.prop_regions,
#                    original_regions=orig_list, orig=orig)                           
#     return new_part

def discretize_switched(ppp, hybrid_sys, disc_params=None, plot=False):
    
    if disc_params is None:
        disc_params = {'N':1, 'trans_length':1}
    
    logger.info('discretizing hybrid system')
    
    modes = hybrid_sys.modes
    mode_nums = hybrid_sys.disc_domain_size
    
    abstractions = dict()
    for mode in modes:
        logger.debug(30*'-'+'\n')
        logger.info('Abstracting mode: ' + str(mode))
        
        cont_dyn = hybrid_sys.dynamics[mode]
        
        absys = discretize(
            ppp, cont_dyn,
            min_cell_volume=0.01,
            plotit=False,
            **disc_params[mode]
        )
        logger.debug('Mode Abstraction:\n' + str(absys) +'\n')
        
        abstractions[mode] = absys
    
    (merged_abstr, ap_labeling) = merge_partitions(abstractions)
    n = len(merged_abstr.ppp.regions)
    logger.info('Merged partition has: ' + str(n) + ', states')
    
    trans = dict()
    for mode in modes:
        cont_dyn = hybrid_sys.dynamics[mode]
        
        params = disc_params[mode]
        
        trans[mode] = get_transitions(
            merged_abstr, mode, cont_dyn,
            N=params['N'], trans_length=params['trans_length']
        )
    
    merge_abstractions(merged_abstr, trans,
                       abstractions, modes, mode_nums)
    merged_abstr.disc_params = disc_params
    
    if plot:
        plot_mode_partitions(abstractions, merged_abstr)
    
    return merged_abstr

def plot_mode_partitions(abstractions, merged_abs):
    try:
        from tulip.graphics import newax
    except:
        warnings.warn('could not import newax, no partitions plotted.')
        return
    
    ax, fig = newax()
    
    for mode, ab in abstractions.iteritems():
        ab.ppp.plot(plot_numbers=False, ax=ax, trans=ab.ts)
        plot_annot(ax)
        fname = 'part_' + str(mode) + '.pdf'
        fig.savefig(fname)
    
    merged_abs.ppp.plot(plot_numbers=False, trans=merged_abs.ts, ax=ax)
    plot_annot
    fname = 'part_merged' + '.pdf'
    fig.savefig(fname)

def plot_annot(ax):
    fontsize = 5
    
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    ax.set_xlabel('$v_1$', fontsize=fontsize+6)
    ax.set_ylabel('$v_2$', fontsize=fontsize+6)

def merge_abstractions(merged_abstr, trans, abstr, modes, mode_nums):
    """Construct merged transitions.
    
    @type merged_part: AbstractSysDyn
    
    @type abstr: list of AbstractSysDyn
    
    @type hybrid_sys: HybridSysDyn
    """
    # TODO: check equality of atomic proposition sets
    aps = abstr[modes[0]].ts.atomic_propositions
    
    logger.info('APs: ' + str(aps))
    
    sys_ts = trs.OpenFTS()
    
    # create stats
    n = len(merged_abstr.ppp.regions)
    states = ['s'+str(i) for i in xrange(n) ]
    sys_ts.states.add_from(states)
    
    sys_ts.atomic_propositions.add_from(aps)
    
    # copy AP labels from regions to discrete states
    ppp2ts = states
    for (i, state) in enumerate(ppp2ts):
        props =  merged_abstr.ppp.regions[i].props
        sys_ts.states.label(state, props)
    
    # create mode actions
    sys_actions = [str(s) for e,s in modes]
    env_actions = [str(e) for e,s in modes]
    
    # no env actions ?
    if mode_nums[0] == 0:
        actions_per_mode = {
            (e,s):{'sys_actions':str(s)}
            for e,s in modes
        }
        sys_ts.sys_actions.add_from(sys_actions)
    elif mode_nums[1] == 0:
        # no sys actions
        actions_per_mode = {
            (e,s):{'env_actions':str(e)}
            for e,s in modes
        }
        sys_ts.env_actions.add_from(env_actions)
    else:
        actions_per_mode = {
            (e,s):{'env_actions':str(e), 'sys_actions':str(s)}
            for e,s in modes
        }
        sys_ts.env_actions.add_from([str(e) for e,s in modes])
        sys_ts.sys_actions.add_from([str(s) for e,s in modes])
    
    for mode in modes:
        env_sys_actions = actions_per_mode[mode]
        adj = trans[mode]
        
        sys_ts.transitions.add_labeled_adj(
            adj = adj,
            adj2states = states,
            labels = env_sys_actions
        )
    
    merged_abstr.ts = sys_ts
    merged_abstr.ppp2ts = ppp2ts

def get_transitions(abstract_sys, mode, ssys, N=10, closed_loop=True,
                    trans_length=1, abs_tol=1e-7):
    logger.info('checking which transitions remain feasible after merging')
    part = abstract_sys.ppp
    
    # Initialize matrix for pairs to check
    IJ = part.adj.copy()
    if trans_length > 1:
        k = 1
        while k < trans_length:
            IJ = np.dot(IJ, part.adj)
            k += 1
        IJ = (IJ > 0).astype(int)
    
    # Initialize output
    n = len(part.regions)
    transitions = sp.lil_matrix((n, n), dtype=int)
    
    # Do the abstraction
    n_checked = 0
    n_found = 0
    while np.sum(IJ) > 0:
        n_checked += 1
        
        ind = np.nonzero(IJ)
        i = ind[1][0]
        j = ind[0][0]
        IJ[j,i] = 0
        
        logger.debug('checking transition: ' + str(i) + ' -> ' + str(j))
        
        si = part.regions[i]
        sj = part.regions[j]
        
        orig_region_idx = abstract_sys.ppp2orig[mode][i]
        
        subsys_idx = abstract_sys.ppp2pwa[mode][i]
        active_subsystem = ssys.list_subsys[subsys_idx]
        
        # Use original cell as trans_set
        S0 = solve_feasible(
            si, sj, active_subsystem, N,
            closed_loop = closed_loop,
            trans_set = abstract_sys.original_regions[mode][orig_region_idx]
        )
        
        diff = pc.mldivide(si, S0)
        vol2 = pc.volume(diff)
                    
        if vol2 < abs_tol:
            transitions[j,i] = 1 
            msg = '\t Feasible transition.'
            n_found += 1
        else:
            transitions[j,i] = 0
            msg = '\t Not feasible transition.'
        logger.debug(msg)
    logger.info('Checked: ' + str(n_checked))
    logger.info('Found: ' + str(n_found))
    logger.info('Survived merging: ' + str(float(n_found) / n_checked) + ' % ')
            
    return transitions
    
def merge_partitions(abstractions):
    if len(abstractions) == 0:
        warnings.warn('Abstractions empty, nothing to merge.')
        return
    
    # consistency check
    for ab1 in abstractions.itervalues():
        for ab2 in abstractions.itervalues():
            p1 = ab1.ppp
            p2 = ab2.ppp
            
            if p1.prop_regions != p2.prop_regions:
                msg = 'merge: partitions have different sets '
                msg += 'of continuous propositions'
                raise Exception(msg)
            
            if not (p1.domain.A == p2.domain.A).all() or \
            not (p1.domain.b == p2.domain.b).all():
                raise Exception('merge: partitions have different domains')
            
            # check equality of original partitions
            if ab1.original_regions == ab2.original_regions:
                logger.info('original partitions happen to be equal')
    
    init_mode = abstractions.keys()[0]
    all_modes = set(abstractions)
    remaining_modes = all_modes.difference(set([init_mode]))
    
    print('init mode: ' + str(init_mode))
    print('all modes: ' + str(all_modes))
    print('remaining modes: ' + str(remaining_modes))
    
    # initialize iteration data
    prev_modes = [init_mode]
    
    ab0 = abstractions[init_mode]
    regions = ab0.ppp.regions
    parents = range(len(regions) )
    ppp2orig = ab0.ppp2orig
    ppp2pwa = ab0.ppp2pwa
    ap_labeling = {i:reg.props for i,reg in enumerate(regions)}
    
    for cur_mode in remaining_modes:
        ab2 = abstractions[cur_mode]
        
        r = merge_partition_pair(
            regions, ab2, cur_mode, prev_modes,
            parents, ap_labeling,
            ppp2orig, ppp2pwa
        )
        regions, parents, ap_labeling, ppp2orig, ppp2pwa = r
        
        prev_modes += [cur_mode]
    
    new_list = regions
    
    # build adjacency based on spatial adjacencies of
    # component abstractions.
    # which justifies the assumed symmetry of part1.adj, part2.adj
    n_reg = len(new_list)
    
    adj = np.zeros([n_reg, n_reg], dtype=int)
    for i, reg_i in enumerate(new_list):
        for j, reg_j in enumerate(new_list[(i+1):]):
            touching = False
            for mode in abstractions:
                pi = parents[mode][i]
                pj = parents[mode][j]
                
                part = abstractions[mode].ppp
                
                if (part.adj[pi, pj] == 1) or (pi == pj):
                    touching = True
                    break
            
            if not touching:
                continue
            
            if pc.is_adjacent(reg_i, reg_j):
                adj[i,j] = 1
                adj[j,i] = 1
        adj[i,i] = 1
    
    ppp = PropPreservingPartition(
        domain=ab0.ppp.domain,
        regions=new_list,
        prop_regions=ab0.ppp.prop_regions,
        adj=adj
    )
    
    switched_original_regions = {
        mode:abstractions[mode].original_regions for mode in abstractions
    }
    
    abstraction = AbstractSysDyn(
        ppp = ppp,
        original_regions = switched_original_regions,
        ppp2orig = ppp2orig,
        ppp2pwa=ppp2pwa
    )
    
    return (abstraction, ap_labeling)

def merge_partition_pair(
    ab1, ab2,
    cur_mode, prev_modes,
    old_parents, old_ap_labeling,
    ppp2orig, ppp2pwa
):
    # TODO: track initial states: better done automatically with AP 'init'
    
    logger.info('merging partitions')
    
    part1 = ab1.ppp
    part2 = ab2.ppp
    
    modes = prev_modes + [cur_mode]
    
    new_list = []
    orig = {mode:[] for mode in modes}
    subsystems = {mode:[] for mode in modes}
    parents = {mode:dict() for mode in modes}
    ap_labeling = dict()
    
    for i in xrange(len(part1.regions)):
        for j in xrange(len(part2.regions)):
            isect = pc.intersect(part1.regions[i],
                                 part2.regions[j])
            rc, xc = pc.cheby_ball(isect)
            
            # no intersection ?
            if rc < 1e-5:
                continue
            logger.info('merging region: A' + str(i) + ', with: B' + str(j))
            
            # if Polytope, make it Region
            if len(isect) == 0:
                isect = pc.Region([isect])
            
            # label the Region with propositions
            isect.props = part1.regions[i].props.copy()
            
            new_list.append(isect)
            idx = new_list.index(isect)
            
            # keep track of parents
            for mode in old_parents:
                parents[mode][idx] = i
            parents[cur_mode][idx] = j
            
            # keep track of original regions
            for mode in ppp2orig:
                orig[mode] += [ppp2orig[mode][i] ]
            orig[cur_mode] += [ab2.ppp2orig[j] ]
            
            # keep track of subsystems
            for mode in ppp2pwa:
                subsystems[mode] += [ppp2pwa[mode][i] ]
            subsystems[cur_mode] += [ab2.ppp2pwa[j] ]
            
            # union of AP labels from parent states
            ap_label_1 = old_ap_labeling[i]
            ap_label_2 = ab2.ts.states.label_of('s'+str(j))['ap']
            
            logger.debug('AP label 1: ' + str(ap_label_1))
            logger.debug('AP label 2: ' + str(ap_label_2))
            
            # original partitions may be different if pwa_partition used
            # but must originate from same initial partition,
            # i.e., have same continuous propositions, checked above
            #
            # so no two intersecting regions can have different AP labels,
            # checked here
            if ap_label_1 != ap_label_2:
                msg = 'Inconsistent AP labels between intersecting regions\n'
                msg += 'of partitions of switched system.'
                raise Exception(msg)
            
            ap_labeling[idx] = ap_label_1
    
    return new_list, parents, ap_labeling, ppp2orig, ppp2pwa

def _all_dict(r, names='?'):
    f = lambda x: isinstance(x, dict)
    
    n_dict = len(filter(f, r))
    
    if n_dict == 0:
        return False
    elif n_dict == len(r):
        return True
    else:
        msg = 'Mixed dicts with non-dicts among: ' + str(names)
        raise Exception(msg)

"""
Auxiliary discretization and specification functions
"""
import logging
logger = logging.getLogger(__name__)

import numpy as np
from scipy import sparse as sp

from tulip import abstract
import tulip.polytope as pc

def get_transitions(abstract_sys, ssys, N=10, closed_loop=True,
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
    n = part.num_regions
    transitions = sp.lil_matrix((n, n), dtype=int)
    
    # Do the abstraction
    while np.sum(IJ) > 0:
        ind = np.nonzero(IJ)
        i = ind[1][0]
        j = ind[0][0]
        IJ[j,i] = 0
        
        logger.info('checking transition: ' + str(i) + ' -> ' + str(j))
        
        si = part.list_region[i]
        sj = part.list_region[j]
        
        # Use original cell as trans_set
        S0 = abstract.feasible.solve_feasible(
            si, sj, ssys, N,
            closed_loop = closed_loop,
            trans_set = abstract_sys.orig_list_region[abstract_sys.orig[i]]
        )
        
        diff = pc.mldivide(si, S0)
        vol2 = pc.volume(diff)
                    
        if vol2 < abs_tol:
            transitions[j,i] = 1 
            msg = '\t Feasible transition.'
        else:
            transitions[j,i] = 0
            msg = '\t Not feasible transition.'
        logger.info(msg)
            
    return transitions
    
def merge_partitions(abstract1, abstract2):
    logger.info('merging partitions')
    
    part1 = abstract1.ppp
    part2 = abstract2.ppp
    
    if part1.num_prop != part2.num_prop:
        msg = "merge: partitions have different"
        msg += " number of propositions."
        raise Exception(msg)
    
    if part1.list_prop_symbol != part2.list_prop_symbol:
        msg = 'merge: partitions have different propositions'
        raise Exception(msg)
    
    if len(abstract1.orig_list_region) != \
    len(abstract2.orig_list_region):
        msg = "merge: partitions have different"
        msg += " number of original regions"
        raise Exception(msg)
    
    if not (part1.domain.A == part2.domain.A).all() or \
    not (part1.domain.b == part2.domain.b).all():
        raise Exception('merge: partitions have different domains')

    new_list = []
    orig = []
    parent_1 = []
    parent_2 = []
    for i in range(part1.num_regions):
        for j in range(part2.num_regions):
            logger.info('mergin region: A' + str(i) + ', with: B' + str(j))
            
            isect = pc.intersect(part1.list_region[i],
                                 part2.list_region[j])
            rc, xc = pc.cheby_ball(isect)
            
            # no intersection ?
            if rc < 1e-5:
                continue
            
            # if Polytope, make it Region
            if len(isect) == 0:
                isect = pc.Region([isect], [])
            
            isect.list_prop = part1.list_region[i].list_prop
            new_list.append(isect)
            parent_1.append(i)
            parent_2.append(j)
            
            if abstract1.orig[i] != abstract2.orig[j]:
                raise Exception("not same orig, partitions don't have the \
                                  same origin.")
            orig.append(abstract1.orig[i])
    
    adj = np.zeros([len(new_list), len(new_list)], dtype=int)
    for i in range(len(new_list)):
        for j in range(i+1, len(new_list)):
            
            if (part1.adj[parent_1[i], parent_1[j]] == 1) or \
               (part2.adj[parent_2[i], parent_2[j]] == 1) or \
               (parent_1[i] == parent_1[j]) or \
               (parent_2[i] == parent_2[j]):
                if pc.is_adjacent(new_list[i], new_list[j]):
                    adj[i,j] = 1
                    adj[j,i] = 1
        adj[i,i] = 1
    
    ppp = abstract.PropPreservingPartition(
        domain=part1.domain,
        num_prop=part1.num_prop,
        list_region=new_list,
        num_regions=len(new_list),
        list_prop_symbol=part1.list_prop_symbol,
        adj=adj,
        #list_subsys
    )
    
    return abstract.discretization.AbstractSysDyn(
        ppp = ppp,
        ofts = None,
        orig_list_region = abstract1.orig_list_region,
        orig = np.array(orig)
    )

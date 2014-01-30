"""
Auxiliary discretization and specification functions
"""
import logging
logger = logging.getLogger(__name__)

import numpy as np
from scipy import sparse as sp

from tulip import abstract
import tulip.polytope as pc
from tulip import transys as trs

def discretize_hybrid(ppp, hybrid_sys, N=1, trans_len=1):
    
    print('discretizing hybrid system')
    
    modes = hybrid_sys.modes
    
    abstractions = dict()
    for mode in modes:
        print(30*'-'+'\n')
        print('Abstracting mode: ' + str(mode))
        
        cont_dyn = hybrid_sys.dynamics[mode]
        
        absys = abstract.discretize(
            ppp, cont_dyn, N=N,
            trans_length=trans_len,
            min_cell_volume=0.01,
            plotit=False
        )
        print('Mode Abstraction:\n' + str(absys) +'\n')
        
        abstractions[mode] = absys
    
    (merged_abstr, ap_labeling) = merge_partitions(abstractions)
    n = len(merged_abstr.ppp.regions)
    logger.info('Merged partition has: ' + str(n) + ', states')
    
    trans = dict()
    for mode in modes:
        cont_dyn = hybrid_sys.dynamics[mode]
        
        trans[mode] = get_transitions(
            merged_abstr, mode, cont_dyn,
            N=N, trans_length=trans_len
        )
    
    merge_abstractions(merged_abstr, trans, abstractions, modes)

def merge_abstractions(merged_abstr, trans, abstr, modes, mode_nums):
    """Construct merged transitions.
    
    @type merged_part: AbstractSysDyn
    
    @type abstr: list of AbstractSysDyn
    
    @type hybrid_sys: HybridSysDyn
    """
    # allow for different AP sets
    aps1 = abstr[0].ts.atomic_propositions
    aps2 = abstr[1].ts.atomic_propositions
    all_aps = aps1 | aps2
    logger.info('all APs: ' + str(all_aps))
    
    sys_ts = trs.OpenFTS()
    sys_ts.atomic_propositions.add_from(all_aps)
    
    # copy AP labels from parents
    
    
    # ignore singleton modes
    if mode_nums[0]:
        str_modes = [str(s) for e,s in modes]
    elif mode_nums[1]:
        str_modes = [str(e) for e,s in modes]
    else:
        str_modes = [str(e) + '_' + str(s) for e,s in modes]
    
    sys_ts.env_actions.add_from(str_modes)
    
    n = len(merged_abstr.list_reg)
    states = ['s'+str(i) for i in xrange(n) ]
    sys_ts.states.add_from(states)
    
    
    for mode in modes:
        str_mode = str_modes[mode]
        adj = trans[mode]
        
        sys_ts.transitions.add_labeled_adj(
            adj = adj,
            adj2states = states,
            labels = {'env_actions':str_mode}
        )
    
    merged_abstr.ts = sys_ts
    merged_abstr.ppp2ts = states

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
    while np.sum(IJ) > 0:
        ind = np.nonzero(IJ)
        i = ind[1][0]
        j = ind[0][0]
        IJ[j,i] = 0
        
        logger.info('checking transition: ' + str(i) + ' -> ' + str(j))
        
        si = part.regions[i]
        sj = part.regions[j]
        
        orig_region_idx = abstract_sys.ppp2orig[mode][i]
        
        subsys_idx = abstract_sys.ppp.subsystems[mode][i]
        active_subsystem = ssys.list_subsys[subsys_idx]
        
        # Use original cell as trans_set
        S0 = abstract.feasible.solve_feasible(
            si, sj, active_subsystem, N,
            closed_loop = closed_loop,
            trans_set = abstract_sys.original_regions[mode][orig_region_idx]
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
    
def merge_partitions(abstractions):
    logger.info('merging partitions')
    
    mode1, mode2 = abstractions
    
    abstract1 = abstractions[mode1]
    abstract2 = abstractions[mode2]
    
    part1 = abstract1.ppp
    part2 = abstract2.ppp
    
    if part1.prop_regions != part2.prop_regions:
        msg = 'merge: partitions have different sets '
        msg += 'of continuous propositions'
        raise Exception(msg)
    
    if not (part1.domain.A == part2.domain.A).all() or \
    not (part1.domain.b == part2.domain.b).all():
        raise Exception('merge: partitions have different domains')

    new_list = []
    orig = {mode:[] for mode in abstractions}
    subsystems = {mode:[] for mode in abstractions}
    
    parent_1 = dict()
    parent_2 = dict()
    
    ap_labeling = dict()
    for i in range(len(part1.regions)):
        for j in range(len(part2.regions)):
            logger.info('mergin region: A' + str(i) + ', with: B' + str(j))
            
            isect = pc.intersect(part1.regions[i],
                                 part2.regions[j])
            rc, xc = pc.cheby_ball(isect)
            
            # no intersection ?
            if rc < 1e-5:
                continue
            
            # if Polytope, make it Region
            if len(isect) == 0:
                isect = pc.Region([isect], [])
            
            isect.props = part1.regions[i].props
            new_list.append(isect)
            
            idx = new_list.index(isect)
            
            parent_1[idx] = i
            parent_2[idx] = j
            
            orig[mode1] += [abstract1.ppp2orig[i] ]
            orig[mode2] += [abstract2.ppp2orig[j] ]
            
            subsystems[mode1] += [abstract1.ppp.subsystems[i] ]
            subsystems[mode2] += [abstract2.ppp.subsystems[j] ]
            
            # union of AP labels from parent states
            ap_label_1 = abstract1.ts.states.label_of('s'+str(i))['ap']
            ap_label_2 = abstract2.ts.states.label_of('s'+str(j))['ap']
            
            logger.debug('AP label 1: ' + str(ap_label_1))
            logger.debug('AP label 2: ' + str(ap_label_2))
            
            ap_labeling[idx] = ap_label_1 | ap_label_2
    
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
        regions=new_list,
        prop_regions=part1.prop_regions,
        adj=adj,
        subsystems=subsystems
    )
    
    # check equality of original partitions
    if abstract1.original_regions == abstract2.original_regions:
        print('original partitions happen to be equal')
    
    switched_original_regions = {
        mode:abstractions[mode].original_regions for mode in abstractions
    }
    
    abstraction = abstract.discretization.AbstractSysDyn(
        ppp = ppp,
        original_regions = switched_original_regions,
        ppp2orig = orig
    )
    
    return (abstraction, ap_labeling)

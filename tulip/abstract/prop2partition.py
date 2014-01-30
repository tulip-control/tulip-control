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
Proposition preserving partition module.

Restructured by NO, 30 Jun 2013.
"""
from warnings import warn

import numpy as np
from scipy import sparse as sp
import copy

from tulip import polytope as pc
from .plot import plot_partition

hl = 40 * '-'

def prop2part(state_space, cont_props_dict):
    """Main function that takes a domain (state_space) and a list of
    propositions (cont_props), and returns a proposition preserving
    partition of the state space.
    
    @param state_space: problem domain
    @type state_space: polytope.Polytope
    
    @param cont_props_dict: propositions
    @type cont_props_dict: dict of polytope.Polytope
    
    @return: state space quotient partition induced by propositions
    @rtype: PropPreservingPartition
    
    see also
    --------
    PropPreservingPartition,
    polytope.Polytope
    """
    first_poly = [] #Initial Region's list_poly atribute 
    first_poly.append(state_space)
    
    regions = []
    regions.append(
        pc.Region(list_poly=first_poly)
    )
    
    # init partition
    mypartition = PropPreservingPartition(
        domain = copy.deepcopy(state_space),
        regions = regions,
        prop_regions = copy.deepcopy(cont_props_dict)
    )
    
    for cur_prop in cont_props_dict:
        cur_prop_poly = cont_props_dict[cur_prop]
        
        num_reg = len(mypartition.regions)
        prop_holds_reg = []
        
        for i in xrange(num_reg): #i region counter
            region_now = mypartition.regions[i].copy()
            #loop for prop holds
            prop_holds_reg.append(0)
            
            prop_now = mypartition.regions[i].props.copy()
            
            dummy = region_now.intersect(cur_prop_poly)
            
            # does cur_prop hold in dummy ?
            if pc.is_fulldim(dummy):
                dum_prop = prop_now.copy()
                dum_prop.add(cur_prop)
                
                # is dummy a Polytope ?
                if len(dummy) == 0:
                    mypartition.regions[i] = pc.Region(
                        [dummy],
                        dum_prop
                    )
                else:
                    # dummy is a Region
                    dummy.props = dum_prop.copy()
                    mypartition.regions[i] = dummy.copy()
                prop_holds_reg[-1] = 1
            else:
                #does not hold in the whole region
                # (-> no need for the 2nd loop)
                mypartition.regions.append(region_now)
                continue
                
            #loop for prop does not hold
            mypartition.regions.append(
                pc.Region(
                    list_poly=[],
                    props=prop_now
                )
            )
            dummy = region_now.diff(cur_prop_poly)
            
            if pc.is_fulldim(dummy):
                dum_prop = prop_now.copy()
                
                # is dummy a Polytope ?
                if len(dummy) == 0:
                    mypartition.regions[-1] = pc.Region(
                        [pc.reduce(dummy)],
                        dum_prop
                    )
                else:
                    # dummy is a Region
                    dummy.props = dum_prop.copy()
                    mypartition.regions[-1] = dummy.copy()
            else:
                mypartition.regions.pop()
        
        count = 0
        for hold_count in xrange(len(prop_holds_reg)):
            if prop_holds_reg[hold_count]==0:
                mypartition.regions.pop(hold_count-count)
                count+=1
    
    mypartition.adj = find_adjacent_regions(mypartition).copy()
    
    return mypartition

def part2convex(ppp):
    """This function takes a proposition preserving partition and generates 
    another proposition preserving partition such that each part in the new 
    partition is a convex polytope
    
    @type ppp: PropPreservingPartition
    
    @return: refinement into convex polytopes
    @rtype: PropPreservingPartition
    """
    cvxpart = PropPreservingPartition(
        domain=copy.deepcopy(ppp.domain),
        prop_regions=copy.deepcopy(ppp.prop_regions)
    )
    subsys_list = []
    for i in xrange(len(ppp.regions)):
        simplified_reg = ppp.regions[i] + ppp.regions[i]
        
        for j in xrange(len(simplified_reg)):
            region_now = pc.Region(
                [simplified_reg.list_poly[j]],
                ppp.regions[i].props
            )
            cvxpart.regions.append(region_now)
            if ppp.subsystems is not None:
                subsys_list.append(ppp.subsystems[i])
    
    if ppp.subsystems is not None:
        cvxpart.subsystems = subsys_list
    
    cvxpart.adj = find_adjacent_regions(cvxpart).copy()
    
    return cvxpart
    
def pwa_partition(pwa_sys, ppp, abs_tol=1e-5):
    """This function takes a piecewise affine system pwa_sys and a proposition 
    preserving partition ppp whose domain is a subset of the domain of pwa_sys
    and returns a refined proposition preserving partition where in each
    region a unique subsystem of pwa_sys is active.
    
    Modified from Petter Nilsson's code implementing merge algorithm in 
    Nilsson et al. `Temporal Logic Control of Switched Affine Systems with an
    Application in Fuel Balancing`, ACC 2012.
    
    @type pwa_sys: hybrid.PwaSysDyn
    @type ppp: PropPreservingPartition
    
    @return: object with subsystem assignments
    @rtype: PropPreservingPartition
    
    see also
    --------
    discretize.discretize
    """
    if pc.is_fulldim(ppp.domain.diff(pwa_sys.domain) ):
        raise Exception("pwaPartition: "
            "pwa system is not defined everywhere in state space")

    new_list = []
    subsys_list = []
    parent = []
    for i in xrange(len(pwa_sys.list_subsys)):
        for j in xrange(len(ppp.regions)):
            isect = pwa_sys.list_subsys[i].domain.intersect(
                ppp.regions[j]
            )
            
            if pc.is_fulldim(isect):
                rc, xc = pc.cheby_ball(isect)
                if rc < abs_tol:
                    print("Warning: One of the regions in the refined PPP is "
                          "too small, this may cause numerical problems")
                if len(isect) == 0:
                    isect = pc.Region([isect])
                
                isect.props = ppp.regions[j].props.copy()
                subsys_list.append(i)
                new_list.append(isect)
                parent.append(j)
    
    adj = sp.lil_matrix((len(new_list), len(new_list)), dtype=np.int8)
    for i in xrange(len(new_list)):
        for j in xrange(i+1, len(new_list)):
            if (ppp.adj[parent[i], parent[j]] == 1) or \
                    (parent[i] == parent[j]):
                if pc.is_adjacent(new_list[i], new_list[j]):
                    adj[i,j] = 1
                    adj[j,i] = 1
        adj[i,i] = 1
            
    return PropPreservingPartition(
        domain = ppp.domain,
        regions = new_list,
        adj = adj,
        prop_regions = ppp.prop_regions,
        subsystems = subsys_list
    )
                
def add_grid(ppp, grid_size=None, num_grid_pnts=None, abs_tol=1e-10):
    """ This function takes a proposition preserving partition ppp and the size 
    of the grid or the number of grids, and returns a refined proposition 
    preserving partition with grids.
     
    Input:
    
    - `ppp`: a PropPreservingPartition object
    - `grid_size`: the size of the grid,
        type: float or list of float
    - `num_grid_pnts`: the number of grids for each dimension,
        type: integer or list of integer
    
    Output:
    
    - A PropPreservingPartition object with grids
        
    Note: There could be numerical instabilities when the continuous 
    propositions in ppp do not align well with the grid resulting in very small 
    regions. Performace significantly degrades without glpk.
    """
    if (grid_size!=None)&(num_grid_pnts!=None):
        raise Exception("add_grid: Only one of the grid size or number of \
                        grid points parameters is allowed to be given.")
    if (grid_size==None)&(num_grid_pnts==None):
        raise Exception("add_grid: At least one of the grid size or number of \
                         grid points parameters must be given.")
 
    dim=len(ppp.domain.A[0])
    domain_bb = pc.bounding_box(ppp.domain)
    size_list=list()
    if grid_size!=None:
        if isinstance( grid_size, list ):
            if len(grid_size) == dim:
                size_list=grid_size
            else:
                raise Exception(
                    "add_grid: grid_size isn't given in a correct format."
                )
        elif isinstance( grid_size, float ):
            for i in xrange(dim):
                size_list.append(grid_size)
        else:
            raise Exception("add_grid: "
                "grid_size isn't given in a correct format.")
    else:
        if isinstance( num_grid_pnts, list ):
            if len(num_grid_pnts) == dim:
                for i in xrange(dim):
                    if isinstance( num_grid_pnts[i], int ):
                        grid_size=(
                                float(domain_bb[1][i]) -float(domain_bb[0][i])
                            ) /num_grid_pnts[i]
                        size_list.append(grid_size)
                    else:
                        raise Exception("add_grid: "
                            "num_grid_pnts isn't given in a correct format.")
            else:
                raise Exception("add_grid: "
                    "num_grid_pnts isn't given in a correct format.")
        elif isinstance( num_grid_pnts, int ):
            for i in xrange(dim):
                grid_size=(
                        float(domain_bb[1][i])-float(domain_bb[0][i])
                    ) /num_grid_pnts
                size_list.append(grid_size)
        else:
            raise Exception("add_grid: "
                "num_grid_pnts isn't given in a correct format.")
    
    j=0
    list_grid=dict()
    
    while j<dim:
        list_grid[j] = compute_interval(
            float(domain_bb[0][j]),
            float(domain_bb[1][j]),
            size_list[j],
            abs_tol
        )
        if j>0:
            if j==1:
                re_list=list_grid[j-1]
                re_list=product_interval(re_list, list_grid[j])
            else:
                re_list=product_interval(re_list, list_grid[j])
        j+=1
        
    new_list = []
    parent = []
    for i in xrange(len(re_list)):
        temp_list=list()
        j=0
        while j<dim*2:
            temp_list.append([re_list[i][j],re_list[i][j+1]])
            j=j+2
        for j in xrange(len(ppp.regions)):
            tmp = pc.box2poly(temp_list)
            isect = tmp.intersect(ppp.regions[j], abs_tol)
            
            #if pc.is_fulldim(isect):
            rc, xc = pc.cheby_ball(isect)
            if rc > abs_tol/2:
                if rc < abs_tol:
                    print("Warning: "
                        "One of the regions in the refined PPP is too small"
                        ", this may cause numerical problems")
                if len(isect) == 0:
                    isect = pc.Region([isect], [])
                isect.props = ppp.regions[j].props.copy()
                new_list.append(isect)
                parent.append(j)   
    
    adj = sp.lil_matrix((len(new_list), len(new_list)), dtype=np.int8)
    for i in xrange(len(new_list)):
        adj[i,i] = 1
        for j in xrange(i+1, len(new_list)):
            if (ppp.adj[parent[i], parent[j]] == 1) or \
                    (parent[i] == parent[j]):
                if pc.is_adjacent(new_list[i], new_list[j]):
                    adj[i,j] = 1
                    adj[j,i] = 1
            
    return PropPreservingPartition(
        domain = ppp.domain,
        regions = new_list,
        adj = adj,
        prop_regions = ppp.prop_regions
    )

#### Helper functions ####
def compute_interval(low_domain, high_domain, size, abs_tol=1e-7):
    """Helper implementing intervals computation for each dimension.
    """
    list_g=list()
    i=low_domain
    while True:
        if (i+size+abs_tol)>=high_domain:
            list_g.append([i,high_domain])
            break
        else:
            list_g.append([i,i+size])
        i=i+size
    return list_g

def product_interval(list1, list2):
    """Helper implementing combination of all intervals for any two interval lists.
    """
    new_list=list()
    for m in xrange(len(list1)):
        for n in range(len(list2)):
            new_list.append(list1[m]+list2[n])
    return new_list

def find_adjacent_regions(partition):
    """Return region pairs that are spatially adjacent.
    """
    num_reg = len(partition.regions)
    
    adj = sp.lil_matrix(
        (num_reg, num_reg),
        dtype=np.int8
    )
    
    for i in xrange(num_reg):
        adj[i,i] = 1
        for j in xrange(i+1, num_reg):
            adj[i, j] = pc.is_adjacent(
                partition.regions[i],
                partition.regions[j]
            )
            adj[j,i] = adj[i,j]
        
    return adj
################################

class PropPreservingPartition(object):
    """Partition class with following fields:
    
    - domain: the domain we want to partition
        type: Polytope
    
    - regions: Regions of proposition-preserving partition
        type: list of Region
    
    - adj: a sparse matrix showing which regions are adjacent
        order of Regions same as in list C{regions}
        
        type: scipy lil sparse
    
    - prop_regions: map from atomic proposition symbols
        to continuous subsets
        
        type: dict of Polytope or Region
    
    - subsystems: list of indices
        Each partition corresponds to some mode.
        (for switched systems)
        
        In each mode a PwaSubSys is active.
        This PwaSubSys comprises of subsystems,
        which are listed in PwaSubSys.list_subsys.
        
        The list C{subsystems} means:
        
            - i-th Region in C{regions}
            - subsystems[i]-th system in PwaSubSys.list_subsys
                is active in the i-th Region
        
        type: list
    
    see also
    ========
    prop2part
    """
    # TODO: proposition preservation check
    
    def __init__(self,
        domain=None, regions=[],
        adj=None, prop_regions=None, subsystems=None,
        check=True
    ):
        if prop_regions is None:
            self.cont_props = None
        else:
            self.prop_regions = copy.deepcopy(prop_regions)
        
        n = len(regions)
        
        if hasattr(adj, 'shape'):
            (m, k) = adj.shape
            if m != k:
                raise ValueError('adj must be square')
            if m != n:
                msg = "adj size doesn't agree with number of regions"
                raise ValueError(msg)
        
        if check:
            atomic_propositions = set(self.prop_regions)
            
            for region in regions:
                if not region <= domain:
                    msg = 'Partition: Region:\n\n' + str(region) + '\n'
                    msg += 'is not subset of given domain:\n\t'
                    msg += str(domain)
                    raise ValueError(msg)
                
                if self.prop_regions is None:
                    warn('No continuous propositions defined.')
                    continue
                
                if not region.props <= atomic_propositions:
                    msg = 'Partitions: Region labeled with propositions:\n\t'
                    msg += str(region.props) + '\n'
                    msg += 'not all of which are in the '
                    msg += 'continuous atomic propositions:\n\t'
                    msg += str(atomic_propositions)
                    raise ValueError(msg)
        
        self.domain = domain
        self.regions = regions[:]
        self.adj = adj
        self.subsystems = subsystems
        
    def reg2props(self, region_index):
        return self.regions[region_index].props.copy()

    def __str__(self):
        s = '\n' + hl + '\n'
        s += 'Proposition Preserving Partition:\n'
        s += hl + 2*'\n'
        
        s += 'Domain: ' + str(self.domain) + '\n'
        
        for j, region in enumerate(self.regions):
            s += 'Region: ' + str(j) +'\n'
            
            if self.prop_regions is not None:
                s += '\t Propositions: '
                
                active_props = ' '.join(region.props)
                
                if active_props:
                    s += active_props + '\n'
                else:
                    s += '{}\n'
            
            s += str(region)
        
        if hasattr(self.adj, 'todense'):
            s += 'Adjacency matrix:\n'
            s += str(self.adj.todense()) + '\n'
        return s
    
    def plot(self, **kwargs):
        """For details see plot.plot_partition.
        """
        plot_partition(self, **kwargs)

class PPP(PropPreservingPartition):
    """Alias to PropPreservingPartition.
    
    See that for details.
    """
    def __init__(self, **args):
        PropPreservingPartition.__init__(self, **args)

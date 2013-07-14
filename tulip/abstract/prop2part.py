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


import numpy as np
from scipy import sparse as sp
import copy

from tulip import polytope as pc

def prop2part(state_space, cont_props_dict):
    """Main function that takes a domain (state_space) and a list of
    propositions (cont_props), and returns a proposition preserving
    partition of the state space.
    """
    cont_props = cont_props_dict.values()
    num_props = len(cont_props)
    list_regions = []
    first_poly = [] #Initial Region's list_poly atribute 
    first_poly.append(state_space)
    list_regions.append(pc.Region(list_poly=first_poly))
    mypartition = PropPreservingPartition(domain=copy.deepcopy(state_space),
                                          num_prop=num_props,
                                          list_region=list_regions,
                                          list_prop_symbol=copy.deepcopy(cont_props_dict.keys()))
    for prop_count in range(num_props):
        num_reg = mypartition.num_regions
        prop_holds_reg = []
        for i in range(num_reg): #i region counter
            region_now = mypartition.list_region[i].copy()
            #loop for prop holds
            prop_holds_reg.append(0)
            list_prop_now = mypartition.list_region[i].list_prop[:]
            dummy = pc.intersect(region_now, cont_props[prop_count])
            if pc.is_fulldim(dummy):
                dum_list_prop = list_prop_now[:]
                dum_list_prop.append(1)
                if len(dummy) == 0:
                    mypartition.list_region[i] = pc.Region([dummy],dum_list_prop)
                else:
                    dummy.list_prop = dum_list_prop
                    mypartition.list_region[i] = dummy.copy()
                prop_holds_reg[-1] = 1
            else: #does not hold in the whole region (-> no need for the 2nd loop)
                region_now.list_prop.append(0)
                mypartition.list_region.append(region_now)
                continue
                
            #loop for prop does not hold
            mypartition.list_region.append(pc.Region(list_poly=[],
                                                     list_prop=list_prop_now))
            dummy = pc.mldivide(region_now, cont_props[prop_count])
            if pc.is_fulldim(dummy):
                dum_list_prop = list_prop_now[:]
                dum_list_prop.append(0)
                if len(dummy) == 0:
                    mypartition.list_region[-1] = pc.Region([pc.reduce(dummy)],dum_list_prop)
                else:
                    dummy.list_prop = dum_list_prop
                    mypartition.list_region[-1] = dummy.copy()
            else:
                mypartition.list_region.pop()
        count = 0
        for hold_count in range(len(prop_holds_reg)):
            if prop_holds_reg[hold_count]==0:
                mypartition.list_region.pop(hold_count-count)
                count+=1
        num_reg = len(mypartition.list_region)
        mypartition.num_regions = num_reg
    adj = sp.lil_matrix((num_reg,num_reg), dtype=np.int8)
    for i in range(num_reg):
        for j in range(i+1,num_reg):
            dum = pc.is_adjacent(mypartition.list_region[i], mypartition.list_region[j])
            adj[i,j] = dum
            adj[j,i] = dum
        adj[i,i] = 1
    mypartition.adj = adj.copy()
    return mypartition
    
def prop2partconvex(ppp):
    """This function takes a proposition preserving partition and generates another proposition preserving partition     
    such that each part in the new partition is a convex polytope"""
    myconvexpartition = PropPreservingPartition(domain=copy.deepcopy(ppp.domain),
                                          num_prop=ppp.num_prop,
                                          list_prop_symbol=copy.deepcopy(ppp.list_prop_symbol))
    for i in range(ppp.num_regions):
    	simplified_reg = pc.union(ppp.list_region[i],ppp.list_region[i],check_convex=True)
        for j in range(len(simplified_reg)):
            region_now = pc.Region([simplified_reg.list_poly[j]], ppp.list_region[i].list_prop)
            myconvexpartition.list_region.append(region_now) 
    num_reg = len(myconvexpartition.list_region)
    myconvexpartition.num_regions = num_reg
    adj = sp.lil_matrix((num_reg,num_reg), dtype=np.int8)
    for i in range(num_reg):
        for j in range(i+1,num_reg):
            dum = pc.is_adjacent(myconvexpartition.list_region[i], myconvexpartition.list_region[j])
            adj[i,j] = dum
            adj[j,i] = dum
        adj[i,i] = 1 
    myconvexpartition.adj = adj.copy()
    return myconvexpartition
    
def pwa_partition(pwa_sys, ppp, abs_tol=1e-5):
    """ This function takes a piecewise affine system pwa_sys and a proposition 
    preserving partition ppp whose domain is a subset of the domain of pwa_sys
    and returns a refined proposition preserving partition where in each
    region a unique subsystem of pwa_sys is active.
    
    Modified from Petter Nilsson's code implementing merge algorithm in 
    Nilsson et al. `Temporal Logic Control of Switched Affine Systems with an
    Application in Fuel Balancing`, ACC 2012.
    
    Input:
    
    - `pwa_sys`: a PwaSysDyn object
    - `ppp`: a PropPreservingPartition object
    
    Output:
    
    - A PropPreservingPartition object with subsystem assignments
    
    """
    if pc.is_fulldim(pc.mldivide(ppp.domain, pwa_sys.domain)):
        raise Exception("pwaPartition: pwa system is not defined everywhere in the state space")

    new_list = []
    subsys_list = []
    parent = []
    for i in range(len(pwa_sys.list_subsys)):
        for j in range(ppp.num_regions):
            isect = pc.intersect(pwa_sys.list_subsys[i].sub_domain, ppp.list_region[j])
            if pc.is_fulldim(isect):
                rc, xc = pc.cheby_ball(isect)
                if rc < abs_tol:
                    print "Warning: One of the regions in the refined PPP is too small, this may cause numerical problems"
                if len(isect) == 0:
                    isect = pc.Region([isect], [])
                isect.list_prop = ppp.list_region[j].list_prop
                subsys_list.append(i)
                new_list.append(isect)
                parent.append(j)
    
    adj = sp.lil_matrix((num_reg,num_reg), dtype=np.int8)
    for i in range(len(new_list)):
        for j in range(i+1, len(new_list)):
            if (ppp.adj[parent[i], parent[j]] == 1) or \
                    (parent[i] == parent[j]):
                if pc.is_adjacent(new_list[i], new_list[j]):
                    adj[i,j] = 1
                    adj[j,i] = 1
        adj[i,i] = 1
            
    return PropPreservingPartition(domain=ppp.domain,\
                    num_prop=ppp.num_prop, list_region=new_list, num_regions=len(new_list), \
                    adj=adj, list_prop_symbol=ppp.list_prop_symbol, list_subsys = subsys_list)        
        

class PropPreservingPartition:
    """Partition class with following fields
    
    - domain: the domain we want to partition, type: polytope
    - num_prop: number of propositions
    - list_region: proposition preserving regions, type: list of Region
    - num_regions: length of the above list
    - adj: a sparse matrix showing which regions are adjacent, type scipy lil sparse
    - list_prop_symbol: list of symbols of propositions
    - list_subsys: list assigning the subsystem of the piecewise affine system that 
              is active in that region to each region in ppp 

    """
    def __init__(self, domain=None, num_prop=0, list_region=[], num_regions=0, adj=0, list_prop_symbol=None, list_subsys=None):
        self.domain = domain
        self.num_prop = num_prop
        self.list_region = list_region[:]
        self.num_regions = len(list_region)
        self.adj = adj
        self.list_prop_symbol = list_prop_symbol
        self.list_subsys = list_subsys
        
    def reg2props(self, region):
        return [self.list_prop_symbol[n] for (n,p) in enumerate(
                self.list_region[region].list_prop) if p]

    def __str__(self):
        output = "Domain: "+str(self.domain)+"\n"
        if self.list_prop_symbol is not None:
            for j in range(len(self.list_region)):
                output += "Region "+str(j)+", propositions: "+" ".join([self.list_prop_symbol[i] for i in range(len(self.list_prop_symbol)) if self.list_region[j].list_prop[i] != 0])+"\n"
                output += str(self.list_region[j])
        if hasattr(self.adj, "shape"):
            output +="Adjacency matrix:\n"+str(self.adj.todense())+"\n"
        return output

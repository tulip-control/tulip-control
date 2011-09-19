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
# 
# $Id$
""" 
Proposition preserving partition module.
"""


import numpy as np
from time import time
import copy

import polytope as pc

def prop2part2(state_space, cont_props_dict):
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
        prop_holds_poly = []
        prop_not_holds_poly = []
        for i in range(num_reg): #i region counter
            region_now = mypartition.list_region[i].list_poly[:]
            #loop for prop holds
            prop_holds_reg.append(0)
            prop_holds_poly[:] = []
            list_prop_now = mypartition.list_region[i].list_prop[:]
            for j in range(len(region_now)): #j polytope counter
                prop_holds_poly.append(0)
                dummy = pc.Polytope(np.concatenate((region_now[j].A, cont_props[prop_count].A)),
                                    np.concatenate((region_now[j].b, cont_props[prop_count].b)))
                if pc.is_fulldim(dummy):
                    #dummy = pc.reduce(dummy)
                    mypartition.list_region[i].list_poly[j] = pc.reduce(dummy)
                    prop_holds_reg[-1] = 1
                    prop_holds_poly[-1] = 1
            count = 0
            for hold_count in range(len(prop_holds_poly)):
                if prop_holds_poly[hold_count]==0:
                    mypartition.list_region[i].list_poly.pop(hold_count-count)
                    count+=1
            if len(mypartition.list_region[i].list_poly)>0:
                mypartition.list_region[i].list_prop.append(1)
            #loop for prop does not hold
            mypartition.list_region.append(pc.Region(list_poly=[],
                                                     list_prop=list_prop_now))
            for j in range(len(region_now)):
                valid_props = cont_props[prop_count] #eliminateRedundantProps(region_now[j],cont_props[prop_count])
                A_prop = valid_props.A.copy()
                b_prop = valid_props.b.copy()
                num_halfspace = A_prop.shape[0]
                for k in range(pow(2,num_halfspace)-1):
                    signs = pc.num_bin(k,places=num_halfspace)
                    A_now = A_prop.copy()
                    b_now = b_prop.copy()
                    for l in range(len(signs)): 
                        if signs[l]==0:
                            A_now[l] = -A_now[l]
                            b_now[l] = -b_now[l]
                    dummy = pc.Polytope(np.concatenate((region_now[j].A, A_now)),
                                        np.concatenate((region_now[j].b, b_now)))
                    if pc.is_fulldim(dummy):
                        #dummy = pc.reduce(dummy)
                        mypartition.list_region[-1].list_poly.append(pc.reduce(dummy))
            if len(mypartition.list_region[-1].list_poly)>0:
                mypartition.list_region[-1].list_prop.append(0)
            else:
                mypartition.list_region.pop()
        count = 0
        for hold_count in range(len(prop_holds_reg)):
            if prop_holds_reg[hold_count]==0:
                mypartition.list_region.pop(hold_count-count)
                count+=1
        num_reg = len(mypartition.list_region)
        mypartition.num_regions = num_reg
    adj = np.zeros((num_reg,num_reg), dtype=np.int8)
    for i in range(num_reg):
        for j in range(i+1,num_reg):
            adj[i,j] = pc.is_adjacent(mypartition.list_region[i],
                                           mypartition.list_region[j])
    adj =  adj+adj.T+np.eye(num_reg, dtype=np.int8)
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
    adj = np.zeros((num_reg,num_reg), dtype=np.int8)
    for i in range(num_reg):
        for j in range(i+1,num_reg):
            adj[i,j] = pc.is_adjacent(myconvexpartition.list_region[i],
                                           myconvexpartition.list_region[j])
    adj =  adj+adj.T+np.eye(num_reg, dtype=np.int8)
    myconvexpartition.adj = adj.copy()
    return myconvexpartition

class PropPreservingPartition:
    """Partition class with following fields
    
    - domain: the domain we want to partition, type: polytope
    - num_prop: number of propositions
    - list_region: proposition preserving regions, type: list of Region
    - num_regions: length of the above list
    - adj: a matrix showing which regions are adjacent
    - trans: a matrix showing which region is reachable from which region.
             trans[i,j] = 1 means state i is reachable from state j.
    - list_prop_symbol: list of symbols of propositions
    - orig_list_region: original proposition preserving regions
    - orig: list assigning an original proposition preserving region to each
            new region

    """
    
    def __init__(self, domain=None, num_prop=0, list_region=[], num_regions=0, adj=0, trans=0, list_prop_symbol=None, orig_list_region=None, orig=None):
        self.domain = domain
        self.num_prop = num_prop
        self.list_region = list_region[:]
        self.num_regions = len(list_region)
        self.adj = adj
        self.trans = trans
        self.list_prop_symbol = list_prop_symbol
        self.orig_list_region = orig_list_region
        self.orig = orig
        
#def test():
if __name__ == "__main__":
    start_time = time()
    domain = np.array([[0., 2.],[0., 2.]])

    domain_poly_A = np.array(np.vstack([np.eye(2),-np.eye(2)]))
    domain_poly_b = np.array([np.r_[domain[:,1],-domain[:,0]]]).T
    state_space = pc.Polytope(domain_poly_A, domain_poly_b)

    cont_props = []
    
    A0 = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    b0 = np.array([[1., 0., 1., 0.]]).T
    cont_props.append(pc.Polytope(A0, b0))
    
    A1 = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    b1 = np.array([[1., 0., 2., -1.]]).T
    cont_props.append(pc.Polytope(A1, b1))
    
    
    A2 = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    b2 = np.array([[2., -1., 1., 0.]]).T
    cont_props.append(pc.Polytope(A2, b2))

    A3 = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    b3 = np.array([[2., -1., 2., -1.]]).T
    cont_props.append(pc.Polytope(A3, b3))
    
    cont_props_dict = dict({'C0':pc.Polytope(A0, b0),'C1':pc.Polytope(A1, b1),'C2':pc.Polytope(A2, b2),'C3':pc.Polytope(A3, b3) })
    
    mypartition = prop2part2(state_space, cont_props_dict)
    
    #print len(mypartition.list_region)
    A4 = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    b4 = np.array([[0.5, 0., 0.5, 0.]]).T
    poly1 = pc.Polytope(A4,b4)

    r1 = pc.mldivide(mypartition.list_region[3],poly1)
    
    verbose = 0
    
    xx = mypartition.adj
    print xx
    time_elapsed = time()-start_time
    print time_elapsed
    if verbose:
        print 'time', time_elapsed
        #print sys.getrefcount(np.dtype('float64'))
        print '\nAdjacency matrix:\n', mypartition.adj
    
        for i in range(mypartition.num_regions):
            print i+1,"th region has the following generators"
            for j in range(len(mypartition.list_region[i].list_poly)):
                print 'polytope',j,'in region',i+1
                #print mypartition.list_region[i].list_poly[j].generators
            print 'region',i+1,'proposition list is',mypartition.list_region[i].list_prop,'\n'


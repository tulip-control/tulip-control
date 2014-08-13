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
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLxUDING, BUT NOT
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
"""
import logging
logger = logging.getLogger(__name__)

import warnings
import copy

import numpy as np
from scipy import sparse as sp
import polytope as pc

from .plot import plot_partition

try:
    import matplotlib as mpl
except Exception, e:
    logger.error(e)
    mpl = None

_hl = 40 * '-'

def prop2part(state_space, cont_props_dict):
    """Main function that takes a domain (state_space) and a list of
    propositions (cont_props), and returns a proposition preserving
    partition of the state space.

    See Also
    ========
    L{PropPreservingPartition},
    C{polytope.Polytope}
    
    @param state_space: problem domain
    @type state_space: C{polytope.Polytope}
    
    @param cont_props_dict: propositions
    @type cont_props_dict: dict of C{polytope.Polytope}
    
    @return: state space quotient partition induced by propositions
    @rtype: L{PropPreservingPartition}
    """
    first_poly = [] #Initial Region's polytopes
    first_poly.append(state_space)
    
    regions = [pc.Region(first_poly)]
    
    for cur_prop in cont_props_dict:
        cur_prop_poly = cont_props_dict[cur_prop]
        
        num_reg = len(regions)
        prop_holds_reg = []
        
        for i in xrange(num_reg): #i region counter
            region_now = regions[i].copy()
            #loop for prop holds
            prop_holds_reg.append(0)
            
            prop_now = regions[i].props.copy()
            
            dummy = region_now.intersect(cur_prop_poly)
            
            # does cur_prop hold in dummy ?
            if pc.is_fulldim(dummy):
                dum_prop = prop_now.copy()
                dum_prop.add(cur_prop)
                
                # is dummy a Polytope ?
                if len(dummy) == 0:
                    regions[i] = pc.Region([dummy], dum_prop)
                else:
                    # dummy is a Region
                    dummy.props = dum_prop.copy()
                    regions[i] = dummy.copy()
                prop_holds_reg[-1] = 1
            else:
                #does not hold in the whole region
                # (-> no need for the 2nd loop)
                regions.append(region_now)
                continue
                
            #loop for prop does not hold
            regions.append(pc.Region([], props=prop_now) )
            dummy = region_now.diff(cur_prop_poly)
            
            if pc.is_fulldim(dummy):
                dum_prop = prop_now.copy()
                
                # is dummy a Polytope ?
                if len(dummy) == 0:
                    regions[-1] = pc.Region([pc.reduce(dummy)], dum_prop)
                else:
                    # dummy is a Region
                    dummy.props = dum_prop.copy()
                    regions[-1] = dummy.copy()
            else:
                regions.pop()
        
        count = 0
        for hold_count in xrange(len(prop_holds_reg)):
            if prop_holds_reg[hold_count]==0:
                regions.pop(hold_count-count)
                count+=1
    
    mypartition = PropPreservingPartition(
        domain = copy.deepcopy(state_space),
        regions = regions,
        prop_regions = copy.deepcopy(cont_props_dict)
    )
    
    mypartition.adj = pc.find_adjacent_regions(mypartition).copy()
    
    return mypartition

def part2convex(ppp):
    """This function takes a proposition preserving partition and generates 
    another proposition preserving partition such that each part in the new 
    partition is a convex polytope
    
    @type ppp: L{PropPreservingPartition}
    
    @return: refinement into convex polytopes and
        map from new to old Regions
    @rtype: (L{PropPreservingPartition}, list)
    """
    cvxpart = PropPreservingPartition(
        domain=copy.deepcopy(ppp.domain),
        prop_regions=copy.deepcopy(ppp.prop_regions)
    )
    new2old = []
    for i in xrange(len(ppp.regions)):
        simplified_reg = pc.union(ppp.regions[i],
                                  ppp.regions[i],
                                  check_convex=True)
        
        for j in xrange(len(simplified_reg)):
            region_now = pc.Region(
                [simplified_reg[j]],
                ppp.regions[i].props
            )
            cvxpart.regions.append(region_now)
            new2old += [i]
    
    cvxpart.adj = pc.find_adjacent_regions(cvxpart).copy()
    
    return (cvxpart, new2old)
    
def pwa_partition(pwa_sys, ppp, abs_tol=1e-5):
    """This function takes:
    
      - a piecewise affine system C{pwa_sys} and
      - a proposition-preserving partition C{ppp}
          whose domain is a subset of the domain of C{pwa_sys}
    
    and returns a *refined* proposition preserving partition
    where in each region a unique subsystem of pwa_sys is active.
    
    Reference
    =========
    Modified from Petter Nilsson's code
    implementing merge algorithm in:
    
    Nilsson et al.
    `Temporal Logic Control of Switched Affine Systems with an
    Application in Fuel Balancing`, ACC 2012.

    See Also
    ========
    L{discretize}
    
    @type pwa_sys: L{hybrid.PwaSysDyn}
    @type ppp: L{PropPreservingPartition}
    
    @return: new partition and associated maps:
        
        - new partition C{new_ppp}
        - map of C{new_ppp.regions} to C{pwa_sys.list_subsys}
        - map of C{new_ppp.regions} to C{ppp.regions}
    
    @rtype: C{(L{PropPreservingPartition}, list, list)}
    """
    if pc.is_fulldim(ppp.domain.diff(pwa_sys.domain) ):
        raise Exception('pwa system is not defined everywhere ' +
                        'in state space')
    
    # for each subsystem's domain, cut it into pieces
    # each piece is the intersection with
    # a unique Region in ppp.regions
    new_list = []
    subsys_list = []
    parents = []
    for i, subsys in enumerate(pwa_sys.list_subsys):
        for j, region in enumerate(ppp.regions):
            isect = region.intersect(subsys.domain)
            
            if pc.is_fulldim(isect):
                rc, xc = pc.cheby_ball(isect)
                
                if rc < abs_tol:
                    msg = 'One of the regions in the refined PPP is '
                    msg += 'too small, this may cause numerical problems'
                    warnings.warn(msg)
                
                # not Region yet, but Polytope ?
                if len(isect) == 0:
                    isect = pc.Region([isect])
                
                # label with AP
                isect.props = region.props.copy()
                
                # store new Region
                new_list.append(isect)
                
                # keep track of original Region in ppp.regions
                parents.append(j)
                
                # index of subsystem active within isect
                subsys_list.append(i)
    
    # compute spatial adjacency matrix
    n = len(new_list)
    adj = sp.lil_matrix((n, n), dtype=np.int8)
    for i, ri in enumerate(new_list):
        pi = parents[i]
        for j, rj in enumerate(new_list[0:i]):
            pj = parents[j]
            
            if (ppp.adj[pi, pj] == 1) or (pi == pj):
                if pc.is_adjacent(ri, rj):
                    adj[i, j] = 1
                    adj[j, i] = 1
        adj[i, i] = 1
            
    new_ppp = PropPreservingPartition(
        domain = ppp.domain,
        regions = new_list,
        adj = adj,
        prop_regions = ppp.prop_regions
    )
    return (new_ppp, subsys_list, parents)
                
def add_grid(ppp, grid_size=None, num_grid_pnts=None, abs_tol=1e-10):
    """ This function takes a proposition preserving partition ppp and the size 
    of the grid or the number of grids, and returns a refined proposition 
    preserving partition with grids.
     
    Input:
    
      - `ppp`: a L{PropPreservingPartition} object
      - `grid_size`: the size of the grid,
          type: float or list of float
      - `num_grid_pnts`: the number of grids for each dimension,
          type: integer or list of integer
    
    Output:
    
      - A L{PropPreservingPartition} object with grids
        
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
    domain_bb = ppp.domain.bounding_box
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

def post_area(ppp, sys_dyn, N=1, abs_tol=1e-7):

    list_post_area={}
    list_extp_d=pc.extreme(sys_dyn.Wset)

    if list_extp_d==None:
        for i in range(0,len(ppp.regions)):
            list_post_area[i]=[]
            p_current=ppp.regions[i]
            for m in range(len(ppp.regions[i].list_poly)):
                extp=pc.extreme(p_current.list_poly[m])
                j=1
                post_extp_N=extp
                while j <=N:
                     post_extp_N=np.dot(post_extp_N,sys_dyn.A.T)+sys_dyn.K.T
                     #post_extp_N=np.dot(post_extp_N,sys_dyn.A.T)+np.vstack([sys_dyn.K.T,sys_dyn.K.T,sys_dyn.K.T,sys_dyn.K.T])
                     j+=1
                post_area_hull=pc.qhull(post_extp_N)
                list_post_area[i].append(post_area_hull)
           
    else:
        for i in range(0,len(ppp.regions)):
            list_post_area[i]=[]
            list_post_extp_d =[]
            p_current=ppp.regions[i]
            extp=pc.extreme(p_current.list_poly[0])
            #post_extp_n=np.zeros([len(list_extp_d),len(list_extp_d)])
            #post_extp_n=[]
            for m in range(0, len(list_extp_d)):
                post_extp_N=extp
                j=1
                while j<= N:
                     post_extp_N=np.dot(post_extp_N,sys_dyn.A.T)+sys_dyn.K.T+np.dot(list_extp_d[m], sys_dyn.E.T)
                     j+=1
                list_post_extp_d.append(post_extp_N)
                if m==0:
                    post_extp_n = list_post_extp_d[m]
                else:
                    post_extp_n=np.vstack([post_extp_n, list_post_extp_d[m]])
                #post_extp_n=post_extp_N.copy()
                #post_extp_n=np.vstack([post_extp_n, list_post_extp_d[m]])
            #post_extp_n=np.vstack([list_post_extp_d[0],list_post_extp_d[1],list_post_extp_d[2],list_post_extp_d[3]])
            post_area_hull=pc.qhull(post_extp_n)
            list_post_area[i].append(post_area_hull)
    
    return list_post_area

def get_transitions(ppp, post_area):

    transitions = np.zeros([len(ppp.regions),(len(ppp.regions)+1)], dtype = int)

    list_intersect_region=[]
    for i in range(0,len(ppp.regions)):
        for j in range(0,len(ppp.regions)):
            for m in range(len(ppp.regions[i].list_poly)):
                inters_region=pc.intersect(ppp.regions[j],post_area[i][m])
                if pc.is_empty(inters_region)== False:
                    trans=1
                    break
                else:
                    trans=0
            if trans==1:
                transitions[i,j]=1
            else:
                transitions[i,j]=0
                
    for j in range(0,len(ppp.regions)):
        for m in range(len(ppp.regions[j].list_poly)):
            inters_region=pc.mldivide(post_area[j][m],ppp.domain)
            if pc.is_empty(inters_region)== False:
                trans=1
                break
            else:
                trans=0
        if trans==1:
            transitions[j,len(ppp.regions)]=1
        else:
            transitions[j,len(ppp.regions)]=0
    return transitions

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

def find_equilibria(ssd,cont_props,outside_props,eq_props,eps=0.1):
    """ Finds the polytope that contains the equilibrium points

    @param ssd: The dynamics of the switched system
    @type ssd: L{SwitchedSysDyn}

    @param cont_props: The polytope representations of the atomic 
    propositions of the state space to be used in partitiong 
    @type cont_props: dict of polytope.Polytope

    @param outside_props: The set of names of atomic propositions 
    that are beyond the domain (i.e. 'OUTSIDE')
    @type outside_props: set()

    @param eq_props: The set of names of atomic propositions 
    where equilibrium points for a mode can be found
    @type eq_props: set()

    @param eps: The value by which the width of all polytopes
    containing equilibrium points is increased.
    @type eps: float

    Warning: Currently, there is no condition for points where 
    the polytope is critically stable. 

    Warning: if there is a region outside the domain, then it is
    unstable. It seems to ignore the regions outside the domain.
    """
    
    def normalize(A,B):
        """ Normalizes set of equations of the form Ax<=B
        """
        if A.size > 0:
            Anorm = np.sqrt(np.sum(A*A,1)).flatten()     
            pos = np.nonzero(Anorm > 1e-10)[0]
            A = A[pos, :]
            B = B[pos]
            Anorm = Anorm[pos]           
            mult = 1/Anorm
            for i in xrange(A.shape[0]):
                A[i,:] = A[i,:]*mult[i]
            B = B.flatten()*mult
        return A,B

    cont_ss=ssd.cts_ss
    for mode in ssd.modes:
        cont_dyn=ssd.dynamics[mode].list_subsys[0]
        A=cont_dyn.A
        K=cont_dyn.K.T[0]
        #cont_ss.b=np.array([cont_ss.b]).T
        I=np.eye(len(A),dtype=float)
        rank_IA=np.linalg.matrix_rank(I-A)
        concat=np.hstack((I-A,K.reshape(len(A),1)))
        rank_concat=np.linalg.matrix_rank(concat)
        soln=pc.Polytope()
        props_sym='eqpnt_'+str(mode[1])
        eq_props|={props_sym}

        if (rank_IA==rank_concat):
            if (rank_IA==len(A)):
                equil=np.dot(np.linalg.inv(I-A),K)
                print "Equilibrium Points: "+str(mode)
                print equil
                print "---------------------------------"
                if (equil[0]>=(-cont_ss.b[2]) and equil[0]<=cont_ss.b[0] 
                        and equil[1]>=(-cont_ss.b[3]) 
                        and equil[1]<=cont_ss.b[1]):
                    delta=equil/100
                    soln=box2poly([[equil[0]-delta[0], equil[0]+delta[0]],
                        [equil[1]-delta[1], equil[1]+delta[1]]]) 
                else:
                    soln=box2poly([[24.,25.],[24.,25.]])
                    outside_props|={props_sym}
            elif (rank_IA<len(A)):
                #eps=abs(min(np.amin(K),np.amin(I-A)))
                #eps=0.0005
                eps=0.2
                if eps==0:
                    eps=abs(min(np.amin(-K),np.amin(A-I)))
                IAn,Kn = normalize(I-A,K)
                soln=pc.Polytope(np.vstack((IAn,-IAn)), 
                        np.hstack((Kn+eps,-Kn+eps)))

                print "First soln: "+str(mode)
                print soln
                print "---------------------------------"
                relevantsoln=pc.intersect(soln,cont_ss,abs_tol)
                if pc.is_empty(relevantsoln):
                    print "Intersect "+str(mode)+" is empty"
                else:
                    print "Intersect "+str(mode)+" is not empty - good job!!"
                print relevantsoln
                print "---------------------------------"

                if(pc.is_empty(relevantsoln) & ~pc.is_empty(soln)):
                    soln=box2poly([[24.,25.],[24.,25.]])
                    outside_props|={props_sym}
                else:
                    soln=relevantsoln
        
        else:
            #Assuming trajectories go to infinity as there are no 
            #equilibrium points
            soln=box2poly([[24.,25.],[24.,25.]])
            outside_props|={props_sym}
            print str(mode)+" trajectories go to infinity! No solution"

        print "Normalized soln: "+str(mode)
        print soln
        print "---------------------------------"
        cont_props[props_sym]=soln

################################

class PropPreservingPartition(pc.MetricPartition):
    """Partition class with following fields:
    
      - domain: the domain we want to partition
          type: C{Polytope}

      - regions: Regions of proposition-preserving partition
          type: list of C{Region}

      - adj: a sparse matrix showing which regions are adjacent
          order of C{Region}s same as in list C{regions}

          type: scipy lil sparse

      - prop_regions: map from atomic proposition symbols
          to continuous subsets

          type: dict of C{Polytope} or C{Region}
    
    See Also
    ========
    L{prop2part}
    """
    def __init__(self,
        domain=None, regions=[],
        adj=None, prop_regions=None, check=True
    ):
        #super(PropPreservingPartition, self).__init__(adj)
        
        if prop_regions is None:
            self.prop_regions = None
        else:
            try:
                # don't call it
                # use try because it should work
                # vs hasattr, which would look like normal selection
                prop_regions.keys
            except:
                msg = 'prop_regions must be dict.'
                msg += 'Got instead: ' + str(type(prop_regions))
                raise TypeError(msg)
            
            self.prop_regions = copy.deepcopy(prop_regions)
        
        n = len(regions)
        
        if hasattr(adj, 'shape'):
            (m, k) = adj.shape
            if m != k:
                raise ValueError('adj must be square')
            if m != n:
                msg = "adj size doesn't agree with number of regions"
                raise ValueError(msg)
        
        self.regions = regions[:]
        
        if check:
            for region in regions:
                if not region <= domain:
                    msg = 'Partition: Region:\n\n' + str(region) + '\n'
                    msg += 'is not subset of given domain:\n\t'
                    msg += str(domain)
                    raise ValueError(msg)
            
            self.is_symbolic()
        
        self.domain = domain
        super(PropPreservingPartition, self).__init__(domain)
        self.adj = adj
    
    def reg2props(self, region_index):
        return self.regions[region_index].props.copy()
    
    #TODO: iterator over pairs
    #TODO: use nx graph to store partition
    
    def is_symbolic(self):
        """Check that the set of preserved predicates
        are bijectively mapped to the symbols.
        
        Symbols = Atomic Propositions
        """
        if self.prop_regions is None:
            msg = 'No continuous propositions defined.'
            logging.warn(msg)
            warnings.warn(msg)
            return
        
        for region in self.regions:
            if not region.props <= set(self.prop_regions):
                msg = 'Partitions: Region labeled with propositions:\n\t'
                msg += str(region.props) + '\n'
                msg += 'not all of which are in the '
                msg += 'continuous atomic propositions:\n\t'
                msg += str(set(self.prop_regions) )
                raise ValueError(msg)
    
    def preserves_predicates(self):
        """Return True if each Region <= Predicates for the
        predicates in C{prop_regions.values},
        where C{prop_regions} is a bijection to
        "continuous" propositions of the specification's alphabet.
        
        Note
        ====
        1. C{prop_regions} in practice need not be injective.
            It doesnt hurt - though creates unnecessary redundancy.
        
        2. The specification alphabet is fixed an user-defined.
            It should be distinguished from the auxiliary alphabet
            generated automatically during abstraction,
            which defines another partition with
            its own bijection to TS.
        """
        all_props = set(self.prop_regions)
        
        for region in self.regions:
            # Propositions True in Region
            for prop in region.props:
                preimage = self.prop_regions[prop]
                
                if not region <= preimage:
                    return False
            
            # Propositions False in Region
            for prop in all_props.difference(region.props):
                preimage = self.prop_regions[prop]
                
                if region.intersect(preimage).volume > pc.polytope.ABS_TOL:
                    return False
        return True

    def __str__(self):
        """Get informal string representation."""
        s = '\n' + _hl + '\n'
        s += 'Proposition Preserving Partition:\n'
        s += _hl + 2*'\n'
        
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
    
    def plot(
        self, trans=None, ppp2trans=None, only_adjacent=False,
        ax=None, plot_numbers=True, color_seed=None,
        show=False
    ):
        """For details see plot.plot_partition.
        """
        return plot_partition(
            self, trans, ppp2trans, only_adjacent,
            ax, plot_numbers, color_seed, show
        )
    
    def plot_props(self, ax=None):
        """Plot labeled regions of continuous propositions.
        """
        if mpl is None:
            warnings.warn('No matplotlib')
            return
        
        if ax is None:
            ax = mpl.pyplot.subplot(111)
        
        l, u = self.domain.bounding_box
        ax.set_xlim(l[0,0], u[0,0])
        ax.set_ylim(l[1,0], u[1,0])
        
        for (prop, poly) in self.prop_regions.iteritems():
            isect_poly = poly.intersect(self.domain)
            
            isect_poly.plot(ax, color='none', hatch='/')
            isect_poly.text(prop, ax, color='yellow')
        return ax

class PPP(PropPreservingPartition):
    """Alias to L{PropPreservingPartition}.
    
    See that for details.
    """
    def __init__(self, **args):
        PropPreservingPartition.__init__(self, **args)

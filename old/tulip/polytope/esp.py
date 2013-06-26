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
#
#  Reference:
#
#  Colin N. Jones, Eric C. Kerrigan and Jan M. Maciejowski,
#  Equality Set Projection: A new algorithm for the projection of polytopes 
#  in halfspace representation,
#  http://www-control.eng.cam.ac.uk/~cnj22/research/projection.html
#  2004
# 

'''Implementing the non-vertex polytope projection method
Equality Set Projection (ESP) from 
http://www-control.eng.cam.ac.uk/~cnj22/research/projection.html

Very unstable, can not handle complex polytopes.

Created by P. Nilsson, 8/2/11'''

import sys, os

import numpy as np
from scipy import linalg
from cvxopt import matrix, solvers

# Find a lp solver to use
try:
    import cvxopt.glpk
    lp_solver = 'glpk'
except:
    lp_solver = None
        
solvers.options['show_progress']=False
solvers.options['LPX_K_MSGLEV'] = 0

class Ridge:
    '''Contains the following information:
    `E_0`: Equality set of a facet
    `af,bf`: Affine hull of the facet s.t. P_{E_0} = P intersection {x | af x = bf}.
    '''
    def __init__(self, E, a, b):
        self.E_r= E
        self.ar = a
        self.br = b

class Ridge_Facet:
    '''Contains the following information:
    
    `E_r`: Equality set of a ridge
    `ar,br`: Affine hull of the ridge s.t. P_{E_f} intersection {x | ar x = br} defines the ridge,
           where E_f is the equality set of the facet.
    `E_0`: Equality set of a facet
    `af,bf`: Affine hull of the facet.
    '''
    def __init__(self,E_r,ar,br,E_0,af,bf):
        self.E_r = E_r
        self.ar = ar
        self.br = br
        self.E_0 = E_0
        self.af = af
        self.bf = bf

def esp(CC,DD,bb,centered=False,abs_tol=1e-10,verbose=0):
    '''
    Compute the projection of the polytope [C D] x <= b onto the coordinates corresponding
    to C. The projection of the polytope P = {[C D]x <= b where C is M x D and D is M x K is 
    defined as proj(P) = {x in R^d | exist y in R^k s.t Cx + Dy < b}
    '''
    # Remove zero columns and rows
    nonzerorows = np.nonzero( np.sum(np.abs(np.hstack([CC, DD])), axis = 1) > abs_tol)[0]
    nonzeroxcols = np.nonzero( np.sum(np.abs(CC), axis = 0) > abs_tol)[0]
    nonzeroycols = np.nonzero( np.sum(np.abs(DD), axis = 0) > abs_tol)[0]
    
    C = CC[nonzerorows, :].copy()
    D = DD[nonzerorows, :].copy()
    C = C[:, nonzeroxcols]
    D = D[:, nonzeroycols]
    b = bb[nonzerorows].copy()
    
    # Make sure origo is inside polytope
    if not centered:
        xc0,yc0,trans = cheby_center(C,D,b)
        if trans:
            b = b - np.dot(C,xc0).flatten() - np.dot(D,yc0).flatten()
        else:
            b = b
    else:
        trans = False

    d = C.shape[1]
    k = D.shape[1]
    
    if verbose > 0:
        print "Projecting from dim " + str(d+k) + " to " + str(d)
    
    if k == 0:
        # Not projecting
        return C,bb,[]
    
    if d == 1:
        #Projection to 1D      
        c = np.zeros(d+k)
        c[0] = 1
        G = np.hstack([C,D])
        sol = solvers.lp( matrix(c),matrix(G), matrix(b), None, None, lp_solver)            
        if sol['status'] != "optimal":
            raise Exception("esp: projection to 1D is not full-dimensional, LP returned status " + str(sol['status']))
        min_sol = np.array(sol['x']).flatten()
        min_dual_sol = np.array(sol['z']).flatten()
        sol = solvers.lp(-matrix(c),matrix(G), matrix(b), None, None, lp_solver)
        if sol['status'] != "optimal":
            raise Exception("esp: projection to 1D is not full-dimensional, LP returned status " + str(sol['status']))    
        max_sol = np.array(sol['x']).flatten()
        max_dual_sol = np.array(sol['z']).flatten()
        
        x_min = min_sol[0]
        x_max = max_sol[0]
        y_min = min_sol[range(1,k+1)]
        y_max = max_sol[range(1,k+1)]
        
        if is_dual_degenerate(c,G,b,None,None,min_sol,min_dual_sol):
            #Min case, relax constraint a little to avoid infeasibility
            E_min = unique_equalityset(C,D,b,np.array([1.]),x_min+abs_tol/3,abs_tol=abs_tol)
        else:
            E_min = np.nonzero(np.abs(np.dot(G,min_sol)-b) < abs_tol)[0]
        
        if is_dual_degenerate(c,G,b,None,None,max_sol,max_dual_sol):
            #Max case, relax constraint a little to avoid infeasibility
            E_max = unique_equalityset(C,D,b,np.array([1.]),x_max-abs_tol/3,abs_tol=abs_tol) 
        else:
            E_max = np.nonzero(np.abs(np.dot(G,max_sol)-b) < abs_tol)[0]
            
        G = np.array([[1.],[-1.]])
        g = np.array([x_max,-x_min])
        
        # Relocate
        if trans:
            g = g + np.dot(G,xc0)
        # Return zero cols/rows
        E_max = nonzerorows[E_max]
        E_min = nonzerorows[E_min]
        
        if verbose > 0:
           print "Returning projection from dim " + str(d+k) + " to dim 1 \n"
        return G,g,[E_max,E_min]
    
    E = []
    L = []
    
    E_0,af,bf = shoot(C,D,b,abs_tol=abs_tol)  
    ridge_list = ridge(C,D,b,E_0,af,bf,abs_tol=abs_tol,verbose=verbose)

    for i in range(len(ridge_list)):
        r = ridge_list[i]
        L.append(Ridge_Facet(r.E_r,r.ar,r.br,E_0,af,bf))
    
    G = af.T
    g = bf
    
    if verbose > 0:
        print "\nStarting eq set " + str(E_0) + "\nStarting ridges "
        for rr in L:
            print str(rr.E_r)

    E.append(E_0)  
    
    while len(L) > 0:
        rid_fac1 = L[0]
        if verbose > 0:
            print "\nLooking for neighbors to " + str(rid_fac1.E_0) + " and " + str(rid_fac1.E_r) + " .."
        E_adj,a_adj,b_adj = adjacent(C,D,b,rid_fac1,abs_tol=abs_tol)
        if verbose > 0:
            print "found neighbor " + str(E_adj ) + ". \n\nLooking for ridges of neighbor.."
        ridge_list = ridge(C,D,b,E_adj,a_adj,b_adj,abs_tol=abs_tol,verbose=verbose)
        if verbose > 0:
            print "found " + str(len(ridge_list)) + " ridges\n"
        
        found_org = False
        for i in range(len(ridge_list)):
            r = ridge_list[i]
            E_r = r.E_r
            ar = r.ar
            br = r.br
            found = False
            for j in range(len(L)):
                rid_fac2 = L[j]
                A_r = rid_fac2.E_r
                if len(A_r) != len(E_r):
                    continue                
                t1 = np.sort(np.array(A_r))
                t2 = np.sort(np.array(E_r))
                if np.sum(np.abs(t1-t2)) < abs_tol:
                    found = True
                    break
            if found:
                if verbose > 0:
                    print "Ridge " + str(E_r) + " already visited, removing from L.."
                if rid_fac2 == rid_fac1:
                    found_org = True
                L.remove(rid_fac2)
            else:
                if verbose > 0:
                    print "Adding ridge-facet " + str(E_adj) + " " + str(E_r) + ""
                L.append( Ridge_Facet(E_r,ar,br,E_adj,a_adj,b_adj))      
                
        if not found_org:
            print "Expected ridge " + str(rid_fac1.E_r)
            print "but got ridges "
            for rid in ridge_list:
                print rid.E_r
            raise Exception("esp: ridge did not return neighboring ridge as expected")
            
        G = np.vstack([G, a_adj])
        g = np.hstack([g, b_adj])
                
        E.append(E_adj)
    
    # Restore center
    if trans:
        g = g + np.dot(G,xc0)
    
    # Return zero rows
    for Ef in E:
        Ef = nonzerorows[Ef]
    
    return G,g,E
    
def shoot(C,D,b,maxiter=1000,abs_tol=1e-7):
    '''Returns a randomly selected equality set E_0 of P such
    that the projection of the equality set is a facet of the projection
    
    Input:
    
    `C`: Matrix defining the polytope Cx+Dy <= b
    `D`: Matrix defining the polytope Cx+Dy <= b
    `b`: Vector defining the polytope Cx+Dy <= b
    
    Output:
    
    `E_0,af,bf`: Equality set and affine hull
    '''

    d = C.shape[1]
    k = D.shape[1]
    iter = 0
    while True:
        if iter > maxiter:
            raise Exception("shoot: could not find starting equality set")
        gamma = np.random.rand(d) - 0.5
        
        c = np.zeros(k+1)
        c[0] = -1
        G = np.hstack([np.array([np.dot(C,gamma)]).T,D])
        sol = solvers.lp(matrix(c), matrix(G) , matrix(b), None, None, lp_solver)
        opt_sol = np.array(sol['x']).flatten()
        opt_dual = np.array(sol['z']).flatten()
        r_opt = opt_sol[0]
        y_opt = np.array(opt_sol[ range(1,len(opt_sol)) ]).flatten()
        x_opt = r_opt*gamma
                
        E_0 = np.nonzero(np.abs(np.dot(C,x_opt) + np.dot(D,y_opt) - b) < abs_tol)[0]
        DE0 = D[E_0,:]
        CE0 = C[E_0,:]
        b0 = b[E_0]     
        if rank(np.dot(null_space(DE0.T).T, CE0)) == 1:
            break    
        iter += 1
            
    af,bf = proj_aff(CE0,DE0,b0,abs_tol=abs_tol)
        
    if is_dual_degenerate(c,G,b,None,None,opt_sol,opt_dual,abs_tol=abs_tol):
        E_0 = unique_equalityset(C,D,b,af,bf,abs_tol=abs_tol)
    af,bf = proj_aff(C[E_0,:],D[E_0,:],b[E_0])
    if len(bf) > 1:
        raise Exception("shoot: wrong dimension of affine hull")
    return E_0,af.flatten(),bf

def ridge(C,D,b,E,af,bf,abs_tol=1e-7,verbose=0): 
    '''
    Computes all the ridges of a facet in the projection.
    
    Input:
    `C,D,b`: Original polytope data
    `E,af,bf`: Equality set and affine hull of a facet in the projection
    
    Output:
    `ridge_list`: A list containing all the ridges of the facet as Ridge objects
    '''

    d = C.shape[1]
    k = D.shape[1]
    
    Er_list = []
    
    q = C.shape[0]
    
    E_c = np.setdiff1d(range(q),E)
    
    C_E = C[E,:]
    D_E = D[E,:]
    b_E = b[E,:]
    
    C_Ec = C[E_c,:]
    D_Ec = D[E_c,:]
    b_Ec = b[E_c]
    
    S = C_Ec - np.dot( np.dot(D_Ec,linalg.pinv(D_E)) , C_E)
    L = np.dot(D_Ec,null_space(D_E))
    t = b_Ec - np.dot(D_Ec , np.dot(linalg.pinv(D_E) ,  b_E) )
    if rank( np.hstack([C_E, D_E]) ) < k+1:
        if verbose > 1:
            print "Doing recursive ESP call"
        u,s,v = linalg.svd(np.array([af]), full_matrices=1)
        sigma = s[0]
        v = v.T * u[0,0]    # Correct sign
                
        V_hat = v[:,[0]]
        V_tilde = v[:,range(1,v.shape[1])]
        Cnew = np.dot(S,V_tilde)
        Dnew = L
        bnew = t - np.dot(S,V_hat).flatten() * bf / sigma
        Anew = np.hstack([Cnew,Dnew])
        xc2,yc2,cen2 = cheby_center(Cnew,Dnew,bnew)
        bnew = bnew - np.dot(Cnew,xc2).flatten() - np.dot(Dnew,yc2).flatten()
        Gt,gt,E_t = esp(Cnew, Dnew, bnew, centered=True,abs_tol=abs_tol,verbose=0)
        if (len(E_t[0]) == 0) or (len(E_t[1]) == 0):
            raise Exception("ridge: recursive call did not return any equality sets")
        for i in range(len(E_t)):
            E_f = E_t[i]
            er = np.sort( np.hstack([E, E_c[E_f]]) )
            ar = np.dot(Gt[i,:],V_tilde.T).flatten()
            br0 = gt[i].flatten()
            
            # Make orthogonal to facet
            ar = ar - af*np.dot(af.flatten(),ar.flatten())
            br = br0 - bf*np.dot(af.flatten(),ar.flatten())
            
            # Normalize and make ridge equation point outwards
            norm = np.sqrt(np.sum(ar*ar))
            ar = ar*np.sign(br)/norm
            br = br*np.sign(br)/norm
            
            # Restore center
            br = br + np.dot(Gt[i,:],xc2)/norm

            if len(ar) > d:
                raise Exception("ridge: wrong length of new ridge!")
            Er_list.append(Ridge(er,ar,br))  
    
    else:
        if verbose > 0:
            print "Doing direct calculation of ridges"             
        X = np.arange(S.shape[0])
        while len(X) > 0:
            i = X[0]
            X = np.setdiff1d(X,i)
            if np.linalg.norm(S[i,:]) < abs_tol:
                continue
            Si = S[i,:]
            Si = Si / np.linalg.norm(Si)          
            if np.linalg.norm(af - np.dot(Si,af)*Si) > abs_tol:
                
                test1 = null_space(np.vstack([  np.hstack([af, bf])  , np.hstack([ S[i,:], t[i] ])  ]), nonempty=True)
                test2 = np.hstack([S, np.array([t]).T])
                test = np.dot(test1.T , test2.T)
                test = np.sum(np.abs(test), 0)
                Q_i = np.nonzero(test > abs_tol)[0]
                Q = np.nonzero(test < abs_tol)[0]

                X = np.setdiff1d(X,Q)
                
                # Have Q_i     
                Sq = S[Q_i,:]
                tq = t[Q_i]
                
                c = np.zeros(d+1)
                c[0] = 1
                Gup = np.hstack([-np.ones([Sq.shape[0],1]),Sq])
                Gdo = np.hstack([-1, np.zeros(Sq.shape[1])])
                G = np.vstack([Gup, Gdo])
                h = np.hstack([tq, 1])
                
                Al = np.zeros([2, 1])
                Ar = np.vstack([af,S[i,:]])
                A = np.hstack([Al,Ar])
                bb = np.hstack([bf,t[i]])
                
                solvers.options['show_progress']=False
                solvers.options['LPX_K_MSGLEV'] = 0
                sol = solvers.lp(matrix(c), matrix(G) , matrix(h), matrix(A), matrix(bb), lp_solver)
                if sol['status'] == 'optimal':
                    tau = sol['x'][0]
                    if tau < -abs_tol:
                        ar = np.array([S[i,:]]).flatten()
                        br = t[i].flatten()
                        
                        # Make orthogonal to facet
                        ar = ar - af*np.dot(af.flatten(),ar.flatten())
                        br = br - bf*np.dot(af.flatten(),ar.flatten())
            
                        # Normalize and make ridge equation point outwards
                        norm = np.sqrt(np.sum(ar*ar))
                        ar = ar/norm
                        br = br/norm
                        
                        Er_list.append(Ridge(np.sort(np.hstack([E,E_c[Q]])),ar,br))
    return Er_list
        
def adjacent(C,D,b,rid_fac,abs_tol=1e-7):
    '''
    Compute the (unique) adjacent facet.

    Input:
    `rid_fac`: A Ridge_Facet object containing the parameters for a facet and one of
               its ridges.
    
    Output:
    `E_adj,a_adj,b_adj`: The equality set and parameters for the adjacent facet such that 
                         P_{E_adj} = P intersection {x | a_adj x = b_adj}
    '''
        
    E = rid_fac.E_0
    af = rid_fac.af
    bf = rid_fac.bf
    
    E_r = rid_fac.E_r
    ar = rid_fac.ar
    br = rid_fac.br
    
    d = C.shape[1]
    k = D.shape[1]
        
    C_er = C[E_r,:]
    D_er = D[E_r,:]
    b_er = b[E_r]
    
    c = -np.hstack([ar,np.zeros(k)])
    G = np.hstack([C_er,D_er])
    h = b_er
    
    A = np.hstack([af, np.zeros(k)])
    
    sol = solvers.lp(matrix(c), matrix(G) , matrix(h), matrix(A).T, matrix(bf*(1-0.01)), lp_solver)
    
    if sol['status'] != "optimal":
        print G
        print h
        print af
        print bf
        print ar
        print br
        print np.dot(af,ar)
        from scipy import io as sio
        data = {}
        data["C"] = C
        data["D"] = D
        data["b"] = b
        sio.savemat("matlabdata", data) 
        
        import pickle
        pickle.dump(data, open( "polytope.p", "wb" ) )
        
        raise Exception("adjacent: Lp returned status " + str(sol['status']))
    opt_sol = np.array(sol['x']).flatten()
    dual_opt_sol = np.array(sol['z']).flatten()
    x_opt = opt_sol[range(0,d)]
    y_opt = opt_sol[range(d,d+k)]

    if is_dual_degenerate(c.flatten(),G,h,A,bf*(1-0.01),opt_sol,dual_opt_sol,abs_tol=abs_tol):
        # If degenerate, compute affine hull and take preimage
        E_temp = np.nonzero(np.abs(np.dot(G,opt_sol) - h) < abs_tol)[0]
        a_temp,b_temp = proj_aff(C_er[E_temp,:], D_er[E_temp,:], b_er[E_temp], expected_dim=1, abs_tol=abs_tol)
        E_adj = unique_equalityset(C,D,b,a_temp,b_temp,abs_tol=abs_tol)
        if len(E_adj) == 0:          
            from scipy import io as sio
            data = {}
            data["C"] = C
            data["D"] = D
            data["b"] = b
            data["Er"] = E_r + 1
            data["ar"] = ar
            data["br"] = br
            data["Ef"] = E + 1
            data["af"] = af
            data["bf"] = bf
            sio.savemat("matlabdata", data) 
            raise Exception("adjacent: equality set computation returned empty set")
    
    else:
        E_adj = np.nonzero(np.abs(np.dot(C,x_opt) + np.dot(D,y_opt) - b) < abs_tol)[0]
        
    C_eadj = C[E_adj,:]
    D_eadj = D[E_adj,:]
    b_eadj = b[E_adj]
    af_adj,bf_adj = proj_aff(C_eadj,D_eadj,b_eadj,abs_tol=abs_tol)
    return E_adj, af_adj, bf_adj

def proj_aff(Ce,De,be,expected_dim=None,abs_tol=1e-7):
    '''Compute the set aff = {x | Ce x + De y = be} on the form
    aff = ({x | a x = b} intersection {Ce x + De y < be})
    
    Input: Polytope parameters Ce, De and be
    
    Output: Constants a and b'''
    
    # Remove zero columns
    ind = np.nonzero(np.sum(np.abs(De), axis=0) > abs_tol)[0]
    D = De[:,ind]
    if D.shape[1] == 0:   
        a = Ce
        b = be
        a_n, b_n = normalize(a,b)
        if expected_dim != None:
            if expected_dim != b_n.size:
                raise Exception("proj_aff: wrong dimension calculated in 1")
        return a_n.flatten(), b_n
    
    sh = np.shape(D.T)
    m = sh[0]
    n = sh[1]
        
    nDe = null_space(D.T) 
    a = np.dot(nDe.T,Ce)
    b = np.dot(nDe.T,be)
    
    a_n,b_n = normalize(a,b)
    
    if expected_dim != None:
        if expected_dim != b_n.size:
            raise Exception("proj_aff: wrong dimension calculated in 2")
    
    return a_n,b_n
        
def is_dual_degenerate(c,G,h,A,b,x_opt,z_opt,abs_tol=1e-7):
    '''Checks if the pair of dual problems
    
    (P): min c'x        (D): max h'z + b'y
         s.t Gx <= h         s.t G'z + A'y = c
             Ax = b                z <= 0
    
    is dual degenerate, i.e. if (P) has several optimal solutions.
    Optimal solutions x* and z* are required.
    
    Input:
    
    `G,h,A,b`: Parameters of (P)
    `x_opt`: One optimal solution to (P)
    `z_opt`: The optimal solution to (D) corresponding to _inequality constraints_ in (P)
    
    Output:
    `dual`: Boolean indicating whether (P) has many optimal solutions.
    '''                                   

    D = -G
    d = -h.flatten()
    
    mu = -z_opt.flatten() # mu >= 0
    
    I = np.nonzero(np.abs(np.dot(D,x_opt).flatten() - d) < abs_tol)[0]   # Active constraints
    J = np.nonzero(mu > abs_tol)[0]                            # Positive elements in dual opt
    
    i = mu < abs_tol            # Zero elements in dual opt    
    i = i.astype(int)
    j = np.zeros(len(mu), dtype=int)
    j[I] = 1                    # 1 if active
    
    L = np.nonzero(i+j == 2)[0]    # Indices where active constraints have 0 dual opt
    
    nI = len(I)
    nJ = len(J)
    nL = len(L)
    
    DI = D[I,:]       # Active constraints
    DJ = D[J,:]       # Constraints with positive lagrange mult
    DL = D[L,:]       # Active constraints with zero dual opt
        
    dual = 0
    
    if A == None:
        test = DI
    else:
        test = np.vstack([DI,A])
    if rank(test) < np.amin(DI.shape):
        return True
    else:
        if len(L) > 0:
            if A == None:
                Ae = DJ
            else:
                Ae = np.vstack([DJ,A])
            be = np.zeros(Ae.shape[0])
            Ai = -DL
            bi = np.zeros(nL)
            sol = solvers.lp(-matrix(np.sum(DL, axis=0)),matrix(Ai), matrix(bi), matrix(Ae), matrix(be), lp_solver)
            if sol['status'] == "dual infeasible":
                # Dual infeasible -> primal unbounded -> value>epsilon
                return True
            if sol['primal objective'] > abs_tol:
                return True     
    return False

def unique_equalityset(C,D,b,af,bf,abs_tol=1e-7,verbose=0):
    '''Return the equality set E such that
    
    P_E = {x | af x = bf} intersection P
    
    where P is the polytope C x + D y < b
    
    The inequalities have to be satisfied with equality everywhere on
    the face defined by af and bf.'''
    
    if D != None:
        A = np.hstack([C,D])
        a = np.hstack([af, np.zeros(D.shape[1]) ])
    else:
        A = C
        a = af
    E = []
    for i in range(A.shape[0]):
        A_i = np.array(A[i,:])
        b_i = b[i]
        sol = solvers.lp(matrix(A_i), matrix(A) , matrix(b), matrix(a).T, matrix(bf), lp_solver)
        if sol['status'] != "optimal":
            raise Exception("unique_equalityset: LP returned status " + str(sol['status']))
        if np.abs(sol['primal objective'] - b_i) < abs_tol:
            # Constraint is active everywhere
            E.append(i)
    if len(E) == 0:
        raise Exception("unique_equalityset: empty E")
    return np.array(E)

def unique_equalityset2(C,D,b,opt_sol,abs_tol=1e-7):
    
    A = np.hstack([C,D])
    
    E0 = np.nonzero(np.abs(np.dot(A,opt_sol)-b) < abs_tol)[0]
    af,bf = proj_aff(C[E0,:],D[E0,:],b[E0],expected_dim=1)
    
    ineq = np.hstack([af, np.zeros(D.shape[1])])
    G = np.vstack([A, np.vstack([ineq,-ineq])])
    h = np.hstack([b, np.hstack([bf, -bf])])
    
    m = G.shape[0]
    n = G.shape[1]
    
    e = 1e-3
    v = np.vstack([np.zeros([1,n]), np.eye(n)]).T
    v = v - np.array([np.mean(v, axis=1)]).T
    v = v*e
    
    ht = h + np.amin(-np.dot(G,v), axis=1)
    
    H1 = np.hstack([G, -np.eye(m)])
    H2 = np.hstack([G, np.zeros([m,m])])
    H3 = np.hstack([np.zeros([m,n]), -np.eye(m)])
    H = np.vstack([H1, np.vstack([H2,H3]) ])
    h = np.hstack([ht, np.hstack([h, np.zeros(m)]) ])
    c = np.hstack([np.zeros(n), np.ones(m)])
    
    sol = solvers.lp(matrix(c), matrix(H), matrix(h), None, None, lp_solver)
    if not sol['status'] == "optimal":
        raise Exception("unique_equalityset: LP returned status " + str(sol['status']))
    opt_sol2 = np.array(sol['x']).flatten()
    x = opt_sol2[range(0,n)]
    s = opt_sol2[range(n,len(opt_sol2))]
    E = np.nonzero(s > abs_tol)[0]
    print E
    E = np.sort(E[np.nonzero(E < C.shape[0])])
    
    # Check that they define the same projection
    at,bt = proj_aff(C[E,:],D[E,:],b[E])
    if bt.size != 1 or np.sum(np.abs(at - af)) + np.abs(bt - bf) > abs_tol:
        raise Exception("unique_equalityset2: affine hulls not the same")
    return E
    
def cheby_center(C,D,b):
    '''Calculates the chebyshev center for polytope
    
    C x + D y <= b
    
    Input:
    `C, D, b`: Polytope parameters
    
    Output:
    `x_0, y_0`: The chebyshev centra
    `boolean`: True if a point could be found, False otherwise'''
    
    d = C.shape[1]
    k = D.shape[1]
    A = np.hstack([C,D])
    dim = np.shape(A)[1]
    c = -np.r_[np.zeros(dim),1]
    norm2 = np.sqrt(np.sum(A*A, axis=1))
    G = np.c_[A, norm2]
    solvers.options['show_progress']=False
    solvers.options['LPX_K_MSGLEV'] = 0
    sol = solvers.lp(matrix(c), matrix(G), matrix(b), None, None, lp_solver)
    if sol['status'] == "optimal":
        opt = np.array(sol['x'][0:-1]).flatten()
        return opt[range(0,d)], opt[range(d,d+k)], True
    else:
        return np.zeros(d), np.zeros(k), False
    
def normalize(AA,bb,abs_tol=1e-7):
    '''
    Normalize the equations A x = b such that
    A'A = 1 and b > 0. Also removes duplicate
    lines.
    
    '''
    if AA.size == 0:
        return AA,bb
        
    dim = AA.size/bb.size
    
    A = AA.copy().reshape(bb.size, dim)
    b = bb.copy().reshape(bb.size,1)
    
    # Remove zero lines
    keepind = np.nonzero(np.sum(np.abs(np.hstack([A,b])), axis=1) > abs_tol) [0]
    A = A[keepind,:]
    b = b[keepind]
    
    # Normalize
    anorm = np.sqrt(np.sum(A*A, axis=1))
    for i in range(len(anorm)):
        A[i,:] = A[i,:]*np.sign(b[i,0])/anorm[i]
        b[i,0] = np.sign(b[i,0])*b[i,0]/anorm[i]
    
    # Remove duplicate rows
    keep_row = []
    for i in range(len(anorm)):
        unique = True
        for j in range(i+1,len(anorm)):
            test = np.sum(np.abs(A[i,:] - A[j,:])) + np.abs(b[i,0]-b[j,0])
            if test < abs_tol:
                unique = False
                break
        if unique:
            keep_row.append(i)
            
    A_n = A[keep_row,:]
    b_n = b[keep_row,0]
    if A_n.size == dim:         # Return flat A if only one row
        A_n = A_n.flatten()
    return A_n, b_n.flatten()
    
def rank(A, eps=1e-15):
    u, s, vh = linalg.svd(A)
    m = A.shape[0]
    n = A.shape[1]
    tol = np.amax([m,n]) * np.amax(s) * eps
    return np.sum(s > tol)

def null_space(A, eps=1e-15, nonempty=False):
    '''Returns the null space N_A to matrix A
    such that A N_A = 0'''
    u,s,v = linalg.svd(A, full_matrices=1)
    m = A.shape[0]
    n = A.shape[1]
    tol = np.amax([m,n]) * np.amax(s) * eps
    rank = np.sum(s > tol)
    N_space = v[range(rank,n),:].T
    
    if nonempty and (len(N_space) == 0):
        N_space = v[range(np.amax(n-1,1),n),:]

    return N_space

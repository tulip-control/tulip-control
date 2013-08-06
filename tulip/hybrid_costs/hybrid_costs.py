import sys, os, time, subprocess
from copy import deepcopy
import numpy as np
from scipy import io as sio
from scipy.linalg import block_diag
from cvxopt import matrix, solvers
from scipy import linalg
import random
from scipy import optimize

import polytope as pc
from prop2part import PropPreservingPartition
from errorprint import printWarning, printError, printInfo


#def get_input_helper_all(x0, ssys, H0, H1, N, R, Q):
#    '''Calculates the sequence u_seq such that
#    - x(t+1) = A x(t) + B u(t)
#    - x(k) \in H0 for k = 0,...,M
#    - x(k) \in H1 for k = M+1,...,N
#    - u(k) \in U for k = 0,...,N-1
#    and minimizes \sum x'(k) R x(k) + u'(k) Q u(k) for k = 0,...,N
#    over all values of M.
#    '''
#    minCost = np.Inf
#    optU = []
#    for k in xrange(1,N):
#        u, cost = get_input_helper_switch(x0, ssys, H0, H1, N, R, Q, k)
#        if cost < minCost:
#            minCost = cost
#            optU = u
#    return optU, minCost
#
#
#def get_input_helper(x0, ssys, H0, H1, N, R, Q):
#    '''Calculates the sequence u_seq such that
#    - x(t+1) = A x(t) + B u(t)
#    - x(k) \in H0 for k = 0,...,N-1
#    - x(k) \in H1 for k = N
#    - u(k) \in U for k = 0,...,N-1
#    and minimizes \sum x'(k) R x(k) + u'(k) Q u(k) for k = 0,...,N
#    '''
#    return get_input_helper_switch(x0, ssys, H0, H1, N, R, Q, N-1)
#
#  
#def get_input_helper_switch(x0, ssys, H0, H1, N, R, Q, switch):
#    '''Calculates the sequence u_seq such that
#    - x(t+1) = A x(t) + B u(t)
#    - x(k) \in H0 for k = 0,...,switch
#    - x(k) \in H1 for k = switch+1,...,N
#    - u(k) \in U for k = 0,...,N-1
#    and minimizes \sum x'(k) R x(k) + u'(k) Q u(k) for k = 0,...,N
#    '''
#    n = ssys.A.shape[1]
#    m = ssys.B.shape[1]
#    
#    # Create dictionary with powers of A matrix
#    A_dict = dict()     #(k --> A^k) key-value pairs
#    A_dict[0] = ssys.A.copy()
#    for k in xrange(1,N+1):
#        A_dict[k] = np.dot(A_dict[k-1],ssys.A)
#    
#    ## Create the X vector.  X = S_x X(0) + S_u U
#    S_x = np.eye(n)
#    for k in xrange(1,N+1):
#        S_x = np.vstack((S_x, A_dict[k]))
#    assert S_x.shape == (n*(N+1),n)
#
#    S_u = np.zeros((n*(N+1),m*N))
#    for row in xrange(1,N+1):      # First row is zeros (no control effect)
#        for col in xrange(0,k):
#            if row-col-1 >= 0:
#                S_u[n*row:n*(row+1),n*col:n*(col+1)] = np.dot(A_dict[row-col-1],ssys.B)
#            else:
#                break
#    assert S_u.shape == (n*(N+1),m*N)
#    
#    ## Define the cost function:  
#    # J:= X' Q_blk X + U' R_blk U  = U'HU + 2 x'(0)FU + x'(0)Yx(0),
#    # where H := (S_u' Q_blk S_u + R_blk), F := (S_x' Q_blk S_u), and Y := (S_x' Q_blk S_x)
#    Q_blk = block_diag(*[Q]*(N+1))
#    R_blk = block_diag(*[R]*N)
#    
#    H = np.dot(S_u.T, np.dot(Q_blk,S_u)) + R_blk
#    F = np.dot(S_x.T, np.dot(Q_blk,S_u))
#    Y = np.dot(S_x.T, np.dot(Q_blk,S_x))
#    assert H.shape == (m*N,m*N)
#    assert F.shape == (n,m*N)
#    assert Y.shape == (n,n)
#
#    ## Create the input constraint matrices
#    # D u(k) <= d for all k = 0,...,N-1
#    # D_blk = blockdiag(D,...,D) and d_blk = [d',...,d']
#    D_blk = block_diag(*[ssys.Uset.A]*N)
#    d_blk = np.tile(ssys.Uset.b,(1,N))[0]
#    
#    ## Create the state constraint matrices
#    # H0 x(k) <= h0 for all k = 0,...,N-1 and H1 x(N) <= h1
#    # H_blk = blockdiag(H0,H0,...,H1) and h_blk = [h0',h0',,...,h1']
#    H_before = block_diag(*[H0.A]*(switch+1))
#    H_after = block_diag(*[H1.A]*(N-switch))
#    H_blk = block_diag(H_before,H_after)
#    h_before = np.tile(H0.b,(1,(switch+1)))[0]
#    h_after = np.tile(H1.b,(1,(N-switch)))[0]
#    h_blk = np.hstack((h_before,h_after))
#    
#    # Combine input and state constraints
#    G = np.vstack((np.dot(H_blk, S_u), D_blk))
#    h = np.hstack((h_blk-np.dot(H_blk, np.dot(S_x,x0)), d_blk))
#        
#    P = 2*matrix(H)     #factor of 2 for cvxopt input
#    q = matrix(2*np.dot(x0,F)) 
#    G = matrix(G)
#    h = matrix(h)
#            
#    sol = solvers.qp(P,q,G,h)
#    
#    if sol['status'] != "optimal":
#        raise Exception("getInputHelper: QP solver finished with status " + \
#                        str(sol['status']))
#    u = np.array(sol['x']).flatten()
#    cost = sol['primal objective'] + np.dot(x0,np.dot(Y,x0))
#    
#    return u.reshape(N, m), cost



def cst_min_cost_bi(ssys, H0, H1, xf, N, R, Q):
    
    def cstr(x, *args):
        A,b = args[6:9]
        return  b - np.dot(A,x)  # b - Ax >= 0

    def func(x, *args):
        ssys, H0, H1, N, R, Q = args[0:6]
        try:
            u, cost = get_input_helper(x, ssys, H0, H1, N, R, Q)
        except:
            pass
        return cost
    # Constraint to start in H0 (Ax <= b)
    A = H0.A
    b = H0.b
        
    # Guess point in middle of H0
    rd,xd = pc.cheby_ball(H0)
    x_init = xd.flatten() #-xf  #Coordinate shift
        
    # Call nonlinear solver from scipy.optimize
    soln = optimize.fmin_slsqp(func, x_init, args = (ssys, H0, H1, N, R, Q, A, b), \
                               f_ieqcons = cstr, \
                               full_output=True, iprint = 0)   # cstr >= 0 is default
    if soln[3] == 0:      #soln = (out,fx,its,imode,smode)
        return soln[1]
    else:
        print "Solver returned non-optimal solution!"
        return None



def cst_min_cost(ssys, H0, H1, N, R, Q):
    
    switch = N-1
    
    # Constraint to start in H0 (Ax <= b)
    A = H0.A
    b = H0.b
    
    '''Calculates the sequence u_seq such that
    - x(t+1) = A x(t) + B u(t)
    - x(k) \in H0 for k = 0,...,switch
    - x(k) \in H1 for k = switch+1,...,N
    - u(k) \in U for k = 0,...,N-1
    and minimizes \sum x'(k) R x(k) + u'(k) Q u(k) for k = 0,...,N
    '''
    n = ssys.A.shape[1]
    m = ssys.B.shape[1]
    
    # Create dictionary with powers of A matrix
    A_dict = dict()     #(k --> A^k) key-value pairs
    A_dict[0] = ssys.A.copy()
    for k in xrange(1,N+1):
        A_dict[k] = np.dot(A_dict[k-1],ssys.A)
    
    ## Create the X vector.  X = S_x X(0) + S_u U
    S_x = np.eye(n)
    for k in xrange(1,N+1):
        S_x = np.vstack((S_x, A_dict[k]))
    assert S_x.shape == (n*(N+1),n)

    S_u = np.zeros((n*(N+1),m*N))
    for row in xrange(1,N+1):      # First row is zeros (no control effect)
        for col in xrange(0,k):
            if row-col-1 >= 0:
                S_u[n*row:n*(row+1),n*col:n*(col+1)] = np.dot(A_dict[row-col-1],ssys.B)
            else:
                break
    assert S_u.shape == (n*(N+1),m*N)
    
    ## Define the cost function:  
    # J:= X' Q_blk X + U' R_blk U  = U'HU + 2 x'(0)FU + x'(0)Yx(0),
    # where H := (S_u' Q_blk S_u + R_blk), F := (S_x' Q_blk S_u), and Y := (S_x' Q_blk S_x)
    Q_blk = block_diag(*[Q]*(N+1))
    R_blk = block_diag(*[R]*N)
    
    H = np.dot(S_u.T, np.dot(Q_blk,S_u)) + R_blk
    F = np.dot(S_x.T, np.dot(Q_blk,S_u))
    Y = np.dot(S_x.T, np.dot(Q_blk,S_x))
    assert H.shape == (m*N,m*N)
    assert F.shape == (n,m*N)
    assert Y.shape == (n,n)

    ## Create the input constraint matrices
    # D u(k) <= d for all k = 0,...,N-1
    # D_blk = blockdiag(D,...,D) and d_blk = [d',...,d']
    D_blk = block_diag(*[ssys.Uset.A]*N)
    d_blk = np.tile(ssys.Uset.b,(1,N))[0]
    
    ## Create the state constraint matrices
    # H0 x(k) <= h0 for all k = 0,...,N-1 and H1 x(N) <= h1
    # H_blk = blockdiag(H0,H0,...,H1) and h_blk = [h0',h0',,...,h1']
    H_before = block_diag(*[H0.A]*(switch+1))
    H_after = block_diag(*[H1.A]*(N-switch))
    H_blk = block_diag(H_before,H_after)
    h_before = np.tile(H0.b,(1,(switch+1)))[0]
    h_after = np.tile(H1.b,(1,(N-switch)))[0]
    h_blk = np.hstack((h_before,h_after))
    
    # Combine input and state constraints
    G = np.vstack((np.dot(H_blk, S_u), D_blk))
    h = np.hstack((h_blk-np.dot(H_blk, np.dot(S_x,x0)), d_blk))
        
    P = 2*matrix(H)     #factor of 2 for cvxopt input
    q = matrix(2*np.dot(x0,F)) 
    G = matrix(G)
    h = matrix(h)
            
    sol = solvers.qp(P,q,G,h)
    
    if sol['status'] != "optimal":
        raise Exception("getInputHelper: QP solver finished with status " + \
                        str(sol['status']))
    u = np.array(sol['x']).flatten()
    cost = sol['primal objective'] + np.dot(x0,np.dot(Y,x0))
    
    return u.reshape(N, m), cost
        



def cst_max_cost(ssys, H0, H1, N, R, Q):
    '''Calculates the maximum cost of a trajectory under the constrained LQR problem.
    Uses the fact that the constrained LQR value function is convex in initial state after
    minimization over control inputs u, so a maximum value must occur on a vertex of the polygon.'''
    vertices = pc.extreme(H0)
    
    max_cost = -np.Inf
    for v in vertices:
        u,cost = get_input_helper(v, ssys, H0, H1, N, R, Q)
        if cost > max_cost:
            max_cost = cost
            
    return max_cost


def cst_expected_cost(num_samples, ssys, H0, H1, N, R, Q):
    '''Calculates the expected cost of a trajectory starting in H0 and ending in H1.'''
    
    total_cost = 0.0
    pts = sample_pts_poly(H0,num_samples)
    while len(pts)>0:
        x0 = pts.pop()
        u,cost = get_input_helper(x0, ssys, H0, H1, N, R, Q)
        total_cost += cost
    return total_cost / float(num_samples)




def lqr_value(ssys, N, R, Q):
    '''Calculates the discrete-time finite-horizon LQR value function at initial stage.
    Value(x) = x.T * P0 * x for state x from the initial stage.'''
    
    A = ssys.A
    B = ssys.B
    P_N = Q
    Pdict = dict()
    Pdict[N] = P_N
    
    for k in xrange(N,0,-1):
        P_k = Pdict[k]
        #P_{k-1} = Q + A.T*(P_k - P_k*B * (R + B.T*P_k*B)^-1 * B.T*P_k)*A    
        Pdict[k-1] = Q + A.T*(P_k - P_k*B* linalg.solve(R + B.T*P_k*B, B.T*P_k) )*A
    
    return Pdict[0]


def lqr_min_cost(ssys, H0, xf, N, R, Q):
    '''Calculates the minimum cost of a trajectory under the unconstrained LQR problem.
    Uses the fact that the unconstrained LQR value function is convex.
    min_x (x-xf)' P0 (x-xf) s.t. Gx <= h.
    Note: xf is not chosen as part of the optimization, or else the LQR value would be
    zero by selecting x = xf on the boundary between the two regions.'''
    
    P0 = lqr_value(ssys, N, R, Q)
        
    P = 2.0*matrix(P0)        #scale for cvxopt
    q = matrix( -2.0*np.dot(xf.T, P0) )
    G = matrix(H0.A)
    h = matrix(H0.b)
    
    sol = solvers.qp(P,q,G,h)
    if sol['status'] == 'optimal':
        min_cost = sol['primal objective'] + np.dot(xf.T, np.dot(P0,xf)) #Coordinate shift
    else:
        print "QP solver returned non-optimal solution!"
        return None
    return min_cost


def lqr_max_cost(ssys, H0, xf, N, R, Q):
    '''Calculates the maximum cost of a trajectory under the unconstrained LQR problem.
    Uses the fact that the unconstrained LQR value function is convex, so a maximum value
    must occur on a vertex of the polygon.'''
    # TODO: Make xf a free variable that is uncontrolled

    P0 = lqr_value(ssys, N, R, Q)
    vertices = pc.extreme(H0)
    
    max_cost = -np.Inf
    for v in vertices:
        v = v - xf  #Coordinate shift
        cost = np.dot(v,np.dot(P0,v))     # V(v) = v.T*P0*v from unconstrained LQR theory
        if cost > max_cost:
            max_cost = cost
    
    return max_cost


def lqr_expected_cost(num_samples, ssys, H0, H1, N, R, Q):
    '''Calculates the expected cost of a trajectory starting in P0 and ending in P1.
    Uses an LQR approximation.'''
    
    P0 = lqr_value(ssys, N, R, Q)
    
    total_cost = 0.0
    pts_x0 = sample_pts_poly(H0,num_samples)
    pts_xf = sample_pts_poly(H1,num_samples)
    while len(pts_x0)>0:
        x = pts_x0.pop() - pts_xf.pop()     #Coordinate shift
        cost = np.dot(x,np.dot(P0,x))     # V(v) = v.T*P0*v from unconstrained LQR theory
        total_cost += cost
    return total_cost / float(num_samples)


def sample_pts_poly(H0,num_pts):
    '''Generate uniform sampled points inside a polygon H0.'''
    # TODO: Replace with outer ball approximation
    pts = list()
        
    rd,xd = pc.cheby_ball(H0)
    xd = xd.flatten()
    expand_rd = 1.5     #expansion factor to make polygon more inside the Chebyshev ball
    rd *= expand_rd
    
    # Sample new point in H0
    for k in xrange(num_pts):
        while True:
            z = np.random.multivariate_normal(xd,rd**2*np.eye(2))       #sample Gaussian centered on Cheby ball
            if pc.is_inside(H0, z):
                pts.append(z)
                break
    return pts

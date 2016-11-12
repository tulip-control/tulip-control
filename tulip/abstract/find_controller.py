# Copyright (c) 2011-2014 by California Institute of Technology
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
Algorithms related to controller synthesis for discretized dynamics.

Primary functions:
    - L{get_input}
    
Helper functions:
    - L{get_input_helper}
    - L{is_seq_inside}

See Also
========
L{discretize}
"""
from __future__ import absolute_import

import logging
import numpy as np
import polytope as pc

from .feasible import solve_feasible, createLM, _block_diag2


logger = logging.getLogger(__name__)


def get_input(
    x0, ssys, abstraction,
    start, end,
    R=[], r=[], Q=[], mid_weight=0.0,
    test_result=False,ord=1
):
    """Compute continuous control input for discrete transition.

    Computes a continuous control input sequence
    which takes the plant:
        
        - from state C{start}
        - to state C{end}

    These are states of the partition C{abstraction}.
    The computed control input is such that::
        
        f(x, u) = |Rx|_inf+|Qu|_inf+r'x+mid_weight *|xc-x(0)|_inf

    be minimal.

    C{xc} is the chebyshev center of the final cell.
    If no cost parameters are given, then the defaults are:

        - Q = I
        - mid_weight = 3

    Notes
    =====
    1. The same horizon length as in reachability analysis
        should be used in order to guarantee feasibility.

    2. If the closed loop algorithm has been used
        to compute reachability the input needs to be
        recalculated for each time step
        (with decreasing horizon length).

        In this case only u(0) should be used as
        a control signal and u(1) ... u(N-1) discarded.

    3. The "conservative" calculation makes sure that
        the plant remains inside the convex hull of the
        starting region during execution, i.e.::

            x(1), x(2) ...  x(N-1) are
            \in conv_hull(starting region).

        If the original proposition preserving partition
        is not convex, then safety cannot be guaranteed.

    @param x0: initial continuous state
    @type x0: numpy 1darray

    @param ssys: system dynamics
    @type ssys: L{LtiSysDyn}

    @param abstraction: abstract system dynamics
    @type abstraction: L{AbstractPwa}

    @param start: index of the initial state in C{abstraction.ts}
    @type start: int >= 0

    @param end: index of the end state in C{abstraction.ts}
    @type end: int >= 0

    @param R: state cost matrix for::
            x = [x(1)' x(2)' .. x(N)']'
        If empty, zero matrix is used.
    @type R: size (N*xdim x N*xdim)

    @param r: cost vector for state trajectory:
        x = [x(1)' x(2)' .. x(N)']'
    @type r: size (N*xdim x 1)

    @param Q: input cost matrix for control input::
            u = [u(0)' u(1)' .. u(N-1)']'
        If empty, identity matrix is used.
    @type Q: size (N*udim x N*udim)

    @param mid_weight: cost weight for |x(N)-xc|_2

    @param test_result: performs a simulation
        (without disturbance) to make sure that
        the calculated input sequence is safe.
    @type test_result: bool

    @param ord: norm used for cost function::
	f(x, u) = |Rx|_{ord} + |Qu|_{ord} + r'x + mid_weight *|xc-x(0)|_{ord}
    @type ord: ord \in {1,2,'inf'}

    @return: array A where row k contains the
        control input: u(k)
        for k = 0,1 ... N-1
    @rtype: (N x m) numpy 2darray
    """

    #@param N: horizon length
    #@type N: int >= 1

    #@param conservative:
    #    if True,
    #    then force plant to stay inside initial
    #    state during execution.
    #    
    #    Otherwise, plant is forced to stay inside
    #    the original proposition preserving cell.
    #@type conservative: bool

    #@param closed_loop: should be True
    #    if closed loop discretization has been used.
    #@type closed_loop: bool

    part = abstraction.ppp
    regions = part.regions

    ofts = abstraction.ts
    original_regions = abstraction.orig_ppp
    orig = abstraction._ppp2orig

    params = abstraction.disc_params
    N = params['N']
    conservative = params['conservative']
    closed_loop = params['closed_loop']

    if (len(R) == 0) and (len(Q) == 0) and \
    (len(r) == 0) and (mid_weight == 0):
        # Default behavior
        Q = np.eye(N*ssys.B.shape[1])
        R = np.zeros([N*x0.size, N*x0.size])
        r = np.zeros([N*x0.size,1])
        mid_weight = 3
    if len(R) == 0:
        R = np.zeros([N*x0.size, N*x0.size])
    if len(Q) == 0:
        Q = np.eye(N*ssys.B.shape[1])    
    if len(r) == 0:
        r = np.zeros([N*x0.size,1])

    if (R.shape[0] != R.shape[1]) or (R.shape[0] != N*x0.size):
        raise Exception("get_input: "
            "R must be square and have side N * dim(state space)")

    if (Q.shape[0] != Q.shape[1]) or (Q.shape[0] != N*ssys.B.shape[1]):
        raise Exception("get_input: "
            "Q must be square and have side N * dim(input space)")
    if ofts is not None:
        start_state = start
        end_state = end

        if end_state not in ofts.states.post(start_state):
            raise Exception('get_input: '
                'no transition from state s' +str(start) +
                ' to state s' +str(end)
            )
    else:
        print("get_input: "
            "Warning, no transition matrix found, assuming feasible")

    if (not conservative) & (orig is None):
        print("List of original proposition preserving "
            "partitions not given, reverting to conservative mode")
        conservative = True

    P_start = regions[start]
    P_end = regions[end]

    n = ssys.A.shape[1]
    m = ssys.B.shape[1]

    idx = range((N-1)*n, N*n)

    if conservative:
        # Take convex hull or P_start as constraint
        if len(P_start) > 0:
            if len(P_start) > 1:
                # Take convex hull
                vert = pc.extreme(P_start[0])
                for i in range(1, len(P_start)):
                    vert = np.vstack([
                        vert,
                        pc.extreme(P_start[i])
                    ])
                P1 = pc.qhull(vert)
            else:
                P1 = P_start[0]
        else:
            P1 = P_start
    else:
        # Take original proposition preserving cell as constraint
        P1 = original_regions[orig[start]]
        # must be convex (therefore single polytope?)
        if len(P1) > 0:
            if len(P1) == 1:
                P1 = P1[0]
            else:
                print P1
                raise Exception("conservative = False flag requires "
                                "original regions to be convex")

    if len(P_end) > 0:
        low_cost = np.inf
        low_u = np.zeros([N,m])

        # for each polytope in target region
        for P3 in P_end:
            if mid_weight > 0:
                rc, xc = pc.cheby_ball(P3)
                R[
                    np.ix_(
                        range(n*(N-1), n*N),
                        range(n*(N-1), n*N)
                    )
                ] += mid_weight*np.eye(n)

                r[idx, :] += -mid_weight*xc
            #try:
                u, cost = get_input_helper_LP(
                    x0, ssys, P1, P3, N, R, r, Q, ord,
                    closed_loop=closed_loop
                )
                r[idx, :] += mid_weight*xc
            #except:
            #    r[idx, :] += mid_weight*xc
            #    continue
            
            if cost < low_cost:
                low_u = u
                low_cost = cost

        if low_cost == np.inf:
            raise Exception("get_input: Did not find any trajectory")
    else:
        P3 = P_end
        if mid_weight > 0:
            rc, xc = pc.cheby_ball(P3)
            R[
                np.ix_(
                    range(n*(N-1), n*N),
                    range(n*(N-1), n*N)
                )
            ] += mid_weight*np.eye(n)
            r[idx, :] += -mid_weight*xc
        low_u, cost = get_input_helper_LP(
            x0, ssys, P1, P3, N, R, r, Q, ord,
            closed_loop=closed_loop
        )

    if test_result:
        good = is_seq_inside(x0, low_u, ssys, P1, P3)
        if not good:
            print("Calculated sequence not good")
    return low_u

def get_input_helper_LP(
    x0, ssys, P1, P3, N, R, r, Q, ord=1,
    closed_loop=True
):
    """Calculates the sequence u_seq such that:

      - x(t+1) = A x(t) + B u(t) + K
      - x(k) \in P1 for k = 0,...N
      - x(N) \in P3
      - [u(k); x(k)] \in PU
    
    and minimizes x'Rx + 2*r'x + u'Qu
    """
    n = ssys.A.shape[1]
    m = ssys.B.shape[1]

    list_P = []
    if closed_loop:
        temp_part = P3
        list_P.append(P3)
        for i in xrange(N-1,0,-1): 
            temp_part = solve_feasible(
                P1, temp_part, ssys, N=1,
                closed_loop=False, trans_set=P1
            )
            list_P.insert(0, temp_part)
        list_P.insert(0,P1)
        L,M = createLM(ssys, N, list_P, disturbance_ind=[1])
    else:
        list_P.append(P1)
        for i in xrange(N-1,0,-1):
            list_P.append(P1)
        list_P.append(P3)
        L,M = createLM(ssys, N, list_P)

    # Remove first constraint on x(0)
    L = L[range(list_P[0].A.shape[0], L.shape[0]),:]
    M = M[range(list_P[0].A.shape[0], M.shape[0]),:]

    # Separate L matrix
    Lx = L[:,range(n)]
    Lu = L[:,range(n,L.shape[1])] 

    M = M - Lx.dot(x0).reshape(Lx.shape[0],1)

    B_diag = ssys.B
    for i in xrange(N-1):
        B_diag = _block_diag2(B_diag,ssys.B)
    K_hat = np.tile(ssys.K, (N,1))
    A_it = ssys.A.copy()
    A_row = np.zeros([n, n*N])
    A_K = np.zeros([n*N, n*N])
    A_N = np.zeros([n*N, n])

    for i in xrange(N):
        A_row = ssys.A.dot(A_row)
        A_row[np.ix_(
            range(n),
            range(i*n, (i+1)*n)
        )] = np.eye(n)

        A_N[np.ix_(
            range(i*n, (i+1)*n),
            range(n)
        )] = A_it

        A_K[np.ix_(
            range(i*n,(i+1)*n),
            range(A_K.shape[1])
        )] = A_row

        A_it = ssys.A.dot(A_it)
    Ct = A_K.dot(B_diag)
    if ord == 1:
        # f(\epsilon,u) = sum(\epsilon)
        c_LP = np.hstack((np.ones((1,N*(n+m))),r.T.dot(Ct)))
        # Constraints -\epsilon <= Q*u + R*x <= \epsilon
        # x = A_N*x0 + Ct*u --> ignore the first constant part
        # x  = Ct*u
        G_LP = np.vstack((
            np.hstack((-np.eye(N*n),np.zeros((N*n,N*m)),-R.dot(Ct))),
            np.hstack((-np.eye(N*n),np.zeros((N*n,N*m)),R.dot(Ct))),
            np.hstack((np.zeros((N*m,N*n)),-np.eye(N*m),-Q)),
            np.hstack((np.zeros((N*m,N*n)),-np.eye(N*m),Q)),
            np.hstack((np.zeros((Lu.shape[0],N*n+N*m)),Lu))
        ))
        h_LP = np.vstack((np.zeros((2*N*(n+m),1)),M))
    elif ord == 2:
        raise Exception("2-norm is Not Implemented")
        return
    elif ord == np.inf:
        c_LP = np.hstack((np.ones((1,2)),r.T.dot(Ct)))
        G_LP = np.vstack((
            np.hstack((-np.ones((N*n,1)),np.zeros((N*n,1)),-R.dot(Ct))),
            np.hstack((-np.ones((N*n,1)),np.zeros((N*n,1)),R.dot(Ct))),
            np.hstack((np.zeros((N*m,1)),-np.ones((N*m,1)),-Q)),
            np.hstack((np.zeros((N*m,1)),-np.ones((N*m,1)),Q)),
            np.hstack((np.zeros((Lu.shape[0],2)),Lu))
        ))
        h_LP = np.vstack((np.zeros((2*N*(n+m),1)),M))

    sol = pc.polytope.lpsolve(c_LP.flatten(), G_LP, h_LP)

    if sol['status'] != 0:
        raise Exception("getInputHelper: "
            "LP solver finished with status " +
            str(sol['status'])
        )
    var = np.array(sol['x']).flatten()
    u = var[-N*m:]
    cost = sol['fun']

    return u.reshape(N, m), cost

def is_seq_inside(x0, u_seq, ssys, P0, P1):
    """Checks if the plant remains inside P0 for time t = 1, ... N-1
    and  that the plant reaches P1 for time t = N.
    Used to test a computed input sequence.
    No disturbance is taken into account.

    @param x0: initial point for execution
    @param u_seq: (N x m) array where row k is input for t = k

    @param ssys: dynamics
    @type ssys: L{LtiSysDyn}

    @param P0: C{Polytope} where we want x(k) to remain for k = 1, ... N-1

    @return: C{True} if x(k) \in P0 for k = 1, .. N-1 and x(N) \in P1.
        C{False} otherwise  
    """
    N = u_seq.shape[0]
    x = x0.reshape(x0.size, 1)

    A = ssys.A
    B = ssys.B
    if len(ssys.K) == 0:
        K = np.zeros(x.shape)
    else:
        K = ssys.K

    inside = True
    for i in xrange(N-1):
        u = u_seq[i,:].reshape(u_seq[i, :].size, 1)
        x = A.dot(x) + B.dot(u) + K

        if not pc.is_inside(P0, x):
            inside = False

    un_1 = u_seq[N-1,:].reshape(u_seq[N-1, :].size, 1)
    xn = A.dot(x) + B.dot(un_1) + K

    if not pc.is_inside(P1, xn):
        inside = False

    return inside

def find_discrete_state(x0, part):
    """Return index identifying the discrete state
    to which the continuous state x0 belongs to.

    Notes
    =====
    1. If there are overlapping partitions
        (i.e., x0 belongs to more than one discrete state),
        then return the first discrete state ID

    @param x0: initial continuous state
    @type x0: numpy 1darray

    @param part: state space partition
    @type part: L{PropPreservingPartition}

    @return: if C{x0} belongs to some
        discrete state in C{part},
        then return the index of that state

        Otherwise return None, i.e., in case
        C{x0} does not belong to any discrete state.
    @rtype: int
    """
    for (i, region) in enumerate(part):
        if pc.is_inside(region, x0):
             return i
    return None

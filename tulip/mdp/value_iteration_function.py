# First cut of value iteration for a product MDP.
# Notes: [(L0, K0), ...] is list of tuples (bad, good).
# Dependencies: jbern_MDP_overhaul for MDP methods.
# Originally: James Bern 8/6/2013
#
# 8/9: made search for V0 much more efficient.
# 8/12: reintroduced discounting.
# 8/12: reintroduced costs.
# 8/16: wrote robust_value_iteration 
# 8/16: added and checked the two small examples
# 8/20: P_V_linear_program became own function.

from jbern_MDP_functions import *

from copy import deepcopy
from pprint import pprint

import numpy as np
from cvxopt import matrix
from cvxopt import solvers
from cvxopt import mul

# Helper Function
def value_iteration(mdp, epsilon=0.00001, V1o=set(), unknown_states=None, gamma=1, costs_enabled=False):
    """
    Modified/specialized AIMA algorithm for value iteration,
    which is used to perform value iteration over either all
    of a product MDP or part of it if graph search has been used
    to optimize.
    Called by product_MDP_value_iteration().

    NOTE unknown_states is at least prod.states - AMECs, and at most
    prod.states - (V1 U V0).
    """

    T = deepcopy(mdp.T)

    # Safely defaults to all non-V1 states in T.
    unknown_states = unknown_states
    if unknown_states == None:
        unknown_states = set(mdp.states) - V1o

    V1 = V1o # Solely used to diff. b/t s part of V0 vs. V1

    U1 = {s : 0.0 for s in unknown_states}

    gamma = gamma

    while True:
        U = deepcopy(U1)

        delta = 0

        for s in unknown_states:
            U1[s] = 0
            actions = T[s].keys()
            action_values_lst = []

            for a in actions:

                # optional cost, NOTE must pass costs_enabled=True
                if costs_enabled:
                    try:
                        U1[s] += C[s][a]
                    except:
                        raise Exception, "costs not fully defined"

                # vector of constant transition probabilities 
                P = []
                # vector of destination state values
                V = []

                for tup in T[s][a]:
                    s_f, p = tup
                    # determine value_s1
                    if s_f in unknown_states:
                        v = U[s_f]
                    elif s_f in V1:
                        v = gamma * 1.0
                    else: # s_f in V0
                        v = 0.0
                    P.append(p)
                    V.append(v)

                action_value = np.dot(np.array(P), np.array(V))
                action_values_lst.append(action_value)

            U1[s] = gamma * max(action_values_lst)


            delta = max(delta, abs(U1[s] - U[s]))

        # optional discounting
        if gamma == 1.0:
            e_factor = 1.0
        else:
            e_factor = (1 - gamma) / gamma

        if delta < e_factor * epsilon:
            return U1

def robust_value_iteration(mdp, epsilon=0.00001, V1o=set(), unknown_states=None, gamma=1, costs_enabled=False):
    """
    Value iteration for use in an UncertainMDP.
    Currently only supports interval probabilities.

    A modified version of value_iteration.
    """

    T = deepcopy(mdp.T)

    # Safely defaults to all non-V1 states in T.
    unknown_states = unknown_states
    if unknown_states == None:
        unknown_states = set(mdp.states) - V1o

    V1 = V1o # Solely used to diff. b/t s part of V0 vs. V1

    U1 = {s : 0.0 for s in unknown_states}

    gamma = gamma

    while True:
        U = deepcopy(U1)

        delta = 0

        for s in unknown_states:
            U1[s] = 0
            actions = T[s].keys()
            action_values_lst = []

            for a in actions:

                # optional cost, NOTE must pass costs_enabled=True
                if costs_enabled:
                    try:
                        U1[s] += C[s][a]
                    except:
                        raise Exception, "costs not fully defined"

                # vector of __interval__ transition probabilities 
                P = []
                # vector of destination state values
                V = []

                # set up P and V lists
                for tup in T[s][a]:
                    s_f, p_interval = tup

                    # determine value_s1
                    if s_f in unknown_states:
                        v = U[s_f]
                    elif s_f in V1:
                        v = gamma * 1.0
                    else: # s_f in V0
                        v = 0.0

                    P.append(p_interval)
                    V.append(v)

                # solve linear program for updated p values
                P = P_V_linear_program(P, V)
                # convert V into a cvxopt matrix for taking dot product
                V = matrix(V)

                # take the dot product of the fixed P and V
                action_value = sum(mul(P, V))
                #print action_value
                action_values_lst.append(action_value)

            # robot maximizes over the set of minimized actions
            # update U1[s] accordingly
            U1[s] = gamma * max(action_values_lst)
            #print U1

            delta = max(delta, abs(U1[s] - U[s]))

        # optional discounting
        if gamma == 1.0:
            e_factor = 1.0
        else:
            e_factor = (1 - gamma) / gamma

        if delta < e_factor * epsilon:
            return U1

def P_V_linear_program(P, V):
    """
    The environment minimizes dot(P, V) by fixing P with restriction 
    that entries of P sum to 1.
    Requires elements of P are IntervalProbability's, and
    elements of V are floats.
    Takes P, V as lists, and returns P as a cvxopt matrix.
    """
    assert type(P) == list
    assert type(V) == list
    for v in V:
        assert type(v) == float
    for p in P:
        assert isinstance(p, IntervalProbability), "P assumed matrix of IntervalProbability's"

    n = len(P)

    # c matrix, to be used as c:
    # call it V for clarity/consistency
    # form: col of state values
    V = matrix(V)
    #print V

    # G matrix, to be used as G:
    # form: -1  0  0 ...
    #       +1  0  0 ...
    #        0 -1  0 ...
    #        0 +1  0 ...
    #        ...........
    G = np.ndarray((2*n, n))
    G_n = -1 * np.identity(n)
    G_p = np.identity(n)
    for i in range(n):
        r = 2*i
        G[r] = G_n[i]
        G[r+1] = G_p[i]
    G = matrix(G)
    #print G

    # h matrix, to be used as h:
    # form: col of -p1F, +p1C, -p2F, +P2C, ...
    # (where F denotes floor/lower-bound on interval
    #  and C denotes ceiling/upper-bound on interval)
    h = []
    for p_interval in P:
        h.append(-1*p_interval.low)
        h.append(p_interval.high)
    h = matrix(h)
    #print h

    # A matrix, to be used as A:
    # form: same as c
    A = matrix(1.0, (1,n))
    #print A

    # b matrix
    # form: 1x1 with entry 1
    b = matrix(1.0, (1,1))
    #print b

    solvers.options["show_progress"] = False
    P = solvers.lp(V, G, h, A, b)['x']
    #for p in P:
    #    print p
    #print
    return P

def product_MDP_value_iteration(prod, AMECs, gamma=1):
    """
    Run value iteration on a product mdp with LK list.

    NOTE doesn't run graph search when gamma != 1.0,
    since the discounted value of a state v1 in V1 but not in AMECs
    is not 1.0.

    Chooses between robust_value_iteration and value_iteration().
    """
    assert prod.states == set(prod.T.keys()), "states doesn't match T"

    AMECs = AMECs

    V0 = set()
    V1 = set()

    # 1-out all states in AMEC
    for AMEC in AMECs:
        states = AMEC[0]
        V1.update(states)

    # Save a copy of V1 for if you want to compare time-savings of graph search
    # V1_safe = deepcopy(V1)

    #GRAPH SEARCH BLOCK
    print "\n* Running a graph search for V0. *\n"

    ##V0 Search (custom)
    # Use can_reach method to compute set of states
    # that can possibly reach an AMEC.
    can_reach_AMEC = prod.can_reach(V1)
    print "Found members of V0:"
    pprint(prod.states - can_reach_AMEC)
    V0.update(prod.states - can_reach_AMEC)

    ##V1 Search B+K pg. 859
    new_V1 = set()
    if gamma == 1.0:
        print "\n* Running a graph search for V1 *"
        prod2 = deepcopy(prod)

        U = deepcopy(V0)
        while len(U) > 0:
            R = deepcopy(U)
            while len(R) > 0:
                u = R.pop()

                for t_a_tup in prod2.pre_s_a(u):
                    t, a = t_a_tup
                    if t not in U:
                        del prod2.T[t][a]
                        if len(prod2.T[t]) == 0:
                            R.add(t)
                            U.add(t)
                # all incoming edges have been removed
                # remove u and its outoing edges from M
                del prod2.T[u]
            # Determine states s that cannot reach V1 in modified
            U_new = set()
            for s in prod.states:
                if s not in U and s in V0:
                    U_new.add(s)
            U = deepcopy(U_new)
        new_V1 = prod2.T.keys() # NOTE don't use states since it's no longer accurate.

        print "\nFound members of V1:"
        pprint(new_V1)

    else:
        print "\n* Skipping V1 graph search due to discounting. *"
        pass

    V1.update(new_V1)
    #END GRAPH SEARCH BLOCK

    print "\n* Running appropriate form of value iteration on unknown states *"

    # Perform value iteration to finish off
    unknown_states = deepcopy(prod.states)
    unknown_states -= V1
    unknown_states -= V0

    # switch case for types of value iteration.
    if prod.probability_type == float:
        print "\nvalue_iteration finds: "
        final_dict = value_iteration(prod, V1o=V1, unknown_states=unknown_states, gamma=gamma,
            costs_enabled=False)
    elif prod.probability_type == IntervalProbability:
        print "\nrobust_value_iteration finds: "
        final_dict = robust_value_iteration(prod, V1o=V1, unknown_states=unknown_states,
                gamma=gamma, costs_enabled=False)
    else:
        raise NotImplementedError, "using unsupported type of MDP."
    pprint(final_dict)

    print "\nusing stored members of V0 and V1 to build up final dict:"
    for v1 in V1:
        final_dict[v1] = 1.0
    for v0 in V0:
        final_dict[v0] = 0.0

    return final_dict

def __main__():
    print "\n" + "*"*80 + "\n"
    # Small example to test product_MDP_value_iteration
    # -for a normal MDP.
    # NOTE checked by hand: 0.333...
    prod = MDP(name="mdp_val_example")
    prod.states = set([0,1,2,3])
    prod.T = {0 : {'a':[(0,.25), (1,.25), (3,.5)]},
              1 : {'a':[(2,1)]},
              2 : {'a':[(1,1)]},
              3 : {'a':[(3,1)]}
             }
    prod.sanity_check()
    LK_lst = [(None, 2)]
    AMECs = accepting_max_end_components(prod, LK_lst)
    print prod.name
    pprint(product_MDP_value_iteration(prod, AMECs))

    print "\n" + "*"*80 + "\n"
    # Small example to test product_MDP_value_iteration
    # -for an UncertainMDP.
    # NOTE checked by hand: 0.2
    prod = UncertainMDP(name="mdp_robust_val_example")
    p25 = IntervalProbability(.15, .35)
    p50 = IntervalProbability(.4, .6)
    one = IntervalProbability.one()
    prod.states = set([0,1,2,3])
    prod.T = {0 : {'a':[(0,p25), (1,p25), (3,p50)]},
              1 : {'a':[(2,one)]},
              2 : {'a':[(1,one)]},
              3 : {'a':[(3,one)]}
             }
    prod.sanity_check()
    LK_lst = [(None, 2)]
    AMECs = accepting_max_end_components(prod, LK_lst)
    print prod.name
    pprint(product_MDP_value_iteration(prod, AMECs))

if __name__ == "__main__":
    __main__()


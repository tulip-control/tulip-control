# First cut of value iteration for a product MDP.
# Notes: [(L0, K0), ...] is list of tuples (bad, good).
# Dependencies: jbern_MDP_overhaul for MDP methods.
# Originally: James Bern 8/6/2013

from copy import deepcopy
from pprint import pprint
from jbern_MDP_overhaul import *

# Helper Function
def value_iteration(mdp, epsilon=0.001, V1=set(), unknown_states=None):
    """
    Modified/specialized AIMA algorithm for value iteration,
    which is used to perform value iteration over either all
    of a product MDP or part of it if graph search has been used
    to optimize.
    Used in product_MDP_value_iteration().
    """

    T = deepcopy(mdp.T)

    # Safely defaults to all non-V1 states in T.
    unknown_states = unknown_states
    if unknown_states == None:
        unknown_states = set(mdp.states) - V1

    V1 = V1 # Solely used to diff. b/t s part of V0 vs. V1

    U1 = {s : 0.0 for s in unknown_states}

    while True:
        U = deepcopy(U1)

        delta = 0

        for s in unknown_states:
            actions = T[s].keys()
            state_value = []

            for a in actions:
                action_values = []
                for tup in T[s][a]:
                    s1, p = tup
                    # determine value_s1
                    if s1 in unknown_states:
                        value_s1 = U[s1]
                    elif s1 in V1:
                        value_s1 = 1.0
                    else: # s1 in V0
                        value_s1 = 0.0
                    action_values.append(p * value_s1)

                state_value.append(sum(action_values))

            U1[s] = max(state_value)

            delta = max(delta, abs(U1[s] - U[s]))

        if delta < epsilon:
            return U1

def product_MDP_value_iteration(prod, LK_lst, dumb=False):
    """
    Run value iteration on a product mdp with LK list.

    dumb allows for a conservative search for V1, i.e.
    without assuming the robot always takes the best action.

    Uses value_iteration().
    """
    assert prod.states == set(prod.T.keys()), "states doesn't match T"

    L = [LK[0] for LK in LK_lst] # TODO add fields to MDP class
    K = [LK[1] for LK in LK_lst]
    print "\n* Searching for AMECs *\n"

    MECs = max_end_components(prod, show_steps=False)
    AMECs = []
    for MEC in MECs:
        got_L = False
        got_K = False
        states_set = MEC[0]
        for state in states_set:
            # (s, q) convention
            if type(state) is tuple:
                q = state[1]
            # <int> or <str> convention (produced by generate_product_MDP)
            else:
                q = state # i.e. <int> for Eric's simple example
            if q in L:
                got_L = True
            elif q in K:
                got_K = True
        if got_K and not got_L:
            AMECs.append(MEC)

    print "Found " + str(len(AMECs)) + " AMECs:"
    pprint(AMECs)
    assert len(AMECs) > 0, "No AMECs found."

    #print "L: " + str(L)
    #print "K: " + str(K)
    #pprint(AMECs)

    V0 = set()
    V1 = set()

    # 1-out all states in AMEC
    for AMEC in AMECs:
        states = AMEC[0]
        V1.update(states)

    # Save a copy of V1 to compare time-savings of graph search
    V1_safe = deepcopy(V1)

    ## GRAPH SEARCH BLOCK
    print "\n* Running a graph search. *\n"

    # # V0 Search
    def can_reach(mdp, B):
        """
        Returns set of states that can reach B.
        """
        # Initialize with B
        can_reach_B = deepcopy(B)
        changed = True
        while changed:
            changed = False
            shell_can_reach = deepcopy(can_reach_B)
            for x in can_reach_B:
                pre_s = mdp.pre_s(x)
                shell_can_reach.update(pre_s)
            if len(shell_can_reach) > len(can_reach_B):
                changed = True
                can_reach_B = deepcopy(shell_can_reach)
        return can_reach_B

    # Compute set of states that could possibly reach an AMEC
    can_reach_AMEC = can_reach(prod, V1)
    print "Found members of V0:"
    pprint(prod.states - can_reach_AMEC)
    V0.update(prod.states - can_reach_AMEC)

    # # V1 Search B+K pg. 859
    new_V1 = set()
    if not dumb:
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

    else: # dumb
        for s_o in prod.states:
            goes_to = set([s_o]) # Everywhere s_o can go.
            changed = True
            while changed:
                shell_goes = deepcopy(goes_to)
                changed = False
                for s in goes_to:
                    post_s = prod.post_s(s)
                    shell_goes.update(post_s)
                if len(shell_goes) > len(goes_to):
                    changed = True
                    goes_to = deepcopy(shell_goes)

            # If s_o does not ever go to any states in V0 => V1 
            if len(goes_to.intersection(V0)) == 0:
                new_V1.add(s_o)

    print "\nFound members of V1:"
    pprint(new_V1)
    V1.update(new_V1)
    ## END GRAPH SEARCH BLOCK

    print "\n* Running Value Iteration on Unknown States *\n"

    # Perform value iteration to finish off
    unknown_states = deepcopy(prod.states)
    unknown_states -= V1
    unknown_states -= V0

    print "unkown_states:"
    pprint(unknown_states)

    print "\nfor comparison, value_iteration _WITHOUT_ graph search finds: "
    pprint(value_iteration(prod, V1=V1_safe))

    print "\nvalue_iteration _WITH_ graph search finds: "
    final_dict = value_iteration(prod, V1=V1, unknown_states=unknown_states)
    pprint(final_dict)

    print "\nusing hashed V0 and V1, build up the final value dict:"
    for v1 in V1:
        final_dict[v1] = 1.0
    for v0 in V0:
        final_dict[v0] = 0.0
    pprint(final_dict)

    return final_dict

def __main__():
    prod = MDP()
    prod.states = set([0,1,2,3,4,5])
    prod.T = {0 : {'a':[(3,1)], 'b':[(0, .2), (1,.6), (3,.2)]},
              1 : {'a':[(2,1)]},
              2 : {'a':[(1,1)]},
              3 : {'a':[(3,1)]},
              4 : {'a':[(3,1)]},
              5 : {'a':[(2,1)]}}
    pprint(product_MDP_value_iteration(prod, [(None, 2)], dumb=False))

if __name__ == "__main__":
    __main__()







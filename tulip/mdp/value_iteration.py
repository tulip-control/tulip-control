# First cut of value iteration for a product MDP.
# Notes: [(L0, K0), ...] is list of tuples (bad, good).
# Dependencies: networkx for graph search, jbern_MDP_overhaul for MDP methods.
# Originally: James Bern 8/6/2013

from copy import deepcopy
from pprint import pprint
import networkx
from transys import *
from jbern_MDP_overhaul import *


"""
# More complicated example
print "\n\n** More Complicated Example (uses generate_product_MDP()) **"
mdp = mdp_example()
pprint(mdp.T)
print

spec = "G F s2"
LK_lst = [(None, 'q1')]
L = [LK[0] for LK in LK_lst]
K = [LK[1] for LK in LK_lst]

ra = RabinAutomaton()
ra.states.add_from(['q0', 'q1'])
ra.states.add_initial('q0')
alphabet = ['s2', '!s2']
ra.alphabet.add_from(alphabet)
ra.transitions.add_labeled('q0', 'q0', '!s2')
ra.transitions.add_labeled('q0', 'q1', 's2')
ra.transitions.add_labeled('q1', 'q0', '!s2')
ra.transitions.add_labeled('q1', 'q1', 's2')

prod = generate_product_MDP(mdp, ra)
#pprint(prod.T)
print prod.states
print
#"""


#'''
# Eric's simple example
print "\n\n** Eric's Simple Example **"
LK_lst = [(None, 2)]
L = [LK[0] for LK in LK_lst]
K = [LK[1] for LK in LK_lst]

prod = MDP()
prod.T = {0 : {'a':[(3,1)], 'b':[(0, .2), (1,.6),(3,.2)]},
          1 : {'a':[(2,1)]},
          2 : {'a':[(1,1)]},
          3 : {'a':[(3,1)]},
          5 : {'a':[(2,1)]}}
prod.states = set(prod.T.keys())
prod.init_states = [0] # Not used.
#'''


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

#print "L: " + str(L)
#print "K: " + str(K)
#pprint(AMECs)

# V[s] = <float>
V = {s : None for s in prod.states}
V0 = set()
V1 = set()

# 1-out all states in AMEC
for AMEC in AMECs:
    states = AMEC[0]
    for state in states:
        V[state] = 1.0
        V1.add(state)

# Save a copy of V1 and R so to compare time-savings of graph search
V1_safe = deepcopy(V1)
R_safe = {}
for v in prod.states:
    if v in V1_safe:
        R_safe[v] = 1.0
    else:
        R_safe[v] = 0.0

#''' GRAPH SEARCH BLOCK
# TODO strengthen conditions for V0 and V1 (will require substantial rewrtiting)
#
# Do a networkx graph search to save time with:
# -Determine states that must reach V1 and states that must reach V0
# -(i.e. never reach V1) and add to V1 and V0 resp.
successors = {}
G = prod.induce_digraph(prod.states, prod.T) # TODO mdp.graph_representation()
print
print "* Running a graph search. *"
print
print "NetworkX successor dict:"
pprint({state : networkx.dfs_successors(G, state) for state in (prod.states - V1)})
print

changed = False
while not changed:
    changed = False
    # Note all have prior condition not part of an AMEC.
    for state in (prod.states - V1):
        successors = networkx.dfs_successors(G, state)

        # V0 Condition (a): no successors
        # -This is really a side-effect of how NetworkX outputs dfs_successors().
        if successors == {}:
            print "Found a member of V0: " + str(state)
            V[state] = 0.0
            V0.add(state)
            changed = True

        else:
            # Strip out set of successors
            successor_lst = []
            for s_val in successors.values():
                successor_lst += s_val
            successor_set = set(successor_lst)

            # V0 Condition (b): successors are a subset of VO
            if successor_set.issubset(V0):
                print "Found a member of V0: " + str(state)
                V[state] = 0.0
                V0.add(state)
                changed = True

            # V1 Condition: successors are a subset of V1
            elif successor_set.issubset(V1):
                print "Found a member of V1: " + str(state)
                V[state] = 1.0
                V1.add(state)
                changed = True
print
print "* Completed graph search. *"
print
#END TIME SAVING BLOCK''' 

# Perform value iteration to finish off
# -First initialize rewards dict.
R = {}
for s in prod.states:
    if s in V1:
        R[s] = 1.0
    else: # in V0 => 0 or not in (V0 U V1) => init. @ 0
        R[s] = 0.0

#print "V:\n"
#pprint(V)
#print "\nV1:"
#pprint(V1)
#print "V0:"
#pprint(V0)

unknown_states = deepcopy(prod.states)
unknown_states -= V1
unknown_states -= V0

print "value_iteration _WITHOUT_ time saving returns: "
pprint(value_iteration(prod, R=R_safe, V1=V1_safe)) # Note use of both old R, old V1.
print
print "value_iteration _WITH_ time saving returns: "
final_dict = value_iteration(prod, R=R, V1=V1, unknown_states=unknown_states)
pprint(final_dict)
print
print "from either one we can generate the final value dict:"
# Add the time saving version to V1 and V0
for v1 in V1:
    final_dict[v1] = 1.0
for v0 in V0:
    final_dict[v0] = 0.0
pprint(final_dict)





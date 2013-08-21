# Originally: James Bern 8/17/2013
#
# 8/20: Policies complete.

from jbern_uncertain_MDP_overhaul import *
from jbern_MDP_functions import *
from value_iteration_function import *
from transys import RabinAutomaton # FORNOW TODO make current
from itertools import cycle
import networkx

likely = IntervalProbability(.7, .9)
unlikely = IntervalProbability(.1, .3)

def gen_uncertain_grid_T(num_rows, num_cols, name='s'):
    """
    Helper function that generates a canonical grid world.
    Could be easily extended if need arises.

    names is a list of lists of states (matrix of state names)
    """
    names = []
    num_rows = num_rows
    num_cols = num_cols
    for r in range(num_rows):
        row = []
        for c in range(num_cols):
            i = r*num_cols + c
            row.append('s' + str(i))
        names.append(row)
    print names

    # Attempt to grab the spec'd adj. name, if not valid coord's,
    # -default to returning argument name.
    def grab(name, drow=0, dcol=0):
        for search_row in range(num_rows):
            for search_col in range(num_cols):
                if names[search_row][search_col] == name:
                    row = search_row
                    col = search_col
        ROW = row + drow
        COL = col + dcol
        if ROW in range(num_rows) and COL in range(num_cols):
            return names[ROW][COL]
        else:
            return name

    # Build up the T dictionary
    T = {}
    for row in range(num_rows):
        for col in range(num_cols):
            name = names[row][col]
            # 'u'
            u = grab(name, drow=-1)
            # 'r'
            r = grab(name, dcol=1)
            # 'd'
            d = grab(name, drow=1)
            # 'l'
            l = grab(name, dcol=-1)

            T[name] = {'u' : [(u, likely), (r, unlikely), (l, unlikely)],
                       'r' : [(r, likely), (u, unlikely), (d, unlikely)],
                       'd' : [(d, likely), (r, unlikely), (l, unlikely)],
                       'l' : [(l, likely), (u, unlikely), (d, unlikely)]}
    return T

# Define a 3 x 3 MDP w/ canonical (.8, .1, .1) moves:
# s0 s1 s2
# s3 s4 s5
# s6 s7 s8
mdp = UncertainMDP(name="uncertain_example")
mdp.T = gen_uncertain_grid_T(3, 3)
for state in mdp.T.keys():
    mdp.states.add(state)
# Add a trap state that catches:
# above s0
# left of s0
# left of s3
#
# Add in trap state t0
# Trap is a loop to avoid violation of non-blocking assumption
# -NOTE no actions would trigger assertion
mdp.states.add('t0')
mdp.T['t0'] = {"trapped" : [('t0', IntervalProbability.one())]}
# Modify s0
mdp.T['s0']['u'] = [('t0', likely), ('t0', unlikely), ('s1', unlikely)]
mdp.T['s0']['r'] = [('t0', unlikely), ('s1', likely), ('s3', unlikely)]
mdp.T['s0']['d'] = [('t0', unlikely), ('s3', likely), ('s1', unlikely)]
mdp.T['s0']['l'] = [('t0', likely), ('t0', unlikely), ('s1', unlikely)]
# Modify s3
mdp.T['s3']['u'] = [('t0', unlikely), ('s0', likely), ('s4', unlikely)]
mdp.T['s3']['d'] = [('t0', unlikely), ('s6', likely), ('s4', unlikely)]
mdp.T['s3']['l'] = [('t0', likely), ('s0', unlikely), ('s6', unlikely)]
# Condense the MDP down so we can run a sanity check.
mdp.condense_split_dynamics()
mdp.sanity_check()
pprint(mdp.T)

# Define Rabin Automaton for spe
spec = "& G F s2 G F s7"
ra = RabinAutomaton()
# TODO have all this read in from ltl2dstar
ra.states.add_from(['q' + str(i) for i in range(5)])
ra.states.add_initial('q0')
#
# TODO custom check function that checks for conjunction (+/- !).
#
ra.transitions.add_labeled('q0', 'q0', '!s2&!s7', check=False)
ra.transitions.add_labeled('q0', 'q1', 's2&!s7', check=False)
ra.transitions.add_labeled('q0', 'q0', '!s2&s7', check=False)
ra.transitions.add_labeled('q0', 'q4', 's2&s7', check=False)
#
ra.transitions.add_labeled('q1', 'q1', '!s2&!s7', check=False)
ra.transitions.add_labeled('q1', 'q1', 's2&!s7', check=False)
ra.transitions.add_labeled('q1', 'q2', '!s2&s7', check=False)
ra.transitions.add_labeled('q1', 'q4', 's2&s7', check=False)
#
ra.transitions.add_labeled('q2', 'q2', '!s2&!s7', check=False)
ra.transitions.add_labeled('q2', 'q3', 's2&!s7', check=False)
ra.transitions.add_labeled('q2', 'q2', '!s2&s7', check=False)
ra.transitions.add_labeled('q2', 'q4', 's2&s7', check=False)
#
ra.transitions.add_labeled('q3', 'q1', '!s2&!s7', check=False)
ra.transitions.add_labeled('q3', 'q1', 's2&!s7', check=False)
ra.transitions.add_labeled('q3', 'q2', '!s2&s7', check=False)
ra.transitions.add_labeled('q3', 'q4', 's2&s7', check=False)
#
ra.transitions.add_labeled('q4', 'q2', '!s2&!s7', check=False)
ra.transitions.add_labeled('q4', 'q1', 's2&!s7', check=False)
ra.transitions.add_labeled('q4', 'q2', '!s2&s7', check=False)
ra.transitions.add_labeled('q4', 'q4', 's2&s7', check=False)
#
# list form [(L0, K0), (L1, K1), ...], L bad K good.
LK_lst = [(None, 'q3'), (None, 'q4')]

# Generate the product.
prod = generate_product_MDP(mdp, ra, check_deterministic=True)
pprint(prod.T)

# Two policies: (1) min-path to reach AMEC (Wongpiromsarn)
# --------------(2) round robin to visit all states in AMEC
# https://530cb0a7-a-62cb3a1a-s-sites.googlegroups.com/site/tichakorn/iros12_inc.pdf
#
# Prelim.'s for (1)
AMECs = accepting_max_end_components(prod, LK_lst)
ssp_prod, Sr, S0, B, ssp_to_original = gen_stochastic_shortest_path(prod, AMECs)

# assert states in AMECs (set of tup of state, action dict)
# -and B (set of states) assert be identical
#check_states = set()
#for AMEC in AMECs:
#    check_states.update(AMEC[0])
#assert check_states == B

# Generate V_ssp, and backtrack to V_orig using mapping returned by
# -gen_stochastic_shortest_path()
V_ssp = product_MDP_value_iteration(ssp_prod, AMECs)
V_orig = {}
for s_ssp in ssp_to_original.keys():
    original_states = ssp_to_original[s_ssp]
    for s_orig in original_states:
        V_orig[s_orig] = V_ssp[s_ssp]

# Part One: Wongpiromsarn
# Helper function.
def gen_actMax(prod, V_orig, Sr):
    """
    A modified version of gen_best_actions_dict that handles a
    precomputed V and IntervalProbability probabilities.
    """
    actMax = {}
    for s_i in Sr:
        actions = prod.actions(s_i)

        # Find best actions from a state using MDP probabilities.
        best_actions = set()
        action_values = {}
        for a in actions:
            P = []
            V = []
            for tup in prod.T[s_i][a]:
                s_f, p = tup
                assert isinstance(p, IntervalProbability), "unsupported p"
                P.append(p)
                V.append(V_orig[s_f])
            P = P_V_linear_program(P, V)
            V = matrix(V)
            action_values[a] = sum(mul(P, V))
        max_value = max(action_values.values())
        for a in action_values.keys():
            if action_values[a] == max_value:
                best_actions.add(a)
        actMax[s_i] = best_actions
    return actMax
#
# Generate digraph that only allows actions in act_Max.
# In this case it does not matter that the graph representation loses
# -the information of some destination states are "tied together" by
# -by their action.
actMax = gen_actMax(prod, V_orig, Sr)
G = prod.induce_nx_digraph(states=prod.states, actions={s : actMax[s] for s in Sr})
#
# Generate dictionary of min number of moves required to reach an AMEC.
AMEC_distances = {s : -1 for s in Sr}
for s in Sr:
    min_lst = []
    for b in B:
        distance = -1 + len(networkx.shortest_path(G, source=s, target=b))
        min_lst.append(distance)
    AMEC_distances[s] = min(min_lst)
#
pprint(AMEC_distances)
print
#
# Compute mu_policy for (1).
# -Rule: take an action that has destination that has shorter distance
mu_policy = {}
for s_i in Sr:
    init_distance = AMEC_distances[s_i]
    actions = actMax[s_i]

    # Unambiguous s_i: trivial case. 
    if len(actions) == 1:
        mu_policy[s_i] = deepcopy(actions).pop()
        destinations = prod.destinations(s_i, a)
        continue
    # For ambiguous s_i: find an action that meets the Rule.
    for a in actions:
        destinations = prod.destinations(s_i, a)
        for s_f in destinations:
            # Case 1: s_f in AMEC
            if s_f in B:
                mu_policy[s_i] = a
                break
            # Case 2: s_f in S0
            if s_f not in B.union(Sr):
                continue
            # Case 3: s_f in Sr
            # => Check: final_distance < init_distance
            final_distance = AMEC_distances[s_f]
            if final_distance < init_distance:
                assert init_distance - final_distance == 1
                mu_policy[s_i] = a
#
pprint(mu_policy)

# Prelim.'s for (2)
def determine_AMEC(AMECs, s_in_AMEC):
    for AMEC in AMECs:
        if s_in_AMEC in AMEC[0]:
            return AMEC
    raise Exception, "s_in_AMEC is not in an AMEC."

# Part Two: round-robin to visit all states in AMEC.
def gen_round_robin_dict(AMECs, AMEC_state):
    round_robin_dict = {}
    our_AMEC = determine_AMEC(AMECs, AMEC_state)
    our_AMEC_states = our_AMEC[0]
    for state in our_AMEC_states:
        allowed_actions = our_AMEC[1][state]
        # Use itertools cycles for convenience and reliability (TODO check overhead).
        round_robin_dict[state] = cycle(allowed_actions)
    return round_robin_dict
#
def next_action_from(round_robin_dict, state):
    """
    Returns the next action from a state in a round_robin_dict.
    """
    return round_robin_dict[state].next()
#
round_robin_dict = gen_round_robin_dict(AMECs, AMEC_state=('s1','q1'))
print
pprint(round_robin_dict)


# MDP example for TuLiP, demonstrating: product-generation and simple
# ---------------------------------------a simple two-pronged policty
# Originally: James Bern 8/6/2013
#
# 8/9: Rewrote gen_grid_T to take num_rows and num_cols
# 8/9: Tested example for 15x15 MDP x 4-state Rabin => ~1000 state product
# 8/12: First cut of two-pronged policy.
# 8/17: (FIX) Error in gen_round_robin_dict corrected
# -------AMEC (garbage) -> our_AMEC
# 8/21: Simulation added.

from jbern_MDP_overhaul import *
from jbern_MDP_functions import *
from value_iteration_function import product_MDP_value_iteration
from transys import RabinAutomaton # FORNOW TODO make current w/ TuLiP

from itertools import cycle

from pprint import pprint

import random

def gen_grid_T(num_rows, num_cols, name='s'):
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
    print(names)

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

            T[name] = {'u' : [(u, .8), (r, .1), (l, .1)],
                       'r' : [(r, .8), (u, .1), (d, .1)],
                       'd' : [(d, .8), (r, .1), (l, .1)],
                       'l' : [(l, .8), (u, .1), (d, .1)]}
    return T

# Define a 3 x 3 MDP w/ canonical (.8, .1, .1) moves:
# s0 s1 s2
# s3 s4 s5
# s6 s7 s8
mdp = MDP(name="video_mdp")
mdp.T = gen_grid_T(3, 3)
for state in mdp.T.keys():
    mdp.states.add(state)
mdp.initial_states = set(['s0'])
# Add a trap state that catches:
# above s0
# left of s0
# left of s3
#
# Add in trap state t0
# Trap is a loop to avoid violation of non-blocking assumption
# -NOTE no actions would trigger assertion
mdp.states.add('t0')
mdp.T['t0'] = {"loop" : [('t0', 1)]}
# Modify s0
mdp.T['s0']['u'] = [('t0', .9), ('s1', .1)]
mdp.T['s0']['r'] = [('t0', .1), ('s1', .8), ('s3', .1)]
mdp.T['s0']['d'] = [('t0', .1), ('s3', .8), ('s1', .1)]
mdp.T['s0']['l'] = [('t0', .9), ('s1', .1)]
# Modify s3
mdp.T['s3']['u'] = [('t0', .1), ('s0', .8), ('s4', .1)]
mdp.T['s3']['d'] = [('t0', .1), ('s6', .8), ('s4', .1)]
mdp.T['s3']['l'] = [('t0', .8), ('s0', .1), ('s6', .1)]
# Condense the MDP down so we can run a sanity check.
mdp.condense_split_dynamics()
mdp.sanity_check()

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

# Two policies: (1) deterministic discounted value iteration to reach AMEC
# --------------(2) round robin to visit all states in AMEC

# Prelim.'s for (1)
AMECs = accepting_max_end_components(prod, LK_lst)
nonAMEC_states = set()
AMEC_states = set()
for AMEC in AMECs:
    AMEC_states.update(AMEC[0])
nonAMEC_states = prod.states.difference(AMEC_states)

# Prelim.'s for (2)
def determine_AMEC(AMECs, s_in_AMEC):
    for AMEC in AMECs:
        if s_in_AMEC in AMEC[0]:
            return AMEC
    raise Exception, "s_in_AMEC is not in an AMEC."

# Part One: memoryless goto_AMEC policy done with discounting.
V_for_memoryless = product_MDP_value_iteration(prod, AMECs, gamma=.9)
pprint(V_for_memoryless)
mu_policy = gen_best_actions_dict(prod, V_for_memoryless, states=nonAMEC_states)
pprint(mu_policy)
#
# # Demonstration of memoryless policy.
print("\npolicy one:")
pprint(mu_policy)

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
# # Demonstration of a state exhibiting the desired cyclic behavior.
round_robin_dict = gen_round_robin_dict(AMECs, AMEC_state=('s1','q1'))
print("\npolicy two:")
pprint(round_robin_dict)
print
for i in range(10):
    for state in round_robin_dict.keys():
        print("state: " + str(state) )
        print("\taction: " + str(next_action_from(round_robin_dict, state)) )
        break

# Run Simulation.
NUM_STEPS = 20
init_sq = deepcopy(prod.initial_states).pop()
round_robin_dict = None
generated_round_robin_dict = False
for i in xrange(NUM_STEPS):
    print("\nSimulation step: {}".format(i) )
    print("Init sq: {}".format(init_sq) )
    # Choose which policy to use.
    if init_sq not in AMEC_states:
        print("Operating under policy 1, with action:")
        #FORNOW
        action = deepcopy(mu_policy[init_sq]).pop()
    else:
        # Only generate the round_robin_dict once, and using approp.
        # -AMEC_state.
        if not generated_round_robin_dict:
            round_robin_dict = gen_round_robin_dict(AMECs, AMEC_state=init_sq)
            generated_round_robin_dict = True
        print("Operating under policy 2, with action: ")
        action = next_action_from(round_robin_dict, init_sq)
    print action
    # Random'ly determine final_s (in the original MDP).
    # (This block would be replaced by 
    # -the random motion modeled by the MDP.)
    possible_final_s = [tup[0][0] for tup in prod.T[init_sq][action]] #SUGAR
    print("potential next s: {}".format(possible_final_s) )
    random_draw = random.random()
    cumm_prob = 0.0
    for sq_p_tup in prod.T[init_sq][action]:
        sq_tup, p = sq_p_tup
        cumm_prob += p
        if random_draw < cumm_prob:
            final_s = sq_tup[0]
            break
    print("Next s: {}".format(final_s) )
    # Backtrack out the final_sq from the product MDP.
    # (This block would be replaced by e.g. IR LED query of state.)
    for sq_p_tup in prod.T[init_sq][action]:
        sq_tup = sq_p_tup[0]
        if final_s == sq_tup[0]:
            final_sq = sq_tup
    # Update the init_sq
    init_sq = final_sq


















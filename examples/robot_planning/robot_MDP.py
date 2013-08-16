# MDP example for TuLiP, demonstrating: product-generation and simple
# ---------------------------------------a simple two-pronged policty
# Dependencies: jbern_MDP_overhaul, value_iteration_function
# Originally: James Bern 8/6/2013
#
# 8/9: Rewrote gen_grid_T to take num_rows and num_cols
# 8/9: Tested example for 15x15 MDP x 4-state Rabin => ~1000 state product
# 8/12: First cut of two-pronged policy.

from jbern_MDP_overhaul import *
from value_iteration_function import product_MDP_value_iteration
from itertools import cycle

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

# Two iteration schemes: memoryless discounting/min-path (to reach AMEC), and
# -----------------------probabilistic/round robin (to visit all states in AMEC).

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
memoryless_policy = gen_best_action_dict(prod, V_for_memoryless, states=nonAMEC_states)
#
# # Demonstration of memoryless policy.
print
pprint(memoryless_policy)

# Part Two: round-robin to visit all states in AMEC.
def gen_round_robin_dict(AMECs, AMEC_state):
    round_robin_dict = {}
    our_AMEC = determine_AMEC(AMECs, AMEC_state)
    our_AMEC_states = our_AMEC[0]
    for state in our_AMEC_states:
        allowed_actions = AMEC[1][state]
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
print
for i in range(10):
    for state in round_robin_dict.keys():
        print "state: " + str(state)
        print "\taction: " + str(next_action_from(round_robin_dict, state))
        break



















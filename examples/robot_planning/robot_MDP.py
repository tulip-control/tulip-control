from jbern_MDP_overhaul import *
from value_iteration_function import product_MDP_value_iteration

def gen_grid_T(names):
    """
    Helper function that generates a canonical grid world.
    Could be easily extended if need arises.

    names is a list of lists of states (matrix of state names)
    """
    num_rows = len(names)
    num_cols = len(names[0])
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
mdp.states.update(['s' + str(i) for i in range(9)])
names = [['s0', 's1', 's2'], ['s3', 's4', 's5'], ['s6', 's7', 's8']]
mdp.T = gen_grid_T(names)
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

# Define Rabin Automaton for spec
spec = "& G F s2 G F s7"
ra = RabinAutomaton()
# TODO have all this read in from ltl2dstar
ra.states.add_from(['q' + str(i) for i in range(5)])
ra.states.add_initial('q0')

# TODO custom check function that checks for conjunction (+/- !).

ra.transitions.add_labeled('q0', 'q0', '!s2&!s7', check=False)
ra.transitions.add_labeled('q0', 'q1', 's2&!s7', check=False)
ra.transitions.add_labeled('q0', 'q0', '!s2&s7', check=False)
ra.transitions.add_labeled('q0', 'q4', 's2&s7', check=False)

ra.transitions.add_labeled('q1', 'q1', '!s2&!s7', check=False)
ra.transitions.add_labeled('q1', 'q1', 's2&!s7', check=False)
ra.transitions.add_labeled('q1', 'q2', '!s2&s7', check=False)
ra.transitions.add_labeled('q1', 'q4', 's2&s7', check=False)

ra.transitions.add_labeled('q2', 'q2', '!s2&!s7', check=False)
ra.transitions.add_labeled('q2', 'q3', 's2&!s7', check=False)
ra.transitions.add_labeled('q2', 'q2', '!s2&s7', check=False)
ra.transitions.add_labeled('q2', 'q4', 's2&s7', check=False)

ra.transitions.add_labeled('q3', 'q1', '!s2&!s7', check=False)
ra.transitions.add_labeled('q3', 'q1', 's2&!s7', check=False)
ra.transitions.add_labeled('q3', 'q2', '!s2&s7', check=False)
ra.transitions.add_labeled('q3', 'q4', 's2&s7', check=False)

ra.transitions.add_labeled('q4', 'q2', '!s2&!s7', check=False)
ra.transitions.add_labeled('q4', 'q1', 's2&!s7', check=False)
ra.transitions.add_labeled('q4', 'q2', '!s2&s7', check=False)
ra.transitions.add_labeled('q4', 'q4', 's2&s7', check=False)

# list form [(L0, K0), (L1, K1), ...], L bad K good.
LK_lst = [(None, 'q3'), (None, 'q4')]

prod = generate_product_MDP(mdp, ra, check_deterministic=True)

V = product_MDP_value_iteration(prod, LK_lst)

# TODO two iteration schemes, discounting/min-path (to reach AMEC), and
# ------probabilistic/round robin (to visit all states in AMEC)





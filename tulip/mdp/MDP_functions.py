# To hold all accessory functions.  Note: functions should work for both
# MDPs and UncertainMDPs.

# Originally: James Bern 8/17/2013
# jbern@caltech.edu
#
# 8/17 gen_best_action_dict now returns all best actions (i.e. returns
#--all equally best actions if they exist.

from transys_old import RabinAutomaton # TODO make current w/ TuLiP
from MDP import *
from uncertain_MDP import *
from probability_types import *

import networkx

from copy import deepcopy
from pprint import pprint

############################################################################
# Generate Product MDP
############################################################################

def generate_product_MDP(mdp, ra, check_deterministic=False):
    """
    Generate a product MDP from self (MDP) and a DRA.
    """
    #assert type(ra) == RabinAutomaton
    #TODO every state in prod has one action (non-blocking)
    def satisfies(ra, q, sp, qp, s=None):
        """
        Check transition label against state label of
        destination state.
        Assumes a conjunctive label of form "!s2&t0&s4".
        NOTE currently only one letter is allowed, i.e. trap0 will not work.
        Should be swapped out/generalized as needed.
        """
        def gen_lst_form(label):
            """
            Helper function: Throw label into a list and trim out "s"'s.
            """
            label = label.split("&")
            lst_form = []
            for sub in label:
                if sub[0] == "!":
                    sub = sub[0] + sub[2:]
                else:
                    sub = sub[1:]
                lst_form.append(sub)
            return lst_form

        # Iterate through all of the RA's transitions,
        # -looking for one that satisfies.
        return_bool = False
        # Way of making sure that there is <= 1 satisfying transition.
        num_satisfying_transitions = 0
        for transition in ra.transitions():
            q_i = transition[0]
            q_f = transition[1]
            # Not a potentially satisfying transition.
            # -(wrong initial or wrong final)
            if q_i != q or q_f != qp:
                continue
            # ELSE
            # We've now hit a potentially satisfying transition.
            # -(right initial and final state)
            satisfies = True
            L = gen_lst_form(transition[2]["in_alphabet"])

            sp_int = int(sp[1:])
            for sub in L:
                if sub[0] == "!":
                    if sp_int == int(sub[1:]):
                        satisfies = False
                else:
                    if sp_int != int(sub):
                        satisfies = False

            # TODO Match overall system.
            DEBUGGING = False
            if DEBUGGING:
                print str((s, q))
                print str((sp, qp))
                print L
                print satisfies
                print ''

            # Good Check To Have:
            # assert not found_satisfying, "There exists more than one valid transition"
            if satisfies:
                return_bool = True
                num_satisfying_transitions += 1
            # Since, NOTE another transition may __potentailly_ satisfy
            else:
                continue
        assert num_satisfying_transitions <= 1, "Found >1 satisfying transitions"
        return return_bool

    if mdp.probability_type == float:
        prod = MDP()
    elif mdp.probability_type == IntervalProbability:
        prod = UncertainMDP()
    else:
        raise NotImplementedError

    prod.initial_states = set()
    prod.states = set() # form: [(s0, q0), ...]
    prod.T = {}
    for s in mdp.states:
        for q in ra.states():
            prod.states.add((s, q))
            prod.T[(s,q)] = {}

    # Product initial states.
    for s in mdp.initial_states:
        for q_0 in ra.states.initial:
            for q in ra.states():
                if satisfies(ra=ra, qp=q, q=q_0, sp=s):
                    prod.initial_states.add((s, q))

    # Product transitions
    for s_i_q_i_tup in prod.states:
        # Grab every state in S x Q, call s_i, q_i
        s_i, q_i = s_i_q_i_tup
        actions = mdp.T[s_i].keys()
        # Consider all actions a for s_i
        for a in actions:
            for s_f_p_tup in mdp.T[s_i][a]:
                # Consider all possible s_f for each action a
                s_f, p = s_f_p_tup
                # Consider all s_f x Q
                for q_f in ra.states():
                    # We now have qp, q, sp as on 881
                    if satisfies(ra=ra, qp=q_f, q=q_i, sp=s_f): # Why work either s_i, s_f?
                        prod_p = p
                        if a not in prod.T[(s_i, q_i)].keys():
                            prod.T[(s_i, q_i)][a] = []
                        prod.T[(s_i, q_i)][a].append(((s_f, q_f), prod_p))

    # Product labelling.
    pass

    # TODO check rule with Eric.
    # Run simple check of determinism of product:
    # Check that there are no repeated s's in a dest_dict
    if check_deterministic:
        for action_dict in prod.T.values():
            for dest_dict in action_dict.values():
                check_lst = []
                for tup in dest_dict:
                    dest = tup[0]
                    s = dest[0]
                    check_lst.append(s)
            # (Uniqueness condition)
            assert len(set(check_lst)) == len(check_lst), \
                "Did you condense? D" + str(dest_dict) + " " + str(check_lst)
        print "Determinism checks."

    return prod


def generate_product_MDP_example():
    """
    This is a trivial MDP cast from B+K pg.202 Traffic Light Example
    Notation is consistent w/ B+K pg.881, where sp corresponds to s'.
    """
    # Set up MDP.
    # (Note that this is actually a Markov Chain.)
    mdp = MDP()
    mdp.states = ['s0', 's1']
    mdp.initial_states = ['s0'] # FORNOW TODO use States object
    mdp.T = {}
    mdp.T['s0'] = {'f' : [('s1', 1)]}
    mdp.T['s1'] = {'b' : [('s0', 1)]}

    # RabinAutomaton for the spec.
    # (Note that this is actually a Buchi automaton.)
    ra = RabinAutomaton()
    ra.states.add_from(['q0', 'q1', 'q2'])
    ra.states.add_initial('q0')

    alphabet = ['s0', 's1', 's2', 's3', '!s0', '!s1', '!s2', '!s3', '!s-1']
    ra.alphabet.add_from(alphabet)
    ra.transitions.add_labeled('q0', 'q0', '!s-1')
    ra.transitions.add_labeled('q0', 'q1', '!s1')
    ra.transitions.add_labeled('q1', 'q1', '!s1')
    ra.transitions.add_labeled('q1', 'q2', 's1')
    ra.transitions.add_labeled('q2', 'q2', '!s-1')
    #pprint(ra.transitions())

    prod = generate_product_MDP(mdp, ra)
    return prod

############################################################################
# Max End Components
############################################################################

def max_end_components(MDP, show_steps=False):
    """
    Compute and return the maximal end components of an MDP, or product MDP.
    See: pg.878 B+K
    Note: uses a call to networkx.strongly_connected_components()
    """
    show_steps = show_steps

    m = MDP.T
    A = {} # form: A[s][actions]
    for s in m.keys(): # TODO States
        A[s] = m[s].keys()[:]
    MEC = set([])
    MEC_new = [set(m.keys()[:])]

    while MEC != MEC_new:
        MEC = MEC_new[:]
        MEC_new = []
        for T in MEC:
            R = set() # set of states to be removed

            # Compute nontrivial SCCs T1, ..., Tk of digraph G(T,A_T)
            A_T = {t : set(A[t]) for t in T}

            G = MDP.induce_digraph(states=T, actions=A_T)
            SCCs = networkx.strongly_connected_components(G)

            # Remove trivial SCCs
            keep_SCCs = []
            for SCC in SCCs:
                if not (len(SCC) == 1 and len(A[SCC[0]]) == 0):
                        keep_SCCs.append(SCC)
                elif show_steps:
                    print "removing trivial SCC " + str(SCC)
            SCCs = keep_SCCs

            for T_i in SCCs:
                for s in T_i:
                    keep_actions = []
                    for a in A[s]:
                        destinations = [dest_prob_tup[0] for dest_prob_tup in m[s][a]]
                        if set(destinations).issubset(set(T_i)):
                            keep_actions.append(a)
                        elif show_steps:
                            print "removing action " + str(a)
                    A[s] = keep_actions
                    if len(A[s]) == 0:
                        if show_steps:
                            print "removing state " + str(s)
                        R.add(s)

            while (len(R) > 0):
                # remove s from R and from T
                s = R.pop()
                T.remove(s)
                for t_b_tup in MDP.pre_s_a(s):
                    t, b = t_b_tup
                    # Restrict to t part of T
                    if t not in T:
                        continue
                    if b in A[t]:
                        if show_steps:
                            print "removing action " + str(b)
                        A[t].remove(b)
                    if len(A[t]) == 0:
                        if show_steps:
                            print "removing state " + str(t)
                        R.add(t)

            for T_i in SCCs:
                if len(set(T).intersection(set(T_i))) > 0:
                    MEC_new.append( (set(T) & set(T_i)) )

    # Return block.
    output = []
    for T in MEC:
        A_T = {t: set(A[t]) for t in T}
        output.append((T, A_T))
    return output


def max_end_components_example():
    """
    Simple max-end-component example which can be checked
    by hand.
    """
    m = MDP()

    m.T = {}
    m.states = set(['s1','s2','s3','s4'])
    for state in m.states:
        m.T[state] = {}
    m.T['s1']['none1'] = [('s2', .5), ('s4', .5)]
    m.T['s1']['1/3'] = [('s1', .2), ('s3', .8)]
    m.T['s2']['none2'] = [('s1', 1)]
    m.T['s3']['1/3'] = [('s1', 1)]
    m.T['s2']['2'] = [('s2', 1)]
    m.T['s4']['4'] = [('s4', 1)]
    m.sanity_check()

    return max_end_components(m)

############################################################################
# Accepting Max End Components
############################################################################

def accepting_max_end_components(prod, LK_lst, allow_no_AMECs=False):
    """
    Compute accepting max end components given L, K.
    """
    L = [LK[0] for LK in LK_lst] # TODO add fields to MDP class
    K = [LK[1] for LK in LK_lst]

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
        if got_L:
            print "Warning: called accepting_max_end_components with " + \
                    "L states.  Due to nature of Rabin, winning set " + \
                    "may not be an AMEC!"

    if len(AMECs) == 0:
        if not allow_no_AMECs:
            raise Exception, "No AMECs found."
        print "Warning: accepting_max_end_components returning None"
        return None
    return AMECs


def accepting_max_end_components_example():
    prod = MDP()
    prod.states = set([1,2,3,4])
    prod.T[1] = {'2/3' : [(2,.5), (3,.5)], '4' : [(4,1)]}
    prod.T[2] = {'1' : [(1,1)], '4' : [(4,1)]}
    prod.T[3] = {'1' : [(1,1)], '4' : [(4,1)]}
    prod.T[4] = {'4' : [(4,1)]}
    prod.sanity_check()
    return [accepting_max_end_components(prod, [(None, 4)]),
            accepting_max_end_components(prod, [(None, 1)]),
            accepting_max_end_components(prod, [(None, 1), (None, 4)]),
            accepting_max_end_components(prod, [(2, 1)], allow_no_AMECs=True)]

############################################################################
# Generate SSP MDP
############################################################################

def gen_stochastic_shortest_path(prod, AMECs):
    """
    Generate an MDP that obeys ssp assumptions.
    See: cds.caltech.edu/~ewolff/wolff_cdc_12.pdf
    """

    # Algorithm 2: end component elimination

    # S: set of all states
    S = prod.states

    # B: union of all AMECs
    B = set()
    for AMEC in AMECs:
        B.update(AMEC[0])

    # S0: states that cannot reach B
    can_reach_B = prod.can_reach(B)
    S0 = S.difference(can_reach_B)

    # Sr: remainder
    Sr = S.difference(B.union(S0))

    # Max end components in Sr:
    reach_MDP = MDP()
    # (a1) create Sr MDP
    for s in Sr:
        reach_MDP.states.add(s)
        reach_MDP.T[s] = deepcopy(prod.T[s])
    # (a2) prune out actions that have destinations not in Sr
    for s in Sr:
        for a in reach_MDP.T[s].keys():
            if not set(reach_MDP.destinations(s, a)).issubset(Sr):
                del reach_MDP.T[s][a]
    # (b) compute maximal end components
    reach_MECs = max_end_components(reach_MDP)

    ## Swap Sr MECs for new states:
    # (a1) determine sets of states
    new_states = set(['h' + str(i) for i in xrange(len(reach_MECs))])
    reach_MEC_states = set()
    for MEC in reach_MECs:
        reach_MEC_states.update(MEC[0])
    # (a2) create S_hat and Sr_hat
    S_hat = S.union(new_states) - reach_MEC_states
    # Sr_hat = Sr.union(new_states) - reach_MEC_states # XXX

    ## Create A_hat dict which holds (s, a) pairs:
    A_hat = {}
    # (1) S - reach_MEC_states
    for s in S.difference(reach_MEC_states):
        pairs = [(s, a) for a in prod.actions(s)]
        A_hat[s] = set(pairs)
    # (2) new_states
    reach_MEC_copy = deepcopy(reach_MECs)
    reach_MEC_dict = {si : reach_MEC_copy.pop() for si in new_states}
    for si in new_states:
        pairs = []
        Ci, Di = reach_MEC_dict[si]
        for s in Ci:
            for a in prod.actions(s) - Di[s]: # FORNOW
                pairs.append((s, a))
        A_hat[si] = set(pairs)

    # Assert that replacement occured as expected.
    assert len(A_hat) == len(S) - len(reach_MEC_states) + len(reach_MECs), \
            "Fatal error in generation of A_hat."
    #print "\nA_hat"
    #pprint(A_hat)
    #print

    ## Create the ssp_MDP:
    ssp_MDP = UncertainMDP()
    ssp_MDP.T = {s : {} for s in S_hat}
    for s in S_hat:
        for (u, a) in A_hat[s]:
            # Add all actions TODO more conservative dict addition
            ssp_MDP.T[s][(u,a)] = []

            # Init. checking statements for use in linking to prob's of original
            if u not in prod.states:
                continue
            if a not in prod.actions(u):
                continue

            # old states: follow prod
            for t in S.difference(reach_MEC_states):
                # Final checking statement
                if t not in prod.destinations(u, a):
                    continue
                else:
                    # add in
                    new_tup = (t, prod.probability(u, a, t))
                    ssp_MDP.T[s][(u,a)].append(new_tup)
            # new states: sum over t in prod
            for si in new_states:
                Ci = reach_MEC_dict[si][0]
                prob = IntervalProbability.zero()
                for t in Ci:
                    # Final checking statement
                    if t not in prod.destinations(u, a):
                        continue
                    else:
                        prob += prod.probability(u, a, t)
                # add in
                if prob != IntervalProbability.zero():
                    new_tup = (si, prob)
                    ssp_MDP.T[s][(u,a)].append(new_tup)

    # Algorithm 1: append the terminal state.
    # -and link it to B and S0 
    t_s = "t-1"
    t_a = "terminal"
    ssp_MDP.T[t_s] = {t_a : [(t_s, IntervalProbability.one())]}
    ssp_MDP.states = set(ssp_MDP.T.keys()) #FORNOW
    for state in B.union(S0):
        ssp_MDP.T[state][t_a] = [(t_s, IntervalProbability.one())]

    ssp_MDP.sanity_check()

    # mapping back to original
    ssp_to_original = {}
    for s in ssp_MDP.states: #FORNOW
        # map back to a reach_MEC
        if s in new_states:
            ssp_to_original[s] = deepcopy(reach_MEC_dict[s][0])
        # terminal states not considered in mapping
        elif s == t_s:
            pass
        # s in S0 U AMECs, maps to self
        else:
            ssp_to_original[s] = set([s])

    return ssp_MDP, Sr, S0, B, ssp_to_original


def gen_stochastic_shortest_path_example():
    prod = uncertain_mdp_example()
    AMECs = accepting_max_end_components(prod, [(None, 2)])
    return gen_stochastic_shortest_path(prod, AMECs)

############################################################################
# Misc/Special Use Functions
############################################################################

def gen_best_actions_dict(mdp, V, states=None, verbose=False):
    """
    Used for memoryless policies on discounted product value iteration.
    """
    print "\nWarning: gen_best_actions_dict should only be used with a V " + \
            "that utilized discounting or costs."

    best_actions_dict = {}

    # Default to considering all states
    if states == None:
        states = mdp.states

    for state in states:
        # Grab all possible actions from state.
        actions = mdp.T[state]

        # Find best actions from a state using MDP probabilities.
        best_actions = set()
        action_values = {}
        for action in actions:
            action_value = 0
            for tup in mdp.T[state][action]:
                s_f, p = tup
                # for now an inefficient switch case here
                # TODO remove or move to outside if we need it.
                if mdp.probability_type == float:
                    action_value += (V[s_f] * p)
                elif mdp.probability_type == IntervalProbability:
                    action_value += (V[s_f] * p.low)
                else:
                    raise NotImplementedError
            action_values[action] = action_value
        max_value = max(action_values.values())
        for a in action_values.keys():
            if action_values[a] == max_value:
                best_actions.add(a)
        best_actions_dict[state] = best_actions

    return best_actions_dict

############################################################################
# Main Function
############################################################################

def __main__():
    print "\n" + "*"*80 + "\n"
    # Small example to test generate_product_MDP
    # NOTE Checked against book (B+K).
    # TODO Check again (that was a while ago)
    prod = generate_product_MDP_example()
    pprint(prod.T)

    print "\n" + "*"*80 + "\n"
    # Small example to test max_end_components
    # NOTE Checked by hand: <s4 "4"> <s2 "2"> <s1 "1/3", s3 "1/3">
    max_end_components = max_end_components_example()
    pprint(max_end_components)

    print "\n" + "*"*80 + "\n"
    # Small example to test accepting_max_end_components
    # NOTE Checked by hand: <4 "4">
    # ----------------------<1 "2/3", 2 "1", 3 "1">
    # ----------------------both above
    # ----------------------NONE FOUND
    accepting_max_end_components = accepting_max_end_components_example()
    pprint(accepting_max_end_components)

    print "\n" + "*"*80 + "\n"
    # Small example to test gen_stochastic_shortest_path_example
    # NOTE Uses uncertain_mdp_example
    # NOTE Checked by hand:
    # Sr 3,4,5
    # S0 0
    # B  1,2
    # 
    # set([3,4]) decomposes as:
    # -0/1/5 -> 0/1/d(5), with d(5) likely
    # set([5]) decomposes as:
    # -0/2/3/4 -> 0/2/d(3,4), with d(3,4) one 
    # -(NOTE <.8, .9> + <.8, .9> wraps to <1.0, 1.0>)
    ssp_MDP, Sr, S0, B, ssp_to_original = gen_stochastic_shortest_path_example()
    pprint(ssp_MDP.T)
    pprint(Sr)
    pprint(S0)
    pprint(B)
    pprint(ssp_to_original)

if __name__ == '__main__':
    __main__()


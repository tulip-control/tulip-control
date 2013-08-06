# First cut of an MDP class, and associated methods for TuLiP.
# Originally: James Bern 8/2/2013
# 
# 8/5: MDP states and init states are now sets.
# 8/6: value_iteration() rewritten for use in product MDPs.

from copy import deepcopy
from pprint import pprint
import networkx
from transys import *

############################################################################
# MDP
############################################################################

class MDP:
    def __init__(self, name="DEFAULT MDP"):
        self.name = name
        self.states = set()
        self.initial_states = set()
        self.T = {} # form: T[s_i][a] = [(s_f_0, p_0), ...]
        self.R = {} # form: R[s_i][a] = <float>

    def sanity_check(self):
        """
        * assert(probabilities for each action normalized to one
        """
        for s_i in self.T.keys():
            for a in self.T[s_i]:
                assert(sum([dest_prob_tup[1] for dest_prob_tup in self.T[s_i][a]]) == 1)
                #print "Probabilities for " + str((s_i, a)) + " sum to 1"

    def pre(self, s_f):
        """
        Simple one-step pre.
        """
        pre = set()
        for s_i in self.T.keys():
            for a in self.T[s_i].keys():
                if s_f in self.destinations(s_i, a):
                    pre.add((s_i, a))
        return pre

    def destinations(self, s_i, a):
        """
        One-step post from s_i under action a.
        """
        return [dest_prob_tup[0] for dest_prob_tup in self.T[s_i][a]]

    def induce_digraph(self, states, actions):
        """
        Induce a digraph on an MDP given list/set of states and
        a dictionary of actions avaialabe at state.
        """
        G = {}
        for s in states:
            neighbors = set()
            for a in actions[s]:
                neighbors.update(self.destinations(s, a))
            G[s] = neighbors
        return G

def mdp_example():
    """
    A toy-MDP representing a four-square grid-world:

    s0 s1
    s3 s2

    s2 is terminal

    R(s0) = R(s1) = R(s3) = 0
    R(s2) = 1

    From any square can do a move(target adjacent square) of form:
    --target adjacent square w/ p=.9
    --other  adjacent square w/ p=.1

    Expected:
    V0 ~ .8
    V1 = V3 ~ .9
    V2 = 1
    """
    self = MDP()
    self.states = set(['s0', 's1', 's2', 's3', 's9'])
    self.init_states = ['s0'] # Ask about extending to multiple states

    self.T = {} # T[state][action][i] = ('si', p)
    self.R = {} # R[state][action] = <float>reward
    for state in self.states:
        self.T[state] = {}
        self.R[state] = {}

    # Initialized transitions
    self.T['s0'] = {'a1' : [('s1', .9), ('s3', .1)],
                    'a2' : [('s1', .1), ('s3', .9)]}
    self.T['s1'] = {'a1' : [('s2', .9), ('s0', .1)],
                    'a2' : [('s2', .1), ('s0', .9)]}
    self.T['s2'] = {'a1' : [('s3', .9), ('s1', .1)],
                    'a2' : [('s3', .1), ('s1', .9)]}
    self.T['s3'] = {'a1' : [('s0', .9), ('s2', .1)],
                    'a2' : [('s0', .1), ('s2', .9)],
                    'a9' : [('s9', 1)]}
    self.T['s9'] = {'a9' : [('s9', 1)]}
    # Initialize rewards
    self.R['s0'] = {'a1' : 0,
                    'a2' : 0}
    self.R['s1'] = {'a1' : 0,
                    'a2' : 0}
    self.R['s3'] = {'a1' : 0,
                    'a2' : 0}

    # Currently just checks probabilities.
    self.sanity_check()

    return self

############################################################################
# Generate Product MDP
############################################################################
# TODO Add check that it's deterministic.
# check_deterministic=False
def generate_product_MDP(mdp, ra, check_deterministic=False):
    """
    Generate a product MDP from self (MDP) and a DRA.
    """
    #assert type(ra) == RabinAutomaton

    def satisfies(ra, q, sp, qp, s=None):
        """
        Check transition label against state label of
        destination state.
        Assumes a conjunctive label of form "!s2&s0&s4".
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

        satisfies = False
        for transition in ra.transitions():
            q_i = transition[0]
            q_f = transition[1]
            if q_i != q or q_f != qp:
                continue

            # Assuming we've now wandered into the right transition (i.e. transition
            # -actually exists):
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

            # Debugging block, pass s in function call (original state)
            # -in order to view prev. state.
            # -otherwise s will default to none.
            if False:
                print str((s, q))
                print str((sp, qp))
                print L
                print satisfies
                print ''

            break
        return satisfies

    # Initialize the product
    prod = MDP()
    prod.initial_states = set()
    prod.states = set() # form: [(s0, q0), ...]
    prod.T = {}
    for s in mdp.states:
        for q in ra.states():
            prod.states.add((s, q))
            prod.T[(s,q)] = {}

    # Product initial states.
    for s in mdp.initial_states: # TODO use States object when actually writing MDP class
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

    # TODO
    # Run simple check of determinism of product:
    # -That there is <=1 enabled action and also
    # -one enabled transition for that action.
    if check_deterministic:
        for action_dict in prod.T.values():
            assert len(action_dict) in [0, 1], str(action_dict) + " is invalid."

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
            A_T = {t : A[t] for t in T}
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
                for t_b_tup in MDP.pre(s):
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
                if len(set(T) & set(T_i)) > 0:
                    MEC_new.append( (set(T) & set(T_i)) )

    # Return block.
    output = []
    for T in MEC:
        A_T = {t: A[t] for t in T}
        output.append((T, A_T))
    return output

def max_end_components_example():
    """
    Simple max-end-component example which can be checked
    by hand.
    """
    m = MDP()

    m.T = {}
    m.states = ['s1','s2','s3','s4']
    for state in m.states:
        m.T[state] = {}
    m.T['s1']['a1'] = [('s2', .5), ('s4', .5)]
    m.T['s1']['a2'] = [('s1', .2), ('s3', .8)]
    m.T['s2']['foo'] = [('s1', 1)]
    m.T['s3']['bar'] = [('s1', 1)]
    m.T['s2']['save'] = [('s2', 1)]

    return max_end_components(m)

############################################################################
# Run Examples
############################################################################
"""
print ''
mdp = mdp_example()
pprint(mdp.T)
print ''

print ''
prod = generate_product_MDP_example()
pprint(prod.T)
print ''

print ''
max_end_components = max_end_components_example()
pprint(max_end_components)
print ''
"""
############################################################################
# Misc.
############################################################################
def value_iteration(mdp, epsilon=0.001, R={}, V1=set(), gamma=1, unknown_states=None):
    """
    Modified/specialized AIMA algorithm for value iteration,
    which is used to perform value iteration over either all
    of a product MDP or part of it if graph search has been used
    to optimize.
    """

    gamma  = gamma
    assert gamma == 1, "discounting not currently implemented"

    T = mdp.T

    # Safely defaults to all non-V1 states in T.
    unknown_states = unknown_states
    if unknown_states == None:
        unknown_states = set(mdp.states) - V1

    V1 = V1
    assert V1 != {}, "You have to first isolate at least the AMECs!"

    U1 = {}
    for s in unknown_states:
        U1[s] = 0

    while True:
        U = deepcopy(U1)
        delta = 0

        for s in unknown_states:
            actions = T[s].keys()
            state_value = []

            # I've added in here the notion of a terminal state w/
            # a one-time reward. (For a recurrent reward on an accepting state,
            # one could add in the trivial action {'triv' : (<same state>, 1)}.)
            # Now the rule becomes: if in V1
            if s in V1:
                state_value = [0]

            else:
                for a in actions:
                    action_values = []
                    for (s1, p) in T[s][a]:
                        # determine value_s1
                        if s1 in unknown_states:
                            value_s1 = U[s1]
                        elif s1 in V1:
                            value_s1 = 1
                        else: # s1 in V0
                            value_s1 = 0
                        action_values.append(p * value_s1)

                    state_value.append(sum(action_values))

            U1[s] = R[s] + max(state_value)
            delta = max(delta, abs(U1[s] - U[s]))

        if delta < epsilon: # to support discounted rewards: * (1 - gamma) / gamma:
            return U













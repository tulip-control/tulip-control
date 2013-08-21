# First cut of an MDP class, and associated methods for TuLiP.
# Originally: James Bern 8/2/2013
# 
# 8/5: MDP states and init states are now sets.
# 8/6: value_iteration() rewritten for use in product MDPs.
# 8/6: (FIX) check_deterministic keyword-argument check fixed.
# 8/7: (FIX) generalized generate_product_MDP() to handle case of multiple
# ------transitions for a given (s_i, s_f) in a RA.
# 8/7: more pre/post functions and destinations() now returns a set.
# 8/7: (CHK) value_iteration() now tested and working for use in
# ------evaluation product MDPs.
# 8/7: added check that in satisfies() in generate_product_MDP()
# ------there are <=1 satisfying transitions. TODO check w/ Eric
# 8/7: added all states have actions to sanity check.
# 8/7: added MDP.states and MDP.T compatibility to sanity check.
# 8/7: added condense_split_dynamics()
# ------NOTE all code except sanity_check should work without it
# 8/12: began implementation of costs, TODO how to deal with costs
# -------in a product (should automatically generate from original MDP costs)
# 8/14: a MEC now returns as:
# -------(set-of-states, {state : set-of-actions for state in set-of-states})
# -------instead of with a list of actions
# 8/19: added assertion that all s_f are actual states
# 8/20: induce_digraph() now has sane default values
# 8/20: started type checking NOTE not in UncertainMDP

from copy import deepcopy
from pprint import pprint
import networkx

############################################################################
# MDP
############################################################################
class MDP:
    def __init__(self, name="DEFAULT MDP"):
        self.name = name
        self.states = set()
        self.initial_states = set()
        self.T = {} # transitions | form: T[s_i][a] = [(s_f_0, p_0), ...]
        self.C = {} # costs       | form: C[s_i][a] = <float>
        self.probability_type = float

    def sanity_check(self):
        """
        # type check
        * assert(all s_i in transition dict)
        * assert(all s_i have actions)
        # assert(all s_f are states)
        * assert(probabilities for each action normalized to one
        """
        assert type(self.states) == set, "states must be a set"
        for state in self.states:
            assert type(state) in [str, int], "state must be int of str"
            for action in self.T[state].keys():
                assert type(action) == str, "actions must be str"

        assert self.states == set(self.T.keys()), "compatibility of T and states"

        for s_i in self.T.keys():
            assert len(self.T[s_i]) > 0, "State " + str(s_i) + " has no actions."
            for a in self.T[s_i]:
                assert abs(1.0 - \
                        sum([dest_prob_tup[1] for dest_prob_tup in self.T[s_i][a]])) \
                        < .0001,\
                    "Probabilities for A" + str((s_i, a)) + " don't sum to 1."
                for s_f in self.destinations(s_i, a):
                    assert s_f in self.states, "s_f: " + str(s_f) + " is not in states."

    def actions(self, s):
        """
        Return set of actions available at s.
        """
        return set(self.T[s].keys())

    def probability(self, s_i, a, s_f):
        lst = self.T[s_i][a]
        for tup in lst:
            if tup[0] == s_f:
                return tup[1]
        raise Exception, "Invalid probability query."

    def pre_s_a(self, s_f):
        """
        Simple one-step pre.
        Returns set of (state, action).
        """
        pre = set()
        for s_i in self.T.keys():
            for a in self.T[s_i].keys():
                if s_f in self.destinations(s_i, a):
                    pre.add((s_i, a))
        return pre


    def pre_s(self, s_f):
        """
        Simple one-step pre.
        Returns set of state.
        """
        pre = set()
        for s_i in self.T.keys():
            for a in self.T[s_i].keys():
                if s_f in self.destinations(s_i, a):
                    pre.add(s_i)
        return pre

    def post_s(self, s_i):
        """
        Simple one step post.
        Returns set of state
        """
        post = set()
        for a in self.T[s_i]:
            post.update(self.destinations(s_i, a))
        return post

    def destinations(self, s_i, a):
        """
        One-step post from s_i under action a.
        """
        return set([dest_prob_tup[0] for dest_prob_tup in self.T[s_i][a]])

    # NOTE This function is required for max_end_components()
    # TODO Find a way to use induce_nx_digraph instead.
    # The problem seems to be in networkx's code.
    def induce_digraph(self, states=None, actions=None):
        """
        Returns graph in the form of a dict of sets.

        Note: the notion of actions is lost, and all that
        returned is a graph.
        i.e. {'a' : [('s0', 0.5), ('s1', 0.5)]} will be indistinguishable
        from {'a' : [('s0', 1.0})], 'b' : [('s1', 1.0)]}
        """
        # Default to all states and all actions for all states.
        if states == None:
            print "no states passed to induce_nx_digraph"
            assert actions == None, "actions passed but states not passed."
            states = self.states
        if actions == None:
            print "no actions passed to induce_nx_digraph"
            actions = {s : self.actions(s) for s in states}

        G = {}
        for s in states:
            neighbors = set()
            if s in actions.keys():
                s_actions = actions[s]
                for a in s_actions:
                    neighbors.update(self.destinations(s, a))
            G[s] = neighbors
        return G

    def induce_nx_digraph(self, states=None, actions=None):
        """
        Induce a networkx-digraph.

        Note: the notion of actions is lost, and all that
        returned is a graph.
        i.e. {'a' : [('s0', 0.5), ('s1', 0.5)]} will be indistinguishable
        from {'a' : [('s0', 1.0})], 'b' : [('s1', 1.0)]}
        """
        # Default to all states and all actions for all states.
        if states == None:
            print "no states passed to induce_nx_digraph"
            assert actions == None, "actions passed but states not passed."
            states = self.states
        if actions == None:
            print "no actions passed to induce_nx_digraph"
            actions = {s : self.actions(s) for s in states}

        G = {}
        for s in states:
            neighbors = set()
            if s in actions.keys():
                s_actions = actions[s]
                for a in s_actions:
                    neighbors.update(self.destinations(s, a))
            G[s] = list(neighbors)
        return networkx.from_dict_of_lists(G)

    def condense_split_dynamics(self):
        """
        Destructively condense split dynamics,
        (i.e. [('s0', .5), ('s0', .5)] -> [('s0', 1)]).
        """
        for s in self.states:
            for key in self.T[s].keys():
                lst = self.T[s][key][:]
                new_lst = []
                for tup in lst:
                    # each tup will be <added> to new_lst
                    added = False
                    for new_tup in new_lst:
                        # if have same dest
                        if tup[0] == new_tup[0]:
                            # condense the two tuples into one
                            update_tup = (tup[0], tup[1] + new_tup[1])
                            new_lst.remove(new_tup)
                            new_lst.append(update_tup)
                            added = True
                            break
                    if not added:
                        new_lst.append(tup)
                self.T[s][key] = new_lst

    def can_reach(mdp, B):
        """
        Returns set of states that can reach a set of a states B.
        """
        # Initialize with B
        can_reach_B = deepcopy(B)
        check_next = deepcopy(can_reach_B) # check_next is the frontier
        # Idea is only find pre of new states
        changed = True

        while changed:
            changed = False
            shell_can_reach = deepcopy(can_reach_B)
            for x in check_next:
                pre_s = mdp.pre_s(x)
                shell_can_reach.update(pre_s)
            if len(shell_can_reach) > len(can_reach_B):
                changed = True
                check_next = shell_can_reach.difference(can_reach_B)
                can_reach_B = deepcopy(shell_can_reach)
        return can_reach_B


def mdp_example():
    """
    A toy-MDP representing a four-square grid-world:

    s0 s1
    s3 s2

    s2 is terminal

    From any square can do a move(target adjacent square) of form:
    --target adjacent square w/ p=.9
    --other  adjacent square w/ p=.1

    Expected:
    V0 ~ .8
    V1 = V3 ~ .9
    V2 = 1
    """
    mdp = MDP()
    mdp.states = set(['s0', 's1', 's2', 's3', 's9'])
    mdp.init_states = ['s0'] # Ask about extending to multiple states

    mdp.T = {} # T[state][action][i] = ('si', p)
    mdp.C = {} # C[state][action] = <float>reward
    for state in mdp.states:
        mdp.T[state] = {}
        mdp.C[state] = {}

    # Initialized transitions
    mdp.T['s0'] = {'a1' : [('s1', .9), ('s3', .1)],
                    'a2' : [('s1', .1), ('s3', .9)]}
    mdp.T['s1'] = {'a1' : [('s2', .9), ('s0', .1)],
                    'a2' : [('s2', .1), ('s0', .9)]}
    mdp.T['s2'] = {'a1' : [('s3', .9), ('s1', .1)],
                    'a2' : [('s3', .1), ('s1', .9)]}
    mdp.T['s3'] = {'a1' : [('s0', .9), ('s2', .1)],
                    'a2' : [('s0', .1), ('s2', .9)],}
    # Note if 's3' also has some garbage action we don't find it as in V1.
    mdp.T['s9'] = {'a9' : [('s9', 1)]}
    # Initialize costs
    mdp.C['s0'] = {'a1' : 0,
                    'a2' : 0}
    mdp.C['s1'] = {'a1' : 0,
                    'a2' : 0}
    mdp.C['s3'] = {'a1' : 0,
                    'a2' : 0}

    # Run a sanity check.
    mdp.sanity_check()

    return mdp

def main():
    # NOTE: Checked for correct .T
    mdp = mdp_example()
    pprint(mdp.T)

if __name__ == "__main__":
    main()

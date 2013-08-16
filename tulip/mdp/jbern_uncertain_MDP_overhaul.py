# Originally: James Bern 8/13/2013

from jbern_MDP_overhaul import *
from copy import deepcopy
from pprint import pprint

"""
A first cut at a class for Probability (for uncertain MDPs)
and Transition.
"""

class IntervalProbability:
    """
    A single interval of probability for use in an UncertainMDP.
    """
    def __init__(self, low, high):
        # Case 2: a finite set of ranges of probabilities
        assert type(low) is float
        assert type(high) is float
        assert low <= high, "not of form (this-float-less-than, this-float)"
        self.low = low
        self.high = high
        self.interval = (low, high)
        assert self.low <= 1.0 and self.low >= 0.0, str(self.low) + " is invalid low"
        assert self.high <= 1.0 and self.high >= 0.0, str(self.high) + " is invalid high"

    def __repr__(self):
        return "<%s, %s>" % (self.low, self.high)

    def __eq__(self, other):
        return self.interval == other.interval

    def __ne__(self, other):
        return self.interval != other.interval

    def __add__(self, other):
        """
        Standard interval addition function with a ceiling at 1.0 for low and high.
        """
        result = (self.low + other.low, self.high + other.high)
        return IntervalProbability(min(1.0, result[0]), min(1.0, result[1]))

    @staticmethod
    def zero():
        return IntervalProbability(0.0, 0.0)

    @staticmethod
    def one():
        return IntervalProbability(1.0, 1.0)

    def contains(self, p):
        low, high = self.interval
        return p >= low and p <= high


class UncertainMDP(MDP):
    """
    Uncertain MDP class using Probability objects for probabilities.

    Extends: MDP
    """
    def __init__(self, name="DEFAULT_UNCERTAIN_MDP"):
        MDP.__init__(self, name=name)
        self.probability_type = IntervalProbability

    #@overrides
    def sanity_check(self):
        """
        * assert(all states in transition dict)
        * assert(all states have actions)
        * FORNOW all probabilities are interval probabilities
        """

        assert self.states == set(self.T.keys()), "compatibility of T and states"
        T = self.T
        for s_i in T.keys():
            assert len(self.T[s_i]) > 0, "State " + str(s_i) + " has no actions."
            for a in self.T[s_i]:
                for tup in T[s_i][a]:
                    assert isinstance(tup[1], self.probability_type), str(tup) + \
                            " is invalid for an UncertainMDP."


def uncertain_mdp_example():
    # Likely and unlikely interval probabilities.
    likely = IntervalProbability(0.8, 0.9)
    unlikely  = IntervalProbability(0.1, 0.2)
    one = IntervalProbability.one()

    prod = UncertainMDP()
    prod.LK_lst = [(None, 2)]
    prod.T = {0 : {'a':[(0, unlikely), (6, unlikely)]},
              1 : {'b':[(2, one)]},
              2 : {'c':[(1, one)]},
              5 : {'d':[(2, likely), (0, unlikely)], 'D' : [(6, one)]},
              6 : {'e':[(0, likely), (6, unlikely)]},
             }
    prod.states = set(prod.T.keys())
    prod.init_states = [0] # Not used.
    prod.sanity_check()
    return prod


def gen_stochastic_shortest_path(prod, LK_lst):
    """
    Generate an MDP that obeys ssp assumptions.
    See: cds.caltech.edu/~ewolff/wolff_cdc_12.pdf
    """

    # Algorithm 2: end component elimination

    # S: set of all states
    S = prod.states

    # B: union of all AMECs
    AMECs = accepting_max_end_components(prod, LK_lst)
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
    print "\nA_hat"
    pprint(A_hat)
    print

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

    # Algorithm 1: append the terminal state, just to condensed states.
    t = "terminal"
    one = IntervalProbability(1.0, 1.0)
    ssp_MDP.T[t] = {"u" : [(t, one)]}
    for state in ssp_MDP.states:
        ssp_MDP.T[state]["u"] = [(t, one)]

    ssp_MDP.states = set(ssp_MDP.T.keys()) #FORNOW
    ssp_MDP.sanity_check()
    # TODO ssp_MDP.recallibrate_probabilities()
    print "ssp_MDP.T"
    return ssp_MDP


def gen_stochastic_shortest_path_example():
    pass


def main():
    print
    uncertain_mdp = uncertain_mdp_example()
    pprint(uncertain_mdp.T)
    print
    #ssp_MDP = gen_stochastic_shortest_path_example()
    #pprint(ssp_MDP.T)

if __name__ == "__main__":
    main()


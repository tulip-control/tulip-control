# The UncertainMDP class.
#
# Originally: James Bern 8/13/2013
# jbern@caltech.edu
#
# 8/20: better example

from MDP import MDP
from probability_types import *

from pprint import pprint

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
    prod.T = {0 : {'0:0':[(0, one)]},
              1 : {'1:2':[(2, one)]},
              2 : {'2:1':[(1, one)]},
              3 : {'3:0/1/5':[(0, unlikely), (1, unlikely), (5, likely)],
                   '3:4' : [(4, one)]},
              4 : {'4:3' : [(3, one)]},
              5 : {'5:0/2/3/4':[(0, unlikely), (2, unlikely), (3, likely), (4, likely)],
                   '5:5':[(5, one)]},
             }
    prod.states = set(prod.T.keys())
    prod.init_states = [0] # Not used.
    prod.sanity_check()
    return prod





def main():
    # NOTE: Checked for correct .T
    uncertain_mdp = uncertain_mdp_example()
    pprint(uncertain_mdp.T)

if __name__ == "__main__":
    main()


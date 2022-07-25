# This test includes unit tests for MarkovChain and MarkovDecisionProcess of
# the tulip.transys module
#
# * mc_test(): a unit test for the MarkovChain class
# * mdp_test(): a unit test for the MarkovDecisionProcess class

from tulip.transys import MarkovChain as MC
from tulip.transys import MarkovDecisionProcess as MDP


def mc_test():
    states = {"green", "red"}
    transitions = {
        ("green", "green", 0.8),
        ("green", "red", 0.2),
        ("red", "green", 0.5),
        ("red", "red", 0.5),
    }
    init = "green"
    mc = _construct_mdpmc(states, transitions, init)
    assert len(mc.states) == len(states)
    assert len(mc.states.initial) == 1
    for state in mc.states:
        mc.states[state]["ap"] == {state}
    for transition in transitions:
        mc_transitions = mc.transitions.find(transition[0], transition[1])
        assert len(mc_transitions) == 1
        for mc_transition in mc_transitions:
            assert mc_transition[2]["probability"] == transition[2]


def mdp_test():
    states = {"c8", "c4", "c9"}
    transitions = {
        ("c8", "c8", 0.1, "acc"),
        ("c8", "c8", 0.8, "brake"),
        ("c8", "c4", 0.9, "acc"),
        ("c8", "c4", 0.2, "brake"),
        ("c4", "c4", 0.2, "acc"),
        ("c4", "c4", 0.9, "brake"),
        ("c4", "c9", 0.8, "acc"),
        ("c4", "c9", 0.1, "brake"),
        ("c9", "c9", 1.0, "brake"),
    }
    init = "c8"
    actions = ["acc", "brake"]
    mc = _construct_mdpmc(states, transitions, init, actions)
    assert len(mc.states) == len(states)
    assert len(mc.states.initial) == 1
    for state in mc.states:
        mc.states[state]["ap"] == {state}
    for transition in transitions:
        mc_transitions = mc.transitions.find(transition[0], transition[1])
        assert len(mc_transitions) >= 1
        for mc_transition in mc_transitions:
            assert mc_transition[2]["action"] in actions
            if mc_transition[2]["action"] == transition[3]:
                assert mc_transition[2]["probability"] == transition[2]


def _construct_mdpmc(states, transitions, init, actions=None):
    if actions is not None:
        ts = MDP()
        ts.actions.add_from(actions)
    else:
        ts = MC()
    ts.states.add_from(states)
    ts.states.initial.add(init)
    for transition in transitions:
        attr = {"probability": transition[2]}
        if len(transition) > 3:
            attr["action"] = transition[3]
        ts.transitions.add(
            transition[0],
            transition[1],
            attr,
        )
    for s in states:
        ts.atomic_propositions.add(s)
        ts.states[s]["ap"] = {s}
    return ts

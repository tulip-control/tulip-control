# This test includes unit tests for each element needed to formulate and solve
# the minimum violation planning problem.
# It also includes an integration test of minimum violation planning
# using the example from the EECI 2020 computer lab.
#
# * vectorcost_test(): a unit test for tulip.transys.cost module
# * wks_test(): a unit test for tulip.transys.WeightedKripkeStructure module
# * wfa_test(): a unit test for tulip.transys.automata.WeightedFiniteStateAutomaton module
# * composition_test(): a unit test for tulip.transys.compositions module
# * prioritized_safety_test(): a unit test for tulip.spec.prioritized_safety module
# * graph_algorithm_test(): a unit test for tulip.transys.graph_algorithms module
# * mvp_test(): an integration test for minimum-violation planning based
#   on the EECI 2020 computer lab.

import copy
from tulip.transys.cost import VectorCost
from tulip.transys import WeightedKripkeStructure as WKS
from tulip.transys.automata import WeightedFiniteStateAutomaton as WFA
from tulip.transys.compositions import ks_synchronous_parallel
from tulip.spec.prioritized_safety import PrioritizedSpecification
from tulip.transys.mathset import PowerSet
from tulip.transys.graph_algorithms import (
    dijkstra_single_source_multiple_targets,
    dijkstra_multiple_sources_multiple_targets,
)
from tulip.mvp import solve as solve_mvp


def vectorcost_test():
    num_item = 10
    a = VectorCost([2 * i for i in range(num_item)])
    assert len(a) == num_item
    for i in range(num_item):
        assert a[i] == 2 * i

    b = a + 1
    c = 1 + a
    assert b >= a
    assert b > a
    assert a < b
    assert a <= b
    assert b == c
    assert a != c
    assert len(b) == num_item
    assert len(c) == num_item

    i = 0
    for b_item in b:
        assert b_item == 2 * i + 1
        i += 1

    d = VectorCost([2 * i if i != 2 else 2 * i + 1 for i in range(num_item)])

    assert d > a


def wks_test():
    states = {"c8", "c4", "c9"}
    transitions = {
        ("c8", "c8", 1),
        ("c8", "c4", 0),
        ("c4", "c4", 1),
        ("c4", "c9", 0),
        ("c9", "c9", 0),
    }
    init = "c8"

    ts = _construct_wks(states, transitions, init)

    assert len(ts.states) == len(states)
    for state in ts.states:
        ts.states[state]["ap"] == {state}

    for transition in transitions:
        ts_transitions = ts.transitions.find(transition[0], transition[1])
        assert len(ts_transitions) == 1
        for ts_transition in ts_transitions:
            assert ts_transition[2]["cost"] == transition[2]


def wfa_test():
    propositions = ["h4", "a4"]
    fa = _construct_wfa(propositions, [[]])
    assert len(fa.states) == 1
    assert len(fa.transitions.find()) == 2 ** len(propositions)


def composition_test():
    # Test deterministic transitions
    states1 = {"c1", "c2"}
    transitions1 = {
        ("c1", "c2", 1),
        ("c2", "c1", 0),
    }
    init1 = "c1"

    states2 = {"c1", "c2", "c3"}
    transitions2 = {
        ("c1", "c2"),
        ("c2", "c3"),
        ("c3", "c3"),
    }
    init2 = "c1"

    states3 = {"c4", "c5"}
    transitions3 = {
        ("c4", "c5", 1),
        ("c5", "c4", 0),
    }
    init3 = "c4"

    ts1 = _construct_wks(states1, transitions1, init1)
    ts2 = _construct_wks(states2, transitions2, init2)
    ts3 = _construct_wks(states3, transitions3, init3)

    ts = ks_synchronous_parallel([ts1, ts2, ts3])
    assert isinstance(ts, WKS)

    assert len(ts.states) == len(states1) * len(states2) * len(states3)
    assert len(ts.states.initial) == 1
    init = list(ts.states.initial)[0]
    init_expected = (init1, init2, init3)

    visited = {init}
    visited_expected = {init_expected}
    Q = [init]
    Q_expected = [init_expected]

    while len(Q) > 0:
        v = Q.pop(0)
        v_expected = Q_expected.pop(0)
        transitions = ts.transitions.find(v)
        transitions1 = ts1.transitions.find(v_expected[0])
        transitions2 = ts2.transitions.find(v_expected[1])
        transitions3 = ts3.transitions.find(v_expected[2])
        assert len(transitions) == 1
        assert (
            transitions[0][2]["cost"]
            == transitions1[0][2]["cost"] + transitions3[0][2]["cost"]
        )
        if transitions[0][1] not in visited:
            visited.add(transitions[0][1])
            visited_expected.add(
                (transitions1[0][1], transitions2[0][1], transitions3[0][1])
            )
            Q.append(transitions[0][1])
            Q_expected.append(
                (transitions1[0][1], transitions2[0][1], transitions3[0][1])
            )

    assert len(Q_expected) == 0


def prioritized_safety_test():
    fa1 = _construct_wfa(["a4", "h4"], [["a4"]])
    fa2 = _construct_wfa(["a4", "h4"], [["h4"]])
    fa3 = _construct_wfa(["a4", "h4"], [[]])

    automata = [fa1, fa2, fa3]
    priority = [1, 2, 3]
    level = [0, 0, 1]

    spec = PrioritizedSpecification()
    for i in [0, 2, 1]:
        spec.add_rule(automata[i], priority=priority[i], level=level[i])

    for i in range(len(automata)):
        assert spec[i].automaton() == automata[i]
        assert spec[i].priority() == priority[i]
        assert spec[i].level() == level[i]

    i = 0
    for aut in spec:
        assert aut.automaton() == automata[i]
        assert aut.priority() == priority[i]
        assert aut.level() == level[i]
        i += 1

    assert i == len(automata)
    assert len(spec) == len(automata)
    assert len(spec.get_rules_at(0)) == 2
    assert len(spec.get_rules_at(1)) == 1
    assert len(spec.get_rules_at(2)) == 0
    assert len(spec.get_rules()) == len(automata)
    assert len(set(spec.get_states())) == 1
    assert len(set(spec.get_initial_states())) == 1
    assert len(set(spec.get_accepting_states())) == 1
    assert spec.get_num_levels() == 2


def graph_algorithm_test():
    states = {"c1", "c2", "c3", "c4", "c5"}
    transitions = {
        ("c1", "c2", 1),
        ("c2", "c3", 2),
        ("c1", "c4", 2),
        ("c3", "c5", 1),
        ("c4", "c5", 4),
    }

    ts = _construct_wks(states, transitions, "c1")
    result = dijkstra_single_source_multiple_targets(ts, "c1", ["c5"])
    assert result[0] == 4
    assert result[1] == ["c1", "c2", "c3", "c5"]

    result = dijkstra_multiple_sources_multiple_targets(ts, ["c2", "c4"], ["c5"])
    assert result[0] == 3
    assert result[1] == ["c2", "c3", "c5"]


def mvp_test():
    states_a = {"c8", "c4", "c9"}
    transitions_a = {
        ("c8", "c8", 1),
        ("c8", "c4", 0),
        ("c4", "c4", 1),
        ("c4", "c9", 0),
        ("c9", "c9", 0),
    }
    init_a = "c8"

    states_h = {"c3", "c4", "c5", "c6"}
    transitions_h = {
        ("c3", "c4"),
        ("c4", "c5"),
        ("c5", "c6"),
        ("c6", "c6"),
    }
    init_h = "c3"

    states_l = {"green", "red"}
    transitions_l = {
        ("green", "red"),
        ("red", "red"),
    }
    init_l = "green"

    ts_a = _construct_wks(states_a, transitions_a, init_a, "a")
    ts_h = _construct_wks(states_h, transitions_h, init_h, "h")
    ts_l = _construct_wks(states_l, transitions_l, init_l)

    ts = ks_synchronous_parallel([ts_a, ts_h, ts_l])

    # To define the transition !(h4 & a4), we define 2 sets:
    #   * ap_without_h4 contains all the atomic propositions except 'h4'
    #   * ap_without_a4 contains all the atomic propositions except 'a4'
    # The subset of 2^{AP} corresponding to !(h4 & a4) is the union of
    # PowerSet(ap_without_h4) and PowerSet(ap_without_a4).
    fa1 = _construct_wfa(ts.atomic_propositions, [["a4"], ["h4"]])

    # To define the transition !(red & (a8 | a4)), we define 2 sets:
    #   * ap_without_red contains all the atomic propositions except 'red'
    #   * ap_without_a4a8 contains all the atomic propositions except 'a4' and 'a8'
    fa2 = _construct_wfa(ts.atomic_propositions, [["red"], ["a4", "a8"]])

    # Define the prioritized safety specification
    spec = PrioritizedSpecification()
    spec.add_rule(fa1, priority=1, level=0)
    spec.add_rule(fa2, priority=1, level=1)

    # Solve the minimum violation planning problem
    (cost, state_path, product_path, wpa) = solve_mvp(ts, "a9", spec)
    assert cost == [0, 2, 1]
    assert state_path == [
        ("c8", "c3", "green"),
        ("c8", "c4", "red"),
        ("c4", "c5", "red"),
        ("c9", "c6", "red"),
    ]
    assert product_path == [
        ("null", ("q0", "q0")),
        (("c8", "c3", "green"), ("q0", "q0")),
        (("c8", "c4", "red"), ("q0", "q0")),
        (("c4", "c5", "red"), ("q0", "q0")),
        (("c9", "c6", "red"), ("q0", "q0")),
    ]


def _construct_wks(states, transitions, init, ap_key=None):
    ts = WKS()
    ts.states.add_from(states)
    ts.states.initial.add(init)
    for transition in transitions:
        if len(transition) == 3:
            ts.transitions.add(
                transition[0], transition[1], {"cost": transition[2]},
            )
        else:
            ts.transitions.add(transition[0], transition[1])

    for s in states:
        if not ap_key:
            ts.atomic_propositions.add(s)
            ts.states[s]["ap"] = {s}
        else:
            ap = ap_key + s[1:]
            ts.atomic_propositions.add(ap)
            ts.states[s]["ap"] = {ap}

    return ts


def _construct_wfa(all_propositions, false_propositions):
    # Construct WFA for invariant spec
    # ! \wedge_{props \in false_propositions} \vee_{prop \in props} prop
    fa = WFA()
    fa.atomic_propositions.add_from(all_propositions)
    fa.states.add_from({"q0"})
    fa.states.initial.add("q0")
    fa.states.accepting.add("q0")

    transition_letters = set()
    for propositions in false_propositions:
        props_without_false = copy.deepcopy(fa.atomic_propositions)
        for prop in propositions:
            props_without_false.remove(prop)
        transition_letters |= set(PowerSet(props_without_false))

    for letter in transition_letters:
        fa.transitions.add("q0", "q0", letter=letter)
    return fa

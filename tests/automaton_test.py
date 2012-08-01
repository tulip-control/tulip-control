#!/usr/bin/env python
"""
SCL; 26 Jul 2012.
"""

from StringIO import StringIO
import copy
from tulip.automaton import Automaton, AutomatonState


REFERENCE_AUTFILE = """
smv file: /Users/scott/scott_centre/scm/tulip-xml-branch/examples/specs/robot_simple.smv
spc file: /Users/scott/scott_centre/scm/tulip-xml-branch/examples/specs/robot_simple.spc
priority kind: 3
Specification is realizable...
==== Building an implementation =========
-----------------------------------------
State 0 with rank 0 -> <park:1, cellID:0, X0reach:0>
	With successors : 1, 2
State 1 with rank 1 -> <park:0, cellID:0, X0reach:0>
	With successors : 3, 4
State 2 with rank 1 -> <park:1, cellID:0, X0reach:0>
	With successors : 3, 4
State 3 with rank 1 -> <park:0, cellID:1, X0reach:0>
	With successors : 5, 6
State 4 with rank 1 -> <park:1, cellID:1, X0reach:0>
	With successors : 5, 6
State 5 with rank 1 -> <park:0, cellID:4, X0reach:0>
	With successors : 7, 6
State 6 with rank 1 -> <park:1, cellID:4, X0reach:0>
	With successors : 7, 6
State 7 with rank 1 -> <park:0, cellID:3, X0reach:0>
	With successors : 8, 9
State 8 with rank 1 -> <park:0, cellID:4, X0reach:1>
	With successors : 15, 16
State 9 with rank 1 -> <park:1, cellID:4, X0reach:1>
	With successors : 10, 11
State 10 with rank 0 -> <park:0, cellID:4, X0reach:0>
	With successors : 12, 13
State 11 with rank 0 -> <park:1, cellID:4, X0reach:0>
	With successors : 12, 13
State 12 with rank 0 -> <park:0, cellID:1, X0reach:0>
	With successors : 14, 0
State 13 with rank 0 -> <park:1, cellID:1, X0reach:0>
	With successors : 14, 0
State 14 with rank 0 -> <park:0, cellID:0, X0reach:0>
	With successors : 1, 2, 1, 2
State 15 with rank 0 -> <park:0, cellID:4, X0reach:1>
	With successors : 17, 18
State 16 with rank 0 -> <park:1, cellID:4, X0reach:1>
	With successors : 12, 13
State 17 with rank 0 -> <park:0, cellID:1, X0reach:1>
	With successors : 19, 20
State 18 with rank 0 -> <park:1, cellID:1, X0reach:1>
	With successors : 14, 0
State 19 with rank 0 -> <park:0, cellID:0, X0reach:1>
	With successors : 21, 22
State 20 with rank 0 -> <park:1, cellID:0, X0reach:1>
	With successors : 1, 2
State 21 with rank 1 -> <park:0, cellID:0, X0reach:1>
	With successors : 19, 20
State 22 with rank 1 -> <park:1, cellID:0, X0reach:1>
	With successors : 14, 0
-----------------------------------------
Games time: 12
Checking realizability time: 14
Strategy time: 240
"""


def automaton2xml_test():
    ref_aut = Automaton(states_or_file=StringIO(REFERENCE_AUTFILE))
    aut = Automaton()
    aut.loadXML(ref_aut.dumpXML())
    assert aut == ref_aut
    aut.states[0].transition = []
    assert aut != ref_aut


def node_copy_test():
    node1 = AutomatonState(id=0, state={"nyan":0, "cat":1}, transition=[0])
    node2 = node1.copy()
    assert node2 is not node1
    assert node2.id == 0
    assert node2.state == {"nyan":0, "cat":1}
    assert node2.transition == [0]


class basic_Automaton_test():
    def setUp(self):
        self.empty_aut = Automaton()
        self.singleton = Automaton(states_or_file=[AutomatonState(id=0, state={"x":0}, transition=[0])])
        self.chain_len = 8
        base_state = dict([("x"+str(k),0) for k in range(self.chain_len)])
        self.chain_aut = Automaton()
        for k in range(self.chain_len):
            base_state["x"+str(k)] = 1
            self.chain_aut.addAutState(AutomatonState(id=k, state=base_state, transition=[(k+1)%self.chain_len]))
            base_state["x"+str(k)] = 0

    def tearDown(self):
        pass

    def test_equality(self):
        assert self.empty_aut == self.empty_aut
        assert self.singleton == self.singleton
        assert self.empty_aut != self.singleton

    def test_size(self):
        assert len(self.empty_aut) == 0  # Should give same result as size()
        assert self.empty_aut.size() == 0
        assert self.singleton.size() == 1
        assert len(self.chain_aut) == self.chain_len

    def test_copy(self):
        A = self.singleton.copy()
        assert len(A) == 1
        B = copy.copy(self.singleton)
        assert A == B
        assert A is not B

        C = self.chain_aut.copy()
        assert len(C) == self.chain_len
        assert C == self.chain_aut
        base_state = dict([("x"+str(k),0) for k in range(self.chain_len)])
        base_state["x0"] = 1
        assert C.findAutState(base_state) != -1
        C.findAutState(base_state).state["x0"] = 0
        assert C != self.chain_aut

#!/usr/bin/env python
"""
SCL; 29 Dec 2011.
"""

from StringIO import StringIO
from tulip.automaton import Automaton


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

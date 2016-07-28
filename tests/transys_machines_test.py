#!/usr/bin/env python
"""
Tests for transys.machines (part of transys subpackage)
"""
import logging
logging.basicConfig()
logger = logging.getLogger(__name__)

from tulip.transys import machines

def test_strip_ports():
    mealy = machines.MealyMachine()
    mealy.add_inputs({'door':{'open', 'closed'}})
    mealy.add_outputs({'led':{'on', 'off'},
                       'window':{'open', 'closed'}})

    mealy.add_nodes_from(xrange(10))
    mealy.states.initial.add(0)

    mealy.add_edge(0, 1, door='open', window='open', led='on')
    mealy.add_edge(0, 2, door='open', window='open', led='off')
    mealy.add_edge(2, 3, door='closed', window='closed', led='on')
    mealy.add_edge(3, 4, door='open', window='closed', led='on')
    mealy.add_edge(9, 5, door='closed', window='open', led='off')

    new = machines.strip_ports(mealy, {'window'})

    assert('door' in new.inputs)
    assert('led' in new.outputs)
    assert('window' not in new.outputs)

    edges = [
        (0, 1, dict(door='open', led='on')),
        (0, 2, dict(door='open', led='off')),
        (2, 3, dict(door='closed', led='on')),
        (3, 4, dict(door='open', led='on')),
        (9, 5, dict(door='closed', led='off')),
    ]

    assert(len(edges) == len(new.edges()))
    for (u, v, d), (x, y, b) in zip(new.edges_iter(data=True), edges):
        assert(u == x)
        assert(v == y)
        assert(d == b)

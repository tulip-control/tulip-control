#!/usr/bin/env python
"""Tests for the export mechanisms of tulip.dumpsmach."""
import networkx as nx
from nose.tools import assert_raises
from tulip import spec, synth, dumpsmach


class basic_test:
    def setUp(self):
        self.triv = spec.GRSpec(env_vars="x", sys_vars="y",
                                env_init="x", env_prog="x",
                                sys_init="y", sys_prog="y && x")
        self.triv_M = synth.synthesize("gr1c", self.triv)

        self.dcounter = spec.GRSpec(sys_vars={"y": (0, 5)}, sys_init=["y=0"],
                                    sys_prog=["y=0", "y=5"])
        self.dcounter_M = synth.synthesize("gr1c", self.dcounter)

    def tearDown(self):
        self.dcounter = None
        self.dcounter_M = None

    def test_python_case(self):
        compile(dumpsmach.python_case(self.triv_M),
                filename="<string>", mode="exec")
        # print(dumpsmach.python_case(self.dcounter_M))
        compile(dumpsmach.python_case(self.dcounter_M),
                filename="<string>", mode="exec")


def test_nx():
    g = nx.DiGraph()
    g.inputs = {'a': '...', 'b': '...'}
    g.outputs = {'c': '...', 'd': '...'}
    start = 'Sinit'
    g.add_edge(start, 0, a=0, b=0, c=0, d=0)
    g.add_edge(0, 1, a=0, b=1, c=0, d=1)
    g.add_edge(1, 2, a=1, b=0, c=1, d=1)
    print(dumpsmach.python_case(g, classname='Machine', start='Sinit'))
    exec dumpsmach.python_case(g, classname='Machine', start='Sinit')
    m = Machine()  # previous line creates the class `Machine`
    # Sinit -> 0
    out = m.move(a=0, b=0)
    assert out == dict(c=0, d=0)
    # 0 -> 1
    out = m.move(a=0, b=1)
    assert out == dict(c=0, d=1)
    # invalid input for index 2 in time sequence
    with assert_raises(ValueError):
        m.move(a=1, b=1)
    # 1 -> 2
    out = m.move(a=1, b=0)
    assert out == dict(c=1, d=1)
    # dead-end
    with assert_raises(Exception):
        m.move(a=1, b=0)

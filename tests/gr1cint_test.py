#!/usr/bin/env python
"""
Tests for the interface with gr1c.
"""
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('tulip.spec.lexyacc').setLevel(logging.WARNING)
import networkx as nx
from nose.tools import raises
import os
from tulip.spec import GRSpec, translate
from tulip.interfaces import gr1c


REFERENCE_SPECFILE = """
# For example, regarding states as bitvectors, 1011 is not in winning
# set, while 1010 is. (Ordering is x ze y zs.)

ENV: x ze;
SYS: y zs;

ENVINIT: x & !ze;
ENVTRANS: [] (zs -> ze') & []((!ze & !zs) -> !ze');
ENVGOAL: []<>x;

SYSINIT: y;
SYSTRANS:;
SYSGOAL: []<>y&x & []<>!y & []<> !ze;
"""

REFERENCE_AUTXML = """<?xml version="1.0" encoding="UTF-8"?>
<tulipcon xmlns="http://tulip-control.sourceforge.net/ns/1" version="1">
  <env_vars><item key="x" value="boolean" /></env_vars>
  <sys_vars><item key="y" value="boolean" /></sys_vars>
  <spec><env_init></env_init><env_safety></env_safety><env_prog></env_prog><sys_init></sys_init><sys_safety></sys_safety><sys_prog></sys_prog></spec>
  <aut type="basic">
    <node>
      <id>0</id><anno></anno><child_list> 1 2</child_list>
      <state><item key="x" value="1" /><item key="y" value="0" /></state>
    </node>
    <node>
      <id>1</id><anno></anno><child_list> 1 2</child_list>
      <state><item key="x" value="0" /><item key="y" value="0" /></state>
    </node>
    <node>
      <id>2</id><anno></anno><child_list> 1 0</child_list>
      <state><item key="x" value="1" /><item key="y" value="1" /></state>
    </node>
  </aut>
  <extra></extra>
</tulipcon>
"""

REFERENCE_AUTJSON_smallbool = """
{"version": 1,
 "gr1c": "0.10.2",
 "date": "2015-10-10 16:56:17",
 "extra": "",

 "ENV": [{"x": "boolean"}],
 "SYS": [{"y": "boolean"}],

 "nodes": {
"0x1E8FA40": {
    "state": [0, 0],
    "mode": 0,
    "rgrad": 1,
    "initial": false,
    "trans": ["0x1E8FA00"] },
"0x1E8FA00": {
    "state": [1, 1],
    "mode": 1,
    "rgrad": 1,
    "initial": false,
    "trans": ["0x1E8FA40"] },
"0x1E8F990": {
    "state": [0, 1],
    "mode": 0,
    "rgrad": 1,
    "initial": true,
    "trans": ["0x1E8FA00"] }
}}
"""


class basic_test:
    def setUp(self):
        self.f_un = GRSpec(
            env_vars="x",
            sys_vars="y",
            env_init="x",
            env_prog="x",
            sys_init="y",
            sys_safety=["y -> X(!y)", "!y -> X(y)"],
            sys_prog="y && x",
            moore=False,
            plus_one=False,
            qinit='\A \E')
        self.dcounter = GRSpec(
            env_init=['True'],
            sys_vars={"y": (0, 5)},
            sys_init=["y=0"],
            sys_prog=["y=0", "y=5"],
            moore=False,
            plus_one=False,
            qinit='\A \E')

    def tearDown(self):
        self.f_un = None
        self.dcounter = None

    def test_check_syntax(self):
        assert gr1c.check_syntax(REFERENCE_SPECFILE)
        assert not gr1c.check_syntax("foo")

    def test_to_gr1c(self):
        assert gr1c.check_syntax(translate(self.f_un, 'gr1c'))
        assert gr1c.check_syntax(translate(self.dcounter, 'gr1c'))

    def test_check_realizable(self):
        assert not gr1c.check_realizable(self.f_un)
        self.f_un.sys_safety = []
        assert gr1c.check_realizable(self.f_un)
        self.f_un.qinit = '\A \A'
        self.f_un.env_init = ['x', 'y = 0', 'y = 5']
        self.f_un.sys_init = list()
        assert gr1c.check_realizable(self.f_un)
        assert gr1c.check_realizable(self.dcounter)
        self.dcounter.qinit = '\A \A'
        self.dcounter.sys_init = list()
        assert gr1c.check_realizable(self.dcounter)

    def test_synthesize(self):
        self.f_un.sys_safety = list()  # Make it realizable
        g = gr1c.synthesize(self.f_un)
        assert g is not None
        assert len(g.env_vars) == 1 and 'x' in g.env_vars
        assert len(g.sys_vars) == 1 and 'y' in g.sys_vars

        g = gr1c.synthesize(self.dcounter)
        assert g is not None
        assert len(g.env_vars) == 0
        assert len(g.sys_vars) == 1 and 'y' in g.sys_vars
        assert len(g) == 2

        # In the notation of gr1c SYSINIT: True;, so the strategy must
        # account for every initial state, i.e., for y=0, y=1, y=2, ...
        self.dcounter.qinit = '\A \A'
        self.dcounter.sys_init = list()
        g = gr1c.synthesize(self.dcounter)
        assert g is not None
        print g
        assert len(g.env_vars) == 0
        assert len(g.sys_vars) == 1 and 'y' in g.sys_vars
        assert len(g) == 6, len(g)


class GR1CSession_test:
    def setUp(self):
        self.spec_filename = "trivial_partwin.spc"
        with open(self.spec_filename, "w") as f:
            f.write(REFERENCE_SPECFILE)
        self.gs = gr1c.GR1CSession("trivial_partwin.spc",
                                   env_vars=["x", "ze"],
                                   sys_vars=["y", "zs"])

    def tearDown(self):
        self.gs.close()
        os.remove(self.spec_filename)

    def test_numgoals(self):
        assert self.gs.numgoals() == 3

    def test_reset(self):
        assert self.gs.reset()

    def test_getvars(self):
        vars_str = self.gs.getvars()
        vars_list = [vi.strip() for vi in vars_str.split(",")]
        assert vars_list == ["x (0)", "ze (1)", "y (2)", "zs (3)"]

    def test_getindex(self):
        assert self.gs.getindex({"x": 0, "y": 0, "ze": 0, "zs": 0}, 0) == 1
        assert self.gs.getindex({"x": 0, "y": 0, "ze": 0, "zs": 0}, 1) == 1

    def test_iswinning(self):
        assert self.gs.iswinning({"x": 1, "y": 1, "ze": 0, "zs": 0})
        assert not self.gs.iswinning({"x": 1, "y": 1, "ze": 0, "zs": 1})

    def test_env_next(self):
        assert (self.gs.env_next({"x": 1, "y": 1, "ze": 0, "zs": 0}) ==
                [{'x': 0, 'ze': 0}, {'x': 1, 'ze': 0}])
        assert (self.gs.env_next({"x": 1, "y": 1, "ze": 0, "zs": 1}) ==
                [{'x': 0, 'ze': 1}, {'x': 1, 'ze': 1}])

    def test_sys_nexta(self):
        assert (
            self.gs.sys_nexta(
                {"x": 1, "y": 1, "ze": 0, "zs": 0},
                {"x": 0, "ze": 0}
            ) == [
                {'y': 0, 'zs': 0},
                {'y': 0, 'zs': 1},
                {'y': 1, 'zs': 0},
                {'y': 1, 'zs': 1}
            ])

    def test_sys_nextfeas(self):
        assert self.gs.sys_nextfeas(
            {"x": 1, "y": 1, "ze": 0, "zs": 0},
            {"x": 0, "ze": 0}, 0) == [{'y': 0, 'zs': 0}, {'y': 1, 'zs': 0}]


def test_aut_xml2mealy():
    g = gr1c.load_aut_xml(REFERENCE_AUTXML)
    assert g.env_vars == {"x": "boolean"}
    assert g.sys_vars == {"y": "boolean"}
    print(g.nodes())
    assert len(g) == 3

def test_load_aut_json():
    g = gr1c.load_aut_json(REFERENCE_AUTJSON_smallbool)
    assert g.env_vars == dict(x='boolean'), (g.env_vars)
    assert g.sys_vars == dict(y='boolean'), (g.sys_vars)
    # `REFERENCE_AUTJSON_smallbool` defined above
    h = nx.DiGraph()
    nodes = {0: '0x1E8FA40', 1: '0x1E8FA00', 2: '0x1E8F990'}
    h.add_node(nodes[0], state=dict(x=0, y=0),
               mode=0, rgrad=1, initial=False)
    h.add_node(nodes[1], state=dict(x=1, y=1),
               mode=1, rgrad=1, initial=False)
    h.add_node(nodes[2], state=dict(x=0, y=1),
               mode=0, rgrad=1, initial=True)
    edges = [(nodes[0], nodes[1]),
             (nodes[1], nodes[0]),
             (nodes[2], nodes[1])]
    h.add_edges_from(edges)
    # compare
    for u, d in h.nodes_iter(data=True):
        assert u in g, (u, g.nodes())
        d_ = g.node[u]
        for k, v in d.iteritems():
            v_ = d_.get(k)
            assert v_ == v, (k, v, v_, d, d_)
    h_edges = set(h.edges_iter())
    g_edges = set(g.edges_iter())
    assert h_edges == g_edges, (h_edges, g_edges)


@raises(ValueError)
def synth_init_illegal_check(init_option):
    spc = GRSpec(moore=False, plus_one=False, qinit=init_option)
    gr1c.synthesize(spc)


def synth_init_illegal_test():
    for init_option in ["Caltech", 1]:
        yield synth_init_illegal_check, init_option


@raises(ValueError)
def realiz_init_illegal_check(init_option):
    spc = GRSpec(moore=False, plus_one=False, qinit=init_option)
    gr1c.check_realizable(spc)


def realiz_init_illegal_test():
    for init_option in ["Caltech", 1]:
        yield realiz_init_illegal_check, init_option

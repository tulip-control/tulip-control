#!/usr/bin/env python
"""
Tests for the interface with gr1c.
"""
import logging
logging.basicConfig(level=logging.DEBUG)

from nose.tools import raises
import os

from tulip.spec import GRSpec
from tulip.interfaces import gr1cint


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


class basic_test:
    def setUp(self):
        self.f_un = GRSpec(env_vars="x", sys_vars="y",
                           env_init="x", env_prog="x",
                           sys_init="y",sys_safety=["y -> X(!y)","!y -> X(y)"],
                           sys_prog="y && x")
        self.dcounter = GRSpec(sys_vars={"y": (0,5)}, sys_init=["y=0"],
                               sys_prog=["y=0", "y=5"])

    def tearDown(self):
        self.f_un = None
        self.dcounter = None

    def test_check_syntax(self):
        assert gr1cint.check_syntax(REFERENCE_SPECFILE)
        assert not gr1cint.check_syntax("foo")

    def test_to_gr1c(self):
        assert gr1cint.check_syntax(self.f_un.to_gr1c() )
        assert gr1cint.check_syntax(self.dcounter.to_gr1c() )

    def test_check_realizable(self):
        assert not gr1cint.check_realizable(self.f_un,
                                            init_option="ALL_ENV_EXIST_SYS_INIT")
        self.f_un.sys_safety = []
        assert gr1cint.check_realizable(self.f_un,
                                        init_option="ALL_ENV_EXIST_SYS_INIT")
        assert gr1cint.check_realizable(self.f_un,
                                        init_option="ALL_INIT")

        assert gr1cint.check_realizable(self.dcounter,
                                        init_option="ALL_ENV_EXIST_SYS_INIT")
        self.dcounter.sys_init = []
        assert gr1cint.check_realizable(self.dcounter,
                                        init_option="ALL_INIT")

    def test_synthesize(self):
        self.f_un.sys_safety = []  # Make it realizable
        mach = gr1cint.synthesize(self.f_un,
                                  init_option="ALL_ENV_EXIST_SYS_INIT")
        assert mach is not None
        assert len(mach.inputs) == 1 and mach.inputs.has_key("x")
        assert len(mach.outputs) == 1 and mach.outputs.has_key("y")

        mach = gr1cint.synthesize(self.dcounter,
                                  init_option="ALL_ENV_EXIST_SYS_INIT")
        assert mach is not None
        assert len(mach.inputs) == 0
        assert len(mach.outputs) == 1 and mach.outputs.has_key("y")
        assert len(mach.states) == 3

        # In the notation of gr1c SYSINIT: True;, so the strategy must
        # account for every initial state, i.e., for y=0, y=1, y=2, ...
        self.dcounter.sys_init = []
        mach = gr1cint.synthesize(self.dcounter,
                                  init_option="ALL_INIT")
        assert mach is not None
        print mach
        assert len(mach.inputs) == 0
        assert len(mach.outputs) == 1 and mach.outputs.has_key("y")
        assert len(mach.states) == 7


class GR1CSession_test:
    def setUp(self):
        self.spec_filename = "trivial_partwin.spc"
        with open(self.spec_filename, "w") as f:
            f.write(REFERENCE_SPECFILE)
        self.gs = gr1cint.GR1CSession("trivial_partwin.spc", env_vars=["x","ze"], sys_vars=["y","zs"])

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
        assert self.gs.getindex({"x":0, "y":0, "ze":0, "zs":0}, 0) == 1
        assert self.gs.getindex({"x":0, "y":0, "ze":0, "zs":0}, 1) == 1

    def test_iswinning(self):
        assert self.gs.iswinning({"x":1, "y":1, "ze":0, "zs":0})
        assert not self.gs.iswinning({"x":1, "y":1, "ze":0, "zs":1})

    def test_env_next(self):
        assert self.gs.env_next({"x":1, "y":1, "ze":0, "zs":0}) == [{'x': 0, 'ze': 0}, {'x': 1, 'ze': 0}]
        assert self.gs.env_next({"x":1, "y":1, "ze":0, "zs":1}) == [{'x': 0, 'ze': 1}, {'x': 1, 'ze': 1}]

    def test_sys_nexta(self):
        assert self.gs.sys_nexta({"x":1, "y":1, "ze":0, "zs":0}, {"x":0, "ze":0}) == [{'y': 0, 'zs': 0}, {'y': 0, 'zs': 1}, {'y': 1, 'zs': 0}, {'y': 1, 'zs': 1}]

    def test_sys_nextfeas(self):
        assert self.gs.sys_nextfeas({"x":1, "y":1, "ze":0, "zs":0}, {"x":0, "ze":0}, 0) == [{'y': 0, 'zs': 0}, {'y': 1, 'zs': 0}]


def test_load_autxml():
    (spec, mach) = gr1cint.load_aut_xml(REFERENCE_AUTXML)
    assert spec.env_vars == {"x": "boolean"}
    assert spec.sys_vars == {"y": "boolean"}
    assert len(mach.states) == 4
    assert len(mach.inputs) == 1 and mach.inputs.has_key("x")
    assert len(mach.outputs) == 1 and mach.outputs.has_key("y")

@raises(ValueError)
def synth_init_illegal_check(init_option):
    spc = GRSpec()
    gr1cint.synthesize(spc, init_option=init_option)

def synth_init_illegal_test():
    for init_option in ["Caltech", 1]:
        yield synth_init_illegal_check, init_option

@raises(ValueError)
def realiz_init_illegal_check(init_option):
    spc = GRSpec()
    gr1cint.check_realizable(spc, init_option=init_option)

def realiz_init_illegal_test():
    for init_option in ["Caltech", 1]:
        yield realiz_init_illegal_check, init_option

#!/usr/bin/env python
"""
Warning: some tests in this file call time.sleep() during thread
interactions.  This seems unavoidable given the presence of internal
random stalling in the run() method of parallelsim.Strategy

SCL; 9 December 2012.
"""

import time
from tulip.automaton import Automaton
from tulip.parallelsim import Strategy


_thread_timeout = 3  # seconds

## The below automaton is a strategy for the following GR(1) specification.
# ENV: x;
# SYS: y;
#
# ENVINIT: x;
# ENVTRANS:;
# ENVGOAL: []<>x;
#
# SYSINIT: y;
# SYSTRANS:;
# SYSGOAL: []<>y&x & []<>!y;
REFERENCE_XMLFRAGMENT = """  <aut>
    <node>
      <id>0</id><name></name>
      <child_list>1 2</child_list>
      <state><item key="y" value="0" /><item key="x" value="1" /></state>
    </node>
    <node>
      <id>1</id><name></name>
      <child_list>1 2</child_list>
      <state><item key="y" value="0" /><item key="x" value="0" /></state>
    </node>
    <node>
      <id>2</id><name></name>
      <child_list>1 0</child_list>
      <state><item key="y" value="1" /><item key="x" value="1" /></state>
    </node>
  </aut>
"""

class ParallelSimStrategy_test():
    def setUp(self):
        self.aut = Automaton()
        self.aut.loadXML(REFERENCE_XMLFRAGMENT)
        self.V = {"x":0, "y":0}
        self.strategy_thread = Strategy(self.aut, self.V, ["x"], ["y"], "toy",
                                        Tmin=1, Tmax=1,  # Update time in ms
                                        runtime=10)  # Total allowed runtime

    def tearDown(self):
        self.strategy_thread.runtime = .1
        self.strategy_thread = None
        del self.aut

    def test_single_autcomm(self):
        self.strategy_thread.start()
        assert self.V["x"] == 0
        assert self.V["y"] == 0
        time.sleep(2)
        # Automaton should not change current node until x is set
        assert self.V["x"] == 0
        assert self.V["y"] == 0

        self.V["x"] = 1  # Go!

        # Round 1
        start_t = time.time()
        while self.V["y"] == 0 and time.time() - start_t < _thread_timeout:
            pass
        assert time.time() - start_t < _thread_timeout
        start_t = time.time()
        while self.V["y"] == 1 and time.time() - start_t < _thread_timeout:
            pass
        assert time.time() - start_t < _thread_timeout
        
        # Round 2
        start_t = time.time()
        while self.V["y"] == 0 and time.time() - start_t < _thread_timeout:
            pass
        assert time.time() - start_t < _thread_timeout
        start_t = time.time()
        while self.V["y"] == 1 and time.time() - start_t < _thread_timeout:
            pass
        assert time.time() - start_t < _thread_timeout

        # If execution reaches this point, then test run succeeded.
        # Kill the thread via internal timeout
        self.strategy_thread.runtime = .1

#!/usr/bin/env python
"""
Tests for solver.py
"""

from tulip.solver import SolverInput
import tulip.gridworld as gw
import benchmark.benchmark_tools as bm
import nose.tools as nt

REFERENCE_GWFILE = """
# A very small example, realizable by itself.
6 10
*  G*
  ***  ***
         *
I  *  *G *
  ****** *
*
"""

def varNames_test():
    test_strs = """foo.bar
    foo.bar.baz
    foos_0.bar
    foos_0_0.bar
    gfoos.bar
    dfoos_0.bar
    dfoos_1.bar"""
    svi = SolverInput()
    svi.addModule("foo", instances=1)
    svi.addModule("foos", instances=2)
    
    svi.addModule("gfoos", sys_vars={"bar" : "boolean"}, instances=1)
    svi.globalize("gfoos")
    
    svi.addModule("dfoos", sys_vars={"bar" : "boolean"}, instances=2)
    svi.decompose("dfoos", globalize=True)
    
    svi.addModule("foos_0", instances=3)
    for s in test_strs.split("\n"):
        svi.setSolver("NuSMV")
        nt.eq_(s, svi.canonical(svi.varName(s)))
        svi.setSolver("SPIN")
        nt.eq_(s, svi.canonical(svi.varName(s)))
        
def gridworld_single_actor_test():
    Z = gw.random_world(size=(10,10), num_goals=2)
    rlz = dict()
    for solv in ["NuSMV", "SPIN"]:
        (slvi, paths) = bm.solve_paths(Z, solv=solv, verbose=0)
        if paths:
            for p in paths:
                assert(gw.verify_path(Z, p))
        rlz[solv] = (paths is not None)
    assert(all(rlz.values()) or not any(rlz.values()))
    
def gridworld_multi_actor_test():
    Z = gw.random_world(size=(8,8), num_goals=1)
    rlz = dict()
    for solv in ["NuSMV", "SPIN"]:
        (slvi, paths) = bm.solve_paths(Z, instances=2, solv=solv, verbose=0)
        if paths:
            for p in paths:
                assert(gw.verify_path(Z, p))
            assert(gw.verify_mutex(paths))
        rlz[solv] = (paths is not None)
    assert(all(rlz.values()) or not any(rlz.values()))
    
def gridworld_realizability_test():
    Z = gw.GridWorld(REFERENCE_GWFILE, prefix="Y")
    for solv in ["NuSMV", "SPIN"]:
        (slvi, paths) = bm.solve_paths(Z, solv=solv, verbose=0)
        if paths:
            for p in paths:
                assert(gw.verify_path(Z, p))
        assert paths is not None
        
if __name__ == "__main__":
    varNames_test()
    gridworld_single_actor_test()

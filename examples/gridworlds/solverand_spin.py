#!/usr/bin/env python
"""
Usage: solverand.py [H W]

will generate a random deterministic gridworld problem of the height H
and width W (default is 5 by 10), try to solve and, if realizable,
dump a DOT file named "exampledet.dot" depicting the strategy
automaton.  Example usage for 3 by 5 size is

  $ ./solverand.py 3 5
  $ dot -Tpng -O exampledet.dot

The resulting PNG image built by dot is in the file named
"exampledet.dot.png" or similar.


SCL; 28 June 2012.
"""

import sys, time
import tulip.gridworld as gw
from tulip import gr1cint
import tulip.spinint as spinint
import tulip.automaton as automaton
from subprocess import call

if __name__ == "__main__":
    if len(sys.argv) > 3 or "-h" in sys.argv:
        print "Usage: solverand.py [H W]"
        exit(1)

    if len(sys.argv) >= 3:
        (height, width) = (int(sys.argv[1]), int(sys.argv[2]))
    else:
        (height, width) = (5, 10)

    Z = gw.random_world((height, width),
                        wall_density=0.2,
                        num_init=1,
                        num_goals=2)
    print Z
    
    start = time.time()
    gr1spec = Z.spec()
    # generate SMV spec from discrete transitions
    print "Generating transition system"
    pp = Z.discreteTransitionSystem()
        
    print "Assembling progress specification"
    sp = ["[]<>(" + x + ")" for x in gr1spec.sys_prog]
    initials = { k : True for k in [Z[x] for x in Z.init_list]}
    spinint.generateSPINInput({}, ["", " & ".join(sp)],
                                    {}, pp, "random_grid.pml", initials)

    print "Computing strategy"
    if spinint.computeStrategy("random_grid.pml", "random_grid.aut", 1):
        print "Writing automaton"
        aut = automaton.Automaton('', [])
        aut.loadSPINAut("random_grid.aut", [])
        for state in aut.states:
            # translate cellID -> proposition
            rewrittenState = False
            for (n, p) in enumerate(pp.list_region[state.state["cellID"]].list_prop):
                if p:
                    prop = pp.list_prop_symbol[n]
                    state.state[prop] = "TRUE"
                    rewrittenState = True
            if rewrittenState:
                del(state.state["cellID"])
        print Z.pretty(show_grid=True, path=gw.extractPath(aut))
        aut.writeDotFile("random_grid.dot", hideZeros=True)
        call("dot random_grid.dot -Tpng -o random_grid.png".split())
    else:
        print "Strategy cannot be realized."
    print "SPIN solved in " + str(time.time() - start) + "s"

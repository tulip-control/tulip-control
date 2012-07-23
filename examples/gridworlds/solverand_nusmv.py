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
import tulip.nusmvint as nusmvint
import tulip.automaton as automaton
from subprocess import call

def verify_path(W, path, seq):
    goals = W.goal_list[:]
    if seq:
        # Check if path visits all goals in gridworld W in the correct order
        for p in path:
            if not goals: break
            if goals[0] == p:
                del(goals[0])
        if goals:
            return False
    else:
        # Check if path visits all goals
        for g in goals:
            if not g in path:
                return False
    # Ensure that path does not intersect an obstacle
    for p in path:
        if not W.isEmpty(p):
            return False
    return True

if __name__ == "__main__":
    if len(sys.argv) > 4 or "-h" in sys.argv:
        print "Usage: solverand.py [H W] [goals]"
        exit(1)

    if len(sys.argv) >= 3:
        (height, width) = (int(sys.argv[1]), int(sys.argv[2]))
        if len(sys.argv) >= 4:
            ngoals = int(sys.argv[3])
        else:
            ngoals = 2
    else:
        (height, width) = (5, 10)

    #Z = gw.narrow_passage((height, width), 2)
    Z = gw.random_world((height, width),
                        wall_density=0.2,
                        num_init=1,
                        num_goals=ngoals)
    print Z
    
    start = time.time()
    print "Creating GR1 spec"
    gr1spec = Z.spec()
    SOLVE_MODE = 2
    SEQ = True
    if SOLVE_MODE == 0:
        # directly convert GRSpec to SMV
        svars = {k : "boolean" for k in gr1spec.sys_vars}
        spec = gr1spec.toSMVSpec()
        nusmvint.writeSMV("random_grid.smv", spec, sys_vars=svars)
    elif SOLVE_MODE == 1:
        # generate SMV spec from discrete dynamics
        pp = Z.dumpPPartition()
        pp.trans = pp.adj.tolist()
        for n in range(0, len(pp.trans)):
            pp.trans[n][n] = 1
    elif SOLVE_MODE == 2:
        # generate SMV spec from discrete transitions
        print "Generating transition system"
        pp = Z.discreteTransitionSystem()
        
    if SOLVE_MODE in (1,2):
        print "Assembling progress specification"
        # Not logically necessary but speeds up checking
        sp = ["[]<>(" + x + ")" for x in gr1spec.sys_prog]
        #sp = []
        init_spec = " | ".join([Z[x] for x in Z.init_list])
        sp.append("(" + init_spec + ")")
        if SEQ:
            # Goal sequencing
            sp.append("(goal = 0)")
            for (n,g) in enumerate(Z.goal_list):
                sp.append("[]((goal != %d) | <>(%s))" % (n, Z[g]))
                sp.append("[](goal = %d -> (next(goal = %d) | (%s & next(goal = %d))))"
                            % (n, n, Z[g], n+1))
            # Reset goal count
            sp.append("[](goal = %d -> next(goal = 0))" % len(Z.goal_list))
            # Progress for 'goal': eventually reach all goals in order
            sp.append("[]<>(goal = %d)" % len(Z.goal_list))
            discvars = {"goal" : "{0...%d}" % len(Z.goal_list)}
        else:
            discvars = {}
        nusmvint.generateNuSMVInput(discvars, ["", " & ".join(sp)],
                                        {}, pp, "random_grid.smv")

    print "Computing strategy"
    if nusmvint.computeStrategy("random_grid.smv", "random_grid.aut", 1):
        print "Writing automaton"
        aut = automaton.Automaton('', [])
        aut.loadSMVAut("random_grid.aut", [])
        if SOLVE_MODE in (1,2):
            for state in aut.states:
                # translate cellID -> proposition
                props = pp.reg2props(state.state["cellID"])
                if props:
                    for p in props: state.state[p] = True
                    del(state.state["cellID"])
        assert(verify_path(Z, gw.extractPath(aut), SEQ))
        print Z.pretty(show_grid=True, path=gw.extractPath(aut), goal_order=SEQ)
        aut.writeDotFile("random_grid.dot", hideZeros=True)
        call("dot random_grid.dot -Tpng -o random_grid.png".split())
    else:
        print "Strategy cannot be realized."
    print "NuSMV solved in " + str(time.time() - start) + "s"

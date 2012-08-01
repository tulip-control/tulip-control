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

import sys, time, resource, copy
import tulip.gridworld as gw
from tulip import solver, automaton, ltl_parse
from subprocess import call
import random
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
from itertools import combinations

# Max. memory usage
MEMORY_LIMIT=900000000
N_ROBOTS=1
assert_message="ASSERT FAILED"
get_assert_msg = lambda: assert_message

def verify_path(W, path, seq):
    global assert_message
    goals = W.goal_list[:]
    if seq:
        # Check if path visits all goals in gridworld W in the correct order
        for p in path:
            if not goals: break
            if goals[0] == p:
                del(goals[0])
            elif p in goals:
                return False
        if goals:
            return False
    else:
        # Check if path visits all goals
        for g in goals:
            if not g in path:
                assert_message = "Path does not visit goal " + str(g)
                return False
    # Ensure that path does not intersect an obstacle
    for p in path:
        if not W.isEmpty(p):
            assert_message = "Path intersects obstacle at " + str(p)
            return False
    return True
    
def verify_mutex(paths):
    # sanity check - all paths same length
    if not all(len(p) == len(paths[0]) for p in paths):
        assert_message = "Paths are different lengths"
        return False
    for t in zip(*paths):
        # Coordinates in each tuple must be unique
        if not len(set(t)) == len(t):
            assert_message = "Non-unique coordinates in tuple " + str(t)
            return False
    return True
    
def run_simulation(Z, paths, jitter=0.0):
    colors = 'rgbcmyk'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    Z.plot()
    def update_line(num, dlist, lines):
        for (p,t), d in zip(lines, dlist):
            t.set_data(d[...,:num+1])
            p.set_data(d[...,num])
        #fig.savefig("anim/out%03d.png" % num)
        return lines,

    data = []
    lines = []
    for n,path in enumerate(paths):
        arr = np.array([[x,y] for (y,x) in path]).transpose()
        arr = np.add(arr, jitter*(np.random.rand(*arr.shape) - 0.5))
        data.append(arr)
        l, = ax.plot([], [], 'o', color=colors[n], markersize=10.0, zorder=2)
        l_trail, = ax.plot([], [], '-', color=colors[n], zorder=1)
        lines.append((l, l_trail))
    
    ani = anim.FuncAnimation(fig, update_line, len(paths[0]), fargs=(data,lines),
        interval=500)
    plt.show()
    
def compress(paths):
    # Remove superfluous path elements from a path list
    pzip = zip(*paths)
    acc = []
    for n in range(len(pzip)-1):
        if not pzip[n] == pzip[n+1]:
            acc.append(pzip[n])
    acc.append(pzip[-1])
    return zip(*acc)

if __name__ == "__main__":
    if len(sys.argv) > 4 or "-h" in sys.argv:
        print "Usage: solverand.py [H W] [goals]"
        exit(1)

    if len(sys.argv) >= 3:
        (height, width) = (int(sys.argv[1]), int(sys.argv[2]))
    else:
        (height, width) = (5, 10)
    if len(sys.argv) >= 4:
        ngoals = int(sys.argv[3])
    else:
        ngoals = 2

    #Z = gw.narrow_passage((height, width), 2)
    Z = gw.random_world((height, width),
                        wall_density=0.2,
                        num_init=N_ROBOTS,
                        num_goals=ngoals)
    print Z
    
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT, -1))
    
    start = time.time()
    print "Creating GR1 spec"
    gr1spec = Z.spec()
    SOLVE_MODE = 2
    SEQ = False
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
        shufflespec = gr1spec.sys_prog[:]
        random.shuffle(shufflespec)
        sp = ["[]<>(" + x + ")" for x in shufflespec]
        #sp = []
        #init_spec = " | ".join([Z[x] for x in Z.init_list])
        initials = { Z[x] : True for x in Z.init_list }
        #sp.append("(" + init_spec + ")")
        if SEQ:
            # Goal sequencing
            sp.append("(goal = 0)")
            for (n,g) in enumerate(Z.goal_list):
                sp.append("[]((goal != %d) | <>(%s))" % (n, Z[g]))
                sp.append("[](goal = %d -> (next(goal = %d) | (%s & next(goal = %d))))"
                            % (n, n, Z[g], n+1))
                # Avoid this goal when looking for another
                sp.append("[]((goal != %d) -> !%s)" % (n, Z[g]))
            # Reset goal count
            sp.append("[](goal = %d -> next(goal = 0))" % len(Z.goal_list))
            # Progress for 'goal': eventually reach all goals in order
            sp.append("[]<>(goal = %d)" % len(Z.goal_list))
            discvars = {"goal" : "{0...%d}" % len(Z.goal_list)}
        else:
            discvars = {}
        solverinput = solver.SolverInput()
        ddmodel = solver.discDynamicsModel(discvars, ["", " & ".join(sp)],
                    {}, pp, initials)
        solverinput.addModule("robot", *ddmodel, instances=N_ROBOTS)
        # Mutex
        for (n, m) in combinations(range(N_ROBOTS), 2):
            if n != m:
                solverinput.addSpec(ltl_parse.parse(
                    "[](robot_%d.cellID != robot_%d.cellID)" % (n, m)))
        solverinput.setSolver("SPIN")
        solverinput.decompose("robot")
        solverinput.write("random_grid.smv")
        #solverinput = nusmvint.generateNuSMVInput(discvars, ["", " & ".join(sp)],
        #                            {}, pp, "random_grid.smv", initials)
        #sp.extend(["[]<>(" + x + ")" for x in shufflespec])
        #nusmvint.generateNuSMVInput(discvars, ["", " & ".join(sp)],
        #                                {}, pp, "random_grid_2.smv")

    print "Computing strategy"
    #nusmvint.computeStrategy("random_grid_2.smv", "random_grid.aut", 1)
    if solverinput.solve("random_grid.aut", 1):
        print "Writing automaton"
        aut = solverinput.automaton()
        if SOLVE_MODE in (1,2):
            for state in aut.states:
                # translate cellID -> proposition
                for k in state.state.keys():
                    var = k.rsplit(".")
                    if var[-1] == "cellID":
                        props = pp.reg2props(state.state[k])
                        if props:
                            for p in props:
                                var[-1] = p
                                state.state[".".join(var)] = True
                            del(state.state[k])
        paths = compress([ gw.extractPath(aut, "robot_%d" % n) for n in range(N_ROBOTS) ])
        for n in range(N_ROBOTS):
            assert verify_path(Z, paths[n], SEQ), get_assert_msg()
            print Z.pretty(show_grid=True, path=paths[n], goal_order=SEQ)
        assert(verify_mutex(paths))
        aut.writeDotFile("random_grid.dot", hideZeros=True)
        call("dot random_grid.dot -Tpng -o random_grid.png".split())
        run_simulation(Z, paths, jitter=0.2)
    else:
        print "Strategy cannot be realized."
    print "NuSMV solved in " + str(time.time() - start) + "s"

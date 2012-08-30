#!/usr/bin/env python
"""
Usage: random_gridworld_benchmark.py [iterations] [size_limit]

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

import sys, time, os, signal
import tulip.gridworld as gw
from tulip import gr1cint
import tulip.nusmvint as nusmvint
import tulip.spinint as spinint
import tulip.automaton as automaton
from tulip import jtlvint
from subprocess import call
from argparse import ArgumentParser
from tulip.ltl_parse import parse as ltl_parse

RESULTS_FILE="random_gridworld_benchmark.dat"
NUM_GOALS = 2
WALL_DENS = 0.2
# total (OS + user) CPU time of children
chcputime = (lambda: (lambda x: x[2] + x[3])(os.times()))

class TimeoutError(Exception):
    pass

def alarmhandler(signum, frame):
    raise TimeoutError()
def termhandler(signum, frame):
    print "All child processes terminated."
    
def verify_path(W, path, seq=False):
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
                print "Verification error: path avoids goal", g
                return False
    # Ensure that path does not intersect an obstacle
    for p in path:
        if not W.isEmpty(p):
            print "Verification error: path visits obstacle", p
            return False
    return True
    
def translate_aut(aut, pp):
    for state in aut.states:
        # translate cellID -> proposition
        props = pp.reg2props(state.state["cellID"])
        if props:
            for p in props: state.state[p] = True
            del(state.state["cellID"])

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-o", "--output", help="Benchmark output file", 
                            type=str, default=RESULTS_FILE)
    parser.add_argument("-n", "--iterations", help="Test W random worlds per grid size",
                            type=int, default=5, metavar="W")
    parser.add_argument("-l", "--size-limit", help="Largest grid size",
                            type=int, default=10)
    parser.add_argument("-s", "--start-size", help="Initial grid size",
                            type=int, default=2)
    parser.add_argument("-i", "--world-iterations", help="Check each world ITER times",
                            type=int, default=1, metavar="ITER")
    parser.add_argument("-t", "--timeout", help="Disqualification time in seconds",
                            type=int, default=0)
    parser.add_argument("-w", "--wall-density", help="Wall-density step (<= 1/ITER)",
                            type=float, default=0)
    args = parser.parse_args()
    (limit, iters, init, checks, disqual) = (args.size_limit, args.iterations,
                            args.start_size, args.world_iterations, args.timeout)
    solvers = [ "jtlv", "NuSMV", "gr1c", "SPIN" ]
    disqualified = { k : False for k in solvers }

    signal.signal(signal.SIGTERM, termhandler)
    try:
        f = open(args.output, "w")
        f.write("Solver W H Goals WDensity Time NStates\n")
        for dim in range(init, limit+1):
            wdensity = 0
            for run in range(iters):
                if args.wall_density == 0:
                    wdensity = WALL_DENS
                else:
                    wdensity += args.wall_density
                if wdensity > 1:
                    wdensity -= 1
                try:
                    Z = gw.random_world((dim, dim),
                                        wall_density=wdensity,
                                        num_init=1,
                                        num_goals=NUM_GOALS)
                except ValueError:
                    # World full!
                    print "Could not create world due to insufficient empty space"
                    continue
                gr1spec = Z.spec()
                print Z
                
                for solve in range(checks):
                    rlz = 0
                    print "%dx%d: world %d, run %d, wall density %.2f" \
                             % (dim, dim, run, solve, wdensity)
                    if not disqualified["NuSMV"]:
                        try:
                            signal.signal(signal.SIGALRM, alarmhandler)
                            signal.alarm(disqual)
                            start = time.time()
                            # generate SMV spec from discrete transitions
                            pp = Z.discreteTransitionSystem()
                                
                            # add progress and init requirements
                            sp = ["[]<>(" + x + ")" for x in gr1spec.sys_prog]
                            init_spec = " | ".join([Z[x] for x in Z.init_list])
                            sp.append("(" + init_spec + ")")
                            nusmvint.generateNuSMVInput({}, ["", " & ".join(sp)],
                                                            {}, pp, "random_grid.smv")
                            
                            chcpuoffset = chcputime()
                            if nusmvint.computeStrategy("random_grid.smv", "random_grid_nusmv.aut"):
                                aut = automaton.Automaton()
                                aut.loadSMVAut("random_grid_nusmv.aut")
                                translate_aut(aut, pp)
                                assert(verify_path(Z, gw.extractPath(aut)))
                                nstates = len(aut)
                                rlz += 1
                            else:
                                nstates = -1
                            ctime = chcputime() - chcpuoffset
                            
                            signal.alarm(0)
                            print "NuSMV solved in %.4fs, %.2fs CPU, %d states" \
                                     % (time.time() - start, ctime, nstates)
                            f.write("NuSMV %d %d %d %.2f %.4f %d\n" % (dim, dim,
                                         NUM_GOALS, wdensity, ctime, nstates))
                        except TimeoutError:
                            os.killpg(os.getpid(), signal.SIGTERM)
                            disqualified["NuSMV"] = True
                            print "NuSMV was disqualified"
                    
                    if not disqualified["SPIN"]:
                        try:
                            signal.signal(signal.SIGALRM, alarmhandler)
                            signal.alarm(disqual)
                            start = time.time()
                            
                            sp = ["[]<>(" + x + ")" for x in gr1spec.sys_prog]
                            initials = { k : True for k in [Z[x] for x in Z.init_list]}
                            spinint.generateSPINInput({}, ["", " & ".join(sp)],
                                                            {}, pp, "random_grid.pml", initials)
                            chcpuoffset = chcputime()
                            if spinint.computeStrategy("random_grid.pml", "random_grid_spin.aut"):
                                aut = automaton.Automaton()
                                aut.loadSPINAut("random_grid_spin.aut")
                                translate_aut(aut, pp)
                                assert(verify_path(Z, gw.extractPath(aut)))
                                nstates = len(aut)
                                rlz += 1
                            else:
                                nstates = -1
                            ctime = chcputime() - chcpuoffset
                            
                            signal.alarm(0)
                            print "SPIN solved in %.4fs, %.2fs CPU, %d states" \
                                     % (time.time() - start, ctime, nstates)
                            f.write("SPIN %d %d %d %.2f %.4f %d\n" % (dim, dim,
                                         NUM_GOALS, wdensity, ctime, nstates))
                        except TimeoutError:
                            os.killpg(os.getpid(), signal.SIGTERM)
                            disqualified["SPIN"] = True
                            print "SPIN was disqualified"
                    
                    if not disqualified["gr1c"]:
                        try:
                            signal.signal(signal.SIGALRM, alarmhandler)
                            signal.alarm(disqual)
                            start = time.time()
                            chcpuoffset = chcputime()
                            aut = gr1cint.synthesize(gr1spec)
                            ctime = chcputime() - chcpuoffset
                            
                            if aut:
                                assert(verify_path(Z, gw.extractPath(aut)))
                                rlz += 1
                                nstates = len(aut)
                            else:
                                nstates = -1
                            
                            signal.alarm(0)
                            print "gr1c solved in %.4fs, %.2fs CPU, %d states" \
                                     % (time.time() - start, ctime, nstates)
                            f.write("gr1c %d %d %d %.2f %.4f %d\n" % (dim, dim,
                                         NUM_GOALS, wdensity, ctime, nstates))
                        except TimeoutError:
                            os.killpg(os.getpid(), signal.SIGTERM)
                            disqualified["gr1c"] = True
                            print "gr1c was disqualified"
                    
                    if not disqualified["jtlv"]:
                        try:
                            signal.signal(signal.SIGALRM, alarmhandler)
                            signal.alarm(disqual)
                            start = time.time()
                            spec = gr1spec.toJTLVSpec()
                            ast = ltl_parse(spec[1])
                            spec[1] = ast.toJTLV()
                            sysvars = { k : "boolean" for k in gr1spec.sys_vars }
                            jtlvint.generateJTLVInput({}, sysvars, spec,
                                    smv_file="random_grid_jtlv.smv", spc_file="random_grid_jtlv.spc",
                                    file_exist_option="r")
                            chcpuoffset = chcputime()
                            if jtlvint.computeStrategy(smv_file="random_grid_jtlv.smv",
                                    spc_file="random_grid_jtlv.spc", aut_file="random_grid_jtlv.aut",
                                    file_exist_option="r"):
                                aut = automaton.Automaton("random_grid_jtlv.aut")
                                assert(verify_path(Z, gw.extractPath(aut)))
                                nstates = len(aut)
                                rlz += 1
                            else:
                                nstates = -1
                            ctime = chcputime() - chcpuoffset
                            
                            signal.alarm(0)
                            print "jtlv solved in %.4fs, %.2fs CPU, %d states" \
                                    % (time.time() - start, ctime, nstates)
                            f.write("jtlv %d %d %d %.2f %.4f %d\n" % (dim, dim,
                                         NUM_GOALS, wdensity, ctime, nstates))
                        except (TimeoutError, jtlvint.JTLVError):
                            os.killpg(os.getpid(), signal.SIGTERM)
                            disqualified["jtlv"] = True
                            print "jtlv was disqualified"
                    # Either all solvers realize a strategy or none do
                    assert(rlz == 0 or rlz == len([x for x in solvers if not disqualified[x]]))
                    if not False in disqualified.values():
                        # all checkers disqualified, exit
                        break
                    f.flush()
    except KeyboardInterrupt:
        print "Benchmark aborted."
    finally:
        f.close()
            

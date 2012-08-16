#!/usr/bin/env python
#
# Copyright (c) 2011, 2012 by California Institute of Technology
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the California Institute of Technology nor
#    the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CALTECH
# OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
#
# $Id$
# NuSMV interface
import sys, resource, os
import tulip.gridworld as gw
from tulip import solver
from tulip.solver_common import SolverException
import numpy as np
import string
from itertools import combinations

default_gw = { "size" : (5,5), "wall_density" : 0.2, "num_init" : 1, "num_goals" : 1,
        "num_robots" : 1, "goal_sequence" : False, "obstacle_size" : (1,1) }
datatype = { "solver" : 'S16', "width" : 'i4', "height" : 'i4',
        "num_init" : 'i4', "num_goals" : 'i4', "num_robots" : 'i4',
        "wall_density" : 'f8', "goal_sequence" : 'b', "cpu_time" : 'f8',
        "aut_size" : 'i4', "path_length" : 'i4', "realizable" : 'b',
        "size" : 'i4', "description_length" : 'i4', "state_space" : 'i4',
        "obstacle_size" : 'i4', "spec_nodes" : 'i4', "transitions" : 'i4',
        "memory" : 'i4' }
record = [ "solver", "width", "height", "num_init", "num_goals", "num_robots",
        "wall_density", "goal_sequence", "obstacle_size", "transitions",
        "cpu_time", "aut_size", "path_length", "realizable", "spec_nodes", "memory" ]
descriptions = { "solver" : "Solver",
        "width" : "World width",
        "height" : "World height",
        "num_init" : "Number of initial positions",
        "num_goals" : "Number of goals",
        "num_robots" : "Number of robots", 
        "wall_density" : "Wall density",
        "goal_sequence" : "Visit goals in sequence",
        "cpu_time" : "CPU time (s)",
        "aut_size" : "Automaton size",
        "path_length" : "Length of path",
        "realizable" : "Realizability",
        "size" : "Number of cells",
        "description_length" : "Description length (bits)",
        "state_space" : "Size of state space",
        "obstacle_size" : "Size of obstacles (cells)",
        "spec_nodes" : "Number of nodes in specification AST",
        "transitions" : "Number of transitions",
        "memory" : "Peak memory usage (KB)" }

def strfmt(dtype):
    format_map = { 'S' : '%s', 'i' : '%d', 'f' : '%.4f' }
    return [ format_map[dtype.fields[n][0].kind] for n in dtype.names ]

def gridworld_model(Z, goal_sequence=False, sp=None):
    if sp is None:
        sp = []
    
    initials = { Z[x] : True for x in Z.init_list }
    sp.extend([ "[]<>(%s)" % Z[x] for x in Z.goal_list ])
    discvars = {}
    
    if goal_sequence:
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
    
    pp = Z.discreteTransitionSystem()
    gwmodel = solver.discDynamicsModel(discvars, ["", " & ".join(sp)],
                {}, pp, initials)
    return gwmodel
    
def gridworld_problem(size=(5,5), wall_density=0.2, num_init=1, num_goals=1,
        num_robots=1, goal_sequence=False, obstacle_size=(1,1)):
    Z = gw.random_world(size, wall_density, num_init, num_goals, obstacle_size=obstacle_size)
    model = gridworld_model(Z, goal_sequence)
    slvi = solver.SolverInput()
    slvi.addModule("grid", *model, instances=num_robots)
    if num_robots > 1:
        # Mutex
        for (n, m) in combinations(range(num_robots), 2):
            if n != m:
                slvi.addSpec("[](grid_%d.cellID != grid_%d.cellID)" % (n, m))
    return (slvi, Z)
    
def gridworld_solve_data(slvi, Z, opts):
    if slvi.realized:
        aut = slvi.automaton()
        aut_size = len(aut)
        pp = Z.discreteTransitionSystem()
        solver.restore_propositions(aut, pp)
        if opts["num_robots"] > 1:
                paths = [gw.extractPath(aut, "grid_%d" % n) for n in range(opts["num_robots"])]
        else:
            paths = [gw.extractPath(aut, "grid")]
        for p in paths: assert(gw.verify_path(Z, p, opts["goal_sequence"]))
        assert(gw.verify_mutex(paths))
        path_length = len(gw.compress_paths(paths)[0])
    else:
        aut_size = -1
        path_length = -1
    return { "solver" : slvi.solver, "cpu_time" : slvi.solveTime(),
            "aut_size" : aut_size, "realizable" : slvi.realized,
            "path_length" : path_length, "memory" : slvi.memoryUsage() }

def benchmark_variable(solv, indep, vals, fixed={}):
    # opts contains the static parameters for the gridworld
    opts = dict(default_gw.items() + fixed.items())
    result = []
    try:
        for v in vals:
            opts[indep] = v
            # A specification with more robots than initial positions is trivially
            # unrealizable, so assume that we want as many inits as robots.
            if opts["num_robots"] > opts["num_init"] and "num_init" not in fixed:
                opts["num_init"] = opts["num_robots"]
            print solv, opts
            try:
                (slvi, Z) = gridworld_problem(**opts)
            except ValueError:
                continue
            if opts["num_robots"] > 1 and solv == "SPIN":
                slvi.decompose("grid", globalize=True)
            slvi.setSolver(solv)
            slvi.write("gw_bm.mdl")
            try:
                rlz = slvi.solve("gw_bm.aut")
            except SolverException as e:
                sys.stderr.write(solv + " raised error: " + e.message + "\n")
                break
            outs = gridworld_solve_data(slvi, Z, opts)
            row = dict(opts.items() + outs.items())
            row["height"] = row["size"][0]
            row["width"] = row["size"][1]
            del(row["size"])
            row["obstacle_size"] = row["obstacle_size"][0] * row["obstacle_size"][1]
            row["spec_nodes"] = slvi.specNodes()
            row["transitions"] = slvi.numTransitions(slvi.modules[0]["name"])
            row = tuple([ row[k] for k in record ])
            result.append(row)
    except KeyboardInterrupt:
        pass
    return result
    
def mean_stdev(data, indep, dep, filt={}):
    if filt:
        filtarray = np.array(tuple(filt.values()), dtype=[ (k, datatype[k]) for k in filt.keys() ])
        filtered = data[data[filt.keys()] == filtarray]
    else:
        filtered = data
    indep_vals = np.unique(filtered[indep])
    dtype_dep = [ (dep, 'f8'), (dep + "_std", 'f8') ]
    avgs = np.empty(len(indep_vals), dtype = [ (k, datatype[k]) for k in indep ] + dtype_dep)
    for n, val in enumerate(indep_vals):
        selection = filtered[filtered[indep] == val]
        mean = selection[dep].mean()
        stdev = selection[dep].std()
        # output mean, standard error
        row = list(val) + [ mean, stdev/np.sqrt(len(selection[dep])) ]
        avgs[n] = tuple(row)
    return avgs
    
def compute_fields(data):
    # Computed fields
    comp_fields = ["size", "description_length"]
    comp_rectype = np.dtype(data.dtype.descr + [ (k, datatype[k]) for k in comp_fields])
    ret = np.empty(data.shape, comp_rectype)
    # Fill in original fields
    for field in data.dtype.fields:
        ret[field] = data[field]
    
    # Let m be the number of cells in the world, and n be the number of actors
    # Size = m = width * height
    ret["size"] = ret["width"] * ret["height"]
    
    # Description length:
    # 1) 2m bits
    #       A cell is either empty, an obstacle, a goal or an initial
    # 2) m + (i+g)log m     where i = #initials, g = #goals
    #       A cell is either empty or obstacle, and the locations of initials
    #       and goals are represented by cell numbers.
    # (2) is shorter than (1) for i+g < m/log(m). m/log(m) is bounded below by
    #       sqrt(m) for m > 16, so i+g < sqrt(m) (~dim) is sufficient.
    ret["description_length"] = (ret["size"] + np.ceil(np.log2(ret["size"]))*
                (ret["num_init"] + ret["num_goals"]))
    return ret

color = 1
def plotstrings(ident, xcol, ycol, outfile, eqtype, filterval=None):
    global color
    expform = (string.Template("exp(${ID}_a*x + ${ID}_b)"), "exp(%.3g*x + %.3g)")
    linform = (string.Template("${ID}_a*x + ${ID}_b"), "%.3g*x + %.3g")
    powform = lambda p: (string.Template("${ID}_a*x**%d + ${ID}_b" % p), "%.3g*x**" + str(p) + " + %.3g")

    plotfit = string.Template("""${ID}_a = ${ID}_b = 0.5
    ${ID}_f(x) = $FORMULA
    fit ${ID}_f(x) "$FILENAME" using $XCOL:$YCOL via ${ID}_a, ${ID}_b
    """)
    plotfit_filter = string.Template("""${ID}_a = ${ID}_b = 0.5
    ${ID}_f(x) = $FORMULA
    fit ${ID}_f(x) "$FILENAME" using $XCOL:((stringcolumn(1) eq "$FILTER") ? $$$YCOL : 1/0) via ${ID}_a, ${ID}_b
    """)
    
    plottpl = string.Template("\"$FILENAME\" using $XCOL:$YCOL:$ERRCOL with errorbars \
    title \"$ID\" lt $COLOR, ${ID}_f(x) title sprintf(\"$ID fit: $FORMULA\", ${ID}_a, ${ID}_b) lt $COLOR")
    plottpl_filter = string.Template("\"$FILENAME\" using $XCOL:((stringcolumn(1) eq \"$FILTER\") ? $$$YCOL : 1/0):$ERRCOL \
    with errorbars title \"$ID\" lt $COLOR, ${ID}_f(x) title sprintf(\"$ID fit: $FORMULA\", ${ID}_a, ${ID}_b) lt $COLOR")
    
    if eqtype == "exp":
        eqn = expform
    elif eqtype == "lin":
        eqn = linform
    elif eqtype.startswith("pow_"):
        eqn = powform(int(eqtype.rpartition("_")[2]))
        
    fx = eqn[0].substitute(ID=ident)
    if filterval:
        fit = plotfit_filter.substitute(ID=ident, FILENAME=outfile, XCOL=xcol,
                YCOL=ycol, FORMULA=fx, FILTER=filterval)
        plot = plottpl_filter.substitute(ID=ident, FILENAME=outfile, XCOL=xcol,
                YCOL=ycol, ERRCOL=ycol+1, COLOR=color, FORMULA=eqn[1], FILTER=filterval)
    else:
        fit = plotfit.substitute(ID=ident, FILENAME=outfile, XCOL=xcol,
                YCOL=ycol, FORMULA=fx)
        plot = plottpl.substitute(ID=ident, FILENAME=outfile, XCOL=xcol,
                YCOL=ycol, ERRCOL=ycol+1, COLOR=color, FORMULA=eqn[1])
    color = color + 1
    return (fit, plot)
    
def write_plotfile(prefix, plotstrs, xlabel, ylabel, log=False):    
    pf = string.Template("""
    set xlabel "$XAXIS"
    set ylabel "$YAXIS"
    set terminal png font "" 10
    set output "$FN_PNG"
    """)
    pl = []
    with open(prefix + ".plt", "w") as f:
        for fit, plot in plotstrs:
            f.write(fit)
            pl.append(plot)
        if log:
            f.write("set log y")
        s = pf.substitute(FN_PNG=prefix + ".png", XAXIS=xlabel,
                    YAXIS=ylabel)
        f.write(s)
        f.write("plot " + ", ".join(pl))
        
def simple_benchmark(prefix, x, xr, y, fixed={}, solvers=["NuSMV", "SPIN"],
         eqtype="lin", overwrite=False, cvar=None, filt={}):
    # if xr is a generator, make it a list so we can iterate multiple times
    xr = list(xr)
    # Don't recalculate if the data already exists
    if overwrite or not os.path.exists(prefix + "_raw.dat"):
        data = []
        for s in solvers:
            data.extend(benchmark_variable(s, x, xr, fixed))
        arr = np.array(data, dtype=[ (k, datatype[k]) for k in record ])
        with open(prefix + "_raw.dat", "w") as f:
            # header
            f.write("#" + " ".join(record) + "\n")
            np.savetxt(f, arr, fmt=strfmt(arr.dtype))
    else:
        try:
            arr = np.genfromtxt(prefix + "_raw.dat", dtype=[ (k, datatype[k]) for k in record ])
        except ValueError:
            print >>sys.stderr, "Warning: automatically determining field types"
            arr = np.genfromtxt(prefix + "_raw.dat", dtype=None, names=True)
    arr = compute_fields(arr)
    if cvar is not None:
        # Switch from a controlled value to a computed one
        x = cvar
    out = mean_stdev(arr, ["solver", x], y, filt)
    np.savetxt(prefix + ".dat", out, fmt=strfmt(out.dtype))
    plotstrs = []
    ci = lambda col: colidx(out, col)
    if isinstance(eqtype, str):
        eqtype = [eqtype for s in solvers]
    for s, eq in zip(solvers, eqtype):
        plotstrs.append(plotstrings(s, ci(x), ci(y), prefix + ".dat", eq, s))
    write_plotfile(prefix, plotstrs, descriptions[x], descriptions[y], (eqtype == "exp"))
        
colidx = lambda arr, name: arr.dtype.names.index(name) + 1
nrange = lambda lower, upper, n: (x for x in range(lower, upper+1) for y in range(n))
limit_mem = lambda limit: resource.setrlimit(resource.RLIMIT_AS, (limit*1024*1024, -1)) # MiB

MEMORY_LIMIT=1000
if __name__ == "__main__":
    limit_mem(MEMORY_LIMIT)
    simple_benchmark("bm-obstacles/square,1-10x1-10,30x30grid.SPIN,NuSMV", "obstacle_size", ((x, y) for x in nrange(1, 10, 1) for y in nrange(1, 10, 5)), "cpu_time", fixed = { "size" : (30,30), "wall_density" : 0.25 })

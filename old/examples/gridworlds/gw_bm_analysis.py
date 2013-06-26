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
# Take averages of the output from the gridworld benchmark script.
import numpy as np
import sys
import os
import string

expform = (string.Template("exp(${SOLVER}_a*x + ${SOLVER}_b)"), "exp(%.3f*x + %.3f)")
linform = (string.Template("${SOLVER}_a*x + ${SOLVER}_b"), "%.3f*x + %.3f")

plotfit = string.Template("""${SOLVER}_a = ${SOLVER}_b = 0.5
${SOLVER}_f(x) = $FORMULA
fit ${SOLVER}_f(x) \"$FILENAME\" using $XCOL:((stringcolumn(1) eq "$SOLVER") ? $$$YCOL : 1/0) via ${SOLVER}_a, ${SOLVER}_b
""")

plottpl = string.Template("\"$FILENAME\" using $XCOL:((stringcolumn(1) eq \"$SOLVER\") ? $$$YCOL : 1/0):$ERRCOL with errorbars \
title \"$SOLVER\" lt $COLOR, ${SOLVER}_f(x) title sprintf(\"$SOLVER fit: $FORMULA\", ${SOLVER}_a, ${SOLVER}_b) lt $COLOR")

pf = string.Template("""
set xlabel "$XAXIS"
set ylabel "$YAXIS"
set terminal png font "" 10
set output "$FN_PNG"
""")

columns = ["", "Solver", "Cells", "Goals", "WDensity", "AvgTime", "StDevTime", "AvgStates", "StDevStates"]
colnames = ["", "Solver", "Grid cells", "Number of goals", "Wall Density", "CPU time (s)", "", "Number of states", ""]
err = { columns.index("AvgTime") : columns.index("StDevTime"),
        columns.index("AvgStates") : columns.index("StDevStates") }
        
if len(sys.argv) < 4:
    print "Usage: gw_bm_analysis.py [data file] [x-col] [y-col] <exp/lin>"
    sys.exit(0)
d = np.genfromtxt(sys.argv[1], dtype="S16, i4, i4, i4, f8, f8, i4", names=True)
xcol = columns.index(sys.argv[2])
ycol = columns.index(sys.argv[3])
if len(sys.argv) >= 5:
    EXP = (sys.argv[4] == "exp")
else:
    # Default linear fit
    EXP = False
    
if EXP: eqn = expform
else: eqn = linform

avgs = []
solvers = ["NuSMV", "jtlv", "gr1c", "SPIN"]
# List of columns specifying dimension of a grid
dimension = ["W", "H", "Goals", "WDensity"]
for solver in solvers:
    s_data = d[d["Solver"] == solver]
    for dim in np.unique(s_data[dimension]):
        # Mean & error in the mean
        times = s_data[s_data[dimension] == dim]["Time"]
        time_mean = times.mean()
        time_stdev = times.std()/np.sqrt(len(times))
        states = s_data[s_data[dimension] == dim]["NStates"]
        states_mean = states.mean()
        states_stdev = states.std()/np.sqrt(len(states))
        avgs.append((solver, dim[0] * dim[1], dim[2], dim[3], time_mean,
                    time_stdev, states_mean, states_stdev))

(prefix, ext) = os.path.splitext(sys.argv[1])
outfile = prefix + ".avg" + ext
pltfile = prefix + ".avg.plt"
pngfile = prefix + ".png"

with open(outfile, "w") as f:
    f.write(" ".join(columns[1:]) + "\n")
    for a in avgs:
        f.write("%s %d %d %.4f %.4f %.4f %.4f %.4f\n" % a)

with open(pltfile, "w") as f:
    pl = []
    for (n, solver) in enumerate(solvers):
        fx = eqn[0].substitute(SOLVER=solver)
        s = plotfit.substitute(SOLVER=solver, FILENAME=outfile, XCOL=xcol,
                YCOL=ycol, FORMULA=fx)
        f.write(s)
        s = plottpl.substitute(SOLVER=solver, FILENAME=outfile, XCOL=xcol,
                YCOL=ycol, ERRCOL=err[ycol], COLOR=n, FORMULA=eqn[1])
        pl.append(s)
    s = pf.safe_substitute(FN_PNG=pngfile, XAXIS=colnames[xcol],
                YAXIS=colnames[ycol])
    f.write(s)
    if EXP: f.write("set log y\n")
    f.write("plot " + ", ".join(pl))

#!/usr/bin/env python
"""
Usage: genrandt.py [-p] [-i FILE] [-t N] [-g G] [-b] [H W]

will generate a random gridworld of height H and width W (default is
5x10), with N trolls (default 1) at random positions, G goals (default 2)
at random positions, and print the resulting specification.

Troll region radii are set using the variable TROLL_RADIUS (default 1).

If the flag "-p" is given, then generate a plot of the grid.  If the -i
flag is given in addition to -p, then save the plot to a PDF FILE.  If the
flag -b is given, instead of integer-valued variables, use representation
where there is one boolean variable in the specification per grid cell.
"""
from __future__ import print_function

import sys
import matplotlib as mpl
mpl.use('agg')  # change the backend as available in your environment
import matplotlib.pyplot as plt
import tulip.gridworld as gw


TROLL_RADIUS = 1

if len(sys.argv) > 11 or "-h" in sys.argv:
    print("Usage: genrandt.py [-p] [-i FILE] [-t N] [-g G] [-b] [H W]")
    sys.exit(1)

if "-b" in sys.argv:
    nonbool = False
    sys.argv.remove("-b")
else:
    nonbool = True

try:
    targ_ind = sys.argv.index("-t")
    if targ_ind > len(sys.argv)-2:
        print("Invalid use of -t flag.  Try \"-h\"")
        sys.exit(1)
except ValueError:
    targ_ind = -1
if targ_ind < 0:
    N = 1
else:
    N = int(sys.argv[targ_ind+1])

try:
    garg_ind = sys.argv.index("-g")
    if garg_ind > len(sys.argv)-2:
        print("Invalid use of -g flag.  Try \"-h\"")
        sys.exit(1)
except ValueError:
    garg_ind = -1
if garg_ind < 0:
    num_goals = 2
else:
    num_goals = int(sys.argv[garg_ind+1])

if "-p" in sys.argv:
    print_pretty = True
    sys.argv.remove("-p")
    try:
        iarg_ind = sys.argv.index("-i")+1
        if iarg_ind > len(sys.argv)-1:
            print("Invalid use of -i flag.  Try \"-h\"")
            sys.exit(1)
    except ValueError:
        iarg_ind = -1
else:
    print_pretty = False

if len(sys.argv) >= 3 and sys.argv[-2][0] != "-":
    (height, width) = (int(sys.argv[-2]), int(sys.argv[-1]))
else:
    (height, width) = (5, 10)

Z, troll_list = gw.random_world((height, width),
                                wall_density=0.2,
                                num_init=1,
                                num_goals=num_goals, num_trolls=N)
for i in range(len(troll_list)):
    troll_list[i] = (troll_list[i][0], TROLL_RADIUS)

print(Z.pretty(show_grid=True, line_prefix="## "))

print(Z.dumps(line_prefix="# "))
(spec, moves_N) = gw.add_trolls(Z, troll_list)
print(spec.pretty())

if print_pretty:
    Z.plot(font_pt=0, troll_list=troll_list)
    if iarg_ind == -1:
        plt.show()
    else:
        plt.savefig(sys.argv[iarg_ind])

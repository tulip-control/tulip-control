#!/usr/bin/env python
"""Usage: genrand.py [-b] [-s] [-p] [H W]

will generate a random gridworld of the height H and width W (default
is 5 by 10) and dump the resulting description string.  An example
use-case is to save the result to a file for later fun by

  $ ./genrand.py 20 70 > goodtimes.txt

You could then load this in your code using, e.g,

  with open("goodtimes.txt", "r") as f:
      X = GridWorld(f.read())

If the flag "-s" is given, then a specification using TuLiP LTL syntax
is printed with the description string given in the comments.  If the
flag "-p" is given, then pretty-print the gridworld into the comments.

Use the flag -b to use a representation where there is one boolean
variable in the specification per grid cell.  Otherwise (default),
support for nonboolean domains is used.
"""
from __future__ import print_function

from os import environ as os_environ
import sys
import tulip.gridworld as gw

if "TULIP_REGRESS" in os_environ:
    import numpy
    numpy.random.seed(0)


if len(sys.argv) > 5 or "-h" in sys.argv:
    print("Usage: genrand.py  [-s] [-p] [H W]")
    sys.exit(1)

if "-b" in sys.argv:
    nonbool = False
    sys.argv.remove("-b")
else:
    nonbool = True

if "-s" in sys.argv:
    dump_spec = True
    sys.argv.remove("-s")
else:
    dump_spec = False

if "-p" in sys.argv:
    print_pretty = True
    sys.argv.remove("-p")
else:
    print_pretty = False

if len(sys.argv) >= 3:
    (height, width) = (int(sys.argv[1]), int(sys.argv[2]))
else:
    (height, width) = (5, 10)

Z = gw.random_world((height, width),
                    wall_density=0.2,
                    num_init=1,
                    num_goals=2)

if dump_spec:
    if print_pretty:
        print(Z.pretty(show_grid=True, line_prefix="## "))
    print(Z.dumps(line_prefix="# "))
    print(Z.spec(nonbool=nonbool).pretty())
else:
    if print_pretty:
        print(Z.pretty(show_grid=True, line_prefix="# "))
    print(Z.dumps())

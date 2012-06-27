#!/usr/bin/env python
"""
Usage: genrand.py [-s] [-p] [H W]

will generate a random gridworld of the height H and width W (default
is 5 by 10) and dump the resulting description string.  An example
use-case is to save the result to a file for later fun by

  $ ./genrand.py 20 70 > goodtimes.txt

You could then load this in your code using, e.g,

  with open("goodtimes.txt", "r") as f:
      X = GridWorld(f.read())

If the flag "-s" is given, then a gr1c specification is printed with
the description string given in the comments.  If the flag "-p" is
given, then pretty-print the gridworld into the comments.


SCL; 27 June 2012.
"""

import sys
import tulip.gridworld as gw


if __name__ == "__main__":
    if len(sys.argv) > 5 or "-h" in sys.argv:
        print "Usage: genrand.py  [-s] [-p] [H W]"
        exit(1)

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
            print Z.pretty(show_grid=True, line_prefix="## ")
        print Z.dumps(line_prefix="# ")
        print Z.spec().dumpgr1c()
    else:
        if print_pretty:
            print Z.pretty(show_grid=True, line_prefix="# ")
        print Z.dumps()

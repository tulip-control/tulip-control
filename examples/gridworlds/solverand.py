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

import sys
import tulip.gridworld as gw
from tulip import gr1cint


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
    
    if not gr1cint.check_realizable(Z.spec()):
        print "Not realizable."
    else:
        aut = gr1cint.synthesize(Z.spec())
        aut.writeDotFile("exampledet.dot", hideZeros=True)

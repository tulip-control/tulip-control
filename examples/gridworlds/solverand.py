#!/usr/bin/env python
"""
Usage: solverand.py [H W]

will generate a random deterministic gridworld of height H and width W
(default is 5 by 10), try to solve it and, if realizable, create an
SVG file named "ctrl-solverand.svg" that depicts the controller, which
is a Mealy machine.  Example usage for 3 by 5 size is

  $ ./solverand.py 3 5
"""
from __future__ import print_function

import sys
import tulip.gridworld as gw
from tulip import synth


if len(sys.argv) > 3 or "-h" in sys.argv:
    print("Usage: solverand.py [H W]")
    sys.exit(1)

if len(sys.argv) >= 3:
    (height, width) = (int(sys.argv[1]), int(sys.argv[2]))
else:
    (height, width) = (5, 10)

Z = gw.random_world((height, width),
                    wall_density=0.2,
                    num_init=1,
                    num_goals=2)
print(Z)

spc = Z.spec()
spc.moore = False
spc.qinit = r'\A \E'
if not synth.is_realizable(spc, solver='omega'):
    print("Not realizable.")
else:
    ctrl = synth.synthesize(spc, solver='omega')
    if not ctrl.save('ctrl-solverand.svg'):
        print(ctrl)

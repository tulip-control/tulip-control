#!/usr/bin/env python
"""
Usage: drawrand.py [H W]

will generate and draw a random gridworld of the height H and width W;
default is 5 by 10.

SCL; 27 June 2012.
"""

import sys
import matplotlib.pyplot as plt
import tulip.gridworld as gw


if __name__ == "__main__":
    if (len(sys.argv) != 3 and len(sys.argv) != 1) or "-h" in sys.argv:
        print "Usage: drawrand.py [H W]"
        exit(1)

    if len(sys.argv) == 3:
        (height, width) = (int(sys.argv[1]), int(sys.argv[2]))
    else:
        (height, width) = (5, 10)

    gw.random_world((height, width),
                    wall_density=0.2,
                    num_init=1,
                    num_goals=2
                    ).plot(font_pt=12, show_grid=True)
    plt.show()

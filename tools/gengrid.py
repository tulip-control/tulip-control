#!/usr/bin/env python
"""
Generate uniform grid in the plane and output as YAML polytopes (vertices).

SCL; 3 Apr 2012
"""

import numpy as np
import sys


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print "Usage: %s x0 x1 dx y0 y1 dy" % sys.argv[0]
        exit(1)

    part_name = "initial_part"
    prefix = "X"
    cell_counter = 0
    dx = float(sys.argv[3])
    dy = float(sys.argv[6])
    print "initial_part:"
    for x in np.arange(float(sys.argv[1]), float(sys.argv[2]), dx):
        for y in np.arange(float(sys.argv[4]), float(sys.argv[5]), dx):
            print "  X"+str(cell_counter)+":\n    V: |"
            print "      "+str(x)+" "+str(y)
            print "      "+str(x+dx)+" "+str(y)
            print "      "+str(x+dx)+" "+str(y+dy)
            print "      "+str(x)+" "+str(y+dy)
            cell_counter += 1

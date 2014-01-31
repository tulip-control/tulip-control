#!/usr/bin/env python
"""
Tests for the polytope subpackage.
"""
import numpy as np
from tulip import polytope as pc

# unit square
Ab = np.array([[0.0, 1.0, 1.0],
               [0.0, -1.0, 0.0],
               [1.0, 0.0, 1.0],
               [-1.0, 0.0, 0.0]])

A = Ab[:,0:2]
b = Ab[:, 2]

p = pc.Polytope(A, b)
p2 = pc.Polytope(A, 2*b)

assert(p <= p2)
assert(not p2 <= p)
assert(not p2 == p)

r = pc.Region([p])
r2 = pc.Region([p2])

assert(r <= r2)
assert(not r2 <= r)
assert(not r2 == r)

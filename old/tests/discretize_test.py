#!/usr/bin/env python
"""
SCL; 26 August 2012.
"""

import numpy as np
from tulip.discretize import _block_diag2


def block_diag2_test():
    tol = 1e-16
    assert np.all(np.abs(_block_diag2(np.eye(3), np.eye(4)) - np.eye(7)) < tol)
    assert np.all(_block_diag2(np.zeros((3,2), dtype=np.int16), np.zeros(4, dtype=np.int16)) == np.zeros((4,6), dtype=np.int16))
    A = np.array([[1, 2, 3.],
                  [1, 2, -3],
                  [5, 0, 0]])
    B = np.array([[-.1, 0,],
                  [.003, .1]])
    C = np.array([[1, 2, 3., 0, 0],
                  [1, 2, -3, 0, 0],
                  [5, 0, 0, 0, 0],
                  [0,0,0, -.1, 0,],
                  [0,0,0, .003, .1]])
    assert np.all(_block_diag2(A, B) == C)

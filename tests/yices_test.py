#!/usr/bin/env python
"""
Based on test code moved here from bottom of tulip/rhtlputil.py

SCL; 31 Dec 2011.
"""

import os

from tulip.rhtlputil import *


class yices_test:

    def setUp(self):
        self.tmpdir = "tmpspec"
        self.ysfile = os.path.join(self.tmpdir, "tmp.ys")

    def tearDown(self):
        os.remove(self.ysfile)
        os.rmdir(self.tmpdir)
        
    def test_yicesSolveSat(self):
        vardict = {'a': 'boolean', 'b': '{0, 2, 3, -15}', 'x': [14, -3]}
        expr='!a -> (!(b = 2) | b < 0 & x > 0)'
        toYices(expr=expr, allvars=vardict,
                ysfile=self.ysfile, verbose=3)
        assert yicesSolveSat(expr=expr, allvars=vardict,
                             ysfile=self.ysfile,
                             verbose=3) == (True, "(= a false)\n(= b -15)\n(= x 14)\n")

def test_evalExpr():
    vardict = {'x':1, 'y':0, 'z':1, 'w':0}
    expr = 'x=0 -> y < 1 & z >= 2 -> w'
    assert not evalExpr(expr=expr, vardict=vardict, verbose=3)
    expr = '(x=0 -> y < 1) & (z >= 2 -> w)'
    assert evalExpr(expr=expr, vardict=vardict, verbose=3)

def test_findCycle():
    graph = [[1,2,3], [2], [], [2]]
    assert findCycle(graph=graph, verbose=3) == []

    graph = [[2,3], [0], [0,1], [2]]
    assert findCycle(graph=graph, verbose=3) == [0, 2, 0]
    assert findCycle(graph=graph, W0ind=[0], verbose=3) == []

def test_expr2ysstr():
    expr = "x & (y+1 >= 1 | (z -> w))"
    assert expr2ysstr(expr=expr, verbose=3) == "(and x (or (>= (+ y 1) 1) (=> z w)))"

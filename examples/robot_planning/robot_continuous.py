#!/usr/bin/env python
"""
The example is an extension of robot_discrete_simple.py by including continuous dynamics. 

Nok Wongpiromsarn (nok@cds.caltech.edu)
September 2, 2010

minor refactoring by SCL <slivingston@caltech.edu>
1 May 2011.

Small modifications by Yuchen Lin.
12 Aug 2011

NO, system and cont. prop definitions based on TuLiP 1.x
2 Jul, 2013
"""

#@importvar@
import sys, os
from numpy import array

from tulip import *
import tulip.polytope as pc
from tulip.abstract import prop2part
#@importvar_end@

# Continuous state space
#@contdyn@
cont_state_space = pc.Polytope.from_box(array([[0., 2.],[0., 3.]]))


# Continuous dynamics: \dot{x} = u_x, \dot{y} = u_y
A = array([[1.0, 0.],[ 0., 1.0]])
B = array([[0.1, 0.],[ 0., 0.1]])
U = pc.Polytope.from_box(array([[-1., 1.],[-1., 1.]]))

sys_dyn = hybrid.LtiSysDyn(A,B,[],[],U,[], cont_state_space)
#@contdyn_end@

# Continuous proposition
cont_props = {}
cont_props['X1'] = pc.Polytope.from_box(array([[0., 1.],[0., 1.]]))
cont_props['X2'] = pc.Polytope.from_box(array([[1., 2.],[2., 3.]]))

# Compute the proposition preserving partition of the continuous state space
cont_partition = prop2part.prop2part(cont_state_space, cont_props)

# TO DO (when the relevant pieces are in place):
# Specifications
# Discretization
# Synthesis
# Simulation

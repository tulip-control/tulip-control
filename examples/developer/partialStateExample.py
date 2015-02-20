#!/usr/bin/env python
#
# WARNING: This example may not yet be working.  Please check again in
#          the upcoming release.
#
"""
This example is an extension of robot_discrete.py by including continuous
dynamics with disturbances.

Petter Nilsson (pettni@kth.se)
August 14, 2011

NO, system and cont. prop definitions based on TuLiP 1.x
2 Jul, 2013
NO, TuLiP 1.x discretization
17 Jul, 2013
OM, Testing with shrunk Polytopes
4 Oct, 2014
"""
#
# Note: This code is commented to allow components to be extracted into
# the tutorial that is part of the users manual.  Comments containing
# strings of the form @label@ are used for this purpose.

import logging
logging.basicConfig(level=logging.INFO)

# @import_section@
import numpy as np

from tulip import spec, synth, hybrid
from polytope import box2poly
from tulip.abstract import prop2part, discretize
from tulip.abstract.plot import plot_partition
from tulip.abstract.prop2partition import shrinkPartition, shrinkPoly
from tulip.hybrid import generateFilter
from cvxopt import matrix
# @import_section_end@

show = True

# @dynamics_section@
# Problem parameters
input_bound = 20.0
uncertainty = 0.001
epsilon = 0.02
filter_bound = 1 - uncertainty/epsilon
# Continuous state space
cont_state_space = box2poly([[0., 3.], [0., 2.]])

# Continuous dynamics
A = np.array([[0.95, 0.2], [ 0., 0.95]]) #need (A,C) observable
B = np.array([[0.2, 0.], [ 0., 0.2]])
C = np.array([[1.0, 1.0]])
E = np.array([[1.0,0.], [0.,1.0]])

# Available control, possible disturbances
U = input_bound *np.array([[-1., 1.], [-1., 1.]])
W = uncertainty *np.array([[-1., 1.], [-1., 1.]])

# Convert to polyhedral representation
U = box2poly(U)
W = box2poly(W)

# Construct the LTI system describing the dynamics
sys_dyn = hybrid.LtiOutSysDyn(A=A,B=B,C=C, E=E, K=None, Uset=U, Wset=W, domain=cont_state_space)
L = generateFilter(A, C, filter_bound, use_mosek=False)
sys_dyn_hat = sys_dyn.generateObservedDynamics(L,epsilon)
# @dynamics_section_end@

# @partition_section@
# Define atomic propositions for relevant regions of state space
cont_props = {}
cont_props['home'] = shrinkPoly(box2poly([[0., 1.], [0., 1.]]),epsilon)
cont_props['lot'] = shrinkPoly(box2poly([[2., 3.], [1., 2.]]),epsilon)

# Compute the proposition preserving partition of the continuous state space
cont_partition = prop2part(cont_state_space, cont_props)
plot_partition(cont_partition) if show else None
cont_partition = shrinkPartition(cont_partition, epsilon)
plot_partition(cont_partition) if show else None
# @partition_section_end@

# @discretize_section@
# Given dynamics & proposition-preserving partition, find feasible transitions
disc_dynamics = discretize(
    cont_partition, sys_dyn_hat, closed_loop=True, conservative=True,
    N=1, min_cell_volume=0.01, plotit=show, trans_length=3
)
# @discretize_section_end@

"""Visualize transitions in continuous domain (optional)"""
plot_partition(disc_dynamics.ppp, disc_dynamics.ts,
               disc_dynamics.ppp2ts) if show else None

"""Specifications"""
# Environment variables and assumptions
env_vars = {'park'}
env_init = set()                # empty set
env_prog = '!park'
env_safe = set()                # empty set

# System variables and requirements
sys_vars = {'X0reach'}
sys_init = {'X0reach'}          
sys_prog = {'home'}               # []<>home
sys_safe = {'(X(X0reach) <-> lot) || (X0reach && !park)'}
sys_prog |= {'X0reach'}

# Create the specification
specs = spec.GRSpec(env_vars, sys_vars, env_init, sys_init,
                    env_safe, sys_safe, env_prog, sys_prog)

# @synthesize_section@
"""Synthesize"""
ctrl = synth.synthesize('gr1c', specs,
                        sys=disc_dynamics.ts, ignore_sys_init=True)

# Generate a graphical representation of the controller for viewing
#if not ctrl.save('continuous.png'):
#    print(ctrl)
# @synthesize_section_end@

# Simulation

#!/usr/bin/env python
"""Controller synthesis for system with continuous (linear) dynamics.

This example is an extension of `robot_discrete.py`,
by including continuous dynamics with disturbances.
The dynamics is linear over a bounded set that is a polytope.
"""
# Petter Nilsson (pettni@kth.se)
# August 14, 2011
# NO, system and cont. prop definitions based on TuLiP 1.x
# 2 Jul, 2013
# NO, TuLiP 1.x discretization
# 17 Jul, 2013

# Note: This code is commented to allow components to be extracted into
# the tutorial that is part of the users manual.  Comments containing
# strings of the form @label@ are used for this purpose.

# @import_section@
from __future__ import print_function

import logging

import numpy as np
from tulip import spec, synth, hybrid
from polytope import box2poly
from tulip.abstract import prop2part, discretize
from tulip.abstract.plot import plot_partition
# @import_section_end@


logging.basicConfig(level=logging.WARNING)
show = False

# @dynamics_section@
# Problem parameters
input_bound = 1.0
uncertainty = 0.01

# Continuous state space
cont_state_space = box2poly([[0., 3.], [0., 2.]])

# Continuous dynamics
# (continuous-state, discrete-time)
A = np.array([[1.0, 0.], [ 0., 1.0]])
B = np.array([[0.1, 0.], [ 0., 0.1]])
E = np.array([[1,0], [0,1]])

# Available control, possible disturbances
U = input_bound *np.array([[-1., 1.], [-1., 1.]])
W = uncertainty *np.array([[-1., 1.], [-1., 1.]])

# Convert to polyhedral representation
U = box2poly(U)
W = box2poly(W)

# Construct the LTI system describing the dynamics
sys_dyn = hybrid.LtiSysDyn(A, B, E, None, U, W, cont_state_space)
# @dynamics_section_end@

# @partition_section@
# Define atomic propositions for relevant regions of state space
cont_props = {}
cont_props['home'] = box2poly([[0., 1.], [0., 1.]])
cont_props['lot'] = box2poly([[2., 3.], [1., 2.]])

# Compute the proposition preserving partition of the continuous state space
cont_partition = prop2part(cont_state_space, cont_props)
plot_partition(cont_partition) if show else None
# @partition_section_end@

# @discretize_section@
# Given dynamics & proposition-preserving partition, find feasible transitions
disc_dynamics = discretize(
    cont_partition, sys_dyn, closed_loop=True,
    N=8, min_cell_volume=0.1, plotit=show
)
# @discretize_section_end@

# Visualize transitions in continuous domain (optional)
plot_partition(disc_dynamics.ppp, disc_dynamics.ts,
               disc_dynamics.ppp2ts) if show else None

# Specifications
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
specs.qinit = '\E \A'

# @synthesize_section@
# Synthesize
ctrl = synth.synthesize(specs,
                        sys=disc_dynamics.ts, ignore_sys_init=True)
assert ctrl is not None, 'unrealizable'


# Generate a graphical representation of the controller for viewing
if not ctrl.save('continuous.png'):
    print(ctrl)
# @synthesize_section_end@

# Simulation

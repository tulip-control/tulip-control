#!/usr/bin/env python
#
# WARNING: This example may not yet be working.  Please check again in
#          the upcoming release.
#
"""
Continuous example with MATLAB export at the end. Mostly identical to
examples/robot_planning/continuous.py
"""
#
# Note: This code is commented to allow components to be extracted into
# the tutorial that is part of the users manual.  Comments containing
# strings of the form @label@ are used for this purpose.
#
# @import_section@
import sys
sys.path.append('../')
import numpy as np
import tomatlab
from tulip import spec, synth, hybrid
from polytope import box2poly
from tulip.abstract import prop2part, discretize, find_controller
# @import_section_end@

visualize = False
from tulip.abstract.plot import plot_partition

# @dynamics_section@
# Problem parameters
input_bound = 1.0
uncertainty = 0.01

# Continuous state space
cont_state_space = box2poly([[0., 3.], [0., 2.]])

# Continuous dynamics
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
plot_partition(cont_partition, show=visualize)
# @partition_section_end@

# @discretize_section@
# Given dynamics & proposition-preserving partition, find feasible transitions
disc_params = {'closed_loop':True, 'N':8, 'min_cell_volume':0.1,
               'plotit':visualize, 'conservative':False}
disc_dynamics = discretize(cont_partition, sys_dyn, **disc_params)
# @discretize_section_end@

"""Visualize transitions in continuous domain (optional)"""
plot_partition(disc_dynamics.ppp, disc_dynamics.ts,
               disc_dynamics.ppp2ts, show=visualize)

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
# Unrealizable spec ?
if ctrl is None:
    sys.exit()

# Export Simulink Model
tomatlab.export('robot_continuous.mat', ctrl, sys_dyn, disc_dynamics,
                disc_params)

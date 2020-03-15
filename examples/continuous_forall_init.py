#!/usr/bin/env python
"""Simulation example with continuous dynamics."""
from __future__ import division
from __future__ import print_function

import logging
import random

import numpy as np
import polytope as pc
from polytope import box2poly
try:
    from matplotlib import pyplot as plt
except ImportError:
    plt = None

from tulip import hybrid, spec, synth
from tulip.abstract import prop2part, discretize
from tulip.abstract.plot import plot_partition
from tulip.abstract import find_controller
from tulip.abstract.plot import simulate2d, pick_point_in_polytope


logging.basicConfig(level='WARNING')


show = False
# Problem parameters
input_bound = 10.0
uncertainty = 0.01
# Continuous state space
cont_state_space = box2poly([[0., 3.], [0., 2.]])
# Continuous dynamics
A = np.array([[1.0, 0.], [0., 1.0]])
B = np.array([[0.1, 0.], [0., 0.1]])
E = np.array([[1, 0], [0, 1]])
# Available control, possible disturbances
U = input_bound * np.array([[-1., 1.], [-1., 1.]])
W = uncertainty * np.array([[-1., 1.], [-1., 1.]])
# Convert to polyhedral representation
U = box2poly(U)
W = box2poly(W)
# Construct the LTI system describing the dynamics
sys_dyn = hybrid.LtiSysDyn(A, B, E, None, U, W, cont_state_space)
# Define atomic propositions for relevant regions of state space
cont_props = {}
cont_props['home'] = box2poly([[0., 1.], [0., 1.]])
cont_props['lot'] = box2poly([[2., 3.], [1., 2.]])
# Compute the proposition preserving partition of
# the continuous state space
cont_partition = prop2part(cont_state_space, cont_props)
plot_partition(cont_partition) if show else None
# Given dynamics & proposition-preserving partition,
# find feasible transitions
disc_dynamics = discretize(
    cont_partition, sys_dyn, closed_loop=False,
    conservative=True,
    N=5, min_cell_volume=0.1, plotit=show)
# Visualize transitions in continuous domain (optional)
plot_partition(disc_dynamics.ppp, disc_dynamics.ts,
               disc_dynamics.ppp2ts) if show else None
#
# Specification
# Environment variables and assumptions
env_vars = {'park'}
env_init = {'X0reach'}  # qinit == '\A \A'
env_prog = '!park'
env_safe = set()
# System variables and requirements
sys_vars = {'X0reach'}
sys_init = set()  # qinit == '\A \A'
sys_prog = {'home'}  # []<>home
sys_safe = {'(X(X0reach) <-> lot) || (X0reach && !park)'}
sys_prog |= {'X0reach'}
# Create the specification
specs = spec.GRSpec(env_vars, sys_vars, env_init, sys_init,
                    env_safe, sys_safe, env_prog, sys_prog)
specs.qinit = '\A \A'
#
# Synthesize
disc_dynamics.ts.states.initial.add_from(disc_dynamics.ts.states)
ctrl = synth.synthesize(specs,
                        sys=disc_dynamics.ts,
                        ignore_sys_init=False)
assert ctrl is not None, 'unrealizable'
# Generate a graphical representation of the controller for viewing
if not ctrl.save('continuous.png'):
    print(ctrl)
#
# Simulation
print('\n Simulation starts \n')
T = 100
# let us pick an environment signal
env_inputs = [{'park': random.randint(0, 1)} for b in range(T + 1)]

# Set up parameters for get_input()
disc_dynamics.disc_params['conservative'] = True
disc_dynamics.disc_params['closed_loop'] = False


def pick_initial_state(ctrl, disc_dynamics):
    """Construct initial discrete and continuous state

    for `qinit == '\A \A'`.
    """
    # pick initial discrete state
    init_edges = ctrl.edges('Sinit', data=True)
    u, v, edge_data = next(iter(init_edges))
    assert u == 'Sinit', u
    d_init = edge_data
    # pick initial continuous state
    s0_part = edge_data['loc']
    init_poly = disc_dynamics.ppp.regions[s0_part].list_poly[0]
    x_init = pick_point_in_polytope(init_poly)
    s0_part_ = find_controller.find_discrete_state(
        x_init, disc_dynamics.ppp)
    assert s0_part == s0_part_, (s0_part, s0_part_)
    return d_init, x_init


# for `qinit == '\A \A'`
d_init, x_init = pick_initial_state(ctrl, disc_dynamics)
simulate2d(env_inputs, sys_dyn, ctrl, disc_dynamics, T,
           d_init=d_init, x_init=x_init, qinit=specs.qinit)

#!/usr/bin/env python
"""Controller synthesis for system with piecewise-affine continuous dynamics."""
# This example is an extension of `robot_continuous.py`
# by Petter Nilsson and Nok Wongpiromsarn.
# Necmiye Ozay, August 26, 2012
from __future__ import print_function

import numpy as np
from polytope import box2poly
from tulip.abstract import discretize
from tulip.abstract import prop2part
from tulip.abstract.plot import plot_strategy
from tulip.hybrid import LtiSysDyn
from tulip.hybrid import PwaSysDyn
from tulip import spec
from tulip import synth


# set to `True` if `matplotlib.pyplot` is available
plotting = False

# Problem parameters
input_bound = 0.4
uncertainty = 0.05

# Continuous state space
cont_state_space = box2poly([[0., 3.], [0., 2.]])

# Assume, for instance, our robot is traveling on
# a nonhomogenous surface (xy plane),
# resulting in different dynamics at different
# parts of the plane.
#
# Since the continuous state space in this example
# is just xy position, different dynamics in
# different parts of the surface can be modeled
# using LtiSysDyn subsystems subsys0 and subsys1.
#
# Together they comprise a Piecewise Affine System:

# @subsystem0@
def subsys0():
    A = np.array([[1.1052, 0.], [ 0., 1.1052]])
    B = np.array([[1.1052, 0.], [ 0., 1.1052]])
    E = np.array([[1,0], [0,1]])

    U = box2poly([[-1., 1.], [-1., 1.]])
    U.scale(input_bound)

    W = box2poly([[-1., 1.], [-1., 1.]])
    W.scale(uncertainty)

    dom = box2poly([[0., 3.], [0.5, 2.]])

    sys_dyn = LtiSysDyn(A, B, E, None, U, W, dom)
    #sys_dyn.plot()

    return sys_dyn
# @subsystem0_end@

# @subsystem1@
def subsys1():
    A = np.array([[0.9948, 0.], [0., 1.1052]])
    B = np.array([[-1.1052, 0.], [0., 1.1052]])
    E = np.array([[1, 0], [0, 1]])

    U = box2poly([[-1., 1.], [-1., 1.]])
    U.scale(input_bound)

    W = box2poly([[-1., 1.], [-1., 1.]])
    W.scale(uncertainty)

    dom = box2poly([[0., 3.], [0., 0.5]])

    sys_dyn = LtiSysDyn(A, B, E, None, U, W, dom)
    #sys_dyn.plot()

    return sys_dyn
# @subsystem1_end@

# @pwasystem@
subsystems = [subsys0(), subsys1()]

# Build piecewise affine system from its subsystems
sys_dyn = PwaSysDyn(subsystems, cont_state_space)
if plotting:
    ax = sys_dyn.plot()
    ax.figure.savefig('pwa_sys_dyn.pdf')
# @pwasystem_end@

# Continuous proposition
cont_props = {}
cont_props['home'] = box2poly([[0., 1.], [0., 1.]])
cont_props['lot'] = box2poly([[2., 3.], [1., 2.]])

# Compute the proposition preserving partition
# of the continuous state space
cont_partition = prop2part(cont_state_space, cont_props)
if plotting:
    ax = cont_partition.plot()
    cont_partition.plot_props(ax=ax)
    ax.figure.savefig('spec_ppp.pdf')

disc_dynamics = discretize(
    cont_partition, sys_dyn, closed_loop=True,
    N=8, min_cell_volume=0.1, plotit=plotting, save_img=True,
    cont_props=cont_props)
if plotting:
    ax = disc_dynamics.plot(show_ts=True)
    ax.figure.savefig('abs_pwa.pdf')
    disc_dynamics.ts.save('ts.pdf')

# Specifications

# Environment variables and assumptions
env_vars = {'park'}
env_init = set()                # empty set
env_prog = '!park'
env_safe = set()                # empty set

# System variables and requirements
sys_vars = {'X0reach'}

# []<>home
sys_prog = {'home'}

# [](park -> <> lot)
sys_init = {'X0reach'}
sys_safe = {'X(X0reach) <-> lot || (X0reach && !park)'}
sys_prog |= {'X0reach'}

# Create the specification
specs = spec.GRSpec(env_vars, sys_vars, env_init, sys_init,
                    env_safe, sys_safe, env_prog, sys_prog)
specs.moore = True
specs.qinit = '\E \A'

# Synthesize
ctrl = synth.synthesize(specs,
                        sys=disc_dynamics.ts, ignore_sys_init=True)
assert ctrl is not None, 'unrealizable'
if plotting:
    ax = plot_strategy(disc_dynamics, ctrl)
    ax.figure.savefig('pwa_proj_mealy.pdf')

# Save graphical representation of controller for viewing
if not ctrl.save('pwa.png'):
    print(ctrl)

# Simulation

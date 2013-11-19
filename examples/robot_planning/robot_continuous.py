#!/usr/bin/env python
"""
This example is an extension of robot_discrete.py by including continuous
dynamics with disturbances.

Petter Nilsson (pettni@kth.se)
August 14, 2011

NO, system and cont. prop definitions based on TuLiP 1.x
2 Jul, 2013
NO, TuLiP 1.x discretization
17 Jul, 2013
"""
import numpy as np

from tulip import spec, synth, hybrid
import tulip.polytope as pc
from tulip.abstract import prop2part, discretize

visualize = False
if visualize:
    import networkx as nx
    from tulip.polytope.plot import plot_partition
else:
    def plot_partition(a, b=None):
        return

# Problem parameters
input_bound = 1.0
uncertainty = 0.01

"""Quotient partition induced by propositions"""
# Continuous state space
cont_state_space = pc.Polytope.from_box([[0., 3.],[0., 2.]])

# Continuous propositions
cont_props = {}
cont_props['home'] = pc.Polytope.from_box([[0., 1.],[0., 1.]])
cont_props['lot'] = pc.Polytope.from_box([[2., 3.],[1., 2.]])

# Compute the proposition preserving partition of the continuous state space
cont_partition = prop2part.prop2part(cont_state_space, cont_props)
plot_partition(cont_partition)

"""Dynamics abstracted to discrete transitions, given initial partition"""
# Continuous dynamics
A = np.array([[1.0, 0.],[ 0., 1.0]])
B = np.array([[0.1, 0.],[ 0., 0.1]])
E = np.array([[1,0],[0,1]])

# available control, possible disturbances
U = input_bound *np.array([[-1., 1.],[-1., 1.]])
W = uncertainty *np.array([[-1., 1.],[-1., 1.]])

U = pc.Polytope.from_box(U)
W = pc.Polytope.from_box(W)

sys_dyn = hybrid.LtiSysDyn(A, B, E, [], U, W, cont_state_space)

# Given dynamics & proposition-preserving partition, find feasible transitions
disc_dynamics = discretize.discretize(
    cont_partition, sys_dyn, closed_loop=True,
    N=8, min_cell_volume=0.1, verbose=10, plotting=True
)

"""Visualize transitions in continuous domain (optional)"""
if visualize:
    plot_partition(
        disc_dynamics.ppp,
        np.array(nx.to_numpy_matrix(disc_dynamics.ofts) )
    )
    print(disc_dynamics.ofts)

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
sys_safe = {'X(X0reach) == lot || (X0reach && !park)'}
sys_prog |= {'X0reach'}

# Create the specification
specs = spec.GRSpec(env_vars, sys_vars, env_init, sys_init,
                    env_safe, sys_safe, env_prog, sys_prog)

"""Synthesize"""
ctrl = synth.synthesize('jtlv', specs, disc_dynamics.ofts)

# Unrealizable spec ?
if isinstance(ctrl, list):
    for counterexample in ctrl:
        print('counterexamples: ' +str(ctrl) +'\n')
    exit(1)

# Generate a graphical representation of the controller for viewing
if not ctrl.save('robot_continuous.png', 'png'):
    print(ctrl)

# Simulation

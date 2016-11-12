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
#
# Note: This code is commented to allow components to be extracted into
# the tutorial that is part of the users manual.  Comments containing
# strings of the form @label@ are used for this purpose.

import logging
logging.basicConfig(level=logging.INFO)

# @import_section@
import numpy as np
import polytope as pc
from tulip import spec, synth, hybrid
from polytope import box2poly
from tulip.abstract import prop2part, discretize
from tulip.abstract.plot import plot_partition
from tulip.abstract.find_controller import *
#import find_controller_LP
# @import_section_end@
show = False

# @dynamics_section@
# Problem parameters
input_bound = 10.0
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
plot_partition(cont_partition) if show else None
# @partition_section_end@

# @discretize_section@
# Given dynamics & proposition-preserving partition, find feasible transitions
disc_dynamics = discretize(
    cont_partition, sys_dyn, closed_loop=False,
    conservative=True,
    N=5, min_cell_volume=0.1, plotit=show
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
ctrl = synth.synthesize('omega', specs,
                        sys=disc_dynamics.ts, ignore_sys_init=True)
assert ctrl is not None, 'unrealizable'

# Generate a graphical representation of the controller for viewing
if not ctrl.save('continuous.png'):
    print(ctrl)
# @synthesize_section_end@

# Simulation

print '\n Simulation starts \n'
T = 100;

# let us pick an environment signal
# from: http://code.activestate.com/recipes/577944-random-binary-list/

from random import *
randParkSignal = [randint(0,1) for b in range(1,T+1)]

# initialization is a bit hacky (need to be consistent with ctrl init
# states)

s0_part = ctrl[ctrl.edges('Sinit')[1][0]][ctrl.edges('Sinit')[1][1]][0]['loc']

init_poly_v = pc.extreme(disc_dynamics.ppp[s0_part][0])
x_init = sum(init_poly_v)/init_poly_v.shape[0]
x = [x_init[0]]
y = [x_init[1]]

N = disc_dynamics.disc_params['N']
s0_part = find_discrete_state([x[0],y[0]],disc_dynamics.ppp)
ctrl = synth.determinize_machine_init(ctrl, {'loc':s0_part})
(s, dum) = ctrl.reaction('Sinit', {'park':randParkSignal[0]})
print dum,'\n'
for i in range(0,T):
    (s, dum) = ctrl.reaction(s, {'park':randParkSignal[i]})
    u = get_input(
            np.array([x[i*N],y[i*N]]),
            sys_dyn,
            disc_dynamics,
            s0_part,
            disc_dynamics.ppp2ts.index(dum['loc']),
            mid_weight=5,
            test_result=True)
    for ind in range(N):
        s_now = np.dot(
                    sys_dyn.A, [x[-1],y[-1]]
                    ) + np.dot(sys_dyn.B,u[ind])
        x.append(s_now[0])
        y.append(s_now[1])

    s0_part = find_discrete_state([x[-1],y[-1]],disc_dynamics.ppp)
    s0_loc = disc_dynamics.ppp2ts[s0_part]
    print s0_loc, dum['loc']
    print dum,'\n'

show_traj = True
if show_traj:
    import matplotlib.pyplot as plt
    plt.plot(x)
    plt.plot(y)
    plt.show()

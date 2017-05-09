"""test hybrid construction"""
from __future__ import print_function

import logging
logging.basicConfig(level=logging.INFO)

import time
import numpy as np

import matplotlib as mpl
mpl.use('Agg')

from tulip import abstract, hybrid
from polytope import box2poly

input_bound = 0.4
uncertainty = 0.05

cont_state_space = box2poly([[0., 3.], [0., 2.]])

cont_props = {}
cont_props['home'] = box2poly([[0., 1.], [0., 1.]])
cont_props['lot'] = box2poly([[2., 3.], [1., 2.]])

sys_dyn = dict()

allh = [0.5, 1.1, 1.5]

modes = []
modes.append(('normal', 'fly'))
modes.append(('refuel', 'fly'))
modes.append(('emergency', 'fly'))

"""First PWA mode"""
def subsys0(h):
    A = np.array([[1.1052, 0.], [ 0., 1.1052]])
    B = np.array([[1.1052, 0.], [ 0., 1.1052]])
    E = np.array([[1,0], [0,1]])

    U = box2poly([[-1., 1.], [-1., 1.]])
    U.scale(input_bound)

    W = box2poly([[-1., 1.], [-1., 1.]])
    W.scale(uncertainty)

    dom = box2poly([[0., 3.], [h, 2.]])

    sys_dyn = hybrid.LtiSysDyn(A, B, E, None, U, W, dom)

    return sys_dyn

def subsys1(h):
    A = np.array([[0.9948, 0.], [0., 1.1052]])
    B = np.array([[-1.1052, 0.], [0., 1.1052]])
    E = np.array([[1, 0], [0, 1]])

    U = box2poly([[-1., 1.], [-1., 1.]])
    U.scale(input_bound)

    W = box2poly([[-1., 1.], [-1., 1.]])
    W.scale(uncertainty)

    dom = box2poly([[0., 3.], [0., h]])

    sys_dyn = hybrid.LtiSysDyn(A, B, E, None, U, W, dom)

    return sys_dyn

for mode, h in zip(modes, allh):
    subsystems = [subsys0(h), subsys1(h)]
    sys_dyn[mode] = hybrid.PwaSysDyn(subsystems, cont_state_space)

"""Switched Dynamics"""

# collect env, sys_modes
env_modes, sys_modes = zip(*modes)
msg = 'Found:\n'
msg += '\t Environment modes: ' + str(env_modes)
msg += '\t System modes: ' + str(sys_modes)

switched_dynamics = hybrid.SwitchedSysDyn(
    disc_domain_size=(len(env_modes), len(sys_modes)),
    dynamics=sys_dyn,
    env_labels=env_modes,
    disc_sys_labels=sys_modes,
    cts_ss=cont_state_space
)

print(switched_dynamics)

ppp = abstract.prop2part(cont_state_space, cont_props)
ppp, new2old = abstract.part2convex(ppp)

"""Discretize to establish transitions"""
start = time.time()

N = 8
trans_len=1

disc_params = {}
for mode in modes:
    disc_params[mode] = {'N':N, 'trans_length':trans_len}

swab = abstract.multiproc_discretize_switched(
    ppp, switched_dynamics, disc_params,
    plot=True, show_ts=True
)
print(swab)
axs = swab.plot(show_ts=True)
for i, ax in enumerate(axs):
    ax.figure.savefig('swab_' + str(i) + '.pdf')

#ax = sys_ts.ts.plot()

elapsed = (time.time() - start)
print('Discretization lasted: ' + str(elapsed))

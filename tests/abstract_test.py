"""test correctness of transition directions"""
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np

from tulip import abstract, hybrid
from tulip.polytope import box2poly

input_bound = 0.4

modes = []
modes.append(('normal', 'fly'))
modes.append(('refuel', 'fly'))
env_modes, sys_modes = zip(*modes)

def subsys0():
    dom = box2poly([[0., 3.], [0., 2.]])
    
    A = np.eye(2)
    B = np.eye(2)
    
    U = box2poly([[0., 1.],
                  [0., 1.]])
    U.scale(input_bound)
    
    sys_dyn = hybrid.LtiSysDyn(A, B, Uset=U, domain=dom)
    
    return sys_dyn

def subsys1():
    dom = box2poly([[0., 3.], [0., 2.]])
    
    A = np.eye(2)
    B = np.eye(2)
    
    U = box2poly([[0., 0.],
                  [-1., 0.]])
    U.scale(input_bound)
    
    sys_dyn = hybrid.LtiSysDyn(A, B, Uset=U, domain=dom)
    
    return sys_dyn

cont_state_space = box2poly([[0., 3.], [0., 2.]])
pwa_sys = dict()
pwa_sys[('normal', 'fly')] = hybrid.PwaSysDyn([subsys0()], cont_state_space)
pwa_sys[('refuel', 'fly')] = hybrid.PwaSysDyn([subsys1()], cont_state_space)

switched_dynamics = hybrid.HybridSysDyn(
    disc_domain_size=(len(env_modes), len(sys_modes)),
    dynamics=pwa_sys,
    env_labels=env_modes,
    disc_sys_labels=sys_modes,
    cts_ss=cont_state_space
)

cont_props = {}
cont_props['home'] = box2poly([[0., 1.], [0., 1.]])
cont_props['lot'] = box2poly([[2., 3.], [1., 2.]])

ppp = abstract.prop2part(cont_state_space, cont_props)
ppp, new2old = abstract.part2convex(ppp)

N = 8
trans_len=1

disc_params = {}
for mode in modes:
    disc_params[mode] = {'N':N, 'trans_length':trans_len}

swab = abstract.discretize_switched(
    ppp, switched_dynamics, disc_params,
    plot=True, show_ts=True, only_adjacent=False
)

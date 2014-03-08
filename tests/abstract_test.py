"""
Tests for the abstraction from continuous dynamics to logic
"""
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np

from tulip import abstract, hybrid
from tulip.polytope import box2poly

input_bound = 0.4

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

def transition_directions_test():
    """
    unit test for correctness of abstracted transition directions, with:
    
      - uni-directional control authority
      - no disturbance
    """
    modes = []
    modes.append(('normal', 'fly'))
    modes.append(('refuel', 'fly'))
    env_modes, sys_modes = zip(*modes)
    
    cont_state_space = box2poly([[0., 3.], [0., 2.]])
    pwa_sys = dict()
    pwa_sys[('normal', 'fly')] = hybrid.PwaSysDyn(
        [subsys0()], cont_state_space
    )
    pwa_sys[('refuel', 'fly')] = hybrid.PwaSysDyn(
        [subsys1()], cont_state_space
    )
    
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
    
    ts = swab.modes[('normal', 'fly')].ts
    edges = {('s0', 's0'), ('s1', 's1'), ('s2', 's2'), ('s3', 's3'),
             ('s4', 's4'), ('s5', 's5'),
             ('s1', 's2'), ('s1', 's4'),
             ('s2', 's3'), ('s2', 's5'), ('s2', 's0'),
             ('s3', 's0'),
             ('s4', 's5'),
             ('s5', 's0')}
    assert(set(ts.edges() ) == edges)
    
    ts = swab.ts
    edges.remove(('s2', 's0'))
    assert(set(ts.edges() ) == edges)
    for i, j in edges:
        assert(ts[i][j][0]['env_actions'] == 'normal')
        assert(ts[i][j][0]['sys_actions'] == 'fly')

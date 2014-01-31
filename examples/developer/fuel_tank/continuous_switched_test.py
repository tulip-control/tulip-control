"""test hybrid construction"""
import logging
logging.basicConfig(level=logging.INFO)

import time
import numpy as np

from tulip import abstract, hybrid
from tulip.polytope import box2poly

input_bound = 0.4
uncertainty = 0.05

cont_state_space = box2poly([[0., 3.], [0., 2.]])

cont_props = {}
cont_props['home'] = box2poly([[0., 1.], [0., 1.]])
cont_props['lot'] = box2poly([[2., 3.], [1., 2.]])

sys_dyn = dict()

modes = []
modes.append(('normal', 'fly'))
modes.append(('refuel', 'fly'))

"""First PWA mode"""
def subsys0():
    A = np.array([[1.1052, 0.], [ 0., 1.1052]])
    B = np.array([[1.1052, 0.], [ 0., 1.1052]])
    E = np.array([[1,0], [0,1]])
    
    U = box2poly([[-1., 1.], [-1., 1.]])
    U.scale(input_bound)
    
    W = box2poly([[-1., 1.], [-1., 1.]])
    W.scale(uncertainty)
    
    dom = box2poly([[0., 3.], [0.5, 2.]])
    
    sys_dyn = hybrid.LtiSysDyn(A, B, E, [], U, W, dom)
    #sys_dyn.plot()
    
    return sys_dyn

def subsys1():
    A = np.array([[0.9948, 0.], [0., 1.1052]])
    B = np.array([[-1.1052, 0.], [0., 1.1052]])
    E = np.array([[1, 0], [0, 1]])
    
    U = box2poly([[-1., 1.], [-1., 1.]])
    U.scale(input_bound)
    
    W = box2poly([[-1., 1.], [-1., 1.]])
    W.scale(uncertainty)
    
    dom = box2poly([[0., 3.], [0., 0.5]])
    
    sys_dyn = hybrid.LtiSysDyn(A, B, E, [], U, W, dom)
    #sys_dyn.plot()
    
    return sys_dyn

subsystems0 = [subsys0(), subsys1()]
sys_dyn[modes[0]] = hybrid.PwaSysDyn(subsystems0, cont_state_space)

"""Second PWA mode"""
def subsys2():
    A = np.array([[1.1052, 0.], [ 0., 1.1052]])
    B = np.array([[1.1052, 0.], [ 0., 1.1052]])
    E = np.array([[1,0], [0,1]])
    
    U = box2poly([[-1., 1.], [-1., 1.]])
    U.scale(input_bound)
    
    W = box2poly([[-1., 1.], [-1., 1.]])
    W.scale(uncertainty)
    
    dom = box2poly([[0., 3.], [1.0, 2.]])
    
    sys_dyn = hybrid.LtiSysDyn(A, B, E, [], U, W, dom)
    #sys_dyn.plot()
    
    return sys_dyn

def subsys3():
    A = np.array([[0.9948, 0.], [0., 1.1052]])
    B = np.array([[-1.1052, 0.], [0., 1.1052]])
    E = np.array([[1, 0], [0, 1]])
    
    U = box2poly([[-1., 1.], [-1., 1.]])
    U.scale(input_bound)
    
    W = box2poly([[-1., 1.], [-1., 1.]])
    W.scale(uncertainty)
    
    # the only difference is here
    dom = box2poly([[0., 3.], [0., 1.0]])
    
    sys_dyn = hybrid.LtiSysDyn(A, B, E, [], U, W, dom)
    #sys_dyn.plot()
    
    return sys_dyn

subsystems1 = [subsys2(), subsys3()]
sys_dyn[modes[1]] = hybrid.PwaSysDyn(subsystems1, cont_state_space)


"""Switched Dynamics"""

# collect env, sys_modes
env_modes, sys_modes = zip(*modes)
msg = 'Found:\n'
msg += '\t Environment modes: ' + str(env_modes)
msg += '\t System modes: ' + str(sys_modes)

switched_dynamics = hybrid.HybridSysDyn(
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

sys_ts = abstract.discretize_switched(
    ppp, switched_dynamics, disc_params)

print(sys_ts)
sys_ts.ppp.plot()
sys_ts.ts.plot()

elapsed = (time.time() - start)
print('Discretization lasted: ' + str(elapsed))

"""
Aerial refueling double tank balancing example.

The system consists of two fuel tanks, T1 and T2.
Fuel is drawn at a constant rate from T2.
There are two dynamic modes:

    - 'normal operation' and
    - 'air refuel mode'.

In 'air refuel mode' fuel is added
to T1 at a constant rate.

The objective is to keep the fuel levels in
the tanks close to each other.

reference
=========
    Nilsson P.; Ozay N.; Topcu U.; Murray R.M.
     Temporal Logic Control of Switched Affine Systems
     with an Application in Fuel Balancing
    2012 American Control Conference
"""

import os
import pickle
import numpy as np

from tulip import hybrid, abstract, spec, synth
import polytope as pc
from tulip.abstract.plot import plot_strategy

import sys
sys.path.append('../')
import tomatlab

"""Problem variables"""
tank_capacity = 10      # Maximum tank capacity
fuel_consumption = 1    # Rate at which fuel is drawn from tank 2
refill_rate = 3         # Rate at which fuel is refilled in tank 1 during refuel mode
input_lb = 0            # Lower bound on fuel move rate from tank 1 to 2
input_ub = 3            # Upper bound on fuel move rate from tank 1 to 2
max_vol_diff = 2        # Enforce |v1-v2| to be lower than this value
fin_vol_diff = 1        # Enforce |v1-v2| to alw. ev. be lower than this value
max_refuel_level = 8    # Above this level no refueling will take place
N = 1                   # Horizon length
disturbance = 0.0       # Absolute uncertainty in fuel consumption
init_lower = 6          # Lower bound for possible initial volumes
init_upper = 8          # Upper bound for possible initial volumes


"""State space and propositions"""
cont_ss = pc.Polytope(
    np.array([[1,0],
              [-1,0],
              [0,1],
              [0,-1]]),
    np.array([tank_capacity, 0, tank_capacity, 0])
)

cont_props = {}
cont_props['no_refuel'] = pc.Polytope(
    np.array([[1,0],
              [-1,0],
              [0,1],
              [0,-1]]),
    np.array([tank_capacity, 0, tank_capacity, -max_refuel_level])
)
cont_props['vol_diff'] = pc.Polytope(
    np.array([[-1,0],
              [0,-1],
              [-1,1],
              [1,-1]]),
    np.array([0,0,max_vol_diff,max_vol_diff])
)
cont_props['initial'] = pc.Polytope(
    np.array([[1,0],
             [-1,0],
             [0, 1],
             [0,-1]]),
    np.array([init_upper, -init_lower, init_upper, -init_lower])
)
cont_props['critical'] = pc.Polytope(
    np.array([[-1,0],
              [0,-1],
              [1,1]]),
    np.array([0, 0, 2*fuel_consumption])
)

"""Dynamics"""
A = np.eye(2)
B = np.array([[-1],[1]])
E = np.array([[0],[1]])
K1 = np.array([[0.],[-fuel_consumption]])
K2 = np.array([[refill_rate],[-fuel_consumption]])
U1 = pc.Polytope(
    np.array([
        [1,0,0],
        [-1,0,0],
        [1,-1,0]
    ]),
    np.array([input_ub, -input_lb, 0])
)
U2 = pc.Polytope(
    np.array([
        [1,0,0],
        [-1,0,0],
        [1,-1,0]
    ]),
    np.array([input_ub,-input_lb, refill_rate])
)
W = pc.Polytope(np.array([[1],[-1]]),
                np.array([disturbance, disturbance]))

# Normal operation dynamics
cont_dyn_normal = hybrid.LtiSysDyn(A, B, E, K1, U1, W, domain=cont_ss)

# Aerial refueling mode dynamics
cont_dyn_refuel = hybrid.LtiSysDyn(A, B, E, K2, U2, W, domain=cont_ss)

"""Switched Dynamics"""
env_modes = ('normal', 'refuel')
sys_modes = ('fly',)

pwa_normal = hybrid.PwaSysDyn([cont_dyn_normal], domain=cont_ss)
pwa_refuel = hybrid.PwaSysDyn([cont_dyn_refuel], domain=cont_ss)

dynamics_dict = {
    ('normal', 'fly') : pwa_normal,
    ('refuel', 'fly') : pwa_refuel
}

switched_dynamics = hybrid.SwitchedSysDyn(
    cts_ss=cont_ss,
    disc_domain_size=(len(env_modes), len(sys_modes)),
    dynamics=dynamics_dict,
    env_labels=env_modes,
    disc_sys_labels=sys_modes
)

"""Create convex proposition preserving partition"""
ppp = abstract.prop2part(cont_ss, cont_props)
ppp, new2old = abstract.part2convex(ppp)


"""Discretize to establish transitions"""

disc_params = {}
disc_params[('normal', 'fly')] = {'N':N, 'trans_length':3}
disc_params[('refuel', 'fly')] = {'N':N, 'trans_length':3}

sys_ts = abstract.discretize_switched(
    ppp, switched_dynamics, disc_params, plot=False
)

"""Specs"""
env_vars = set()
sys_disc_vars = set()

env_init = {'env_actions = normal'}

env_safe = {'no_refuel -> X(env_actions = normal)',
            '(critical & (env_actions = normal)) -> X(env_actions = refuel)',
            '(!critical & env_actions = normal) -> X(env_actions = normal)',
            '(!no_refuel & env_actions = refuel) -> X(env_actions = refuel)'}
env_prog = {'env_actions = refuel'}

# relate switching actions to u_in (env_actions)
sys_init = {'initial'}
sys_safe = {'vol_diff'}
sys_prog = {'True'}

specs = spec.GRSpec(env_vars, sys_disc_vars,
                    env_init, sys_init,
                    env_safe, sys_safe,
                    env_prog, sys_prog)
print(specs.pretty())

"""Synthesis"""
print("Starting synthesis")

ctrl = synth.synthesize(
    'gr1c', specs, sys=sys_ts.ts, ignore_sys_init=True,
)

disc_params = disc_params[('normal', 'fly')]
tomatlab.export('fuel_tank.mat', ctrl, switched_dynamics, sys_ts, disc_params)

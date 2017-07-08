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

CODE KEY:
- ## : beginning of major step; e.g., dynamics definition

For a smaller example, consider adjusting the script to use instead
the following parameters:
tank_capacity = 5
input_ub = 10
max_vol_diff = 1
max_refuel_level = 3
init_lower = 1
init_upper = 3

reference
=========
    Nilsson P.; Ozay N.; Topcu U.; Murray R.M.
     Temporal Logic Control of Switched Affine Systems
     with an Application in Fuel Balancing
    2012 American Control Conference
"""
from __future__ import print_function

import logging
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logging.getLogger('tulip').setLevel(logging.ERROR)
logging.getLogger('omega').setLevel(logging.WARNING)

log = logging.getLogger('multiprocessing')
#log.setLevel(logging.ERROR)

import os
import pickle
import numpy as np
#from scipy import io as sio
#import matplotlib
import matplotlib as mpl
mpl.use('Agg')

from tulip import hybrid, abstract, spec, synth
import polytope as pc
from tulip.abstract.plot import plot_strategy
#from tulip.graphics import newax

## Problem variables
tank_capacity = 10      # Maximum tank capacity
fuel_consumption = 1    # Rate at which fuel is drawn from tank 2
refill_rate = 3         # Rate at which fuel is refilled in tank 1 during refuel mode
input_lb = 0            # Lower bound on fuel move rate from tank 1 to 2
input_ub = 3            # Upper bound on fuel move rate from tank 1 to 2
max_vol_diff = 2        # Enforce |v1-v2| to be lower than this value
fin_vol_diff = 1        # Enforce |v1-v2| to always eventually. be lower than this value
max_refuel_level = 8    # Above this level no refueling will take place
N = 1                   # Horizon length
disturbance = 0.0       # Absolute uncertainty in fuel consumption
init_lower = 6          # Lower bound for possible initial volumes
init_upper = 8          # Upper bound for possible initial volumes
fast = True             # Use smaller domain to increase speed

imgpath = './'
fontsize = 18

## State space and propositions
if fast:
    cont_ss = pc.Polytope(
        np.array([[1,0],
                  [-1,0],
                  [0,1],
                  [0,-1],
                  [1,-1],
                  [-1,1]]),
        np.array([tank_capacity, 1, tank_capacity,
                  1, 2*max_vol_diff, 2*max_vol_diff])
    )
else:
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
cont_props['vol_diff2'] = pc.Polytope(
    np.array([[-1,0],
              [0,-1],
              [-1,1],
              [1,-1]]),
    np.array([0,0,fin_vol_diff,fin_vol_diff])
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

## Dynamics
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

## Switched Dynamics
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

## Create convex proposition preserving partition
ppp = abstract.prop2part(cont_ss, cont_props)
ppp, new2old = abstract.part2convex(ppp)

ax = ppp.plot_props()
ax.figure.savefig(imgpath + 'cprops.pdf')

ax = ppp.plot()
ax.figure.savefig(imgpath + 'ppp.pdf')

## Discretize to establish transitions
if os.name == "posix":
    start = os.times()[2]
    logger.info('start time: ' + str(start))
else:
    logger.info('Timing currently only available for POSIX platforms (not Windows)')

disc_params = {}
disc_params[('normal', 'fly')] = {'N':N, 'trans_length':3}
disc_params[('refuel', 'fly')] = {'N':N, 'trans_length':3}

sys_ts = abstract.multiproc_discretize_switched(
    ppp, switched_dynamics, disc_params, plot=True
)

if os.name == "posix":
    end = os.times()[2]
    logger.info('end time: ' + str(end))
    elapsed = (end - start)
    logger.info('Discretization lasted: ' + str(elapsed))

## Save abstraction to save debugging time
fname = './abstract_switched.pickle'
#pickle.dump(sys_ts, open(fname, 'wb') )

#sys_ts = pickle.load(open(fname, 'r') )

## Specifications
env_vars = set()
sys_disc_vars = set()

env_init = {'env_actions = "normal"'}
#env_init |= {'initial'}

env_safe = {'no_refuel -> X(env_actions = "normal")',
            '(critical & (env_actions = "normal")) -> X(env_actions = "refuel")',
            '(!critical & env_actions = "normal") -> X(env_actions = "normal")',
            '(!no_refuel & env_actions = "refuel") -> X(env_actions = "refuel")'}
env_prog = {'env_actions = "refuel"'}

# relate switching actions to u_in (env_actions)
sys_init = {'initial'}
sys_safe = {'vol_diff'}
sys_prog = {'True'} #{'vol_diff2'}

specs = spec.GRSpec(env_vars, sys_disc_vars,
                    env_init, sys_init,
                    env_safe, sys_safe,
                    env_prog, sys_prog)
print(specs.pretty())

## Synthesis
print("Starting synthesis")
if os.name == "posix":
    start = os.times()[2]

specs.moore = False
specs.qinit = r'\A \E'
ctrl = synth.synthesize(
    specs, sys=sys_ts.ts, ignore_sys_init=True,
    #action_vars=('u_in', 'act')
)
if os.name == "posix":
    end = os.times()[2]
    elapsed = (end - start)
    logger.info('Synthesis lasted: ' + str(elapsed))

logger.info(ctrl)
ctrl.save(imgpath + 'double_tank.pdf')

ax = plot_strategy(sys_ts, ctrl)
ax.figure.savefig(imgpath + 'proj_mealy.pdf')

## Simulate
# num_it = 25
# init_state = {}
# init_state['u_in'] = 0

# destfile = 'rsdisturbance_example.gexf'
# states = grsim.grsim(
#     [aut], env_states=[init_state], num_it=num_it,
#     deterministic_env=False, graph_vis=False,
#     destfile=destfile
# )

# uin_arr = []
# cellid_arr = []
# for (autID, state) in states:
#     uin_arr.append(state.state['u_in'])
#     cellid_arr.append(state.state['cellID'])
# uin_arr = np.array(uin_arr)
# cellid_arr = np.array(cellid_arr)

# rc, x = pc.cheby_ball(new_part.regions[cellid_arr[0]])
# x = x.flatten()
# x_arr = x.copy()
# u_arr = np.zeros(1)
# for i in range(1, len(cellid_arr)):
#     if uin_arr[i-1] == 0:
#         u = abstract.get_input(
#             x, cont_dyn_normal, new_part,
#             cellid_arr[i-1], cellid_arr[i], 1, mid_weight=10
#         )
#         x = np.dot(cont_dyn_normal.A,x).flatten() + \
#             np.dot(cont_dyn_normal.B,u).flatten() + \
#             cont_dyn_normal.K.flatten()
#     else:
#         u = abstract.get_input(
#             x, cont_dyn_refuel, new_part, cellid_arr[i-1],
#             cellid_arr[i], 1, Q=[], mid_weight=10
#         )
#         x = np.dot(cont_dyn_refuel.A,x).flatten() + \
#             np.dot(cont_dyn_refuel.B,u).flatten() + \
#             cont_dyn_refuel.K.flatten()

#     u_arr = np.hstack([u_arr, u.flatten()])
#     x_arr = np.vstack([x_arr, x])

# data = {}
# data['x'] = x_arr
# data['u'] = u_arr[range(1,u_arr.shape[0]), :]
# data['u_in'] = uin_arr

# sio.savemat("matlabdata", data)

# ax = plot_partition(new_part, plot_numbers=False, show=False)
# for tick in ax.xaxis.get_major_ticks():
#     tick.label1.set_fontsize(fontsize)
# for tick in ax.yaxis.get_major_ticks():
#     tick.label1.set_fontsize(fontsize)
# plt.xlabel('$v_1$', fontsize=fontsize+6)
# plt.ylabel('$v_2$', fontsize=fontsize+6)

# # Plot transitions
# arr_size=0.2
# for i in range(1,x_arr.shape[0]):
#     x = x_arr[i-1,0]
#     y = x_arr[i-1,1]
#     dx = x_arr[i,0] - x
#     dy = x_arr[i,1] - y
#     arr = matplotlib.patches.Arrow(
#         float(x), float(y), float(dx),
#         float(dy), width=arr_size
#     )
#     ax.add_patch(arr)

# ax.plot(x_arr[0,0], x_arr[0,1], 'og')
# ax.plot(x_arr[-1,0], x_arr[-1,1], 'or')

# plt.savefig(imgpath + 'simulation.pdf')

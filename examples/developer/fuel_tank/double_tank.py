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
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

polylogger = logging.getLogger('tulip.polytope')
polylogger.setLevel(logging.WARN)

abs_logger = logging.getLogger('tulip.abstract')
abs_logger.setLevel(logging.WARN)

logging.getLogger('tulip.gr1cint').setLevel(logging.DEBUG)

import numpy as np
import time

#import matplotlib
#import matplotlib.pyplot as plt
#from tulip.graphics import newax

from tulip import hybrid, abstract, spec, synth
from tulip import polytope as pc
from tulip.abstract.plot import plot_partition

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
fast = False             # Use smaller domain to increase speed

fontsize = 18

"""State space and propositions"""
if fast:
    cont_ss = pc.Polytope(
        np.array([
            [1,0], [-1,0], [0,1],
            [0,-1], [1,-1], [-1,1]
        ]),
        np.array([
            tank_capacity, 1, tank_capacity,
            1, 2*max_vol_diff, 2*max_vol_diff
        ])
    )
else:
    cont_ss = pc.Polytope(
        np.array([
            [1,0], [-1,0], [0,1], [0,-1]
        ]),
        np.array([
            tank_capacity, 0, tank_capacity, 0
        ])
    )

cont_props = {}
cont_props['no_refuel'] = pc.Polytope(
    np.array([
        [1,0], [-1,0], [0,1], [0,-1]
    ]),
    np.array([
        tank_capacity, 0, tank_capacity, -max_refuel_level
    ])
)
cont_props['vol_diff'] = pc.Polytope(
    np.array([[-1,0],[0,-1],[-1,1],[1,-1]]),
    np.array([0,0,max_vol_diff,max_vol_diff])
)
cont_props['vol_diff2'] = pc.Polytope(
    np.array([
        [-1,0], [0,-1], [-1,1], [1,-1]
    ]),
    np.array([
        0,0,fin_vol_diff,fin_vol_diff
    ])
)
cont_props['initial'] = pc.Polytope(
    np.array([
        [1,0], [-1,0], [0,1], [0,-1]
    ]),
    np.array([
        init_upper, -init_lower, init_upper, -init_lower
    ])
)
cont_props['critical'] = pc.Polytope(
    np.array([
        [-1,0], [0,-1], [1,1]
    ]),
    np.array([
        0, 0, 2*fuel_consumption
    ]))

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

switched_dynamics = hybrid.HybridSysDyn(
    cts_ss=cont_ss,
    disc_domain_size=(len(env_modes), len(sys_modes)),
    dynamics=dynamics_dict,
    env_labels=env_modes,
    disc_sys_labels=sys_modes
)

"""Create convex proposition preserving partition"""
ppp = abstract.prop2part(cont_ss, cont_props)
ppp = abstract.part2convex(ppp)

"""Discretize to establish transitions"""
start = time.time()

sys_ts = abstract.discretize_switched(ppp, switched_dynamics, N)

elapsed = (time.time() - start)
logger.info('Discretization lasted: ' + str(elapsed))

"""Plot partitions"""
"""
ax, fig = newax()
def plotidy(ax):
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    ax.set_xlabel('$v_1$', fontsize=fontsize+6)
    ax.set_ylabel('$v_2$', fontsize=fontsize+6)

disc_ss_normal.ppp.plot(plot_numbers=False, ax=ax)
plotidy(ax)
fig.savefig('part_normal.pdf')

disc_ss_refuel.ppp.plot(plot_numbers=False, ax=ax)
plotidy(ax)
fig.savefig('part_refuel.pdf')

new_part.ppp.plot(plot_numbers=False, ax=ax)
plotidy(ax)
fig.savefig('part_merged.pdf')
"""

"""Specs"""
env_vars = set()
sys_disc_vars = set()

env_init = {'u_in = normal'}
#env_init |= {'initial'}

env_safe = {'no_refuel -> X(u_in = normal)',
            '(critical & (u_in = normal)) -> X(u_in = refuel)',
            '(!critical & u_in = normal) -> X(u_in = normal)',
            '(!no_refuel & u_in = refuel) -> X(u_in = refuel)'}
env_prog = {'u_in = refuel'}

# relate switching actions to u_in
sys_init = {'initial'}
sys_safe = {'vol_diff'}
#sys_prog = {'vol_diff2'}

specs = spec.GRSpec(env_vars, sys_disc_vars,
                    env_init, sys_init,
                    env_safe, sys_safe,
                    env_prog, sys_prog)
print(specs.pretty())

"""Synthesis"""
print("Starting synthesis")
start = time.time()

ctrl = synth.synthesize(
    'gr1c', specs, sys=sys_ts.ts, ignore_sys_init=True,
    actions_must='xor', action_vars=('u_in', 'act')
)
print(ctrl)

elapsed = (time.time() - start)
logger.info('Synthesis lasted: ' + str(elapsed))

exit

"""Simulate"""
num_it = 25
init_state = {}
init_state['u_in'] = 0

destfile = 'rsdisturbance_example.gexf'
states = grsim.grsim([aut], env_states=[init_state], num_it=num_it,
                     deterministic_env=False, graph_vis=False,
                     destfile=destfile)

uin_arr = []
cellid_arr = []
for (autID, state) in states:
    uin_arr.append(state.state['u_in'])
    cellid_arr.append(state.state['cellID'])
uin_arr = np.array(uin_arr)
cellid_arr = np.array(cellid_arr)

rc, x = pc.cheby_ball(new_part.regions[cellid_arr[0]])
x = x.flatten()
x_arr = x.copy()
u_arr = np.zeros(1)
for i in range(1, len(cellid_arr)):
    if uin_arr[i-1] == 0:
        u = abstract.get_input(x, cont_dyn_normal, new_part,
            cellid_arr[i-1], cellid_arr[i], 1, mid_weight=10)
        x = np.dot(cont_dyn_normal.A,x).flatten() + np.dot(cont_dyn_normal.B,u).flatten() + \
            cont_dyn_normal.K.flatten()
    else:
        u = abstract.get_input(x, cont_dyn_refuel, new_part, cellid_arr[i-1],
            cellid_arr[i], 1, Q=[], mid_weight=10)
        x = np.dot(cont_dyn_refuel.A,x).flatten() + np.dot(cont_dyn_refuel.B,u).flatten() + \
            cont_dyn_refuel.K.flatten()
    u_arr = np.hstack([u_arr, u.flatten()])
    x_arr = np.vstack([x_arr, x])
    
data = {}
data['x'] = x_arr
data['u'] = u_arr[range(1,u_arr.shape[0]), :]
data['u_in'] = uin_arr
from scipy import io as sio
sio.savemat("matlabdata", data)

ax = plot_partition(new_part, plot_numbers=False, show=False)
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
plt.xlabel('$v_1$', fontsize=fontsize+6)
plt.ylabel('$v_2$', fontsize=fontsize+6)

# Plot transitions
arr_size=0.2
for i in range(1,x_arr.shape[0]):
    x = x_arr[i-1,0]
    y = x_arr[i-1,1]
    dx = x_arr[i,0] - x
    dy = x_arr[i,1] - y
    arr = matplotlib.patches.Arrow(float(x),float(y),float(dx),float(dy),width=arr_size)
    ax.add_patch(arr)

ax.plot(x_arr[0,0], x_arr[0,1], 'og')
ax.plot(x_arr[-1,0], x_arr[-1,1], 'or')

plt.savefig('simulation.eps')

'''TuLiP double tank example. 

The system consists of two fuel tanks, T1 and T2.
Fuel is drawn at a constant rate from T2. There are two dynamic modes, 'normal
operation' and 'air refuel mode'. In 'air refuel mode' fuel is added to T1
at a constant rate. The objective is to keep the fuel levels in the tanks
close to each other.'''

import numpy as np
import os, sys
from subprocess import call
import re
from copy import deepcopy
import time

import matplotlib
import matplotlib.pyplot as plt

from tulip import *
import tulip.polytope as pc
from tulip.polytope.plot import *

from tank_functions import *

# Problem variables
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

start = time.time()

# Specify where the smv file, spc file and aut file will go
testfile = 'fuel_tank'
smvfile = testfile+'.smv'
spcfile = testfile+'.spc'
autfile = testfile+'.aut'

# Dynamics
A = np.eye(2)
B = np.array([[-1],[1]])
E = np.array([[0],[1]])
K1 = np.array([[0.],[-fuel_consumption]])
K2 = np.array([[refill_rate],[-fuel_consumption]])
U1 = pc.Polytope(np.array([[1,0,0],[-1,0,0],[1,-1,0]]), \
                np.array([input_ub,-input_lb, 0]))
U2 = pc.Polytope(np.array([[1,0,0],[-1,0,0],[1,-1,0]]), \
                np.array([input_ub,-input_lb, refill_rate]))
D = pc.Polytope(np.array([[1],[-1]]), np.array([disturbance, disturbance]))
cont_dyn_normal = discretize.CtsSysDyn(A,B,E,K1,U1,D)    # Normal operation dynamics
cont_dyn_refuel = discretize.CtsSysDyn(A,B,E,K2,U2,D)    # Aerial refueling mode dynamics

# State space and propositions
if fast:
    cont_ss = pc.Polytope(np.array([[1,0],[-1,0],[0,1],[0,-1],[1,-1],[-1,1]]),
                      np.array([tank_capacity,1,tank_capacity,1,2*max_vol_diff,
                      2*max_vol_diff]))
else:
    cont_ss = pc.Polytope(np.array([[1,0],[-1,0],[0,1],[0,-1]]),
                      np.array([tank_capacity,0,tank_capacity,0]))
cont_props = {}
cont_props['no_refuel'] = pc.Polytope(np.array([[1,0],[-1,0],[0,1],[0,-1]]),
                np.array([tank_capacity,0,tank_capacity,-max_refuel_level]))
cont_props['vol_diff'] = pc.Polytope(np.array([[-1,0],[0,-1],[-1,1],[1,-1]]), \
                                     np.array([0,0,max_vol_diff,max_vol_diff]))
cont_props['vol_diff2'] = pc.Polytope(np.array([[-1,0],[0,-1],[-1,1],[1,-1]]), \
                                     np.array([0,0,fin_vol_diff,fin_vol_diff]))
cont_props['initial'] = pc.Polytope(np.array([[1,0],[-1,0],[0,1],[0,-1]]), \
                        np.array([init_upper,-init_lower,init_upper,-init_lower]))
cont_props['critical'] = pc.Polytope(np.array([[-1,0],[0,-1],[1,1]]), \
                                     np.array([0,0,2*fuel_consumption]))

# Create convex proposition preserving partition                      
ppp = prop2part.prop2part2(cont_ss, cont_props)
ppp = prop2part.prop2partconvex(ppp)

# Discretize to establish transitions

disc_ss_normal = discretize.discretize(ppp, cont_dyn_normal, N=N, \
                                trans_length=2, use_mpt=False, \
                                min_cell_volume=.01)
                                
# ax = plot_partition(disc_ss_normal, plot_numbers=False, show=False)
# for tick in ax.xaxis.get_major_ticks():
#     tick.label1.set_fontsize(fontsize)
# for tick in ax.yaxis.get_major_ticks():
#     tick.label1.set_fontsize(fontsize)

#plt.xlabel('$v_1$', fontsize=fontsize+6)
#plt.ylabel('$v_2$', fontsize=fontsize+6)
#plt.savefig('part_normal.eps')
                                      
disc_ss_refuel = discretize.discretize(ppp, cont_dyn_refuel, N=N, \
                                trans_length=3, use_mpt=False, \
                                min_cell_volume=.01)

# ax = plot_partition(disc_ss_refuel, plot_numbers=False, show=False)
# for tick in ax.xaxis.get_major_ticks():
#     tick.label1.set_fontsize(fontsize)
# for tick in ax.yaxis.get_major_ticks():
#     tick.label1.set_fontsize(fontsize)
#plt.xlabel('$v_1$', fontsize=fontsize+6)
#plt.ylabel('$v_2$', fontsize=fontsize+6)
#plt.savefig('part_refuel.eps')

# Merge partitions and get transition matrices for both dynamic modes
# in the merged partition
new_part = merge_partitions(disc_ss_normal, disc_ss_refuel)

# ax = plot_partition(new_part, plot_numbers=False, show=False)
# for tick in ax.xaxis.get_major_ticks():
#     tick.label1.set_fontsize(fontsize)
# for tick in ax.yaxis.get_major_ticks():
#     tick.label1.set_fontsize(fontsize)
#plt.xlabel('$v_1$', fontsize=fontsize+6)
#plt.ylabel('$v_2$', fontsize=fontsize+6)
#plt.savefig('part_merged.eps')

print new_part.num_regions

trans_normal = get_transitions(new_part, cont_dyn_normal, N=1, trans_length=3)
trans_refuel = get_transitions(new_part, cont_dyn_refuel, N=1, trans_length=4)

elapsed = (time.time() - start)
print "Discretization took " + str(elapsed)

# Variable dictionaries
env_vars = {'u_in': "{0, 2}"}
sys_disc_vars = {}

# Specs
# assumption = 'initial'
assumption = ' (u_in=0)'
assumption += '& ([](no_refuel -> next(u_in = 0)))'
#assumption += '& ([]<> (u_in = 2))'
assumption += '& ([]( (critical & (u_in=0)) -> next(u_in = 2)))'
assumption += '& ([]((!critical & u_in=0) -> next(u_in=0)))'
assumption += '& ([]((!no_refuel & u_in=2) -> next(u_in=2)))'
guarantee = 'initial & ([]vol_diff) & ([]<>vol_diff2)'
#guarantee = 'initial & ([]vol_diff)'


asd = raw_input("Starting discretization")
start = time.time()


# Create JTLV files
create_files(new_part, trans_normal, trans_refuel, 'u_in', 0, 2, env_vars, \
            sys_disc_vars, [assumption, guarantee], smvfile, spcfile) 

# Check realizability
realizability = jtlvint.checkRealizability(smv_file=smvfile, spc_file=spcfile,
                                           aut_file=autfile, verbose=3)
                                           
jtlvint.computeStrategy(smv_file=smvfile, spc_file=spcfile, aut_file=autfile,
                        priority_kind=3, verbose=3)
                        
elapsed = (time.time() - start)
print "Synthesis took " + str(elapsed)

                        
aut = automaton.Automaton(autfile, [], 3)          

# Simulate
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

rc, x = pc.cheby_ball(new_part.list_region[cellid_arr[0]])
x = x.flatten()
x_arr = x.copy()
u_arr = np.zeros(1)
for i in range(1, len(cellid_arr)):
    if uin_arr[i-1] == 0:
        u = discretize.get_input(x, cont_dyn_normal, new_part, \
            cellid_arr[i-1], cellid_arr[i], 1, mid_weight=10)
        x = np.dot(cont_dyn_normal.A,x).flatten() + np.dot(cont_dyn_normal.B,u).flatten() + \
            cont_dyn_normal.K.flatten()
    else:
        u = discretize.get_input(x, cont_dyn_refuel, new_part, cellid_arr[i-1], \
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
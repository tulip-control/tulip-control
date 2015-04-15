#!/usr/bin/env python

from tulip import spec, synth, transys, hybrid
import numpy as np
from scipy import sparse as sp
import polytope as pc
from tulip.abstract import prop2part, find_discrete_state

""" 
Set up simple system to test synthesis
"""

env_sws = transys.AFTS()
env_sws.owner = 'env'

env_sws.sys_actions.add_from({'off','on',})

# str states
n = 3
states = transys.prepend_with(range(n), 's')
env_sws.states.add_from(set(states) )
env_sws.states.initial.add('s0')

env_sws.atomic_propositions.add_from(['low','medium','high'])
state_labels = [{'low'}, {'medium'}, {'high'}]

for state, label in zip(states, state_labels):
    env_sws.states.add(state, ap=label)

progmap={}

# mode1 transitions
trans_on = np.array([[1,1,0],
                      [1,0,1],
                      [0,0,1]])
progmap['on'] = ('s0','s1')

env_sws.transitions.add_adj(
    sp.lil_matrix(trans_on), states, sys_actions='on'
)
                      
# mode2 transitions
trans_off = np.array([[1,1,0],
                      [1,0,1],
                      [0,1,0]])
env_sws.transitions.add_adj(
    sp.lil_matrix(trans_off), states, sys_actions='off'
)
progmap['off'] = ('s1','s2')
env_sws.set_progress_map(progmap)
# This is what is visible to the outside world (and will go into synthesis method)

env_vars = set()
env_init = set()
env_prog = set() #{'sys_actions="off" && (eloc="s1" || eloc="s2")','sys_actions="on" && (eloc="s1" || eloc="s0")'}
env_safe = set()   

sys_vars = set()
sys_init = set()
sys_prog = {'low','high'}
sys_safe = set()

specs = spec.GRSpec(env_vars, sys_vars, env_init, sys_init,
                    env_safe, sys_safe, env_prog, sys_prog)

ctrl = synth.synthesize('gr1c', specs, env=env_sws,ignore_env_init=True)

if not ctrl.save('only_mode_controlled.eps'):
    print(ctrl)

cts_ss=pc.box2poly([[0.,3.],[0., 1.]])
cont_props={}
cont_props['low'] = pc.box2poly([[0., 1.], [0., 1.]])
cont_props['medium'] = pc.box2poly([[1., 2.], [0., 1.]])
cont_props['high'] = pc.box2poly([[2.,3.],[1.,1.]])

ppp=prop2part(cts_ss,cont_props)
pc.plot_partition(ppp,show=True)
dynamics={}
A_off = np.array([[1., 0.],[0., 1.]])
B_zero = np.array([[1., 0.],[0., 1.]])
K_on = np.array([[0.1], [0.]])
K_off = np.array([[-0.1], [0.]])

dynamics['on']=hybrid.LtiSysDyn(A_off,  B_zero, None, K_on, None, None, cts_ss)
dynamics['off']=hybrid.LtiSysDyn(A_off,  B_zero, None, K_off, None, None, cts_ss)

T=[1.5]
C=[0.5]
s0_part = find_discrete_state([T[0],C[0]],ppp)
mach = synth.determinize_machine_init(ctrl,{'sys_actions':'on'})
sim_hor = 100
N=1

(s1, dum) = mach.reaction('Sinit', {'eloc': 's1'})

for sim_time in range(sim_hor):
    sysnow=dum['sys_actions']

    for ind in range(N):
        x = np.dot(
                dynamics[sysnow].A, [T[-1],C[-1]]
                ) + dynamics[sysnow].K.flatten()
        T.append(x[0])
        C.append(x[1])

    s0_part = find_discrete_state([T[-1],C[-1]],ppp)
    #s0_loc = disc_dynamics.ppp2ts[s0_part]
    print s1, 's'+str(s0_part), dum['sys_actions']
    if sim_time < 10:
    	(s1, dum) = mach.reaction(s1, {'eloc': 's0'})
    elif sim_time <=50:
    	(s1, dum) = mach.reaction(s1, {'eloc': 's1'})
    else:
    	(s1, dum) = mach.reaction(s1, {'eloc': 's2'})


print "Hello World"

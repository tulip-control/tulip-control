#!/usr/bin/env python

from tulip import spec, synth, transys
import numpy as np
from scipy import sparse as sp

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
print(env_sws)

env_vars = {'human'}
env_init = {'!human'}
env_prog = {'human'}
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


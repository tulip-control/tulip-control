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

env_vars = set()
env_init = set()
env_prog = set()

# manual construct
#prog={'((!low && !medium) || sys_actions != "on")', '((!high && !medium) || sys_actions != "off")'}

#more automated construct (ideally should go into synth.synthesize)

prog =set()
for x in env_sws.sys_actions:
    sp='(('
    for i, y in enumerate(env_sws.progress_map[x]):
        sp+= 'eloc != "'
        sp+=y
        if i!=len(env_sws.progress_map[x])-1:
            sp+='" && '
    sp+='") || sys_actions !="'
    sp+=x
    sp+='")'
    prog|={sp}

env_prog |=prog

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

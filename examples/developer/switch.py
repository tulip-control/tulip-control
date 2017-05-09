"""
Simple example with uncontrolled switching, for debugging

 6 cell robot example.
     +---+---+---+
     | 3 | 4 | 5 |
     +---+---+---+
     | 0 | 1 | 2 |
     +---+---+---+
"""
from __future__ import print_function

from tulip import spec, synth, transys
import numpy as np
from scipy import sparse as sp

sys_swe = transys.FTS()
sys_swe.env_actions.add_from({'sun', 'rain'})

# Environment actions are mutually exclusive.
n = 2
states = ['s' + str(i) for i in range(n) ]
sys_swe.states.add_from(states)
sys_swe.states.initial |= ['s0']

# different transitions possible, depending on weather
transmat1 = sp.lil_matrix(np.array(
                [[0,1],
                 [1,0]]
            ))
sys_swe.transitions.add_adj(transmat1, states, env_actions='sun')

# avoid being killed by environment
transmat2 = sp.lil_matrix(np.array(
                [[1,0],
                 [0,1]]
            ))
sys_swe.transitions.add_adj(transmat2, states, env_actions='rain')

# atomic props
sys_swe.atomic_propositions.add_from(['home', 'lot'])

sys_swe.states.add(states[0], ap={'home'})
sys_swe.states.add(states[1], ap={'lot'})

print(sys_swe)

sys_swe.save('sys_swe.pdf')

# (park & sun) & []<>!park && []<>sum
env_vars = {'park'}
env_init = {'park', 'env_actions = "sun"'}
env_prog = {'!park','env_actions = "sun"'}
env_safe = set()

# (s0 & mem) & []<> home & [](park -> <>lot)
sys_vars = {'mem'}
sys_init = {'mem'}
sys_prog = {'home'}               # []<>home
sys_safe = {'next(mem) <-> lot || (mem && !park)'}
sys_prog |= {'mem'}

# Create the specification
specs = spec.GRSpec(env_vars, sys_vars, env_init, sys_init,
                    env_safe, sys_safe, env_prog, sys_prog)
specs.moore = False
specs.qinit = '\A \E'
# Controller synthesis
ctrl = synth.synthesize(specs, sys=sys_swe,
                        ignore_sys_init=True)

if not ctrl.save('switch.pdf'):
    print(ctrl)

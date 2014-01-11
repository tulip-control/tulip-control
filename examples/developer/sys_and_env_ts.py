"""
example to illustrate the combined use of
an environment and a system transition system.
"""
import logging
logging.basicConfig(filename='sys_and_env_ts.log',
                    level=logging.DEBUG, filemode='w')

from tulip import transys, spec, synth

# the system's spatial layout:
#     +----+----+----+
#     | X3 | X4 | X5 |
#     +----+----+----+
#     | X0 | X1 | X2 |
#     +----+----+----+

sys = transys.FTS()
sys.states.add_from(['X0', 'X1', 'X2', 'X3', 'X4', 'X5'])
sys.states.initial.add_from(['X0', 'X1'])

sys.transitions.add_from({'X0'}, {'X1', 'X3'})
sys.transitions.add_from({'X1'}, {'X0', 'X4', 'X2'})
sys.transitions.add_from({'X2'}, {'X1', 'X5'})
sys.transitions.add_from({'X3'}, {'X0', 'X4'})
sys.transitions.add_from({'X4'}, {'X3', 'X1', 'X5'})
# compared to home_parking, this car can stay in lot next
sys.transitions.add_from({'X5'}, {'X4', 'X2', 'X5'})

sys.atomic_propositions.add_from({'home', 'lot'})
sys.states.label('X0', 'home')
sys.states.label('X5', 'lot')

"""Park as an env AP
"""
env0 = transys.FTS()
env0.states.add_from({'e0', 'e1'})
env0.states.initial.add_from({'e0', 'e1'})

env0.atomic_propositions.add('park')
env0.states.label('e0', 'park')

env0.transitions.add_from({'e0', 'e1'}, {'e0', 'e1'})
logging.info(env0)

# barely realizable: assumption necessary
env_prog = '!park'

sys_vars = {'X0reach'}
sys_init = {'X0reach'}
sys_prog = {'home'}

# one additional requirement: if in lot,
# then stay there until park signal is turned off
sys_safe = {'(X (X0reach) <-> lot) || (X0reach && !park)',
            '((lot & park) -> X(lot))'}
sys_prog |= {'X0reach'}

specs = spec.GRSpec(sys_vars=sys_vars, sys_init=sys_init,
                    sys_safety=sys_safe,
                    env_prog=env_prog, sys_prog=sys_prog)

ctrl = synth.synthesize('gr1c', specs, sys=sys, env=env0)
ctrl.save('sys_and_env_ts0.pdf')
logging.info(ctrl)

"""Park as an env action
"""
env1 = transys.FTS()
env1.states.add('e0')
env1.states.initial.add('e0')

env1.actions.add_from({'park', ''})

env1.transitions.add_labeled('e0', 'e0', 'park')
env1.transitions.add_labeled('e0', 'e0', '')
logging.info(env1)

specs = spec.GRSpec(sys_vars=sys_vars, sys_init=sys_init,
                    sys_safety=sys_safe,
                    env_prog=env_prog, sys_prog=sys_prog)

ctrl = synth.synthesize('gr1c', specs, sys=sys, env=env1)
ctrl.save('sys_and_env_ts1.pdf')
logging.info(ctrl)

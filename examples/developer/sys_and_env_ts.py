"""
example to illustrate the combined use of
an environment and a system transition system.
"""
import logging
logging.basicConfig(filename='sys_and_env_ts.log',
                    level=logging.DEBUG, filemode='w')
logger = logging.getLogger(__name__)

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

sys.transitions.add_from(
    [('X0', x) for x in {'X1', 'X3'}] +
    [('X1', x) for x in {'X0', 'X4', 'X2'}] +
    [('X2', x) for x in {'X1', 'X5'}] +
    [('X3', x) for x in {'X0', 'X4'}] +
    [('X4', x) for x in {'X3', 'X1', 'X5'}] +
    # compared to home_parking, this car can stay in lot next
    [('X5', x) for x in {'X4', 'X2', 'X5'}]
)

sys.atomic_propositions.add_from({'home', 'lot'})
sys.states.add('X0', ap={'home'})
sys.states.add('X5', ap={'lot'})

"""Park as an env AP
"""
env0 = transys.FTS()
env0.owner = 'env'
env0.states.add_from({'e0', 'e1'})
env0.states.initial.add_from({'e0', 'e1'})

env0.atomic_propositions.add('park')
env0.states.add('e0', ap={'park'})

env0.transitions.add_from([
    ('e0', 'e0'), ('e0', 'e1'),
    ('e1', 'e0'), ('e1', 'e1')
])
logger.info(env0)

# barely realizable: assumption necessary
env_prog = '!park'

sys_vars = {'mem'}
sys_init = {'mem'}
sys_prog = {'home'}

# one additional requirement: if in lot,
# then stay there until park signal is turned off
sys_safe = {'(X(mem) <-> lot) || (mem && !park)',
            '((lot && park) -> X(lot))'}
sys_prog |= {'mem'}

specs = spec.GRSpec(sys_vars=sys_vars, sys_init=sys_init,
                    sys_safety=sys_safe,
                    env_prog=env_prog, sys_prog=sys_prog)
specs.moore = False
specs.qinit = '\A \E'
ctrl = synth.synthesize(specs, sys=sys, env=env0)
ctrl.save('sys_and_env_ts0.pdf')
logger.info(ctrl)

"""Park as an env action
"""
env1 = transys.FTS()
env1.owner = 'env'
env1.states.add('e0')
env1.states.initial.add('e0')

env1.env_actions.add_from({'park', 'none'})

env1.transitions.add('e0', 'e0', env_actions='park')
env1.transitions.add('e0', 'e0', env_actions='none')
logger.info(env1)

env_prog = ['! (env_actions = "park")']
sys_safe = {'(X(mem) <-> lot) || (mem && ! (env_actions = "park"))',
            '((lot && (env_actions = "park")) -> X(lot))'}

specs = spec.GRSpec(sys_vars=sys_vars, sys_init=sys_init,
                    sys_safety=sys_safe,
                    env_prog=env_prog, sys_prog=sys_prog)
specs.moore = False
specs.qinit = '\A \E'
ctrl = synth.synthesize('omega', specs, sys=sys, env=env1)
ctrl.save('sys_and_env_ts1.pdf')
env1.save('env1.pdf')
logger.info(ctrl)

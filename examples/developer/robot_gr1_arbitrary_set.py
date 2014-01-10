"""
Two examples for arbitrary finite domains with gr1c:

    - one manually coded
    - one with arbitrary (unnumbered) sys TS states
"""
# will probably need also a flag which type to use

from tulip import spec, synth
from tulip import transys as trs

env_vars = {'park'}
env_init = set()
env_safe = set()
env_prog = '!park'

sys = trs.FTS()

sys_vars = {}
sys_vars['loc'] = ['a', 'b', 'c', 'd', 'e', 'foo']

sys_init = {'loc=a'}
sys_safe = {
    'loc=a -> next(loc=b || loc=d)',
    'loc=b -> next(loc=a || loc=e || loc=c)',
    'loc=c -> next(loc=b || loc=foo)',
    'loc=d -> next(loc=a || loc=e)',
    'loc=e -> next(loc=d || loc=b || loc=foo)',
    'loc=foo -> next(loc=e || loc=c)',
}
sys_prog = set()

sys_vars['X0reach'] = 'boolean'
sys_init |= {'X0reach'}
sys_safe |= {'next(X0reach) <-> (loc=a) || (X0reach && !park)'}
sys_prog |= {'X0reach'}

specs = spec.GRSpec(env_vars, sys_vars, env_init, sys_init,
                    env_safe, sys_safe, env_prog, sys_prog)

ctrl = synth.synthesize('gr1c', specs)

if not ctrl.save('robot_gr1_arbitrary_set.png', 'png'):
    print(ctrl)

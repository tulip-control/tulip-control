"""
Two examples for arbitrary finite domains with gr1c:

    - one manually coded
    - one with arbitrary (unnumbered) sys TS states
"""
# will probably need also a flag which type to use

from tulip import spec, synth
from tulip import transys as trs

env_vars = {'park'}
env_prog = '!park'

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

sys_vars['X0reach'] = 'boolean'
sys_init |= {'X0reach'}
sys_safe |= {'next(X0reach) <-> (loc=a) || (X0reach && !park)'}
sys_prog = {'X0reach'}

specs = spec.GRSpec(env_vars=env_vars, sys_vars=sys_vars,
                    sys_init=sys_init, sys_safety=sys_safe,
                    env_prog=env_prog, sys_prog=sys_prog)

ctrl = synth.synthesize('gr1c', specs)
ctrl.save('robot_gr1_arbitrary_set.png', 'png')

"""
2nd example
"""
sys = trs.FTS()

states = {'a', 'b', 'c', 'd', 'e', 'foo'}
sys.states.add_from(states)
sys.states.initial.add('a')

sys.transitions.add_from({'a'}, {'b', 'd'})
sys.transitions.add_from({'b'}, {'a', 'e', 'c'})
sys.transitions.add_from({'c'}, {'b', 'foo'})
sys.transitions.add_from({'d'}, {'a', 'e'})
sys.transitions.add_from({'e'}, {'d', 'b', 'foo'})
sys.transitions.add_from({'foo'}, {'e', 'c'})

sys.atomic_propositions |= {'cave'}
sys.states.label('a', 'cave')

sys_vars = {'X0reach':'boolean'}
sys_init = {'X0reach'}

# if we don't want to use an extra AP to label a,
# then we need to either know that tulip internally names
# the state variable as 'loc', or
# provide an option to pass a name for the state var.
# The latter approach can help a user avoid name
# conflicts, in case, e.g., she wants to reserve
# 'loc' for different use
sys_safe = {'next(X0reach) <-> (cave) || (X0reach && !park)'}

specs = spec.GRSpec(env_vars=env_vars, sys_vars=sys_vars,
                    sys_init=sys_init, sys_safety=sys_safe,
                    env_prog=env_prog, sys_prog=sys_prog)

ctrl = synth.synthesize('gr1c', specs, sys=sys)
ctrl.save('robot_gr1_arbitrary_set2.png', 'png')

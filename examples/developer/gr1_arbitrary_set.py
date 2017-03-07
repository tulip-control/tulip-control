"""
Two examples for arbitrary finite domains:

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

sys_init = {'loc= "a"'}
sys_safe = {
    'loc="a" -> next(loc="b" || loc="d")',
    'loc="b" -> next(loc="a" || loc="e" || loc="c")',
    'loc="c" -> next(loc="b" || loc="foo")',
    'loc="d" -> next(loc="a" || loc="e")',
    'loc="e" -> next(loc="d" || loc="b" || loc="foo")',
    'loc="foo" -> next(loc="e" || loc="c")',
}

sys_vars['mem'] = 'boolean'
sys_init |= {'mem'}
sys_safe |= {'next(mem) <-> (loc="a") || (mem && !park)'}
sys_prog = {'mem'}

specs = spec.GRSpec(env_vars=env_vars, sys_vars=sys_vars,
                    sys_init=sys_init, sys_safety=sys_safe,
                    env_prog=env_prog, sys_prog=sys_prog)
specs.moore = False
specs.qinit = '\A \E'
ctrl = synth.synthesize(specs)
ctrl.save('gr1_arbitrary_set0.pdf')

"""
2nd example
"""
sys = trs.FTS()

states = {'a', 'b', 'c', 'd', 'e', 'foo'}
sys.states.add_from(states)
sys.states.initial.add('a')

sys.transitions.add_from(
    [('a', x) for x in {'b', 'd'}] +
    [('b', x) for x in {'a', 'e', 'c'}] +
    [('c', x) for x in {'b', 'foo'}] +
    [('d', x) for x in {'a', 'e'}] +
    [('e', x) for x in {'d', 'b', 'foo'}] +
    [('foo', x) for x in {'e', 'c'}]
)

sys.atomic_propositions |= {'cave'}
sys.states.add('a', ap={'cave'})

sys_vars = {'mem': 'boolean'}
sys_init = {'mem'}

# if we don't want to use an extra AP to label a,
# then we need to either know that tulip internally names
# the state variable as 'loc', or
# provide an option to pass a name for the state var.
# The latter approach can help a user avoid name
# conflicts, in case, e.g., she wants to reserve
# 'loc' for different use
sys_safe = {'next(mem) <-> (cave) || (mem && !park)'}

specs = spec.GRSpec(env_vars=env_vars, sys_vars=sys_vars,
                    sys_init=sys_init, sys_safety=sys_safe,
                    env_prog=env_prog, sys_prog=sys_prog)
specs.moore = False
specs.qinit = '\A \E'
ctrl = synth.synthesize(specs, sys=sys, solver='omega')
ctrl.save('gr1_arbitrary_set1.pdf')

"""
Tests for the tulip.synth module.
"""
import logging
logging.getLogger('tulip').setLevel(logging.ERROR)
logging.getLogger('tulip.interfaces.omega').setLevel(logging.DEBUG)
logging.getLogger('omega').setLevel(logging.WARNING)
from nose.tools import assert_raises
import numpy as np
from scipy import sparse as sp
from tulip import spec, synth, transys


def sys_fts_2_states():
    sys = transys.FTS()
    sys.states.add_from(['X0', 'X1'])
    sys.states.initial.add_from(['X0', 'X1'])

    sys.transitions.add('X0', 'X1')
    sys.transitions.add('X1', 'X0')
    sys.transitions.add('X1', 'X1')

    sys.atomic_propositions.add_from({'home', 'lot'})
    sys.states.add('X0', ap={'home'})
    sys.states.add('X1', ap={'lot'})

    # sys.plot()
    return sys


def env_fts_2_states():
    env = transys.FTS()
    env.owner = 'env'

    env.states.add_from({'e0', 'e1'})
    env.states.initial.add('e0')

    # Park as an env action
    env.env_actions.add_from({'park', 'go'})

    env.transitions.add('e0', 'e0', env_actions='park')
    env.transitions.add('e0', 'e0', env_actions='go')

    # env.plot()
    return env


def env_ofts_bool_actions():
    env = transys.FTS()
    env.owner = 'env'

    env.states.add_from({'e0', 'e1', 'e2'})
    env.states.initial.add('e0')

    env.env_actions.add_from({'park', 'go'})
    env.sys_actions.add_from({'up', 'down'})

    env.transitions.add('e0', 'e1', env_actions='park',
                        sys_actions='up')
    env.transitions.add('e1', 'e2', env_actions='go',
                        sys_actions='down')
    # env.plot()
    return env


def env_ofts_int_actions():
    env = env_ofts_bool_actions()

    env.env_actions.add('stop')
    env.sys_actions.add('hover')

    return env


def parking_spec():
    # barely realizable: assumption necessary
    env_prog = '! (eact = park)'

    sys_vars = {'X0reach'}
    sys_init = {'X0reach'}
    sys_prog = {'home'}

    # one additional requirement: if in lot,
    # then stay there until park signal is turned off
    sys_safe = {'(X (X0reach) <-> lot) || (X0reach && !(eact = "park") )',
                '((lot & (eact = "park") ) -> X(lot))'}
    sys_prog |= {'X0reach'}

    specs = spec.GRSpec(sys_vars=sys_vars, sys_init=sys_init,
                        sys_safety=sys_safe,
                        env_prog=env_prog, sys_prog=sys_prog)
    return specs


def test_sys_fts_int_states():
    """Sys FTS has 3 states, must become 1 int var in GR(1).
    """
    sys = sys_fts_2_states()
    sys.sys_actions_must = 'mutex'
    sys.states.add('X2')

    spec = synth.sys_to_spec(
        sys,
        ignore_initial=False,
        statevar='loc',
        bool_actions=False
    )

    assert 'X0' not in spec.sys_vars
    assert 'X1' not in spec.sys_vars
    assert 'X2' not in spec.sys_vars

    assert 'eloc' not in spec.sys_vars
    assert 'loc' in spec.sys_vars
    assert sorted(spec.sys_vars['loc']) == ['X0', 'X1', 'X2']


def test_env_fts_int_states():
    """Env FTS has 3 states, must become 1 int var in GR(1).
    """
    env = env_fts_2_states()
    env.env_actions_must = 'mutex'
    env.states.add('e2')

    spec = synth.env_to_spec(
        env,
        ignore_initial=False,
        statevar='eloc',
        bool_actions=False
    )

    assert 'e0' not in spec.env_vars
    assert 'e1' not in spec.env_vars
    assert 'e2' not in spec.env_vars

    assert 'loc' not in spec.env_vars
    assert 'eloc' in spec.env_vars
    print(spec.env_vars['eloc'])
    assert sorted(spec.env_vars['eloc']) == ['e0', 'e1', 'e2']


def test_sys_fts_no_actions():
    """Sys FTS has no actions."""
    sys = sys_fts_2_states()
    sys.sys_actions_must = 'mutex'

    spec = synth.sys_to_spec(
        sys,
        ignore_initial=False,
        statevar='loc',
        bool_actions=False)
    assert 'actions' not in spec.sys_vars


def test_env_fts_bool_actions():
    """Env FTS has 2 actions, bools requested."""
    env = env_fts_2_states()
    env.env_actions_must = 'mutex'

    spec = synth.env_to_spec(
        env,
        ignore_initial=False,
        statevar='eloc',
        bool_actions=True,
    )

    assert 'sys_actions' not in spec.env_vars
    assert 'env_actions' not in spec.env_vars

    assert 'park' in spec.env_vars
    assert spec.env_vars['park'] == 'boolean'

    assert 'go' in spec.env_vars
    assert spec.env_vars['go'] == 'boolean'


def test_env_fts_int_actions():
    """Env FTS actions must become 1 int var in GR(1).
    """
    env = env_fts_2_states()
    env.env_actions_must = 'mutex'
    env.env_actions.add('stop')

    spec = synth.env_to_spec(
        env,
        ignore_initial=False,
        statevar='eloc',
        bool_actions=False
    )

    assert 'park' not in spec.env_vars
    assert 'go' not in spec.env_vars
    assert 'stop' not in spec.env_vars

    assert 'sys_actions' not in spec.env_vars
    assert 'env_actions' in spec.env_vars

    print spec.env_vars['env_actions']
    assert (set(spec.env_vars['env_actions']) ==
            {'park', 'go', 'stop', 'env_actionsnone'})


def test_env_ofts_bool_actions():
    """Env OpenFTS has 2 actions, bools requested.
    """
    env = env_ofts_int_actions()
    env.env_actions_must = 'mutex'
    env.env_actions.remove('stop')
    env.sys_actions.remove('hover')

    spec = synth.env_to_spec(
        env,
        ignore_initial=False,
        statevar='eloc',
        bool_actions=True
    )

    _check_ofts_bool_actions(spec)


def test_sys_ofts_bool_actions():
    """Sys OpenFTS has 2 actions, bools requested.
    """
    sys = env_ofts_int_actions()
    sys.owner = 'sys'
    sys.sys_actions_must = 'mutex'
    sys.env_actions.remove('stop')
    sys.sys_actions.remove('hover')

    spec = synth.sys_to_spec(
        sys,
        ignore_initial=False,
        statevar='loc',
        bool_actions=True
    )

    _check_ofts_bool_actions(spec)


def _check_ofts_bool_actions(spec):
    """Common assertion checking for 2 functions above."""
    assert 'park' in spec.env_vars
    assert spec.env_vars['park'] == 'boolean'

    assert 'go' in spec.env_vars
    assert spec.env_vars['go'] == 'boolean'

    assert 'up' in spec.sys_vars
    assert spec.sys_vars['up'] == 'boolean'

    assert 'down' in spec.sys_vars
    assert spec.sys_vars['down'] == 'boolean'

    assert 'env_actions' not in spec.env_vars
    assert 'sys_actions' not in spec.sys_vars


def test_env_ofts_int_actions():
    """Env OpenFTS actions must become 1 int var in GR(1)."""
    env = env_ofts_int_actions()
    env.sys_actions_must = 'mutex'
    spec = synth.env_to_spec(
        env,
        ignore_initial=False,
        statevar='eloc',
        bool_actions=False)
    _check_ofts_int_actions(spec)


def test_sys_ofts_int_actions():
    """Sys OpenFTS actions must become 1 int var in GR(1).
    """
    sys = env_ofts_int_actions()
    sys.owner = 'sys'
    sys.sys_actions_must = 'mutex'
    spec = synth.sys_to_spec(
        sys,
        ignore_initial=False,
        statevar='loc',
        bool_actions=False)
    _check_ofts_int_actions(spec)


def _check_ofts_int_actions(spec):
    """Common assertion checking for 2 function above."""
    print(spec.env_vars)
    print(spec.sys_vars)
    assert 'park' not in spec.env_vars
    assert 'go' not in spec.env_vars
    assert 'stop' not in spec.env_vars

    assert 'up' not in spec.sys_vars
    assert 'down' not in spec.sys_vars
    assert 'hover' not in spec.sys_vars

    assert 'env_actions' in spec.env_vars
    assert set(spec.env_vars['env_actions']) == {'park', 'go', 'stop'}

    assert 'sys_actions' in spec.sys_vars
    assert (set(spec.sys_vars['sys_actions']) == {'up', 'down', 'hover',
            'sys_actionsnone'})


def test_only_mode_control():
    """Unrealizable due to non-determinism.

    Switched system with 2 modes: 'left', 'right'.
    Modes are controlled by the system.
    States are controlled by the environment.

    So only control over dynamics is through mode switching.
    Transitions are thus interpreted as non-deterministic.

    This can model uncertain outcomes in the real world,
    e.g., due to low quality actuators or
    bad low-level feedback controllers.
    """
    # Create a finite transition system
    env_sws = transys.FTS()
    env_sws.owner = 'env'

    env_sws.sys_actions.add_from({'right', 'left'})

    # str states
    n = 4
    states = transys.prepend_with(range(n), 's')

    env_sws.atomic_propositions.add_from(['home', 'lot'])

    # label TS with APs
    ap_labels = [set(), set(), {'home'}, {'lot'}]
    for i, label in enumerate(ap_labels):
        state = 's' + str(i)
        env_sws.states.add(state, ap=label)

    # mode1 transitions
    transmat1 = np.array([[0, 1, 0, 1],
                          [0, 1, 0, 0],
                          [0, 1, 0, 1],
                          [0, 0, 0, 1]])
    env_sws.transitions.add_adj(
        sp.lil_matrix(transmat1), states, {'sys_actions': 'right'})

    # mode2 transitions
    transmat2 = np.array([[1, 0, 0, 0],
                          [1, 0, 1, 0],
                          [0, 0, 1, 0],
                          [1, 0, 1, 0]])
    env_sws.transitions.add_adj(
        sp.lil_matrix(transmat2), states, {'sys_actions': 'left'})

    env_vars = {'park'}
    env_init = {'eloc = "s0"', 'park'}
    env_prog = {'!park'}
    env_safe = set()

    sys_vars = {'X0reach'}
    sys_init = {'X0reach'}
    sys_prog = {'home'}
    sys_safe = {'next(X0reach) <-> lot || (X0reach && !park)'}
    sys_prog |= {'X0reach'}

    specs = spec.GRSpec(env_vars, sys_vars, env_init, sys_init,
                        env_safe, sys_safe, env_prog, sys_prog)

    r = synth.is_realizable('omega', specs, env=env_sws, ignore_env_init=True)
    assert not r


def multiple_env_actions_test():
    multiple_env_actions_check('omega')

def multiple_env_actions_check(solver='omega'):
    """Two env players, 3 states controlled by sys.

    sys wins marginally, due to assumption on
    next combination of actions by env players.
    """
    # 1 <---> 2
    # 1  ---> 3

    env_actions = [
        {
            'name': 'env_alice',
            'values': transys.MathSet({'left', 'right'})
        },
        {
            'name': 'env_bob',
            'values': transys.MathSet({'bleft', 'bright'})
        }
    ]

    sys = transys.FTS(env_actions)
    sys.states.add_from({'s1', 's2', 's3'})
    sys.states.initial.add_from({'s1'})

    sys.add_edge('s1', 's2', env_alice='left', env_bob='bright')
    sys.add_edge('s2', 's1', env_alice='left', env_bob='bright')
    # at state 3 sys loses
    sys.add_edge('s1', 's3', env_alice='right', env_bob='bleft')

    logging.debug(sys)

    env_safe = {('(loc = "s1") -> X( (env_alice = "left") && '
                 '(env_bob = "bright") )')}
    sys_prog = {'loc = "s1"', 'loc = "s2"'}

    specs = spec.GRSpec(
        env_safety=env_safe,
        sys_prog=sys_prog,
        moore=False,
        plus_one=False,
        qinit='\A \E')
    r = synth.is_realizable(solver, specs, sys=sys)
    assert r
    # slightly relax assumption
    specs = spec.GRSpec(
        sys_prog=sys_prog,
        moore=False,
        plus_one=False,
        qinit='\A \E')
    r = synth.is_realizable(solver, specs, sys=sys)
    assert not r


def test_var_name_conflicts():
    """Check redefinitions between states, actions, atomic props."""
    conversion_raises = lambda x, y: assert_raises(
        Exception, spec=x, sys=y,
        ignore_initial=True,
        statevar='loc',
        bool_actions=False)

    # FTS to spec

    # states vs APs
    sys = transys.FTS()
    sys.states.add('out')
    sys.atomic_propositions.add('out')

    conversion_raises(synth.sys_to_spec, sys)

    # states vs sys_actions
    sys = transys.FTS()
    sys.states.add('out')
    sys.sys_actions.add('out')

    conversion_raises(synth.sys_to_spec, sys)

    sys = transys.FTS()
    sys.states.add('sys_actions')
    sys.sys_actions.add('out')

    conversion_raises(synth.sys_to_spec, sys)

    # states vs env_actions
    env = transys.FTS()
    env.states.add('out')
    env.env_actions.add('out')

    conversion_raises(synth.env_to_spec, env)

    env = transys.FTS()
    env.states.add('env_actions')
    env.env_actions.add('out')

    conversion_raises(synth.env_to_spec, env)

    # APs vs sys_actions
    sys = transys.FTS()
    sys.states.add('s0')
    sys.atomic_propositions.add('out')
    sys.env_actions.add('out')

    conversion_raises(synth.sys_to_spec, sys)

    sys = transys.FTS()
    sys.states.add('s0')
    sys.atomic_propositions.add('sys_actions')
    sys.env_actions.add('out')

    conversion_raises(synth.sys_to_spec, sys)

    # APs vs env_actions
    env = transys.FTS()
    env.states.add('s0')
    env.atomic_propositions.add('out')
    env.env_actions.add('out')

    conversion_raises(synth.env_to_spec, env)

    env = transys.FTS()
    env.states.add('s0')
    env.atomic_propositions.add('env_actions')
    env.env_actions.add('out')

    conversion_raises(synth.env_to_spec, env)

    # OpenFTS to spec

    # states vs APs
    sys = transys.FTS()
    sys.states.add('out')
    sys.atomic_propositions.add('out')

    conversion_raises(synth.sys_to_spec, sys)

    conversion_raises(synth.env_to_spec, sys)

    # states vs sys_actions
    sys = transys.FTS()
    sys.states.add('out')
    sys.sys_actions.add('out')

    conversion_raises(synth.sys_to_spec, sys)

    conversion_raises(synth.env_to_spec, sys)

    sys = transys.FTS()
    sys.states.add('sys_actions')
    sys.sys_actions.add('out')

    conversion_raises(synth.sys_to_spec, sys)

    conversion_raises(synth.env_to_spec, sys)

    # states vs env_actions
    sys = transys.FTS()
    sys.states.add('out')
    sys.env_actions.add('out')

    conversion_raises(synth.sys_to_spec, sys)

    conversion_raises(synth.env_to_spec, sys)

    sys = transys.FTS()
    sys.states.add('env_actions')
    sys.env_actions.add('out')

    conversion_raises(synth.sys_to_spec, sys)

    conversion_raises(synth.env_to_spec, sys)

    # sys_actions vs APs
    sys = transys.FTS()
    sys.states.add('s0')
    sys.sys_actions.add('out')
    sys.atomic_propositions.add('out')

    conversion_raises(synth.sys_to_spec, sys)

    conversion_raises(synth.env_to_spec, sys)

    sys = transys.FTS()
    sys.states.add('s0')
    sys.sys_actions.add('out')
    sys.atomic_propositions.add('sys_actions')

    conversion_raises(synth.sys_to_spec, sys)

    conversion_raises(synth.env_to_spec, sys)

    # env_actions vs APs
    sys = transys.FTS()
    sys.states.add('s0')
    sys.env_actions.add('out')
    sys.atomic_propositions.add('out')

    conversion_raises(synth.sys_to_spec, sys)

    conversion_raises(synth.env_to_spec, sys)

    sys = transys.FTS()
    sys.states.add('s0')
    sys.env_actions.add('out')
    sys.atomic_propositions.add('env_actions')

    conversion_raises(synth.sys_to_spec, sys)

    conversion_raises(synth.env_to_spec, sys)


def test_determinize_machine_init():
    mach = transys.MealyMachine()
    mach.add_inputs({'a': {0, 1}})
    mach.add_outputs({'b': {0, 1}, 'c': {0, 1}})
    u = 'Sinit'
    mach.add_nodes_from([u, 1, 2])

    # initial reactions:

    # to input: a=0
    mach.add_edge(u, 1, a=0, b=0, c=1)
    mach.add_edge(u, 2, a=0, b=0, c=1)
    mach.add_edge(u, 1, a=0, b=1, c=0)
    mach.add_edge(u, 2, a=0, b=1, c=1)

    # to input: a=1
    mach.add_edge(u, 1, a=1, b=0, c=1)
    mach.add_edge(u, 2, a=1, b=0, c=1)
    mach.add_edge(u, 1, a=1, b=1, c=0)
    mach.add_edge(u, 2, a=1, b=1, c=1)

    # determinize all outputs arbitrarily
    detmach = synth.determinize_machine_init(mach)
    assert detmach is not mach

    for a in {0, 1}:
        edges = [(i, j) for (i, j, d) in detmach.edges_iter(u, data=True)
                 if d['a'] == a]
        assert len(edges) == 1

    # determinize output b arbitrarily,
    # but output c is constrained to the initial value 0
    detmach = synth.determinize_machine_init(mach, {'c': 0})

    for a in {0, 1}:
        edges = [(i, j, d) for (i, j, d) in detmach.edges_iter(u, data=True)
                 if d['a'] == a]
        assert len(edges) == 1

        ((i, j, d), ) = edges
        assert j == 1
        assert d['b'] == 1


class synthesize_test:
    def setUp(self):
        self.f_triv = spec.GRSpec(
            sys_vars="y",
            moore=False,
            plus_one=False)
        self.trivial_unreachable = spec.GRSpec(
            sys_vars="y",
            sys_prog="False",
            moore=False,
            plus_one=False)

    def tearDown(self):
        self.f_triv = None

    def test_gr1c_basic(self):
        g = synth.synthesize("omega", self.f_triv)
        assert isinstance(g, transys.MealyMachine)

    def test_unrealizable(self):
        assert synth.synthesize("omega", self.trivial_unreachable) is None


if __name__ == '__main__':
    multiple_env_actions_test()

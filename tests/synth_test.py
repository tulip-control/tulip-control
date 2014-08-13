"""
Tests for the tulip.synth module.
"""
import logging
logging.basicConfig(level=logging.WARNING)

from tulip import spec, synth, transys
import numpy as np
from scipy import sparse as sp

def sys_fts_2_states():
    sys = transys.FTS()
    sys.states.add_from(['X0', 'X1'])
    sys.states.initial.add_from(['X0', 'X1'])
    
    sys.transitions.add('X0', 'X1')
    sys.transitions.add('X1', 'X0')
    sys.transitions.add('X1', 'X1')
    
    sys.atomic_propositions.add_from({'home', 'lot'})
    sys.states.add('X0', ap='home')
    sys.states.add('X1', ap='lot')
    
    #sys.plot()
    return sys

def env_fts_2_states():
    env = transys.FTS()
    env.states.add_from({'e0', 'e1'})
    env.states.initial.add('e0')
    
    # Park as an env action
    env.actions.add_from({'park', 'go'})
    
    env.transitions.add('e0', 'e0', actions='park')
    env.transitions.add('e0', 'e0', actions='go')
    
    #env.plot()
    return env

def env_ofts_bool_actions():
    env = transys.OpenFTS()
    env.states.add_from({'e0', 'e1', 'e2'})
    env.states.initial.add('e0')
    
    env.env_actions.add_from({'park', 'go'})
    env.sys_actions.add_from({'up', 'down'})
    
    env.transitions.add('e0', 'e1', env_actions='park',
                                    sys_actions='up')
    env.transitions.add('e1', 'e2', env_actions='go',
                                    sys_actions='down')
    
    #env.plot()
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
    sys_safe = {'(X (X0reach) <-> lot) || (X0reach && !(eact = park) )',
                '((lot & (eact = park) ) -> X(lot))'}
    sys_prog |= {'X0reach'}
    
    specs = spec.GRSpec(sys_vars=sys_vars, sys_init=sys_init,
                        sys_safety=sys_safe,
                        env_prog=env_prog, sys_prog=sys_prog)
    return specs

def test_sys_fts_bool_states():
    """Sys FTS has 2 states, must become 2 bool vars in GR(1)
    """
    sys = sys_fts_2_states()
    sys.actions_must = 'mutex'

    spec = synth.sys_to_spec(
        sys,
        ignore_initial=False,
        bool_states=False,
        action_vars=('eact', 'act'),
        bool_actions=False
    )
    
    assert('loc' not in spec.sys_vars)
    assert('eloc' not in spec.sys_vars)
    
    assert('X0' in spec.sys_vars)
    assert(spec.sys_vars['X0'] == 'boolean')
    
    assert('X1' in spec.sys_vars)
    assert(spec.sys_vars['X1'] == 'boolean')

def test_env_fts_bool_states():
    """Env FTS has 2 states, must become 2 bool vars in GR(1).
    """
    env = env_fts_2_states()
    env.actions_must = 'mutex'
    
    spec = synth.env_to_spec(
        env,
        ignore_initial=False,
        bool_states=False,
        action_vars=('eact', 'act'),
        bool_actions=False
    )
    
    assert('loc' not in spec.env_vars)
    assert('eloc' not in spec.env_vars)
    
    assert('e0' in spec.env_vars)
    assert(spec.env_vars['e0'] == 'boolean')
    
    assert('e1' in spec.env_vars)
    assert(spec.env_vars['e1'] == 'boolean')

def test_sys_fts_int_states():
    """Sys FTS has 3 states, must become 1 int var in GR(1).
    """
    sys = sys_fts_2_states()
    sys.actions_must='mutex'
    sys.states.add('X2')
    
    spec = synth.sys_to_spec(
        sys,
        ignore_initial=False,
        bool_states=False,
        action_vars=('eact', 'act'),
        bool_actions=False
    )
    
    assert('X0' not in spec.sys_vars)
    assert('X1' not in spec.sys_vars)
    assert('X2' not in spec.sys_vars)
    
    assert('eloc' not in spec.sys_vars)
    assert('loc' in spec.sys_vars)
    assert(spec.sys_vars['loc'] == (0, 2))

def test_env_fts_int_states():
    """Env FTS has 3 states, must become 1 int var in GR(1).
    """
    env = env_fts_2_states()
    env.actions_must='mutex'
    env.states.add('e2')
    
    spec = synth.env_to_spec(
        env,
        ignore_initial=False,
        bool_states=False,
        action_vars=('eact', 'act'),
        bool_actions=False
    )
    
    assert('e0' not in spec.env_vars)
    assert('e1' not in spec.env_vars)
    assert('e2' not in spec.env_vars)
    
    assert('loc' not in spec.env_vars)
    assert('eloc' in spec.env_vars)
    assert(spec.env_vars['eloc'] == (0, 2))

def test_sys_fts_no_actions():
    """Sys FTS has no actions.
    """
    sys = sys_fts_2_states()
    sys.actions_must='mutex'
    
    spec = synth.sys_to_spec(
        sys,
        ignore_initial=False,
        bool_states=False,
        action_vars=('eact', 'act'),
        bool_actions=False
    )
    
    assert('actions' not in spec.sys_vars)

def test_env_fts_bool_actions():
    """Env FTS has 2 actions, bools requested.
    """
    env = env_fts_2_states()
    env.actions_must='mutex'
    
    spec = synth.env_to_spec(
        env,
        ignore_initial=False,
        bool_states=False,
        action_vars=('eact', 'act'),
        bool_actions=True,
    )
    
    assert('sys_actions' not in spec.env_vars)
    assert('env_actions' not in spec.env_vars)
    
    assert('park' in spec.env_vars)
    assert(spec.env_vars['park'] == 'boolean')
    
    assert('go' in spec.env_vars)
    assert(spec.env_vars['go'] == 'boolean')

def test_env_fts_int_actions():
    """Env FTS actions must become 1 int var in GR(1).
    """
    env = env_fts_2_states()
    env.actions_must='mutex'
    env.actions.add('stop')
    
    spec = synth.env_to_spec(
        env,
        ignore_initial=False,
        bool_states=False,
        action_vars=('eact', 'act'),
        bool_actions=False
    )
    
    assert('park' not in spec.env_vars)
    assert('go' not in spec.env_vars)
    assert('stop' not in spec.env_vars)
    
    assert('sys_actions' not in spec.env_vars)
    assert('env_actions' in spec.env_vars)    
    
    print spec.env_vars['env_actions']
    assert(set(spec.env_vars['env_actions']) ==
           {'park', 'go', 'stop', 'env_actionsnone'})

def test_env_ofts_bool_actions():
    """Env OpenFTS has 2 actions, bools requested.
    """
    env = env_ofts_int_actions()
    env.actions_must='mutex'
    env.env_actions.remove('stop')
    env.sys_actions.remove('hover')
    
    spec = synth.env_to_spec(
        env,
        ignore_initial=False,
        bool_states=False,
        action_vars=('eact', 'act'),
        bool_actions=True
    )
    
    _check_ofts_bool_actions(spec)

def test_sys_ofts_bool_actions():
    """Sys OpenFTS has 2 actions, bools requested.
    """
    sys = env_ofts_int_actions()
    sys.actions_must='mutex'
    sys.env_actions.remove('stop')
    sys.sys_actions.remove('hover')
    
    spec = synth.sys_to_spec(
        sys,
        ignore_initial=False,
        bool_states=False,
        action_vars=('eact', 'act'),
        bool_actions=True
    )
    
    _check_ofts_bool_actions(spec)

def _check_ofts_bool_actions(spec):
    """Common assertion checking for 2 functions above.
    """
    assert('park' in spec.env_vars)
    assert(spec.env_vars['park'] == 'boolean')
    
    assert('go' in spec.env_vars)
    assert(spec.env_vars['go'] == 'boolean')
    
    assert('up' in spec.sys_vars)
    assert(spec.sys_vars['up'] == 'boolean')
    
    assert('down' in spec.sys_vars)
    assert(spec.sys_vars['down'] == 'boolean')
    
    assert('env_actions' not in spec.env_vars)
    assert('sys_actions' not in spec.sys_vars)

def test_env_ofts_int_actions():
    """Env OpenFTS actions must become 1 int var in GR(1).
    """
    env = env_ofts_int_actions()
    env.actions_must='mutex'
    
    spec = synth.env_to_spec(
        env,
        ignore_initial=False,
        bool_states=False,
        action_vars=('eact', 'act'),
        bool_actions=False
    )
    
    _check_ofts_int_actions(spec)

def test_sys_ofts_int_actions():
    """Sys OpenFTS actions must become 1 int var in GR(1).
    """
    sys = env_ofts_int_actions()
    sys.actions_must='mutex'
    
    spec = synth.sys_to_spec(
        sys,
        ignore_initial=False,
        bool_states=False,
        action_vars=('eact', 'act'),
        bool_actions=False
    )
    
    _check_ofts_int_actions(spec)

def _check_ofts_int_actions(spec):
    """Common assertion checking for 2 function above.
    """
    print(spec.env_vars)
    assert('park' not in spec.env_vars)
    assert('go' not in spec.env_vars)
    assert('stop' not in spec.env_vars)
    
    assert('up' not in spec.sys_vars)
    assert('down' not in spec.sys_vars)
    assert('hover' not in spec.sys_vars)
    
    assert('env_actions' in spec.env_vars)
    assert(set(spec.env_vars['env_actions']) == {'park', 'go', 'stop'})
    
    assert('sys_actions' in spec.sys_vars)
    assert(set(spec.sys_vars['sys_actions']) == {'up', 'down', 'hover'})

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
    env_sws = transys.OpenFTS()
    
    env_sws.sys_actions.add_from({'right','left'})
    
    # str states
    n = 4
    states = transys.prepend_with(range(n), 's')
    
    env_sws.atomic_propositions.add_from(['home','lot'])
    
    # label TS with APs
    ap_labels = [set(),set(),{'home'},{'lot'}]
    for i, label in enumerate(ap_labels):
        state = 's' + str(i)
        env_sws.states.add(state, ap=label)
    
    
    # mode1 transitions
    transmat1 = np.array([[0,1,0,1],
                          [0,1,0,0],
                          [0,1,0,1],
                          [0,0,0,1]])
    env_sws.transitions.add_adj(
        sp.lil_matrix(transmat1), states, {'sys_actions':'right'}
    )
                          
    # mode2 transitions
    transmat2 = np.array([[1,0,0,0],
                          [1,0,1,0],
                          [0,0,1,0],
                          [1,0,1,0]])
    env_sws.transitions.add_adj(
        sp.lil_matrix(transmat2), states, {'sys_actions':'left'}
    )
    
    env_vars = {'park'}
    env_init = {'eloc = 0', 'park'}
    env_prog = {'!park'}
    env_safe = set()
    
    sys_vars = {'X0reach'}
    sys_init = {'X0reach'}          
    sys_prog = {'home'}
    sys_safe = {'next(X0reach) <-> lot || (X0reach && !park)'}
    sys_prog |= {'X0reach'}
    
    specs = spec.GRSpec(env_vars, sys_vars, env_init, sys_init,
                        env_safe, sys_safe, env_prog, sys_prog)
    
    r = synth.is_realizable('gr1c', specs, env=env_sws, ignore_env_init=True)
    assert(not r)

def multiple_env_actions_test():
    """Two env players, 3 states controlled by sys.
    
    sys wins marginally, due to assumption on
    next combination of actions by env players.
    """
    # 1 <---> 2
    #    ---> 3
    
    env_actions = [('env_alice', transys.MathSet({'left', 'right'}) ),
                   ('env_bob', transys.MathSet({'left', 'right'}) )]
    
    sys = transys.OpenFTS(env_actions)
    sys.states.add_from({'s1', 's2', 's3'})
    sys.states.initial.add_from({'s1'})
    
    sys.add_edge('s1', 's2', env_alice='left', env_bob='right')
    sys.add_edge('s1', 's3', env_alice='right', env_bob='left') # at state 3 sys loses
    sys.add_edge('s2', 's1', env_alice='left', env_bob='right')
    
    logging.debug(sys)
    
    env_safe = {'(loc = s1) -> X( (env_alice = left) && (env_bob = right) )'}
    sys_prog = {'loc = s1', 'loc = s2'}
    
    specs = spec.GRSpec(env_safety=env_safe, sys_prog=sys_prog)
    
    r = synth.is_realizable('gr1c', specs, sys=sys)
    assert(r)
    
    # slightly relax assumption
    specs = spec.GRSpec(sys_prog=sys_prog)
    
    r = synth.is_realizable('gr1c', specs, sys=sys)
    assert(not r)


class synthesize_test:
    def setUp(self):
        self.f_triv = spec.GRSpec(sys_vars="y")

    def tearDown(self):
        self.f_triv = None

    def test_gr1c_basic(self):
        assert isinstance(synth.synthesize("gr1c", self.f_triv),
                          transys.MealyMachine)

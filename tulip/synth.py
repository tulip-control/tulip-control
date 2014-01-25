# Copyright (c) 2012, 2013 by California Institute of Technology
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 
# 3. Neither the name of the California Institute of Technology nor
#    the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CALTECH
# OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
"""
Interface to library of synthesis tools, e.g., JTLV, gr1c
"""
import logging
logger = logging.getLogger(__name__)

from warnings import warn

from tulip import transys
from tulip.spec import GRSpec
from tulip import jtlvint
from tulip import gr1cint

hl = '\n' +60*'-'

def pstr(s):
    return '(' +str(s) +')'

def _disj(set0):
    return " || ".join([
        "(" +str(x) +")"
        for x in set0
    ])

def _conj(set0):
    return " && ".join([
        "(" +str(x) +")"
        for x in set0
    ])

def _conj_intersection(set0, set1, parenth=True):
    if parenth:
        return " && ".join([
            "("+str(x)+")"
            for x in set0
            if x in set1
        ])
    else:
        return " && ".join([
            str(x)
            for x in set0
            if x in set1
        ])

def _conj_neg(set0, parenth=True):
    if parenth:
        return " && ".join([
            "!("+str(x)+")"
            for x in set0
        ])
    else:
        return " && ".join([
            "!"+str(x)
            for x in set0
        ])

def _conj_neg_diff(set0, set1, parenth=True):
    if parenth:
        return " && ".join([
            "!("+str(x)+")"
            for x in set0
            if x not in set1
        ])
    else:
        return " && ".join([
            "!"+str(x)
            for x in set0
            if x not in set1
        ])

def mutex(iterable):
    """Mutual exclusion for all time.
    """
    iterable = filter(lambda x: x != '', iterable)
    if not iterable:
        return []
    if len(iterable) <= 1:
        return []
    
    return [_conj([
        '!(' + str(x) + ') || (' + _conj_neg_diff(iterable, [x]) +')'
        for x in iterable
    ]) ]

def exactly_one(iterable):
    """N-ary xor.
    
    Contrast with pure mutual exclusion.
    """
    if len(iterable) <= 1:
        return [pstr(x) for x in iterable]
    
    return ['(' + _disj([
        '(' +str(x) + ') && ' + _conj_neg_diff(iterable, [x])
        for x in iterable
    ]) + ')']

def _conj_action(label, action_type, nxt=False):
    if action_type not in label:
        return ''
    action = label[action_type]
    if action is '':
        return ''
    if nxt:
        return ' && X' + pstr(action)
    else:
        return ' && ' + pstr(action)

def create_states(states, variables, trans, statevar, bool_states):
    """Create bool or int state variables in GR(1).
    
    Return map of TS states to spec variable valuations.
    
    @param states: TS states
    
    @param variables: to be augmented with state variables
    
    @param trans: to be augmented with state variable constraints.
        Such a constraint is necessary only in case of bool states.
        It requires that exactly one bool variable be True at a time.
    
    @param statevar: name to use for int-valued state variabe.
    
    @param bool_states: if True, then use bool variables.
        Otherwise use int-valued variable.
        The latter is overridden in case < 3 states exist,
        to avoid issues with gr1c.
    """
    # too few states for a gr1c int variable ?
    if len(states) < 3:
        bool_states = True
    
    if bool_states:
        state_ids = {x:x for x in states}
        variables.update({s:'boolean' for s in states})
        trans += exactly_one(states)
    else:
        state_ids, domain = states2ints(states, statevar)
        variables[statevar] = domain
    return state_ids

def states2ints(states, statevar):
    """Return states of form 'statevar = #'.
    
    where # is obtained by dropping the 1st char
    of each given state.
    
    @type states: iterable of str,
        each str of the form: letter + number
    
    @param statevar: name of int variable representing
        the current state
    @type statevar: str
    
    @rtype: {state : state_id}
    """
    letter_int = True
    for state in states:
        try:
            int(state[1:])
        except:
            letter_int = False
            break
    
    if letter_int:
        # this allows the user to control numbering
        strip_letter = lambda x: statevar + ' = ' + x[1:]
        state_ids = {x:strip_letter(x) for x in states}
        n_states = len(states)
        domain = (0, n_states-1)
    else:
        setloc = lambda s: statevar + ' = ' + s
        state_ids = {s:setloc(s) for s in states}
        domain = list(states)
    
    return (state_ids, domain)

def sys_to_spec(sys, ignore_initial=False, bool_states=False):
    """Convert system's transition system to GR(1) representation.
    
    The term GR(1) representation is preferred to GR(1) spec,
    because an FTS can represent sys_init, sys_safety, but
    not other spec forms.
    
    @type sys: transys.FTS | transys.OpenFTS
    
    @param ignore_initial: Do not include initial state info from TS.
        Enable this to mask absence of OpenFTS initial states.
        Useful when initial states are specified in another way,
        e.g., directly augmenting the spec part.
    @type check_initial_exist: bool
    
    @param bool_states: if True,
        then use one bool variable for each state,
        otherwise use an int variable called loc.
    @type bool_states: bool
    
    @rtype: GRSpec
    """
    if isinstance(sys, transys.FiniteTransitionSystem):
        (sys_vars, sys_init, sys_trans) = fts2spec(
            sys, ignore_initial, bool_states, 'loc'
        )
        return GRSpec(sys_vars=sys_vars, sys_init=sys_init,
                      sys_safety=sys_trans)
    elif isinstance(sys, transys.OpenFiniteTransitionSystem):
        return sys_open_fts2spec(sys, ignore_initial, bool_states)
    else:
        raise TypeError('synth.sys_to_spec does not support ' +
            str(type(sys)) +'. Use FTS or OpenFTS.')

def env_to_spec(env, ignore_initial=False, bool_states=False):
    """Convert environment transition system to GR(1) representation.
    
    For details see also sys_to_spec.
    
    @type env: transys.FTS | transys.OpenFTS
    
    @type bool_states: bool
    """
    if isinstance(env, transys.FiniteTransitionSystem):
        (env_vars, env_init, env_trans) = fts2spec(
            env, ignore_initial, bool_states, 'eloc'
        )
        return GRSpec(env_vars=env_vars, env_init=env_init,
                      env_safety=env_trans)
    elif isinstance(env, transys.OpenFiniteTransitionSystem):
        return env_open_fts2spec(env, ignore_initial, bool_states)
    else:
        raise TypeError('synth.env_to_spec does not support ' +
            str(type(env)) +'. Use FTS or OpenFTS.')

def fts2spec(fts, ignore_initial=False, bool_states=False,
             statevar='loc'):
    """Convert closed FTS to GR(1) representation.
    
    So fts on its own is not the complete problem spec.
    
    @param fts: transys.FiniteTransitionSystem
    
    @rtype: GRSpec
    """
    assert(isinstance(fts, transys.FiniteTransitionSystem))
    
    aps = fts.aps
    states = fts.states
    
    sys_vars = {ap:'boolean' for ap in aps}
    sys_vars.update({act:'boolean' for act in fts.actions})
    
    sys_trans = []
    
    state_ids = create_states(states, sys_vars, sys_trans,
                              statevar, bool_states)
    
    init = sys_init_from_ts(states, state_ids, aps, ignore_initial)
    
    sys_trans += sys_trans_from_ts(states, state_ids, fts.transitions)
    sys_trans += ap_trans_from_ts(states, state_ids, aps)
    
    sys_trans += mutex(fts.actions)
    
    return (sys_vars, init, sys_trans)

def sys_open_fts2spec(ofts, ignore_initial=False, bool_states=False):
    """Convert OpenFTS to GR(1) representation.
    
    Note that not any GR(1) can be represented by an OpenFTS,
    as the OpenFTS is currently defined.
    A GameStructure would be needed instead.
    
    Use the spec to add more information,
    for example to specify env_init, sys_init that
    involve both sys_vars and env_vars.
    
    For example, an OpenFTS cannot represent how the
    initial valuation of env_vars affects the allowable
    initial valuation of sys_vars, which is represented
    by the state of OpenFTS.
    
    Either OpenFTS can be extended in the future,
    or a game structure added.
    
    notes
    -----
    
    1. Currently each env_action becomes a bool env_var.
        In the future a candidate option is to represent the
        env_actions as an enumeration.
        This would avoid the exponential cost incurred by bools.
        A map between the enum and named env_actions
        will probably also be needed.
    
    @param ofts: transys.OpenFiniteTransitionSystem
    
    @rtype: GRSpec
    """
    assert(isinstance(ofts, transys.OpenFiniteTransitionSystem))
    
    aps = ofts.aps
    states = ofts.states
    trans = ofts.transitions
    env_actions = ofts.env_actions
    sys_actions = ofts.sys_actions
    
    sys_vars = {ap:'boolean' for ap in aps}
    sys_vars.update({act:'boolean' for act in sys_actions})
    sys_trans = mutex(sys_actions)
    
    env_vars = list(env_actions)
    env_trans = mutex(env_actions)
    
    statevar = 'loc'
    state_ids = create_states(states, sys_vars, sys_trans,
                              statevar, bool_states)
    
    sys_init = sys_init_from_ts(states, state_ids, aps, ignore_initial)
    
    sys_trans += sys_trans_from_ts(states, state_ids, trans)
    sys_trans += ap_trans_from_ts(states, state_ids, aps)
    
    env_trans += env_trans_from_sys_ts(states, state_ids, trans, env_vars)
    
    return GRSpec(
        sys_vars=sys_vars, env_vars=env_vars,
        sys_init=sys_init,
        env_safety=env_trans, sys_safety=sys_trans
    )

def env_open_fts2spec(ofts, ignore_initial, bool_states=False):
    assert(isinstance(ofts, transys.OpenFiniteTransitionSystem))
    
    aps = ofts.aps
    states = ofts.states
    trans = ofts.transitions
    env_actions = ofts.env_actions
    sys_actions = ofts.sys_actions
    
    # since APs are tied to env states, let them be env variables
    env_vars = {ap:'boolean' for ap in aps}
    env_vars.update({act:'boolean' for act in env_actions})
    env_trans = mutex(env_actions)
    
    sys_vars = list(sys_actions)
    # some duplication here, because we don't know
    # whether the user will provide a system TS as well
    # and whether that TS will contain all the system actions
    # defined in the environment TS
    sys_trans = exactly_one(ofts.sys_actions)
    
    statevar = 'eloc'
    state_ids = create_states(states, env_vars, env_trans,
                              statevar, bool_states)
    
    env_init = sys_init_from_ts(states, state_ids, aps, ignore_initial)
    
    env_trans += env_trans_from_env_ts(states, state_ids, trans, sys_actions)
    env_trans += ap_trans_from_ts(states, state_ids, aps)
    
    return GRSpec(
        sys_vars=sys_vars, env_vars=env_vars,
        env_init=env_init,
        env_safety=env_trans, sys_safety=sys_trans
    )

def sys_init_from_ts(states, state_ids, aps, ignore_initial=False):
    """Initial state, including enforcement of exactly one.
    
    APs also considered for the initial state.
    """
    init = []
    
    # don't ignore labeling info
    for state in states.initial:
        state_id = state_ids[state]
        label = states.label_of(state)
        ap_str = sprint_aps(label, aps)
        if not ap_str:
            continue
        init += ['!(' + pstr(state_id) + ') || (' + ap_str +')']
    
    # skip ?
    if ignore_initial:
        return init
    
    if not states.initial:
        msg = 'FTS has no initial states.\n'
        msg += 'Enforcing this renders False the GR(1):\n'
        msg += ' - guarantee if this is a system TS,\n'
        msg += '   so the spec becomes trivially False.\n'
        msg += ' - assumption if this is an environment TS,\n'
        msg += '   so the spec becomes trivially True.'
        warn(msg)
        
        init += [_conj_neg(state_ids.itervalues() ) ]
        return init
        
    init += [_disj([state_ids[s] for s in states.initial])]
    return init

def sys_trans_from_ts(states, state_ids, trans):
    """Convert transition relation to GR(1) sys_safety.
    
    The transition relation may be closed or open,
    i.e., depend only on system, or also on environment actions.
    
    @type trans: FiniteTransitionSystem.transitions |
        OpenFiniteTransitionSystem.transitions
    
    No mutexes enforced by this function among:
        
        - sys states
        - env actions
    """
    sys_trans = []
    
    # Transitions
    for from_state in states:
        from_state_id = state_ids[from_state]
        cur_trans = trans.find([from_state])
        
        # no successor states ?
        if not cur_trans:
            sys_trans += ['('+str(from_state_id) +') -> X(' +
                _conj_neg(state_ids.itervalues() ) +')']
            continue
        
        cur_str = []
        for (from_state, to_state, label) in cur_trans:
            to_state_id = state_ids[to_state]
            
            precond = '(' + str(from_state_id) + ')'
            postcond = '(' + str(to_state_id) +')'
            
            postcond += _conj_action(label, 'env_actions')
            postcond += _conj_action(label, 'sys_actions')
            # system FTS given
            postcond += _conj_action(label, 'actions')
            
            cur_str += ['(' + precond + ') -> X(' + postcond + ')']
            
        sys_trans += [_disj(cur_str) ]
    return sys_trans

def env_trans_from_sys_ts(states, state_ids, trans, env_vars):
    """Convert environment actions to GR(1) env_safety.
    
    This constrains the actions available next to the environment
    based on the system OpenFTS.
    
    Might become optional in the future,
    depending on the desired way of defining env behavior.
    """
    env_trans = []
    if not env_vars:
        return env_trans
    
    for from_state in states:
        from_state_id = state_ids[from_state]
        cur_trans = trans.find([from_state])
        
        # no successor states ?
        if not cur_trans:
            env_trans += ['(' +str(from_state_id) +') -> X(' +
                _conj_neg(env_vars) + ')']
            continue
        
        # collect possible next env actions
        next_env_actions = set()
        for (from_state, to_state, label) in cur_trans:
            if 'env_actions' not in label:
                continue
            
            env_action = label['env_actions']
            next_env_actions.add(env_action)
        next_env_actions = _disj(next_env_actions)
        
        env_trans += ["(" +str(from_state_id) +") -> X(" +
                      next_env_actions +")"]
    return env_trans

def env_trans_from_env_ts(states, state_ids, trans, sys_actions):
    """Convert environment TS transitions to GR(1) representation.
    
    This contributes to the \rho_e(X, Y, X') part of the spec,
    i.e., constrains the next environment state variables' valuation
    depending on the previous environment state variables valuation
    and the previous system action (system output).
    """
    env_trans = []
    
    for from_state in states:
        from_state_id = state_ids[from_state]
        cur_trans = trans.find([from_state])
        
        # no successor states ?
        if not cur_trans:
            env_trans += [pstr(from_state_id) +' -> X(' +
                _conj_neg(state_ids.itervalues() ) + ')']
                
            msg = 'Environment dead-end found.\n'
            msg += 'If sys can force env to dead-end,\n'
            msg += 'then GR(1) assumption becomes False,\n'
            msg += 'and spec trivially True.'
            warn(msg)
            
            continue
        
        cur_list = []
        found_free = False # any environment transition
        # not conditioned on the previous system output ?
        for (from_state, to_state, label) in cur_trans:
            to_state_id = state_ids[to_state]
            
            precond = pstr(from_state_id)
            postcond = 'X' + pstr(to_state_id)
            
            postcond += _conj_action(label, 'env_actions', nxt=True)
            
            # environment FTS given
            postcond += _conj_action(label, 'actions', nxt=True)
            postcond += _conj_action(label, 'sys_actions')
            
            if not _conj_action(label, 'sys_actions'):
                found_free = True
            
            cur_list += [pstr(postcond) ]
        
        # can sys kill env by setting all previous sys outputs to False ?
        # then env assumption becomes False,
        # so the spec trivially True: avoid this
        if not found_free:
            cur_list += [_conj_neg(sys_actions)]
        
        env_trans += [pstr(precond) + ' -> (' + _disj(cur_list) +')']
    return env_trans

def ap_trans_from_ts(states, state_ids, aps):
    """Require atomic propositions to follow states according to label.
    """
    trans = []
    
    # no AP labels ?
    if not aps:
        return trans
    
    for state in states:
        label = states.label_of(state)
        state_id = state_ids[state]
        
        tmp = sprint_aps(label, aps)
        if not tmp:
            continue
        
        trans += ["X(("+ str(state_id) +") -> ("+ tmp +"))"]
    return trans

def sprint_aps(label, aps):
    if label.has_key("ap"):
        tmp0 = _conj_intersection(aps, label['ap'], parenth=False)
    else:
        tmp0 = ''
    
    if label.has_key("ap"):
        tmp1 = _conj_neg_diff(aps, label['ap'], parenth=False)
    else:
        tmp1 = _conj_neg(aps, parenth=False)
    
    if len(tmp0) > 0 and len(tmp1) > 0:
        tmp = tmp0 +' && '+ tmp1
    else:
        tmp = tmp0 + tmp1
    return tmp

def synthesize(option, specs, env=None, sys=None,
               ignore_env_init=False, ignore_sys_init=False,
               bool_states=False, verbose=0):
    """Function to call the appropriate synthesis tool on the spec.

    Beware!  This function provides a generic interface to a variety
    of routines.  Being under active development, the types of
    arguments supported and types of objects returned may change
    without notice.

    @param option: Magic string that declares what tool to invoke,
        what method to use, etc.  Currently recognized forms:

          - C{"gr1c"}: use gr1c for GR(1) synthesis via L{gr1cint}.
          - C{"jtlv"}: use JTLV for GR(1) synthesis via L{jtlvint}.
    @type specs: L{spec.GRSpec}
    
    @param env: A transition system describing the environment:
        
            - states controlled by environment
            - input: sys_actions
            - output: env_actions
            - initial states constrain the environment
        
        This constrains the transitions available to
        the environment, given the outputs from the system.
        
        Note that an OpenFTS with only sys_actions is
        equivalent to an FTS for the environment.
    @type env: transys.FTS | transys.OpenFTS
    
    @param sys: A transition system describing the system:
        
            - states controlled by the system
            - input: env_actions
            - output: sys_actions
            - initial states constrain the system
        
        Note that an OpenFTS with only sys_actions is
        equivalent to an FTS for the system.
    @type sys: transys.FTS | transys.OpenFTS
    
    @param ignore_sys_init: Ignore any initial state information
        contained in env.
    @type ignore_sys_init: bool
    
    @param ignore_env_init: Ignore any initial state information
        contained in sys.
    @type ignore_env_init: bool
    
    @param bool_states: if True,
        then use one bool variable for each state.
        Otherwise use a single int variable for all states.
        
        Currently int state implemented only for gr1c.
    @type bool_states: bool
    
    @type verbose: bool
    
    @return: If spec is realizable,
        then return a Mealy machine implementing the strategy.
        Otherwise return None.
    @rtype: transys.MealyMachine | None
    """
    # not yet implemented for jtlv
    if bool_states is False and option is 'jtlv':
        warn('Int state not yet available for jtlv solver.\n' +
             'Using bool states.')
        bool_states = True
    
    specs = spec_plus_sys(specs, env, sys,
                          ignore_env_init, ignore_sys_init,
                          bool_states)
    
    if option == 'gr1c':
        ctrl = gr1cint.synthesize(specs, verbose=verbose)
    elif option == 'jtlv':
        ctrl = jtlvint.synthesize(specs, verbose=verbose)
    else:
        raise Exception('Undefined synthesis option. '+\
                        'Current options are "jtlv" and "gr1c"')
    
    try:
        logger.debug('Mealy machine has: n = ' +
            str(len(ctrl.states) ) +' states.')
    except:
        logger.debug('No Mealy machine returned.')
    
    # no controller found ?
    # exploring unrealizability with counterexamples or other means
    # can be done by calling a dedicated other function, not this
    if not isinstance(ctrl, transys.MealyMachine):
        return None
    
    return ctrl

def is_realizable(option, specs, env=None, sys=None,
                  ignore_env_init=False, ignore_sys_init=False,
                  bool_states=False, verbose=0):
    """Check realizability.
    
    For details see synthesize.
    """
    if bool_states is False and option is 'jtlv':
        warn('Int state not yet available for jtlv solver.\n' +
             'Using bool states.')
        bool_states = True
    
    specs = spec_plus_sys(specs, env, sys,
                          ignore_env_init, ignore_sys_init,
                          bool_states)
    
    if option == 'gr1c':
        r = gr1cint.check_realizable(specs, verbose=verbose)
    elif option == 'jtlv':
        r = jtlvint.check_realizable(specs, verbose=verbose)
    else:
        raise Exception('Undefined synthesis option. '+\
                        'Current options are "jtlv" and "gr1c"')
    return r

def spec_plus_sys(specs, env=None, sys=None,
                  ignore_env_init=False, ignore_sys_init=False,
                  bool_states=True):
    if sys is not None:
        sys_formula = sys_to_spec(sys, ignore_sys_init, bool_states)
        specs = specs | sys_formula
        logger.debug('sys TS:\n' + str(sys_formula.pretty() ) + hl)
    if env is not None:
        env_formula = env_to_spec(env, ignore_env_init, bool_states)
        specs = specs | env_formula
        logger.debug('env TS:\n' + str(env_formula.pretty() ) + hl)
        
    logger.debug('Overall Spec:\n' + str(specs.pretty() ) +hl)
    return specs

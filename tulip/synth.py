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

import warnings

from tulip import transys
from tulip.spec import GRSpec
from tulip.interfaces import jtlv
from tulip.interfaces import gr1c

_hl = '\n' +60*'-'

def _pstr(s):
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
        if x != ''
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
        return [_pstr(x) for x in iterable]
    
    return ['(' + _disj([
        '(' +str(x) + ') && ' + _conj_neg_diff(iterable, [x])
        for x in iterable
    ]) + ')']

def _conj_action(actions_dict, action_type, nxt=False, ids=None):
    """Return conjunct if C{action_type} in C{actions_dict}.
    
    @param actions_dict: C{dict} with pairs C{action_type_name : action_value}
    @type actions_dict: dict
    
    @param action_type: key to look for in C{actions_dict}
    @type action_type: hashable (here typically a str)
    
    @param nxt: prepend or not with the next operator
    @type nxt: bool
    
    @param ids: map C{action_value} -> value used in solver input, e.g., for gr1c
    @type ids: dict
    
    @return:
        - conjunct (includes C{&&} operator) if:
          
            - C{action_type} in C{actions_dict}, and
            - C{action_value} is not the empty string (modeling "no constrain")
          
          includes next operator (C{X}) if C{nxt = True}.
        - empty string otherwise
    @rtype: str
    """
    if action_type not in actions_dict:
        return ''
    action = actions_dict[action_type]
    if ids is not None:
        action = ids[action]
    if action is '':
        return ''
    if nxt:
        return ' X' + _pstr(action)
    else:
        return _pstr(action)

def _conj_actions(actions_dict, solver_expr=None, nxt=False):
    """Conjunction of multiple action types.
    
    Includes solver expression substitution.
    See also L{_conj_action}.
    """
    logger.debug('conjunction of actions: ' + str(actions_dict))
    logger.debug('mapping to solver equivalents: ' + str(solver_expr))
    
    if not actions_dict:
        logger.debug('actions_dict empty, returning empty string\n')
        return ''
    
    if solver_expr is not None:
        actions = [solver_expr[type_name][action_value]
                   for type_name, action_value in actions_dict.iteritems()]
    else:
        actions = actions_dict
    
    logger.debug('after substitution: ' + str(actions))
    
    conjuncted_actions = _conj(actions)
    logger.debug('conjuncted actions: ' + str(conjuncted_actions) +'\n')
    
    if nxt:
        return ' X' + _pstr(conjuncted_actions)
    else:
        return _pstr(conjuncted_actions)

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
        logger.debug('states modeled as Boolean variables')
        
        state_ids = {x:x for x in states}
        variables.update({s:'boolean' for s in states})
        trans += exactly_one(states)
    else:
        logger.debug('states not modeled as Booleans')
        
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
    # TODO: merge with actions2ints, don't strip letter
    msg = 'mapping states: ' + str(states) +\
          '\n\t to expressions understood by solver.'
    logger.debug(msg)
    
    if not states:
        raise Exception('No states given, got: ' + str(states))
    
    letter_int = True
    for state in states:
        if not isinstance(state, str):
            msg = 'States must be strings, not anything else.\n' +\
                  'Got instead: ' + str(state) +\
                  ', of type: ' + str(type(state) )
            raise TypeError(msg)
        
        try:
            int(state[1:])
        except:
            letter_int = False
    
    logger.debug('all states are strings')
    if letter_int:
        logger.debug('all states are like "x1" where x some character')
        
        # this allows the user to control numbering
        strip_letter = lambda x: statevar + ' = ' + x[1:]
        state_ids = {x:strip_letter(x) for x in states}
        state_ints = {int(x[1:]) for x in states}
        n_states = len(states)
        domain = (0, n_states-1)
        solver_range = set(range(0, n_states))
        
        logger.debug('after stripping the character: ' +\
                     'state_ids = ' + str(state_ids))
        
        # any missing integers ?
        if state_ints != solver_range:
            msg = 'some integers within string states missing:' +\
                  'compare given:\n\t' + str(state_ints) +\
                  '\n to same length range:\n\t' + str(solver_range) +\
                  '\n Will try to model them as arbitrary finite domain...'
            logger.error(msg)
            letter_int = False
    
    # try arbitrary finite domain
    if not letter_int:
        logger.debug('string states modeled as an arbitrary finite domain')
        setloc = lambda s: statevar + ' = ' + str(s)
        state_ids = {s:setloc(s) for s in states}
        domain = list(states)
    
    return (state_ids, domain)

def create_actions(
    actions, variables, trans, init,
    actionvar, bool_actions, actions_must
):
    """Represent actions by bool or int GR(1) variables.
    
    Similar to the int representation of states.
    
    If the actions are:
    
       - mutually exclusive (use_mutex == True)
       - bool actions have not been requested (bool_actions == False)
      
    then an int variable represents actions in GR(1).
    
    If actions are not mutually exclusive,
    then only bool variables can represent them.
    
    Suppose N actions are defined.
    The int variable is allowed to take N+1 values.
    The additional value corresponds to all actions being False.
    
    If FTS actions are integers,
    then the additional action is an int value.
    
    If FTS actions are strings (e.g., 'park', 'wait'),
    then the additional action is 'none'.
    They are treated by gr1cint as an arbitrary finite domain.
    
    An option 'min_one' is internally available,
    in order to allow only N values of the action variable.
    This requires that at least one action be True each time.
    Combined with a mutex constraint, it yields an n-ary xor constraint.
    
    @return: mapping from FTS actions, to GR(1) actions.
        If bools are used, then GR(1) are the same.
        Otherwise, they map to e.g. 'act = wait'
    @rtype: dict
    """
    if not actions:
        logger.debug('actions empty, empty dict for solver expr')
        return dict()
    
    logger.debug('creating actions from: ' + str(actions) )
    
    # options for modeling actions
    if actions_must is None:
        use_mutex = False
        min_one = False
    elif actions_must == 'mutex':
        use_mutex = True
        min_one = False
    elif actions_must == 'xor':
        use_mutex = True
        min_one = True
    else:
        raise Exception('Unknown value: actions_must = ' +
                        str(actions_must) )
    
    yesno = lambda x: 'Yes' if x else 'No'
    msg = 'options for modeling actions:\n\t' +\
          'mutex: ' + yesno(use_mutex) +'\n\t' +\
          'min_one: ' + yesno(min_one)
    logger.debug(msg)
    
    # too few values for gr1c ?
    #if len(actions) < 3:
    #    bool_actions = True
    
    # no mutex -> cannot use int variable
    if not use_mutex:
        logger.debug('not using mutex: Booleans must model actions')
        bool_actions = True
    
    if bool_actions:
        logger.debug('actions modeled as Boolean variables')
        
        action_ids = {x:x for x in actions}
        variables.update({a:'boolean' for a in actions})
        
        # single action ?
        if not mutex(action_ids.values()):
            return action_ids
        
        if use_mutex and not min_one:
            trans += ['X (' + mutex(action_ids.values())[0] + ')']
            init += mutex(action_ids.values())
        elif use_mutex and min_one:
            trans += ['X (' + exactly_one(action_ids.values())[0] + ')']
            init += exactly_one(action_ids.values())
        elif min_one:
            raise Exception('min_one requires mutex')
    else:
        logger.debug('actions not modeled as Booleans')
        assert(use_mutex)
        action_ids, domain = actions2ints(actions, actionvar, min_one)
        variables[actionvar] = domain
        
        msg = 'created solver variable: ' + str(actionvar) + '\n\t' +\
              'with domain: ' + str(domain)
        logger.debug(msg)
    
    msg = 'for tulip variable: ' + str(actionvar) +\
          ' (an action type)\n\t' +\
          'the map from [tulip action values] ---> ' +\
          '[solver expressions] is:\n' + 2*'\t' + str(action_ids)
    logger.debug(msg)
    return action_ids

def actions2ints(actions, actionvar, min_one=False):
    msg = 'mapping domain of action_type: ' + str(actionvar) +\
          '\n\t to expressions understood by solver.'
    logger.debug(msg)
    
    int_actions = True
    for action in actions:
        if not isinstance(action, int):
            logger.debug('not all actions are integers')
            int_actions = False
            break
    if int_actions:
        logger.debug('actions modeled as an integer variable')
        
        action_ids = {x:x for x in actions}
        n_actions = len(actions)
        
        # extra value modeling all False ?
        if min_one:
            n = n_actions -1
        else:
            n = n_actions
        domain = (0, n)
    else:
        logger.debug('modeling actions as arbitrary finite domain')
        
        setact = lambda s: actionvar + ' = ' + s
        action_ids = {s:setact(s) for s in actions}
        domain = list(actions)
        if not min_one:
            domain += [actionvar + 'none']
            
            msg = 'domain has been extended, because all actions\n\t' +\
                  'could be False (constraint: min_one = False).'
            logger.debug(msg)
        
    return (action_ids, domain)

def sys_to_spec(
    sys, ignore_initial, bool_states,
    action_vars, bool_actions
):
    """Convert system's transition system to GR(1) representation.
    
    The term GR(1) representation is preferred to GR(1) spec,
    because an FTS can represent sys_init, sys_safety, but
    not other spec forms.
    
    @type sys: L{transys.FTS} or L{transys.OpenFTS}
    
    @param ignore_initial: Do not include initial state info from TS.
        Enable this to mask absence of OpenFTS initial states.
        Useful when initial states are specified in another way,
        e.g., directly augmenting the spec part.
    
    @param bool_states: if True,
        then use one bool variable for each state,
        otherwise use an int variable called loc.
    @type bool_states: bool
    
    @rtype: L{GRSpec}
    """
    if isinstance(sys, transys.FiniteTransitionSystem):
        (sys_vars, sys_init, sys_trans) = fts2spec(
            sys, ignore_initial, bool_states, 'loc',
            'sys_actions', bool_actions
        )
        return GRSpec(sys_vars=sys_vars, sys_init=sys_init,
                      sys_safety=sys_trans)
    elif isinstance(sys, transys.OpenFiniteTransitionSystem):
        return sys_open_fts2spec(
            sys, ignore_initial, bool_states,
            action_vars, bool_actions
        )
    else:
        raise TypeError('synth.sys_to_spec does not support ' +
            str(type(sys)) +'. Use FTS or OpenFTS.')

def env_to_spec(
    env, ignore_initial, bool_states,
    action_vars, bool_actions
):
    """Convert environment transition system to GR(1) representation.
    
    For details see also L{sys_to_spec}.
    
    @type env: L{transys.FTS} or L{transys.OpenFTS}
    
    @type bool_states: bool
    """
    if isinstance(env, transys.FiniteTransitionSystem):
        (env_vars, env_init, env_trans) = fts2spec(
            env, ignore_initial, bool_states, 'eloc',
            'env_actions', bool_actions
        )
        return GRSpec(env_vars=env_vars, env_init=env_init,
                      env_safety=env_trans)
    elif isinstance(env, transys.OpenFiniteTransitionSystem):
        return env_open_fts2spec(
            env, ignore_initial, bool_states,
            action_vars, bool_actions
        )
    else:
        raise TypeError('synth.env_to_spec does not support ' +
            str(type(env)) +'. Use FTS or OpenFTS.')

def fts2spec(
    fts, ignore_initial=False, bool_states=False,
    statevar='loc', actionvar=None,
    bool_actions=False
):
    """Convert closed FTS to GR(1) representation.
    
    Single player + Multiple action types
    =====================================
    So fts on its own is not the complete problem spec.
    Currently L{FTS} supports only a single set of actions.
    
    If you have only one player (either env or sys),
    with multiple action types, then use an L{OpenFTS},
    without any action types for its opponent.
    
    Make sure that, depending on the player,
    C{'env'} or C{'sys'} are part of the action type names,
    so that L{synth.synthesize} can recognize them.
    
    @param fts: L{transys.FiniteTransitionSystem}
    
    @rtype: (dict, list, list)
    @return: (sys_vars, sys_init, sys_trans), where each element
        corresponds to the similarly-named attribute of L{GRSpec}.
    """
    assert(isinstance(fts, transys.FiniteTransitionSystem))
    
    aps = fts.aps
    states = fts.states
    actions = fts.actions
    
    sys_init = []
    sys_trans = []
    
    sys_vars = {ap:'boolean' for ap in aps}
    
    action_ids = create_actions(
        actions, sys_vars, sys_trans, sys_init,
        actionvar, bool_actions, fts.actions_must
    )
    
    state_ids = create_states(states, sys_vars, sys_trans,
                              statevar, bool_states)
    
    sys_init += sys_init_from_ts(states, state_ids, aps, ignore_initial)
    
    sys_trans += sys_trans_from_ts(
        states, state_ids, fts.transitions,
        action_ids=action_ids
    )
    tmp_init, tmp_trans = ap_trans_from_ts(states, state_ids, aps)
    sys_init += tmp_init
    sys_trans += tmp_trans
    
    return (sys_vars, sys_init, sys_trans)

def sys_open_fts2spec(
    ofts, ignore_initial=False, bool_states=False,
    action_vars=None, bool_actions=False
):
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
    
    @param ofts: L{transys.OpenFiniteTransitionSystem}
    
    @rtype: L{GRSpec}
    """
    assert(isinstance(ofts, transys.OpenFiniteTransitionSystem))
    
    aps = ofts.aps
    states = ofts.states
    trans = ofts.transitions
    
    sys_init = []
    sys_trans = []
    env_init = []
    env_trans = []
    
    sys_vars = {ap:'boolean' for ap in aps}
    env_vars = dict()
    
    actions = ofts.actions
    
    sys_action_ids = dict()
    env_action_ids = dict()
    
    for action_type, codomain in actions.iteritems():
        msg = 'action_type:\n\t' + str(action_type) +'\n'
        msg += 'with codomain:\n\t' + str(codomain)
        logger.debug(msg)
        
        if 'sys' in action_type:
            logger.debug('Found sys action')
            
            action_ids = create_actions(
                codomain, sys_vars, sys_trans, sys_init,
                action_type, bool_actions, ofts.sys_actions_must
            )
            
            logger.debug('Updating sys_action_ids with:\n\t' + str(action_ids))
            sys_action_ids[action_type] = action_ids
        elif 'env' in action_type:
            logger.debug('Found env action')
            
            action_ids = create_actions(
                codomain, env_vars, env_trans, env_init,
                action_type, bool_actions, ofts.env_actions_must
            )
            
            logger.debug('Updating env_action_ids with:\n\t' + str(action_ids))
            env_action_ids[action_type] = action_ids
    
    statevar = 'loc'
    state_ids = create_states(states, sys_vars, sys_trans,
                              statevar, bool_states)
    
    sys_init += sys_init_from_ts(states, state_ids, aps, ignore_initial)
    
    sys_trans += sys_trans_from_ts(
        states, state_ids, trans,
        sys_action_ids=sys_action_ids, env_action_ids=env_action_ids
    )
    tmp_init, tmp_trans = ap_trans_from_ts(states, state_ids, aps)
    sys_init += tmp_init
    sys_trans += tmp_trans
    
    env_trans += env_trans_from_sys_ts(
        states, state_ids, trans, env_action_ids
    )
    
    return GRSpec(
        sys_vars=sys_vars, env_vars=env_vars,
        env_init=env_init, sys_init=sys_init,
        env_safety=env_trans, sys_safety=sys_trans
    )

def env_open_fts2spec(
    ofts, ignore_initial=False, bool_states=False,
    action_vars=None, bool_actions=False
):
    assert(isinstance(ofts, transys.OpenFiniteTransitionSystem))
    
    aps = ofts.aps
    states = ofts.states
    trans = ofts.transitions
    
    sys_init = []
    sys_trans = []
    env_init = []
    env_trans = []
    
    # since APs are tied to env states, let them be env variables
    env_vars = {ap:'boolean' for ap in aps}
    sys_vars = dict()
    
    actions = ofts.actions
    
    sys_action_ids = dict()
    env_action_ids = dict()
    
    for action_type, codomain in actions.iteritems():
        if 'sys' in action_type:
            action_ids = create_actions(
                codomain, sys_vars, sys_trans, sys_init,
                action_type, bool_actions, ofts.sys_actions_must
            )
            sys_action_ids[action_type] = action_ids
        elif 'env' in action_type:
            action_ids = create_actions(
                codomain, env_vars, env_trans, env_init,
                action_type, bool_actions, ofts.env_actions_must
            )
            env_action_ids[action_type] = action_ids
    
    # some duplication here, because we don't know
    # whether the user will provide a system TS as well
    # and whether that TS will contain all the system actions
    # defined in the environment TS
    
    statevar = 'eloc'
    state_ids = create_states(states, env_vars, env_trans,
                              statevar, bool_states)
    
    env_init += sys_init_from_ts(states, state_ids, aps, ignore_initial)
    
    env_trans += env_trans_from_env_ts(
        states, state_ids, trans,
        env_action_ids=env_action_ids, sys_action_ids=sys_action_ids
    )
    tmp_init, tmp_trans = ap_trans_from_ts(states, state_ids, aps)
    env_init += tmp_init
    env_trans += tmp_trans
    
    return GRSpec(
        sys_vars=sys_vars, env_vars=env_vars,
        env_init=env_init, sys_init=sys_init,
        env_safety=env_trans, sys_safety=sys_trans
    )

def sys_init_from_ts(states, state_ids, aps, ignore_initial=False):
    """Initial state, including enforcement of exactly one.
    """
    init = []
    
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
        raise Exception(msg)
        
        init += ['False']
        return init
        
    init += [_disj([state_ids[s] for s in states.initial])]
    return init

def sys_trans_from_ts(
    states, state_ids, trans,
    action_ids=None, sys_action_ids=None, env_action_ids=None):
    """Convert transition relation to GR(1) sys_safety.
    
    The transition relation may be closed or open,
    i.e., depend only on system, or also on environment actions.
    
    No mutexes enforced by this function among:
        
        - sys states
        - env actions
    
    An edge attribute 'previous' can be optionally set to
    an iterable of edge attribute keys.
    The actions with those action_types those keys
    will not be prepended by the next operator.
    
    This enables defining both current and next actions, e.g.,
        
    some_action && X(some_other_action)
    
    About label type checking: in principle everything should work the
    same if the base class LabeledDiGraph was replaced by MultiDiGraph,
    so that users can play around with their own bare graphs,
    when they don't need the label typing overhead.

    @param trans: L{Transitions} as from the transitions
        attribute of L{FiniteTransitionSystem} or
        L{OpenFiniteTransitionSystem}.
    
    @param action_ids: same as C{sys-action_ids}
        Caution: to be removed in a future release
    
    @param sys_action_ids: dict of dicts
        outer dict keyed by action_type
        each inner dict keyed by action_value
        each inner dict value is the solver expression for that action value
        
        for example an action type with an
        arbitrary finite discrete codomain can be modeled either:
        
          - as Boolean variables, so each possible action value
            becomes a different Boolean variable with the same
            name, thus C{sys_action_ids[action_type]} will be
            the identity map on C{action_values} for that C{action_type}.
          
          - as integer variables, so each possible action value
            becomes a different expression in the solver (e.g. gr1c)
            input format. Then C{sys_action_ids[action_type]} maps
            C{action_value} -> solver expression of the form:
                
            C{action_type = i}
            
            where C{i} corresponds to that particular  C{action_type}.
    
    @param env_action_ids: same as C{sys-action_ids}
    """
    logger.debug('modeling sys transitions in logic')
    sys_trans = []
    
    # Transitions
    for from_state in states:
        from_state_id = state_ids[from_state]
        precond = _pstr(from_state_id)
        
        cur_trans = trans.find([from_state])
        
        msg = 'from state: ' + str(from_state) +\
              ', the available transitions are:\n\t' + str(cur_trans)
        logger.debug(msg)
        
        # no successor states ?
        if not cur_trans:
            logger.debug('state: ' + str(from_state) + ' is deadend !')
            sys_trans += [precond + ' -> X(False)']
            continue
        
        cur_str = []
        for (from_state, to_state, label) in cur_trans:
            to_state_id = state_ids[to_state]
            
            postcond = ['X' + _pstr(to_state_id)]
            
            logger.debug('label = ' + str(label))
            if 'previous' in label:
                previous = label['previous']
            else:
                previous = set()
            logger.debug('previous = ' + str(previous))
            
            env_actions = {k:v for k,v in label.iteritems() if 'env' in k}
            prev_env_act = {k:v for k,v in env_actions.iteritems()
                            if k in previous}
            next_env_act = {k:v for k,v in env_actions.iteritems()
                            if k not in previous}
            
            postcond += [_conj_actions(prev_env_act, env_action_ids, nxt=False)]
            postcond += [_conj_actions(next_env_act, env_action_ids, nxt=True)]
            
            sys_actions = {k:v for k,v in label.iteritems() if 'sys' in k}
            prev_sys_act = {k:v for k,v in sys_actions.iteritems()
                            if k in previous}
            next_sys_act = {k:v for k,v in sys_actions.iteritems()
                            if k not in previous}
            
            postcond += [_conj_actions(prev_sys_act, sys_action_ids, nxt=False)]
            postcond += [_conj_actions(next_sys_act, sys_action_ids, nxt=True)]
            
            # if system FTS given
            # in case 'actions in label, then action_ids is a dict,
            # not a dict of dicts, because certainly this came
            # from an FTS, not an OpenFTS
            if 'actions' in previous:
                postcond += [_conj_action(label, 'actions',
                                          ids=action_ids, nxt=False)]
            else:
                postcond += [_conj_action(label, 'actions',
                                          ids=action_ids, nxt=True)]
            
            cur_str += [_conj(postcond)]
            
            msg = 'guard to state: ' + str(to_state) +\
                  ', with state_id: ' + str(to_state_id) +\
                  ', has post-conditions: ' + str(postcond)
            logger.debug(msg)
            
        sys_trans += [precond + ' -> (' + _disj(cur_str) + ')']
    return sys_trans

def env_trans_from_sys_ts(states, state_ids, trans, env_action_ids):
    """Convert environment actions to GR(1) env_safety.
    
    This constrains the actions available next to the environment
    based on the system OpenFTS.
    
    Purpose is to prevent env from blocking sys by purely
    picking a combination of actions for which sys has no outgoing
    transition from that state.
    
    Might become optional in the future,
    depending on the desired way of defining env behavior.
    
    @param env_action_ids: dict of dicts, see L{sys_trans_from_ts}.
    """
    env_trans = []
    
    # this probably useless for multiple action types
    if not env_action_ids:
        return env_trans
    
    for from_state in states:
        from_state_id = state_ids[from_state]
        precond = _pstr(from_state_id)
        
        cur_trans = trans.find([from_state])
        
        # no successor states ?
        if not cur_trans:
            # nothing modeled for env, since sys has X(False) anyway
            #for action_type, codomain_map in env_action_ids.iteritems():
            #env_trans += [precond + ' -> X(' + s + ')']
            continue
        
        # collect possible next env actions
        next_env_action_combs = set()
        for (from_state, to_state, label) in cur_trans:
            env_actions = {k:v for k,v in label.iteritems() if 'env' in k}
            
            if not env_actions:
                continue
            
            logger.debug('env_actions: ' + str(env_actions))
            logger.debug('env_action_ids: ' + str(env_action_ids))
            
            env_action_comb = _conj_actions(env_actions, env_action_ids)
            
            logger.debug('env_action_comb: ' + str(env_action_comb))
            
            next_env_action_combs.add(env_action_comb)
        next_env_actions = _disj(next_env_action_combs)
        
        logger.debug('next_env_actions: ' + str(next_env_actions))
        
        # no next env actions ?
        if not next_env_actions:
            continue
        
        env_trans += [precond + ' -> X(' +
                      next_env_actions + ')']
    return env_trans

def env_trans_from_env_ts(
    states, state_ids, trans,
    action_ids=None, env_action_ids=None, sys_action_ids=None
):
    """Convert environment TS transitions to GR(1) representation.
    
    This contributes to the \rho_e(X, Y, X') part of the spec,
    i.e., constrains the next environment state variables' valuation
    depending on the previous environment state variables valuation
    and the previous system action (system output).
    """
    env_trans = []
    
    for from_state in states:
        from_state_id = state_ids[from_state]
        precond = _pstr(from_state_id)
        
        cur_trans = trans.find([from_state])
        
        # no successor states ?
        if not cur_trans:
            env_trans += [precond + ' -> X(False)']
                
            msg = 'Environment dead-end found.\n'
            msg += 'If sys can force env to dead-end,\n'
            msg += 'then GR(1) assumption becomes False,\n'
            msg += 'and spec trivially True.'
            warnings.warn(msg)
            
            continue
        
        cur_list = []
        found_free = False # any environment transition
        # not conditioned on the previous system output ?
        for (from_state, to_state, label) in cur_trans:
            to_state_id = state_ids[to_state]
            
            postcond = ['X' + _pstr(to_state_id)]
            
            env_actions = {k:v for k,v in label.iteritems() if 'env' in k}
            postcond += [_conj_actions(env_actions, env_action_ids, nxt=True)]
            
            # remember: this is an environment FTS, so no next for sys
            sys_actions = {k:v for k,v in label.iteritems() if 'sys' in k}
            postcond += [_conj_actions(sys_actions, sys_action_ids)]
            
            postcond += [_conj_action(label, 'actions', nxt=True, ids=action_ids)]
            
            # todo: test this claus
            if not sys_actions:
                found_free = True
            
            cur_list += [_conj(postcond) ]
        
        # can sys kill env by setting all previous sys outputs to False ?
        # then env assumption becomes False,
        # so the spec trivially True: avoid this
        if not found_free and sys_action_ids:
            msg = 'no free env outgoing transition found\n' +\
                  'instead will take disjunction with negated sys actions'
            logger.debug(msg)
            
            for action_type, codomain in sys_action_ids.iteritems():
                conj = _conj_neg(codomain.itervalues() )
                cur_list += [conj]
                
                msg = 'for action_type: ' + str(action_type) +'\n' +\
                      'with codomain: ' + str(codomain) +'\n' +\
                      'the negated conjunction is: ' + str(conj)
                logger.debug(msg)
        
        env_trans += [_pstr(precond) + ' -> (' + _disj(cur_list) +')']
    return env_trans

def ap_trans_from_ts(states, state_ids, aps):
    """Require atomic propositions to follow states according to label.
    """
    init = []
    trans = []
    
    # no AP labels ?
    if not aps:
        return (init, trans)
    
    # initial labeling
    for state in states:
        state_id = state_ids[state]
        label = states[state]
        ap_str = sprint_aps(label, aps)
        if not ap_str:
            continue
        init += ['!(' + _pstr(state_id) + ') || (' + ap_str +')']
    
    # transitions of labels
    for state in states:
        label = states[state]
        state_id = state_ids[state]
        
        tmp = sprint_aps(label, aps)
        if not tmp:
            continue
        
        trans += ["X(("+ str(state_id) +") -> ("+ tmp +"))"]
    
    return (init, trans)

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

def synthesize(
    option, specs, env=None, sys=None,
    ignore_env_init=False, ignore_sys_init=False,
    bool_states=False, action_vars=None,
    bool_actions=False, rm_deadends=True
):
    """Function to call the appropriate synthesis tool on the specification.

    Beware!  This function provides a generic interface to a variety
    of routines.  Being under active development, the types of
    arguments supported and types of objects returned may change
    without notice.

    @param option: Magic string that declares what tool to invoke,
        what method to use, etc.  Currently recognized forms:

          - C{"gr1c"}: use gr1c for GR(1) synthesis via L{interfaces.gr1c}.
          - C{"jtlv"}: use JTLV for GR(1) synthesis via L{interfaces.jtlv}.
    @type specs: L{spec.GRSpec}
    
    @param env: A transition system describing the environment:
        
            - states controlled by environment
            - input: sys_actions
            - output: env_actions
            - initial states constrain the environment
        
        This constrains the transitions available to
        the environment, given the outputs from the system.
        
        Note that an L{OpenFTS} with only sys_actions is
        equivalent to an L{FTS} for the environment.
    @type env: L{transys.FTS} or L{transys.OpenFTS}
    
    @param sys: A transition system describing the system:
        
            - states controlled by the system
            - input: env_actions
            - output: sys_actions
            - initial states constrain the system
        
        Note that an OpenFTS with only sys_actions is
        equivalent to an FTS for the system.
    @type sys: L{transys.FTS} L{transys.OpenFTS}
    
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
    
    @param action_vars: for the integer variables modeling
        environment and system actions in GR(1).
        Effective only when >2 actions for each player.
    @type action_vars: 2-tuple of str:
        
        (env_action_var_name, sys_action_var_name)
        
        Default: ('eact', 'act')
        
        (must be valid variable name)
    
    @param bool_actions: model actions using bool variables
    @type bool_actions: bool

    @param rm_deadends: if True,
        then the returned strategy contains no terminal states.
    @type rm_deadends: bool
    
    @return: If spec is realizable,
        then return a Mealy machine implementing the strategy.
        Otherwise return None.
    @rtype: L{transys.MealyMachine} or None
    """
    specs = spec_plus_sys(specs, env, sys,
                          ignore_env_init, ignore_sys_init,
                          bool_states, action_vars,
                          bool_actions)
    
    if option == 'gr1c':
        ctrl = gr1c.synthesize(specs)
    elif option == 'jtlv':
        ctrl = jtlv.synthesize(specs)
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

    if rm_deadends:
        ctrl.remove_deadends()

    return ctrl

def is_realizable(
    option, specs, env=None, sys=None,
    ignore_env_init=False, ignore_sys_init=False,
    bool_states=False, action_vars=None,
    bool_actions=False
):
    """Check realizability.
    
    For details see L{synthesize}.
    """
    specs = spec_plus_sys(
        specs, env, sys,
        ignore_env_init, ignore_sys_init,
        bool_states, action_vars, bool_actions
    )
    
    if option == 'gr1c':
        r = gr1c.check_realizable(specs)
    elif option == 'jtlv':
        r = jtlv.check_realizable(specs)
    else:
        raise Exception('Undefined synthesis option. '+\
                        'Current options are "jtlv" and "gr1c"')
    
    if r:
        logger.debug('is realizable')
    else:
        logger.debug('is not realizable')
    
    return r

def _default_action_vars():
    return ('eact', 'act')

def spec_plus_sys(
    specs, env, sys,
    ignore_env_init, ignore_sys_init,
    bool_states, action_vars, bool_actions
):
    if sys is not None:
        sys_formula = sys_to_spec(sys, ignore_sys_init, bool_states,
                                  action_vars, bool_actions)
        specs = specs | sys_formula
        logger.debug('sys TS:\n' + str(sys_formula.pretty() ) + _hl)
    if env is not None:
        env_formula = env_to_spec(env, ignore_env_init, bool_states,
                                  action_vars, bool_actions)
        specs = specs | env_formula
        logger.debug('env TS:\n' + str(env_formula.pretty() ) + _hl)
        
    logger.info('Overall Spec:\n' + str(specs.pretty() ) +_hl)
    return specs

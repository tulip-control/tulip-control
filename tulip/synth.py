# Copyright (c) 2012-2014 by California Institute of Technology
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
from __future__ import absolute_import

import logging
logger = logging.getLogger(__name__)

import warnings

from . import transys
from .spec import GRSpec
from .interfaces import jtlv
from .interfaces import gr1c

_hl = '\n' +60*'-'

def _pstr(s):
    return '(' + str(s) + ')'

def _disj(set0):
    return ' || '.join([
        '(' + str(x) + ')'
        for x in set0
    ])

def _conj(set0):
    return ' && '.join([
        '(' + str(x) + ')'
        for x in set0
        if x != ''
    ])

def _conj_intersection(set0, set1, parenth=True):
    if parenth:
        return ' && '.join([
            '(' + str(x) + ')'
            for x in set0
            if x in set1
        ])
    else:
        return ' && '.join([
            str(x)
            for x in set0
            if x in set1
        ])

def _conj_neg(set0, parenth=True):
    if parenth:
        return ' && '.join([
            '!(' + str(x) + ')'
            for x in set0
        ])
    else:
        return ' && '.join([
            '!' + str(x)
            for x in set0
        ])

def _conj_neg_diff(set0, set1, parenth=True):
    if parenth:
        return ' && '.join([
            '!(' + str(x) + ')'
            for x in set0
            if x not in set1
        ])
    else:
        return ' && '.join([
            '!' + str(x)
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

# duplicate states are impossible, because each networkx vertex is unique
# non-contiguous integerss for states fine: you are lossing efficiency
# - synth doesn't care about that

def iter2var(states, variables, statevar, bool_states, must):
    """Represent finite domain in GR(1).
    
    An integer or string variable can be used,
    or multiple Boolean variables.
    
    If the possible values are:
    
      - mutually exclusive (use_mutex == True)
      - bool actions have not been requested (bool_actions == False)
    
    then an integer or string variable represents the variable in GR(1).
    
    If all values are integers, then an integer is used.
    If all values are strings, then a string variable is used.
    Otherwise an exception is raised, unless Booleans have been requested.
    
    If the values are not mutually exclusive,
    then only Boolean variables can represent them.
    
    Suppose N ossible values are defined.
    The int variable is allowed to take N+1 values.
    The additional value corresponds to all, e.g., actions, being False.
    
    If FTS values are integers,
    then the additional action is an int value.
    
    If FTS values are strings (e.g., 'park', 'wait'),
    then the additional action is 'none'.
    They are treated by C{spec} as an arbitrary finite domain.
    
    An option C{min_one} is internally available,
    in order to allow only N values of the variable.
    This requires that the variable takes at least one value each time.
    
    Combined with a mutex constraint, it yields an n-ary xor constraint.
    
    @param states: values of domain.
    @type states: iterable container of C{int}
        or iterable container of C{str}
    
    @param variables: to be augmented with integer or string variable
        or Boolean variables.
    
    @param statevar: name to use for integer or string valued variabe.
    @type statevar: C{str}
    
    @param bool_states: if True, then use bool variables.
        Otherwise use integer or string valued variable.
    
    @return: C{tuple} of:
      - mapping from values to GR(1) actions.
        If Booleans are used, then GR(1) are the same.
        Otherwise, they map to e.g. 'act = "wait"' or 'act = 3'
        
      - constraints to be added to C{trans} and/or C{init} in GR(1)
    @rtype: C{dict}, C{list}
    """
    if not states:
        logger.debug('empty container, so empty dict for solver expr')
        return dict(), None
    
    logger.debug('mapping domain: ' + str(states) + '\n\t'
                 'to expression understood by a GR(1) solver.')
    
    assert(must in {'mutex', 'xor', None})
    
    # options for modeling actions
    if must in {'mutex', 'xor'}:
        use_mutex = True
    else:
        use_mutex = False
    
    if must == 'xor':
        min_one = True
    else:
        min_one = False
    
    # no mutex -> cannot use int variable
    if not use_mutex:
        logger.debug('not using mutex: Booleans must model actions')
        bool_states = True
    
    logger.debug(
        'options for modeling actions:\n\t'
        'mutex: ' + str(use_mutex) +'\n\t'
        'min_one: ' + str(min_one)
    )
    
    all_str = all(isinstance(x, str) for x in states)
    
    if bool_states:
        logger.debug('states modeled as Boolean variables')
        if not all_str:
            raise TypeError('If Boolean, all states must be strings.')
        
        state_ids = {x:x for x in states}
        variables.update({s:'boolean' for s in states})
        
        # single action ?
        if len(mutex(state_ids.values()) ) == 0:
            return state_ids
        
        # handle multiple actions
        if use_mutex and not min_one:
            constraint = mutex(state_ids.values())[0]
        elif use_mutex and min_one:
            constraint = exactly_one(state_ids.values())[0]
        elif min_one:
            raise Exception('min_one requires mutex')
    else:
        logger.debug('states not modeled as Booleans')
        if statevar in variables:
            raise ValueError('state variable: ' + str(statevar) +
                             ' already exists in: ' + str(variables))
        
        all_int = all(isinstance(x, int) for x in states)
        
        if all_int:
            logger.debug('all states are integers')
            
            # extra value modeling all False ?
            if min_one:
                n = max(states)
            else:
                n = max(states) + 1
            
            f = lambda x: statevar + ' = ' + str(x)
            domain = (min(states), n)
            
            logger.debug('created solver variable: ' + str(statevar) +
                         '\n\t with domain: ' + str(domain))
        elif all_str:
            logger.debug('all states are strings')
            assert(use_mutex)
            
            f = lambda x: statevar + ' = "' + str(x) + '"'
            domain = list(states)
            
            if not min_one:
                domain += [statevar + 'none']
                
                logger.debug('domain has been extended, because all actions\n\t'
                             'could be False (constraint: min_one = False).')
        else:
            raise TypeError('Integer and string states must not be mixed.')
        
        state_ids = {x:f(x) for x in states}
        variables[statevar] = domain
        constraint = None
    
    logger.debug(
        'for tulip variable: ' + str(statevar) + '\n'
        'the map from [tulip action values] ---> '
        '[solver expressions] is:\n' + 2*'\t' + str(state_ids)
    )
    return state_ids, constraint

def _add_actions(constraint, init, trans):
    if constraint is None:
        return
    trans += ['X (' + constraint[0] + ')']
    init += constraint

def _fts2spec(
    fts, ignore_initial=False,
    statevar='loc', actionvar=None,
    bool_states=False, bool_actions=False
):
    """Convert closed FTS to GR(1) representation.
    """
    raise Exception('deprecated')
    
    assert(isinstance(fts, transys.FiniteTransitionSystem))
    
    aps = fts.aps
    states = fts.states
    actions = fts.actions
    
    sys_init = []
    sys_trans = []
    
    sys_vars = {ap:'boolean' for ap in aps}
    
    action_ids, constraint = iter2var(
        actions, sys_vars, actionvar, bool_actions, fts.actions_must
    )
    _add_actions(constraint, sys_init, sys_trans)
    
    state_ids, constraint = iter2var(states, sys_vars, statevar,
                                     bool_states, must='xor')
    if constraint is not None:
        sys_trans += constraint
    
    sys_init += _sys_init_from_ts(states, state_ids, aps, ignore_initial)
    
    sys_trans += _sys_trans_from_ts(
        states, state_ids, fts.transitions,
        action_ids=action_ids
    )
    tmp_init, tmp_trans = _ap_trans_from_ts(states, state_ids, aps)
    sys_init += tmp_init
    sys_trans += tmp_trans
    
    return (sys_vars, sys_init, sys_trans)

def sys_to_spec(
    ofts, ignore_initial, statevar,
    bool_states=False, bool_actions=False
):
    """Convert transition system to GR(1) fragment of LTL.
    
    The attribute C{FTS.owner} defines who controls the system,
    as described next. It can take values C{'env'} or C{'sys'}.
    
    The following are represented by variables controlled by C{ofts.owner}:
    
      - the current state
      - the atomic propositions annotating states
      - the system actions annotating edges
    
    The following are represented by variables controlled by the other player:
    
      - the environment actions annotating edges
    
    Multiple types of environment and system actions can be defined.
    Make sure that, depending on the player,
    C{'env'} or C{'sys'} are part of the action type names,
    so that L{synth.synthesize} can recognize them.
    
    Caution
    =======
    There are aspects of L{FTS} that
    need to be separately specified in a logic formula.
    
    An example are the initial conditions constraining the values
    of environment and system actions.
    
    See also
    ========
    L{sys_trans_from_ts}, L{env_open_fts2spec},
    L{create_actions}, L{create_states}
    
    @param ofts: L{FTS}
    
    @param ignore_initial: Do not include initial state info from TS.
        Enable this to mask absence of FTS initial states.
        Useful when initial states are specified in another way,
        e.g., directly augmenting the spec part.
    @type ignore_initial: C{bool}
    
    @param state_var: name to be used for the integer or string
        variable that equals the current transition system state.
    @type state_var: C{str}
    
    @param bool_states: deprecated as inefficient
        
        if C{True}, then use one Boolean variable
        to represent each state in GR(1).
        Otherwise use a single integer variable,
        different values of which correspond to states of C{ofts}.
    @type bool_states: bool
    
    @param bool_actions: Similar to C{bool_states}.
        For each type of system actions,
        and each type of environment actions:
        
          - if C{True}, then for each possible value of that action type,
            use a different Boolean variable to represent it.
            
          - Otherwise use a single integer variable,
            that ranges over the possible action values.
    
    @return: logic formula in GR(1) form representing C{ofts}.
    @rtype: L{GRSpec}
    """
    if not isinstance(ofts, transys.FiniteTransitionSystem):
        raise TypeError('ofts must be FTS, got instead: ' + str(type(ofts)))
    
    assert(ofts.owner == 'sys')
    
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
            
            action_ids, constraint = iter2var(
                codomain, sys_vars,
                action_type, bool_actions, ofts.sys_actions_must
            )
            _add_actions(constraint, sys_init, sys_trans)
            
            logger.debug('Updating sys_action_ids with:\n\t' + str(action_ids))
            sys_action_ids[action_type] = action_ids
        elif 'env' in action_type:
            logger.debug('Found env action')
            
            action_ids, constrait = iter2var(
                codomain, env_vars,
                action_type, bool_actions, ofts.env_actions_must
            )
            _add_actions(constraint, env_init, env_trans)
            
            logger.debug('Updating env_action_ids with:\n\t' + str(action_ids))
            env_action_ids[action_type] = action_ids
    
    state_ids, constraint = iter2var(states, sys_vars, statevar, bool_states, must='xor')
    if constraint is not None:
        sys_trans += constraint
    
    sys_init += _sys_init_from_ts(states, state_ids, aps, ignore_initial)
    
    sys_trans += _sys_trans_from_ts(
        states, state_ids, trans,
        sys_action_ids=sys_action_ids, env_action_ids=env_action_ids
    )
    tmp_init, tmp_trans = _ap_trans_from_ts(states, state_ids, aps)
    sys_init += tmp_init
    sys_trans += tmp_trans
    
    env_trans += _env_trans_from_sys_ts(
        states, state_ids, trans, env_action_ids
    )
    
    return GRSpec(
        sys_vars=sys_vars, env_vars=env_vars,
        env_init=env_init, sys_init=sys_init,
        env_safety=env_trans, sys_safety=sys_trans
    )

def env_to_spec(
    ofts, ignore_initial, statevar,
    bool_states=False, bool_actions=False
):
    """Convert env transition system to GR(1) representation.
    
    The following are represented by environment variables:
    
      - the current state
      - the atomic propositions annotating states
      - the environment actions annotating edges
    
    The following are represented by system variables:
    
      - the system actions annotating edges
    
    Multiple types of environment and system actions can be defined.
    
    For more details see L{sys_to_spec}.
    
    See also
    ========
    L{sys_open_fts2spec}
    """
    if not isinstance(ofts, transys.FiniteTransitionSystem):
        raise TypeError('ofts must be FTS, got instead: ' + str(type(ofts)))
    
    assert(ofts.owner == 'env')
    
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
            action_ids, constraint = iter2var(codomain, sys_vars, action_type,
                                              bool_actions, ofts.sys_actions_must)
            _add_actions(constraint, sys_init, sys_trans)
            sys_action_ids[action_type] = action_ids
        elif 'env' in action_type:
            action_ids, constraint = iter2var(codomain, env_vars, action_type,
                                              bool_actions, ofts.env_actions_must)
            _add_actions(constraint, env_init, env_trans)
            env_action_ids[action_type] = action_ids
    
    # some duplication here, because we don't know
    # whether the user will provide a system TS as well
    # and whether that TS will contain all the system actions
    # defined in the environment TS
    
    state_ids, constraint = iter2var(states, env_vars, statevar, bool_states, must='xor')
    if constraint is not None:
        env_trans += constraint
    
    env_init += _sys_init_from_ts(states, state_ids, aps, ignore_initial)
    
    env_trans += _env_trans_from_env_ts(
        states, state_ids, trans,
        env_action_ids=env_action_ids, sys_action_ids=sys_action_ids
    )
    tmp_init, tmp_trans = _ap_trans_from_ts(states, state_ids, aps)
    env_init += tmp_init
    env_trans += tmp_trans
    
    return GRSpec(
        sys_vars=sys_vars, env_vars=env_vars,
        env_init=env_init, sys_init=sys_init,
        env_safety=env_trans, sys_safety=sys_trans
    )

def _sys_init_from_ts(states, state_ids, aps, ignore_initial=False):
    """Initial state, including enforcement of exactly one.
    """
    init = []
    
    # skip ?
    if ignore_initial:
        return init
    
    if not states.initial:
        msg = (
            'FTS has no initial states.\n'
            'Enforcing this renders False the GR(1):\n'
            ' - guarantee if this is a system TS,\n'
            '   so the spec becomes trivially False.\n'
            ' - assumption if this is an environment TS,\n'
            '   so the spec becomes trivially True.'
        )
        raise Exception(msg)
        
        init += ['False']
        return init
        
    init += [_disj([state_ids[s] for s in states.initial])]
    return init

def _sys_trans_from_ts(
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
        attribute of L{FTS}.
    
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
        
        msg = ('from state: ' + str(from_state) +\
               ', the available transitions are:\n\t' + str(cur_trans))
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
            
            msg = (
                'guard to state: ' + str(to_state) +
                ', with state_id: ' + str(to_state_id) +
                ', has post-conditions: ' + str(postcond)
            )
            logger.debug(msg)
            
        sys_trans += [precond + ' -> (' + _disj(cur_str) + ')']
    return sys_trans

def _env_trans_from_sys_ts(states, state_ids, trans, env_action_ids):
    """Convert environment actions to GR(1) env_safety.
    
    This constrains the actions available next to the environment
    based on the system FTS.
    
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

def _env_trans_from_env_ts(
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
                
            msg = (
                'Environment dead-end found.\n'
                'If sys can force env to dead-end,\n'
                'then GR(1) assumption becomes False,\n'
                'and spec trivially True.'
            )
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
                
                msg = (
                    'for action_type: ' + str(action_type) +'\n' +
                    'with codomain: ' + str(codomain) +'\n' +
                    'the negated conjunction is: ' + str(conj)
                )
                logger.debug(msg)
        
        env_trans += [_pstr(precond) + ' -> (' + _disj(cur_list) +')']
    return env_trans

def _ap_trans_from_ts(states, state_ids, aps):
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
        ap_str = _sprint_aps(label, aps)
        if not ap_str:
            continue
        init += ['!(' + _pstr(state_id) + ') || (' + ap_str +')']
    
    # transitions of labels
    for state in states:
        label = states[state]
        state_id = state_ids[state]
        
        tmp = _sprint_aps(label, aps)
        if not tmp:
            continue
        
        trans += ['X((' + str(state_id) + ') -> (' + tmp + '))']
    
    return (init, trans)

def _sprint_aps(label, aps):
    if label.has_key('ap'):
        tmp0 = _conj_intersection(aps, label['ap'], parenth=False)
    else:
        tmp0 = ''
    
    if label.has_key('ap'):
        tmp1 = _conj_neg_diff(aps, label['ap'], parenth=False)
    else:
        tmp1 = _conj_neg(aps, parenth=False)
    
    if len(tmp0) > 0 and len(tmp1) > 0:
        tmp = tmp0 +' && '+ tmp1
    else:
        tmp = tmp0 + tmp1
    return tmp

def synthesize_many(specs, ts=None, ignore_init=None,
                    bool_actions=None, solver='gr1c'):
    """Synthesize from logic specs and multiple transition systems.
    
    @type specs: L{GRSpec}
    
    @type ts: C{dict} of L{FiniteTransitionSystem}
    
    @type ignore_init: C{set} of keys from C{ts}
    
    @type bool_actions: C{set} of keys from C{ts}
    
    @param solver: 'gr1c' or 'jtlv'
    @type solver: str
    """
    assert(isinstance(ts, dict))
    
    for name, t in ts.iteritems():
        assert(isinstance(t, transys.FiniteTransitionSystem))
        
        ignore = name in ignore_init
        bool_act = name in bool_actions
        statevar = name + '_state'
        
        if t.owner == 'sys':
            specs |= sys_to_spec(t, ignore, statevar,
                                 bool_actions=bool_act)
        elif t.owner == 'env':
            specs |= env_to_spec(t, ignore, statevar,
                                 bool_actions=bool_act)
    
    if solver == 'gr1c':
        ctrl = gr1c.synthesize(specs)
    elif solver == 'jtlv':
        ctrl = jtlv.synthesize(specs)
    else:
        raise Exception('Unknown solver: ' + str(solver) + '. '
                        'Available solvers: "jtlv" and "gr1c"')
    
    try:
        logger.debug('Mealy machine has: n = ' +
                     str(len(ctrl.states) ) +' states.')
    except:
        logger.debug('No Mealy machine returned.')
    
    # no controller found ?
    # counterstrategy not constructed by synthesize
    if not isinstance(ctrl, transys.MealyMachine):
        return None
    
    ctrl.remove_deadends()

    return ctrl

def synthesize(
    option, specs, env=None, sys=None,
    ignore_env_init=False, ignore_sys_init=False,
    bool_states=False,
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
    @type env: L{FTS}
    
    @param sys: A transition system describing the system:
        
            - states controlled by the system
            - input: env_actions
            - output: sys_actions
            - initial states constrain the system
    
    @type sys: L{FTS}
    
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
    specs = _spec_plus_sys(
        specs, env, sys,
        ignore_env_init,
        ignore_sys_init,
        bool_states,
        bool_actions
    )
    
    if option == 'gr1c':
        strategy = gr1c.synthesize(specs)
    elif option == 'jtlv':
        strategy = jtlv.synthesize(specs)
    else:
        raise Exception('Undefined synthesis option. ' +
                        'Current options are "jtlv" and "gr1c"')
    
    ctrl = strategy2mealy(strategy, specs)
    
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
    bool_states=False,
    bool_actions=False
):
    """Check realizability.
    
    For details see L{synthesize}.
    """
    specs = _spec_plus_sys(
        specs, env, sys,
        ignore_env_init, ignore_sys_init,
        bool_states, bool_actions
    )
    
    if option == 'gr1c':
        r = gr1c.check_realizable(specs)
    elif option == 'jtlv':
        r = jtlv.check_realizable(specs)
    else:
        raise Exception('Undefined synthesis option. ' +
                        'Current options are "jtlv" and "gr1c"')
    
    if r:
        logger.debug('is realizable')
    else:
        logger.debug('is not realizable')
    
    return r

def _spec_plus_sys(
    specs, env, sys,
    ignore_env_init, ignore_sys_init,
    bool_states, bool_actions
):
    if sys is not None:
        sys_formula = sys_to_spec(
            sys, ignore_sys_init,
            bool_states=bool_states,
            bool_actions=bool_actions
        )
        specs = specs | sys_formula
        logger.debug('sys TS:\n' + str(sys_formula.pretty() ) + _hl)
    
    if env is not None:
        env_formula = env_to_spec(
            env, ignore_env_init,
            bool_states=bool_states,
            bool_actions=bool_actions
        )
        specs = specs | env_formula
        logger.debug('env TS:\n' + str(env_formula.pretty() ) + _hl)
        
    logger.info('Overall Spec:\n' + str(specs.pretty() ) +_hl)
    return specs

def strategy2mealy(A, spec):
    """Convert strategy to Mealy transducer.
    
    Note that the strategy is a deterministic game graph,
    but the input C{A} is given as the contraction of
    this game graph (which directly )
    
    @param A: strategy
    @type A: C{networkx.DiGraph}
    
    @return: tuple of the form (L{GRSpec}, L{MealyMachine}).  Either
        or both can be None if the corresponding part is missing.
        Note that the returned GRSpec instance depends only on what is
        in the given tulipcon XML string x, not on the argument spec0.
    """
    logger.info('converting strategy (compact) to Mealy machine')
    
    env_vars = spec.env_vars
    sys_vars = spec.sys_vars
    
    mach = transys.MealyMachine()
    
    inputs = transys.machines.create_machine_ports(env_vars)
    mach.add_inputs(inputs)
    
    outputs = transys.machines.create_machine_ports(sys_vars)
    mach.add_outputs(outputs)
    
    str_vars = {
        k:v for k, v in env_vars.items()
        if isinstance(v, list)
    }
    str_vars.update({
        k:v for k, v in sys_vars.items()
        if isinstance(v, list)
    })
    
    mach.states.add_from(A)
    
    # transitions labeled with I/O
    for u in A:
        for v in A.successors_iter(u):
            d = A.node[v]['state']
            d = _int2str(d, str_vars)
            mach.transitions.add(u, v, **d)
            
            logger.info('node: ' + str(v) + ', state: ' + str(d))
    
    # special initial state, for first reaction
    initial_state = 'Sinit'
    mach.states.add(initial_state)
    mach.states.initial.add(initial_state)
    
    # fix an ordering for keys
    # because tuple(dict.iteritems()) is not safe:
    # https://docs.python.org/2/library/stdtypes.html#dict.items
    try:
        u = next(iter(A))
        keys = A.node[u]['state'].keys()
    except:
        logger.warn('strategy has no states.')
    
    # to store tuples of dict values for fast search
    init_valuations = set()
    isinit = spec.compile_init(no_str=True)
    
    # Mealy reaction to initial env input
    tmp = dict()
    for u, d in A.nodes_iter(data=True):
        var_values = d['state']
        vals = tuple(var_values[k] for k in keys)
        
        # already an initial valuation ?
        if vals in init_valuations:
            continue
        
        # add edge: Sinit -> u ?
        tmp.update(var_values)
        if eval(isinit, tmp):
            label = _int2str(var_values, str_vars)
            mach.transitions.add(initial_state, u, **label)
            
            # remember variable values to avoid
            # spurious non-determinism wrt the machine's memory
            init_valuations.add(vals)
            
            logger.debug('found initial state: ' + str(u))
        
        logger.debug(
            'in state: ' + str(u) +
            ', var values: ' + str(var_values)
        )
    
    if not mach.successors('Sinit'):
        import pprint
        raise Exception(
            'The machine obtained from the strategy '
            'does not have any initial states !\n'
            'The strategy is:\n'
            'vertices:' + pprint.pformat(A.nodes(data=True)) + 2*'\n' +
            'edges:\n' + str(A.edges()) + 2*'\n' +
            'and the machine:\n' + str(mach) + 2*'\n' +
            'and the specification is:\n' + str(spec.pretty()) + 2*'\n'
        )
    
    return mach

def _int2str(label, str_vars):
    """Replace integers with string values for string variables in C{label}.
    """
    label = dict(label)
    label.update({k:str_vars[k][int(v)] for k,v in label.iteritems()
                                        if k in str_vars})
    return label

def mask_outputs(machine):
    """Erase outputs from each edge where they are zero.
    """
    for u, v, d in machine.edges_iter(data=True):
        for k in d:
            if k in machine.outputs and d[k] == 0:
                d.pop(k)

# Copyright (c) 2012-2015 by California Institute of Technology
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
"""Interface to library of synthesis tools, e.g., JTLV, gr1c"""
from __future__ import absolute_import
import copy
import logging
import pprint
import warnings

from tulip.interfaces import gr1c
from tulip.interfaces import gr1py
from tulip.interfaces import jtlv
from tulip.interfaces import omega as omega_int
try:
    from tulip.interfaces import slugs
except ImportError:
    slugs = None
from tulip.spec import GRSpec
from tulip import transys


logger = logging.getLogger(__name__)
_hl = '\n' + 60 * '-'


def _pstr(s):
    return '(' + str(s) + ')'


def _disj(set0):
    return ' || '.join([
        '(' + str(x) + ')'
        for x in set0])


def _conj(set0):
    return ' && '.join([
        '(' + str(x) + ')'
        for x in set0
        if x != ''])


def _conj_intersection(set0, set1, parenth=True):
    if parenth:
        return ' && '.join([
            '(' + str(x) + ')'
            for x in set0
            if x in set1])
    else:
        return ' && '.join([
            str(x)
            for x in set0
            if x in set1])


def _conj_neg(set0, parenth=True):
    if parenth:
        return ' && '.join([
            '!(' + str(x) + ')'
            for x in set0])
    else:
        return ' && '.join([
            '!' + str(x)
            for x in set0])


def _conj_neg_diff(set0, set1, parenth=True):
    if parenth:
        return ' && '.join([
            '!(' + str(x) + ')'
            for x in set0
            if x not in set1])
    else:
        return ' && '.join([
            '!' + str(x)
            for x in set0
            if x not in set1])


def mutex(iterable):
    """Mutual exclusion for all time."""
    iterable = filter(lambda x: x != '', iterable)
    if not iterable:
        return list()
    if len(iterable) <= 1:
        return []
    return [_conj([
        '!(' + str(x) + ') || (' + _conj_neg_diff(iterable, [x]) + ')'
        for x in iterable])]


def exactly_one(iterable):
    """N-ary xor.

    Contrast with pure mutual exclusion.
    """
    if len(iterable) <= 1:
        return [_pstr(x) for x in iterable]
    return ['(' + _disj([
        '(' + str(x) + ') && ' + _conj_neg_diff(iterable, [x])
        for x in iterable]) + ')']


def _conj_action(actions_dict, action_type, nxt=False, ids=None):
    """Return conjunct if C{action_type} in C{actions_dict}.

    @param actions_dict: C{dict} with pairs C{action_type_name : action_value}
    @type actions_dict: dict

    @param action_type: key to look for in C{actions_dict}
    @type action_type: hashable (here typically a str)

    @param nxt: prepend or not with the next operator
    @type nxt: bool

    @param ids: map C{action_value} -> value used in solver input,
        for example, for gr1c
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
    logger.debug('conjuncted actions: ' + str(conjuncted_actions) + '\n')
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

    Suppose N possible values are defined.
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

    @param statevar: name to use for integer or string valued variable.
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
    assert must in {'mutex', 'xor', None}
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
        'mutex: ' + str(use_mutex) + '\n\t'
        'min_one: ' + str(min_one))
    all_str = all(isinstance(x, str) for x in states)
    if bool_states:
        logger.debug('states modeled as Boolean variables')
        if not all_str:
            raise TypeError('If Boolean, all states must be strings.')
        state_ids = {x: x for x in states}
        variables.update({s: 'boolean' for s in states})
        # single action ?
        if len(mutex(state_ids.values())) == 0:
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
            assert use_mutex
            f = lambda x: statevar + ' = "' + str(x) + '"'
            domain = list(states)
            if not min_one:
                domain += [statevar + 'none']

                logger.debug(
                    'domain has been extended, because all actions\n\t'
                    'could be False (constraint: min_one = False).')
        else:
            raise TypeError('Integer and string states must not be mixed.')
        state_ids = {x: f(x) for x in states}
        variables[statevar] = domain
        constraint = None
    logger.debug(
        'for tulip variable: ' + str(statevar) + '\n'
        'the map from [tulip action values] ---> '
        '[solver expressions] is:\n' + 2 * '\t' + str(state_ids))
    return state_ids, constraint


def _add_actions(constraint, init, trans):
    if constraint is None:
        return
    trans += ['X (' + constraint[0] + ')']
    init += constraint


def _fts2spec(
    fts, ignore_initial,
    statevar, actionvar=None,
    bool_states=False, bool_actions=False
):
    """Convert closed FTS to GR(1) representation."""
    raise Exception('deprecated')
    assert isinstance(fts, transys.FiniteTransitionSystem)
    aps = fts.aps
    states = fts.states
    actions = fts.actions
    sys_init = list()
    sys_trans = list()
    sys_vars = {ap: 'boolean' for ap in aps}
    action_ids, constraint = iter2var(
        actions, sys_vars, actionvar, bool_actions, fts.actions_must)
    _add_actions(constraint, sys_init, sys_trans)
    state_ids, constraint = iter2var(states, sys_vars, statevar,
                                     bool_states, must='xor')
    if constraint is not None:
        sys_trans += constraint
    sys_init += _sys_init_from_ts(states, state_ids, aps, ignore_initial)
    sys_trans += _sys_trans_from_ts(
        states, state_ids, fts.transitions,
        action_ids=action_ids)
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
    assert ofts.owner == 'sys'
    aps = ofts.aps
    states = ofts.states
    trans = ofts.transitions
    # init
    sys_init = list()
    sys_trans = list()
    env_init = list()
    env_trans = list()
    sys_vars = {ap: 'boolean' for ap in aps}
    env_vars = dict()
    actions = ofts.actions
    sys_action_ids = dict()
    env_action_ids = dict()
    for action_type, codomain in actions.iteritems():
        msg = 'action_type:\n\t' + str(action_type) + '\n'
        msg += 'with codomain:\n\t' + str(codomain)
        logger.debug(msg)
        if 'sys' in action_type:
            logger.debug('Found sys action')
            action_ids, constraint = iter2var(
                codomain, sys_vars,
                action_type, bool_actions, ofts.sys_actions_must)
            _add_actions(constraint, sys_init, sys_trans)
            logger.debug('Updating sys_action_ids with:\n\t' + str(action_ids))
            sys_action_ids[action_type] = action_ids
        elif 'env' in action_type:
            logger.debug('Found env action')
            action_ids, constrait = iter2var(
                codomain, env_vars,
                action_type, bool_actions, ofts.env_actions_must)
            _add_actions(constraint, env_init, env_trans)
            logger.debug('Updating env_action_ids with:\n\t' + str(action_ids))
            env_action_ids[action_type] = action_ids
    state_ids, constraint = iter2var(states, sys_vars, statevar,
                                     bool_states, must='xor')
    if constraint is not None:
        sys_trans += constraint
    sys_init += _sys_init_from_ts(states, state_ids, aps, ignore_initial)
    sys_trans += _sys_trans_from_ts(
        states, state_ids, trans,
        sys_action_ids=sys_action_ids, env_action_ids=env_action_ids)
    tmp_init, tmp_trans = _ap_trans_from_ts(states, state_ids, aps)
    sys_init += tmp_init
    sys_trans += tmp_trans
    env_trans += _env_trans_from_sys_ts(
        states, state_ids, trans, env_action_ids)
    return GRSpec(
        sys_vars=sys_vars, env_vars=env_vars,
        env_init=env_init, sys_init=sys_init,
        env_safety=env_trans, sys_safety=sys_trans)


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
    assert ofts.owner == 'env'
    aps = ofts.aps
    states = ofts.states
    trans = ofts.transitions
    # init
    sys_init = list()
    sys_trans = list()
    env_init = list()
    env_trans = list()
    # since APs are tied to env states, let them be env variables
    env_vars = {ap: 'boolean' for ap in aps}
    sys_vars = dict()
    actions = ofts.actions
    sys_action_ids = dict()
    env_action_ids = dict()
    for action_type, codomain in actions.iteritems():
        if 'sys' in action_type:
            action_ids, constraint = iter2var(
                codomain, sys_vars, action_type,
                bool_actions, ofts.sys_actions_must)
            _add_actions(constraint, sys_init, sys_trans)
            sys_action_ids[action_type] = action_ids
        elif 'env' in action_type:
            action_ids, constraint = iter2var(
                codomain, env_vars, action_type,
                bool_actions, ofts.env_actions_must)
            _add_actions(constraint, env_init, env_trans)
            env_action_ids[action_type] = action_ids
    # some duplication here, because we don't know
    # whether the user will provide a system TS as well
    # and whether that TS will contain all the system actions
    # defined in the environment TS
    state_ids, constraint = iter2var(states, env_vars, statevar,
                                     bool_states, must='xor')
    if constraint is not None:
        env_trans += constraint
    env_init += _sys_init_from_ts(states, state_ids, aps, ignore_initial)
    env_trans += _env_trans_from_env_ts(
        states, state_ids, trans,
        env_action_ids=env_action_ids, sys_action_ids=sys_action_ids)
    tmp_init, tmp_trans = _ap_trans_from_ts(states, state_ids, aps)
    env_init += tmp_init
    env_trans += tmp_trans
    return GRSpec(
        sys_vars=sys_vars, env_vars=env_vars,
        env_init=env_init, sys_init=sys_init,
        env_safety=env_trans, sys_safety=sys_trans)


def _sys_init_from_ts(states, state_ids, aps, ignore_initial=False):
    """Initial state, including enforcement of exactly one."""
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
            '   so the spec becomes trivially True.')
        raise Exception(msg)
        init += ['False']
        return init
    init += [_disj([state_ids[s] for s in states.initial])]
    return init


def _sys_trans_from_ts(
    states, state_ids, trans,
    action_ids=None, sys_action_ids=None, env_action_ids=None
):
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
    sys_trans = list()
    # Transitions
    for from_state in states:
        from_state_id = state_ids[from_state]
        precond = _pstr(from_state_id)
        cur_trans = trans.find([from_state])
        msg = ('from state: ' + str(from_state) +
               ', the available transitions are:\n\t' + str(cur_trans))
        logger.debug(msg)
        # no successor states ?
        if not cur_trans:
            logger.debug('state: ' + str(from_state) + ' is deadend !')
            sys_trans += [precond + ' -> X(False)']
            continue
        cur_str = list()
        for (from_state, to_state, label) in cur_trans:
            to_state_id = state_ids[to_state]
            postcond = ['X' + _pstr(to_state_id)]
            logger.debug('label = ' + str(label))
            if 'previous' in label:
                previous = label['previous']
            else:
                previous = set()
            logger.debug('previous = ' + str(previous))
            env_actions = {k: v for k, v in label.iteritems() if 'env' in k}
            prev_env_act = {k: v for k, v in env_actions.iteritems()
                            if k in previous}
            next_env_act = {k: v for k, v in env_actions.iteritems()
                            if k not in previous}
            postcond += [_conj_actions(prev_env_act, env_action_ids,
                                       nxt=False)]
            postcond += [_conj_actions(next_env_act, env_action_ids,
                                       nxt=True)]
            sys_actions = {k: v for k, v in label.iteritems() if 'sys' in k}
            prev_sys_act = {k: v for k, v in sys_actions.iteritems()
                            if k in previous}
            next_sys_act = {k: v for k, v in sys_actions.iteritems()
                            if k not in previous}
            postcond += [_conj_actions(prev_sys_act, sys_action_ids,
                                       nxt=False)]
            postcond += [_conj_actions(next_sys_act, sys_action_ids,
                                       nxt=True)]
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
                ', has post-conditions: ' + str(postcond))
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
    env_trans = list()
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
            # for action_type, codomain_map in env_action_ids.iteritems():
            # env_trans += [precond + ' -> X(' + s + ')']
            continue
        # collect possible next env actions
        next_env_action_combs = set()
        for (from_state, to_state, label) in cur_trans:
            env_actions = {k: v for k, v in label.iteritems() if 'env' in k}
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
    env_trans = list()
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
                'and spec trivially True.')
            warnings.warn(msg)
            continue
        cur_list = list()
        found_free = False  # any environment transition
        # not conditioned on the previous system output ?
        for (from_state, to_state, label) in cur_trans:
            to_state_id = state_ids[to_state]
            postcond = ['X' + _pstr(to_state_id)]
            env_actions = {k: v for k, v in label.iteritems() if 'env' in k}
            postcond += [_conj_actions(env_actions, env_action_ids, nxt=True)]
            # remember: this is an environment FTS, so no next for sys
            sys_actions = {k: v for k, v in label.iteritems() if 'sys' in k}
            postcond += [_conj_actions(sys_actions, sys_action_ids)]
            postcond += [_conj_action(label, 'actions', nxt=True,
                                      ids=action_ids)]
            # todo: test this claus
            if not sys_actions:
                found_free = True
            cur_list += [_conj(postcond)]
        # can sys kill env by setting all previous sys outputs to False ?
        # then env assumption becomes False,
        # so the spec trivially True: avoid this
        if not found_free and sys_action_ids:
            msg = 'no free env outgoing transition found\n' +\
                  'instead will take disjunction with negated sys actions'
            logger.debug(msg)
            for action_type, codomain in sys_action_ids.iteritems():
                conj = _conj_neg(codomain.itervalues())
                cur_list += [conj]
                msg = (
                    'for action_type: ' + str(action_type) + '\n' +
                    'with codomain: ' + str(codomain) + '\n' +
                    'the negated conjunction is: ' + str(conj))
                logger.debug(msg)
        env_trans += [_pstr(precond) + ' -> (' + _disj(cur_list) + ')']
    return env_trans


def _ap_trans_from_ts(states, state_ids, aps):
    """Require atomic propositions to follow states according to label.
    """
    init = list()
    trans = list()
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
        init += ['!(' + _pstr(state_id) + ') || (' + ap_str + ')']
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
    if 'ap' in label:
        tmp0 = _conj_intersection(aps, label['ap'], parenth=False)
    else:
        tmp0 = ''
    if 'ap' in label:
        tmp1 = _conj_neg_diff(aps, label['ap'], parenth=False)
    else:
        tmp1 = _conj_neg(aps, parenth=False)
    if len(tmp0) > 0 and len(tmp1) > 0:
        tmp = tmp0 + ' && ' + tmp1
    else:
        tmp = tmp0 + tmp1
    return tmp


def build_dependent_var_table(fts, statevar):
    """Return a C{dict} of substitution rules for dependent variables.

    The dependent variables in a transition system are the
    atomic propositions that are used to label states.

    They are "dependent" because their values are completely
    determined by knowledge of the current state in the
    transition system.

    The returned substitutions can be used

    @type fts: L{FTS}

    @param statevar: name of variable used for the current state
        For example if it is 'loc', then the states
        C{'s0', 's1'} are mapped to::

          {'s0': '(loc = "s0")',
           's1': '(loc = "s1")'}

    @type state_ids: C{dict}

    @rtype: C{{'p': '((loc = "s1") | (loc = "s2") | ...)', ...}}
        where:

          - C{'p'} is a proposition in C{fts.atomic_propositions}
          - the states "s1", "s2" are labeled with C{'p'}
          - C{loc} is the string variable used for the state of C{fts}.
    """
    state_ids, __ = iter2var(fts.states, variables=dict(), statevar=statevar,
                             bool_states=False, must='xor')
    ap2states = map_ap_to_states(fts)
    return {k: _disj(state_ids[x] for x in v)
            for k, v in ap2states.iteritems()}


def map_ap_to_states(fts):
    """For each proposition find the states labeled with it.

    @type fts: L{FTS}

    @rtype: C{{'p': s, ...}} where C{'p'} a proposition and
        C{s} a set of states in C{fts}.
    """
    table = {p: set() for p in fts.atomic_propositions}
    for u in fts:
        for p in fts.node[u]['ap']:
            table[p].add(u)
    return table


def synthesize_many(specs, ts=None, ignore_init=None,
                    bool_actions=None, solver='gr1c'):
    """Synthesize from logic specs and multiple transition systems.

    The transition systems are composed synchronously, i.e.,
    they all have to take a transition at each time step.
    The synchronous composition is obtained by taking the
    conjunction of the formulas describing each transition system.

    The states of each transition system can be either:

      - all integers, or
      - all strings

    In either case the transition system state will be
    represented in logic with a single variable,
    that ranges over a finite set of integers or strings, respectively.

    The keys of C{ts} are used to name each state variable.
    So the logic formula for C{ts['name']} will be C{'name'}.

    Who controls this state variable is determined from
    the attribute C{FTS.owner} that can take the values:

      - C{'env'}
      - C{'sys'}

    For example:

      >>> ts.states.add_from(xrange(4))
      >>> ts['door'].owner = 'env'

    will result in a logic formula with
    an integer variable C{'door'}
    controlled by the environment and
    taking values over C{{0, 1, 2, 3}}.

    The example:

      >>> ts.states.add_from(['a', 'b', 'c'])
      >>> ts['door'].owner = 'sys'

    will instead result in a string variable C{'door'}
    controlled by the system and taking
    values over C{{'a', 'b', 'c'}}.

    @type specs: L{GRSpec}

    @type ts: C{dict} of L{FiniteTransitionSystem}

    @type ignore_init: C{set} of keys from C{ts}

    @type bool_actions: C{set} of keys from C{ts}

    @param solver: 'gr1c' or 'slugs' or 'jtlv'
    @type solver: str
    """
    assert isinstance(ts, dict)
    for name, t in ts.iteritems():
        assert isinstance(t, transys.FiniteTransitionSystem)
        ignore = name in ignore_init
        bool_act = name in bool_actions
        statevar = name
        if t.owner == 'sys':
            specs |= sys_to_spec(t, ignore, statevar,
                                 bool_actions=bool_act)
        elif t.owner == 'env':
            specs |= env_to_spec(t, ignore, statevar,
                                 bool_actions=bool_act)
    if solver == 'gr1c':
        ctrl = gr1c.synthesize(specs)
    elif solver == 'slugs':
        if slugs is None:
            raise ValueError('Import of slugs interface failed. ' +
                             'Please verify installation of "slugs".')
        ctrl = slugs.synthesize(specs)
    elif solver == 'jtlv':
        ctrl = jtlv.synthesize(specs)
    else:
        raise Exception('Unknown solver: ' + str(solver) + '. '
                        'Available solvers: "jtlv", "gr1c", and "slugs"')
    try:
        logger.debug('Mealy machine has: n = ' +
                     str(len(ctrl.states)) + ' states.')
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
    rm_deadends=True
):
    """Function to call the appropriate synthesis tool on the specification.

    There are three attributes of C{specs} that define what
    kind of controller you are looking for:

    1. C{moore}: What information the controller knows when deciding the next
       values of controlled variables:
        - Moore: can read current state,
          but not next environment variable values, or
        - Mealy: can read current state and next environment variable values.

    2. C{qinit}: Quantification of initial variable values:
        Whether all states that satisfy a predicate should be winning,
        or the initial values of some (or all) the variables is
        subject to the synthesizer's choice.

    3. C{plus_one}: The form of assume-guarantee specification,
        i.e., how the system guarantees relate to assumptions about the
        environment.

    For more details about these attributes, see L{GRSpec}.

    The states of the transition system can be either:

      - all integers, or
      - all strings

    For more details of how the transition system is represented in
    logic look at L{synthesize_many}.

    Beware!
    =======
    This function provides a generic interface to a variety
    of routines.  Being under active development, the types of
    arguments supported and types of objects returned may change
    without notice.

    @param option: Magic string that declares what tool to invoke,
        what method to use, etc.  Currently recognized forms:

        For GR(1) synthesis:

          - C{"gr1c"}: use gr1c via L{interfaces.gr1c}.
            written in C using CUDD, symbolic

          - C{"gr1py"}: use gr1py via L{interfaces.gr1py}.
            Python, enumerative

          - C{"omega"}: use omega via L{interfaces.omega}.
            Python using C{dd} or Cython using CUDD, symbolic

          - C{"slugs"}: use slugs via L{interfaces.slugs}.
            C++ using CUDD, symbolic

          - C{"jtlv"}: use JTLV via L{interfaces.jtlv}.
            Java, symbolic
            (deprecated)

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

    @param ignore_env_init: Ignore any initial state information
        contained in env.
    @type ignore_env_init: bool

    @param ignore_sys_init: Ignore any initial state information
        contained in sys.
    @type ignore_sys_init: bool

    @param rm_deadends: return a strategy that contains no terminal states.
    @type rm_deadends: bool

    @return: If spec is realizable,
        then return a Mealy machine implementing the strategy.
        Otherwise return None.
    @rtype: L{MealyMachine} or None
    """
    specs = _spec_plus_sys(
        specs, env, sys,
        ignore_env_init,
        ignore_sys_init)
    if option == 'gr1c':
        strategy = gr1c.synthesize(specs)
    elif option == 'slugs':
        if slugs is None:
            raise ValueError('Import of slugs interface failed. ' +
                             'Please verify installation of "slugs".')
        strategy = slugs.synthesize(specs)
    elif option == 'gr1py':
        strategy = gr1py.synthesize(specs)
    elif option == 'omega':
        strategy = omega_int.synthesize_enumerated_streett(specs)
    elif option == 'jtlv':
        strategy = jtlv.synthesize(specs)
        if isinstance(strategy, list):
            # Discard counter-examples, because here we only care that
            # it is not realizable.
            strategy = None
    else:
        raise Exception('Undefined synthesis option. ' +
                        'Current options are "gr1c", ' +
                        '"slugs", "gr1py", "omega", and "jtlv".')

    # While the return values of the solver interfaces vary, we expect
    # here that strategy is either None to indicate unrealizable or a
    # networkx.DiGraph ready to be passed to strategy2mealy().
    if strategy is None:
        return None

    ctrl = strategy2mealy(strategy, specs)
    logger.debug('Mealy machine has: n = ' +
                 str(len(ctrl.states)) + ' states.')

    if rm_deadends:
        ctrl.remove_deadends()
    return ctrl


def is_realizable(
    option, specs, env=None, sys=None,
    ignore_env_init=False, ignore_sys_init=False
):
    """Check realizability.

    For details see L{synthesize}.
    """
    specs = _spec_plus_sys(
        specs, env, sys,
        ignore_env_init, ignore_sys_init)
    if option == 'gr1c':
        r = gr1c.check_realizable(specs)
    elif option == 'slugs':
        if slugs is None:
            raise ValueError('Import of slugs interface failed. ' +
                             'Please verify installation of "slugs".')
        r = slugs.check_realizable(specs)
    elif option == 'gr1py':
        r = gr1py.check_realizable(specs)
    elif option == 'omega':
        r = omega_int.is_realizable(specs)
    elif option == 'jtlv':
        r = jtlv.check_realizable(specs)
    else:
        raise Exception('Undefined synthesis option. ' +
                        'Current options are "jtlv", "gr1c", ' +
                        '"slugs", and "gr1py"')
    if r:
        logger.debug('is realizable')
    else:
        logger.debug('is not realizable')
    return r


def _spec_plus_sys(
    specs, env, sys,
    ignore_env_init, ignore_sys_init
):
    if sys is not None:
        if hasattr(sys, 'state_varname'):
            statevar = sys.state_varname
        else:
            logger.info('sys.state_varname undefined. '
                        'Will use the default variable name: "loc".')
            statevar = 'loc'
        sys_formula = sys_to_spec(
            sys, ignore_sys_init,
            bool_states=False,
            bool_actions=False,
            statevar=statevar)
        # consider sys just a formula,
        # not a synthesis problem
        # so overwrite settings
        if hasattr(sys, 'moore'):
            cp = sys
        else:
            cp = specs
        sys_formula.moore = cp.moore
        sys_formula.plus_one = cp.plus_one
        sys_formula.qinit = cp.qinit
        specs = specs | sys_formula
        logger.debug('sys TS:\n' + str(sys_formula.pretty()) + _hl)
    if env is not None:
        if hasattr(env, 'state_varname'):
            statevar = sys.state_varname
        else:
            logger.info('env.state_varname undefined. '
                        'Will use the default variable name: "eloc".')
            statevar = 'eloc'
        env_formula = env_to_spec(
            env, ignore_env_init,
            bool_states=False,
            bool_actions=False,
            statevar=statevar)
        if hasattr(env, 'moore'):
            cp = env
        else:
            cp = specs
        env_formula.moore = cp.moore
        env_formula.plus_one = cp.plus_one
        env_formula.qinit = cp.qinit
        specs = specs | env_formula
        logger.debug('env TS:\n' + str(env_formula.pretty()) + _hl)
    logger.info('Overall Spec:\n' + str(specs.pretty()) + _hl)
    return specs


def strategy2mealy(A, spec):
    """Convert strategy to Mealy transducer.

    Note that the strategy is a deterministic game graph,
    but the input C{A} is given as the contraction of
    this game graph.

    @param A: strategy
    @type A: C{networkx.DiGraph}

    @type spec: L{GRSpec}

    @rtype: L{MealyMachine}
    """
    assert len(A) > 0
    logger.info('converting strategy (compact) to Mealy machine')
    env_vars = spec.env_vars
    sys_vars = spec.sys_vars
    mach = transys.MealyMachine()
    inputs = transys.machines.create_machine_ports(env_vars)
    mach.add_inputs(inputs)
    outputs = transys.machines.create_machine_ports(sys_vars)
    mach.add_outputs(outputs)
    str_vars = {
        k: v for k, v in env_vars.iteritems()
        if isinstance(v, list)}
    str_vars.update({
        k: v for k, v in sys_vars.iteritems()
        if isinstance(v, list)})
    mach.states.add_from(A)
    # transitions labeled with I/O
    for u in A:
        for v in A.successors_iter(u):
            d = A.node[v]['state']
            d = _int2str(d, str_vars)
            mach.transitions.add(u, v, **d)

            logger.info('node: {v}, state: {d}'.format(v=v, d=d))
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
    except Exception:
        logger.warn('strategy has no states.')
    # to store tuples of dict values for fast search
    isinit = spec.compile_init(no_str=True)
    # Mealy reaction to initial env input
    init_valuations = set()
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
            #
            # in other words,
            # "state" omits the strategy's memory
            # hidden (existentially quantified)
            # so multiple nodes can be labeled with the same state
            #
            # non-uniqueness here would be equivalent to
            # multiple choices for initializing the hidden memory.
            init_valuations.add(vals)
            logger.debug('found initial state: {u}'.format(u=u))
        logger.debug('machine vertex: {u}, has var values: {v}'.format(
                     u=u, v=var_values))
    n = len(A)
    m = len(mach)
    assert m == n + 1, (n, m)
    if not mach.successors('Sinit'):
        raise Exception(
            'The machine obtained from the strategy '
            'does not have any initial states !\n'
            'The strategy is:\n'
            'vertices:' + pprint.pformat(A.nodes(data=True)) + 2 * '\n' +
            'edges:\n' + str(A.edges()) + 2 * '\n' +
            'and the machine:\n' + str(mach) + 2 * '\n' +
            'and the specification is:\n' + str(spec.pretty()) + 2 * '\n')
    return mach


def _int2str(label, str_vars):
    """Replace integers with string values for string variables.

    @param label: mapping from variable names, to integer (as strings)
    @type label: C{dict}

    @param str_vars: mapping that defines those variables that
        should be converted from integer to string variables.
        Each variable is mapped to a list of strings that
        comprise its range. This list defines how integer values
        correspond to string literals for that variable.
    @type str_vars: C{dict}

    @rtype: C{dict}
    """
    label = dict(label)
    label.update({k: str_vars[k][int(v)]
                 for k, v in label.iteritems()
                 if k in str_vars})
    return label


def mask_outputs(machine):
    """Erase outputs from each edge where they are zero."""
    for u, v, d in machine.edges_iter(data=True):
        for k in d:
            if k in machine.outputs and d[k] == 0:
                d.pop(k)


def determinize_machine_init(mach, init_out_values=None):
    """Return a determinized copy of C{mach} with given initial outputs.

    The transducers produced by synthesis can have multiple
    initial output valuations as possible reactions to a
    given input valuation.

    Possible reasons for this are:

      1. the system does not have full control over its initial state.
        For example the option "ALL_INIT" of C{gr1c}.

      2. the strategy returned by the solver has multiple
        vertices that satisfy the initial conditions.

    Case 1
    ======
    Requires an initial condition to be specified for
    each run of the transducer, because the transducer does
    not have full freedom to pick the initial output values.

    Note that solver options like "ALL_INIT"
    assume that the system has no control initially.
    Any output valuation that satisfies the initial
    conditions can occur.

    However, this may be too restrictive.
    The system may have control over the initial values of
    some outputs, but not others.

    For the outputs it can initially control,
    the non-determinism resulting from synthesis is redundancy
    and can be removed arbitrarily, as in Case 2.

    Case 2
    ======
    The function L{strategy2mealy} returns a transducer that
    for each initial input valuation,
    for each initial output valuation,
    reacts with a unique transition.

    But this can yield multile reactions to a single input,
    even for solver options like "ALL_ENV_EXIST_SYS_INIT" for C{gr1c}.
    The reason is that there can be multiple strategy vertices
    that satisfy the initial conditions, but the solver
    included them not because they are needed as initial reactions,
    but to be visited later by the strategy.

    These redundant initial reactions can be removed,
    and because the system has full control over their values,
    they can be removed in an arbitrary manner,
    keeping only a single reaction, for each input valuation.

    Algorithm
    =========
    Returns a deterministic transducer.
    This means that at each transducer vertex,
    for each input valuation,
    there is only a single reaction (output valuation) available.

    The non-determinism is resolved for the initial reaction
    by ensuring the outputs given in C{init_out_values}
    take those values.
    The remaining outputs are determinized arbitrarily.

    See also
    ========
    L{synthesize}, L{strategy2mealy}

    @param mach: possibly non-deterministic transducer,
        as produced, for example, by L{synthesize}.
    @type mach: L{MealyMachine}

    @param init_out_values: mapping from output ports that
        the system cannot control initially,
        to the initial values they take in this instance of the game.
    @type init_out_values: C{dict}

    @rtype: L{MealyMachine}
    """
    mach = copy.deepcopy(mach)
    if init_out_values is None:
        init_out_values = dict()
    '''determinize given outputs (uncontrolled)'''
    # restrict attention to given output ports only
    given_ports = tuple(k for k in mach.outputs if k in init_out_values)
    rm_edges = set()
    for i, j, key, d in mach.edges_iter(['Sinit'], data=True, keys=True):
        for k in given_ports:
            if d[k] != init_out_values[k]:
                rm_edges.add((i, j, key))
                break
    mach.remove_edges_from(rm_edges)
    '''determinize arbitrarily any remnant non-determinism'''
    # input valuations already seen
    # tuples of values used for efficiency (have __hash__)
    possible_inputs = set()
    # fix a key order
    inputs = tuple(k for k in mach.inputs)
    rm_edges = set()
    for i, j, key, d in mach.edges_iter(['Sinit'], data=True, keys=True):
        in_values = tuple(d[k] for k in inputs)
        # newly encountered input valuation ?
        if in_values not in possible_inputs:
            possible_inputs.add(in_values)
            continue
        else:
            rm_edges.add((i, j, key))
    mach.remove_edges_from(rm_edges)
    return mach

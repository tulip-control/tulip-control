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
"""Interface to library of synthesis tools, e.g., `gr1c`, `omega`."""
import collections.abc as _abc
import copy
import logging
import pprint
import typing as _ty
import warnings

import networkx as _nx

import tulip.interfaces.gr1c as gr1c
import tulip.interfaces.gr1py as gr1py
import tulip.interfaces.omega as omega_int
try:
    import tulip.interfaces.slugs as slugs
except ImportError:
    slugs = None
import tulip.spec as _spec
import tulip.transys as transys


__all__ = [
    'mutex',
    'exactly_one',
    'sys_to_spec',
    'env_to_spec',
    'build_dependent_var_table',
    'synthesize_many',
    'synthesize',
    'is_realizable',
    'strategy2mealy',
    'mask_outputs',
    'determinize_machine_init']


_logger = logging.getLogger(__name__)
_hl = '\n' + 60 * '-'


Formulas = list[str]
Solver = _ty.Literal[
    'gr1c',
    'gr1py',
    'omega',
    'slugs',
    ]
maybe_mealy = (
    transys.MealyMachine |
    None)
maybe_fts = (
    transys.FTS |
    None)


def _pstr(s) -> str:
    return f'({s})'


def _disj(set0) -> str:
    return ' || '.join(map(
        _pstr, set0))


def _conj(set0) -> str:
    nonempty = filter(None, set0)
    return ' && '.join(map(
        _pstr, nonempty))


def _conj_intersection(
        set0,
        set1,
        parenth:
            bool=True
        ) -> str:
    conjuncts = filter(
        set1.__contains__,
        set0)
    if parenth:
        conjuncts = map(
            _pstr, conjuncts)
    return ' && '.join(conjuncts)


def _conj_neg(
        set0,
        parenth:
            bool=True
        ) -> str:
    if parenth:
        return ' && '.join([
            f'!({x})'
            for x in set0])
    else:
        return ' && '.join([
            f'!{x}'
            for x in set0])


def _conj_neg_diff(
        set0,
        set1,
        parenth:
            bool=True
        ) -> str:
    items = filter(
        lambda x:
            x not in set1,
        set0)
    if parenth:
        items = map(_pstr, items)
    return ' && '.join(
        f'!{x}'
        for x in items)


def mutex(iterable) -> list[str]:
    """Mutual exclusion for all time."""
    iterable = list(filter(None, iterable))
    if not iterable or len(iterable) <= 1:
        return list()
    return [_conj([
        f'!({x}) || ({_conj_neg_diff(iterable, [x])})'
        for x in iterable])]


def exactly_one(iterable) -> list[str]:
    """N-ary xor.

    Contrast with pure mutual exclusion.
    """
    if len(iterable) <= 1:
        return list(map(
            _pstr, iterable))
    def conjoin(x):
        return _conj_neg_diff(iterable, [x])
    def formula(x):
        return f'({x}) && {conjoin(x)}'
    disjunction = _disj(map(formula, iterable))
    return [f'({disjunction})']


def _conj_action(
        actions_dict:
            dict,
        action_type:
            _abc.Hashable,
        nxt:
            bool=False,
        ids:
            dict |
            None=None
        ) -> str:
    """Return conjunct if `action_type` in `actions_dict`.

    @param actions_dict:
        `dict` with pairs `action_type_name : action_value`
    @param action_type:
        key to look for in `actions_dict`
        (here typically a str)
    @param nxt:
        prepend or not with the next operator
    @param ids:
        map `action_value` -> value used in solver input,
        for example, for gr1c
    @return:
        - conjunct (includes `&&` operator) if:

            - `action_type` in `actions_dict`, and
            - `action_value` is not
              the empty string (modeling "no constrain")

          includes next operator (`X`) if `nxt = True`.
        - empty string otherwise
    """
    if action_type not in actions_dict:
        return ''
    action = actions_dict[action_type]
    if ids is not None:
        action = ids[action]
    if action == '':
        return ''
    if nxt:
        return f' X{_pstr(action)}'
    return _pstr(action)


def _conj_actions(
        actions_dict:
            dict,
        solver_expr:
            dict |
            None=None,
        nxt:
            bool=False
        ) -> str:
    """Conjunction of multiple action types.

    Includes solver expression substitution.
    See also `_conj_action`.
    """
    _logger.debug(
        f'conjunction of actions: {actions_dict}')
    _logger.debug(
        'mapping to solver equivalents: '
        f'{solver_expr}')
    if not actions_dict:
        _logger.debug('actions_dict empty, returning empty string\n')
        return ''
    if solver_expr is not None:
        actions = [
            solver_expr[
                type_name][
                action_value]
            for type_name, action_value in
                actions_dict.items()]
    else:
        actions = actions_dict
    _logger.debug(
        f'after substitution: {actions}')
    conjuncted_actions = _conj(actions)
    _logger.debug(
        'conjuncted actions: '
        f'{conjuncted_actions}\n')
    if nxt:
        return f' X{_pstr(conjuncted_actions)}'
    return _pstr(conjuncted_actions)

# duplicate states are impossible,
# because each networkx vertex is unique
# non-contiguous integers for states work too,
# though a less efficient representation.


def iter2var(
        states:
            _abc.Iterable[int | str],
        variables:
            dict,
        statevar:
            str,
        bool_states:
            bool,
        must:
            _ty.Literal[
                'mutex',
                'xor',
                None]
        ) -> tuple[
            dict,
            list[str] | None]:
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
    They are treated by `spec` as an arbitrary finite domain.

    An option `min_one` is internally available,
    in order to allow only N values of the variable.
    This requires that the variable takes at least one value each time.

    Combined with a mutex constraint, it yields an n-ary xor constraint.

    @param states:
        values of domain.
    @param variables:
        to be augmented with
        integer or
        string variable or
        Boolean variables.
    @param statevar:
        name to use for integer or string valued variable.
    @param bool_states:
        if True, then use bool variables.
        Otherwise use integer or string valued variable.
    @return:
        `tuple` of:
        - mapping from values to GR(1) actions.
          If Booleans are used, then GR(1) are the same.
          Otherwise, they map to e.g. 'act = "wait"' or 'act = 3'
        - constraints to be added to `trans` and/or `init` in GR(1)
    """
    if not states:
        _logger.debug('empty container, so empty dict for solver expr')
        return dict(), None
    _logger.debug(
        f'mapping domain: {states}\n'
        '\tto expression understood by '
        'a GR(1) solver.')
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
        _logger.debug(
            'not using mutex: '
            'Booleans must model actions')
        bool_states = True
    _logger.debug(
        'options for modeling actions:\n\t'
        f'mutex: {use_mutex}\n'
        f'\tmin_one: {min_one}')
    all_str = all(
        isinstance(x, str)
        for x in states)
    if bool_states:
        _logger.debug(
            'states modeled as Boolean variables')
        if not all_str:
            raise TypeError(
                'If Boolean, all states must be strings.')
        state_ids = {x: x for x in states}
        variables.update({
            s: 'boolean'
            for s in states})
        # single action ?
        if len(mutex(state_ids.values())) == 0:
            return state_ids, None
        # handle multiple actions
        if use_mutex and not min_one:
            constraint = mutex(state_ids.values())[0]
        elif use_mutex and min_one:
            constraint = exactly_one(state_ids.values())[0]
        elif min_one:
            raise Exception('min_one requires mutex')
        else:
            constraint = 'True'
        constraint = [constraint]
    else:
        _logger.debug('states not modeled as Booleans')
        if statevar in variables:
            raise ValueError(
                f'state variable: {statevar}'
                f' already exists in: {variables}')
        all_int = all(isinstance(x, int) for x in states)
        if all_int:
            _logger.debug('all states are integers')
            # extra value modeling all False ?
            if min_one:
                n = max(states)
            else:
                n = max(states) + 1
            f = lambda x: f'{statevar} = {x}'
            domain = (min(states), n)
            _logger.debug(
                f'created solver variable: {statevar}'
                f'\n\t with domain: {domain}')
        elif all_str:
            _logger.debug('all states are strings')
            assert use_mutex
            f = lambda x: f'{statevar} = "{x}"'
            domain = list(states)
            if not min_one:
                domain += [f'{statevar}none']
                _logger.debug(
                    'domain has been extended, because all actions\n\t'
                    'could be False (constraint: min_one = False).')
        else:
            raise TypeError(
                'Integer and string states must not be mixed.')
        state_ids = {
            x: f(x)
            for x in states}
        variables[statevar] = domain
        constraint = None
    tabs = 2 * '\t'
    _logger.debug(
        f'for tulip variable: {statevar}\n'
        'the map from [tulip action values] ---> '
        '[solver expressions] is:\n'
        f'{tabs}{state_ids}')
    return state_ids, constraint


def _add_actions(
        constraint:
            list[str],
        init:
            list,
        trans:
            list):
    if constraint is None:
        return
    trans += [f'X ({constraint[0]})']
    init += constraint


def _fts2spec(
        fts:
            transys.FiniteTransitionSystem,
        ignore_initial:
            bool,
        statevar:
            str,
        actionvar:
            str |
            None=None,
        bool_states:
            bool=False,
        bool_actions:
            bool=False
        ) -> tuple:
    """Convert closed FTS to GR(1) representation."""
    raise Exception('deprecated')
    assert isinstance(fts, transys.FiniteTransitionSystem)
    aps = fts.aps
    states = fts.states
    actions = fts.actions
    sys_init = list()
    sys_trans = list()
    sys_vars = {
        ap: 'boolean'
        for ap in aps}
    action_ids, constraint = iter2var(
        actions, sys_vars,
        actionvar, bool_actions,
        fts.actions_must)
    _add_actions(constraint, sys_init, sys_trans)
    state_ids, constraint = iter2var(
        states, sys_vars,
        statevar, bool_states,
        must='xor')
    if constraint is not None:
        sys_trans += constraint
    sys_init += _sys_init_from_ts(
        states, state_ids, aps, ignore_initial)
    sys_trans += _sys_trans_from_ts(
        states, state_ids, fts.transitions,
        action_ids=action_ids)
    tmp_init, tmp_trans = _ap_trans_from_ts(
        states, state_ids, aps)
    sys_init += tmp_init
    sys_trans += tmp_trans
    return (
        sys_vars,
        sys_init,
        sys_trans)


def sys_to_spec(
        ofts:
            transys.FiniteTransitionSystem,
        ignore_initial:
            bool,
        statevar:
            str,
        bool_states:
            bool=False,
        bool_actions=False
        ) -> _spec.GRSpec:
    """Convert transition system to GR(1) fragment of LTL.

    The attribute `FTS.owner` defines who controls the system,
    as described next. It can take values `'env'` or `'sys'`.

    The following are represented by
    variables controlled by `ofts.owner`:

      - the current state
      - the atomic propositions annotating states
      - the system actions annotating edges

    The following are represented by
    variables controlled by the other player:

      - the environment actions annotating edges

    Multiple types of environment and system actions can be defined.
    Make sure that, depending on the player,
    `'env'` or `'sys'` are part of the action type names,
    so that `synth.synthesize` can recognize them.

    Caution
    =======
    There are aspects of `FTS` that
    need to be separately specified in a logic formula.

    An example are the initial conditions constraining the values
    of environment and system actions.

    See also
    ========
    `sys_trans_from_ts}, `env_open_fts2spec`,
    `create_actions`, `create_states`

    @param ignore_initial:
        Do not include initial state info from TS.
        Enable this to mask absence of FTS initial states.
        Useful when initial states are specified in another way,
        e.g., directly augmenting the spec part.
    @param state_var:
        name to be used for the integer or string
        variable that equals the current transition system state.
    @param bool_states:
        deprecated as inefficient

        if `True`, then use one Boolean variable
        to represent each state (one-hot encoding).
        Otherwise use a single integer variable,
        different values of which correspond to states of
        `ofts` (binary encoding).
    @param bool_actions:
        Similar to `bool_states`.
        For each type of system actions,
        and each type of environment actions:

          - if `True`, then for each possible value of that action type,
            use a different Boolean variable to represent it.

          - Otherwise use a single integer variable,
            that ranges over the possible action values.
    @return:
        logic formula in GR(1) form representing `ofts`.
    """
    if not isinstance(ofts, transys.FiniteTransitionSystem):
        raise TypeError(
            'ofts must be FTS, '
            f'got instead: {type(ofts)}')
    assert ofts.owner == 'sys'
    aps = ofts.aps
    states = ofts.states
    trans = ofts.transitions
    assert not set(aps).intersection(states)
    # init
    sys_init = list()
    sys_trans = list()
    env_init = list()
    env_trans = list()
    sys_vars = {
        ap: 'boolean'
        for ap in aps}
    env_vars = dict()
    actions = ofts.actions
    sys_action_ids = dict()
    env_action_ids = dict()
    for action_type, codomain in actions.items():
        if set(codomain).intersection(aps):
            raise AssertionError(
                codomain, aps)
        if set(codomain).intersection(states):
            raise AssertionError(
                codomain, states)
        if action_type in states:
            raise AssertionError(
                action_type)
        if action_type in aps:
            raise AssertionError(
                action_type)
        msg = f'action_type:\n\t{action_type}\n'
        msg += f'with codomain:\n\t{codomain}'
        _logger.debug(msg)
        if 'sys' in action_type:
            _logger.debug('Found sys action')
            action_ids, constraint = iter2var(
                codomain, sys_vars,
                action_type, bool_actions,
                ofts.sys_actions_must)
            _add_actions(constraint, sys_init, sys_trans)
            _logger.debug(
                'Updating sys_action_ids with:\n'
                f'\t{action_ids}')
            sys_action_ids[action_type] = action_ids
        elif 'env' in action_type:
            _logger.debug('Found env action')
            action_ids, constraint = iter2var(
                codomain, env_vars,
                action_type, bool_actions,
                ofts.env_actions_must)
            _add_actions(constraint, env_init, env_trans)
            _logger.debug(
                'Updating env_action_ids with:\n'
                f'\t{action_ids}')
            env_action_ids[action_type] = action_ids
    state_ids, constraint = iter2var(
        states, sys_vars,
        statevar, bool_states,
        must='xor')
    if constraint is not None:
        sys_trans += constraint
    sys_init += _sys_init_from_ts(
        states, state_ids, aps, ignore_initial)
    sys_trans += _sys_trans_from_ts(
        states, state_ids, trans,
        sys_action_ids=sys_action_ids,
        env_action_ids=env_action_ids)
    tmp_init, tmp_trans = _ap_trans_from_ts(
        states, state_ids, aps)
    sys_init += tmp_init
    sys_trans += tmp_trans
    env_trans += _env_trans_from_sys_ts(
        states, state_ids, trans, env_action_ids)
    return _spec.GRSpec(
        sys_vars=sys_vars,
        env_vars=env_vars,
        env_init=env_init,
        sys_init=sys_init,
        env_safety=env_trans,
        sys_safety=sys_trans)


def env_to_spec(
        ofts:
            transys.FiniteTransitionSystem,
        ignore_initial:
            bool,
        statevar:
            str,
        bool_states:
            bool=False,
        bool_actions:
            bool=False
        ) -> _spec.GRSpec:
    """Convert env transition system to GR(1) representation.

    The following are represented by environment variables:

      - the current state
      - the atomic propositions annotating states
      - the environment actions annotating edges

    The following are represented by system variables:

      - the system actions annotating edges

    Multiple types of environment and system actions can be defined.

    For more details see `sys_to_spec`.

    See also
    ========
    `sys_open_fts2spec`
    """
    if not isinstance(ofts, transys.FiniteTransitionSystem):
        raise TypeError(
            'ofts must be FTS, '
            f'got instead: {type(ofts)}')
    assert ofts.owner == 'env'
    aps = ofts.aps
    states = ofts.states
    trans = ofts.transitions
    assert not set(aps).intersection(states)
    # init
    sys_init = list()
    sys_trans = list()
    env_init = list()
    env_trans = list()
    # since APs are tied to env states,
    # let them be env variables
    env_vars = {ap: 'boolean' for ap in aps}
    sys_vars = dict()
    actions = ofts.actions
    sys_action_ids = dict()
    env_action_ids = dict()
    for action_type, codomain in actions.items():
        if set(codomain).intersection(aps):
            raise AssertionError(
                codomain, aps)
        if set(codomain).intersection(states):
            raise AssertionError(
                codomain, states)
        if action_type in states:
            raise AssertionError(
                action_type)
        if action_type in aps:
            raise AssertionError(
                action_type)
        if 'sys' in action_type:
            action_ids, constraint = iter2var(
                codomain, sys_vars,
                action_type, bool_actions,
                ofts.sys_actions_must)
            _add_actions(
                constraint,
                sys_init,
                sys_trans)
            sys_action_ids[action_type] = action_ids
        elif 'env' in action_type:
            action_ids, constraint = iter2var(
                codomain, env_vars,
                action_type, bool_actions,
                ofts.env_actions_must)
            _add_actions(
                constraint,
                env_init,
                env_trans)
            env_action_ids[action_type] = action_ids
    # some duplication here,
    # because we don't know
    # whether the user will provide
    # a system TS as well
    # and whether that TS will contain
    # all the system actions
    # defined in the environment TS
    state_ids, constraint = iter2var(
        states, env_vars,
        statevar, bool_states,
        must='xor')
    if constraint is not None:
        env_trans += constraint
    env_init += _sys_init_from_ts(
        states, state_ids, aps, ignore_initial)
    env_trans += _env_trans_from_env_ts(
        states, state_ids, trans,
        env_action_ids=env_action_ids,
        sys_action_ids=sys_action_ids)
    tmp_init, tmp_trans = _ap_trans_from_ts(
        states, state_ids, aps)
    env_init += tmp_init
    env_trans += tmp_trans
    return _spec.GRSpec(
        sys_vars=sys_vars,
        env_vars=env_vars,
        env_init=env_init,
        sys_init=sys_init,
        env_safety=env_trans,
        sys_safety=sys_trans)


def _sys_init_from_ts(
        states,
        state_ids,
        aps,
        ignore_initial:
            bool=False
        ) -> Formulas:
    """Initial state, including enforcement of exactly one."""
    init = list()
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
        init.append('False')
        return init
    init.append(
        _disj(map(
            state_ids.__getitem__,
            states.initial)))
    return init


def _sys_trans_from_ts(
        states,
        state_ids,
        trans,
        action_ids=None,
        sys_action_ids=None,
        env_action_ids=None
        ) -> Formulas:
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

    @param trans:
        `Transitions` as from the transitions
        attribute of `FTS`.
    @param action_ids:
        same as `sys-action_ids`
        Caution: to be removed in a future release
    @param sys_action_ids:
        dict of dicts
        outer dict keyed by action_type
        each inner dict keyed by action_value
        each inner dict value is the
        solver expression for that action value

        for example an action type with an
        arbitrary finite discrete codomain can be modeled either:

          - as Boolean variables, so each possible action value
            becomes a different Boolean variable with the same
            name, thus `sys_action_ids[action_type]` will be
            the identity map on `action_values` for that `action_type`.

          - as integer variables, so each possible action value
            becomes a different expression in the solver (e.g. gr1c)
            input format. Then `sys_action_ids[action_type]` maps
            `action_value` -> solver expression of the form:

            `action_type = i`

            where `i` corresponds to that particular  `action_type`.
    @param env_action_ids:
        same as `sys-action_ids`
    """
    _logger.debug('modeling sys transitions in logic')
    sys_trans = list()
    # Transitions
    for from_state in states:
        from_state_id = state_ids[from_state]
        precond = _pstr(from_state_id)
        cur_trans = trans.find([from_state])
        msg = (
            f'from state: {from_state}'
            ', the available transitions are:\n'
            f'\t{cur_trans}')
        _logger.debug(msg)
        # no successor states ?
        if not cur_trans:
            _logger.debug(
                f'state: {from_state} is deadend !')
            sys_trans.append(
                f'{precond} -> X(False)')
            continue
        cur_str = list()
        for from_state, to_state, label in cur_trans:
            to_state_id = state_ids[to_state]
            postcond = [
                f'X{_pstr(to_state_id)}']
            _logger.debug(
                f'label = {label}')
            if 'previous' in label:
                previous = label['previous']
            else:
                previous = set()
            _logger.debug(
                f'previous = {previous}')
            env_actions = {
                k: v
                for k, v in
                    label.items()
                if 'env' in k}
            prev_env_act = {
                k: v
                for k, v in
                    env_actions.items()
                if k in previous}
            next_env_act = {
                k: v
                for k, v in
                    env_actions.items()
                if k not in previous}
            postcond += [_conj_actions(prev_env_act, env_action_ids,
                                       nxt=False)]
            postcond += [_conj_actions(next_env_act, env_action_ids,
                                       nxt=True)]
            sys_actions = {
                k: v
                for k, v in
                    label.items()
                if 'sys' in k}
            prev_sys_act = {
                k: v
                for k, v in
                    sys_actions.items()
                if k in previous}
            next_sys_act = {
                k: v
                for k, v in
                    sys_actions.items()
                if k not in previous}
            postcond += [
                _conj_actions(
                    prev_sys_act, sys_action_ids,
                    nxt=False)]
            postcond += [
                _conj_actions(
                    next_sys_act, sys_action_ids,
                    nxt=True)]
            # if system FTS given
            # in case 'actions in label,
            # then action_ids is a dict,
            # not a dict of dicts,
            # because certainly this came
            # from an FTS, not an OpenFTS
            if 'actions' in previous:
                postcond += [
                    _conj_action(
                        label, 'actions',
                        ids=action_ids,
                        nxt=False)]
            else:
                postcond += [
                    _conj_action(
                        label, 'actions',
                        ids=action_ids,
                        nxt=True)]
            cur_str += [_conj(postcond)]
            msg = (
                f'guard to state: {to_state}'
                f', with state_id: {to_state_id}'
                f', has post-conditions: {postcond}')
            _logger.debug(msg)
        sys_trans += [
            f'{precond} -> ({_disj(cur_str)})']
    return sys_trans


def _env_trans_from_sys_ts(
        states,
        state_ids,
        trans,
        env_action_ids
        ) -> Formulas:
    """Convert environment actions to GR(1) env_safety.

    This constrains the actions available next to the environment
    based on the system FTS.

    Purpose is to prevent env from blocking sys by purely
    picking a combination of actions for which sys has no outgoing
    transition from that state.

    Might become optional in the future,
    depending on the desired way of defining env behavior.

    @param env_action_ids:
        dict of dicts,
        see `sys_trans_from_ts`.
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
            # nothing modeled for env,
            # since sys has X(False) anyway
            # for action_type, codomain_map in env_action_ids.items():
            # env_trans += [f'{precond} -> X({s})']
            continue
        # collect possible next env actions
        next_env_action_combs = set()
        for from_state, to_state, label in cur_trans:
            env_actions = {
                k: v
                for k, v in
                    label.items()
                if 'env' in k}
            if not env_actions:
                continue
            _logger.debug(
                f'env_actions: {env_actions}')
            _logger.debug(
                f'env_action_ids: {env_action_ids}')
            env_action_comb = _conj_actions(
                env_actions, env_action_ids)
            _logger.debug(
                f'env_action_comb: {env_action_comb}')
            next_env_action_combs.add(env_action_comb)
        next_env_actions = _disj(next_env_action_combs)
        _logger.debug(
            f'next_env_actions: {next_env_actions}')
        # no next env actions ?
        if not next_env_actions:
            continue
        env_trans += [
            f'{precond} -> X({next_env_actions})']
    return env_trans


def _env_trans_from_env_ts(
        states,
        state_ids,
        trans,
        action_ids=None,
        env_action_ids=None,
        sys_action_ids=None
        ) -> Formulas:
    r"""Convert environment TS transitions to GR(1) representation.

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
            env_trans += [f'{precond} -> X(False)']
            msg = (
                'Environment dead-end found.\n'
                'If sys can force env to dead-end,\n'
                'then GR(1) assumption becomes False,\n'
                'and spec trivially True.')
            warnings.warn(msg)
            continue
        cur_list = list()
        found_free = False
            # any environment transition
        # not conditioned on the previous system output ?
        for from_state, to_state, label in cur_trans:
            to_state_id = state_ids[to_state]
            postcond = [f'X{_pstr(to_state_id)}']
            env_actions = {
                k: v
                for k, v in
                    label.items()
                if 'env' in k}
            postcond += [
                _conj_actions(
                    env_actions,
                    env_action_ids,
                    nxt=True)]
            # remember: this is
            # an environment FTS,
            # so no next for sys
            sys_actions = {
                k: v
                for k, v in
                    label.items()
                if 'sys' in k}
            postcond += [
                _conj_actions(
                    sys_actions,
                    sys_action_ids)]
            postcond += [
                _conj_action(
                    label, 'actions',
                    nxt=True,
                    ids=action_ids)]
            # todo: test this clause
            if not sys_actions:
                found_free = True
            cur_list += [_conj(postcond)]
        # can sys block env by
        # setting all previous sys outputs to False ?
        # then env assumption becomes False,
        # so the spec trivially True: avoid this
        if not found_free and sys_action_ids:
            msg = (
                'no free env outgoing transition found\n'
                'instead will take disjunction with negated sys actions')
            _logger.debug(msg)
            for action_type, codomain in sys_action_ids.items():
                conj = _conj_neg(codomain.values())
                cur_list += [conj]
                msg = (
                    f'for action_type: {action_type}\n'
                    f'with codomain: {codomain}\n'
                    f'the negated conjunction is: {conj}')
                _logger.debug(msg)
        env_trans += [
            f'{_pstr(precond)} -> '
            f'({_disj(cur_list)})']
    return env_trans


def _ap_trans_from_ts(
        states,
        state_ids,
        aps) -> tuple[
            Formulas,
            Formulas]:
    """Require atomic propositions to follow states according to label."""
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
        init += [
            f'!({_pstr(state_id)}) || ({ap_str})']
    # transitions of labels
    for state in states:
        label = states[state]
        state_id = state_ids[state]
        tmp = _sprint_aps(label, aps)
        if not tmp:
            continue
        trans += [
            f'X(({state_id}) -> ({tmp}))']
    return (init, trans)


def _sprint_aps(label, aps) -> str:
    if 'ap' in label:
        tmp0 = _conj_intersection(
            aps, label['ap'],
            parenth=False)
    else:
        tmp0 = ''
    if 'ap' in label:
        tmp1 = _conj_neg_diff(
            aps, label['ap'],
            parenth=False)
    else:
        tmp1 = _conj_neg(
            aps,
            parenth=False)
    if tmp0 and tmp1:
        tmp = f'{tmp0} && {tmp1}'
    else:
        tmp = f'{tmp0}{tmp1}'
    return tmp


def build_dependent_var_table(
        fts:
            transys.FTS,
        statevar:
            str
        ) -> dict[str, str]:
    """Return substitution rules for dependent variables.

    The dependent variables in a transition system are the
    atomic propositions that are used to label states.

    They are "dependent" because their values are completely
    determined by knowledge of the current state in the
    transition system.

    The returned substitutions can be used

    @param statevar:
        name of variable used for the current state
        For example if it is 'loc', then the states
        `'s0', 's1'` are mapped to:

        ```python
        {'s0': '(loc = "s0")',
         's1': '(loc = "s1")'}
        ```
    @return:
        `{'p': '((loc = "s1") | (loc = "s2") | ...)', ...}`
        where:

          - `'p'` is a proposition in
            `fts.atomic_propositions`
          - the states "s1", "s2" are labeled with `'p'`
          - `loc` is the string variable used for
            the state of `fts`.
    """
    state_ids, __ = iter2var(
        fts.states,
        variables=dict(),
        statevar=statevar,
        bool_states=False,
        must='xor')
    ap2states = map_ap_to_states(fts)
    return {k: _disj(state_ids[x] for x in v)
            for k, v in ap2states.items()}


def map_ap_to_states(
        fts:
            transys.FTS
        ) -> dict[str, set]:
    """For each proposition find the states labeled with it.

    @return:
        `{'p': s, ...}` where
        `'p'` a proposition and
        `s` a set of states in `fts`.
    """
    table = {
        p: set()
        for p in fts.atomic_propositions}
    for u in fts:
        for p in fts.nodes[u]['ap']:
            table[p].add(u)
    return table


def synthesize_many(
        specs:
            _spec.GRSpec,
        ts:
            dict[
                str,
                transys.FiniteTransitionSystem] |
            None=None,
        ignore_init:
            set |
            None=None,
        solver:
            Solver='omega'
        ) -> maybe_mealy:
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
    that ranges over a finite set of
    integers or strings, respectively.

    The keys of `ts` are used to name each state variable.
    So the logic formula for `ts['name']` will be `'name'`.

    Who controls this state variable is determined from
    the attribute `FTS.owner` that can take the values:

      - `'env'`
      - `'sys'`

    For example:

    ```python
    ts.states.add_from(range(4))
    ts['door'].owner = 'env'
    ```

    will result in a logic formula with
    an integer variable `'door'`
    controlled by the environment and
    taking values over `{0, 1, 2, 3}`.

    The example:

    ```python
    ts.states.add_from(['a', 'b', 'c'])
    ts['door'].owner = 'sys'
    ```

    will instead result in a string variable `'door'`
    controlled by the system and taking
    values over `{'a', 'b', 'c'}`.

    @param ignore_init:
        `set` of keys from `ts`
    @param solver:
        See function `synthesize` for
        available options.
    """
    assert isinstance(ts, dict), ts
    for name, t in ts.items():
        if not isinstance(t, transys.FiniteTransitionSystem):
            raise AssertionError(t)
        ignore = name in ignore_init
        statevar = name
        match t.owner:
            case 'sys':
                sys_spec = sys_to_spec(t, ignore, statevar)
                _copy_options_from_ts(sys_spec, t, specs)
                specs |= sys_spec
            case 'env':
                env_spec = env_to_spec(t, ignore, statevar)
                _copy_options_from_ts(env_spec, t, specs)
                specs |= env_spec
    return _synthesize(
        specs, solver,
        rm_deadends=True)


def synthesize(
        specs:
            _spec.GRSpec,
        env:
            maybe_fts=None,
        sys:
            maybe_fts=None,
        ignore_env_init:
            bool=False,
        ignore_sys_init:
            bool=False,
        rm_deadends:
            bool=True,
        solver:
            Solver='omega'
        ) -> maybe_mealy:
    """Call synthesis tool `solver` on the specification.

    There are three attributes of `specs` that define what
    kind of controller you are looking for:

    1. `moore`:
       What information the controller knows
       when deciding the next
       values of controlled variables:
        - Moore: can read current state,
          but not next environment variable values, or
        - Mealy: can read current state and
          next environment variable values.

    2. `qinit`:
        Quantification of initial variable values:
        Whether all states that satisfy
        a predicate should be winning,
        or the initial values of
        some (or all) the variables is
        subject to the synthesizer's choice.

    3. `plus_one`:
        The form of assume-guarantee specification,
        i.e., how the system guarantees relate to
        assumptions about the environment.

    For more details about these attributes, see `GRSpec`.

    The states of the transition system can be either:

      - all integers, or
      - all strings

    For more details of how
    the transition system is represented in
    logic look at `synthesize_many`.

    Beware!
    =======
    This function provides a generic interface to a variety
    of routines.  Being under active development, the types of
    arguments supported and types of objects returned may change
    without notice.

    @param env:
        A transition system describing the environment:

            - states controlled by environment
            - input: sys_actions
            - output: env_actions
            - initial states constrain the environment

        This constrains the transitions available to
        the environment, given the outputs from the system.
    @param sys:
        A transition system describing the system:

            - states controlled by the system
            - input: env_actions
            - output: sys_actions
            - initial states constrain the system
    @param ignore_env_init:
        Ignore any initial state information
        contained in env.
    @param ignore_sys_init:
        Ignore any initial state information
        contained in sys.
    @param rm_deadends:
        return a strategy that contains no terminal states.
    @param solver:
        Magic string that declares what tool to invoke,
        what method to use, etc.  Currently recognized forms:

        For GR(1) synthesis:

          - `"gr1c"`:
            use gr1c via `interfaces.gr1c`.
            written in C using CUDD, symbolic

          - `"gr1py"`:
            use gr1py via `interfaces.gr1py`.
            Python, enumerative

          - `"omega"`:
            use omega via `interfaces.omega`.
            Python using `dd` or Cython using CUDD, symbolic

          - `"slugs"`:
            use slugs via `interfaces.slugs`.
            C++ using CUDD, symbolic
    @return:
        If spec is realizable,
        then return a Mealy machine implementing the strategy.
        Otherwise return None.
    """
    specs = _spec_plus_sys(
        specs, env, sys,
        ignore_env_init,
        ignore_sys_init)
    return _synthesize(specs, solver, rm_deadends)


def _synthesize(
        specs:
            _spec.GRSpec,
        solver:
            Solver,
        rm_deadends:
            bool
        ) -> maybe_mealy:
    """Return `MealyMachine` that implements `specs`."""
    match solver:
        case 'gr1c':
            strategy = gr1c.synthesize(specs)
        case 'slugs':
            if slugs is None:
                raise ValueError(
                    'Import of slugs interface failed. '
                    'Please verify installation of "slugs".')
            strategy = slugs.synthesize(specs)
        case 'gr1py':
            strategy = gr1py.synthesize(specs)
        case 'omega':
            strategy = omega_int.synthesize_enumerated_streett(specs)
        case _:
            options = {'gr1c', 'gr1py', 'omega', 'slugs'}
            raise ValueError(
                f'Unknown solver: "{solver}". '
                f'Available options are: {options}')
    return _trim_strategy(
        strategy, specs,
        rm_deadends=rm_deadends)


def _trim_strategy(
        strategy:
            maybe_mealy,
        specs:
            _spec.GRSpec,
        rm_deadends:
            bool
        ) -> maybe_mealy:
    """Return `MealyMachine` without deadends, or `None`.

    If `strategy is None`, then return `None`.

    @param rm_deadends:
        if `True`, then remove deadends
        from the Mealy machine
    """
    # While the return values of
    # the solver interfaces vary,
    # we expect here that strategy is
    # either None to indicate unrealizable or
    # a networkx.DiGraph ready to be
    # passed to strategy2mealy().
    if strategy is None:
        return None
    ctrl = strategy2mealy(strategy, specs)
    _logger.debug(
        'Mealy machine has: '
        f'n = {len(ctrl.states)} states.')
    if rm_deadends:
        ctrl.remove_deadends()
    return ctrl


def is_realizable(
        specs:
            _spec.GRSpec,
        env:
            maybe_fts=None,
        sys:
            maybe_fts=None,
        ignore_env_init:
            bool=False,
        ignore_sys_init:
            bool=False,
        solver:
            Solver='omega'
        ) -> bool:
    """Check realizability.

    For details see `synthesize`.
    """
    specs = _spec_plus_sys(
        specs, env, sys,
        ignore_env_init, ignore_sys_init)
    match solver:
        case 'gr1c':
            r = gr1c.check_realizable(specs)
        case 'slugs':
            if slugs is None:
                raise ValueError(
                    'Import of slugs interface failed. '
                    'Please verify installation of "slugs".')
            r = slugs.check_realizable(specs)
        case 'gr1py':
            r = gr1py.check_realizable(specs)
        case 'omega':
            r = omega_int.is_realizable(specs)
        case _:
            raise ValueError(
                'Undefined synthesis solver: '
                f'{solver = }'
                'Available options are "gr1c", '
                '"slugs", and "gr1py"')
    if r:
        _logger.debug('is realizable')
    else:
        _logger.debug('is not realizable')
    return r


def _spec_plus_sys(
        specs:
            _spec.GRSpec,
        env:
            maybe_fts,
        sys:
            maybe_fts,
        ignore_env_init:
            bool,
        ignore_sys_init:
            bool
        ) -> _spec.GRSpec:
    if sys is not None:
        if hasattr(sys, 'state_varname'):
            statevar = sys.state_varname
        else:
            _logger.info(
                'sys.state_varname undefined. '
                'Will use the default '
                'variable name: "loc".')
            statevar = 'loc'
        sys_formula = sys_to_spec(
            sys, ignore_sys_init,
            statevar=statevar)
        _copy_options_from_ts(
            sys_formula, sys, specs)
        if specs.qinit == r'\A \A':
            sys_formula.env_init.extend(
                sys_formula.sys_init)
            sys_formula.sys_init = list()
        specs = specs | sys_formula
        pp_formula = sys_formula.pretty()
        _logger.debug(
            'sys TS:\n'
            f'{pp_formula}{_hl}')
    if env is not None:
        if hasattr(env, 'state_varname'):
            statevar = env.state_varname
        else:
            _logger.info(
                'env.state_varname undefined. '
                'Will use the default '
                'variable name: "eloc".')
            statevar = 'eloc'
        env_formula = env_to_spec(
            env, ignore_env_init,
            statevar=statevar)
        if specs.qinit == r'\A \A':
            env_formula.env_init.extend(
                env_formula.sys_init)
            env_formula.sys_init = list()
        _copy_options_from_ts(env_formula, env, specs)
        specs = specs | env_formula
        _logger.debug(
            'env TS:\n'
            f'{env_formula.pretty()}{_hl}')
    _logger.info(
        'Overall Spec:\n'
        f'{specs.pretty()}{_hl}')
    return specs


def _copy_options_from_ts(
        ts_spec,
        ts,
        specs
        ) -> None:
    """Copy `moore, qinit, plus_one` from `ts`, if set.

    Otherwise copy the values of those attributes from `specs`.
    """
    if hasattr(ts, 'moore'):
        cp = ts
    else:
        cp = specs
    ts_spec.moore = cp.moore
    ts_spec.plus_one = cp.plus_one
    ts_spec.qinit = cp.qinit


def strategy2mealy(
        A:
            _nx.DiGraph,
        spec:
            _spec.GRSpec
        ) -> transys.MealyMachine:
    """Convert strategy to Mealy transducer.

    Note that the strategy is a deterministic game graph,
    but the input `A` is given as the contraction of
    this game graph.

    @param A:
        strategy
    """
    if not A:
        raise AssertionError(
            'graph `A` has no nodes, '
            f'{A = }')
    _logger.info(
        'converting strategy (compact) '
        'to Mealy machine')
    env_vars = spec.env_vars
    sys_vars = spec.sys_vars
    mach = transys.MealyMachine()
    inputs = transys.machines.create_machine_ports(env_vars)
    mach.add_inputs(inputs)
    outputs = transys.machines.create_machine_ports(sys_vars)
    mach.add_outputs(outputs)
    str_vars = {
        k: v
        for k, v in
            env_vars.items()
        if isinstance(v, list)}
    str_vars.update({
        k: v
        for k, v in
            sys_vars.items()
        if isinstance(v, list)})
    mach.states.add_from(A)
    all_vars = dict(env_vars)
    all_vars.update(sys_vars)
    u = next(iter(A))
    strategy_vars = A.nodes[u]['state'].keys()
    assert set(all_vars).issubset(strategy_vars)
    # transitions labeled with I/O
    for u in A:
        for v in A.successors(u):
            d = A.nodes[v]['state']
            d = {
                k: v
                for k, v in
                    d.items()
                if k in all_vars}
            d = _int2str(d, str_vars)
            mach.transitions.add(
                u, v,
                attr_dict=None,
                check=False,
                **d)
            _logger.info(
                f'node: {v}, state: {d}')
    # special initial state, for first reaction
    initial_state = 'Sinit'
    mach.states.add(initial_state)
    mach.states.initial.add(initial_state)
    # fix an ordering for keys
    keys = list(all_vars)
    if hasattr(A, 'initial_nodes'):
        _init_edges_using_initial_nodes(
            A, mach, keys,
            all_vars, str_vars,
            initial_state)
    else:
        _init_edges_using_compile_init(
            spec, A, mach, keys,
            all_vars, str_vars,
            initial_state)
    n = len(A)
    m = len(mach)
    if m != n + 1:
        raise AssertionError(
            n, m)
    if not mach.successors('Sinit'):
        newlines = 2 * '\n'
        nodes = pprint.pformat(
            A.nodes(data=True))
        raise AssertionError(
            'The machine obtained from the strategy '
            'does not have any initial states !\n'
            'The strategy is:\n'
            f'vertices:{nodes}{newlines}'
            f'edges:\n{A.edges()}{newlines}'
            f'and the machine:\n{mach}{newlines}'
            'and the specification is:\n'
            f'{spec.pretty()}{newlines}')
    return mach


def _init_edges_using_initial_nodes(
        A:
            _nx.DiGraph,
        mach:
            transys.MealyMachine,
        keys,
        all_vars,
        str_vars:
            dict,
        initial_state
        ) -> None:
    assert A.initial_nodes
    init_valuations = set()
    for u in A.initial_nodes:
        d = A.nodes[u]['state']
        vals = tuple(d[k] for k in keys)
        # already an initial valuation ?
        if vals in init_valuations:
            continue
        init_valuations.add(vals)
        d = {
            k: v
            for k, v in
                d.items()
            if k in all_vars}
        d = _int2str(d, str_vars)
        mach.transitions.add(
            initial_state, u,
            attr_dict=None,
            **d)


def _init_edges_using_compile_init(
        spec:
            _spec.GRSpec,
        A:
            _nx.DiGraph,
        mach:
            transys.MealyMachine,
        keys,
        all_vars,
        str_vars:
            dict,
        initial_state
        ) -> None:
    init_valuations = set()
    # to store tuples of
    # dict values for fast search
    isinit = spec.compile_init(no_str=True)
    # Mealy reaction to initial env input
    init_valuations = set()
    tmp = dict()
    for u, d in A.nodes(data=True):
        var_values = d['state']
        vals = tuple(var_values[k] for k in keys)
        # already an initial valuation ?
        if vals in init_valuations:
            continue
        # add edge: Sinit -> u ?
        tmp.update(var_values)
        if eval(isinit, tmp):
            var_values = {
                k: v
                for k, v in
                    var_values.items()
                if k in all_vars}
            label = _int2str(var_values, str_vars)
            mach.transitions.add(
                initial_state, u,
                attr_dict=None,
                check=False,
                **label)
            # remember variable values to avoid
            # spurious non-determinism wrt
            # the machine's memory
            #
            # in other words,
            # "state" omits the strategy's memory
            # hidden (existentially quantified)
            # so multiple nodes can be
            # labeled with the same state
            #
            # non-uniqueness here would be
            # equivalent to multiple choices for
            # initializing the hidden memory.
            init_valuations.add(vals)
            _logger.debug(
                f'found initial state: {u}')
        _logger.debug(
            f'machine vertex: {u}, '
            f'has var values: {var_values}')


def _int2str(
        label:
            dict,
        str_vars:
            dict
        ) -> dict:
    """Replace integers with string values for string variables.

    @param label:
        mapping from variable names, to integer (as strings)
    @param str_vars:
        mapping that defines those variables that
        should be converted from integer to string variables.
        Each variable is mapped to a list of strings that
        comprise its range. This list defines how integer values
        correspond to string literals for that variable.
    """
    label = dict(label)
    label.update({
        k:
            str_vars[k][int(v)]
        for k, v in
            label.items()
        if k in str_vars})
    return label


def mask_outputs(
        machine:
            transys.MealyMachine
        ) -> None:
    """Erase outputs from each edge where they are zero."""
    for _, _, d in machine.edges(data=True):
        for k in d:
            pop = (
                k in machine.outputs and
                d[k] == 0)
            if pop:
                d.pop(k)


def determinize_machine_init(
        mach:
            transys.MealyMachine,
        init_out_values:
            dict |
            None=None
        ) -> transys.MealyMachine:
    """Return a determinized copy of `mach` with given initial outputs.

    The transducers produced by synthesis can have multiple
    initial output valuations as possible reactions to a
    given input valuation.

    Possible reasons for this are:

      1. the system does not have full control over its initial state.
        For example the option "ALL_INIT" of `gr1c`.

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
    The function `strategy2mealy` returns a transducer that
    for each initial input valuation,
    for each initial output valuation,
    reacts with a unique transition.

    But this can yield multile reactions to a single input,
    even for solver options like "ALL_ENV_EXIST_SYS_INIT" for `gr1c`.
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
    by ensuring the outputs given in `init_out_values`
    take those values.
    The remaining outputs are determinized arbitrarily.

    See also
    ========
    `synthesize`, `strategy2mealy`

    @param mach:
        possibly non-deterministic transducer,
        as produced, for example, by `synthesize`.
    @param init_out_values:
        mapping from output ports that
        the system cannot control initially,
        to the initial values they take in
        this instance of the game.
    """
    mach = copy.deepcopy(mach)
    if init_out_values is None:
        init_out_values = dict()
    '''determinize given outputs (uncontrolled)'''
    # restrict attention to
    # given output ports only
    given_ports = tuple(filter(
        init_out_values.__contains__,
        mach.outputs))
    rm_edges = set()
    edges = mach.edges(
        ['Sinit'],
        data=True,
        keys=True)
    for i, j, key, d in edges:
        for k in given_ports:
            if d[k] != init_out_values[k]:
                rm_edges.add((i, j, key))
                break
    mach.remove_edges_from(rm_edges)
    #
    # determinize arbitrarily any
    # remnant non-determinism
    #
    # input valuations already seen
    # tuples of values used for
    # efficiency (have __hash__)
    possible_inputs = set()
    # fix a key order
    inputs = tuple(mach.inputs)
    rm_edges = set()
    edges = mach.edges(
        ['Sinit'],
        data=True,
        keys=True)
    for i, j, key, d in edges:
        in_values = tuple(d[k] for k in inputs)
        # newly encountered input valuation ?
        if in_values not in possible_inputs:
            possible_inputs.add(in_values)
            continue
        else:
            edge = (i, j, key)
            rm_edges.add(edge)
    mach.remove_edges_from(rm_edges)
    return mach

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
from copy import deepcopy
from warnings import warn

from tulip import transys
from tulip.spec import GRSpec
from tulip import jtlvint
from tulip import gr1cint

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

def sys_to_spec(sys, ignore_initial=False, bool_states=False):
    """Convert finite transition system to GR(1) representation.
    
    The term GR(1) representation is preferred to GR(1) spec,
    because an FTS can represent sys_init, sys_safety, but
    not other spec forms.
    
    @type sys: transys.FiniteTransitionSystem |
               transys.OpenFiniteTransitionSystem
    
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
        return fts2spec(sys, ignore_initial, bool_states)
    elif isinstance(sys, transys.OpenFiniteTransitionSystem):
        return open_fts2spec(sys, ignore_initial, bool_states)
    else:
        raise TypeError('synth.sys_to_spec does not support ' +
            str(type(sys)) +'. Use FTS or OpenFTS.')

def states2bools(states):
    """Return dict(state : state).
    
    @rtype: {state : state_id}
    """
    return {x:x for x in states}

def states2ints(states):
    """Return states of form 'loc = #'.
    
    where # is obtained by dropping the 1st char
    of each given state.
    
    @type states: iterable of str,
        each str of the form: letter + number
    
    @rtype: {state : state_id}
    """
    strip_letter = lambda x: 'loc = ' + x[1:]
    return {x:strip_letter(x) for x in states}

def fts2spec(fts, ignore_initial=False, bool_states=False):
    """Convert closed FTS to GR(1) representation.
    
    So fts on its own is not the complete problem spec.
    
    @param fts: transys.FiniteTransitionSystem
    
    @rtype: GRSpec
    """
    assert(isinstance(fts, transys.FiniteTransitionSystem))
    
    aps = fts.aps
    states = fts.states
    
    sys_vars = list(aps)
    sys_vars.extend([a for a in fts.actions])
    
    trans = []
    
    if bool_states:
        state_ids = states2bools(states)
        sys_vars.extend([s for s in states])
        trans += sys_state_mutex(states)
    else:
        state_ids = states2ints(states)
        n_states = len(states)
        sys_vars += ['loc [0,' +str(n_states-1) +']']
    
    init = sys_init_from_ts(states, state_ids, aps, ignore_initial)
    
    trans += sys_trans_from_ts(states, state_ids, fts.transitions)
    trans += ap_trans_from_ts(states, state_ids, aps)
    
    trans += pure_mutex(fts.actions)
    
    return GRSpec(sys_vars=sys_vars, sys_init=init, sys_safety=trans)

def open_fts2spec(ofts, ignore_initial=False, bool_states=False):
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
    
    sys_vars = list(aps)
    sys_vars.extend([a for a in ofts.sys_actions])
    
    sys_trans = []    
    
    if bool_states:
        state_ids = states2bools(states)
        sys_vars.extend([s for s in states])
        sys_trans += sys_state_mutex(states)
    else:
        state_ids = states2ints(states)
        n_states = len(states)
        sys_vars += ['loc [0,' +str(n_states-1) +']']
    
    env_vars = list(ofts.env_actions)
    
    sys_init = sys_init_from_ts(states, state_ids, aps, ignore_initial)
    
    sys_trans += sys_trans_from_open_ts(states, state_ids, trans, env_vars)
    sys_trans += ap_trans_from_ts(states, state_ids, aps)
    sys_trans += pure_mutex(ofts.sys_actions)
    
    env_trans = env_trans_from_open_ts(states, state_ids, trans, env_vars)
    env_trans += pure_mutex(env_vars)
    
    return GRSpec(
        sys_vars=sys_vars, env_vars=env_vars,
        sys_init=sys_init,
        env_safety=env_trans,
        sys_safety=sys_trans
    )

def sys_init_from_ts(states, state_ids, aps, ignore_initial=False):
    """Initial state, including enforcement of exactly one.
    
    APs also considered for the initial state.
    """
    init = []
    
    # don't ignore labeling info
    for state in states.initial:
        label = states.label_of(state)
        
        ap_init = _conj_intersection(aps, label["ap"])
        init.append(ap_init)
        
        if len(init[-1]) > 0:
            init[-1] += " && "
        
        init[-1] += _conj_neg_diff(aps, label["ap"])
        
        state_id = state_ids[state]
        init[-1] = "!("+str(state_id)+") || ("+init[-1]+")"
    
    # skip ?
    if ignore_initial:
        return init
    
    if not states.initial:
        msg = 'FTS has no initial states.\n'
        msg += 'Enforcing this renders False the GR(1) guarantee.'
        warn(msg)
        
        init += [_conj_neg(states.intervalues() ) ]
        return init
        
    init += [_disj([state_ids[s] for s in states.initial])]
    return init

def sys_trans_from_ts(states, state_ids, trans):
    """Convert environment actions to GR(1) representation.
    """
    sys_trans = []
    
    for from_state in states:
        post = states.post(from_state)
        from_state_id = state_ids[from_state]
        
        # no successor states ?
        if not post:
            sys_trans += ['('+str(from_state_id) +') -> X('+ _conj_neg(state_ids.itervalues() ) +')']
            continue
        
        cur_trans = trans.find([from_state])
        cur_str = []
        for (from_state, to_state, label) in cur_trans:
            precond = '(' + str(from_state_id) + ')'
            
            to_state_id = state_ids[to_state]
            postcond = '(' + str(to_state_id) + ')'
            
            if 'actions' in label:
                sys_action = label['actions']
                postcond += ' && (' +str(sys_action) +')'
            
            cur_str += ['(' + precond + ') -> X( ' + postcond + ')']
        
        sys_trans += [_disj(cur_str) ]
    return sys_trans

def sys_state_mutex(states):
    """Require next exactly one.
    
    Combined with [] requires exactly one
    for all time except initially.
    
    Contrast with the pure mutual exclusion implemented by:
        spec.form.mutex
    """
    return ['(' + exactly_one(states) + ')']

def pure_mutex(iterable):
    """Mutual exclusion for all time.
    """
    if not iterable:
        return []
    
    return [_conj([
        '(' + str(x) + ') -> (' + _conj_neg_diff(iterable, [x]) +')'
        for x in iterable
    ]) ]

def exactly_one(iterable):
    """N-ary xor.
    """
    return _disj([
        '(' +str(x) + ') && ' + _conj_neg_diff(iterable, [x])
        for x in iterable
    ])

def sys_trans_from_open_ts(states, state_ids, trans, env_vars):
    """Convert sys transitions and env actions to GR(1) sys_safety.
    
    Mutexes not enforced by this function:
        
        - among sys states
        - among env actions
    """
    sys_trans = []
    
    # Transitions
    for from_state in states:
        from_state_id = state_ids[from_state]
        cur_trans = trans.find([from_state])
        
        # no successor states ?
        if not cur_trans:
            sys_trans += ['('+str(from_state_id) +') -> X('+ _conj_neg(states) +')']
            continue
        
        cur_str = []
        for (from_state, to_state, label) in cur_trans:
            to_state_id = state_ids[to_state]
            
            precond = '(' + str(from_state_id) + ')'
            postcond = '(' + str(to_state_id) +')'
            
            if 'env_actions' in label:
                env_action = label['env_actions']
                postcond += ' && (' + str(env_action) + ')'
            
            if 'sys_actions' in label:
                sys_action = label['sys_actions']
                postcond += ' && (' + str(sys_action) + ')'
            
            cur_str += ['(' + precond + ') -> X(' + postcond + ')']
            
        sys_trans += [_disj(cur_str) ]
    return sys_trans

def env_trans_from_open_ts(states, state_ids, trans, env_vars):
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
        
        env_trans += ["(" +str(from_state_id) +") -> X("+ next_env_actions +")"]
    return env_trans

def ap_trans_from_ts(states, state_ids, aps):
    """Require atomic propositions to follow states according to label.
    """
    trans = []
    
    for state in states:
        label = states.label_of(state)
        state_id = state_ids[state]
        
        if label.has_key("ap"):
            tmp0 = _conj_intersection(aps, label["ap"], parenth=False)
        else:
            tmp0 = ""
        
        if label.has_key("ap"):
            tmp1 = _conj_neg_diff(aps, label["ap"], parenth=False)
        else:
            tmp1 = _conj_neg(aps, parenth=False)
        
        if len(tmp0) > 0 and len(tmp1) > 0:
            tmp = tmp0 +' && '+ tmp1
        else:
            tmp = tmp0 + tmp1
        
        trans += ["X(("+ str(state_id) +") -> ("+ tmp +"))"]
    return trans

def synthesize(option, specs, sys=None, ignore_ts_init=False,
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
    @param sys: A transition system that should be expressed with the
        specification (spec).
    
    @param ignore_ts_init: Ignore any initial state information
        contained in the ts.
    @type ignore_ts_init: bool
    
    @param bool_states: if True,
        then use one bool variable for each state.
        Otherwise use a single int variable for all states.
        
        Currently int state implemented only for gr1c.
    @type bool_states: bool
    
    @type verbose: bool
    
    @return: If spec is realizable,
        then return a Mealy machine implementing the strategy.
        Otherwise return list of counterexamples.
    @rtype: transys.Mealy or list
    """
    # not yet implemented for jtlv
    if bool_states is False and option is 'jtlv':
        warn('Int state not yet available for jtlv solver.\n' +
             'Using bool states.')
        bool_states = True
    
    specs = spec_plus_sys(specs, sys, ignore_ts_init, bool_states)
    
    if option == 'gr1c':
        ctrl = gr1cint.synthesize(specs, verbose=verbose)
    elif option == 'jtlv':
        ctrl = jtlvint.synthesize(specs, verbose=verbose)
    else:
        raise Exception('Undefined synthesis option. '+\
                        'Current options are "jtlv" and "gr1c"')
    return ctrl

def is_realizable(option, specs, sys=None, ignore_ts_init=False,
                  verbose=0):
    """Check realizability.
    
    For details see synthesize.
    """
    specs = spec_plus_sys(specs, sys, ignore_ts_init)
    
    if option == 'gr1c':
        r = gr1cint.check_realizable(specs, verbose=verbose)
    elif option == 'jtlv':
        r = jtlvint.check_realizable(specs, verbose=verbose)
    else:
        raise Exception('Undefined synthesis option. '+\
                        'Current options are "jtlv" and "gr1c"')
    return r

def spec_plus_sys(specs, sys=None, ignore_ts_init=False, bool_states=True):
    if sys is not None:
        sys = deepcopy(sys)
        
        sform = sys_to_spec(sys, ignore_ts_init, bool_states)
        specs = specs | sform
    return specs
